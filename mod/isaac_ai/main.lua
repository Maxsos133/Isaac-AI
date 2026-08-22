--[[
  Isaac AI bridge mod.

  Turns Isaac into a synchronous RL environment. On every logic tick the mod
  sends an observation over a TCP socket and then BLOCKS until the agent replies
  with an action. That blocking read is what makes the environment synchronous:
  the game cannot advance past a tick the agent has not answered, so there is no
  such thing as a stale observation or a missed action.

  Player control goes through MC_INPUT_ACTION rather than synthetic keystrokes,
  so the window needs no focus and many instances can run at once.

  Per-instance configuration comes from the environment:
    ISAAC_AI_PORT      TCP port to connect to      (default 9999)
    ISAAC_AI_INSTANCE  label echoed in handshake   (default "unknown")
]]

local socket = require("socket")
local json = require("json")

local mod = RegisterMod("isaac_ai", 1)

local PROTOCOL_VERSION = 1
local HOST = "127.0.0.1"
local PORT = tonumber(os.getenv("ISAAC_AI_PORT") or "") or 9999
local INSTANCE = os.getenv("ISAAC_AI_INSTANCE") or "unknown"

-- Cap the entity list so one crowded room cannot blow up message size.
local MAX_ENTITIES = 32

---------------------------------------------------------------------------
-- Connection
---------------------------------------------------------------------------

local tcp = nil
local connected = false
local disabled = false        -- set after an unrecoverable socket error
local step_index = 0

-- Latest action requested by the agent. Movement and shooting are each in
-- {-1, 0, 1} per axis.
local action = { mx = 0, my = 0, sx = 0, sy = 0, bomb = false, item = false, drop = false }

-- When set, the input callback declines every button and the keyboard drives
-- the player instead. Used to put a human on exactly the encounters the agent
-- is scored on, which is the only way to tell whether a task the agent keeps
-- failing is hard or simply unfair.
local human = false

local function clearAction()
    action.mx, action.my, action.sx, action.sy = 0, 0, 0, 0
    action.bomb, action.item, action.drop = false, false, false
end

local function fail(reason)
    Isaac.DebugString("ISAAC_AI fatal: " .. tostring(reason))
    disabled = true
    connected = false
    clearAction()
    if tcp then
        pcall(function() tcp:close() end)
        tcp = nil
    end
end

local function connect()
    tcp = socket.tcp()
    tcp:settimeout(5.0)
    local ok, err = tcp:connect(HOST, PORT)
    if not ok then
        Isaac.DebugString("ISAAC_AI connect failed: " .. tostring(err))
        tcp = nil
        return false
    end
    tcp:setoption("tcp-nodelay", true)
    -- No timeout on gameplay reads: the agent is allowed to think for as long
    -- as it needs, and the game simply waits.
    tcp:settimeout(nil)
    connected = true

    local hello = json.encode({
        t = "hello",
        protocol = PROTOCOL_VERSION,
        instance = INSTANCE,
        port = PORT,
    })
    tcp:send(hello .. "\n")
    Isaac.DebugString("ISAAC_AI connected instance=" .. INSTANCE .. " port=" .. PORT)
    return true
end

---------------------------------------------------------------------------
-- Event accumulation
--
-- Rewards are computed on the Python side, but the raw events have to be
-- captured here because they happen between ticks.
---------------------------------------------------------------------------

-- Damage attribution, for diagnosis only. This never reaches the observation,
-- so it cannot change a policy's input or invalidate a checkpoint.
--
-- Deaths are 67% of episodes on floor-v19b and `damage_taken` is the largest
-- single term in the reward, and nothing here reported what the damage came
-- *from*. Aiming, enemy response, stranding and wall-grinding are all now fixed
-- or best-ever while rooms_cleared sits at 0.29, so the next decision is between
-- "cannot fight" and "cannot avoid hazards" — and it was about to be made with
-- no measurement either way, which is how this project has lost runs before.
local DAMAGE_CATEGORIES = {
    "enemy", "projectile", "spikes", "fire", "explosion", "creep", "other",
}

local function zeroDamageBy()
    local out = {}
    for _, name in ipairs(DAMAGE_CATEGORIES) do
        out[name] = 0.0
    end
    return out
end

local events = {
    damage_taken = 0.0,
    damage_dealt = 0.0,
    kills = 0,
    tears_fired = 0,
    room_cleared = false,
    new_room = false,
    new_level = false,
    died = false,
    game_started = false,
    is_continued = false,
    -- Always every category, always a number. An empty Lua table is ambiguous
    -- to json.encode (array or object depending on the library), and omitting
    -- the zeros is the exact mistake that made end-reason shares sum to 1.047.
    damage_by = zeroDamageBy(),
}

local function resetEvents()
    events.damage_taken = 0.0
    events.damage_dealt = 0.0
    events.kills = 0
    events.tears_fired = 0
    events.room_cleared = false
    events.new_room = false
    events.new_level = false
    events.died = false
    events.game_started = false
    events.is_continued = false
    for _, name in ipairs(DAMAGE_CATEGORIES) do
        events.damage_by[name] = 0.0
    end
end

-- Resolved from the game's own constants with the same missing-name warning the
-- pickup, room and grid maps use. A flag that does not resolve reads 0 and is
-- then never matched, so it degrades to "other" rather than mis-attributing.
local DAMAGE_FLAG_MISSING = {}
local function damageFlag(name)
    local value = DamageFlag and DamageFlag[name]
    if value == nil then
        DAMAGE_FLAG_MISSING[#DAMAGE_FLAG_MISSING + 1] = name
        return 0
    end
    return value
end

local FLAG_SPIKES = damageFlag("DAMAGE_SPIKES")
local FLAG_FIRE = damageFlag("DAMAGE_FIRE")
local FLAG_EXPLOSION = damageFlag("DAMAGE_EXPLOSION")
local FLAG_ACID = damageFlag("DAMAGE_ACID")

if #DAMAGE_FLAG_MISSING > 0 then
    Isaac.DebugString("ISAAC_AI warn: unresolved damage flags: "
        .. table.concat(DAMAGE_FLAG_MISSING, ", "))
end

-- Which of DAMAGE_CATEGORIES one hit on the player belongs to.
--
-- Flags are checked before the source, because a spike or a fire reports a
-- source entity too and the environmental cause is the one worth knowing. Must
-- never throw: this runs inside MC_ENTITY_TAKE_DMG, and an error there skips the
-- send and blocks every instance in the fleet forever. The caller wraps it, and
-- it is written to fall through to "other" rather than to fail.
local function classifyDamage(flags, source)
    flags = flags or 0
    if FLAG_SPIKES ~= 0 and (flags & FLAG_SPIKES) ~= 0 then return "spikes" end
    if FLAG_ACID ~= 0 and (flags & FLAG_ACID) ~= 0 then return "creep" end
    if FLAG_FIRE ~= 0 and (flags & FLAG_FIRE) ~= 0 then return "fire" end
    if FLAG_EXPLOSION ~= 0 and (flags & FLAG_EXPLOSION) ~= 0 then
        return "explosion"
    end
    if source ~= nil and source.Type ~= nil and source.Type ~= 0 then
        if source.Type == EntityType.ENTITY_PROJECTILE then return "projectile" end
        local entity = source.Entity
        if entity ~= nil and entity:IsVulnerableEnemy() then return "enemy" end
    end
    return "other"
end

mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG,
    function(_, entity, amount, flags, source)
        local player = entity:ToPlayer()
        if player then
            events.damage_taken = events.damage_taken + amount
            local ok, category = pcall(classifyDamage, flags, source)
            if not ok or category == nil then category = "other" end
            events.damage_by[category] =
                (events.damage_by[category] or 0.0) + amount
        elseif entity:IsVulnerableEnemy() then
            events.damage_dealt = events.damage_dealt + amount
        end
        return nil
    end)

mod:AddCallback(ModCallbacks.MC_POST_ENTITY_KILL, function(_, entity)
    -- Do NOT gate this on IsVulnerableEnemy(): by the time the kill callback
    -- runs the entity is already dead and reports false, so the counter never
    -- incremented and the kill reward silently never fired. Identify enemies
    -- structurally instead.
    if entity:ToNPC() then
        events.kills = events.kills + 1
    end
end)

-- Counting shots is what makes accuracy measurable: damage dealt alone cannot
-- distinguish "aims well" from "fires constantly and occasionally connects".
if ModCallbacks.MC_POST_FIRE_TEAR then
    mod:AddCallback(ModCallbacks.MC_POST_FIRE_TEAR, function()
        events.tears_fired = events.tears_fired + 1
    end)
end

-- Fires as a room's clear award is about to spawn, which is the game's own
-- signal that the last enemy died. The combat environment counted enemies
-- instead because it spawned into already-cleared rooms; on real floors this is
-- the authoritative event.
mod:AddCallback(ModCallbacks.MC_PRE_SPAWN_CLEAN_AWARD, function()
    events.room_cleared = true
    return nil
end)

mod:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, function()
    events.new_room = true
end)

mod:AddCallback(ModCallbacks.MC_POST_NEW_LEVEL, function()
    events.new_level = true
end)

mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, function(_, isContinued)
    events.game_started = true
    events.is_continued = isContinued and true or false
    step_index = 0
end)

mod:AddCallback(ModCallbacks.MC_POST_GAME_END, function(_, isGameOver)
    if isGameOver then
        events.died = true
    end
end)

---------------------------------------------------------------------------
-- Observation
---------------------------------------------------------------------------

-- How many enemies the last scenario asked for. Declared here, above
-- buildObservation, on purpose.
local scenario_enemies = 0

-- Why the last `new_room` request did or did not move the player. Diagnostic
-- only; nothing in the observation encoding reads it.
--
-- `probe_room_contents.py` sampled 200 rooms with `new_room: True` and found
-- room index 84 every time and zero interior tiles on all six planes — i.e. the
-- start room, unmoved. The flag has been reported as merely *disabled* in
-- `curriculum.py` for several runs; it is inert. Reading the code cannot say
-- whether `RoomShape.ROOMSHAPE_1x1` fails to resolve (which would empty
-- `candidates` for every room), whether the level has no matching rooms, or
-- whether `ChangeRoom` does not take effect from inside the blocked update, so
-- this reports each of those separately.
local scenario_jump = {
    requested = false,   -- new_room was asked for
    room_types = 0,      -- descriptors passing the ROOM_DEFAULT test
    candidates = 0,      -- ...that also passed the shape test
    shape_enum = false,  -- RoomShape.ROOMSHAPE_1x1 resolved at all
    changed = false,     -- ChangeRoom was actually called
    before = -1,
    after = -1,
}

-- Ticks remaining during which a navigation room keeps being swept clear.
-- Declared up here so the update callback can see it.
local nav_clear_ticks = 0

-- Defined ahead of buildObservation deliberately: a Lua local is only visible
-- to code compiled after it, so declaring this lower down would silently make
-- the call a nil global lookup.
local function countLiveEnemies()
    local total = 0
    for _, e in ipairs(Isaac.GetRoomEntities()) do
        if e:IsVulnerableEnemy() and e:IsActiveEnemy(false) and not e:IsDead() then
            total = total + 1
        end
    end
    return total
end

-- What an entity *means*, resolved here from the game's own constants for the
-- same reason door categories are: a raw Type/Variant integer implies an
-- ordering that does not exist, and hardcoding the numbers on the Python side
-- lets them drift silently against a game update.
--
-- Flags rather than one category, because they are not mutually exclusive — a
-- spiked chest is both a chest and a thing that hurts you, and the agent needs
-- both facts. An entity matching nothing gets every flag at zero rather than a
-- default, so a broken mapping stays visible instead of being absorbed.
--
-- Defined above collectEntities deliberately (see countLiveEnemies).
local PICKUP_CONSUMABLE, PICKUP_CHEST, PICKUP_HOSTILE = {}, {}, {}
local PICKUP_PEDESTAL = {}
local BOMB_HOSTILE = {}
local MAPPING_MISSING = {}

local function mark(set, enum, names)
    for _, name in ipairs(names) do
        local value = enum and enum[name]
        if value == nil then
            MAPPING_MISSING[#MAPPING_MISSING + 1] = name
        else
            set[value] = true
        end
    end
end

mark(PICKUP_CONSUMABLE, PickupVariant, {
    "PICKUP_HEART", "PICKUP_COIN", "PICKUP_KEY", "PICKUP_BOMB",
    "PICKUP_LIL_BATTERY", "PICKUP_PILL", "PICKUP_TAROTCARD",
    "PICKUP_GRAB_BAG", "PICKUP_TRINKET",
})
mark(PICKUP_PEDESTAL, PickupVariant, {"PICKUP_COLLECTIBLE"})

-- Chests are derived from the enum by name rather than listed, because listing
-- them by hand was tried and got it wrong: it missed PICKUP_LOCKEDCHEST (60)
-- and PICKUP_REDCHEST (360), which between them were 41% of every pickup the
-- fleet observed. The flag read zero, which looks exactly like "no chests
-- appeared" rather than like a hole in the mapping. There are thirteen chest
-- variants in this build; deriving them also survives a game update adding one.
local chest_names = {}
if PickupVariant then
    for name, value in pairs(PickupVariant) do
        if type(value) == "number" and name:find("CHEST") then
            PICKUP_CHEST[value] = true
            chest_names[#chest_names + 1] = name
        end
    end
end
if #chest_names == 0 then
    MAPPING_MISSING[#MAPPING_MISSING + 1] = "PickupVariant.*CHEST*"
end
-- The two that damage on contact or on opening, and are otherwise pixel-close
-- to a normal chest. This is the flag that actually earns its place.
mark(PICKUP_HOSTILE, PickupVariant, {"PICKUP_SPIKEDCHEST", "PICKUP_MIMICCHEST"})
-- The agent never places bombs (the act message hardcodes bomb = false), so any
-- live bomb is the room's, not its own. Troll bombs are still worth their own
-- flag: they are thrown at the player and detonate on a short fuse.
mark(BOMB_HOSTILE, BombVariant, {"BOMB_TROLL", "BOMB_SUPERTROLL"})

if #MAPPING_MISSING > 0 then
    Isaac.DebugString("ISAAC_AI warn: unresolved entity constants: "
        .. table.concat(MAPPING_MISSING, ", "))
end

-- Log what each name actually resolved to. A name can exist and still not mean
-- what the mapping assumes, and then the flag is simply never true — which
-- looks exactly like the variant being rare. Printing the numbers makes the
-- mapping auditable against the variants the probe observes.
local function dumpResolved(label, enum, names)
    local parts = {}
    for _, name in ipairs(names) do
        local value = enum and enum[name]
        parts[#parts + 1] = name .. "=" .. tostring(value)
    end
    Isaac.DebugString("ISAAC_AI map " .. label .. ": " .. table.concat(parts, " "))
end

table.sort(chest_names)
Isaac.DebugString("ISAAC_AI map chest (" .. #chest_names .. "): "
    .. table.concat(chest_names, " "))
dumpResolved("consumable", PickupVariant, {
    "PICKUP_HEART", "PICKUP_COIN", "PICKUP_KEY", "PICKUP_BOMB",
    "PICKUP_LIL_BATTERY", "PICKUP_PILL", "PICKUP_TAROTCARD",
    "PICKUP_GRAB_BAG", "PICKUP_TRINKET",
})
dumpResolved("hostile", PickupVariant, {"PICKUP_SPIKEDCHEST", "PICKUP_MIMICCHEST"})
dumpResolved("pedestal", PickupVariant, {"PICKUP_COLLECTIBLE"})

local function entityFlags(e)
    local consumable, pedestal, chest, hostile, flying = false, false, false, false, false
    if e.Type == EntityType.ENTITY_PICKUP then
        consumable = PICKUP_CONSUMABLE[e.Variant] == true
        pedestal = PICKUP_PEDESTAL[e.Variant] == true
        chest = PICKUP_CHEST[e.Variant] == true
        hostile = PICKUP_HOSTILE[e.Variant] == true
    elseif e.Type == EntityType.ENTITY_BOMB then
        hostile = BOMB_HOSTILE[e.Variant] == true
    end
    -- Flight changes what an obstacle means: a flying enemy ignores the rocks
    -- and pits the grid encodes, so the same layout implies a different threat.
    local ok, flies = pcall(function() return e:IsFlying() end)
    flying = ok and flies == true
    return consumable, pedestal, chest, hostile, flying
end

local function collectEntities(playerPos)
    local out = {}
    local room = Game():GetRoom()

    for _, e in ipairs(Isaac.GetRoomEntities()) do
        local include = false
        local kind = nil

        if e:IsVulnerableEnemy() and e:IsActiveEnemy(false) then
            include, kind = true, "enemy"
        elseif e.Type == EntityType.ENTITY_PROJECTILE then
            include, kind = true, "projectile"
        elseif e.Type == EntityType.ENTITY_PICKUP then
            include, kind = true, "pickup"
        elseif e.Type == EntityType.ENTITY_BOMB then
            include, kind = true, "bomb"
        end

        if include then
            local consumable, pedestal, chest, hostile, flying = entityFlags(e)
            out[#out + 1] = {
                k = kind,
                t = e.Type,
                v = e.Variant,
                s = e.SubType,
                -- Semantic identity. Without these a chest, a coin, a heart and
                -- an item pedestal are the identical vector, and a spiked chest
                -- or mimic is indistinguishable from a normal one.
                consumable = consumable,
                pedestal = pedestal,
                chest = chest,
                hostile = hostile,
                flying = flying,
                x = e.Position.X,
                y = e.Position.Y,
                vx = e.Velocity.X,
                vy = e.Velocity.Y,
                hp = e.HitPoints,
                mhp = e.MaxHitPoints,
                boss = e:IsBoss(),
                d = (e.Position - playerPos):Length(),
            }
        end
    end

    -- Nearest first, then truncate: what is close matters most.
    table.sort(out, function(a, b) return a.d < b.d end)
    while #out > MAX_ENTITIES do
        table.remove(out)
    end
    return out
end

-- The room's grid: rocks, spikes, pits, poop, TNT.
--
-- These are grid tiles rather than entities, so the entity allowlist never saw
-- them and the agent has been playing floors blind to every obstacle in the
-- room. A feedforward policy re-decides from scratch each tick, so it cannot
-- remember bumping into something — an unseen rock is permanently unlearnable,
-- and it will push into it forever while its tears are eaten by walls it has no
-- input for. Unexplained damage from spikes lands in the value function as pure
-- noise, which is the same argument the curse-door flag already exists for.
--
-- Reported as one small integer per cell so the shape carries the layout:
--   0 free   1 solid (blocks movement and tears)   2 hazard   3 pit
-- Categorised here from the game's own GridEntityType constants, so no grid
-- integers are hardcoded on the Python side where they could drift.
-- A tile's properties, as a bitmask rather than one class, because they are not
-- mutually exclusive and pretending they were lost real information: a spiked
-- rock blocks movement *and* damages on contact, and being forced to pick one
-- meant it was reported as an ordinary rock. Watching the agent, it walked into
-- spikes and fires repeatedly.
--
-- **Bits 0-2 keep exactly the membership the old three classes had.** The new
-- bits are additive, so each grid plane the network already learned is
-- unchanged and a resumed policy transfers exactly — the planes are the leading
-- dimension of the flattened grid, so new ones append at the end.
-- DAMAGING therefore overlaps HAZARD deliberately; it exists to carry the
-- spiked rock that SOLID alone hides.
local GRID_FREE = 0
local GRID_SOLID = 1 << 0        -- blocks movement and tears
local GRID_HAZARD = 1 << 1       -- damages on contact (as originally defined)
local GRID_PIT = 1 << 2          -- blocks anything that cannot fly
local GRID_DAMAGING = 1 << 3     -- will hurt you, including the spiked rock
local GRID_DESTRUCTIBLE = 1 << 4 -- can be removed by shooting it
local GRID_RETRACTABLE = 1 << 5  -- on/off spikes: sometimes safe to stand on
local GRID_BITS = 6

-- Membership resolved once from the game's own constants, with the same
-- missing-name warning as everywhere else, rather than a chain of comparisons.
-- A name that disappears in a game update becomes visible instead of silently
-- dropping a tile type into "free floor", which is the worst possible default:
-- it would tell the agent a rock is walkable.
local GRID_PROPERTIES = {}

local function markGrid(names, bits)
    for _, name in ipairs(names) do
        local value = GridEntityType and GridEntityType[name]
        if value == nil then
            MAPPING_MISSING[#MAPPING_MISSING + 1] = name
        else
            GRID_PROPERTIES[value] = (GRID_PROPERTIES[value] or 0) | bits
        end
    end
end

-- Bits 0-2: exactly the membership the original three classes had. Do not
-- change these without accepting that every trained grid plane shifts meaning.
markGrid({"GRID_ROCK", "GRID_ROCKB", "GRID_ROCKT", "GRID_ROCK_BOMB",
          "GRID_ROCK_ALT", "GRID_ROCK_SS", "GRID_ROCK_SPIKED",
          "GRID_ROCK_GOLD", "GRID_PILLAR", "GRID_POOP", "GRID_WALL",
          "GRID_DOOR"}, GRID_SOLID)
markGrid({"GRID_SPIKES", "GRID_SPIKES_ONOFF", "GRID_TNT",
          "GRID_FIREPLACE"}, GRID_HAZARD)
markGrid({"GRID_PIT"}, GRID_PIT)

-- New, additive. GRID_ROCK_SPIKED is the one that matters: it is solid, so the
-- old encoding filed it as an ordinary rock and never said it hurts.
markGrid({"GRID_ROCK_SPIKED", "GRID_SPIKES", "GRID_SPIKES_ONOFF", "GRID_TNT",
          "GRID_FIREPLACE"}, GRID_DAMAGING)
-- Shootable out of the way. The agent has no bomb action, so rocks are not
-- destructible to it however destructible they are in principle.
markGrid({"GRID_POOP", "GRID_TNT", "GRID_FIREPLACE"}, GRID_DESTRUCTIBLE)
-- Sometimes retracted and safe to cross, unlike permanent spikes.
markGrid({"GRID_SPIKES_ONOFF"}, GRID_RETRACTABLE)

local function gridClass(kind)
    if kind == nil then return GRID_FREE end
    return GRID_PROPERTIES[kind] or GRID_FREE
end

local function tileProperties(grid)
    local kind = grid:GetType()
    -- A doorway is only a wall while it is shut. GRID_DOOR was in the solid
    -- set unconditionally, so the obstacle grid told the agent that every exit
    -- in the room was a wall — including the one the shaping was steering it
    -- towards. The egocentric window makes that worse in one direction: a 1x1
    -- room is 9 tiles tall and the window is 9 tall, so the top and bottom
    -- walls are *always* in view, while the side walls usually are not. So
    -- "solid straight ahead" was a far more reliable signal vertically, and the
    -- measured policy backs away from up doors harder the closer it gets.
    if GridEntityType and kind == GridEntityType.GRID_DOOR then
        local ok, door = pcall(function() return grid:ToDoor() end)
        if ok and door and door:IsOpen() then
            return GRID_FREE
        end
        return GRID_SOLID
    end
    return GRID_PROPERTIES[kind] or GRID_FREE
end

local function collectGrid(room)
    local out = {}
    local size = room:GetGridSize()
    for index = 0, size - 1 do
        local grid = room:GetGridEntity(index)
        out[#out + 1] = grid and tileProperties(grid) or GRID_FREE
    end
    return out
end

-- What kind of room a door leads to. Categorised here, using the game's own
-- RoomType constants, so no room-type integers are hardcoded on the Python
-- side where they could drift silently against a game update.
--
-- Curse is its own category because it is the one that costs health simply to
-- walk through: without a flag for it the agent sees a normal door, takes
-- unexplained damage, and the inconsistency becomes noise in the value function.
local DOOR_CATEGORY = {}
if RoomType then
    DOOR_CATEGORY[RoomType.ROOM_DEFAULT] = "normal"
    DOOR_CATEGORY[RoomType.ROOM_TREASURE] = "treasure"
    DOOR_CATEGORY[RoomType.ROOM_SHOP] = "shop"
    DOOR_CATEGORY[RoomType.ROOM_BOSS] = "boss"
    DOOR_CATEGORY[RoomType.ROOM_MINIBOSS] = "boss"
    DOOR_CATEGORY[RoomType.ROOM_CURSE] = "curse"
end

-- Rooms whose door needs an explosive. The floor agent has no bomb action at
-- all — `floors.py` hardcodes bomb = false in every message — so these are
-- permanently impassable to it, and `locked` does not mark them: a secret room
-- door is closed and *unlocked*. Measured over 900 fleet steps, the shaping
-- targeted one on 1068 observations, 6% of every step where it had a target,
-- and the agent walked into the wall until its idle limit ran out.
--
-- Resolved by name with the same missing-constant warning as everything else,
-- so a rename in a game update is visible rather than silently reopening this.
local NEEDS_BOMB = {}
for _, name in ipairs({"ROOM_SECRET", "ROOM_SUPERSECRET", "ROOM_ULTRASECRET"}) do
    local value = RoomType and RoomType[name]
    if value == nil then
        MAPPING_MISSING[#MAPPING_MISSING + 1] = name
    else
        NEEDS_BOMB[value] = true
    end
end
-- Last of the three mapping blocks (pickups, grid tiles, room types), so this
-- catches every unresolved name from all of them — hence the generic label.
-- An unresolved grid name is the worst of the three: it would default that tile
-- to free floor and tell the agent a rock is walkable.
if #MAPPING_MISSING > 0 then
    Isaac.DebugString("ISAAC_AI warn: unresolved game constants: "
        .. table.concat(MAPPING_MISSING, ", "))
end

local function categoryOfRoomType(roomType)
    if roomType == nil then
        return "unknown"
    end
    return DOOR_CATEGORY[roomType] or "other"
end

local function doorCategory(door, descriptor)
    -- Prefer the door's own field; fall back to the room descriptor.
    local roomType = door.TargetRoomType
    if roomType == nil and descriptor and descriptor.Data then
        roomType = descriptor.Data.Type
    end
    return categoryOfRoomType(roomType)
end

local function collectDoors()
    local game = Game()
    local room = game:GetRoom()
    local level = game:GetLevel()
    local doors = {}
    for slot = 0, DoorSlot.NUM_DOOR_SLOTS - 1 do
        local door = room:GetDoor(slot)
        if door then
            -- Whether the room behind a door has been seen is what turns
            -- wandering into exploration: without it the agent cannot tell a
            -- new route from the one it just came through.
            local visited = 0
            local descriptor = level:GetRoomByIdx(door.TargetRoomIndex)
            if descriptor and descriptor.VisitedCount then
                visited = descriptor.VisitedCount
            end
            doors[#doors + 1] = {
                slot = slot,
                x = door.Position.X,
                y = door.Position.Y,
                open = door:IsOpen(),
                locked = door:IsLocked(),
                -- The raw room type behind the door, alongside the resolved
                -- category. `locked` does not mean "cannot pass": a secret room
                -- door is closed and *unlocked*, and needs a bomb the agent has
                -- no action for. Reporting the type makes that distinguishable
                -- instead of leaving it indistinguishable from a normal door
                -- that happens to be shut for a fight.
                target_type = door.TargetRoomType,
                -- Impassable without an explosive, which this agent cannot use.
                -- Deliberately a separate field rather than a new door category:
                -- the categories are a fixed-width one-hot inside a flattened
                -- per-slot block, so adding one reshuffles every later column
                -- and a resumed policy would quietly compute something else.
                needs_bomb = NEEDS_BOMB[door.TargetRoomType] == true,
                target = door.TargetRoomIndex,
                visited = visited,
                category = doorCategory(door, descriptor),
            }
        end
    end
    return doors
end

local function levelProgress()
    local level = Game():GetLevel()
    local rooms = level:GetRooms()
    local total, visited = 0, 0
    for index = 0, rooms.Size - 1 do
        local descriptor = rooms:Get(index)
        if descriptor then
            total = total + 1
            if descriptor.VisitedCount and descriptor.VisitedCount > 0 then
                visited = visited + 1
            end
        end
    end
    return total, visited
end

local function buildObservation()
    local game = Game()
    local player = Isaac.GetPlayer(0)

    -- There are ticks with no player: the game-over screen, and the gap while
    -- a restart rebuilds the run. Still report something, otherwise the agent
    -- blocks on a read that never arrives and the whole fleet stalls.
    if not player then
        return {
            t = "obs",
            step = step_index,
            ready = false,
            events = {
                damage_taken = 0.0,
                damage_dealt = 0.0,
                kills = 0,
                room_cleared = false,
                new_room = events.new_room,
                new_level = events.new_level,
                died = events.died,
                game_started = events.game_started,
                is_continued = events.is_continued,
                damage_by = zeroDamageBy(),
            },
            frame = game:GetFrameCount(),
        }
    end

    local level = game:GetLevel()
    local room = game:GetRoom()
    local pos = player.Position
    local rooms_total, rooms_visited = levelProgress()

    return {
        t = "obs",
        step = step_index,
        ready = true,
        player = {
            x = pos.X,
            y = pos.Y,
            -- Which grid tile the player occupies, straight from the game.
            -- Python used to derive this from the room's extents, which assumes
            -- the playable area spans the full grid interior. That holds for a
            -- 1x1 room and is wrong for every "half" shape (IH, IV and friends),
            -- where the playable region is a sub-rectangle of a 15x9 grid whose
            -- remainder is wall — the player then resolved into a wall tile on
            -- 24% of observations in those rooms. `GetGridIndex` is
            -- authoritative for every shape and costs one call.
            grid_index = room:GetGridIndex(pos),
            vx = player.Velocity.X,
            vy = player.Velocity.Y,
            hearts = player:GetHearts(),
            max_hearts = player:GetMaxHearts(),
            soul_hearts = player:GetSoulHearts(),
            bombs = player:GetNumBombs(),
            keys = player:GetNumKeys(),
            coins = player:GetNumCoins(),
            damage = player.Damage,
            speed = player.MoveSpeed,
            tear_delay = player.MaxFireDelay,
            range = player.TearRange,
            can_fly = player.CanFly,
            active_item = player:GetActiveItem(ActiveSlot.SLOT_PRIMARY),
            -- What a `reseed` preserves, reported so Python can tell a pristine
            -- run from one that has acquired something. Derived stats cannot do
            -- this job: a familiar, a tear modifier, a trinket or a held card
            -- moves none of damage/speed/range/tear_delay/max_hearts, so an
            -- instance carrying one reads as untouched and keeps it for every
            -- later episode of that run.
            collectibles = player:GetCollectibleCount(),
            trinket0 = player:GetTrinket(0),
            trinket1 = player:GetTrinket(1),
            card0 = player:GetCard(0),
            pill0 = player:GetPill(0),
            is_dead = player:IsDead(),
        },
        room = {
            index = level:GetCurrentRoomIndex(),
            type = room:GetType(),
            -- The current room run through the same category table the doors
            -- use, which makes the mapping directly checkable: teleport into a
            -- known room type and read back what it resolved to.
            category = categoryOfRoomType(room:GetType()),
            shape = room:GetRoomShape(),
            clear = room:IsClear(),
            -- Spawning into an already-cleared room does not re-lock it, so
            -- IsClear() cannot signal that an encounter is over. Count instead.
            enemies_alive = countLiveEnemies(),
            scenario_enemies = scenario_enemies,
            scenario_jump = scenario_jump,
            top_left_x = room:GetTopLeftPos().X,
            top_left_y = room:GetTopLeftPos().Y,
            bottom_right_x = room:GetBottomRightPos().X,
            bottom_right_y = room:GetBottomRightPos().Y,
            doors = collectDoors(),
            grid = collectGrid(room),
            grid_width = room:GetGridWidth(),
        },
        level = {
            stage = level:GetStage(),
            stage_type = level:GetStageType(),
            curses = level:GetCurses(),
            rooms_total = rooms_total,
            rooms_visited = rooms_visited,
        },
        entities = collectEntities(pos),
        events = {
            damage_taken = events.damage_taken,
            damage_dealt = events.damage_dealt,
            kills = events.kills,
            tears_fired = events.tears_fired,
            room_cleared = events.room_cleared,
            new_room = events.new_room,
            new_level = events.new_level,
            died = events.died,
            game_started = events.game_started,
            is_continued = events.is_continued,
            damage_by = events.damage_by,
        },
        difficulty = game.Difficulty,
        frame = game:GetFrameCount(),
    }
end

---------------------------------------------------------------------------
-- Scenario construction
--
-- The console's `spawn` drops entities directly on the player and cannot pick
-- a position, so encounters are built here instead, where room bounds and free
-- space are known.
---------------------------------------------------------------------------

-- Identify enemies structurally rather than by state. IsVulnerableEnemy() is
-- false while an entity is spawning or briefly invulnerable, so gating on it
-- left bosses and mid-animation enemies alive — the same trap that made the
-- kill counter silently read zero.
local function clearRoomEntities()
    for _, e in ipairs(Isaac.GetRoomEntities()) do
        if e:ToNPC()
            or e.Type == EntityType.ENTITY_PROJECTILE
            or e.Type == EntityType.ENTITY_TEAR then
            e:Remove()
        end
    end
end

local function spawnPosition(minDistance, playerPos)
    local room = Game():GetRoom()
    for _ = 1, 40 do
        local candidate = room:GetRandomPosition(60)
        if (candidate - playerPos):Length() >= minDistance then
            return Isaac.GetFreeNearPosition(candidate, 40)
        end
    end
    -- Nowhere far enough (small room): take any free spot rather than stacking
    -- the encounter on top of the player.
    return Isaac.GetFreeNearPosition(room:GetCenterPos(), 40)
end

-- Spawning into a cleared room leaves its doors open, and the agent can simply
-- walk out. Because enemy counting is per-room, leaving would read as "no
-- enemies left" and pay the clear bonus for running away. Marking the room
-- uncleared restores the game's own combat behaviour: doors shut, and the
-- normal clear award fires when the last enemy dies.
local function lockRoom()
    local room = Game():GetRoom()
    if room.SetClear then
        room:SetClear(false)
    end
    for slot = 0, DoorSlot.NUM_DOOR_SLOTS - 1 do
        local door = room:GetDoor(slot)
        if door then
            door:Close(true)
        end
    end
end

-- Move to a different plain room before an encounter is built.
--
-- Every combat encounter this project has run has happened in the same room,
-- and every policy trained in it has converged on backing onto one wall and
-- firing across: v4 took the left wall, v5 through v7 the top. A single fixed
-- geometry makes one wall permanently correct, which collapses both aiming and
-- movement into something simple enough that the unused action heads stop
-- receiving gradient and the entropy bonus pins them uniform.
--
-- Restricted to ROOM_DEFAULT deliberately. `jumpToRandomRoom`, used by the
-- navigation setup, takes any room at all — which here would eventually drop
-- the player into a treasure or shop room, hand him an item, and permanently
-- change the stats every run is supposed to hold frozen.
local function jumpToPlainRoom()
    scenario_jump.shape_enum = (RoomShape ~= nil
                                and RoomShape.ROOMSHAPE_1x1 ~= nil)
    scenario_jump.room_types = 0
    scenario_jump.candidates = 0
    scenario_jump.changed = false
    if not RoomType then return false end
    local level = Game():GetLevel()
    local rooms = level:GetRooms()
    local current = level:GetCurrentRoomIndex()
    scenario_jump.before = current
    local candidates = {}
    for index = 0, rooms.Size - 1 do
        local descriptor = rooms:Get(index)
        -- Same shape as well as same type. Roughly one plain room in five on a
        -- floor is 2x2, four times the area, and five enemies spread across
        -- that with a 260 tear range is a different task entirely. The
        -- curriculum has a single difficulty scalar and cannot tell "hard
        -- because there are many enemies" from "hard because the room is
        -- enormous", so it lowers the enemy count to hold success at target and
        -- the dial stops meaning anything. Measured: difficulty settled at
        -- 0.73 before room jumping and 0.33-0.39 after, across three runs.
        local plain = descriptor and descriptor.SafeGridIndex
            and descriptor.SafeGridIndex ~= current
            and descriptor.Data
            and descriptor.Data.Type == RoomType.ROOM_DEFAULT
        if plain then
            scenario_jump.room_types = scenario_jump.room_types + 1
            if (not RoomShape
                or descriptor.Data.Shape == RoomShape.ROOMSHAPE_1x1) then
                candidates[#candidates + 1] = descriptor.SafeGridIndex
            end
        end
    end
    scenario_jump.candidates = #candidates
    if #candidates == 0 then return false end
    Game():ChangeRoom(candidates[math.random(#candidates)])
    scenario_jump.changed = true
    scenario_jump.after = Game():GetLevel():GetCurrentRoomIndex()
    return true
end


local function buildScenario(cmd)
    local player = Isaac.GetPlayer(0)
    if not player then return end

    -- Before anything else: the room itself has to change, or clearing and
    -- repositioning just rebuild the same encounter in the same geometry.
    scenario_jump.requested = cmd.new_room and true or false
    if cmd.new_room then
        jumpToPlainRoom()
        player = Isaac.GetPlayer(0) or player
    end

    clearRoomEntities()

    if cmd.heal then
        player:AddHearts(24)
        player:AddSoulHearts(-24)
    end

    -- Move the player before spawning, not after. Encounters rebuild in place,
    -- so without this the position the last fight ended in decides where the
    -- next one is allowed to appear: spawns must clear min_distance, and from
    -- against a wall every legal spot is on one side. Two runs learned to hug a
    -- wall and fire across the room, which is not a combat strategy but control
    -- over the spawn distribution. The navigation setup already repositions for
    -- the same reason; combat never did.
    if cmd.reposition then
        local room = Game():GetRoom()
        local target = room:GetRandomPosition(80)
        player.Position = Isaac.GetFreeNearPosition(target, 40)
    end

    local minDistance = tonumber(cmd.min_distance) or 160
    local spawned = 0

    for _, group in ipairs(cmd.enemies or {}) do
        local count = tonumber(group.count) or 1
        for _ = 1, count do
            local position = spawnPosition(minDistance, player.Position)
            Isaac.Spawn(
                tonumber(group.type) or 10,
                tonumber(group.variant) or 0,
                tonumber(group.subtype) or 0,
                position,
                Vector(0, 0),
                nil
            )
            spawned = spawned + 1
        end
    end

    if spawned > 0 then
        lockRoom()
    end

    scenario_enemies = spawned
    clearAction()
end

-- Strip the room to pure navigation: nothing alive, every door open, full
-- health. Combat was learnable on its own; navigation has only ever been asked
-- for as the first link of a long chain, and three floor runs stalled there.
-- This isolates it so it can be trained the same way combat was.
-- Jump to a random room on the current floor. Far cheaper than regenerating a
-- run, and it stops the agent drilling the same door pair over and over: at one
-- traversal per episode it otherwise shuttles between two rooms until the floor
-- is rebuilt.
local function jumpToRandomRoom()
    local level = Game():GetLevel()
    local rooms = level:GetRooms()
    local candidates = {}
    for index = 0, rooms.Size - 1 do
        local descriptor = rooms:Get(index)
        if descriptor and descriptor.SafeGridIndex
            and descriptor.SafeGridIndex ~= level:GetCurrentRoomIndex() then
            candidates[#candidates + 1] = descriptor.SafeGridIndex
        end
    end
    if #candidates == 0 then return false end
    local choice = candidates[math.random(#candidates)]
    Game():ChangeRoom(choice)
    return true
end

-- Clear a navigation room and keep it open. Repeated for several ticks after
-- setup: an enemy that spawns late also re-locks the doors behind it, so both
-- halves have to be redone, not just the removal.
-- Spikes and moving spike traps are grid tiles, not entities, so removing NPCs
-- leaves them behind. They can kill in a room with nothing alive in it, which
-- costs a run restart and adds damage noise to a task that is only supposed to
-- measure whether the agent can walk to a door.
local function clearGridHazards(room)
    if not room.RemoveGridEntity or not GridEntityType then return end
    for index = 0, room:GetGridSize() - 1 do
        local grid = room:GetGridEntity(index)
        if grid then
            local kind = grid:GetType()
            -- TNT is a hazard here mainly because the policy fires constantly:
            -- the shoot heads get no gradient during navigation, so they stay
            -- near-random and the agent will happily detonate a barrel next to
            -- itself.
            if kind == GridEntityType.GRID_SPIKES
                or kind == GridEntityType.GRID_SPIKES_ONOFF
                or kind == GridEntityType.GRID_TNT then
                room:RemoveGridEntity(index, 0, false)
            end
        end
    end
end

local function sweepNavigationRoom()
    local room = Game():GetRoom()
    clearRoomEntities()
    clearGridHazards(room)
    if room.SetClear then
        room:SetClear(true)
    end
    for slot = 0, DoorSlot.NUM_DOOR_SLOTS - 1 do
        local door = room:GetDoor(slot)
        if door and not door:IsOpen() then
            door:Open()
        end
    end
end

local function setupNavigation(cmd)
    if cmd.random_room then
        jumpToRandomRoom()
    end

    local player = Isaac.GetPlayer(0)
    if not player then return end

    clearRoomEntities()
    player:AddHearts(24)

    local room = Game():GetRoom()
    if room.SetClear then
        room:SetClear(true)
    end
    for slot = 0, DoorSlot.NUM_DOOR_SLOTS - 1 do
        local door = room:GetDoor(slot)
        if door then
            door:TryUnlock(player, true)
            door:Open()
        end
    end

    -- Start somewhere different each time, or the policy can memorise one walk
    -- instead of learning to head for a door from wherever it happens to be.
    if cmd.reposition then
        local target = room:GetRandomPosition(80)
        player.Position = Isaac.GetFreeNearPosition(target, 40)
    end

    -- A room spawns its contents over the ticks following initialisation, so a
    -- single sweep at setup time misses whatever appears next. Keep sweeping
    -- for a few ticks. nav-v4 recorded 11 deaths in a task that is supposed to
    -- contain no enemies at all.
    nav_clear_ticks = 10

    scenario_enemies = 0
    clearAction()
end

---------------------------------------------------------------------------
-- Command handling
---------------------------------------------------------------------------

local function applyCommand(cmd)
    if type(cmd) ~= "table" then return end

    if cmd.t == "human" then
        -- Release the tick without steering. Returning nil from the input
        -- callback is the same path the mod already takes when the bridge is
        -- down, so the keyboard drives the player exactly as it normally would.
        human = true
        clearAction()

    elseif cmd.t == "act" then
        human = false
        action.mx = tonumber(cmd.mx) or 0
        action.my = tonumber(cmd.my) or 0
        action.sx = tonumber(cmd.sx) or 0
        action.sy = tonumber(cmd.sy) or 0
        action.bomb = cmd.bomb and true or false
        action.item = cmd.item and true or false
        action.drop = cmd.drop and true or false

    elseif cmd.t == "reset" then
        -- A fresh episode. Neutral controls first so no input carries across
        -- the boundary, then let the console build the new run.
        clearAction()
        if cmd.seed and cmd.seed ~= "" then
            Isaac.ExecuteCommand("seed " .. tostring(cmd.seed))
        end
        Isaac.ExecuteCommand("restart")

    elseif cmd.t == "scenario" then
        buildScenario(cmd)

    elseif cmd.t == "navsetup" then
        setupNavigation(cmd)

    elseif cmd.t == "command" and type(cmd.value) == "string" then
        -- Escape hatch for curriculum control: spawning enemies, changing
        -- stage, granting items. Used by the training-scenario code.
        Isaac.ExecuteCommand(cmd.value)

    elseif cmd.t == "noop" then
        clearAction()
    end
end

---------------------------------------------------------------------------
-- The synchronous step
---------------------------------------------------------------------------

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, function()
    if disabled then return end

    -- Keep sweeping a navigation room for a few ticks after setup, so anything
    -- the room spawns late is removed before it can reach the player.
    if nav_clear_ticks > 0 then
        nav_clear_ticks = nav_clear_ticks - 1
        sweepNavigationRoom()
    end

    if not connected then
        if not connect() then
            disabled = true
            return
        end
    end

    -- An error anywhere in observation building would otherwise skip the send,
    -- leaving the agent blocked on a read that never arrives and stalling the
    -- whole fleet. Always emit something.
    local built, obs = pcall(buildObservation)
    if not built then
        Isaac.DebugString("ISAAC_AI observation error: " .. tostring(obs))
        obs = {t = "obs", step = step_index, ready = false,
               error = tostring(obs), events = {}}
    end

    local encoded_ok, encoded = pcall(json.encode, obs)
    if not encoded_ok then
        encoded = '{"t":"obs","ready":false,"error":"encode_failed","events":{}}'
    end
    local sent, sendErr = tcp:send(encoded .. "\n")
    if not sent then
        return fail("send: " .. tostring(sendErr))
    end

    -- Events belong to the tick that just reported them.
    resetEvents()

    local line, recvErr = tcp:receive("*l")
    if not line then
        return fail("receive: " .. tostring(recvErr))
    end

    local ok, cmd = pcall(json.decode, line)
    if not ok then
        return fail("decode: " .. tostring(cmd))
    end

    applyCommand(cmd)
    step_index = step_index + 1
end)

---------------------------------------------------------------------------
-- Input driving
---------------------------------------------------------------------------

local AXIS = {
    [ButtonAction.ACTION_LEFT] = function() return -action.mx end,
    [ButtonAction.ACTION_RIGHT] = function() return action.mx end,
    [ButtonAction.ACTION_UP] = function() return -action.my end,
    [ButtonAction.ACTION_DOWN] = function() return action.my end,
    [ButtonAction.ACTION_SHOOTLEFT] = function() return -action.sx end,
    [ButtonAction.ACTION_SHOOTRIGHT] = function() return action.sx end,
    [ButtonAction.ACTION_SHOOTUP] = function() return -action.sy end,
    [ButtonAction.ACTION_SHOOTDOWN] = function() return action.sy end,
}

local BUTTON = {
    [ButtonAction.ACTION_BOMB] = function() return action.bomb end,
    [ButtonAction.ACTION_ITEM] = function() return action.item end,
    [ButtonAction.ACTION_DROP] = function() return action.drop end,
}

mod:AddCallback(ModCallbacks.MC_INPUT_ACTION, function(_, entity, hook, buttonAction)
    -- Once the bridge is down the agent must not keep holding anything.
    if disabled or not connected then return nil end
    -- Human at the controls: decline everything so the game reads the keyboard.
    if human then return nil end

    local axis = AXIS[buttonAction]
    if axis then
        local value = axis()
        if value < 0 then value = 0 end
        if hook == InputHook.GET_ACTION_VALUE then
            return value
        elseif hook == InputHook.IS_ACTION_PRESSED or hook == InputHook.IS_ACTION_TRIGGERED then
            return value > 0
        end
        return nil
    end

    local button = BUTTON[buttonAction]
    if button then
        local pressed = button()
        if hook == InputHook.GET_ACTION_VALUE then
            return pressed and 1.0 or 0.0
        elseif hook == InputHook.IS_ACTION_PRESSED or hook == InputHook.IS_ACTION_TRIGGERED then
            return pressed
        end
    end

    return nil
end)

Isaac.DebugString("ISAAC_AI mod loaded instance=" .. INSTANCE .. " port=" .. PORT)
