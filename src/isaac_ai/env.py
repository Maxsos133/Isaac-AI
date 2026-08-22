"""Vectorized Isaac environment over a fleet of synchronous instances.

Observations are emitted as an entity set rather than a flat feature vector.
A fixed-length vector forces a fixed enemy count and layout, which is a large
part of why the previous project's policies did not survive contact with rooms
they were not trained on. A padded set plus a mask lets one policy handle any
number and mix of entities via a permutation-invariant encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from isaac_ai.bridge import BridgeError, InstanceBridge
from isaac_ai.config import AppConfig

MAX_ENTITIES = 32
# 10 intrinsic features plus 7 player-relative ones. The relative block exists
# because the encoder embeds each entity independently and pools before player
# state is concatenated — so without it, the per-entity embedding never sees
# where the player is, and the relative geometry that aiming depends on is
# destroyed by the pooling. Measured: perfect aim was worth +37.5 points of
# success over the policy that had to infer direction post-pooling.
#
# The last two are closing and tangential speed, added for the same reason.
# Enemy velocity was room-absolute, and the player's own velocity lives in the
# scalar vector which is concatenated *after* pooling — so "is this thing
# coming for me" was not merely hard to infer per entity, it was unavailable,
# exactly as direction-to-target had been. Watching the agent play showed it
# eating contact damage from chargers and never singling them out, which is
# what that missing distinction looks like from outside.
# 17 geometric/kinematic features plus five semantic identity flags appended at
# the end: consumable, pedestal, chest, hostile, flying. Before them a chest, a
# coin, a heart and an item pedestal were the identical vector — and a spiked
# chest or mimic, both of which damage you, looked exactly like a normal chest.
# The mod already sent Type/Variant/SubType all along; nothing read them.
#
# Flags rather than a variant id for the same reason door categories are flags:
# a raw number implies an ordering that does not exist. Resolved in Lua from the
# game's own constants, and an entity matching nothing gets all zeros rather
# than a default, so a broken mapping stays visible.
#
# **New features must be appended, never inserted.** `_warm_start` widens a
# layer by keeping the old columns and zeroing the new ones, which makes a
# resumed policy start out computing exactly what it did before. Inserting
# shifts every later column and scrambles that transfer without erroring.
ENTITY_FEATURES = 22
# Distance to the nearest blocking tile along each of the eight movement
# directions, appended to the scalar block.
#
# The measurement this answers: `diagnose_cornered.py` found blocked retreat
# **3.24x enriched at death** — 41.2% of deaths against a 12.7% baseline, with a
# monotonic gradient through "taking damage" at 25.8%. The agent's only defensive
# tactic is backing away, and it fails about four times in ten because of
# geometry.
#
# Why rays rather than more grid. The obstacle grid has been present since
# floor-v6 and unused in every run: `blocking -> irrelevant` has never exceeded
# 0.029, and `probe_input_sensitivity.py` measures the ego window at **0.013**
# jacobian per value against doors' **0.106** and the entity block's **0.194** —
# 8x to 15x less influential than the inputs the policy actually reads. v14
# through v17 responded to that by adding *more* raw values (planes 3-5, a conv,
# a type embedding) and every run got worse.
#
# What the policy does lean on is **relational** and low-dimensional: doors are
# 13 features each carrying direction and distance to one thing, scalars are 21
# hand-built quantities. So obstacles get the same form — 8 numbers meaning "how
# far can I go this way", which is precisely the quantity a retreat decision
# needs and which 243 raw booleans never expressed.
#
# Order matches the action space: index 0 is up (my = -1), 2 is right (mx = +1),
# 4 is down, 6 is left, with the diagonals between. So ray i is literally "how
# far if I hold this direction", aligned with what the movement heads emit.
RAY_DIRECTIONS = ((0, -1), (1, -1), (1, 0), (1, 1),
                  (0, 1), (-1, 1), (-1, 0), (-1, -1))
# Named rather than written as literals at the point of use, because the plane
# meanings are documented at GRID_CLASSES and nothing else in this file had to
# refer to a single bit before. Plane 1 (hazard) is deliberately absent here.
GRID_SOLID_BIT, GRID_PIT_BIT = 0, 2
# Tiles. A 1x1 room's interior is 13x7, so 8 reaches the far wall from anywhere
# in it and saturates rather than truncating usefully in bigger rooms.
RAY_MAX_TILES = 8

# **Off after floor-v23. The feature works; it does not pay.**
#
# `diagnose_cornered.py` against v19b's numbers, on the same instrument:
#
#     baseline blocked retreat   12.7% -> 6.3%   (halved)
#     at-death blocked retreat   41.2% -> 33.3%
#     free neighbours at death    4.13 -> 5.05
#
# Every absolute number moved the right way and `blocked_rate` (0.058) and
# `ended_died` (0.495) were the lowest of any run. The enrichment *ratio* rose
# 3.24x -> 5.25x only because the baseline improved faster than the tail, which
# is not a regression.
#
# But `rooms_cleared` fell 0.528 -> 0.484 against a continued-training bar of
# ~0.619, and the slope collapsed to +0.0026 per 100k against the lineage's
# +0.0091. What the agent did with better obstacle awareness was get cornered
# less, die less, and stand still more — `ended_idle` 0.503, `still_rate` peaking
# at 0.116. **Give this agent a way to reduce risk and it spends it on the safe
# useless strategy**, because clearing rooms is still the losing trade. v20 did
# exactly this with a larger `kill` reward.
#
# Kept switchable rather than deleted: the mechanism is sound and cheap (10.3us,
# ~9.6% of one core at 20 instances) and this should come back the moment combat
# is worth doing. Turning it on changes SCALAR_FEATURES 21 -> 29, so a resume
# across the switch widens `scalar_encoder.0` and stays exact in that direction
# only — going back the other way needs a checkpoint from before it was on.
ENABLE_OBSTACLE_RAYS = False
# The feature's own width, independent of whether it is wired into the
# observation — `encode_obstacle_rays` still returns eight numbers while the
# switch is off, so its semantics stay under test rather than rotting.
RAY_FEATURES = len(RAY_DIRECTIONS)
# What it actually contributes to the scalar block, which is the thing
# SCALAR_FEATURES has to agree with.
SCALAR_RAY_FEATURES = RAY_FEATURES if ENABLE_OBSTACLE_RAYS else 0

# 21 player/room/level quantities plus the eight obstacle rays appended for
# floor-v23. The rays are last; see RAY_DIRECTIONS and the note at the append.
SCALAR_FEATURES = 21 + SCALAR_RAY_FEATURES

# Named so nothing has to count. Anything indexing this vector positionally
# from the end (`ENTITY_FEATURES - 1`) silently reads a different feature the
# next time one is appended — and reads a *zeroed* flag rather than raising, so
# the breakage looks like a value that simply went to zero.
ENTITY_CLOSING = 15
ENTITY_TANGENTIAL = 16
ENTITY_CONSUMABLE = 17
ENTITY_PEDESTAL = 18
ENTITY_CHEST = 19
ENTITY_HOSTILE = 20
ENTITY_FLYING = 21

# Isaac rooms have eight door slots (four sides, two per side on large rooms).
# Doors are encoded as a fixed-size block rather than as entities because their
# slot index *is* their meaning — slot 0 is always the left door — so position
# in the block carries information that pooling would throw away.
MAX_DOORS = 8
# 7 geometric/state features plus one flag per room category behind the door.
# Categories are flags rather than a scalar type id on purpose: a room-type
# number implies an ordering ("curse is twice boss") that does not exist.
DOOR_CATEGORIES = ("normal", "treasure", "shop", "boss", "curse", "other")
DOOR_FEATURES = 7 + len(DOOR_CATEGORIES)

# The room's obstacle grid — rocks, spikes, pits, poop, TNT. A 1x1 room is 15x9
# tiles including its walls, which is what every training room is; anything
# larger is sampled onto the same frame so the input shape never moves.
#
# Fixed layout rather than the entity set, for the same reason doors are:
# position *is* the meaning here, and pooling would destroy it. Feeding thirty
# rocks through the permutation-invariant path would also crowd out the enemies,
# which is what MAX_ENTITIES exists to hold.
#
# Three channels, one per class the mod reports: solid blocks both movement and
# tears, hazard damages on contact, pit blocks anything that cannot fly.
GRID_WIDTH, GRID_HEIGHT = 15, 9
# One plane per property, and the mod reports a **bitmask** rather than a single
# class, because a tile's properties are not mutually exclusive. Forcing a
# choice lost real information: a spiked rock is solid, so it was filed as an
# ordinary rock and nothing ever said it hurts to touch.
#
#   0 solid         blocks movement and tears
#   1 hazard        damages on contact (the original definition, unchanged)
#   2 pit           blocks anything that cannot fly
#   3 damaging      will hurt you — includes the spiked rock plane 0 hides
#   4 destructible  can be shot away (poop, TNT, fireplace)
#   5 retractable   on/off spikes, sometimes safe to cross
#
# Planes 0-2 keep exactly their old membership and planes 3-5 are additive, so
# every plane a trained policy already reads is unchanged. Planes are the
# leading dimension of the flattened grid, so new ones append at the end and a
# widened encoder transfers exactly. Plane 3 deliberately overlaps plane 1.
#
# **Back to 3 for the v12 reconstruction.** It was raised to 6 for floor-v14, on
# the reasoning that a fresh run has no warm start to keep exact and therefore no
# reason to leave the extra planes dark. That reasoning was sound and the result
# was not: v14 through v18 declined monotonically, 0.26 -> 0.19 -> 0.12 -> 0.08
# cleared, while the observation grew from a 405-value grid to 945.
#
# The planes were never the whole cause — measured on floor-v18 at 665k, moving a
# rock onto the line of fire changed the policy by a total variation of 0.011 and
# spikes on the player by 0.003, so the extra planes were not being read at all.
# But nothing since v12 has been measured against a baseline that works, and the
# way back is to reproduce v12 first and add one thing at a time after.
#
# Bits 0-2 reproduce v12's encoding bit for bit, so this constant is the entire
# switch: the mod still reports the full bitmask and `GRID_ALL_CLASSES` still
# names it, so the tests can tell "not enabled" from "broken".
GRID_CLASSES = 3
# What the mod actually reports, and what 6 would expose. Kept as its own name
# so the tests can tell "not enabled yet" from "broken".
GRID_ALL_CLASSES = 6
GRID_FEATURES = GRID_WIDTH * GRID_HEIGHT * GRID_CLASSES

# The same obstacles again, but centred on the player.
#
# The room-absolute grid above is indexed by room tile and goes through a flat
# Linear, while everything the agent aims and steers with is player-relative —
# and the entity branch is *pooled* before the trunk. So "rock at room tile
# (4, 9)" and "enemy at offset (+140, 0)" only ever met as two unrelated
# summaries, and there was no representation in which "is that rock between me
# and that enemy" existed at all. Measured on floor-v10: moving a rock from
# directly on the line of fire to a far corner changed the policy by a total
# variation of 0.019, and spikes placed against the player changed movement by
# 0.022. The agent was reacting to how many obstacles existed, not where.
#
# This is the same defect the entity encoder had and had fixed — room-absolute
# positions there were worth 37.5 points of success once made player-relative.
#
# Built from the *raw* tile array rather than the downsampled 15x9 frame, so
# local detail is exact even in a large room. Radius 4 spans a 9x9 window, which
# covers the full height of a 1x1 room and most of its width.

EGO_RADIUS = 4
EGO_SIZE = 2 * EGO_RADIUS + 1
EGO_FEATURES = EGO_SIZE * EGO_SIZE * GRID_CLASSES

# Movement and shooting, each as one of {-1, 0, +1} per axis.
AXIS_VALUES = (-1, 0, 1)
ACTION_DIMS = (3, 3, 3, 3)
# Which heads are movement and which are shooting. Worth reporting separately,
# because summed entropy misleads whenever one pair has no gradient: during
# navigation the shoot heads have nothing to learn from and the entropy bonus
# pins them at uniform, contributing an irreducible 2 * ln(3) = 2.197. A total
# of 3.2 therefore means movement is already about half converged, not that the
# policy is still near-random.
MOVE_HEADS = (0, 1)
SHOOT_HEADS = (2, 3)

ENTITY_KINDS = {"enemy": 0, "projectile": 1, "pickup": 2, "bomb": 3}
DOOR_CATEGORY_INDEX = {name: index for index, name in enumerate(DOOR_CATEGORIES)}

# A restart is a few seconds of teardown and floor generation, so allow well
# over that, but not so long that a wedged instance stalls the fleet.
MAX_RESET_TICKS = 300
RESET_REISSUE_TICKS = 90


@dataclass
class StepResult:
    observation: dict[str, np.ndarray]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


def _normalize_position(value: float, low: float, high: float) -> float:
    span = high - low
    if span <= 0:
        return 0.0
    return float(np.clip((value - low) / span * 2.0 - 1.0, -1.5, 1.5))


def encode_observation(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Turn one raw JSON observation into arrays a network can consume."""
    player = obs["player"]
    room = obs["room"]
    level = obs["level"]

    left = float(room["top_left_x"])
    top = float(room["top_left_y"])
    right = float(room["bottom_right_x"])
    bottom = float(room["bottom_right_y"])

    px = _normalize_position(float(player["x"]), left, right)
    py = _normalize_position(float(player["y"]), top, bottom)

    scalars = np.array([
        px,
        py,
        float(player["vx"]) / 10.0,
        float(player["vy"]) / 10.0,
        float(player["hearts"]) / 12.0,
        float(player["max_hearts"]) / 12.0,
        float(player["soul_hearts"]) / 12.0,
        float(player["bombs"]) / 10.0,
        float(player["keys"]) / 10.0,
        float(player["coins"]) / 50.0,
        float(player["damage"]) / 10.0,
        float(player["speed"]) / 2.0,
        float(player["tear_delay"]) / 20.0,
        float(player["range"]) / 500.0,
        1.0 if player["can_fly"] else 0.0,
        1.0 if room["clear"] else 0.0,
        float(level["stage"]) / 12.0,
        float(room["type"]) / 30.0,
        float(room.get("enemies_alive", 0)) / 10.0,
        float(level.get("rooms_visited", 0)) / 20.0,
        # How much of the floor is still unseen — the exploration signal.
        max(0.0, float(level.get("rooms_total", 0))
            - float(level.get("rooms_visited", 0))) / 20.0,
    ], dtype=np.float32)

    # Appended, never inserted. `_warm_start` widens a layer by keeping the old
    # columns and zeroing the new ones, so a policy resumed across this change
    # starts out computing exactly what it did before and the rays earn their
    # influence by gradient. Inserting anywhere earlier would shift every later
    # column and scramble the transfer with no error at all.
    if ENABLE_OBSTACLE_RAYS:
        scalars = np.concatenate([
            scalars,
            encode_obstacle_rays(room, float(player["x"]), float(player["y"]),
                                 player.get("grid_index")),
        ])

    entities = np.zeros((MAX_ENTITIES, ENTITY_FEATURES), dtype=np.float32)
    mask = np.zeros((MAX_ENTITIES,), dtype=np.float32)

    player_x = float(player["x"])
    player_y = float(player["y"])
    span = max(right - left, 1.0)

    for slot, entity in enumerate(obs.get("entities", [])[:MAX_ENTITIES]):
        kind = ENTITY_KINDS.get(entity.get("k", ""), 0)
        max_hp = float(entity.get("mhp") or 0.0)
        hp = float(entity.get("hp") or 0.0)

        # Player-relative geometry, available inside the per-entity embedding.
        dx = float(entity["x"]) - player_x
        dy = float(entity["y"]) - player_y
        distance = math.hypot(dx, dy)
        # The unit vector is what aiming actually needs: which way to shoot,
        # independent of how far away the target is.
        unit_x = dx / distance if distance > 1e-6 else 0.0
        unit_y = dy / distance if distance > 1e-6 else 0.0

        # Velocity relative to the player, split along the line to it. Closing
        # is positive when the gap is shrinking — a charger reads high and a
        # wanderer reads near zero, which is the difference between a threat
        # and scenery. Tangential is what orbiting an enemy looks like.
        relative_vx = float(entity["vx"]) - float(player["vx"])
        relative_vy = float(entity["vy"]) - float(player["vy"])
        closing = -(relative_vx * unit_x + relative_vy * unit_y)
        tangential = relative_vx * -unit_y + relative_vy * unit_x

        entities[slot] = (
            _normalize_position(float(entity["x"]), left, right),
            _normalize_position(float(entity["y"]), top, bottom),
            float(entity["vx"]) / 10.0,
            float(entity["vy"]) / 10.0,
            hp / max(max_hp, 1.0),
            min(max_hp / 50.0, 1.0),
            1.0 if kind == 0 else 0.0,
            1.0 if kind == 1 else 0.0,
            1.0 if kind == 2 else 0.0,
            1.0 if entity.get("boss") else 0.0,
            dx / span,
            dy / span,
            min(distance / span, 2.0),
            unit_x,
            unit_y,
            closing / 10.0,
            tangential / 10.0,
            # Semantic identity, resolved in Lua from the game's own constants.
            # These are APPENDED and must stay last: `_warm_start` transfers a
            # widened layer by keeping the old columns and zeroing the new ones,
            # so a policy trained before them keeps computing exactly what it
            # did. Inserting anywhere earlier shifts every later column and
            # scrambles the transfer silently.
            1.0 if entity.get("consumable") else 0.0,
            1.0 if entity.get("pedestal") else 0.0,
            1.0 if entity.get("chest") else 0.0,
            1.0 if entity.get("hostile") else 0.0,
            1.0 if entity.get("flying") else 0.0,
        )
        mask[slot] = 1.0

    doors = np.zeros((MAX_DOORS, DOOR_FEATURES), dtype=np.float32)
    for door in obs["room"].get("doors", []):
        slot = int(door.get("slot", -1))
        if not 0 <= slot < MAX_DOORS:
            continue
        # A door needing an explosive is not a door to *this* agent — it has no
        # bomb action at all — so it is left out of the observation entirely and
        # the slot reads as plain wall.
        #
        # Removing it from the shaping was not enough on its own, which is worth
        # keeping in mind as a general lesson. `door_potential` stopped steering
        # at secret rooms in floor-v12, and the agent went on walking into them
        # anyway: the encoded door still said `present=1, unvisited=1`, which is
        # byte-identical to a sacrifice-room door that *is* passable, and the
        # policy carried a million steps of "unvisited doors are worth walking
        # to". Killing the reward removes the reinforcement but leaves the habit
        # and the perception that justifies it, so the behaviour decays only as
        # fast as the absence of reward can erode it.
        #
        # Costs no change of shape, so a resumed policy still transfers exactly.
        if door.get("needs_bomb"):
            continue
        dx = float(door["x"]) - player_x
        dy = float(door["y"]) - player_y
        distance = math.hypot(dx, dy)
        doors[slot, :7] = (
            1.0,                                        # this slot has a door
            1.0 if door.get("open") else 0.0,
            1.0 if door.get("locked") else 0.0,
            # Unvisited is the whole point: it marks where exploring leads.
            0.0 if float(door.get("visited", 0)) > 0 else 1.0,
            dx / span,
            dy / span,
            min(distance / span, 2.0),
        )
        # An unrecognised category leaves every flag at zero rather than being
        # silently folded into "normal", so a broken mapping stays visible.
        category = door.get("category", "unknown")
        if category in DOOR_CATEGORY_INDEX:
            doors[slot, 7 + DOOR_CATEGORY_INDEX[category]] = 1.0

    return {"scalars": scalars, "entities": entities, "entity_mask": mask,
            "doors": doors.reshape(-1), "grid": encode_grid(obs["room"]),
            "ego_grid": encode_egocentric_grid(obs["room"], player_x, player_y,
                                              player.get("grid_index"))}


def tile_of_index(room: dict[str, Any], index: Any) -> tuple[int, int] | None:
    """A grid index straight from the game, split into (row, column).

    Preferred over `tile_at` wherever the mod supplies it, because deriving the
    tile from room extents assumes the playable area spans the whole grid
    interior. That is true of a 1x1 room and false of every "half" shape, where
    the playable region is a sub-rectangle of a 15x9 grid whose remainder is
    wall — the player resolved into a wall tile on 24% of observations in those
    rooms before this was used.
    """
    if index is None:
        return None
    width = int(room.get("grid_width") or 0)
    cells = room.get("grid") or []
    if width <= 0 or not cells:
        return None
    index = int(index)
    if not 0 <= index < len(cells):
        return None
    return index // width, index % width


def tile_at(room: dict[str, Any], x: float, y: float) -> tuple[int, int] | None:
    """The raw grid tile containing a room position, or None if unreadable.

    A fallback for arbitrary points — for the *player*, prefer
    `tile_of_index`, which the game answers exactly. This derivation is only
    correct where the playable area spans the full grid interior.

    Shared rather than restated, because three separate probes have now drifted
    from the encoder by keeping their own copy of a rule and then quietly
    measuring the old one. Anything that needs "which tile is this" imports it.

    The two coordinate systems do not describe the same rectangle.
    `GetTopLeftPos`/`GetBottomRightPos` are the centres of the first and last
    *walkable* tiles — the playable area, inside the wall ring — while
    `GetGridWidth` counts the ring too. Scaling the playable span across the
    full width squashes the player into the wall columns near every edge: the
    player's own tile came back solid on 27% of 18,000 live observations, which
    is impossible. So the span covers the interior only, columns 1..width-2,
    which is width-3 tile *steps* between two tile centres.
    """
    cells = room.get("grid") or []
    width = int(room.get("grid_width") or 0)
    if not cells or width <= 0:
        return None
    height = len(cells) // width
    if height <= 0:
        return None

    left = float(room["top_left_x"])
    top = float(room["top_left_y"])
    span_x = max(float(room["bottom_right_x"]) - left, 1.0)
    span_y = max(float(room["bottom_right_y"]) - top, 1.0)
    # Rounded because the endpoints are tile centres, and clamped to the
    # interior so standing in a doorway still resolves.
    column = int(round((x - left) / span_x * max(width - 3, 1))) + 1
    row = int(round((y - top) / span_y * max(height - 3, 1))) + 1
    return (min(height - 2, max(1, row)), min(width - 2, max(1, column)))


def encode_egocentric_grid(room: dict[str, Any], player_x: float,
                           player_y: float,
                           grid_index: Any = None) -> np.ndarray:
    """The obstacles around the player, in the player's own frame.

    Cell (EGO_RADIUS, EGO_RADIUS) is always the tile the player is standing in,
    so "solid one tile to my right" lands at a fixed index no matter where in
    the room that is. That is the whole point: the room-absolute grid puts the
    same fact at a different index every time the player moves, which is why an
    MLP could never tie it to anything.

    Anything outside the room reads as **solid**, not as empty. Off-map is wall,
    and zero-filling would tell the agent it can walk out through the edge —
    the opposite of the truth, and worse than the room-absolute encoding it
    replaces.
    """
    window = np.zeros((GRID_CLASSES, EGO_SIZE, EGO_SIZE), dtype=np.float32)
    cells = room.get("grid") or []
    width = int(room.get("grid_width") or 0)
    if not cells or width <= 0:
        # Nothing readable: leave it empty rather than claiming walls, so a
        # broken payload does not look like a sealed room.
        return window.reshape(-1)

    height = len(cells) // width
    if height <= 0:
        return window.reshape(-1)
    tiles = np.asarray(cells[:height * width], dtype=np.int16).reshape(height, width)

    # The game's own answer where we have it; the extents derivation only as a
    # fallback, since it is wrong for any room whose playable area does not span
    # the full grid interior.
    located = tile_of_index(room, grid_index) or tile_at(room, player_x, player_y)
    if located is None:
        return window.reshape(-1)
    row, column = located

    # Pad with the solid class and slice, rather than looping the 81 cells in
    # Python. This runs once per instance per action repeat — at 20 instances
    # and ~200 steps/s that is 8000 calls a second, so the constant factor is
    # worth caring about. The padding value is what makes off-map read as wall
    # for free.
    #
    # Written out rather than `np.pad`, which measured **11.7us** on a 9x15
    # array — 45% of the whole encode, all of it generic-path overhead, against
    # ~1us for allocate-and-assign.
    # Padded with the solid *bit* (1 << 0), so off-map reads as wall and nothing
    # else — not as a hazard, and not as a pit.
    padded = np.ones((height + 2 * EGO_RADIUS, width + 2 * EGO_RADIUS),
                     dtype=np.int16)
    padded[EGO_RADIUS:EGO_RADIUS + height,
           EGO_RADIUS:EGO_RADIUS + width] = tiles
    view = padded[row:row + EGO_SIZE, column:column + EGO_SIZE]
    for bit in range(GRID_CLASSES):
        window[bit] = (view & (1 << bit)) != 0
    return window.reshape(-1)


def encode_obstacle_rays(room: dict[str, Any], player_x: float,
                         player_y: float,
                         grid_index: Any = None) -> np.ndarray:
    """How far the player can travel in each of the eight directions, in [0, 1].

    1.0 means nothing blocks within `RAY_MAX_TILES`; 0.0 means the very next tile
    that way is solid. See `RAY_DIRECTIONS` for why this exists at all.

    Blocking is solid **or** pit — the two a walking player cannot cross. Hazard
    is deliberately excluded: spikes hurt but are crossable, and a retreat across
    them is a real option that this feature must not hide. `can_fly` is already
    in the scalar block, so the network can learn to discount pits itself rather
    than having that judgement baked in here.

    Off-map counts as blocked, matching `encode_egocentric_grid`'s solid padding:
    the edge of the room is wall, and reporting it as open would invite exactly
    the retreat that gets the agent killed.
    """
    rays = np.ones(RAY_FEATURES, dtype=np.float32)
    cells = room.get("grid") or []
    width = int(room.get("grid_width") or 0)
    if not cells or width <= 0:
        # Unreadable payload reads as open, not as walled in — the same choice
        # `encode_egocentric_grid` makes, so a dropped grid never looks like a
        # sealed room and never fabricates a "trapped" signal.
        return rays

    height = len(cells) // width
    if height <= 0:
        return rays
    tiles = np.asarray(cells[:height * width], dtype=np.int16).reshape(height, width)

    located = tile_of_index(room, grid_index) or tile_at(room, player_x, player_y)
    if located is None:
        return rays
    row, column = located

    blocking = (1 << GRID_SOLID_BIT) | (1 << GRID_PIT_BIT)
    for index, (dcol, drow) in enumerate(RAY_DIRECTIONS):
        travelled = 0
        for step in range(1, RAY_MAX_TILES + 1):
            r, c = row + drow * step, column + dcol * step
            if not (0 <= r < height and 0 <= c < width):
                break
            if tiles[r, c] & blocking:
                break
            travelled = step
        rays[index] = travelled / RAY_MAX_TILES
    return rays


def encode_grid(room: dict[str, Any]) -> np.ndarray:
    """Split the room's obstacle tiles into property planes on a fixed frame.

    The mod reports a **bitmask** per tile, not a class, so a tile can be solid
    and damaging at once — which a spiked rock is, and which the single-class
    encoding had to throw away.

    An all-zero cell says "nothing here", which keeps the input sparse and means
    a room the mod could not read looks empty rather than looking like walls.

    Rooms larger than 1x1 are sampled down onto the same frame rather than
    truncated, so the layout stays recognisable and the input shape never
    depends on the room.
    """
    grid = np.zeros((GRID_CLASSES, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    cells = room.get("grid") or []
    width = int(room.get("grid_width") or 0)
    if not cells or width <= 0:
        return grid.reshape(-1)

    height = len(cells) // width
    if height <= 0:
        return grid.reshape(-1)

    tiles = np.asarray(cells[:height * width], dtype=np.int16).reshape(height, width)

    if (height, width) == (GRID_HEIGHT, GRID_WIDTH):
        for bit in range(GRID_CLASSES):
            grid[bit] = (tiles & (1 << bit)) != 0
        return grid.reshape(-1)

    # Larger room: each output tile covers a block, and reports every property
    # present anywhere in it. Point-sampling instead would silently drop
    # obstacles — and it did: nearest-neighbour with floor division never
    # reaches the final row or column, so a rock in the far corner of a 2x2
    # room became invisible, which is the exact failure this encoding exists to
    # remove. Over-reporting an obstacle is safe; missing one is not.
    for row in range(GRID_HEIGHT):
        top_row = row * height // GRID_HEIGHT
        bottom_row = max(top_row + 1, (row + 1) * height // GRID_HEIGHT)
        for column in range(GRID_WIDTH):
            left = column * width // GRID_WIDTH
            right = max(left + 1, (column + 1) * width // GRID_WIDTH)
            block = tiles[top_row:bottom_row, left:right]
            merged = int(np.bitwise_or.reduce(block, axis=None))
            for bit in range(GRID_CLASSES):
                if merged & (1 << bit):
                    grid[bit, row, column] = 1.0
    return grid.reshape(-1)


def compute_reward(obs: dict[str, Any], config: AppConfig,
                   step_penalty: float | None = None) -> tuple[float, dict[str, Any]]:
    """Thin, mostly-sparse reward. Credit assignment is the learner's job.

    Deliberately contains nothing for room transitions. The underlying event
    fires on *every* crossing, including back into somewhere already cleared,
    so paying it here let an agent shuttle between two rooms indefinitely.
    Exploration is scored in the floor environment, which tracks which rooms
    have actually been visited this episode.
    """
    events = obs["events"]
    rewards = config.rewards

    # Callers with no shaping term of their own can price idling higher; the
    # shared value has to stay under one step of door_shaping for the floor and
    # navigation tasks.
    #
    # Built as a breakdown, with the total summed *from* it. Accumulating a
    # scalar here and deriving the split separately for logging is how you end
    # up reporting a decomposition that does not match the reward actually
    # paid — and this project has been wrong about what a number counted a dozen
    # times. Built this way the two cannot disagree.
    #
    # Restored after the v12 reconstruction reverted it away with the rest of the
    # v15-v17 source. It was added in v16 and is the reason the v18 diagnosis
    # could be made at all: `door_shaping` +2.47 an episode against 12.93 of
    # combat-term magnitude is a measurement, and every rebalance argued before
    # it existed was argued from a split inferred from aggregates.
    parts = {
        "step": rewards.step if step_penalty is None else step_penalty,
        "damage_dealt": rewards.damage_dealt * float(events["damage_dealt"]),
        "damage_taken": rewards.damage_taken * float(events["damage_taken"]),
        "kill": rewards.kill * float(events["kills"]),
        "room_clear": rewards.room_clear if events["room_cleared"] else 0.0,
        "new_level": rewards.new_level if events["new_level"] else 0.0,
        "death": rewards.death if events["died"] else 0.0,
    }
    # Derived here, not sent by the mod.
    events["reward_parts"] = parts

    return sum(parts.values()), events


def decode_action(action: np.ndarray | list[int]) -> tuple[int, int, int, int]:
    mx = AXIS_VALUES[int(action[0])]
    my = AXIS_VALUES[int(action[1])]
    sx = AXIS_VALUES[int(action[2])]
    sy = AXIS_VALUES[int(action[3])]
    return mx, my, sx, sy


class IsaacVecEnv:
    """Steps every instance in lockstep.

    Each instance's game thread is blocked inside its socket read, so sending
    an action is what allows that instance to advance exactly one tick.
    """

    def __init__(self, bridges: list[InstanceBridge], config: AppConfig) -> None:
        self.bridges = bridges
        self.config = config
        self.num_envs = len(bridges)
        self.action_repeat = max(1, config.env.action_repeat)
        self.max_episode_steps = config.env.max_episode_steps

        self._episode_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._latest: list[dict[str, Any] | None] = [None] * self.num_envs
        self._failed: list[bool] = [False] * self.num_envs

    # -- low-level helpers -------------------------------------------------

    def _send_all(self, messages: list[dict[str, Any] | None]) -> None:
        """Release every instance for one tick.

        Sending to all of them before reading any is what keeps the fleet
        parallel. Interleaving send/receive per instance would serialise the
        games: each would sit blocked while the previous one ticked, collapsing
        aggregate throughput to a single instance's 30 Hz.
        """
        for index, message in enumerate(messages):
            if self._failed[index] or message is None:
                continue
            try:
                self.bridges[index].send(message)
            except BridgeError as exc:
                print(f"instance {index} dropped out on send: {exc}")
                self._failed[index] = True

    def _receive_all(self, expect: list[bool]) -> list[dict[str, Any] | None]:
        """Collect the tick result from every instance we released."""
        results: list[dict[str, Any] | None] = [None] * self.num_envs
        for index in range(self.num_envs):
            if self._failed[index] or not expect[index]:
                continue
            try:
                results[index] = self.bridges[index].receive()
            except BridgeError as exc:
                print(f"instance {index} dropped out on receive: {exc}")
                self._failed[index] = True
        return results

    def _exchange_all(self, messages: list[dict[str, Any] | None]
                      ) -> list[dict[str, Any] | None]:
        expect = [msg is not None and not self._failed[i]
                  for i, msg in enumerate(messages)]
        self._send_all(messages)
        return self._receive_all(expect)

    def _prime(self) -> None:
        """Read the observation each instance is already blocked on."""
        for index, bridge in enumerate(self.bridges):
            if self._failed[index]:
                continue
            try:
                self._latest[index] = bridge.receive()
            except BridgeError as exc:
                print(f"instance {index} dropped out during prime: {exc}")
                self._failed[index] = True

    # -- gym-ish API -------------------------------------------------------

    def _restart(self, indices: list[int]) -> None:
        """Restart the given instances and wait until each reports a new run.

        Every instance is driven in parallel, so a full-fleet reset costs about
        as long as a single one.
        """
        if not indices:
            return

        pending = {index for index in indices if not self._failed[index]}

        def absorb(results: list[dict[str, Any] | None]) -> None:
            for index, obs in enumerate(results):
                if obs is None:
                    if self._failed[index]:
                        pending.discard(index)
                    continue
                self._latest[index] = obs
                if obs.get("events", {}).get("game_started"):
                    pending.discard(index)

        # The observation returned by the reset exchange is often the one that
        # already carries game_started. Dropping it means waiting forever for a
        # flag that has come and gone.
        absorb(self._exchange_all([
            {"t": "reset"} if index in pending else None
            for index in range(self.num_envs)
        ]))

        # `restart` takes a number of ticks to tear down and rebuild the run.
        # Re-issue periodically: a reset can be swallowed if it lands on a tick
        # where the console is not accepting commands.
        for tick in range(MAX_RESET_TICKS):
            if not pending:
                break
            reissue = tick > 0 and tick % RESET_REISSUE_TICKS == 0
            command = {"t": "reset"} if reissue else {"t": "noop"}
            messages: list[dict[str, Any] | None] = [
                dict(command) if index in pending else None
                for index in range(self.num_envs)
            ]
            absorb(self._exchange_all(messages))

        if pending:
            print(f"warning: instances {sorted(pending)} never reported a new "
                  f"run after {MAX_RESET_TICKS} ticks")

        for index in indices:
            self._episode_steps[index] = 0
            self._episode_returns[index] = 0.0

    def reset(self) -> dict[str, np.ndarray]:
        """Start a fresh run on every instance and wait until each has begun."""
        if all(latest is None for latest in self._latest):
            self._prime()
        self._restart(list(range(self.num_envs)))
        return self._stack_observations()

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray,
                                                 np.ndarray, np.ndarray, list[dict]]:
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

        messages: list[dict[str, Any] | None] = []
        for index in range(self.num_envs):
            if self._failed[index]:
                messages.append(None)
                terminated[index] = True
                continue
            mx, my, sx, sy = decode_action(actions[index])
            messages.append({"t": "act", "mx": mx, "my": my, "sx": sx, "sy": sy,
                             "bomb": False, "item": False})

        # Hold the action for `action_repeat` ticks, advancing the whole fleet
        # together on each one.
        for _ in range(self.action_repeat):
            for index, obs in enumerate(self._exchange_all(messages)):
                if obs is None:
                    if self._failed[index]:
                        terminated[index] = True
                    continue
                self._latest[index] = obs
                if not obs.get("ready", True):
                    continue
                reward, events = compute_reward(obs, self.config)
                rewards[index] += reward
                if events["died"] or obs["player"]["is_dead"]:
                    terminated[index] = True

        for index in range(self.num_envs):
            self._episode_steps[index] += 1
            self._episode_returns[index] += rewards[index]

            if self._episode_steps[index] >= self.max_episode_steps:
                truncated[index] = True

            if terminated[index] or truncated[index]:
                infos[index]["episode"] = {
                    "r": float(self._episode_returns[index]),
                    "l": int(self._episode_steps[index]),
                }

        return self._stack_observations(), rewards, terminated, truncated, infos

    def reset_done(self, done_mask: np.ndarray) -> None:
        """Restart only the instances whose episodes ended."""
        self._restart([int(index) for index in np.flatnonzero(done_mask)])

    def _stack_observations(self) -> dict[str, np.ndarray]:
        encoded = []
        for index in range(self.num_envs):
            latest = self._latest[index]
            if latest is None or not latest.get("ready", True):
                encoded.append({
                    "scalars": np.zeros(SCALAR_FEATURES, dtype=np.float32),
                    "entities": np.zeros((MAX_ENTITIES, ENTITY_FEATURES), dtype=np.float32),
                    "entity_mask": np.zeros(MAX_ENTITIES, dtype=np.float32),
                    "doors": np.zeros(MAX_DOORS * DOOR_FEATURES, dtype=np.float32),
                    "grid": np.zeros(GRID_FEATURES, dtype=np.float32),
                    "ego_grid": np.zeros(EGO_FEATURES, dtype=np.float32),
                })
            else:
                encoded.append(encode_observation(latest))

        return {
            key: np.stack([item[key] for item in encoded])
            for key in ("scalars", "entities", "entity_mask", "doors", "grid",
                        "ego_grid")
        }

    @property
    def alive_count(self) -> int:
        return sum(1 for failed in self._failed if not failed)

    def failed_indices(self) -> list[int]:
        """Which instances have dropped out, for a caller that can relaunch."""
        return [i for i, failed in enumerate(self._failed) if failed]

    def healthy_bridges(self, exclude: int | None = None) -> list[InstanceBridge]:
        """Bridges still in a run, which a relaunch has to keep ticking.

        A connected instance sits blocked inside its socket read and a blocked
        game stops pumping Win32 messages, so anything that holds the fleet open
        while doing slow work — like walking a relaunched instance through the
        menu — must pump these or their windows become unfocusable.
        """
        return [bridge for i, bridge in enumerate(self.bridges)
                if not self._failed[i] and i != exclude]

    def restore(self, index: int) -> bool:
        """Take a relaunched instance back into the fleet.

        The game is freshly in a run and already blocked on its first send, so
        the observation is waiting to be read — the same situation `_prime`
        handles at start-up, and read the same way.

        Clearing `_failed` is deliberately last: until the first observation is
        in hand there is nothing for `step` to work from, and a half-restored
        instance that `step` believes in would stall the fleet on a read that
        never comes.
        """
        try:
            self._latest[index] = self.bridges[index].receive()
        except BridgeError as exc:
            print(f"instance {index} failed to hand over after relaunch: {exc}")
            return False
        self._failed[index] = False
        return True
