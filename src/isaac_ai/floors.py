"""Floor progression: play an actual run instead of isolated fights.

An episode is one attempt at a floor. The agent starts in the start room and has
to find doors, fight through rooms, and survive on a single health pool.

**Why episodes never heal mid-attempt.** In combat training an episode was one
encounter, so conceding at low health and respawning enemies kept runs alive and
resets cheap. That trick does not transfer: health carrying across rooms is the
whole point of a floor, so healing mid-episode would delete the
resource-management problem we are trying to teach.

**How episodes reset.** `reseed` where possible, `restart` where necessary.
A full restart clears stats, items and progression and rebuilds the run from
scratch, which measured at 0.91s and cost floor-v5 27% of its wall clock across
1698 of them. `reseed` generates a new layout for the current floor and keeps
items and progression, at 0.28s. Two cases still need the expensive path:

  died          a dead player cannot be reseeded around; the run is over
  picked up     reseed keeping items is exactly what makes it cheap, and also
                what would let pickups compound across every later episode
                until the environment is no longer stationary

Reseed also leaves health alone, so the heal is sent with it. Verified over
twenty consecutive resets by `probe_reseed.py`: floors genuinely regenerate,
the player lands in the starting room, is never wedged, and stats hold.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from isaac_ai.bridge import InstanceBridge
from isaac_ai.config import AppConfig
from isaac_ai.env import IsaacVecEnv, compute_reward, decode_action
from isaac_ai.floor_curriculum import FloorCurriculum

import math

# Floor generation lands within a couple of ticks after `reseed`; a few more
# costs nothing next to the 0.63s saved against a full restart.
RESEED_SETTLE_TICKS = 6

# A step that travels less than this while asking to move is not slow, it is
# obstructed. Base Isaac covers several units a tick and action_repeat is 2, so
# a real step moves roughly ten. Calibrated against a random-action control on a
# 20-instance fleet, which registers 2.5% blocked — the floor a policy that
# simply does not steer into walls would sit near.
MOVED_UNITS = 1.0


def door_is_targetable(door: dict[str, Any], unvisited_only: bool = True) -> bool:
    """Whether the shaping may steer at this door.

    Extracted so there is exactly one copy. `probe_door_targets.py` originally
    reimplemented this filter to report what the potential was chasing, and when
    the secret-room case was fixed here the probe went on measuring the old
    behaviour and reported the bug as still present. Any diagnostic that asks
    "what would the shaping do" has to call this rather than restate it.

    `locked` is not the same question as "can I get through this". A secret
    room's door is closed and **unlocked**, and opening it needs a bomb this
    agent has no action for at all — `floors.py` hardcodes `bomb: False`. So
    targeting one means walking confidently into a wall until the idle limit
    ends the attempt and the floor reseeds.

    Measured over 900 fleet steps before the fix: secret doors were seen 2450
    times, were never once open, were never locked, and the potential steered at
    one on 1068 observations — 6% of every step where it had a target. A locked
    library door was correctly skipped 533/533 times over the same run, which is
    what makes this a gap in the filter rather than in the idea.

    Deliberately *not* a check on `open`: doors shut for a fight reopen on the
    clear, and filtering them out would collapse the potential the moment a
    fight starts — a discontinuity paid on entering a room, which is the exact
    shape that plateaued floor-v1 through v3 at -2.885 a transition.
    """
    if unvisited_only and float(door.get("visited", 0)) > 0:
        return False
    if door.get("locked"):
        return False
    if door.get("needs_bomb"):
        return False
    return True


def door_potential(obs: dict[str, Any], unvisited_only: bool = True) -> float:
    """Closeness to the nearest usable door, in [0, 1].

    `unvisited_only` must match what the task actually rewards. On a floor,
    progress means reaching somewhere new, so doors leading back are excluded.
    In the navigation task *any* traversal is success — including back the way
    it came — and a dead-end room has only that one door, so excluding it would
    leave the agent with a flat potential and no guidance toward the one exit
    that ends the episode.

    The range is deliberately non-negative. Shaping around a cycle sums to
    (gamma - 1) * sum(potentials); with a negative potential that product is
    POSITIVE, so a two-room loop paid out about +0.01 a lap. Keeping the
    potential at or above zero makes any cycle cost slightly more than it
    returns, which is the direction that cannot be exploited.

    Used for potential-based shaping. Room clears and new rooms are far apart in
    time, so a policy has no gradient telling it which way to walk — floor-v1
    spent half its episodes wandering until the clock ran out. Shaping of the
    form gamma*phi(s') - phi(s) supplies that gradient and is provably
    policy-invariant, so it cannot introduce a behaviour that the sparse reward
    would not also have preferred.
    """
    if not obs or not obs.get("ready", True):
        return 0.0
    room = obs.get("room")
    if not room:
        return 0.0

    span = max(float(room["bottom_right_x"]) - float(room["top_left_x"]), 1.0)
    player = obs["player"]
    best: float | None = None

    for door in room.get("doors", []):
        if not door_is_targetable(door, unvisited_only):
            continue
        distance = math.dist((float(door["x"]), float(door["y"])),
                             (float(player["x"]), float(player["y"])))
        if best is None or distance < best:
            best = distance

    if best is None:
        # Nothing left to explore here: flat potential, no pull in any direction.
        return 0.0
    return 1.0 - min(best / span, 1.0)


# floor-v22's single variable: drop the door term while the room has live
# enemies, so navigation stops competing with combat inside a fight.
#
# Measured by `probe_door_pull.py` over 30,000 observations under floor-v21:
# **70.7%** of all steps have live enemies, and on **97.1%** of those the shaping
# still has a targetable door, at a mean potential of **0.728** — *higher* than
# the 0.658 it reads in a quiet room. The agent spends most of its life being
# paid to walk away from the thing it is scored on killing.
#
# This corrects a claim STATUS made for several runs: "Isaac locks the doors
# during a fight and `door_potential` skips locked doors". It does not. Of 92,881
# doors observed, **77.5% were shut but only 0.4% were locked** — combat bars a
# door without locking it, and `door_is_targetable` deliberately does not check
# `open`. So the suppression everyone assumed was already happening never was.
#
# Why a gate rather than a smaller `door_shaping`: a coefficient cut weakens
# navigation everywhere, and that has a documented failure here (nav-v1's policy
# gradient came out ten times smaller than the entropy bonus). This removes the
# pull *only* where it competes and leaves it at full strength everywhere else.
#
# **Back on for floor-v28.** floor-v22 measured this as neutral and it was
# reverted; the premise has since changed, which is the only reason to retest a
# change that already failed once.
#
# v22 ran on a policy with `fires ok` at 25% and 0.528 cleared. Removing the
# navigation gradient from a contested room did nothing because there was nothing
# better to do in one — it could not fight. That is also why `ended_idle` barely
# moved: it was not using door shaping to decide anything inside a fight, and had
# no alternative either. The question "does the door pull stop it finishing
# rooms" was put to an agent that could not finish rooms regardless.
#
# floor-v27b can fight. `fires ok` is 84%, it kills **6.64** enemies an episode —
# and finishes **0.85** rooms, clearing only **35%** of the rooms it enters. Two
# rooms' worth of killing converted into less than one clear. Meanwhile
# `door_shaping` is now the largest single term in the reward at **+9.30** an
# episode, and `probe_door_pull.py` measured it live on **97.1%** of combat steps
# at a mean potential of 0.728.
#
# So the agent is paid to walk out of fights it is now capable of winning. That
# is a different situation from the one v22 tested.
#
# **Tripwire, unchanged and this time a real risk:** `ended_idle` is 0.453. If it
# climbs while `rooms_cleared` does not, the gate is starving navigation rather
# than protecting combat, and it reverts again — permanently.
#
# The v22 evidence, kept because it is still the reason to be sceptical:
#
# v22 (1M, resumed from v21) reached cleared 0.528 against v21's 0.470 — but v21
# was already climbing at +0.0089 per 100k, which predicts ~0.559 for the same
# million with no change at all, and v22's own slope was +0.0091, i.e. identical.
# Its first band came in *below* v21's endpoint and then climbed at exactly the
# pre-existing rate. That is a reward change paying for recalibration and buying
# nothing structural.
#
# Two things it did establish, both worth keeping:
#
# 1. `ended_idle` was 0.391 against v21's 0.398. Removing the entire navigation
#    gradient from 70% of the agent's steps changed idling **not at all**, so the
#    agent was never using door shaping to decide what to do inside a fight.
#    That is also why suppressing it changed so little.
#
# 2. The mechanism test failed. The premise was that removing the door pull would
#    let the combat behaviour visible in the doors-removed control surface in
#    normal play. Instead the *control* collapsed:
#
#        v21  doors present 12/32 fires ok, 3 distinct | removed 28/32, 6 distinct
#        v22  doors present  8/32 fires ok, 4 distinct | removed 16/32, 5 distinct
#
#    The gap narrowed because the doors-removed number fell, not because normal
#    play improved. So doors dominate through the **observation**, not through
#    the reward, and no shaping change can reach that.
SUPPRESS_SHAPING_IN_COMBAT = True


def floor_potential(obs: dict[str, Any]) -> float:
    """Progress across the whole floor, in [0, 1]. Continuous through doors.

    `door_potential` alone cannot be used for this. Measured on a real
    transition: standing on an unvisited door it reads 0.990, and one step later
    inside the new room — that door now visited, the next one across the room —
    it reads 0.019. At a shaping coefficient of 4 that pays **-3.885** for the
    step, against a `new_room` bonus of +1.00. Walking through a door cost 2.9
    reward, so floor-v1 through v3 all plateaued around 0.25 rooms an episode.
    The agent was being punished precisely for the behaviour being asked of it.

    The fix is to measure the potential in *floor* units rather than room units:

        rooms_visited + closeness to the nearest unvisited door

    The local term is worth exactly one room, so arriving at a new room consumes
    it (closeness 1 -> 0) at the same moment `rooms_visited` gains 1, and the two
    cancel exactly. Crossing a room raises the potential by up to 1; crossing the
    threshold changes nothing. No discontinuity is left to punish.

    Deliberately *not* divided by `rooms_total`. Normalising to [0, 1] shrinks
    one step of progress by a factor of the floor size, which took the per-step
    gradient to 0.0026 against a step penalty of 0.002 — the same order, which is
    precisely the regime that left nav-v1 maximising randomness instead of
    learning. It stays unnormalised because the shaping is applied undiscounted,
    so an unbounded potential costs nothing: the usual objection to one is the
    -(1-gamma)*phi leak, and at gamma = 1 there is no leak. It remains bounded
    over an episode by the number of rooms on the floor.
    """
    if not obs or not obs.get("ready", True):
        return 0.0
    level = obs.get("level") or {}
    visited = float(level.get("rooms_visited", 0) or 0)

    if SUPPRESS_SHAPING_IN_COMBAT:
        enemies = int((obs.get("room") or {}).get("enemies_alive", 0) or 0)
        if enemies > 0:
            # No local term while the room is contested, so the only gradient
            # during a fight is the combat one. See the constant's note.
            #
            # This cannot punish entering a combat room. The local term is worth
            # at most 1 and `rooms_visited` gains exactly 1 on entry, so the step
            # is `+1 - closeness_before >= 0` — the same cancellation that makes
            # an ordinary door crossing free, and the reason this is a gate on
            # the local term rather than on the whole potential.
            return visited
    return visited + door_potential(obs)


class FloorVecEnv(IsaacVecEnv):
    def __init__(self, bridges: list[InstanceBridge], config: AppConfig,
                 curriculum: FloorCurriculum,
                 max_floor_steps: int = 3000,
                 gamma: float = 0.99) -> None:
        super().__init__(bridges, config)
        self.curriculum = curriculum
        self.max_floor_steps = max_floor_steps
        self.gamma = gamma
        self.shaping_coef = config.rewards.door_shaping
        self._potential = np.zeros(self.num_envs, dtype=np.float64)

        self._rooms_cleared = np.zeros(self.num_envs, dtype=np.int32)
        self._rooms_seen = np.zeros(self.num_envs, dtype=np.int32)
        self._room_transitions = np.zeros(self.num_envs, dtype=np.int32)
        # Which rooms this episode has actually been in. MC_POST_NEW_ROOM fires
        # on every transition, including walking back somewhere already cleared,
        # so it cannot be used to decide what is new.
        self._visited: list[set[int]] = [set() for _ in range(self.num_envs)]
        self._stages_descended = np.zeros(self.num_envs, dtype=np.int32)
        self._idle_steps = np.zeros(self.num_envs, dtype=np.int32)
        # Per-episode reward breakdown. Inferring the split from aggregates is
        # what the v15 rebalance proposal had to do, and an inferred split is a
        # guess wearing a number's clothes.
        self._parts: list[dict[str, float]] = [
            {} for _ in range(self.num_envs)]
        self.restarts = 0
        self.room_exits = 0  # unused on floors; kept so logging stays uniform
        # Movement outcomes, so the blocked-move penalty is measurable from
        # metrics.jsonl instead of needing a fleet probe every time. The two to
        # watch are `blocked_rate` falling towards the 2.5% random-walk floor,
        # and `still_rate` NOT climbing — standing still is the one refuge from
        # the penalty, so it going up means the cure has become the disease.
        self.move_requests = 0
        self.blocked_moves = 0
        self.still_steps = 0
        # Counted here because nothing else was counting it. The trainer logs
        # getattr(env, "deaths", 0), and neither this env nor the combat one
        # defined the attribute, so every floor and combat run ever logged has
        # reported deaths = 0 while the agent died constantly.
        self.deaths = 0
        # Which instances ended their last episode by dying. A dead player
        # cannot be reseeded around, so only those need the expensive restart.
        self._died_last = np.zeros(self.num_envs, dtype=bool)
        # What an untouched run carries, per instance. `reseed` keeps items and
        # progression by design — that is what makes it cheap — so anything the
        # agent acquires would carry across every later episode and the
        # environment would stop being stationary. Comparing against this
        # catches it and forces a real restart.
        #
        # One baseline per instance, seeded from the observation immediately
        # after that instance's own full restart. A single shared baseline
        # seeded lazily from whichever instance happened to finish an episode
        # first records whatever *that* run had already acquired, and then every
        # instance is measured against it forever.
        self._baseline: list[tuple | None] = [None] * self.num_envs

    # -- episode lifecycle -------------------------------------------------

    def _begin_episode(self, indices: list[int]) -> None:
        for index in indices:
            # Seed the potential from the state the episode actually starts in,
            # or the first shaping term would be a spurious jump from zero.
            self._potential[index] = floor_potential(self._latest[index])
            self._visited[index] = set()
            latest = self._latest[index]
            if latest and latest.get("room"):
                # The starting room is not an exploration achievement.
                self._visited[index].add(latest["room"]["index"])
            self._rooms_cleared[index] = 0
            self._rooms_seen[index] = 0
            self._room_transitions[index] = 0
            self._stages_descended[index] = 0
            self._idle_steps[index] = 0
            self._episode_steps[index] = 0
            self._episode_returns[index] = 0.0
            self._parts[index] = {}

    def reset(self) -> dict[str, np.ndarray]:
        if all(latest is None for latest in self._latest):
            self._prime()
        self._restart(list(range(self.num_envs)))
        self.restarts += self.num_envs
        self._seed_baseline(list(range(self.num_envs)))
        self._begin_episode(list(range(self.num_envs)))
        return self._stack_observations()

    def _stats_of(self, index: int) -> tuple | None:
        """Everything a `reseed` would preserve, as one comparable fingerprint.

        Derived stats alone are not enough, and that is not a corner case: a
        familiar, a tear modifier, a trinket, a held card and every consumable
        move none of damage/speed/range/tear_delay/max_hearts, so an instance
        carrying any of them read as pristine and kept it for the rest of the
        run. `collectibles` is the direct count and catches the passives; the
        stats stay as a backstop for anything that changes them without adding
        an item (a devil deal's heart cost, say).

        Consumables are included because the network is *shown* them —
        `bombs`, `keys` and `coins` are three of the scalar inputs — so letting
        them ride across a reseed changes the observation distribution, not
        just the game state.
        """
        latest = self._latest[index]
        if not latest or not latest.get("player"):
            return None
        player = latest["player"]
        return (round(float(player.get("damage", 0)), 2),
                round(float(player.get("speed", 0)), 2),
                round(float(player.get("range", 0)), 1),
                round(float(player.get("tear_delay", 0)), 1),
                int(player.get("max_hearts", 0)),
                int(player.get("collectibles", 0)),
                int(player.get("trinket0", 0)),
                int(player.get("trinket1", 0)),
                int(player.get("card0", 0)),
                int(player.get("pill0", 0)),
                int(player.get("active_item", 0)),
                int(player.get("bombs", 0)),
                int(player.get("keys", 0)),
                int(player.get("coins", 0)))

    def _seed_baseline(self, indices: list[int]) -> None:
        """Record what these instances carry, straight after a full restart.

        A restart rebuilds the run from the frozen save, so this is the one
        moment an instance is known to be pristine. Seeding anywhere else
        bakes in whatever it had already collected.
        """
        for index in indices:
            stats = self._stats_of(index)
            if stats is not None:
                self._baseline[index] = stats

    def _picked_something_up(self, index: int) -> bool:
        """Whether this instance has acquired anything since its last restart."""
        baseline = self._baseline[index]
        if baseline is None:
            return False
        current = self._stats_of(index)
        if current is None:
            return False
        return current != baseline

    def _reseed(self, indices: list[int]) -> None:
        """Regenerate the floor without restarting the run.

        Measured against `restart`: 0.28s versus 0.91s, over resets that were
        costing 27% of floor-v5's wall clock. `reseed` skips the run teardown
        and character re-initialisation and only pays for floor generation.

        Verified safe by `probe_reseed.py` over twenty consecutive resets: the
        floor genuinely regenerates (rooms_total moved between 13 and 17), the
        player lands in the starting room with rooms_visited back to 1, is never
        wedged, and stats do not drift. It does *not* restore health — reseed
        leaves the player as they were — so the heal goes with it.
        """
        if not indices:
            return
        heal: list[dict[str, Any] | None] = [None] * self.num_envs
        reseed: list[dict[str, Any] | None] = [None] * self.num_envs
        for index in indices:
            reseed[index] = {"t": "command", "value": "reseed"}
            heal[index] = {"t": "scenario", "enemies": [], "heal": True,
                           "reposition": False, "new_room": False}

        self._exchange_all(reseed)
        self._exchange_all(heal)
        # Floor generation lands within a couple of ticks; a few more costs
        # nothing and keeps the first observation of the episode coherent.
        for _ in range(RESEED_SETTLE_TICKS):
            settle: list[dict[str, Any] | None] = [
                {"t": "noop"} if reseed[i] is not None else None
                for i in range(self.num_envs)
            ]
            for index, obs in enumerate(self._exchange_all(settle)):
                if obs is not None:
                    self._latest[index] = obs

    def reset_done(self, done_mask: np.ndarray) -> None:
        indices = [int(i) for i in np.flatnonzero(done_mask) if not self._failed[i]]
        if not indices:
            return

        # A dead player cannot be reseeded around — the run is over and only a
        # real restart brings the character back. Everything else (timeouts,
        # stalls) keeps a live player, so it takes the cheap path. Deaths were
        # 26% of episodes in floor-v5, so roughly three resets in four become
        # a third of the price.
        # Reseed keeps items, so an instance that has collected anything must
        # take the expensive path or its pickups compound across every later
        # episode of that run.
        needs_restart = [i for i in indices
                         if self._died_last[i] or self._picked_something_up(i)]
        reseedable = [i for i in indices if i not in set(needs_restart)]

        if needs_restart:
            self._restart(needs_restart)
            self.restarts += len(needs_restart)
            self._seed_baseline(needs_restart)
        self._reseed(reseedable)
        for index in indices:
            self._died_last[index] = False
        self._begin_episode(indices)

    # -- stepping ----------------------------------------------------------

    def step(self, actions: np.ndarray):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

        messages: list[dict[str, Any] | None] = []
        # What each instance was told to do, kept so the blocked-move check can
        # tell "asked to move and could not" from "chose to stand still".
        intents: list[tuple[int, int]] = [(0, 0)] * self.num_envs
        for index in range(self.num_envs):
            if self._failed[index]:
                messages.append(None)
                terminated[index] = True
                continue
            mx, my, sx, sy = decode_action(actions[index])
            intents[index] = (mx, my)
            messages.append({"t": "act", "mx": mx, "my": my, "sx": sx, "sy": sy,
                             "bomb": False, "item": False})

        # Where everyone stood before the action was applied. Captured once per
        # *agent step*, not once per tick: the player accelerates, so the first
        # tick after a direction change travels very little even in open floor,
        # and charging per tick counted that as obstruction. Measured, it took
        # the blocked rate from 14.2% to 22.7% — the gap being acceleration, not
        # walls. The threshold is calibrated against a whole step anyway.
        started = [(latest or {}).get("player") for latest in self._latest]

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
                for name, value in events.get("reward_parts", {}).items():
                    if value:
                        self._parts[index][name] =                             self._parts[index].get(name, 0.0) + value


                # A room is only worth paying for the first time it is entered.
                # Rewarding the transition event instead let the agent shuttle
                # between two cleared rooms for +1.0 a crossing, forever.
                if events.get("new_room"):
                    self._room_transitions[index] += 1
                    room_index = obs.get("room", {}).get("index")
                    if room_index is not None and room_index not in self._visited[index]:
                        self._visited[index].add(room_index)
                        self._rooms_seen[index] += 1
                        reward += self.config.rewards.new_room
                        self._parts[index]["new_room"] =                             self._parts[index].get("new_room", 0.0)                             + self.config.rewards.new_room
                        # Only genuine progress may reset the stall timer, or an
                        # oscillating agent would never time out.
                        self._idle_steps[index] = 0

                # Potential-based shaping: gamma*phi(s') - phi(s).
                # Undiscounted shaping: phi(s') - phi(s), not gamma*phi(s') -
                # phi(s). With gamma < 1 a potential that stays high leaks
                # -(1-gamma)*phi every step, which at coef 4 and phi near 1 is a
                # -0.04 per-step drag on an agent that is doing well. Setting it
                # to 1 removes the leak and still telescopes to
                # phi(end) - phi(start) over the episode.
                potential = floor_potential(obs)
                shaped = self.shaping_coef * (potential - self._potential[index])
                reward += shaped
                self._parts[index]["door_shaping"] =                     self._parts[index].get("door_shaping", 0.0) + shaped
                self._potential[index] = potential

                rewards[index] += reward

                if events.get("room_cleared"):
                    self._rooms_cleared[index] += 1
                    self._idle_steps[index] = 0
                if events.get("new_level"):
                    self._stages_descended[index] += 1
                    self._idle_steps[index] = 0

                if events.get("died") or obs["player"]["is_dead"]:
                    if not terminated[index]:
                        self.deaths += 1
                        self._died_last[index] = True
                        if not events.get("died"):
                            # `compute_reward` pays the death penalty only on
                            # the events flag, which the mod sets from
                            # MC_POST_GAME_END — and that fires as the game-over
                            # screen takes over, which is precisely where mod
                            # callbacks stop running, so the observation
                            # carrying it is usually never sent. `is_dead` is
                            # visible during the death animation and is what
                            # actually arrives. Without this the -10 was inert:
                            # the agent died freely for every floor run, and
                            # the deaths counter read 0 while it happened.
                            rewards[index] += self.config.rewards.floor_death
                            self._parts[index]["floor_death"] =                                 self._parts[index].get("floor_death", 0.0)                                 + self.config.rewards.floor_death
                    terminated[index] = True

        # Did the movement it asked for actually happen, over the whole step?
        # A blocked step is otherwise nearly free: no position change means no
        # potential change, so grinding a wall and drifting in the open pay
        # exactly the same. That absent gradient is why the obstacle grid, and
        # then the egocentric version of it, went unread for three runs —
        # perception cannot beat a reward that does not care.
        #
        # Standing still is deliberately *not* charged. It is strictly better
        # than being blocked and earns nothing either way, so it is a refuge
        # rather than a strategy; `still_rate` is logged so that if it becomes
        # one, it is visible.
        for index in range(self.num_envs):
            if self._failed[index]:
                continue
            if intents[index] == (0, 0):
                self.still_steps += 1
                continue
            prior, latest = started[index], self._latest[index]
            player = (latest or {}).get("player")
            if prior is None or player is None:
                continue
            self.move_requests += 1
            travelled = math.dist(
                (float(player["x"]), float(player["y"])),
                (float(prior.get("x", 0.0)), float(prior.get("y", 0.0))))
            if travelled < MOVED_UNITS:
                self.blocked_moves += 1
                rewards[index] += self.config.rewards.blocked_move
                self._parts[index]["blocked_move"] =                     self._parts[index].get("blocked_move", 0.0)                     + self.config.rewards.blocked_move

        target = self.curriculum.target_rooms()
        for index in range(self.num_envs):
            self._episode_steps[index] += 1
            self._episode_returns[index] += rewards[index]
            self._idle_steps[index] += 1

            # Hitting the curriculum target does NOT end the episode. Ending it
            # there would cap rooms_cleared at the target, making it impossible
            # to tell an agent that clears exactly the bar from one that would
            # have kept going — and it would delete the health-across-rooms
            # pressure that is the whole reason floors exist. A floor attempt
            # runs until the agent dies, stalls, or runs out of time; the target
            # is only how success is scored afterwards.
            reached = self._rooms_cleared[index] >= target

            # Why the attempt ended. "Episode over" lumps together three things
            # with completely different fixes — dying is a combat problem, the
            # 3000-step cap is a pacing problem, and the idle limit is usually
            # the agent pacing between rooms it has already emptied. Without
            # this split there is no way to size any of them.
            reason = None
            if terminated[index]:
                reason = "died" if self._died_last[index] else "dropped"

            # A floor attempt that has stopped making any progress is not worth
            # the game time; end it rather than burn thousands of ticks.
            if self._idle_steps[index] >= self.curriculum.idle_limit:
                truncated[index] = True
                reason = reason or "idle"
            if self._episode_steps[index] >= self.max_floor_steps:
                truncated[index] = True
                reason = reason or "timeout"

            # A failed instance has `terminated` forced true on *every* step, so
            # without this it would emit a fresh zero-return episode every step
            # and flood the window. It never mattered before because a dropped
            # instance used to take the whole trainer down with it; now that a
            # crash only costs one instance, it would.
            if (terminated[index] or truncated[index]) and not self._failed[index]:
                success = bool(reached)
                self.curriculum.record(success, int(self._rooms_cleared[index]))
                # Transitions per distinct room is the backtracking ratio: 1.0
                # means every crossing went somewhere new, while a large value
                # means the agent is pacing between rooms it has already seen.
                seen = max(int(self._rooms_seen[index]), 1)
                infos[index]["episode"] = {
                    "r": float(self._episode_returns[index]),
                    "l": int(self._episode_steps[index]),
                    "success": success,
                    "rooms_cleared": int(self._rooms_cleared[index]),
                    "rooms_seen": int(self._rooms_seen[index]),
                    "transitions": int(self._room_transitions[index]),
                    "backtrack_ratio": float(self._room_transitions[index]) / seen,
                    "descended": int(self._stages_descended[index]),
                    "reason": reason or "unknown",
                    # Whether the room it gave up in still had anywhere to go.
                    # An idle ending with a live door is a stalling problem; one
                    # without is the agent stranded in territory it has already
                    # emptied, where `door_potential` is flat by construction and
                    # nothing tells it which way to walk.
                    "stranded": bool(
                        door_potential(self._latest[index]) <= 0.0),
                    # `stranded` alone cannot carry the diagnosis: Isaac locks
                    # the doors during a fight and `door_potential` skips locked
                    # ones, so an attempt that stalls mid-combat reads as
                    # stranded even though the room is not exhausted at all.
                    # Live enemies separate the two — no doors *and* no enemies
                    # is genuinely nowhere left to go, which is the case a
                    # non-local potential would fix.
                    "reward_parts": dict(self._parts[index]),
                    "enemies_alive": int(
                        ((self._latest[index] or {}).get("room") or {})
                        .get("enemies_alive", 0) or 0),
                }

        return self._stack_observations(), rewards, terminated, truncated, infos
