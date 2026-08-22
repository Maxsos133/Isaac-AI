"""Navigation as its own task: walk through a door.

Three floor runs plateaued at the same place — the agent barely leaving the
start room — while combat trained readily on its own. The difference is the
length of the credit-assignment chain. Combat is a reflex: shoot, something
dies, reward. A floor asks for cross a room, choose a door, walk through it, win
a fight, repeat, with reward only at the end.

So navigation gets the treatment combat got. One episode is one door traversal:
the room is emptied, the doors opened, the player dropped somewhere random, and
success is simply reaching the next room. Tens of steps, not thousands.

If this also fails to learn, the problem is not chain length — it is the
observation or the action space, and that is a different fix entirely.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from isaac_ai.bridge import InstanceBridge
from isaac_ai.config import AppConfig
from isaac_ai.env import IsaacVecEnv, decode_action
from isaac_ai.floors import door_potential
from isaac_ai.nav_curriculum import NavigationCurriculum

# Ticks to let a room setup settle before the policy is asked to act.
SETUP_SETTLE_TICKS = 3
# Regenerate the floor occasionally for genuinely new layouts. Much rarer than
# it used to be, because room variety now comes from teleporting within the
# floor — which costs a few ticks instead of a full floor generation.
EPISODES_PER_FLOOR = 120
# How often an episode starts somewhere other than where the last one ended.
# Lowered from 0.75 once episodes began spanning several distinct rooms: the
# variety now comes from the episodes themselves, and every teleport costs a
# room load — the single most expensive thing in this task.
RANDOM_ROOM_CHANCE = 0.20


def _mark_doors_for_episode(obs: dict[str, Any], visited: set[int]) -> None:
    """Rewrite each door's visit count to mean "seen during THIS episode".

    The game reports a per-run visit count, which saturates: after a while every
    room on the floor reads as visited and the observation can no longer say
    which door leads somewhere new. Success is judged per episode, so the
    observation has to be too — otherwise the agent is asked to reach a new room
    without being told which door that is.
    """
    room = obs.get("room")
    if not room:
        return
    for door in room.get("doors", []):
        door["visited"] = 1 if door.get("target") in visited else 0


def navigation_potential(obs: dict[str, Any]) -> float:
    """Pull toward a door leading somewhere new, falling back to any door.

    Plain nearest-door shaping fights the reward: the nearest door right after
    entering a room is the one just came through, so it guided the agent
    straight back while only new rooms paid. The fallback keeps dead ends
    solvable, where the single door is also the way back.
    """
    value = door_potential(obs, unvisited_only=True)
    if value > 0.0:
        return value
    return door_potential(obs, unvisited_only=False)


class NavigationVecEnv(IsaacVecEnv):
    def __init__(self, bridges: list[InstanceBridge], config: AppConfig,
                 curriculum: NavigationCurriculum,
                 max_nav_steps: int = 150,
                 gamma: float = 1.0) -> None:
        super().__init__(bridges, config)
        self.curriculum = curriculum
        self.max_nav_steps = max_nav_steps
        # Shaping uses gamma = 1 deliberately. With gamma < 1 the term expands
        # to (phi' - phi) - (1 - gamma) * phi': a constant drag proportional to
        # how close the agent already is, which penalises *being* near a door.
        # At the old settings that drag was the same size as the movement
        # signal itself, roughly halving it. Unit gamma telescopes exactly, so
        # there is no drag and a closed loop nets precisely zero.
        self.gamma = gamma
        self.shaping_coef = config.rewards.door_shaping
        self.arrival_reward = config.rewards.navigation_arrival

        self._potential = np.zeros(self.num_envs, dtype=np.float64)
        self._transitions = np.zeros(self.num_envs, dtype=np.int32)
        self._rooms_reached = np.zeros(self.num_envs, dtype=np.int32)
        # Rooms already seen this episode, so crossing back does not score.
        self._visited: list[set[int]] = [set() for _ in range(self.num_envs)]
        self._required = np.ones(self.num_envs, dtype=np.int32)
        self._episodes_on_floor = np.zeros(self.num_envs, dtype=np.int32)
        self._died = np.zeros(self.num_envs, dtype=bool)
        self._rng = np.random.default_rng(0)
        self.restarts = 0
        self.deaths = 0
        self.room_exits = 0

    # -- setup -------------------------------------------------------------

    def _prepare(self, indices: list[int]) -> None:
        """Empty the room, open the doors, and move the player somewhere new."""
        if not indices:
            return
        messages: list[dict[str, Any] | None] = [None] * self.num_envs
        for index in indices:
            if self._failed[index]:
                continue
            messages[index] = {
                "t": "navsetup",
                "reposition": True,
                "random_room": bool(self._rng.random() < RANDOM_ROOM_CHANCE),
            }
        self._exchange_all(messages)

        # The teleport lands us somewhere new, so the episode's history starts
        # from the room we actually end up in.
        for index in indices:
            if not self._failed[index]:
                self._visited[index] = set()

        for _ in range(SETUP_SETTLE_TICKS):
            settle: list[dict[str, Any] | None] = [
                {"t": "noop"} if messages[i] is not None else None
                for i in range(self.num_envs)
            ]
            for index, obs in enumerate(self._exchange_all(settle)):
                if obs is not None:
                    self._latest[index] = obs

        for index in indices:
            self._transitions[index] = 0
            self._rooms_reached[index] = 0
            self._visited[index] = set()
            latest = self._latest[index]
            if latest and latest.get("room"):
                # Where the episode begins is not somewhere it "reached".
                self._visited[index].add(latest["room"]["index"])
            self._required[index] = self.curriculum.required_transitions()
            self._episode_steps[index] = 0
            self._episode_returns[index] = 0.0
            latest = self._latest[index]
            if latest:
                _mark_doors_for_episode(latest, self._visited[index])
            self._potential[index] = navigation_potential(latest)

    def reset(self) -> dict[str, np.ndarray]:
        if all(latest is None for latest in self._latest):
            self._prime()
        everyone = list(range(self.num_envs))
        self._restart(everyone)
        self.restarts += self.num_envs
        self._episodes_on_floor[:] = 0
        self._prepare(everyone)
        return self._stack_observations()

    def reset_done(self, done_mask: np.ndarray) -> None:
        indices = [int(i) for i in np.flatnonzero(done_mask) if not self._failed[i]]
        if not indices:
            return

        # A dead player ends the run, so the room cannot simply be re-prepared —
        # the whole run has to be rebuilt before anything else will work.
        dead = [i for i in indices if self._died[i]]
        stale = [i for i in indices
                 if self._episodes_on_floor[i] >= EPISODES_PER_FLOOR]
        rebuild = sorted(set(dead) | set(stale))

        if rebuild:
            self._restart(rebuild)
            self.restarts += len(rebuild)
            self.deaths += len(dead)
            for index in rebuild:
                self._episodes_on_floor[index] = 0
                self._died[index] = False

        self._prepare(indices)

    # -- stepping ----------------------------------------------------------

    def step(self, actions: np.ndarray):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        arrived = np.zeros(self.num_envs, dtype=bool)
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

        for _ in range(self.action_repeat):
            for index, obs in enumerate(self._exchange_all(messages)):
                if obs is None:
                    if self._failed[index]:
                        terminated[index] = True
                    continue
                self._latest[index] = obs
                if not obs.get("ready", True):
                    continue

                reward = self.config.rewards.step

                # Death must be caught here, during the death animation, while
                # the update loop is still running. Left alone it reaches the
                # game-over screen, where MC_POST_UPDATE stops firing entirely —
                # the mod goes silent and the instance can never recover on its
                # own. Enemies are cleared at setup, so this is rare, but the
                # window between entering a live room and setup running is real.
                if obs["events"].get("died") or obs["player"]["is_dead"]:
                    reward += self.config.rewards.death
                    self._died[index] = True
                    terminated[index] = True
                    continue

                # Only rooms not yet seen this episode count.
                #
                # Counting raw crossings instead let the agent satisfy "reach
                # four rooms" by walking through one door four times — the same
                # mistake as rewarding the floor's room-transition event. It is
                # cheap, it scores full marks, and it teaches nothing beyond the
                # first traversal. Requiring distinct rooms makes the only way
                # to score be actually going somewhere.
                if obs["events"].get("new_room"):
                    self._transitions[index] += 1
                    room_index = obs.get("room", {}).get("index")
                    if room_index is not None and room_index not in self._visited[index]:
                        self._visited[index].add(room_index)
                        self._rooms_reached[index] += 1
                        reward += self.arrival_reward
                        if self._rooms_reached[index] >= self._required[index]:
                            arrived[index] = True
                            terminated[index] = True

                # Relabel doors against this episode's history before either the
                # policy or the shaping looks at them.
                _mark_doors_for_episode(obs, self._visited[index])
                potential = navigation_potential(obs)
                reward += self.shaping_coef * (
                    self.gamma * potential - self._potential[index])
                self._potential[index] = potential

                rewards[index] += reward

        for index in range(self.num_envs):
            self._episode_steps[index] += 1
            self._episode_returns[index] += rewards[index]

            if not terminated[index] and self._episode_steps[index] >= self.max_nav_steps:
                truncated[index] = True

            if terminated[index] or truncated[index]:
                success = bool(arrived[index])
                self._episodes_on_floor[index] += 1
                if not self._failed[index]:
                    self.curriculum.record(success, int(self._rooms_reached[index]))
                reached = max(int(self._rooms_reached[index]), 1)
                infos[index]["episode"] = {
                    "r": float(self._episode_returns[index]),
                    "l": int(self._episode_steps[index]),
                    "success": success,
                    "rooms_reached": int(self._rooms_reached[index]),
                    # Same key the trainer already aggregates.
                    "rooms_seen": int(self._rooms_reached[index]),
                    "transitions": int(self._transitions[index]),
                    # Above 1 means it is crossing back and forth rather than
                    # going somewhere new — the behaviour to watch for.
                    "backtrack_ratio": float(self._transitions[index]) / reached,
                    "required": int(self._required[index]),
                }

        return self._stack_observations(), rewards, terminated, truncated, infos
