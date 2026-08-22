"""Combat training environment.

An episode is one encounter: enemies are spawned into the current room, and it
ends when they are all dead, the player dies, or time runs out.

Episodes reset by rebuilding the encounter in place rather than restarting the
run. A full `restart` regenerates the floor and costs most of a second; respawning
enemies costs a few ticks. Only death forces a real restart, because death ends
the run.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from isaac_ai.bridge import BridgeError, InstanceBridge
from isaac_ai.config import AppConfig
from isaac_ai.curriculum import CombatCurriculum
from isaac_ai.env import IsaacVecEnv, compute_reward, decode_action

# How long to let the encounter settle after spawning before handing control back.
SPAWN_SETTLE_TICKS = 4


class CombatVecEnv(IsaacVecEnv):
    def __init__(self, bridges: list[InstanceBridge], config: AppConfig,
                 curriculum: CombatCurriculum,
                 max_encounter_steps: int = 450,
                 defeat_hearts: int = 1) -> None:
        super().__init__(bridges, config)
        self.curriculum = curriculum
        self.max_encounter_steps = max_encounter_steps
        # Half-hearts remaining at which the encounter is conceded.
        self.defeat_hearts = defeat_hearts
        self.restarts = 0
        # See the note in floors.py: the trainer logs getattr(env, "deaths", 0)
        # and this attribute did not exist, so the figure was always 0.
        self.deaths = 0
        # Should stay at zero: if it climbs, doors are not locking during fights.
        self.room_exits = 0
        self.encounter_reward = config.rewards.room_clear
        self._enemy_target = np.zeros(self.num_envs, dtype=np.int32)
        self._last_outcomes: list[bool] = []
        # Which instances' outcomes are allowed to move the difficulty dial.
        # None means all of them, which is what plain training wants. During
        # distillation the student drives a growing share of episodes and loses
        # most of them; letting those count drags difficulty away from the
        # level the teacher is actually competent at, so the teacher ends up
        # demonstrating on encounters neither of them belongs in.
        self.curriculum_mask: np.ndarray | None = None

    # -- encounter lifecycle ----------------------------------------------

    def _build_encounter(self, indices: list[int]) -> None:
        """Spawn a fresh encounter on each given instance."""
        messages: list[dict[str, Any] | None] = [None] * self.num_envs
        for index in indices:
            if self._failed[index]:
                continue
            encounter = self.curriculum.sample()
            self._enemy_target[index] = encounter.enemy_count
            messages[index] = encounter.to_command()

        self._exchange_all(messages)

        # Let the spawns register before the policy sees the state.
        for _ in range(SPAWN_SETTLE_TICKS):
            settle: list[dict[str, Any] | None] = [
                {"t": "noop"} if messages[i] is not None else None
                for i in range(self.num_envs)
            ]
            for index, obs in enumerate(self._exchange_all(settle)):
                if obs is not None:
                    self._latest[index] = obs

        for index in indices:
            self._episode_steps[index] = 0
            self._episode_returns[index] = 0.0

    def _needs_restart(self, index: int) -> bool:
        """A dead player cannot fight another encounter; the run must restart."""
        latest = self._latest[index]
        if latest is None or not latest.get("ready", True):
            return True
        return bool(latest.get("player", {}).get("is_dead"))

    def reset(self) -> dict[str, np.ndarray]:
        if all(latest is None for latest in self._latest):
            self._prime()
        self._restart(list(range(self.num_envs)))
        self._build_encounter(list(range(self.num_envs)))
        return self._stack_observations()

    def reset_done(self, done_mask: np.ndarray) -> None:
        indices = [int(i) for i in np.flatnonzero(done_mask) if not self._failed[i]]
        if not indices:
            return

        # Only pay for a floor regeneration where the player actually died.
        # Conceding at low health means this should be rare; if `restarts`
        # climbs in step with episodes, the defeat threshold is set too low.
        needs_restart = [i for i in indices if self._needs_restart(i)]
        if needs_restart:
            self.restarts += len(needs_restart)
            self._restart(needs_restart)
        self._build_encounter(indices)

    # -- stepping ----------------------------------------------------------

    def step(self, actions: np.ndarray):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        cleared = np.zeros(self.num_envs, dtype=bool)
        left_room = np.zeros(self.num_envs, dtype=bool)
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

                reward, events = compute_reward(
                    obs, self.config,
                    step_penalty=self.config.rewards.combat_step)
                rewards[index] += reward

                # Ending the encounter just short of death is what keeps resets
                # cheap: a dead player ends the run, and rebuilding a run costs
                # a floor regeneration during which every other instance waits.
                # Defeat is still defeat as far as the learner is concerned.
                hearts = obs["player"]["hearts"] + obs["player"]["soul_hearts"]
                if events.get("died") or obs["player"]["is_dead"]:
                    if not terminated[index]:
                        self.deaths += 1
                    terminated[index] = True
                elif hearts <= self.defeat_hearts:
                    terminated[index] = True
                elif events.get("new_room"):
                    # Enemy counting is per-room, so an agent that walked out
                    # would look like it had cleared the encounter and would be
                    # paid for fleeing. Doors are locked during a fight, so this
                    # should not fire — it is the backstop if one ever opens.
                    left_room[index] = True
                    terminated[index] = True
                elif obs["room"].get("enemies_alive", 0) == 0:
                    # Encounter survived and won.
                    cleared[index] = True
                    terminated[index] = True

            # Killing the last enemy makes the game open the doors again. An
            # instance that keeps applying its held movement for the remaining
            # repeat ticks can walk straight out of the room from a standing
            # start on a door — scored as a failure, and it stays in the new
            # room for every encounter after that. Hold still once done; the
            # fleet still has to tick, so this is a noop rather than nothing.
            for index in range(self.num_envs):
                if terminated[index] and messages[index] is not None:
                    messages[index] = {"t": "noop"}

        for index in range(self.num_envs):
            self._episode_steps[index] += 1
            if cleared[index]:
                rewards[index] += self.encounter_reward
            self._episode_returns[index] += rewards[index]

            if not terminated[index] and self._episode_steps[index] >= self.max_encounter_steps:
                truncated[index] = True

            if left_room[index]:
                self.room_exits += 1

            if terminated[index] or truncated[index]:
                success = bool(cleared[index]) and not bool(left_room[index])
                if not self._failed[index]:
                    if (self.curriculum_mask is None
                            or bool(self.curriculum_mask[index])):
                        self.curriculum.record(success)
                    self._last_outcomes.append(success)
                infos[index]["episode"] = {
                    "r": float(self._episode_returns[index]),
                    "l": int(self._episode_steps[index]),
                    "success": success,
                    "left_room": bool(left_room[index]),
                    "enemies": int(self._enemy_target[index]),
                }

        return self._stack_observations(), rewards, terminated, truncated, infos

    def drain_outcomes(self) -> list[bool]:
        outcomes = self._last_outcomes
        self._last_outcomes = []
        return outcomes
