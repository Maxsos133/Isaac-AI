"""PPO over the vectorized Isaac environment.

Nothing exotic — clipped surrogate objective, GAE, a few epochs per rollout. The
interesting parts of this project are the environment and the curriculum, so the
learner stays deliberately standard and debuggable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from isaac_ai.env import ACTION_DIMS, MOVE_HEADS, SHOOT_HEADS
from isaac_ai.policy import ActorCritic, to_tensors

# Observation keys are discovered from the first observation rather than listed
# here. A hardcoded list is a third place to keep in sync with the encoder and
# the network, and silently drops any key it does not know about.


@dataclass
class PPOConfig:
    rollout_steps: int = 128
    epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    # Per-head entropy ceiling, in nats. Above it a head earns no further bonus,
    # so the term stops being a force pulling towards uniform and becomes only a
    # floor preventing premature collapse.
    #
    # The bonus is otherwise applied to the *summed* entropy of all four heads,
    # which means a head receiving no learning gradient is pushed to the
    # ln(3)=1.099 ceiling at full strength forever. That is the documented cause
    # of every dead axis this project has had. floor-v7b shows it directly:
    # `shoot_x` learned down to 0.863 by 375-500k, then decayed back to 1.084 as
    # the agent got better at exploring and episodes shifted away from fighting,
    # leaving the entropy term as the only force still acting on it.
    #
    # 0.0 disables the ceiling and restores the original behaviour.
    entropy_target: float = 0.0
    # Floor under each individual action's probability, in [0, 1/3).
    #
    # `entropy_target` stops a whole head being dragged to uniform, and that is
    # all it does. It cannot see one action inside a head dying, because the
    # other two carry the entropy: floor-v13 answered an up-door state with
    # move_y = [0.01 up, 0.61 still, 0.37 down], whose entropy is 0.715 — above
    # the 0.5 target, so **zero** gradient pushed P(up) back up. The agent could
    # not leave a room whose only exit was upward, and the logged per-axis
    # entropy read a healthy 0.694 throughout.
    #
    # That is the summed-entropy failure one level down: the instrument built to
    # catch a dead axis is blind to a dead action within an axis.
    #
    # A floor of 0.05 also bounds head entropy at ~0.39, so it subsumes most of
    # what `entropy_target` does while actually protecting the individual
    # action. 0.0 disables it.
    min_action_prob: float = 0.0
    action_floor_coef: float = 1.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class RolloutBuffer:
    steps: int
    num_envs: int
    device: torch.device
    observations: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.actions = torch.zeros((self.steps, self.num_envs, len(ACTION_DIMS)),
                                   dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros((self.steps, self.num_envs), device=self.device)
        self.rewards = torch.zeros((self.steps, self.num_envs), device=self.device)
        self.values = torch.zeros((self.steps, self.num_envs), device=self.device)
        self.dones = torch.zeros((self.steps, self.num_envs), device=self.device)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.observations)

    def allocate(self, sample: dict[str, torch.Tensor]) -> None:
        for key, value in sample.items():
            shape = (self.steps, self.num_envs) + tuple(value.shape[1:])
            self.observations[key] = torch.zeros(shape, device=self.device)

    def insert(self, index: int, obs: dict[str, torch.Tensor],
               actions: torch.Tensor, log_probs: torch.Tensor,
               rewards: torch.Tensor, values: torch.Tensor,
               dones: torch.Tensor) -> None:
        for key in self.keys:
            self.observations[key][index] = obs[key]
        self.actions[index] = actions
        self.log_probs[index] = log_probs
        self.rewards[index] = rewards
        self.values[index] = values
        self.dones[index] = dones


class PPOTrainer:
    def __init__(self, env, config: PPOConfig, device: torch.device,
                 run_dir: Path, recover=None) -> None:
        self.env = env
        self.config = config
        self.device = device
        # Optional callable invoked between updates when instances have dropped
        # out. Returns how many came back. Kept as a callback rather than a
        # fleet reference so nothing here has to know about launchers, windows
        # or SPACE taps; `cli.py` owns that and passes a closure.
        self.recover = recover
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.policy = ActorCritic().to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=config.learning_rate, eps=1e-5)
        self.buffer = RolloutBuffer(config.rollout_steps, env.num_envs, device)
        self.global_step = 0
        self.updates = 0
        self.metrics_path = run_dir / "metrics.jsonl"

    def _episode_summary(self) -> dict:
        """Per-episode extras, aggregated so they mean what they say.

        backtrack_ratio is averaged only over episodes that reached at least one
        room. Including the ones that reached none would score them 0 — reading
        as "no backtracking" when nothing happened at all, so the metric would
        improve as the agent did less.
        """
        extras = self._episode_extras
        if not extras:
            return {}

        summary: dict = {}
        if any("rooms_seen" in e for e in extras):
            summary["rooms_seen"] = round(float(np.mean(
                [e.get("rooms_seen", 0) for e in extras])), 2)

        # What the floor curriculum actually scores on, and it was not being
        # logged. floor-v5 showed rooms_seen climbing to 1.68 while difficulty
        # never left zero, and the reason was invisible: the agent enters rooms
        # it cannot clear. Entering is navigation; clearing is combat, and only
        # one of them was on the chart.
        if any("rooms_cleared" in e for e in extras):
            summary["rooms_cleared"] = round(float(np.mean(
                [e.get("rooms_cleared", 0) for e in extras])), 2)
        if any("descended" in e for e in extras):
            summary["descended"] = round(float(np.mean(
                [e.get("descended", 0) for e in extras])), 3)

        productive = [e for e in extras if e.get("rooms_seen", 0) > 0]
        if productive:
            summary["backtrack_ratio"] = round(float(np.mean(
                [e.get("backtrack_ratio", 0.0) for e in productive])), 2)
            summary["productive_episodes"] = len(productive)

        # Why episodes ended, and how much game time each way costs. "Episode
        # over" covers dying, the step cap and the idle limit, and they have
        # nothing in common — the first is a combat problem, the last is usually
        # the agent pacing between rooms it has already emptied, where the
        # potential is flat by construction. Logged as a share of episodes plus
        # the mean length of the idle ones, which is what turns "it does this
        # sometimes" into a throughput number.
        # Mean reward per episode, per term. Every rebalance until now has been
        # argued from a split *inferred* from aggregates — rooms_seen times a
        # coefficient, enemies assumed at three a room. This reports what was
        # actually paid, so a change can be checked on its first update instead
        # of at the end of a 100-minute run.
        breakdowns = [e.get("reward_parts") for e in extras if e.get("reward_parts")]
        if breakdowns:
            names = sorted({k for b in breakdowns for k in b})
            for name in names:
                summary[f"r_{name}"] = round(
                    float(np.mean([b.get(name, 0.0) for b in breakdowns])), 3)

        reasons = [e.get("reason") for e in extras if e.get("reason")]
        if reasons:
            total = len(reasons)
            # Every reason is logged, including the zeros. Omitting the absent
            # ones seemed tidier and is an analysis trap: averaging each key
            # over only the updates where it appeared made the shares sum to
            # 1.047 on floor-v9b. If this block is present at all then episodes
            # ended, so 0.0 already means "measured, none of them".
            for name in ("died", "idle", "timeout", "dropped"):
                summary[f"ended_{name}"] = round(reasons.count(name) / total, 3)
            idle = [e for e in extras if e.get("reason") == "idle"]
            if idle:
                summary["idle_episode_steps"] = int(np.mean(
                    [e.get("l", 0) for e in idle]))
            summary["episode_steps"] = int(np.mean(
                [e.get("l", 0) for e in extras]))
            # Of the episodes that gave up, how many did so with nowhere left to
            # go in that room. This is the one that says whether a non-local
            # potential is worth building.
            gave_up = [e for e in extras
                       if e.get("reason") in ("idle", "timeout")]
            if gave_up:
                summary["stranded"] = round(float(np.mean(
                    [1.0 if e.get("stranded") else 0.0 for e in gave_up])), 3)
                # The unconfounded version: nowhere to go AND nothing to fight.
                # This is the one that justifies building a non-local potential;
                # `stranded` on its own also counts stalling in a locked fight.
                summary["exhausted"] = round(float(np.mean(
                    [1.0 if (e.get("stranded")
                             and not e.get("enemies_alive")) else 0.0
                     for e in gave_up])), 3)
        return summary

    def _movement_rates(self) -> dict:
        """Blocked and standing-still rates, if the env tracks them.

        Reported as rates over the whole run rather than per window: they move
        slowly and a per-update figure over ~2500 steps is mostly noise.
        """
        requests = getattr(self.env, "move_requests", 0)
        if not requests:
            return {}
        still = getattr(self.env, "still_steps", 0)
        return {
            "blocked_rate": round(
                getattr(self.env, "blocked_moves", 0) / requests, 4),
            "still_rate": round(still / (requests + still), 4),
        }

    def _log(self, record: dict) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def collect(self, observation: dict[str, np.ndarray]):
        """Roll out `rollout_steps` transitions across every instance."""
        episode_returns: list[float] = []
        episode_successes: list[bool] = []
        self._episode_extras: list[dict] = []

        obs_tensors = to_tensors(observation, self.device)
        if not self.buffer.observations:
            self.buffer.allocate(obs_tensors)

        for step in range(self.config.rollout_steps):
            with torch.no_grad():
                actions, log_probs, values = self.policy.act(obs_tensors)

            next_obs, rewards, terminated, truncated, infos = self.env.step(
                actions.cpu().numpy())
            done = terminated | truncated

            self.buffer.insert(
                step, obs_tensors, actions, log_probs,
                torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
                values,
                torch.as_tensor(done, dtype=torch.float32, device=self.device),
            )

            for info in infos:
                if "episode" in info:
                    episode_returns.append(info["episode"]["r"])
                    episode_successes.append(bool(info["episode"].get("success")))
                    self._episode_extras.append(info["episode"])

            if done.any():
                self.env.reset_done(done)
                next_obs = self.env._stack_observations()

            observation = next_obs
            obs_tensors = to_tensors(observation, self.device)
            self.global_step += self.env.num_envs

        with torch.no_grad():
            _, last_values = self.policy(obs_tensors)

        return observation, last_values, episode_returns, episode_successes

    def advantages(self, last_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        advantages = torch.zeros_like(self.buffer.rewards)
        gae = torch.zeros(self.env.num_envs, device=self.device)

        for step in reversed(range(config.rollout_steps)):
            if step == config.rollout_steps - 1:
                next_value = last_values
            else:
                next_value = self.buffer.values[step + 1]
            not_done = 1.0 - self.buffer.dones[step]
            delta = (self.buffer.rewards[step]
                     + config.gamma * next_value * not_done
                     - self.buffer.values[step])
            gae = delta + config.gamma * config.gae_lambda * not_done * gae
            advantages[step] = gae

        returns = advantages + self.buffer.values
        return advantages, returns

    def update(self, advantages: torch.Tensor, returns: torch.Tensor) -> dict:
        config = self.config
        batch = config.rollout_steps * self.env.num_envs

        flat_obs = {key: self.buffer.observations[key].reshape(
            (batch,) + self.buffer.observations[key].shape[2:])
            for key in self.buffer.keys}
        flat_actions = self.buffer.actions.reshape(batch, len(ACTION_DIMS))
        flat_log_probs = self.buffer.log_probs.reshape(batch)
        flat_advantages = advantages.reshape(batch)
        flat_returns = returns.reshape(batch)

        minibatch_size = batch // config.minibatches
        indices = np.arange(batch)
        policy_losses, value_losses, entropies, clip_fractions = [], [], [], []
        move_entropies, shoot_entropies = [], []
        axis_entropies: list[list[float]] = [[] for _ in ACTION_DIMS]
        min_probs: list[list[float]] = [[] for _ in ACTION_DIMS]

        for _ in range(config.epochs):
            np.random.shuffle(indices)
            for start in range(0, batch, minibatch_size):
                subset = indices[start:start + minibatch_size]
                subset_t = torch.as_tensor(subset, device=self.device)

                obs_batch = {key: flat_obs[key][subset_t]
                             for key in self.buffer.keys}
                log_prob, entropy, value, per_head, head_probs = \
                    self.policy.evaluate(obs_batch, flat_actions[subset_t])

                ratio = torch.exp(log_prob - flat_log_probs[subset_t])
                batch_adv = flat_advantages[subset_t]
                batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)

                unclipped = ratio * batch_adv
                clipped = torch.clamp(ratio, 1 - config.clip_range,
                                      1 + config.clip_range) * batch_adv
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(value, flat_returns[subset_t])
                if config.entropy_target > 0:
                    # Clamped per sample, per head: a head already at or above
                    # the target contributes a constant and therefore no
                    # gradient, so nothing pulls it back towards uniform. A head
                    # that collapses below it still gets pushed back up.
                    entropy_loss = torch.stack(
                        [head.clamp(max=config.entropy_target)
                         for head in per_head]).sum(0).mean()
                else:
                    entropy_loss = entropy.mean()

                loss = (policy_loss
                        + config.value_coef * value_loss
                        - config.entropy_coef * entropy_loss)

                # Push any action that has fallen below the floor back up.
                # Hinged, so an action comfortably above it contributes nothing
                # and the policy stays free to be confident where it should be.
                if config.min_action_prob > 0:
                    shortfall = sum(
                        torch.relu(config.min_action_prob - probs).sum(-1).mean()
                        for probs in head_probs)
                    loss = loss + config.action_floor_coef * shortfall

                # The smallest probability in each head, which is the number
                # that would have made floor-v13's dead "up" visible. Entropy
                # could not: the axis read 0.694 while P(up) sat at 0.000.
                with torch.no_grad():
                    for index, probs in enumerate(head_probs):
                        min_probs[index].append(
                            float(probs.min(dim=-1).values.mean()))

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         config.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                # The *true* summed entropy, never the clamped objective, or the
                # logged figure would change meaning the moment a target is set
                # and no run would be comparable with any earlier one.
                entropies.append(float(entropy.mean().item()))
                move_entropies.append(
                    sum(per_head[i].mean().item() for i in MOVE_HEADS))
                shoot_entropies.append(
                    sum(per_head[i].mean().item() for i in SHOOT_HEADS))
                for index, head in enumerate(per_head):
                    axis_entropies[index].append(head.mean().item())
                with torch.no_grad():
                    clip_fractions.append(
                        ((ratio - 1).abs() > config.clip_range).float().mean().item())

        self.updates += 1
        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            # The number to actually watch on a movement task. Max is
            # 2 * ln(3) = 2.197, and unlike the total it can genuinely reach 0.
            "move_entropy": float(np.mean(move_entropies)),
            "shoot_entropy": float(np.mean(shoot_entropies)),
            "clip_fraction": float(np.mean(clip_fractions)),
            # Per axis, because the summed figures hide the failure that
            # actually happens. combat-v5 through v7 all logged a shoot_entropy
            # of ~1.10, which reads as a policy halfway to confident and was in
            # fact one axis frozen on "down" (entropy 0.05) beside one that had
            # given up entirely (1.07, against a ln(3)=1.099 ceiling). The agent
            # could not aim horizontally at all, and the summed number looked
            # unremarkable for three runs and two derived pixel students.
            **{f"entropy_{name}": float(np.mean(axis_entropies[index]))
               for index, name in enumerate(
                   ("move_x", "move_y", "shoot_x", "shoot_y"))},
            # The number per-axis entropy could not give. floor-v13 logged
            # entropy_move_y 0.694 — healthy — while P(up) was 0.000 and the
            # agent could not leave a room whose only exit was upward. Watch
            # these, not the entropies, for an action dying.
            **{f"min_prob_{name}": float(np.mean(min_probs[index]))
               for index, name in enumerate(
                   ("move_x", "move_y", "shoot_x", "shoot_y"))},
        }

    def save(self, name: str = "policy.pt") -> Path:
        path = self.run_dir / name
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "updates": self.updates,
        }, path)
        return path

    def train(self, total_steps: int, log_every: int = 1) -> None:
        observation = self.env.reset()
        started = time.perf_counter()
        recent_returns: list[float] = []
        recent_successes: list[bool] = []

        while self.global_step < total_steps:
            observation, last_values, returns_seen, successes = self.collect(observation)
            recent_returns.extend(returns_seen)
            recent_successes.extend(successes)
            recent_returns = recent_returns[-100:]
            recent_successes = recent_successes[-100:]

            advantages, returns = self.advantages(last_values)
            stats = self.update(advantages, returns)

            # One difficulty step per policy update. Moving it per episode makes
            # the ramp oscillate far faster than the policy can track.
            self.env.curriculum.advance()

            # Bring crashed instances back, between updates rather than mid
            # rollout: the buffer is sized for `num_envs` and a relaunch takes
            # seconds, so doing it here keeps every tensor the right shape and
            # keeps the disturbance out of a batch that is about to be learned
            # from. floor-v27b finished two instances down over 5M steps.
            if self.recover is not None and self.env.alive_count < self.env.num_envs:
                if self.recover():
                    # Relaunching pumps every healthy instance with `noop` for
                    # the seconds the SPACE walk takes, so they have all been
                    # running with neutral controls and some will have wandered
                    # or died. Those episodes are junk; end them here rather
                    # than let them finish and be scored.
                    done = np.ones(self.env.num_envs, dtype=bool)
                    self.env.reset_done(done)
                    observation = self.env._stack_observations()

            if self.updates % log_every == 0:
                elapsed = time.perf_counter() - started
                record = {
                    "update": self.updates,
                    "global_step": self.global_step,
                    "steps_per_second": round(self.global_step / elapsed, 1),
                    "mean_return": round(float(np.mean(recent_returns)), 3)
                    if recent_returns else None,
                    "success_rate": round(float(np.mean(recent_successes)), 3)
                    if recent_successes else None,
                    "episodes": len(recent_returns),
                    "full_restarts": getattr(self.env, "restarts", 0),
                    "room_exits": getattr(self.env, "room_exits", 0),
                    "deaths": getattr(self.env, "deaths", 0),
                    "alive_instances": self.env.alive_count,
                    # The two numbers that say whether the blocked-move penalty
                    # is working. `blocked_rate` should fall towards the 2.5% a
                    # random walk registers; `still_rate` must NOT climb, since
                    # standing still is the one way to dodge the penalty without
                    # learning anything.
                    **self._movement_rates(),
                    # Surfaces back-and-forth pacing, which is invisible in
                    # rooms_cleared and cost a whole run to notice by eye.
                    **self._episode_summary(),
                    **{k: round(v, 4) for k, v in stats.items()},
                    **self.env.curriculum.state(),
                }
                self._log(record)
                # `still_rate` and `blocked_rate` are on the live line, not
                # only in the file. The v16 rebalance tripled still_rate — the
                # exact failure a comment in config.toml said to watch for — and
                # it went unnoticed for two full runs because nobody reads
                # metrics.jsonl while a run is in progress. A tripwire in a file
                # is not a tripwire.
                # `difficulty` is back on the line after floor-v25 moved it off
                # 0.000 for only the second time in this project's history.
                # `target_rooms` comes with it because that is what actually
                # changes the task: when the dial advances, success stops meaning
                # "clear one room" and starts meaning "clear two", so a drop in
                # `success_rate` at that moment is the curriculum working rather
                # than a regression. Without the target on screen that reads as
                # the run falling over.
                curriculum = ""
                if record.get("difficulty") is not None:
                    curriculum = (f"| diff {record['difficulty']:.3f} "
                                  f"t{record.get('target_rooms', '?')} ")
                print(f"update {record['update']:4d} | step {record['global_step']:7d} "
                      f"| {record['steps_per_second']:6.1f} st/s "
                      f"| return {record['mean_return']} "
                      f"| success {record['success_rate']} "
                      f"| cleared {record.get('rooms_cleared')} "
                      f"{curriculum}"
                      f"| still {record.get('still_rate')} "
                      f"| blocked {record.get('blocked_rate')}")

            if self.updates % 20 == 0:
                self.save()

        self.save()
