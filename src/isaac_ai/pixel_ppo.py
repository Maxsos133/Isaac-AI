"""RL fine-tuning for the pixel student, with the asymmetric critic.

Distillation took the student to its teacher's level and stopped, because
imitation cannot exceed what it imitates: measured at frozen difficulty, the
student matches combat-v4 within noise up to 0.4 and both fall away together
above it. Getting further means learning from reward rather than from example,
which is what this does.

The actor still sees only pixels. The critic reads the mod's privileged state
and is discarded at deployment, so the hard half of the problem — inferring
value from a screen — never has to be solved.

Three things here exist because a fine-tune starts from something valuable and
can destroy it, which training from scratch cannot:

**The critic is warmed up first.** Distillation trained the actor and never
touched the critic, so it starts random and its advantages are noise. Feeding
those to the actor would wreck a policy that cost a million steps to build.
Warm-up needs no freezing: the critic reads state and the actor reads pixels
through entirely separate networks, so a value loss produces gradient in the
critic alone. The asymmetry pays for itself here.

**The policy is anchored to the teacher** by a KL penalty. PPO's early updates
are its noisiest, and without an anchor a distilled policy can be undone before
the advantages mean anything. The coefficient is small, so it is a spring rather
than a cage — the student is free to exceed the teacher, just not to forget it.

**The clip range is tighter and the learning rate lower** than the from-scratch
trainer. Fine-tuning wants small steps around a good policy, not the wide
exploration a random initialisation needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from isaac_ai.env import ACTION_DIMS, MOVE_HEADS, SHOOT_HEADS
from isaac_ai.pixel_policy import PixelActorCritic
from isaac_ai.policy import to_tensors


@dataclass
class PixelPPOConfig:
    rollout_steps: int = 64
    epochs: int = 3
    minibatches: int = 4
    # Lower than the from-scratch trainer's 3e-4: the starting policy is good.
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # Tighter than the usual 0.2 for the same reason.
    clip_range: float = 0.1
    # Zero. An entropy bonus is for a policy that still has to explore; on a
    # fine-tune of an already-fitted one it just spreads the distribution.
    # Measured on finetune-v1, where move entropy rose 1.55 -> 1.64 and the
    # sampled policy fell to 0.67 against its own greedy 0.88 at difficulty
    # 0.30 — a policy made worse at being sampled for no gain in what it knew.
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Updates spent training only the critic before the actor is allowed to
    # move. The critic arrives untrained from distillation, and its advantages
    # are noise until it has seen some returns.
    critic_warmup_updates: int = 25
    # Start the critic as a copy of the teacher's value function instead of from
    # random weights. `PrivilegedCritic` is structurally identical to the
    # teacher's network minus its action heads — all fourteen tensors match,
    # value head included — so the teacher can supply a value function already
    # trained for a million steps on this exact task and observation. Learning
    # it again from reward alone is what the warm-up was papering over, and a
    # critic that never becomes informative makes PPO optimise noise: two
    # fine-tuning runs moved difficulty by 0.02 and 0.00 respectively.
    seed_critic_from_teacher: bool = True
    # Strength of the pull back towards the teacher's action distribution.
    # Small: enough to stop an early collapse, not enough to cap the student at
    # the teacher's level, which would defeat the point of fine-tuning at all.
    teacher_kl_coef: float = 0.05


@dataclass
class PixelRollout:
    """Stores frames and privileged state side by side for one rollout."""

    steps: int
    num_envs: int
    pixel_shape: tuple[int, int, int]
    device: torch.device
    state: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # uint8, because frames dominate this buffer: at 12 instances and 64
        # steps a float32 copy would be half a gigabyte for no benefit.
        self.frames = torch.zeros((self.steps, self.num_envs) + self.pixel_shape,
                                  dtype=torch.uint8, device=self.device)
        self.actions = torch.zeros((self.steps, self.num_envs, len(ACTION_DIMS)),
                                   dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros((self.steps, self.num_envs), device=self.device)
        self.rewards = torch.zeros((self.steps, self.num_envs), device=self.device)
        self.values = torch.zeros((self.steps, self.num_envs), device=self.device)
        self.dones = torch.zeros((self.steps, self.num_envs), device=self.device)
        self.teacher_logits = torch.zeros(
            (self.steps, self.num_envs, sum(ACTION_DIMS)), device=self.device)

    def allocate_state(self, sample: dict[str, torch.Tensor]) -> None:
        for key, value in sample.items():
            self.state[key] = torch.zeros(
                (self.steps, self.num_envs) + tuple(value.shape[1:]),
                device=self.device)


def load_student(path: Path, device: torch.device) -> PixelActorCritic:
    """Load a distilled student, actor weights and all."""
    from isaac_ai.env import DOOR_FEATURES, MAX_DOORS, SCALAR_FEATURES

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    key = "student" if "student" in checkpoint else "policy"
    model = PixelActorCritic(tuple(checkpoint["pixel_shape"]),
                             SCALAR_FEATURES, MAX_DOORS * DOOR_FEATURES).to(device)
    model.load_state_dict(checkpoint[key])
    return model


class PixelPPOTrainer:
    def __init__(self, env, capture, student: PixelActorCritic, teacher,
                 config: PixelPPOConfig, device: torch.device,
                 run_dir: Path) -> None:
        self.env = env
        self.capture = capture
        self.policy = student
        self.teacher = teacher
        self.config = config
        self.device = device
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = run_dir / "metrics.jsonl"

        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=config.learning_rate, eps=1e-5)
        self.buffer = PixelRollout(config.rollout_steps, env.num_envs,
                                   capture.shape, device)
        self.global_step = 0
        self.updates = 0
        self._returns: list[float] = []
        self._successes: list[bool] = []

        if config.seed_critic_from_teacher:
            copied, total = self.seed_critic()
            print(f"critic seeded from the teacher: {copied}/{total} tensors")

    def seed_critic(self) -> tuple[int, int]:
        """Copy the teacher's value function into the critic.

        The teacher's value head predicts returns under the *teacher's* policy
        rather than the student's, so this is a starting point and not a
        finished critic — the warm-up still runs. But it starts from something
        that already reads a room correctly instead of from noise.
        """
        teacher_state = self.teacher.state_dict()
        critic_state = self.policy.critic.state_dict()
        transferable = {
            key: teacher_state[key] for key in critic_state
            if key in teacher_state
            and teacher_state[key].shape == critic_state[key].shape
        }
        self.policy.critic.load_state_dict(transferable, strict=False)
        return len(transferable), len(critic_state)

    @property
    def warming_up(self) -> bool:
        return self.updates < self.config.critic_warmup_updates

    def collect(self, observation, frames):
        config = self.config
        state = to_tensors(observation, self.device)
        if not self.buffer.state:
            self.buffer.allocate_state(state)

        for step in range(config.rollout_steps):
            state = to_tensors(observation, self.device)
            pixels = torch.as_tensor(frames, device=self.device)

            with torch.no_grad():
                logits = self.policy.logits(pixels)
                distributions = [Categorical(logits=head) for head in logits]
                actions = torch.stack([d.sample() for d in distributions], dim=-1)
                log_prob = sum(d.log_prob(actions[:, i])
                               for i, d in enumerate(distributions))
                value = self.policy.critic(state)
                teacher_logits, _ = self.teacher(state)

            for key in self.buffer.state:
                self.buffer.state[key][step] = state[key]
            self.buffer.frames[step] = pixels
            self.buffer.actions[step] = actions
            self.buffer.log_probs[step] = log_prob
            self.buffer.values[step] = value
            self.buffer.teacher_logits[step] = torch.cat(teacher_logits, dim=-1)

            observation, rewards, terminated, truncated, infos = self.env.step(
                actions.cpu().numpy())
            done = terminated | truncated

            self.buffer.rewards[step] = torch.as_tensor(
                rewards, dtype=torch.float32, device=self.device)
            self.buffer.dones[step] = torch.as_tensor(
                done, dtype=torch.float32, device=self.device)

            for info in infos:
                if "episode" in info:
                    self._returns.append(info["episode"]["r"])
                    self._successes.append(bool(info["episode"].get("success")))

            if done.any():
                self.env.reset_done(done)
                observation = self.env._stack_observations()
                self.capture.reset(done)
            frames = self.capture.observe()
            self.global_step += self.env.num_envs

        with torch.no_grad():
            last_values = self.policy.critic(to_tensors(observation, self.device))
        return observation, frames, last_values

    def advantages(self, last_values: torch.Tensor):
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

        return advantages, advantages + self.buffer.values

    def update(self, advantages: torch.Tensor, returns: torch.Tensor) -> dict:
        config = self.config
        batch = config.rollout_steps * self.env.num_envs

        frames = self.buffer.frames.reshape((batch,) + self.buffer.pixel_shape)
        state = {key: value.reshape((batch,) + value.shape[2:])
                 for key, value in self.buffer.state.items()}
        actions = self.buffer.actions.reshape(batch, len(ACTION_DIMS))
        old_log_probs = self.buffer.log_probs.reshape(batch)
        teacher_logits = self.buffer.teacher_logits.reshape(batch, -1)
        flat_advantages = advantages.reshape(batch)
        flat_returns = returns.reshape(batch)

        minibatch_size = batch // config.minibatches
        indices = np.arange(batch)
        axis_names = ("move_x", "move_y", "shoot_x", "shoot_y")
        stats = {k: [] for k in ("policy_loss", "value_loss", "entropy",
                                 "teacher_kl", "clip_fraction", "move_entropy")}
        # Per axis, because a summed entropy hides the one failure that matters:
        # combat-v5 onward had a shoot axis frozen and the other uniform, which
        # summed to an unremarkable ~1.10 and went unnoticed for three runs.
        stats.update({f"entropy_{name}": [] for name in axis_names})

        for _ in range(config.epochs):
            np.random.shuffle(indices)
            for start in range(0, batch, minibatch_size):
                subset = torch.as_tensor(indices[start:start + minibatch_size],
                                         device=self.device)

                value = self.policy.critic({k: v[subset] for k, v in state.items()})
                value_loss = F.mse_loss(value, flat_returns[subset])

                # Warm-up: only the critic learns. No freezing is needed —
                # the critic reads privileged state through its own network and
                # the actor reads pixels through another, so this loss simply
                # has no path to the actor's parameters.
                if self.warming_up:
                    loss = value_loss
                    policy_loss = torch.zeros((), device=self.device)
                    entropy_loss = torch.zeros((), device=self.device)
                    kl = torch.zeros((), device=self.device)
                    clipped_fraction = 0.0
                    per_head = None
                else:
                    logits = self.policy.logits(frames[subset])
                    distributions = [Categorical(logits=head) for head in logits]
                    log_prob = sum(d.log_prob(actions[subset][:, i])
                                   for i, d in enumerate(distributions))
                    per_head = [d.entropy() for d in distributions]
                    entropy_loss = sum(per_head).mean()

                    ratio = torch.exp(log_prob - old_log_probs[subset])
                    batch_adv = flat_advantages[subset]
                    batch_adv = ((batch_adv - batch_adv.mean())
                                 / (batch_adv.std() + 1e-8))

                    unclipped = ratio * batch_adv
                    clipped = torch.clamp(ratio, 1 - config.clip_range,
                                          1 + config.clip_range) * batch_adv
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    clipped_fraction = float(
                        ((ratio - 1).abs() > config.clip_range).float().mean())

                    # Anchor to the teacher, per head, on the same soft targets
                    # distillation used.
                    teacher_heads = torch.split(teacher_logits[subset],
                                                list(ACTION_DIMS), dim=-1)
                    kl = sum(
                        F.kl_div(F.log_softmax(student, dim=-1),
                                 F.log_softmax(head, dim=-1),
                                 reduction="batchmean", log_target=True)
                        for student, head in zip(logits, teacher_heads))

                    loss = (policy_loss
                            + config.value_coef * value_loss
                            - config.entropy_coef * entropy_loss
                            + config.teacher_kl_coef * kl)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         config.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"].append(float(policy_loss))
                stats["value_loss"].append(float(value_loss))
                stats["entropy"].append(float(entropy_loss))
                stats["teacher_kl"].append(float(kl))
                stats["clip_fraction"].append(clipped_fraction)
                stats["move_entropy"].append(
                    sum(float(per_head[i].mean()) for i in MOVE_HEADS)
                    if per_head is not None else 0.0)
                for index, name in enumerate(axis_names):
                    stats[f"entropy_{name}"].append(
                        float(per_head[index].mean())
                        if per_head is not None else 0.0)

        result = {key: float(np.mean(values)) for key, values in stats.items()}

        # How much of the return's variance the critic actually explains. Raw
        # value_loss cannot answer this: its scale is set by how variable the
        # returns happen to be, so a large number can mean a bad critic or a
        # volatile task and there is no way to tell them apart. Explained
        # variance is scale-free — 1.0 is perfect, 0.0 is no better than
        # predicting the mean, and negative is worse than that. If this sits
        # near zero the advantages are noise and PPO is optimising nothing,
        # which is indistinguishable from "fine-tuning does not help" unless it
        # is measured.
        with torch.no_grad():
            values = self.buffer.values.reshape(-1)
            target = returns.reshape(-1)
            variance = target.var()
            result["explained_variance"] = float(
                1.0 - (target - values).var() / variance) if variance > 1e-8 else 0.0
        return result

    def _log(self, record: dict) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def save(self, name: str = "student.pt") -> Path:
        path = self.run_dir / name
        torch.save({
            "student": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "updates": self.updates,
            "pixel_shape": self.capture.shape,
        }, path)
        return path

    def train(self, total_steps: int, log_every: int = 1) -> None:
        observation = self.env.reset()
        self.capture.reset()
        frames = self.capture.observe()
        started = time.perf_counter()
        last_log: tuple[int, float] | None = None

        while self.global_step < total_steps:
            observation, frames, last_values = self.collect(observation, frames)
            advantages, returns = self.advantages(last_values)
            # Captured before the counter moves: `warming_up` is derived from
            # the update count, so reading it after the increment reports the
            # state of the *next* update and mislabels the boundary row.
            was_warming = self.warming_up
            stats = self.update(advantages, returns)
            self.updates += 1

            if hasattr(self.env, "curriculum"):
                self.env.curriculum.advance()

            self._returns = self._returns[-100:]
            self._successes = self._successes[-100:]

            if self.updates % log_every == 0:
                now = time.perf_counter()
                if last_log is not None:
                    span = now - last_log[1]
                    current = (self.global_step - last_log[0]) / span if span else 0.0
                else:
                    current = self.global_step / (now - started)
                last_log = (self.global_step, now)

                record = {
                    "update": self.updates,
                    "global_step": self.global_step,
                    "steps_per_second": round(self.global_step / (now - started), 1),
                    "steps_per_second_now": round(current, 1),
                    "warming_up": was_warming,
                    "success_rate": round(float(np.mean(self._successes)), 3)
                    if self._successes else None,
                    "mean_return": round(float(np.mean(self._returns)), 3)
                    if self._returns else None,
                    "episodes": len(self._returns),
                    "room_exits": getattr(self.env, "room_exits", 0),
                    "full_restarts": getattr(self.env, "restarts", 0),
                    "alive_instances": self.env.alive_count,
                    **{k: round(v, 4) for k, v in stats.items()},
                }
                if hasattr(self.env, "curriculum"):
                    record.update(self.env.curriculum.state())
                self._log(record)
                print(f"update {record['update']:4d} "
                      f"| step {record['global_step']:7d} "
                      f"| {record['steps_per_second_now']:6.1f} st/s "
                      f"| {'WARMUP ' if was_warming else ''}"
                      f"success {record['success_rate']} "
                      f"| difficulty {record.get('difficulty')} "
                      f"| v-loss {stats['value_loss']:.3f} "
                      f"| ev {stats['explained_variance']:+.2f} "
                      f"| kl {stats['teacher_kl']:.4f}")

            if self.updates % 20 == 0:
                self.save()

        self.save()
