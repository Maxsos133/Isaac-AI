"""Distilling the state-based teacher into the pixels-only student.

Supervised learning on (frame -> teacher action) pairs. The teacher reads the
mod's privileged state and plays; the student sees only what the screen showed
and learns to predict what the teacher did. No reward is involved, so none of
the credit-assignment problems that have cost this project four navigation runs
apply here — the label is exact and arrives on the same step as the frame.

Two decisions are worth explaining.

**Soft labels.** The student matches the teacher's whole action distribution,
not its argmax. The teacher genuinely hedges when two directions are equally
good, and a student trained on hard labels has to memorise which arbitrary
tie-break to copy instead of learning that the choice was open.

**Episode-level driver assignment.** Pure behaviour cloning only ever sees
states the teacher visits, so the student never learns to recover from its own
mistakes. The fix is to let the student drive sometimes while the teacher keeps
labelling. That share is assigned per *episode* rather than per step: a
half-student, half-teacher trajectory has no honest success rate attached to
it, and on this project a metric that cannot be attributed has been worth less
than no metric at all.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.distributions import Categorical

from isaac_ai.capture import save_stack_image
from isaac_ai.env import ACTION_DIMS, MOVE_HEADS
from isaac_ai.pixel_policy import PixelActorCritic, distillation_loss
from isaac_ai.policy import ActorCritic, to_tensors


@dataclass
class DistillConfig:
    learning_rate: float = 3e-4
    rollout_steps: int = 64
    epochs: int = 2
    minibatch_size: int = 192
    max_grad_norm: float = 0.5
    # How often the student drives an episode, ramped over training. It starts
    # at zero because a student driving from random weights produces nothing
    # but death, and every frame of that is a state the teacher will never be
    # in — labels for a distribution that does not matter yet.
    student_share_start: float = 0.0
    student_share_end: float = 0.9
    student_share_steps: int = 300_000
    # Save what the student is actually looking at, every N updates. Rooms fill
    # with blood as encounters run and only clear when the player dies, so the
    # student's view drifts in a way no privileged metric can show. 0 disables.
    sample_every: int = 20
    # Whether the student drives with its most likely action rather than a
    # sample. Off, because the measured advantage does not survive into the
    # region training actually happens in: on distill-v2's student the mode beat
    # sampling 0.92 to 0.67 at difficulty 0.20 and 0.67 to 0.46 at 0.30, then
    # lost 0.38 to 0.46 at 0.40 and both collapsed at 0.50. A deterministic
    # driver also visits a narrower band of states, which is the opposite of
    # what the student's turn at the wheel is for.
    #
    # This is separate from how the finished student should be *run*, where the
    # mode is clearly better at the difficulties it can handle. `diagnose_student`
    # measures that as its own arm.
    student_greedy: bool = False


def load_teacher(path: Path, device: torch.device) -> ActorCritic:
    """Load a frozen state-based policy, refusing anything that half-fits.

    A partial load is worse than no load here. `--resume` elsewhere in this
    project deliberately copies what matches and reinitialises the rest, which
    is right for continuing training and wrong for a teacher: a teacher with a
    reinitialised trunk still produces confident-looking logits, and the student
    would spend the whole run faithfully imitating noise.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("policy", checkpoint)

    teacher = ActorCritic().to(device)
    expected = teacher.state_dict()
    mismatched = sorted(
        key for key in set(expected) | set(state)
        if key not in state or key not in expected
        or tuple(state[key].shape) != tuple(expected[key].shape)
    )
    if mismatched:
        raise ValueError(
            f"{path} does not fit the current network: {', '.join(mismatched)}.\n"
            "This checkpoint predates a change to the observation encoding "
            "(the door encoder and the wider scalar vector), so it cannot act "
            "as a teacher. Train a fresh combat teacher with the current code "
            "and distil from that."
        )

    teacher.load_state_dict(state)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


class Distiller:
    def __init__(self, env, capture, teacher: ActorCritic,
                 config: DistillConfig, device: torch.device,
                 run_dir: Path) -> None:
        self.env = env
        self.capture = capture
        self.teacher = teacher
        self.config = config
        self.device = device
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = run_dir / "metrics.jsonl"

        self.student = PixelActorCritic(capture.shape, *self._state_widths()).to(device)
        self.optimizer = torch.optim.Adam(self.student.parameters(),
                                          lr=config.learning_rate, eps=1e-5)
        self.global_step = 0
        self.updates = 0

        # True where the student is driving this episode.
        self._student_driven = np.zeros(env.num_envs, dtype=bool)
        self._student_outcomes: list[bool] = []
        self._teacher_outcomes: list[bool] = []
        self._last_log: tuple[int, float] | None = None

    def _state_widths(self) -> tuple[int, int]:
        from isaac_ai.env import DOOR_FEATURES, MAX_DOORS, SCALAR_FEATURES
        return SCALAR_FEATURES, MAX_DOORS * DOOR_FEATURES

    def student_share(self) -> float:
        config = self.config
        if config.student_share_steps <= 0:
            return config.student_share_end
        progress = min(1.0, self.global_step / config.student_share_steps)
        return (config.student_share_start
                + progress * (config.student_share_end - config.student_share_start))

    def _assign_drivers(self, mask: np.ndarray) -> None:
        """Pick who drives each episode that is about to start."""
        share = self.student_share()
        for index in np.flatnonzero(mask):
            self._student_driven[index] = np.random.random() < share

    def collect(self, observation, frames):
        """Roll out, recording a frame and the teacher's logits for each step."""
        config = self.config
        stacked_frames: list[torch.Tensor] = []
        stacked_logits: list[torch.Tensor] = []
        agreements: list[float] = []

        for _ in range(config.rollout_steps):
            state = to_tensors(observation, self.device)
            pixels = torch.as_tensor(frames, device=self.device)

            with torch.no_grad():
                teacher_logits, _ = self.teacher(state)
                teacher_actions = torch.stack(
                    [Categorical(logits=head).sample() for head in teacher_logits],
                    dim=-1)
                student_logits = self.student.logits(pixels)
                if config.student_greedy:
                    student_actions = torch.stack(
                        [head.argmax(dim=-1) for head in student_logits], dim=-1)
                else:
                    student_actions = torch.stack(
                        [Categorical(logits=head).sample()
                         for head in student_logits], dim=-1)

            stacked_frames.append(pixels)
            stacked_logits.append(torch.cat(teacher_logits, dim=-1))
            agreements.append(self._agreement(student_logits, teacher_logits))

            driving = torch.as_tensor(self._student_driven, device=self.device)
            actions = torch.where(driving.unsqueeze(-1),
                                  student_actions, teacher_actions)

            # Every episode feeds the difficulty dial, the student's included.
            #
            # An earlier version masked the student out so difficulty tracked
            # the teacher's competence. That is backwards: the learner here is
            # the student, and a curriculum exists to hold the task at the
            # learner's edge. Measured at fixed difficulty, this student wins
            # 0.46 at 0.30 and 0.00 at 0.50 while the teacher still wins 0.42 —
            # so pinning training to the teacher's level would spend the run
            # somewhere the student cannot learn anything at all.
            #
            # Unmasked, difficulty starts near the teacher's level while it
            # drives most episodes and drifts down to the student's as the
            # share ramps, then climbs again as the student improves. That is
            # the curriculum doing its job. The oscillation seen in distill-v2
            # was never caused by this; it was `advance()` stepping per update
            # while evidence arrived per episode, fixed in CombatCurriculum.
            if hasattr(self.env, "curriculum_mask"):
                self.env.curriculum_mask = None

            observation, _, terminated, truncated, infos = self.env.step(
                actions.cpu().numpy())
            done = terminated | truncated

            for index, info in enumerate(infos):
                if "episode" not in info:
                    continue
                success = bool(info["episode"].get("success"))
                target = (self._student_outcomes if self._student_driven[index]
                          else self._teacher_outcomes)
                target.append(success)

            if done.any():
                self.env.reset_done(done)
                observation = self.env._stack_observations()
                # Frames from before the reset belong to a different episode;
                # keeping them would show the network a scene that teleported.
                self.capture.reset(done)
                self._assign_drivers(done)

            frames = self.capture.observe()
            self.global_step += self.env.num_envs

        return (observation, frames,
                torch.stack(stacked_frames), torch.stack(stacked_logits),
                float(np.mean(agreements)) if agreements else 0.0)

    def _agreement(self, student_logits: list[torch.Tensor],
                   teacher_logits: list[torch.Tensor]) -> float:
        """How often the student's most likely move is the teacher's.

        Compares modes, not samples. Comparing two *sampled* actions looks like
        the same question and is not: two draws from an identical distribution
        only coincide at the collision rate. At combat-v4's move entropy that
        caps a flawless student at 0.50-0.61, so the sampled version reported a
        student 41% of the way there as "barely above the 0.333 chance floor"
        and made a working run look stalled. This version has a ceiling of 1.0
        and means what it says.

        Movement only. The shoot heads sit near uniform whenever there is
        nothing to shoot, so folding them in would score a policy as half-right
        while its movement is entirely wrong.
        """
        matches = [student_logits[i].argmax(dim=-1) == teacher_logits[i].argmax(dim=-1)
                   for i in MOVE_HEADS]
        return float(torch.stack(matches).float().mean())

    def update(self, frames: torch.Tensor, logits: torch.Tensor) -> dict:
        config = self.config
        steps, num_envs = frames.shape[0], frames.shape[1]
        frames = frames.reshape(steps * num_envs, *frames.shape[2:])
        logits = logits.reshape(steps * num_envs, -1)

        splits = list(ACTION_DIMS)
        losses: list[float] = []
        batch = steps * num_envs

        for _ in range(config.epochs):
            order = torch.randperm(batch, device=self.device)
            for start in range(0, batch, config.minibatch_size):
                index = order[start:start + config.minibatch_size]
                student_logits = self.student.logits(frames[index])
                teacher_logits = list(torch.split(logits[index], splits, dim=-1))

                loss = distillation_loss(student_logits, teacher_logits)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(),
                                               config.max_grad_norm)
                self.optimizer.step()
                losses.append(float(loss))

        return {"kl": float(np.mean(losses))}

    def _log(self, record: dict) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def save(self, name: str = "student.pt") -> Path:
        path = self.run_dir / name
        torch.save({
            "student": self.student.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "updates": self.updates,
            "pixel_shape": self.capture.shape,
        }, path)
        return path

    def train(self, total_steps: int, log_every: int = 1) -> None:
        observation = self.env.reset()
        self.capture.reset()
        self._assign_drivers(np.ones(self.env.num_envs, dtype=bool))
        frames = self.capture.observe()
        started = time.perf_counter()

        while self.global_step < total_steps:
            observation, frames, batch_frames, batch_logits, agreement = \
                self.collect(observation, frames)
            stats = self.update(batch_frames, batch_logits)
            self.updates += 1

            if hasattr(self.env, "curriculum"):
                self.env.curriculum.advance()

            if self.updates % log_every == 0:
                now = time.perf_counter()
                elapsed = now - started

                # Two rates, because the cumulative one alone is misleading:
                # being an average since the run began, it starts low and
                # creeps upward all run even after throughput is flat. The
                # pixel path costs ~29% for the first dozen updates while cuDNN
                # picks kernels and the capture sessions fill, and reading that
                # off the cumulative figure makes it look like a slow recovery
                # rather than a one-off.
                if self._last_log is not None:
                    last_step, last_time = self._last_log
                    span = now - last_time
                    current = (self.global_step - last_step) / span if span else 0.0
                else:
                    current = self.global_step / elapsed
                self._last_log = (self.global_step, now)

                # Success is reported per driver. A combined number would climb
                # purely because the teacher drives most episodes early on, and
                # would say nothing at all about the student.
                record = {
                    "update": self.updates,
                    "global_step": self.global_step,
                    "steps_per_second": round(self.global_step / elapsed, 1),
                    "steps_per_second_now": round(current, 1),
                    "kl": round(stats["kl"], 5),
                    "move_agreement": round(agreement, 3),
                    "student_share": round(self.student_share(), 3),
                    "student_success": round(float(np.mean(
                        self._student_outcomes[-100:])), 3)
                    if self._student_outcomes else None,
                    "teacher_success": round(float(np.mean(
                        self._teacher_outcomes[-100:])), 3)
                    if self._teacher_outcomes else None,
                    "student_episodes": len(self._student_outcomes),
                    "alive_instances": self.env.alive_count,
                    # Canaries the state-based trainer logs and this one was
                    # missing. room_exits should stay near zero: it counts
                    # agents that walked out of a fight, which is scored as a
                    # loss and leaves that instance in a different room for
                    # every encounter afterwards.
                    "room_exits": getattr(self.env, "room_exits", 0),
                    "full_restarts": getattr(self.env, "restarts", 0),
                    "deaths": getattr(self.env, "deaths", 0),
                }
                if hasattr(self.env, "curriculum"):
                    record.update(self.env.curriculum.state())
                self._log(record)
                print(f"update {record['update']:4d} "
                      f"| step {record['global_step']:7d} "
                      f"| {record['steps_per_second']:6.1f} st/s "
                      f"| kl {record['kl']:.4f} "
                      f"| agree {record['move_agreement']:.3f} "
                      f"| share {record['student_share']:.2f} "
                      f"| student {record['student_success']} "
                      f"| teacher {record['teacher_success']} "
                      f"| exits {record['room_exits']}")

            if (self.config.sample_every
                    and self.updates % self.config.sample_every == 0):
                save_stack_image(
                    frames[0], self.capture.channels,
                    self.run_dir / "frames" / f"step-{self.global_step:08d}.png")

            if self.updates % 20 == 0:
                self.save()

        self.save()
