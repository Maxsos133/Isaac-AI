"""Does the pixel student actually play, or does it only predict the teacher?

distill-v2 answered the first half of the question well: KL fell from 1.41 to
0.42 and move agreement reached the ceiling a perfect student could score. None
of that says the student can win a fight. `student_success` from that run
cannot answer it either — the curriculum oscillated between difficulty 0.00 and
1.00, so the number is an average over tasks that were never the same twice.

So: three arms at a *fixed* difficulty, run simultaneously on one fleet.

  teacher   the state-based policy the student was distilled from — the target
  student   pixels only, no privileged state anywhere in its path
  random    the floor, because "better than nothing" is not obvious in advance

Arms are assigned per instance and run at the same time, so all three meet the
same curriculum, the same encounters and the same machine load. Sweeping a few
difficulties gives a curve rather than a single number, which is what shows
whether the student degrades gracefully or falls off a cliff.

    .venv/Scripts/python.exe scripts/diagnose_student.py
    .venv/Scripts/python.exe scripts/diagnose_student.py --difficulty 0.3,0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.capture import FleetCapture  # noqa: E402
from isaac_ai.combat import CombatVecEnv  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.curriculum import CombatCurriculum  # noqa: E402
from isaac_ai.distill import load_teacher  # noqa: E402
from isaac_ai.env import ACTION_DIMS  # noqa: E402
from isaac_ai.pixel_policy import PixelActorCritic  # noqa: E402
from isaac_ai.policy import to_tensors  # noqa: E402

# With one student: `greedy` takes its most likely action rather than sampling,
# which mattered while the student was weak and stopped mattering once it fitted
# the teacher well. Given a second student (--student-b), that slot becomes the
# comparison instead — two checkpoints meeting the same encounters on the same
# fleet, which is the only way to tell two training runs apart without the
# between-run variance swamping the difference.
ARMS = ("teacher", "student", "greedy", "random")
ARMS_AB = ("teacher", "student", "student-b", "random")


def load_student(path: Path, device: torch.device) -> PixelActorCritic:
    """Load a student's *actor* only, ignoring a stale privileged critic.

    The critic reads the mod's state and is discarded at deployment, so its
    shape follows whatever ENTITY_FEATURES happened to be when the run was
    trained. Insisting on it would make every earlier student unloadable the
    moment an observation feature is added — which is exactly what stopped
    distill-v3 being comparable against distill-v4. The actor is pixels-only and
    its shape never moved, so that is what gets loaded, and a missing actor
    weight is still a hard error.
    """
    from isaac_ai.env import DOOR_FEATURES, MAX_DOORS, SCALAR_FEATURES

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    weights = checkpoint["student"]
    model = PixelActorCritic(tuple(checkpoint["pixel_shape"]),
                             SCALAR_FEATURES, MAX_DOORS * DOOR_FEATURES).to(device)

    expected = model.state_dict()
    actor = {key: value for key, value in weights.items()
             if not key.startswith("critic")}
    missing = [key for key in expected
               if not key.startswith("critic") and key not in actor]
    wrong = [key for key, value in actor.items()
             if key in expected and value.shape != expected[key].shape]
    if missing or wrong:
        raise ValueError(f"{path}: actor does not fit — missing {missing}, "
                         f"mismatched {wrong}")

    model.load_state_dict(actor, strict=False)
    model.eval()
    return model


def evaluate(env: CombatVecEnv, capture: FleetCapture, teacher, student,
             arm_of: np.ndarray, device: torch.device, episodes: int,
             arms: tuple[str, ...], student_b=None) -> dict[str, list[bool]]:
    """Run until every arm has `episodes` finished episodes."""
    outcomes: dict[str, list[bool]] = {arm: [] for arm in arms}
    observation = env.reset()
    capture.reset()
    frames = capture.observe()
    rng = np.random.default_rng(0)

    while min(len(v) for v in outcomes.values()) < episodes:
        state = to_tensors(observation, device)
        pixels = torch.as_tensor(frames, device=device)

        with torch.no_grad():
            teacher_logits, _ = teacher(state)
            teacher_actions = torch.stack(
                [torch.distributions.Categorical(logits=h).sample()
                 for h in teacher_logits], dim=-1).cpu().numpy()
            student_actions = student.act_from_pixels(pixels).cpu().numpy()
            if student_b is not None:
                other_actions = student_b.act_from_pixels(pixels).cpu().numpy()
            else:
                other_actions = student.act_from_pixels(
                    pixels, deterministic=True).cpu().numpy()

        actions = np.zeros((env.num_envs, len(ACTION_DIMS)), dtype=np.int64)
        for index in range(env.num_envs):
            arm = arms[arm_of[index]]
            if arm == "teacher":
                actions[index] = teacher_actions[index]
            elif arm == "student":
                actions[index] = student_actions[index]
            elif arm in ("greedy", "student-b"):
                actions[index] = other_actions[index]
            else:
                actions[index] = rng.integers(0, 3, size=len(ACTION_DIMS))

        observation, _, terminated, truncated, infos = env.step(actions)
        done = terminated | truncated

        for index, info in enumerate(infos):
            if "episode" in info:
                arm = arms[arm_of[index]]
                if len(outcomes[arm]) < episodes:
                    outcomes[arm].append(bool(info["episode"]["success"]))

        if done.any():
            env.reset_done(done)
            observation = env._stack_observations()
            capture.reset(done)
        frames = capture.observe()

    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=25,
                        help="episodes per arm per difficulty")
    # Reaches 0.8 because the teacher moved: combat-v4 topped out near 0.57 and
    # combat-v5 was still climbing at 0.75, so a sweep ending at 0.6 now stops
    # below where the teacher operates and would miss the interesting half.
    parser.add_argument("--difficulty", default="0.3,0.45,0.6,0.7,0.8",
                        help="comma-separated fixed difficulties to sweep")
    parser.add_argument("--student", default="runs/distill-v4/student.pt")
    parser.add_argument("--student-b", default=None,
                        help="second checkpoint to compare on the same "
                             "encounters; replaces the greedy arm")
    parser.add_argument("--teacher", default="runs/combat-v5/policy.pt")
    # The random arm almost never clears a room, so with the training cap of
    # 450 every one of its episodes runs to the buzzer and it alone sets how
    # long the sweep takes. A tighter cap costs the good arms nothing: the
    # teacher clears in well under this.
    parser.add_argument("--max-steps", type=int, default=250,
                        help="encounter step cap during evaluation")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = load_teacher(config.root / args.teacher, device)
    student = load_student(config.root / args.student, device)
    student_b = (load_student(config.root / args.student_b, device)
                 if args.student_b else None)
    arms = ARMS_AB if student_b is not None else ARMS
    difficulties = [float(x) for x in args.difficulty.split(",")]

    fleet = launcher.bring_up(config, count=args.instances)
    capture = None
    try:
        bridges = fleet.bridges()
        if not bridges:
            raise SystemExit("no instance reached a run")

        curriculum = CombatCurriculum(max_enemies=config.combat.max_enemies)
        env = CombatVecEnv(bridges, config, curriculum,
                           max_encounter_steps=args.max_steps)
        capture = FleetCapture(
            [i.hwnd for i in fleet.instances if i.ready and not i.failed],
            config.pixels)

        # Round-robin so the arms are spread across the fleet rather than
        # clustered, and every arm meets the same encounters at the same time.
        arm_of = np.array([i % len(arms) for i in range(env.num_envs)])
        for arm in arms:
            count = int((arm_of == arms.index(arm)).sum())
            print(f"  {arm:<8} on {count} instance(s)")
        if min(int((arm_of == i).sum()) for i in range(len(arms))) == 0:
            raise SystemExit(f"need at least {len(arms)} instances")

        results: dict[float, dict[str, float]] = {}
        for difficulty in difficulties:
            # Frozen: advance() is never called, so the dial cannot drift and
            # every arm is measured on the same task.
            curriculum.difficulty = difficulty
            print(f"\n== difficulty {difficulty:.2f} "
                  f"({len(curriculum.available())} enemy types) ==")
            outcomes = evaluate(env, capture, teacher, student, arm_of,
                                device, args.episodes, arms, student_b)
            results[difficulty] = {a: float(np.mean(v)) for a, v in outcomes.items()}
            for arm in arms:
                values = outcomes[arm]
                print(f"  {arm:<8} {np.mean(values):.2f} "
                      f"({sum(values)}/{len(values)} encounters won)")

        print(f"\n{'difficulty':<12}" + "".join(f"{a:>11}" for a in arms)
              + f"{'student/teacher':>18}")
        for difficulty, row in results.items():
            ratio = row["student"] / row["teacher"] if row["teacher"] else float("nan")
            print(f"{difficulty:<12.2f}"
                  + "".join(f"{row[a]:>11.2f}" for a in arms)
                  + f"{ratio:>18.2f}")
    finally:
        if capture is not None:
            capture.stop()
        fleet.shutdown()


if __name__ == "__main__":
    main()
