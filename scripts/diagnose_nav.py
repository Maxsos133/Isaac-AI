"""Is the navigation task solvable at all?

Navigation-only training got *worse* over 300k steps, which rules out credit
assignment — the episode is ten seconds long with dense shaping. That leaves two
very different possibilities:

  the task is broken   - doors unreachable, arrivals not detected, something
                         mechanical, in which case no policy can score
  the learning is broken - a scripted controller walks it easily and only the
                         learner fails

A hand-written "walk at the nearest door" arm separates them in one run. This is
the same trick that settled the aiming question.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.env import ACTION_DIMS  # noqa: E402
from isaac_ai.nav_curriculum import NavigationCurriculum  # noqa: E402
from isaac_ai.navigation import NavigationVecEnv  # noqa: E402
from isaac_ai.policy import ActorCritic, to_tensors  # noqa: E402

AXIS_INDEX = {-1: 0, 0: 1, 1: 2}


@dataclass
class Arm:
    name: str
    episodes: int = 0
    successes: int = 0
    steps: int = 0
    lengths: list[int] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)

    def line(self) -> str:
        rate = self.successes / max(self.episodes, 1)
        length = sum(self.lengths) / max(len(self.lengths), 1)
        dist = sum(self.distances) / max(len(self.distances), 1)
        return (f"{self.name:10s} {rate:8.1%} {self.episodes:9d} "
                f"{length:11.0f} {dist:14.0f}")


def nearest_door(obs: dict) -> tuple[float, float] | None:
    """Vector from the player to the closest open, unlocked door."""
    if not obs or not obs.get("ready", True) or "room" not in obs:
        return None
    player = obs["player"]
    best = None
    best_d = None
    for door in obs["room"].get("doors", []):
        if door.get("locked") or not door.get("open"):
            continue
        dx = float(door["x"]) - float(player["x"])
        dy = float(door["y"]) - float(player["y"])
        d = math.hypot(dx, dy)
        if best_d is None or d < best_d:
            best_d, best = d, (dx, dy)
    return best


def scripted_action(vector: tuple[float, float]) -> tuple[int, int]:
    """Head straight at it. Isaac moves on eight directions, so push both axes
    whenever the component is meaningful."""
    dx, dy = vector
    scale = max(abs(dx), abs(dy), 1.0)
    mx = 0
    my = 0
    if abs(dx) / scale > 0.3:
        mx = 1 if dx > 0 else -1
    if abs(dy) / scale > 0.3:
        my = 1 if dy > 0 else -1
    if mx == 0 and my == 0:
        mx = 1 if dx > 0 else -1
    return mx, my


def run_arm(env, name, episodes, policy, device, rng) -> Arm:
    arm = Arm(name=name)
    env.reset()
    observation = env._stack_observations()

    while arm.episodes < episodes:
        raw = list(env._latest)

        if policy is not None:
            with torch.no_grad():
                actions, _, _ = policy.act(to_tensors(observation, device))
            actions = actions.cpu().numpy()
        else:
            actions = np.stack([rng.integers(0, d, size=env.num_envs)
                                for d in ACTION_DIMS], axis=1)
            if name == "scripted":
                for index in range(env.num_envs):
                    vector = nearest_door(raw[index])
                    if vector is not None:
                        mx, my = scripted_action(vector)
                        actions[index][0] = AXIS_INDEX[mx]
                        actions[index][1] = AXIS_INDEX[my]

        for index in range(env.num_envs):
            vector = nearest_door(raw[index])
            if vector is not None:
                arm.distances.append(math.hypot(*vector))

        observation, _, terminated, truncated, infos = env.step(actions)
        arm.steps += env.num_envs

        for info in infos:
            if "episode" in info:
                arm.episodes += 1
                arm.lengths.append(info["episode"]["l"])
                if info["episode"]["success"]:
                    arm.successes += 1

        done = terminated | truncated
        if done.any():
            env.reset_done(done)
            observation = env._stack_observations()

    return arm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--checkpoint", type=str, default="runs/nav-v1/policy.pt")
    args = parser.parse_args()

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = None
    checkpoint = config.root / args.checkpoint
    if checkpoint.exists():
        policy = ActorCritic().to(device)
        policy.load_state_dict(torch.load(checkpoint, map_location=device)["policy"])
        policy.eval()
        print(f"loaded {checkpoint}")

    fleet = launcher.bring_up(config, count=args.instances)
    try:
        if fleet.ready_count == 0:
            print("no instances reached a run")
            return 1

        env = NavigationVecEnv(fleet.bridges(), config, NavigationCurriculum())
        print(f"\n{'arm':10s} {'success':>8s} {'episodes':>9s} "
              f"{'mean steps':>11s} {'door dist px':>14s}")
        print("-" * 58)

        rng = np.random.default_rng(11)
        results = {}
        for name, model in (("random", None), ("scripted", None), ("trained", policy)):
            if name == "trained" and policy is None:
                continue
            arm = run_arm(env, name, args.episodes, model, device, rng)
            results[name] = arm
            print(arm.line())

        print()
        if "scripted" in results:
            scripted = results["scripted"].successes / max(results["scripted"].episodes, 1)
            if scripted < 0.5:
                print("VERDICT: the TASK is broken. A controller that walks straight")
                print("  at the nearest door cannot solve it, so no learner could.")
                print("  Look at reachability, arrival detection, or the door data.")
            else:
                print(f"VERDICT: the task is solvable ({scripted:.0%} scripted).")
                print("  The learner is what is failing — reward scale, the value")
                print("  function, or the policy's use of the door features.")
        return 0
    finally:
        fleet.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
