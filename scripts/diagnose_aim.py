"""Is aiming the ceiling?

Three arms at a fixed difficulty, so aim is the only variable:

  random   - random movement, random shooting          (floor)
  aimbot   - random movement, shoots at nearest enemy  (isolates aim alone)
  trained  - the learned policy                        (what we have)

If aimbot >> random while trained sits near random, the policy has not learned
to aim and aiming is worth a large amount of performance — the bottleneck.
If trained is near aimbot, aim is solved and the ceiling is somewhere else.

Movement is random in both baseline arms on purpose: it holds positioning
constant so the comparison measures aim and nothing else.
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
from isaac_ai.combat import CombatVecEnv  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.curriculum import CombatCurriculum  # noqa: E402
from isaac_ai.env import ACTION_DIMS, AXIS_VALUES  # noqa: E402
from isaac_ai.policy import ActorCritic, to_tensors  # noqa: E402

# The eight directions Isaac can shoot, as (sx, sy) index pairs into AXIS_VALUES.
AXIS_INDEX = {-1: 0, 0: 1, 1: 2}


@dataclass
class ArmStats:
    name: str
    episodes: int = 0
    successes: int = 0
    steps: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    kills: int = 0
    tears: int = 0
    shooting_steps: int = 0
    alignment: list[float] = field(default_factory=list)

    def report(self) -> str:
        success = self.successes / max(self.episodes, 1)
        accuracy = self.damage_dealt / max(self.tears, 1)
        align = sum(self.alignment) / max(len(self.alignment), 1)
        shoot_rate = self.shooting_steps / max(self.steps, 1)
        length = self.steps / max(self.episodes, 1)
        return (f"{self.name:9s} {success:7.1%} {length:8.0f} {self.kills / max(self.episodes,1):7.2f} "
                f"{self.tears / max(self.episodes,1):8.1f} {accuracy:9.3f} {align:10.3f} "
                f"{shoot_rate:8.1%} {self.damage_taken / max(self.episodes,1):8.2f}")


def nearest_enemy(obs: dict) -> tuple[float, float] | None:
    """Vector from the player to the closest live enemy, in room pixels."""
    if not obs or not obs.get("ready", True):
        return None
    enemies = [e for e in obs.get("entities", []) if e.get("k") == "enemy"]
    if not enemies:
        return None
    closest = min(enemies, key=lambda e: e["d"])
    player = obs["player"]
    return closest["x"] - player["x"], closest["y"] - player["y"]


def aim_action(vector: tuple[float, float]) -> tuple[int, int]:
    """Snap a direction onto Isaac's shooting axes.

    Isaac fires along axes, so the best available shot at a target is the axis
    with the larger component. Diagonals fire both axes.
    """
    dx, dy = vector
    if abs(dx) >= abs(dy):
        return (1 if dx > 0 else -1), 0
    return 0, (1 if dy > 0 else -1)


def alignment_of(shoot: tuple[int, int], vector: tuple[float, float]) -> float | None:
    """Cosine between where we shot and where the nearest enemy actually is."""
    sx, sy = shoot
    if sx == 0 and sy == 0:
        return None
    dx, dy = vector
    norm = math.hypot(dx, dy)
    shoot_norm = math.hypot(sx, sy)
    if norm == 0 or shoot_norm == 0:
        return None
    return (sx * dx + sy * dy) / (norm * shoot_norm)


def run_arm(env: CombatVecEnv, name: str, episodes: int,
            policy: ActorCritic | None, device, rng) -> ArmStats:
    stats = ArmStats(name=name)
    env.reset()
    observation = env._stack_observations()

    while stats.episodes < episodes:
        raw = list(env._latest)

        if policy is not None:
            with torch.no_grad():
                actions, _, _ = policy.act(to_tensors(observation, device))
            actions = actions.cpu().numpy()
        else:
            actions = np.stack([rng.integers(0, d, size=env.num_envs)
                                for d in ACTION_DIMS], axis=1)

        # Overriding only the two shooting axes leaves movement untouched, so
        # the gap between an arm and its un-overridden twin is attributable to
        # aim alone. "trained+aim" is the one that matters: it measures what
        # perfect aim would add on top of the movement we already have.
        if name in ("aimbot", "trained+aim"):
            for index in range(env.num_envs):
                vector = nearest_enemy(raw[index])
                if vector is not None:
                    sx, sy = aim_action(vector)
                    actions[index][2] = AXIS_INDEX[sx]
                    actions[index][3] = AXIS_INDEX[sy]

        # Record what we are about to do, against where the enemy is now.
        for index in range(env.num_envs):
            sx = AXIS_VALUES[int(actions[index][2])]
            sy = AXIS_VALUES[int(actions[index][3])]
            if sx or sy:
                stats.shooting_steps += 1
            vector = nearest_enemy(raw[index])
            if vector is not None:
                value = alignment_of((sx, sy), vector)
                if value is not None:
                    stats.alignment.append(value)

        observation, _, terminated, truncated, infos = env.step(actions)
        stats.steps += env.num_envs

        for index in range(env.num_envs):
            latest = env._latest[index]
            if latest and latest.get("ready", True):
                ev = latest.get("events", {})
                stats.damage_dealt += float(ev.get("damage_dealt", 0.0))
                stats.damage_taken += float(ev.get("damage_taken", 0.0))
                stats.kills += int(ev.get("kills", 0))
                stats.tears += int(ev.get("tears_fired", 0))

        for info in infos:
            if "episode" in info:
                stats.episodes += 1
                if info["episode"]["success"]:
                    stats.successes += 1

        done = terminated | truncated
        if done.any():
            env.reset_done(done)
            observation = env._stack_observations()

    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=90)
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=str, default="runs/combat-v2/policy.pt")
    args = parser.parse_args()

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = None
    checkpoint = config.root / args.checkpoint
    if checkpoint.exists():
        policy = ActorCritic().to(device)
        policy.load_state_dict(torch.load(checkpoint, map_location=device)["policy"])
        policy.eval()
        print(f"loaded policy from {checkpoint}")
    else:
        print(f"no checkpoint at {checkpoint}; skipping the trained arm")

    fleet = launcher.bring_up(config, count=args.instances)
    try:
        if fleet.ready_count == 0:
            print("no instances reached a run")
            return 1

        # Difficulty is pinned: the curriculum must not move underneath the
        # comparison or the three arms would be facing different games.
        curriculum = CombatCurriculum(seed=7)
        curriculum.difficulty = args.difficulty
        curriculum.advance = lambda: None  # type: ignore[method-assign]
        env = CombatVecEnv(fleet.bridges(), config, curriculum)

        encounter = curriculum.sample()
        print(f"\nfixed difficulty {args.difficulty} -> {encounter.enemy_count} enemies "
              f"from {len(curriculum.available())} types, {args.episodes} episodes per arm\n")
        print(f"{'arm':9s} {'success':>7s} {'len':>8s} {'kills':>7s} "
              f"{'tears':>8s} {'dmg/tear':>9s} {'aim cos':>10s} {'shoot%':>8s} {'taken':>8s}")
        print("-" * 80)

        rng = np.random.default_rng(3)
        results = []
        arms = (("random", None), ("aimbot", None),
                ("trained", policy), ("trained+aim", policy))
        for name, model in arms:
            if name.startswith("trained") and policy is None:
                continue
            stats = run_arm(env, name, args.episodes, model, device, rng)
            results.append(stats)
            print(stats.report())

        print()
        rate = {s.name: s.successes / max(s.episodes, 1) for s in results}

        if "random" in rate and "aimbot" in rate:
            print(f"aim alone, on random movement: {rate['random']:.1%} -> "
                  f"{rate['aimbot']:.1%} ({rate['aimbot'] - rate['random']:+.1%})")

        if "trained" in rate and "trained+aim" in rate:
            delta = rate["trained+aim"] - rate["trained"]
            print(f"aim on top of trained movement: {rate['trained']:.1%} -> "
                  f"{rate['trained+aim']:.1%} ({delta:+.1%})")
            print()
            if delta > 0.10:
                print("VERDICT: aiming is a real ceiling - perfect aim buys a large "
                      "gain over the current policy. Improving the aim head is worth it.")
            elif delta > 0.03:
                print("VERDICT: aiming is a modest ceiling. Some headroom, but not "
                      "where the biggest win is.")
            else:
                print("VERDICT: aiming is NOT the ceiling - perfect aim adds little "
                      "to the current policy. Look at movement, reward, or capacity.")
        return 0
    finally:
        fleet.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
