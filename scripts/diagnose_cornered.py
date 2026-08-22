"""Do obstacles cause the deaths that `diagnose_death_cause.py` credits to enemies?

That probe measured what dealt the fatal damage: enemies 71.4%, projectiles
20.0%, environmental 5.7%. It was then summarised as "enemies kill it, the room
does not" — which is a stronger claim than the measurement supports. Damage
source is not cause. An agent pinned against a rock while trying to back away
from a gaper dies to the gaper, and the rock is why.

That distinction decides the next run. If retreat is being blocked, an isolated
combat task in an empty room teaches a tactic that fails exactly where it
matters. If it is not, the agent dies in the open to enemies it cannot beat and
the empty room is a fair simplification.

Two facts make this worth measuring rather than arguing. The shoot-axis probe
shows the agent's dominant movement response to an enemy is to *increase*
distance -- `aligns_y` 0/32 and 11/32 on floor-v20, far below the 50% chance
floor -- so it is already trying to flee. And `blocking -> irrelevant` has never
exceeded 0.029 in any run, so it does not represent where obstacles are and
could not plan a retreat around one even in principle.

Measured at each damaging hit and each death, against a baseline sampled from
undamaged steps on the same fleet:

  escape routes      how many of the 8 neighbouring tiles are free
  blocked retreat    asked to move away from the nearest enemy and did not
                     travel -- the mechanism itself, stated directly
  local density      solid/pit tiles in the 3x3 and 5x5 around the player
  recent blocking    share of the last RECENT move requests that failed

Reported as enrichment against the baseline, the way `diagnose_camping.py`
settled the shaping-into-walls question (38.4% of blocked steps had something
solid on the line against a 21.5% baseline -- 1.79x). An enrichment near 1.0
means obstacles are not implicated in deaths.

The window comes from `encode_egocentric_grid`, the same function the policy
reads, rather than a second copy of the tile logic -- a probe that restates the
code answers a question about the probe.

    .venv/Scripts/python.exe scripts/diagnose_cornered.py
    .venv/Scripts/python.exe scripts/diagnose_cornered.py \
        --policy runs/floor-v19b/policy.pt --steps 3000
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.env import (EGO_RADIUS, EGO_SIZE, GRID_CLASSES,  # noqa: E402
                          decode_action, encode_egocentric_grid)
from isaac_ai.floor_curriculum import FloorCurriculum  # noqa: E402
from isaac_ai.floors import MOVED_UNITS, FloorVecEnv  # noqa: E402
from isaac_ai.policy import ActorCritic, to_tensors  # noqa: E402

# Plane 0 is solid, plane 2 is pit. A tile that is either is one the agent
# cannot retreat through; hazard (plane 1) is deliberately excluded, since it is
# crossable and the question here is about being physically stopped.
PLANE_SOLID, PLANE_PIT = 0, 2
# How many recent move requests "recent blocking" looks back over. About a
# second of game time at action_repeat 2.
RECENT = 10


def free_neighbours(window: np.ndarray) -> int:
    """How many of the 8 tiles around the player are passable."""
    grid = window.reshape(GRID_CLASSES, EGO_SIZE, EGO_SIZE)
    blocked = (grid[PLANE_SOLID] > 0) | (grid[PLANE_PIT] > 0)
    free = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if not blocked[EGO_RADIUS + dr, EGO_RADIUS + dc]:
                free += 1
    return free


def density(window: np.ndarray, radius: int) -> int:
    grid = window.reshape(GRID_CLASSES, EGO_SIZE, EGO_SIZE)
    blocked = (grid[PLANE_SOLID] > 0) | (grid[PLANE_PIT] > 0)
    lo, hi = EGO_RADIUS - radius, EGO_RADIUS + radius + 1
    return int(blocked[lo:hi, lo:hi].sum())


def nearest_enemy(obs: dict) -> tuple[float, float] | None:
    best, best_d = None, float("inf")
    for entity in obs.get("entities", []):
        if entity.get("k") != "enemy":
            continue
        d = float(entity.get("d") or 0.0)
        if d < best_d:
            best, best_d = entity, d
    if best is None:
        return None
    return float(best["x"]), float(best["y"])


class Bucket:
    """Running stats for one population (death, damaged, or baseline)."""

    def __init__(self) -> None:
        self.n = 0
        self.free = 0
        self.d3 = 0
        self.d5 = 0
        self.retreat_attempts = 0
        self.retreat_blocked = 0
        self.recent_blocked = 0.0

    def add(self, free: int, d3: int, d5: int,
            retreating: bool, blocked: bool, recent: float) -> None:
        self.n += 1
        self.free += free
        self.d3 += d3
        self.d5 += d5
        self.recent_blocked += recent
        if retreating:
            self.retreat_attempts += 1
            if blocked:
                self.retreat_blocked += 1

    def row(self, name: str) -> str:
        if self.n == 0:
            return f"  {name:<16}{'no samples':>10}"
        rate = (self.retreat_blocked / self.retreat_attempts
                if self.retreat_attempts else float("nan"))
        return (f"  {name:<16}{self.n:>8}{self.free / self.n:>10.2f}"
                f"{self.d3 / self.n:>9.2f}{self.d5 / self.n:>9.2f}"
                f"{rate:>13.1%}{self.recent_blocked / self.n:>10.1%}")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--policy", default="runs/floor-v19b/policy.pt")
    args = parser.parse_args()

    config = load_config()
    count = args.instances or config.instances.count

    checkpoint = torch.load(args.policy, map_location="cpu")
    policy = ActorCritic()
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    print(f"policy {args.policy} ({checkpoint.get('global_step', 0):,} steps)")
    print(f"bringing up {count} instance(s)\n")

    fleet = launcher.bring_up(config, count=count)
    problems: list[str] = []
    try:
        bridges = fleet.bridges()
        if not bridges:
            raise SystemExit("no instance reached a run")
        env = FloorVecEnv(bridges, config, FloorCurriculum())
        observation = env.reset()

        at_death, at_damage, baseline = Bucket(), Bucket(), Bucket()
        recent_blocks = [deque(maxlen=RECENT) for _ in range(env.num_envs)]
        previous_pos: list[tuple[float, float] | None] = [None] * env.num_envs
        previous_intent: list[tuple[int, int]] = [(0, 0)] * env.num_envs
        previous_enemy: list[tuple[float, float] | None] = [None] * env.num_envs
        was_dead = [False] * env.num_envs
        sealed = 0

        for step in range(args.steps):
            with torch.no_grad():
                actions, _, _ = policy.act(
                    to_tensors(observation, torch.device("cpu")))
            actions = actions.numpy()

            for index in range(env.num_envs):
                latest = env._latest[index] or {}
                player = latest.get("player") or {}
                previous_pos[index] = (float(player.get("x", 0.0)),
                                       float(player.get("y", 0.0)))
                previous_enemy[index] = nearest_enemy(latest)
                mx, my, _, _ = decode_action(actions[index])
                previous_intent[index] = (mx, my)

            observation, _, terminated, truncated, _ = env.step(actions)

            for index in range(env.num_envs):
                latest = env._latest[index] or {}
                player = latest.get("player") or {}
                room = latest.get("room") or {}
                events = latest.get("events") or {}
                if not latest.get("ready", True) or not room:
                    continue

                px, py = float(player.get("x", 0.0)), float(player.get("y", 0.0))
                before = previous_pos[index]
                mx, my = previous_intent[index]

                # Same test the environment charges blocked_move on: asked to
                # move, and the position did not change by a meaningful amount.
                blocked = False
                if (mx or my) and before is not None:
                    travelled = math.dist((px, py), before)
                    blocked = travelled < MOVED_UNITS
                    recent_blocks[index].append(1.0 if blocked else 0.0)

                # Retreating: the requested direction points away from where the
                # nearest enemy was when the action was chosen. Dot product > 0
                # against the enemy-to-player vector.
                retreating = False
                enemy = previous_enemy[index]
                if enemy is not None and (mx or my) and before is not None:
                    away = (before[0] - enemy[0], before[1] - enemy[1])
                    if away[0] or away[1]:
                        retreating = (mx * away[0] + my * away[1]) > 0

                window = encode_egocentric_grid(
                    room, px, py, player.get("grid_index"))
                if not window.any():
                    # An unreadable payload encodes as all-zero, which would read
                    # as "wide open" and bias every bucket towards safety.
                    sealed += 1
                    continue

                free = free_neighbours(window)
                d3, d5 = density(window, 1), density(window, 2)
                recent = (sum(recent_blocks[index]) / len(recent_blocks[index])
                          if recent_blocks[index] else 0.0)
                sample = (free, d3, d5, retreating, blocked, recent)

                hurt = float(events.get("damage_taken") or 0.0) > 0.0
                dead = bool(player.get("is_dead"))
                if dead and not was_dead[index]:
                    at_death.add(*sample)
                elif hurt:
                    at_damage.add(*sample)
                else:
                    baseline.add(*sample)
                was_dead[index] = dead

            done = terminated | truncated
            if done.any():
                env.reset_done(done)
                observation = env._stack_observations()

            if step and step % 500 == 0:
                print(f"  step {step}: {at_death.n} deaths, "
                      f"{at_damage.n} damaged steps, {baseline.n} baseline")

        print(f"\n{at_death.n} deaths / {at_damage.n} damaged steps / "
              f"{baseline.n} baseline steps"
              + (f"  ({sealed} unreadable, skipped)" if sealed else ""))
        print(f"\n  {'population':<16}{'n':>8}{'free nbrs':>10}"
              f"{'3x3':>9}{'5x5':>9}{'blkd retreat':>13}{'recent':>10}")
        print(baseline.row("baseline"))
        print(at_damage.row("taking damage"))
        print(at_death.row("at death"))

        if baseline.n == 0:
            problems.append("no baseline samples — nothing to compare against")
        elif at_death.n == 0:
            problems.append("no deaths recorded — raise --steps")
        else:
            base_free = baseline.free / baseline.n
            death_free = at_death.free / at_death.n
            print(f"\n  escape routes at death vs baseline: "
                  f"{death_free:.2f} vs {base_free:.2f} "
                  f"({death_free / base_free:.2f}x)")
            if baseline.retreat_attempts and at_death.retreat_attempts:
                b = baseline.retreat_blocked / baseline.retreat_attempts
                d = at_death.retreat_blocked / at_death.retreat_attempts
                print(f"  blocked retreat at death vs baseline: "
                      f"{d:.1%} vs {b:.1%} "
                      f"({d / b:.2f}x)" if b else "")
            else:
                problems.append(
                    "no retreat attempts in one of the populations — the "
                    "blocked-retreat rate is undefined, not zero")
            print("\n  reading: enrichment near 1.0 means obstacles are not "
                  "implicated in\n  deaths and an empty-room combat task is a "
                  "fair simplification. Well\n  above 1.0 means the agent is "
                  "being cornered, and combat has to be\n  learned somewhere "
                  "that has walls in it.")

    finally:
        fleet.shutdown()

    if problems:
        print("\nPROBLEM:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
