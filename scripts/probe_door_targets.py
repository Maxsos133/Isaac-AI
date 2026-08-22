"""Which doors does the shaping chase that the agent can never walk through?

`door_potential` skips a door only when `locked` is set. That is not the same
question as "can I pass". A secret room's door is closed and **unlocked**, and
opening it needs a bomb — which the floor agent has no action for at all, since
`floors.py` hardcodes `bomb: False` in every message it sends. So the potential
happily targets a wall, the agent walks into it with total confidence, and the
attempt burns its idle limit and reseeds.

This tallies every door the fleet sees by target room type and by whether it was
*ever* observed open. A type that is never open, never locked, and still reaches
the potential is the bug.

    .venv/Scripts/python.exe scripts/probe_door_targets.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.floor_curriculum import FloorCurriculum  # noqa: E402
from isaac_ai.floors import FloorVecEnv, door_is_targetable  # noqa: E402

# Isaac's RoomType values, for reading the tally. Only the ones that matter here.
ROOM_TYPES = {1: "default", 2: "shop", 3: "error", 4: "treasure", 5: "boss",
              6: "miniboss", 7: "secret", 8: "supersecret", 9: "arcade",
              10: "curse", 11: "challenge", 12: "library", 13: "sacrifice",
              14: "devil", 15: "angel", 16: "dungeon", 17: "bossrush",
              18: "isaacs", 19: "barren", 20: "chest", 21: "dice",
              22: "blackmarket", 27: "ultrasecret"}


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--steps", type=int, default=900)
    args = parser.parse_args()

    config = load_config()
    count = args.instances or config.instances.count
    print(f"bringing up {count} instance(s)\n")
    fleet = launcher.bring_up(config, count=count)
    try:
        bridges = fleet.bridges()
        if not bridges:
            raise SystemExit("no instance reached a run")
        env = FloorVecEnv(bridges, config, FloorCurriculum())
        env.reset()

        seen: Counter = Counter()
        ever_open: dict = defaultdict(bool)
        ever_locked: dict = defaultdict(bool)
        targeted: Counter = Counter()
        needs_bomb: Counter = Counter()
        categories: dict = defaultdict(Counter)

        for _ in range(args.steps):
            _, _, terminated, truncated, _ = env.step(
                np.random.randint(0, 3, size=(env.num_envs, 4)))
            done = terminated | truncated
            if done.any():
                env.reset_done(done)

            for index in range(env.num_envs):
                latest = env._latest[index]
                if not latest or not latest.get("ready", True):
                    continue
                room = latest.get("room") or {}
                player = latest.get("player") or {}
                if not room or not player:
                    continue

                # Which door the potential is currently pulling towards. The
                # filter is imported, never restated: an earlier version of this
                # probe kept its own copy, and after the secret-room fix landed
                # it went on reporting the bug as present because only its copy
                # was stale.
                best, best_door = None, None
                for door in room.get("doors", []):
                    kind = int(door.get("target_type", -1))
                    seen[kind] += 1
                    categories[kind][door.get("category", "?")] += 1
                    if door.get("open"):
                        ever_open[kind] = True
                    if door.get("locked"):
                        ever_locked[kind] = True
                    if door.get("needs_bomb"):
                        needs_bomb[kind] += 1
                    if not door_is_targetable(door):
                        continue
                    distance = ((float(door["x"]) - float(player["x"])) ** 2
                                + (float(door["y"]) - float(player["y"])) ** 2)
                    if best is None or distance < best:
                        best, best_door = distance, door
                if best_door is not None:
                    targeted[int(best_door.get("target_type", -1))] += 1

        print(f"\n  {'target room':<14}{'seen':>8}{'ever open':>11}"
              f"{'ever locked':>13}{'needs bomb':>12}{'shaping target':>16}"
              f"   category")
        unreachable = []
        for kind, count in seen.most_common():
            name = ROOM_TYPES.get(kind, f"type {kind}")
            cats = ",".join(sorted(categories[kind]))
            print(f"  {name:<14}{count:>8}{str(ever_open[kind]):>11}"
                  f"{str(ever_locked[kind]):>13}{needs_bomb[kind]:>12}"
                  f"{targeted[kind]:>16}   {cats}")
            if not ever_open[kind] and not ever_locked[kind] and targeted[kind]:
                unreachable.append((name, targeted[kind]))

        print(f"\n  potential had a target on "
              f"{sum(targeted.values())} observations")
        if unreachable:
            print("\nPROBLEM: the shaping targeted doors that were never open "
                  "and never locked —\n  the agent cannot pass these and cannot "
                  "bomb, so it walks into a wall until the idle limit:")
            for name, count in unreachable:
                print(f"    {name}: targeted {count} times")
        else:
            print("\n  every door the potential targeted was openable at some point.")
    finally:
        fleet.shutdown()


if __name__ == "__main__":
    main()
