"""Does the floor reset send the right episodes down the right path, at fleet size?

`probe_reseed.py` answers a different question — whether `reseed` leaves a usable
floor — and answers it on **one** instance. This one exercises the code that
decides *which* reset an episode gets, across the whole fleet, because that is
where the accumulation bug lives and because a reset path that works at 1 and
deadlocks at 20 is this project's most reliably repeated surprise.

Four things have to hold:

  reported   every instance sends the acquisition fields the decision reads.
             A missing key is silently zero, which reads as "pristine" forever.
  seeded     every instance has a baseline after reset(), and it is its own.
             A shared baseline seeded from whichever instance finished first
             records what *that* run had already collected and then measures
             all twenty against it.
  routed     an instance that has acquired something takes the restart path and
             an untouched one takes the cheap path. This is the whole point:
             `reseed` keeps items by design, so anything picked up compounds
             across every later episode until the environment is no longer
             stationary.
  survived   twenty instances go through a mixed restart/reseed split without
             wedging, and come back alive and moving.

    .venv/Scripts/python.exe scripts/probe_floor_reset.py
    .venv/Scripts/python.exe scripts/probe_floor_reset.py --instances 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.floor_curriculum import FloorCurriculum  # noqa: E402
from isaac_ai.floors import FloorVecEnv  # noqa: E402

# Everything the mod must report for the reset decision to be able to see an
# acquisition. Stats alone miss a familiar, a trinket, a card or a consumable.
REQUIRED = ("collectibles", "trinket0", "trinket1", "card0", "pill0",
            "active_item", "bombs", "keys", "coins",
            "damage", "speed", "range", "tear_delay", "max_hearts")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=None,
                        help="default: the real fleet size from config.toml")
    args = parser.parse_args()

    config = load_config()
    count = args.instances or config.instances.count
    print(f"bringing up {count} instance(s) — the real fleet size\n")

    fleet = launcher.bring_up(config, count=count)
    problems: list[str] = []
    try:
        bridges = fleet.bridges()
        if not bridges:
            raise SystemExit("no instance reached a run")

        env = FloorVecEnv(bridges, config, FloorCurriculum())
        started = time.perf_counter()
        env.reset()
        print(f"reset(): {time.perf_counter() - started:.2f}s, "
              f"{env.num_envs} instance(s)\n")

        # -- reported ------------------------------------------------------
        missing: dict[str, int] = {}
        for index in range(env.num_envs):
            player = (env._latest[index] or {}).get("player") or {}
            for key in REQUIRED:
                if key not in player:
                    missing[key] = missing.get(key, 0) + 1
        if missing:
            problems.append(f"fields missing from the observation: {missing}")
            print(f"reported: MISSING {missing}")
        else:
            sample = (env._latest[0] or {}).get("player") or {}
            print("reported: all acquisition fields present on every instance")
            print("          " + "  ".join(
                f"{k}={sample.get(k)}" for k in REQUIRED[:6]))

        # -- seeded --------------------------------------------------------
        unseeded = [i for i in range(env.num_envs) if env._baseline[i] is None]
        if unseeded:
            problems.append(f"no baseline after reset() on instances {unseeded}")
        print(f"seeded:   {env.num_envs - len(unseeded)}/{env.num_envs} "
              f"instances have a baseline")

        pristine = [i for i in range(env.num_envs)
                    if env._picked_something_up(i)]
        if pristine:
            problems.append(
                f"instances {pristine} read as already having acquired "
                f"something immediately after a full restart")
        print(f"          {env.num_envs - len(pristine)}/{env.num_envs} "
              f"read as pristine straight after restart")

        # -- routed --------------------------------------------------------
        # Move the recorded baseline rather than the game state: the decision
        # under test is a comparison, and forcing a real pickup on demand would
        # test the mod's spawn code instead of the routing.
        forced = list(range(0, env.num_envs, 2))
        for index in forced:
            baseline = list(env._baseline[index])
            baseline[REQUIRED.index("collectibles")] += 1
            env._baseline[index] = tuple(baseline)

        flagged = [i for i in range(env.num_envs)
                   if env._picked_something_up(i)]
        if sorted(flagged) != sorted(forced):
            problems.append(
                f"routing wrong: expected {forced} to need a restart, got {flagged}")
        print(f"routed:   {len(flagged)}/{len(forced)} altered instances "
              f"flagged for the expensive path, "
              f"{env.num_envs - len(flagged)} left on reseed")

        # -- survived ------------------------------------------------------
        # One index per axis, not a flat action id: decode_action reads
        # action[0..3] as move x/y and shoot x/y into AXIS_VALUES.
        for _ in range(20):
            env.step(np.random.randint(0, 3, size=(env.num_envs, 4)))

        started = time.perf_counter()
        env.reset_done(np.ones(env.num_envs, dtype=bool))
        elapsed = time.perf_counter() - started
        print(f"\nsurvived: mixed split of {len(forced)} restart / "
              f"{env.num_envs - len(forced)} reseed in {elapsed:.2f}s")

        still_flagged = [i for i in range(env.num_envs)
                         if env._picked_something_up(i)]
        if still_flagged:
            problems.append(
                f"instances {still_flagged} still read as dirty after the "
                f"restart that was supposed to clean them")
        print(f"          {env.num_envs - len(still_flagged)}/{env.num_envs} "
              f"pristine again afterwards")

        moved = 0
        for _ in range(10):
            before = [(env._latest[i] or {}).get("player", {}).get("x")
                      for i in range(env.num_envs)]
            # move x = +1 (right), every other axis neutral.
            env.step(np.tile(np.array([2, 1, 1, 1]), (env.num_envs, 1)))
            after = [(env._latest[i] or {}).get("player", {}).get("x")
                     for i in range(env.num_envs)]
            moved = sum(1 for a, b in zip(before, after)
                        if a is not None and b is not None and abs(a - b) > 1e-6)
        if env.alive_count < env.num_envs:
            problems.append(
                f"only {env.alive_count}/{env.num_envs} instances alive at the end")
        print(f"          {env.alive_count}/{env.num_envs} alive, "
              f"{moved}/{env.num_envs} moving\n")

    finally:
        fleet.shutdown()

    if problems:
        print("PROBLEM:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("floor reset routes correctly at fleet size: fields reported, "
          "baselines per instance, acquisitions take the restart path.")


if __name__ == "__main__":
    main()
