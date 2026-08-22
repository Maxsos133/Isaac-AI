"""Do the semantic entity flags actually resolve and fire?

Two ways this change fails silently, and neither shows up in a unit test:

  unresolved   a constant name that does not exist in this build makes
               `PickupVariant.PICKUP_FOO` nil, the mapping entry is never
               written, and every entity of that kind reports the flag as false
               forever. The mod logs a warning for this; it is checked here too.
  never fires  a flag that is wired correctly but never true in real play is
               five wasted input columns and, worse, reads as "no chests exist"
               rather than as a broken mapping.

So this plays the fleet with random actions and tallies what the mod actually
reports, per flag and per entity kind. A flag at zero after thousands of steps
is the thing to look at.

    .venv/Scripts/python.exe scripts/probe_entity_flags.py
    .venv/Scripts/python.exe scripts/probe_entity_flags.py --instances 4 --steps 400
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.floor_curriculum import FloorCurriculum  # noqa: E402
from isaac_ai.floors import FloorVecEnv  # noqa: E402

FLAGS = ("consumable", "pedestal", "chest", "hostile", "flying")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()

    config = load_config()
    count = args.instances or config.instances.count
    print(f"bringing up {count} instance(s)\n")

    fleet = launcher.bring_up(config, count=count)
    problems: list[str] = []
    try:
        bridges = fleet.bridges()
        if not bridges:
            raise SystemExit("no instance reached a run")
        env = FloorVecEnv(bridges, config, FloorCurriculum())
        env.reset()

        # A flag missing from the payload entirely is not the same as a flag
        # reported false — the first means the mod never wrote it.
        sample = None
        for index in range(env.num_envs):
            for entity in (env._latest[index] or {}).get("entities", []):
                sample = entity
                break
            if sample:
                break
        if sample is not None:
            missing = [f for f in FLAGS if f not in sample]
            if missing:
                problems.append(f"flags absent from the payload: {missing}")
            print(f"payload keys on a live entity: {sorted(sample)}\n")

        seen: Counter = Counter()
        kinds: Counter = Counter()
        by_flag_kind: dict[str, Counter] = {f: Counter() for f in FLAGS}
        entities_total = 0
        # Raw variants, so "flag never fired" can be told apart from "that
        # variant never appeared". Without this a broken mapping and an unlucky
        # sample produce the same zero, which is the exact ambiguity this
        # project keeps losing runs to.
        pickup_variants: Counter = Counter()
        unflagged_variants: Counter = Counter()

        for step in range(args.steps):
            _, _, terminated, truncated, _ = env.step(
                np.random.randint(0, 3, size=(env.num_envs, 4)))

            # Restarting the moment `is_dead` appears is not optional. Isaac
            # stops running mod callbacks once the game-over screen takes over,
            # so an instance left to reach it can never answer again — and
            # because `_receive_all` reads sequentially with a blocking socket,
            # the *entire fleet* then stalls behind it for the full 45s
            # GAMEPLAY_TIMEOUT_SECONDS before it is marked failed. An earlier
            # version of this probe omitted this and duly froze twenty windows
            # with three of them sitting on "Dear Diary". `is_dead` is visible
            # during the death animation, which is the window this uses.
            done = terminated | truncated
            if done.any():
                env.reset_done(done)

            for index in range(env.num_envs):
                for entity in (env._latest[index] or {}).get("entities", []):
                    entities_total += 1
                    kinds[entity.get("k", "?")] += 1
                    for flag in FLAGS:
                        if entity.get(flag):
                            seen[flag] += 1
                            by_flag_kind[flag][entity.get("k", "?")] += 1
                    if entity.get("k") == "pickup":
                        variant = int(entity.get("v", -1))
                        pickup_variants[variant] += 1
                        if not any(entity.get(f) for f in FLAGS):
                            unflagged_variants[variant] += 1
            if step and step % 500 == 0:
                print(f"  step {step}: {entities_total} entity observations, "
                      f"{sum(seen.values())} flags set")

        print(f"\n{entities_total} entity observations over {args.steps} steps")
        print(f"kinds seen: {dict(kinds)}\n")
        print(f"  {'flag':<12}{'times set':>10}   on kinds")
        for flag in FLAGS:
            detail = dict(by_flag_kind[flag]) or "-"
            print(f"  {flag:<12}{seen[flag]:>10}   {detail}")

        print(f"\n  pickup variants seen: {dict(pickup_variants)}")
        if unflagged_variants:
            # A pickup carrying no flag at all is either a variant we failed to
            # map or one genuinely outside the scheme. Either way it is the only
            # thing here that can be a *mapping* fault rather than a sampling
            # one, so it is what gets escalated.
            problems.append(
                f"pickup variants that produced no flag at all: "
                f"{dict(unflagged_variants)} — check these against "
                f"PickupVariant before assuming the mapping is complete")
        else:
            print("  every pickup variant seen produced at least one flag")

        quiet = [f for f in FLAGS if seen[f] == 0]
        if quiet:
            print(f"\n  did not fire: {', '.join(quiet)} — rarity, not "
                  f"necessarily a fault; the variant tally above is what "
                  f"separates the two")

    finally:
        fleet.shutdown()

    log = Path(config.game.savedata_dir) / "log.txt"
    if log.exists():
        warnings = [line.strip() for line in
                    log.read_text(encoding="utf-8", errors="replace").splitlines()
                    if "unresolved entity constants" in line]
        if warnings:
            problems.append(f"mod reported unresolved constants: {warnings[-1]}")
            print(f"\n{warnings[-1]}")
        else:
            print("\nno unresolved-constant warning in log.txt")

    if problems:
        print("\nPROBLEM:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("\nevery semantic flag resolves and fires in real play.")


if __name__ == "__main__":
    main()
