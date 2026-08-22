"""Does a killed instance actually come back?

floor-v27b finished two instances down over 5M steps — the game faults roughly
once an hour across twenty of them, and the bridge fix made that cost one
instance rather than the whole run. `--relaunch-crashed` is meant to close the
rest of the gap, and this is the check that it works before it runs unattended
overnight.

It matters more than a usual probe because the failure it could introduce is
worse than the one it fixes. Run entry needs the foreground, every other
instance is blocked inside its socket read while that happens, and a blocked
Isaac stops pumping Win32 messages — so a mistake here wedges the whole fleet
instead of costing one instance. That is why the flag is off by default and why
this exists.

    kill        one instance's process, the way a crash would
    detect      the env marks it failed on the next step, not before
    relaunch    a new game on the same port, walked back into a run
    restore     the fleet is whole and stepping again

`process.kill()` is not identical to the real fault — an access violation drops
the socket with an RST, which raises `ConnectionResetError`, while a kill closes
it. Both land in the same `_failed` path (that was the v9b bridge fix), so this
exercises the recovery, not the detection.

    .venv/Scripts/python.exe scripts/probe_relaunch.py
    .venv/Scripts/python.exe scripts/probe_relaunch.py --instances 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.floor_curriculum import FloorCurriculum  # noqa: E402
from isaac_ai.floors import FloorVecEnv  # noqa: E402


def step_random(env: FloorVecEnv, steps: int) -> None:
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(
            np.random.randint(0, 3, size=(env.num_envs, 4)))
        done = terminated | truncated
        if done.any():
            env.reset_done(done)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--victim", type=int, default=1,
                        help="which env index to kill")
    args = parser.parse_args()

    config = load_config()
    print(f"bringing up {args.instances} instance(s)\n")
    fleet = launcher.bring_up(config, count=args.instances)
    problems: list[str] = []
    try:
        bridges = fleet.bridges()
        if len(bridges) < 2:
            raise SystemExit("need at least 2 instances to test a relaunch")
        env = FloorVecEnv(bridges, config, FloorCurriculum())
        env.reset()
        started_with = env.alive_count
        print(f"fleet stepping, {started_with} alive")
        step_random(env, 20)

        victim = min(args.victim, env.num_envs - 1)
        instance = next(i for i in fleet.instances
                        if i.index == env.bridges[victim].index)
        print(f"\nkilling env index {victim} "
              f"(fleet instance {instance.index}, pid {instance.process.pid})")
        instance.process.kill()

        # The env only learns about this when a send or receive fails, so step
        # until it does rather than assuming one step is enough.
        for attempt in range(10):
            step_random(env, 1)
            if env.failed_indices():
                print(f"  detected after {attempt + 1} step(s): "
                      f"failed={env.failed_indices()}, alive={env.alive_count}")
                break
        else:
            problems.append("the kill was never detected — the env still "
                            "believes every instance is alive")
            raise SystemExit(1)

        if env.alive_count != started_with - 1:
            problems.append(f"expected {started_with - 1} alive after the kill, "
                            f"got {env.alive_count}")

        print("\nrelaunching...")
        restored = 0
        for index in env.failed_indices():
            if launcher.relaunch(config, fleet, env.bridges[index],
                                 keep_alive=env.healthy_bridges(exclude=index)):
                if env.restore(index):
                    restored += 1

        print(f"\nrestored {restored}, alive {env.alive_count}/{env.num_envs}")
        if env.alive_count != started_with:
            problems.append(f"fleet did not come back: {env.alive_count} alive, "
                            f"expected {started_with}")
        else:
            # Stepping afterwards is the real check. A restored instance that
            # cannot be stepped is worse than one left failed, because `step`
            # will block on a read that never comes and stall all the others.
            print("stepping the whole fleet again...")
            step_random(env, 30)
            print(f"  30 steps clean, alive {env.alive_count}")
            if env.failed_indices():
                problems.append(f"instances dropped again while stepping: "
                                f"{env.failed_indices()}")

    finally:
        fleet.shutdown()

    if problems:
        print("\nPROBLEM:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("\nrelaunch works: a killed instance came back and the fleet stepped on.")


if __name__ == "__main__":
    main()
