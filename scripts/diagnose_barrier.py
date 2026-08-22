"""How much throughput does the lockstep barrier cost?

The vec env sends to every instance, then reads from every instance. Each step
therefore costs the SLOWEST instance, so over a run the cost is a sum of maxima.
Without the barrier it would be a max of sums, which is much smaller when stalls
are bursty and uncorrelated — which room transitions are.

This measures the gap directly: per step, how long the slowest instance took
versus the average one.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from isaac_ai import launcher  # noqa: E402
from isaac_ai.bridge import BridgeError  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.env import ACTION_DIMS  # noqa: E402
from isaac_ai.nav_curriculum import NavigationCurriculum  # noqa: E402
from isaac_ai.navigation import NavigationVecEnv  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--instances", type=int, default=8)
parser.add_argument("--steps", type=int, default=400)
args = parser.parse_args()

config = load_config()
fleet = launcher.bring_up(config, count=args.instances)

try:
    if fleet.ready_count == 0:
        raise SystemExit("no instances reached a run")

    env = NavigationVecEnv(fleet.bridges(), config, NavigationCurriculum())
    env.reset()

    per_instance: list[list[float]] = [[] for _ in range(env.num_envs)]
    step_max: list[float] = []
    step_mean: list[float] = []

    # Patch the receive path to time each instance individually.
    original = env._receive_all

    def timed(expect):
        results = [None] * env.num_envs
        durations = []
        for index in range(env.num_envs):
            if env._failed[index] or not expect[index]:
                continue
            start = time.perf_counter()
            try:
                results[index] = env.bridges[index].receive()
            except BridgeError as exc:
                print(f"instance {index} dropped: {exc}")
                env._failed[index] = True
                continue
            elapsed = time.perf_counter() - start
            per_instance[index].append(elapsed)
            durations.append(elapsed)
        if durations:
            step_max.append(max(durations))
            step_mean.append(sum(durations) / len(durations))
        return results

    env._receive_all = timed  # type: ignore[method-assign]

    rng = np.random.default_rng(5)
    started = time.perf_counter()
    for _ in range(args.steps):
        actions = np.stack([rng.integers(0, d, size=env.num_envs)
                            for d in ACTION_DIMS], axis=1)
        _, _, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated
        if done.any():
            env.reset_done(done)
    elapsed = time.perf_counter() - started

    env._receive_all = original  # type: ignore[method-assign]

    total_max = sum(step_max)
    total_mean = sum(step_mean)
    print(f"\n=== {args.steps} vec steps on {env.num_envs} instances, "
          f"{elapsed:.1f}s wall ===")
    print(f"  waiting on the slowest instance: {total_max:.1f}s")
    print(f"  average instance was ready in:   {total_mean:.1f}s")
    print(f"  barrier overhead:                {total_max - total_mean:.1f}s "
          f"({(total_max - total_mean) / max(total_max, 1e-9):.0%} of read time)")
    print()
    print(f"  median read {statistics.median(step_max) * 1000:6.1f} ms "
          f"| p90 {sorted(step_max)[int(len(step_max) * 0.9)] * 1000:6.1f} ms "
          f"| max {max(step_max) * 1000:7.1f} ms")
    print()
    print(f"  measured {args.steps * env.num_envs / elapsed:.1f} agent steps/s")
    print(f"  ceiling  {env.num_envs * 30 / env.action_repeat:.1f} agent steps/s")
    print(f"  -> running at {(args.steps * env.num_envs / elapsed) / (env.num_envs * 30 / env.action_repeat):.0%} "
          f"of what the 30 Hz tick allows")
finally:
    fleet.shutdown()
