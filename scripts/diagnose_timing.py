"""Separate steady-state stepping cost from episode-reset cost.

Phase 0 sustained a full 30 ticks/s per instance, so any shortfall now is
overhead we added. This tells us which half to look at.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.env import ACTION_DIMS, IsaacVecEnv  # noqa: E402

INSTANCES = 3
STEPS = 120

config = load_config()
fleet = launcher.bring_up(config, count=INSTANCES)
try:
    if fleet.ready_count == 0:
        raise SystemExit("no instances reached a run")

    env = IsaacVecEnv(fleet.bridges(), config)

    print("\ntiming a full reset...")
    t0 = time.perf_counter()
    env.reset()
    reset_seconds = time.perf_counter() - t0
    print(f"  reset took {reset_seconds:.2f}s for {env.num_envs} instance(s)")

    print(f"\ntiming {STEPS} steady-state steps (no resets)...")
    rng = np.random.default_rng(0)
    durations = []
    send_durations = []
    recv_durations = []

    for _ in range(STEPS):
        actions = np.stack([rng.integers(0, d, size=env.num_envs)
                            for d in ACTION_DIMS], axis=1)

        # Time the two halves of one action-repeat round separately.
        messages = [{"t": "act", "mx": 0, "my": 1, "sx": 0, "sy": 0,
                     "bomb": False, "item": False} for _ in range(env.num_envs)]
        t_send = time.perf_counter()
        env._send_all(messages)
        send_durations.append(time.perf_counter() - t_send)

        t_recv = time.perf_counter()
        env._receive_all([True] * env.num_envs)
        recv_durations.append(time.perf_counter() - t_recv)

        t0 = time.perf_counter()
        env.step(actions)
        durations.append(time.perf_counter() - t0)

    def report(name: str, values: list[float]) -> None:
        print(f"  {name:22s} mean {statistics.mean(values) * 1000:7.2f} ms | "
              f"median {statistics.median(values) * 1000:7.2f} ms | "
              f"max {max(values) * 1000:7.2f} ms")

    report("send_all (3 inst)", send_durations)
    report("receive_all (3 inst)", recv_durations)
    report("full env.step", durations)

    ticks = STEPS * env.num_envs * env.action_repeat
    total = sum(durations)
    print(f"\n  steady state: {ticks / total:.1f} game ticks/s "
          f"across {env.num_envs} instances "
          f"({ticks / total / env.num_envs:.1f} per instance, 30 is the cap)")
finally:
    fleet.shutdown()
