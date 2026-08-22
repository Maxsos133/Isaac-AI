"""Confirm navigation episodes start in varied rooms, not the same door pair.

Also a safety check on ChangeRoom: a bad grid index would crash the game rather
than fail politely, so verify it lands somewhere real every time.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.env import ACTION_DIMS  # noqa: E402
from isaac_ai.nav_curriculum import NavigationCurriculum  # noqa: E402
from isaac_ai.navigation import NavigationVecEnv  # noqa: E402

config = load_config()
fleet = launcher.bring_up(config, count=3)

try:
    if fleet.ready_count == 0:
        raise SystemExit("no instances reached a run")

    env = NavigationVecEnv(fleet.bridges(), config, NavigationCurriculum())
    env.reset()

    rooms_at_start: list[int] = []
    episodes = 0
    rng = np.random.default_rng(2)

    while episodes < 45:
        actions = np.stack([rng.integers(0, d, size=env.num_envs)
                            for d in ACTION_DIMS], axis=1)
        _, _, terminated, truncated, infos = env.step(actions)
        episodes += sum(1 for i in infos if "episode" in i)

        done = terminated | truncated
        if done.any():
            env.reset_done(done)
            for index in np.flatnonzero(done):
                latest = env._latest[int(index)]
                if latest and latest.get("room"):
                    rooms_at_start.append(latest["room"]["index"])

    counts = Counter(rooms_at_start)
    print(f"\n{len(rooms_at_start)} episode starts across "
          f"{len(counts)} distinct rooms")
    for room, count in counts.most_common(12):
        print(f"  room {room:4d}: {count}")

    top_share = counts.most_common(1)[0][1] / max(len(rooms_at_start), 1)
    print()
    print(f"  most-used room is {top_share:.0%} of starts")
    if len(counts) >= 4 and top_share < 0.5:
        print("  varied: episodes begin all over the floor")
    else:
        print("  PROBLEM: still concentrated in a couple of rooms")
    print(f"  instances alive: {env.alive_count}/{env.num_envs}")
finally:
    fleet.shutdown()
