"""Verify the room-category mapping against every special room type.

A random walk mostly meets normal doors, so the boss / treasure / shop / curse
branches would go untested. The console can teleport straight into each special
room, and the current room is categorised through the same table the doors use —
so this checks the exact mapping the agent depends on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402

# console target -> the category the mod should report for that room.
# Several spellings per entry: a `goto` that the console does not recognise
# fails silently and leaves the player where they were, so an unverified probe
# would read the previous room and report a mapping bug that does not exist.
EXPECTED = [
    (["s.treasure"], "treasure"),
    (["s.shop"], "shop"),
    (["s.boss", "s.bossroom", "d.boss"], "boss"),
    (["s.curse"], "curse"),
    (["s.secret"], "other"),
    (["s.miniboss", "s.minboss"], "boss"),
    (["s.devil"], "other"),
    (["s.angel"], "other"),
    (["s.arcade"], "other"),
    (["s.library"], "other"),
]

config = load_config()
fleet = launcher.bring_up(config, count=1)


def settle(bridge, ticks: int) -> dict:
    obs: dict = {}
    for _ in range(ticks):
        obs = bridge.receive()
        bridge.send({"t": "noop"})
    return obs


try:
    if fleet.ready_count == 0:
        raise SystemExit("instance did not reach a run")
    bridge = fleet.bridges()[0]
    obs = settle(bridge, 25)
    print(f"\nstart room category: {obs['room'].get('category')} "
          f"(type {obs['room']['type']})")

    print(f"\n{'goto':22s} {'room type':>9s} {'reported':>10s} {'expected':>10s}  result")
    print("-" * 70)

    failures = []
    untested = []

    for targets, expected in EXPECTED:
        before = obs["room"]["index"] if "room" in obs else -1
        arrived = None

        for target in targets:
            bridge.receive()
            bridge.send({"t": "command", "value": f"goto {target}"})
            obs = settle(bridge, 45)
            if not obs.get("ready", True) or "room" not in obs:
                continue
            # The room index changing is the proof the teleport happened.
            if obs["room"]["index"] != before:
                arrived = target
                break

        label = "/".join(targets)
        if arrived is None:
            untested.append(label)
            print(f"{label:22s} {'-':>9s} {'-':>10s} {expected:>10s}  "
                  f"goto did not move us")
            continue

        reported = str(obs["room"].get("category"))
        room_type = obs["room"]["type"]
        ok = reported == expected
        if not ok:
            failures.append((arrived, room_type, reported, expected))
        print(f"{arrived:22s} {room_type:9d} {reported:>10s} {expected:>10s}  "
              f"{'ok' if ok else 'MISMATCH'}")

    print()
    if failures:
        print("mapping is wrong for:")
        for target, room_type, reported, expected in failures:
            print(f"  {target} (room type {room_type}): "
                  f"reported {reported!r}, expected {expected!r}")
    else:
        print("every room type we could reach maps to the expected category")
    if untested:
        print(f"not reachable on this floor, so untested: {', '.join(untested)}")
finally:
    fleet.shutdown()
