"""Verify every enemy in the curriculum roster actually spawns, and that the
mod places them away from the player rather than on top of him.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.curriculum import ROSTER  # noqa: E402

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
    settle(bridge, 20)

    print(f"\n{'enemy':18s} {'spawned':>8s} {'alive':>6s} {'min dist':>9s}  status")
    print("-" * 60)

    failures = []
    for kind in ROSTER:
        bridge.receive()
        bridge.send({
            "t": "scenario",
            "enemies": [{"type": kind.type_id, "variant": kind.variant,
                         "subtype": 0, "count": 3}],
            "min_distance": 200,
            "heal": True,
        })
        obs = settle(bridge, 25)

        enemies = [e for e in obs.get("entities", []) if e["k"] == "enemy"]
        alive = obs["room"].get("enemies_alive", 0)
        distances = [e["d"] for e in enemies]
        closest = min(distances) if distances else 0.0

        ok = len(enemies) >= 3 and closest >= 150
        status = "ok" if ok else "PROBLEM"
        if not ok:
            failures.append(kind.name)
        print(f"{kind.name:18s} {len(enemies):8d} {alive:6d} "
              f"{closest:9.0f}  {status}")

    print("\nclearing the room and confirming the count drops to zero:")
    bridge.receive()
    bridge.send({"t": "scenario", "enemies": [], "min_distance": 200, "heal": True})
    obs = settle(bridge, 25)
    print(f"   enemies_alive = {obs['room'].get('enemies_alive')} "
          f"entities = {len(obs.get('entities', []))}")
    print(f"   player hearts = {obs['player']['hearts']} (heal should restore)")

    if failures:
        print(f"\nroster entries needing different ids: {failures}")
    else:
        print("\nwhole roster spawns correctly at a safe distance")
finally:
    fleet.shutdown()
