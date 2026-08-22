"""Confirm an encounter locks the room.

If doors stay open the agent can walk out mid-fight, and because enemies are
counted per-room that reads as a cleared encounter — paying the clear bonus for
fleeing. This checks the doors actually shut, and then tries to escape.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402

config = load_config()
fleet = launcher.bring_up(config, count=1)


def settle(bridge, ticks: int, action: dict | None = None) -> dict:
    obs: dict = {}
    for _ in range(ticks):
        obs = bridge.receive()
        bridge.send(action or {"t": "noop"})
    return obs


def doors_of(obs: dict) -> tuple[int, int]:
    doors = obs["room"].get("doors", [])
    return sum(1 for d in doors if d["open"]), len(doors)


try:
    if fleet.ready_count == 0:
        raise SystemExit("instance did not reach a run")
    bridge = fleet.bridges()[0]

    obs = settle(bridge, 20)
    open_before, total = doors_of(obs)
    print(f"\nbefore encounter: {open_before}/{total} doors open, "
          f"room clear={obs['room']['clear']}")

    bridge.receive()
    bridge.send({"t": "scenario",
                 "enemies": [{"type": 10, "variant": 0, "subtype": 0, "count": 3}],
                 "min_distance": 200, "heal": True})
    obs = settle(bridge, 20)
    open_after, total = doors_of(obs)
    print(f"after encounter:  {open_after}/{total} doors open, "
          f"room clear={obs['room']['clear']}, "
          f"enemies={obs['room']['enemies_alive']}")

    start_room = obs["room"]["index"]
    print(f"\ntrying to escape from room {start_room} (holding each direction)...")
    escaped = False
    for name, mx, my in (("right", 1, 0), ("left", -1, 0),
                         ("up", 0, -1), ("down", 0, 1)):
        obs = settle(bridge, 90, {"t": "act", "mx": mx, "my": my,
                                  "sx": 0, "sy": 0, "bomb": False, "item": False})
        if obs["room"]["index"] != start_room:
            print(f"   ESCAPED heading {name} -> room {obs['room']['index']}")
            escaped = True
            break
        print(f"   held {name:5s}: still in room {obs['room']['index']}")

    print()
    if escaped:
        print("PROBLEM: the agent can leave mid-encounter")
    elif open_after == 0:
        print("doors lock during an encounter; the agent is held in the fight")
    else:
        print(f"doors report {open_after} open but escape failed - check manually")
finally:
    fleet.shutdown()
