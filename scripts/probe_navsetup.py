"""Verify the navigation setup actually produces a pure-navigation room.

If doors stay locked or enemies survive, the task is not navigation and a whole
run would be spent measuring the wrong thing.
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


def describe(obs: dict) -> str:
    room = obs["room"]
    doors = room.get("doors", [])
    return (f"clear={room['clear']} enemies={room.get('enemies_alive')} "
            f"doors={len(doors)} open={sum(1 for d in doors if d['open'])} "
            f"locked={sum(1 for d in doors if d['locked'])} "
            f"pos=({obs['player']['x']:.0f},{obs['player']['y']:.0f}) "
            f"hearts={obs['player']['hearts']}")


try:
    if fleet.ready_count == 0:
        raise SystemExit("instance did not reach a run")
    bridge = fleet.bridges()[0]

    obs = settle(bridge, 25)
    print(f"\nstart room:      {describe(obs)}")

    # Walk into a neighbouring room so we are somewhere with real contents.
    for mx, my in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        moved = settle(bridge, 90, {"t": "act", "mx": mx, "my": my,
                                    "sx": 0, "sy": 0, "bomb": False, "item": False})
        if moved["room"]["index"] != obs["room"]["index"]:
            obs = moved
            break
    print(f"after moving:    {describe(obs)}")

    positions = []
    for attempt in range(3):
        bridge.receive()
        bridge.send({"t": "navsetup", "reposition": True})
        obs = settle(bridge, 12)
        positions.append((round(obs["player"]["x"]), round(obs["player"]["y"])))
        print(f"after navsetup {attempt + 1}: {describe(obs)}")

    doors = obs["room"].get("doors", [])
    problems = []
    if obs["room"].get("enemies_alive", 0) != 0:
        problems.append("enemies remain")
    if not obs["room"]["clear"]:
        problems.append("room not marked clear")
    if any(not d["open"] for d in doors):
        problems.append("a door is still shut")
    if any(d["locked"] for d in doors):
        problems.append("a door is still locked")
    if len(set(positions)) < 2:
        problems.append(f"player not repositioned (same spot {positions})")

    print()
    if problems:
        print("PROBLEMS: " + "; ".join(problems))
    else:
        print("navsetup gives a clean navigation room: "
              "no enemies, all doors open, player moved each time")
        print(f"  spawn positions across resets: {positions}")
finally:
    fleet.shutdown()
