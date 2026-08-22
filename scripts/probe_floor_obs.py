"""Confirm the floor observation is real, not zeros.

Navigation depends on level and door reporting added to the mod. If any of it
comes back empty the agent is blind to exactly the thing it must learn, and a
training run would look merely "hard" rather than broken.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.env import DOOR_FEATURES, MAX_DOORS, encode_observation  # noqa: E402

config = load_config()
fleet = launcher.bring_up(config, count=1)


def settle(bridge, ticks: int, action: dict | None = None) -> dict:
    obs: dict = {}
    for _ in range(ticks):
        obs = bridge.receive()
        bridge.send(action or {"t": "noop"})
    return obs


try:
    if fleet.ready_count == 0:
        raise SystemExit("instance did not reach a run")
    bridge = fleet.bridges()[0]
    obs = settle(bridge, 30)

    level = obs["level"]
    room = obs["room"]
    print("\n=== level reporting ===")
    print(f"  stage         {level['stage']}")
    print(f"  rooms_total   {level.get('rooms_total')}")
    print(f"  rooms_visited {level.get('rooms_visited')}")
    print(f"  room index    {room['index']}  type {room['type']}  clear {room['clear']}")

    print("\n=== doors in the start room ===")
    for door in room.get("doors", []):
        print(f"  slot {door['slot']}  open {str(door['open']):5s} "
              f"locked {str(door['locked']):5s} target {door['target']:4d} "
              f"visited {door.get('visited')}  category {door.get('category')}")

    encoded = encode_observation(obs)["doors"].reshape(MAX_DOORS, DOOR_FEATURES)
    present = [i for i in range(MAX_DOORS) if encoded[i][0] > 0]
    unvisited = [i for i in present if encoded[i][3] > 0]
    print(f"\n  encoded door slots present: {present}")
    print(f"  of those, unvisited:        {unvisited}")

    print("\n=== walking through a door ===")
    start = room["index"]
    moved = False
    for name, mx, my in (("right", 1, 0), ("left", -1, 0), ("down", 0, 1), ("up", 0, -1)):
        obs = settle(bridge, 120, {"t": "act", "mx": mx, "my": my,
                                   "sx": 0, "sy": 0, "bomb": False, "item": False})
        if obs["room"]["index"] != start:
            print(f"  moved {name} -> room {obs['room']['index']}, "
                  f"type {obs['room']['type']}, clear {obs['room']['clear']}, "
                  f"enemies {obs['room'].get('enemies_alive')}")
            print(f"  rooms_visited now {obs['level'].get('rooms_visited')} "
                  f"(was {level.get('rooms_visited')})")
            moved = True
            break
        print(f"  held {name:5s}: still in room {start}")

    if not moved:
        print("  PROBLEM: never left the start room")

    print("\n=== checking a previously-visited door is marked ===")
    doors = obs["room"].get("doors", [])
    back = [d for d in doors if d.get("visited", 0) > 0]
    print(f"  doors leading to already-visited rooms: {[d['slot'] for d in back]}")
    if back:
        print("  exploration signal works: the way back is distinguishable")
    else:
        print("  PROBLEM: no door reports a visited target")

    # The RoomType lookup is the kind of thing that silently yields "unknown"
    # if the API field is absent, so sweep the floor and check what comes back.
    print("\n=== door categories seen while walking the floor ===")
    seen: dict[str, int] = {}
    rooms: set[int] = set()
    for step in range(14):
        for mx, my in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            obs = settle(bridge, 45, {"t": "act", "mx": mx, "my": my,
                                      "sx": 1, "sy": 0, "bomb": False, "item": False})
            # The player can die out here; a dead run reports no room at all.
            if not obs.get("ready", True) or "room" not in obs:
                bridge.receive()
                bridge.send({"t": "reset"})
                settle(bridge, 120)
                continue
            rooms.add(obs["room"]["index"])
            for door in obs["room"].get("doors", []):
                key = str(door.get("category"))
                seen[key] = seen.get(key, 0) + 1
        print(f"  ...swept {step + 1}/14, {len(rooms)} distinct rooms, "
              f"categories so far: {sorted(seen)}", flush=True)
    for name, count in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"  {name:10s} {count}")
    if seen.get("unknown"):
        print("\n  PROBLEM: some doors report 'unknown' - the RoomType lookup failed")
    else:
        print("\n  every door resolved to a real category")
finally:
    fleet.shutdown()
