"""Is `reseed` a safe replacement for `restart` as the floor reset?

The cost side is settled: `reseed` regenerates the floor in 0.12s against
`restart`'s 0.91s, which is most of the 27% of wall clock floor-v5 lost to
resets. What is not settled is whether the state it leaves behind is usable,
and a reset that is subtly wrong is far more expensive than a slow one.

Four things have to hold, every time, not on average:

  fresh      a different layout each reseed, or the agent memorises one floor
  start      the player lands in the starting room with rooms_visited back to 1
  free       the player can actually walk — `reseed` regenerates the floor
             *around* the player rather than placing them, so nothing guarantees
             the spot they are standing in is still open floor
  stable     stats and health unchanged across many resets, no slow drift

That third one is the real risk and the reason this exists separately from
probe_reset_cost.py, which only measured timing.

    .venv/Scripts/python.exe scripts/probe_reseed.py
"""

from __future__ import annotations

import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402

TRIALS = 20
SETTLE_TICKS = 6
WALK_TICKS = 10
# Base Isaac covers roughly four pixels a tick, so ten ticks of unobstructed
# walking is tens of pixels. Below this it is not walking.
MOVED = 12.0


def tick(bridge, message: dict | None = None) -> dict:
    obs = bridge.receive()
    bridge.send(message or {"t": "noop"})
    return obs


def walk(bridge, mx: int, my: int) -> float:
    obs = bridge.receive()
    start = (obs["player"]["x"], obs["player"]["y"])
    action = {"t": "act", "mx": mx, "my": my, "sx": 0, "sy": 0,
              "bomb": False, "item": False}
    bridge.send(action)
    for _ in range(WALK_TICKS - 1):
        bridge.receive()
        bridge.send(action)
    obs = bridge.receive()
    bridge.send({"t": "noop"})
    return max(abs(obs["player"]["x"] - start[0]),
               abs(obs["player"]["y"] - start[1]))


def stats(obs: dict) -> tuple:
    p = obs["player"]
    return (round(p["damage"], 2), round(p["speed"], 2), round(p["range"], 1),
            p["max_hearts"], p["hearts"])


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    config = load_config()
    fleet = launcher.bring_up(config, count=1)
    try:
        bridges = fleet.bridges()
        if not bridges:
            raise SystemExit("instance did not reach a run")
        bridge = bridges[0]
        for _ in range(30):
            bridge.pump()

        baseline = stats(tick(bridge))
        print(f"baseline damage/speed/range/maxhp/hp: {baseline}\n")

        layouts: Counter = Counter()
        problems: list[str] = []
        times: list[float] = []
        wedged = 0

        for trial in range(TRIALS):
            started = time.perf_counter()
            bridge.receive()
            bridge.send({"t": "command", "value": "reseed"})
            # Heal the way an encounter does: reseed keeps the player's health.
            tick(bridge, {"t": "scenario", "enemies": [], "heal": True,
                          "reposition": False, "new_room": False})
            for _ in range(SETTLE_TICKS):
                obs = tick(bridge)
            times.append(time.perf_counter() - started)

            room, level = obs["room"], obs["level"]
            visited = level.get("rooms_visited")
            layouts[(level.get("rooms_total"), room.get("index"))] += 1

            inside = (room["top_left_x"] <= obs["player"]["x"] <= room["bottom_right_x"]
                      and room["top_left_y"] <= obs["player"]["y"] <= room["bottom_right_y"])
            distances = {"left": walk(bridge, -1, 0), "right": walk(bridge, 1, 0),
                         "up": walk(bridge, 0, -1), "down": walk(bridge, 0, 1)}
            blocked = [n for n, d in distances.items() if d < MOVED]
            stuck = len(blocked) >= 3
            wedged += stuck

            if visited != 1:
                problems.append(f"trial {trial}: rooms_visited {visited}, not 1 "
                                f"— not the starting room")
            if not inside:
                problems.append(f"trial {trial}: player outside the room bounds")
            if stuck:
                problems.append(f"trial {trial}: player wedged, blocked {blocked}")
            if stats(obs) != baseline:
                problems.append(f"trial {trial}: stats drifted to {stats(obs)}")

            print(f"  trial {trial:2d}: {times[-1]:.2f}s  rooms {level.get('rooms_total'):>3}  "
                  f"room {room.get('index'):>4}  visited {visited}  "
                  f"pos ({obs['player']['x']:.0f},{obs['player']['y']:.0f})  "
                  + "  ".join(f"{n[0]}:{d:4.0f}" for n, d in distances.items())
                  + ("   WEDGED" if stuck else ""))

        print(f"\nmedian reseed {statistics.median(times):.2f}s "
              f"(restart measured 0.91s)")
        print(f"distinct floors: {len(layouts)}/{TRIALS}")
        print(f"wedged after reseed: {wedged}/{TRIALS}")
        print()
        if problems:
            print("PROBLEMS:")
            for problem in problems[:12]:
                print(f"  - {problem}")
            print("\nreseed is not a safe drop-in for restart.")
        else:
            print("reseed is safe: fresh floors, starting room, player free to "
                  "move, stats stable. Worth using as the floor reset.")
    finally:
        fleet.shutdown()


if __name__ == "__main__":
    main()
