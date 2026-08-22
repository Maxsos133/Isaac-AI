"""After the teleport, can Isaac actually move?

Combat encounters now jump to a random plain room and teleport the player to a
random free spot in it. `probe_combat_rooms.py` checked that the room changes,
that enemies spawn, and that stats stay frozen — it never checked the one thing
that matters most: whether the player can still walk afterwards.

Rooms differ in their rock, poop and pit layout, `GetFreeNearPosition` only
searches 40 units from the requested point, and a player wedged behind a rock
loses every encounter without any of that showing up as a bug. The timing is
suggestive: combat-v6 had the teleport but not the room jump and reached
difficulty 0.73; every run after the room jump was added settled at 0.33-0.39.

So: teleport repeatedly, and after each one try to walk in all four directions
and see whether the position actually changes.

**A clean result here does not clear the teleport.** A floor has more rooms than
a short run samples — the first pass visited 8 and wedged in none of them, which
says nothing about the layouts it did not see, and rooms that could trap a player
certainly exist. Treat a pass as "no wedging in the rooms sampled" and raise
TRIALS if the answer needs to be stronger than that. The durable fix, if room
jumping is ever re-enabled, is to verify the player can move after each teleport
and re-place him if not, rather than to trust a sample.

    .venv/Scripts/python.exe scripts/probe_stuck.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.curriculum import CombatCurriculum  # noqa: E402

TRIALS = 24
WALK_TICKS = 12
# Isaac moves ~4 px a tick at base speed, so a dozen ticks of unobstructed
# walking is tens of pixels. Anything under this is not walking.
MOVED = 12.0


def walk(bridge, mx: int, my: int) -> float:
    """Hold a direction and report how far the player actually got."""
    obs = bridge.receive()
    start = (obs["player"]["x"], obs["player"]["y"])
    bridge.send({"t": "act", "mx": mx, "my": my, "sx": 0, "sy": 0,
                 "bomb": False, "item": False})
    for _ in range(WALK_TICKS - 1):
        obs = bridge.receive()
        bridge.send({"t": "act", "mx": mx, "my": my, "sx": 0, "sy": 0,
                     "bomb": False, "item": False})
    obs = bridge.receive()
    bridge.send({"t": "noop"})
    end = (obs["player"]["x"], obs["player"]["y"])
    return max(abs(end[0] - start[0]), abs(end[1] - start[1]))


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    config = load_config()
    curriculum = CombatCurriculum(max_enemies=config.combat.max_enemies,
                                  difficulty=0.5)

    fleet = launcher.bring_up(config, count=1)
    try:
        bridges = fleet.bridges()
        if not bridges:
            raise SystemExit("instance did not reach a run")
        bridge = bridges[0]
        for _ in range(20):
            bridge.pump()

        stuck = 0
        blocked_dirs: Counter = Counter()
        rooms_seen: Counter = Counter()

        for trial in range(TRIALS):
            encounter = curriculum.sample()
            # Spawn no enemies: this measures the room and the teleport alone,
            # so a failure to move cannot be blamed on something blocking.
            command = encounter.to_command()
            command["enemies"] = []
            bridge.receive()
            bridge.send(command)
            for _ in range(6):
                bridge.receive()
                bridge.send({"t": "noop"})

            obs = bridge.receive()
            bridge.send({"t": "noop"})
            room = obs["room"]["index"]
            rooms_seen[room] += 1
            position = (round(obs["player"]["x"]), round(obs["player"]["y"]))

            distances = {"left": walk(bridge, -1, 0), "right": walk(bridge, 1, 0),
                         "up": walk(bridge, 0, -1), "down": walk(bridge, 0, 1)}
            blocked = [name for name, moved in distances.items() if moved < MOVED]
            for name in blocked:
                blocked_dirs[name] += 1
            # Wedged means it cannot leave, not that one side is a wall — any
            # spot against a wall legitimately blocks one or two directions.
            wedged = len(blocked) >= 3
            stuck += wedged

            print(f"  trial {trial:2d}: room {room:>4} at {position}  "
                  + "  ".join(f"{n}:{d:5.1f}" for n, d in distances.items())
                  + ("   WEDGED" if wedged else ""))

        print(f"\nrooms visited: {len(rooms_seen)}")
        print(f"blocked directions: {dict(blocked_dirs)}")
        print(f"wedged (3+ directions blocked): {stuck}/{TRIALS} "
              f"({stuck / TRIALS:.0%})")
        print()
        if stuck > TRIALS * 0.1:
            print("PROBLEM: the teleport strands the player often enough to "
                  "explain a large drop in win rate. Every such encounter is "
                  "lost before it starts, and nothing in the metrics says so.")
        elif stuck:
            print("occasional wedging — real but too rare to explain a halving "
                  "of difficulty on its own")
        else:
            print("the teleport never stranded the player; look elsewhere for "
                  "the difficulty regression")
    finally:
        fleet.shutdown()


if __name__ == "__main__":
    main()
