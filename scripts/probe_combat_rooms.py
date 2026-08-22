"""Do combat encounters actually happen in varied rooms now, and safely?

Every combat encounter this project has run happened in the same 520x280 room,
and every policy trained in it backed onto a wall and fired across — v4 the left
wall, v5 through v7 the top, v8 was forming the left again. One fixed geometry
makes one wall permanently correct, aiming stops mattering, and the unused
action heads get pinned uniform by the entropy bonus.

`buildScenario` now jumps to another plain room first. That is not obviously
safe, so this checks it before a training run depends on it:

  varies      the room index and its dimensions actually change between
              encounters — the whole point
  spawns      enemies survive the transition and are alive after the settle
              ticks, rather than being swept by the room change
  settles     SPAWN_SETTLE_TICKS is enough for a room change plus a spawn; the
              navigation setup sweeps for several ticks after ChangeRoom because
              things arrive late, and combat only waits four
  stats       damage/speed/range never move — a treasure or shop room would hand
              over an item and unfreeze the very thing the save snapshots exist
              to hold still

    .venv/Scripts/python.exe scripts/probe_combat_rooms.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.curriculum import CombatCurriculum  # noqa: E402

ENCOUNTERS = 12
SETTLE_PROBE_TICKS = 20


def stats_of(obs: dict) -> tuple:
    p = obs["player"]
    return (round(p["damage"], 2), round(p["speed"], 2), round(p["range"], 1),
            p["max_hearts"])


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
        first = bridge.receive()
        bridge.send({"t": "noop"})
        baseline = stats_of(first)
        print(f"baseline stats damage/speed/range/hearts: {baseline}\n")

        rooms: Counter = Counter()
        shapes: Counter = Counter()
        problems: list[str] = []
        settle_curve: list[int] = []

        for index in range(ENCOUNTERS):
            encounter = curriculum.sample()
            wanted = encounter.enemy_count

            bridge.receive()
            bridge.send(encounter.to_command())

            # Watch enemies appear tick by tick, so "four is enough" is measured
            # rather than assumed.
            alive_by_tick = []
            for _ in range(SETTLE_PROBE_TICKS):
                obs = bridge.receive()
                bridge.send({"t": "noop"})
                alive_by_tick.append(obs["room"].get("enemies_alive", 0))

            room = obs["room"]
            width = room["bottom_right_x"] - room["top_left_x"]
            height = room["bottom_right_y"] - room["top_left_y"]
            rooms[room["index"]] += 1
            shapes[(width, height)] += 1

            settled_at = next((i for i, n in enumerate(alive_by_tick)
                               if n >= wanted), None)
            if settled_at is None:
                problems.append(f"encounter {index}: only {max(alive_by_tick)}/"
                                f"{wanted} enemies ever appeared")
            else:
                settle_curve.append(settled_at + 1)

            if stats_of(obs) != baseline:
                problems.append(f"encounter {index}: stats changed to "
                                f"{stats_of(obs)} — an item was picked up")

            print(f"  encounter {index:2d}: room {room['index']:>4} "
                  f"{width:.0f}x{height:.0f}  wanted {wanted} enemies, "
                  f"alive by tick {alive_by_tick[:6]}"
                  + (f"  settled at tick {settled_at + 1}" if settled_at is not None
                     else "  NEVER SETTLED"))

        print(f"\ndistinct rooms visited:  {len(rooms)}/{ENCOUNTERS}")
        print(f"distinct room shapes:    {len(shapes)}  {sorted(shapes)}")
        if settle_curve:
            print(f"ticks to full spawn:     max {max(settle_curve)}, "
                  f"median {sorted(settle_curve)[len(settle_curve) // 2]}")
            from isaac_ai.combat import SPAWN_SETTLE_TICKS
            if max(settle_curve) > SPAWN_SETTLE_TICKS:
                problems.append(
                    f"spawns need up to {max(settle_curve)} ticks but "
                    f"SPAWN_SETTLE_TICKS is {SPAWN_SETTLE_TICKS} — the policy "
                    f"would see a half-built encounter")

        if len(rooms) < 3:
            problems.append(f"only {len(rooms)} distinct rooms — the jump is "
                            f"not varying the geometry")

        print()
        if problems:
            print("PROBLEMS:")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print("room variety is safe: geometry varies, spawns survive the "
                  "transition, settle ticks suffice, stats never move")
    finally:
        fleet.shutdown()


if __name__ == "__main__":
    main()
