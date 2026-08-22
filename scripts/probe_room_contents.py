"""Do the rooms the combat task could jump to actually contain obstacles?

The combat encounter is built in whatever room the player is standing in, which
is the floor's start room, and the start room is bare. That matters now because
`diagnose_cornered.py` measured blocked retreat at **5.25x** enriched at death on
floor-v23 (33.3% against a 6.3% baseline): a fight learned where retreat always
succeeds would be learning the wrong tactic for the case that actually kills.

The mod can already jump to a random plain room — `jumpToPlainRoom`, reached by
`"new_room": True` on a scenario command. `curriculum.py` sends False, and the
recorded reason is that room-jumping "was added to break the wall attractor by
varying geometry, and it does not: plain rooms on a floor are nearly all the same
520x280 shape". That judged *shape*. This asks a different question — obstacles —
and the same comment is weak evidence *against*, since it calls plain rooms
homogeneous.

So before flipping the flag, measure what is in those rooms:

  interior blocking   solid or pit tiles inside the wall ring, per room
  empty rooms         share with none at all — the number that decides this
  shapes              confirms the ROOMSHAPE_1x1 filter holds, since landing in
                      2x2 rooms is what cost the last attempt 0.73 -> 0.33
                      difficulty

The start room is sampled as a control, because "plain rooms have obstacles" only
matters relative to what the task uses today.

Counting is deliberately restricted to the **interior**: the wall ring is solid
by construction and including it would report every bare room as full of
obstacles, which is the shape of mistake this project keeps making.

    .venv/Scripts/python.exe scripts/probe_room_contents.py
    .venv/Scripts/python.exe scripts/probe_room_contents.py --rooms 200
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.env import IsaacVecEnv  # noqa: E402

SOLID_BIT, PIT_BIT = 0, 2
SETTLE_TICKS = 3


def interior_blocking(room: dict) -> tuple[int, int, int]:
    """(blocking interior tiles, interior tiles, raw width) for one room."""
    cells = room.get("grid") or []
    width = int(room.get("grid_width") or 0)
    if not cells or width <= 0:
        return -1, 0, 0
    height = len(cells) // width
    if height < 3 or width < 3:
        return -1, 0, width
    tiles = np.asarray(cells[:height * width], dtype=np.int16).reshape(height, width)
    inner = tiles[1:-1, 1:-1]
    blocking = ((inner & (1 << SOLID_BIT)) | (inner & (1 << PIT_BIT))) != 0
    return int(blocking.sum()), int(inner.size), width


def sample(env: IsaacVecEnv, new_room: bool,
           rooms: int) -> tuple[list, Counter, Counter, Counter]:
    """Jump (or not) and read what each instance is standing in."""
    counts: list[tuple[int, int]] = []
    shapes: Counter = Counter()
    # Distinct room indices are the control for the whole probe: if the jump is
    # not moving the player, every other column is one room measured N times.
    indices: Counter = Counter()
    jumps: Counter = Counter()
    rounds = max(1, rooms // max(env.num_envs, 1))
    for _ in range(rounds):
        message = {"t": "scenario", "enemies": [], "heal": True,
                   "reposition": True, "new_room": new_room}
        env._exchange_all([dict(message) for _ in range(env.num_envs)])
        # Room generation lands a tick or two later; reading immediately would
        # sample the room being left rather than the one arrived in.
        #
        # **Read the value `_exchange_all` returns, never `env._latest`.**
        # `_exchange_all` only sends and receives; `_latest` is refreshed by
        # `_prime()` at reset and by each env's own `step()` loop, neither of
        # which runs here. An earlier version of this probe read `_latest` and
        # duly reported 200 identical rooms with zero obstacles — 200 reads of
        # one cached reset observation. It was caught only because "every real
        # Basement room is bare" is not believable, which is not a control.
        latest_obs: list[dict | None] = [None] * env.num_envs
        for _ in range(SETTLE_TICKS):
            latest_obs = env._exchange_all(
                [{"t": "noop"} for _ in range(env.num_envs)])
        for index in range(env.num_envs):
            latest = latest_obs[index]
            if not latest or not latest.get("ready", True):
                continue
            room = latest.get("room") or {}
            blocking, interior, _ = interior_blocking(room)
            if blocking < 0 or interior == 0:
                continue
            counts.append((blocking, interior))
            shapes[int(room.get("shape", -1))] += 1
            indices[int(room.get("index", -1))] += 1
            jump = latest.get("scenario_jump")
            if jump:
                jumps[(bool(jump.get("shape_enum")),
                       int(jump.get("room_types", -1)),
                       int(jump.get("candidates", -1)),
                       bool(jump.get("changed")))] += 1
    return counts, shapes, indices, jumps


def report(label: str, counts: list, shapes: Counter,
           indices: Counter, jumps: Counter) -> float:
    if not counts:
        print(f"  {label:<12} no readable rooms")
        return -1.0
    blocking = np.array([c for c, _ in counts], dtype=float)
    interior = np.array([i for _, i in counts], dtype=float)
    empty = float((blocking == 0).mean())
    print(f"  {label:<12}{len(counts):>7}"
          f"{blocking.mean():>12.2f}{np.median(blocking):>10.1f}"
          f"{blocking.max():>7.0f}{(blocking / interior).mean():>11.1%}"
          f"{empty:>10.1%}")
    print(f"               shapes {dict(shapes)}   "
          f"distinct rooms {len(indices)}")
    if jumps:
        print("               jump (shape_enum, room_types, candidates, "
              f"changed): {dict(jumps)}")
    return empty


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--rooms", type=int, default=200)
    args = parser.parse_args()

    config = load_config()
    count = args.instances or config.instances.count
    print(f"bringing up {count} instance(s)\n")

    fleet = launcher.bring_up(config, count=count)
    problems: list[str] = []
    try:
        bridges = fleet.bridges()
        if not bridges:
            raise SystemExit("no instance reached a run")
        env = IsaacVecEnv(bridges, config)
        env.reset()

        print(f"  {'sample':<12}{'rooms':>7}{'blocking':>12}{'median':>10}"
              f"{'max':>7}{'density':>11}{'empty':>10}")
        start_empty = report("start room", *sample(env, False, args.rooms))
        jumped_empty = report("jumped", *sample(env, True, args.rooms))

        print()
        if jumped_empty < 0:
            problems.append("no rooms sampled on the jump path — the probe "
                            "cannot answer the question it was written for")
        elif jumped_empty > 0.5:
            print("  -> most jumped rooms are bare too. Flipping `new_room` "
                  "buys little;\n     obstacles would have to be spawned "
                  "deliberately.")
        else:
            print("  -> jumped rooms carry real obstacles. `new_room: True` is "
                  "the cheap\n     option and needs no new mod code.")
        if start_empty >= 0 and jumped_empty >= 0:
            print(f"\n  start room empty {start_empty:.0%} of the time, "
                  f"jumped {jumped_empty:.0%}")

    finally:
        fleet.shutdown()

    if problems:
        print("\nPROBLEM:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
