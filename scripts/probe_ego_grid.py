"""Is the egocentric window aligned to the player, on real rooms, at fleet size?

The window is only useful if its centre really is the tile the player occupies.
The mapping from position to tile is derived from the room's extents and the raw
grid dimensions, and both of those change on larger rooms — which the offline
tests cannot cover, because they synthesise a 1x1 room.

Three things have to hold on live data:

  centred     the class at the window's centre matches the raw grid at the
              player's own tile. If the mapping drifts, every fact in the window
              is offset by a tile and the encoding is worse than useless.
  standable   the player's own tile is essentially never solid. A player cannot
              stand inside a rock, so a high rate here means the mapping is off
              even when the centre check happens to agree with itself.
  walled      close to a room edge the window shows the off-map wall ring, so
              the agent is told it cannot walk out rather than that the space is
              free.

Room shapes actually observed are reported, because "large rooms are fine" is
only earned if a large room was seen. A synthetic 2x2 case runs regardless.

    .venv/Scripts/python.exe scripts/probe_ego_grid.py
    .venv/Scripts/python.exe scripts/probe_ego_grid.py --instances 4 --steps 300
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
from isaac_ai.env import (EGO_RADIUS, EGO_SIZE, GRID_CLASSES,  # noqa: E402
                          encode_egocentric_grid, tile_at, tile_of_index)
from isaac_ai.floor_curriculum import FloorCurriculum  # noqa: E402
from isaac_ai.floors import FloorVecEnv  # noqa: E402


def window_of(room: dict, px: float, py: float, gi=None) -> np.ndarray:
    flat = encode_egocentric_grid(room, px, py, gi)
    return flat.reshape(GRID_CLASSES, EGO_SIZE, EGO_SIZE)


def raw_tile(room: dict, px: float, py: float, gi=None):
    """The player's tile straight from the payload.

    This mirrors the encoder's mapping, so `centred` only proves the two agree —
    it cannot prove either is right. **`standable` is the independent check**: a
    player cannot be standing inside a rock, so a solid tile under the player
    means the mapping is wrong however self-consistent it looks. That is exactly
    how the playable-area-versus-full-grid mismatch was caught, at 27%.
    """
    cells, width = room.get("grid") or [], int(room.get("grid_width") or 0)
    if not cells or width <= 0:
        return None
    height = len(cells) // width
    located = tile_of_index(room, gi)
    if located is not None:
        r, c = located
        return int(cells[r * width + c]), r, c, height, width
    left, top = float(room["top_left_x"]), float(room["top_left_y"])
    span_x = max(float(room["bottom_right_x"]) - left, 1.0)
    span_y = max(float(room["bottom_right_y"]) - top, 1.0)
    col = min(width - 2, max(1, int(round((px - left) / span_x * max(width - 3, 1))) + 1))
    row = min(height - 2, max(1, int(round((py - top) / span_y * max(height - 3, 1))) + 1))
    return int(cells[row * width + col]), row, col, height, width


def check_synthetic_large_room() -> list[str]:
    """A 2x2 room, constructed so it runs whether or not one is encountered."""
    problems = []
    width, height = 28, 16                      # a 2x2 room's raw tile grid
    cells = [0] * (width * height)
    for x in range(width):
        cells[x] = cells[(height - 1) * width + x] = 1
    for y in range(height):
        cells[y * width] = cells[y * width + width - 1] = 1
    marker_row, marker_col = 9, 20
    cells[marker_row * width + marker_col] = 2   # hazard right of centre
    # Extents are the centres of interior tiles (1,1) and (height-2, width-2),
    # matching what GetTopLeftPos/GetBottomRightPos actually report.
    room = {"top_left_x": 15.0, "top_left_y": 15.0,
            "bottom_right_x": (width - 2 + 0.5) * 10.0,
            "bottom_right_y": (height - 2 + 0.5) * 10.0,
            "grid": cells, "grid_width": width}
    # Stand one tile left of the marker.
    px, py = (marker_col - 1 + 0.5) * 10.0, (marker_row + 0.5) * 10.0
    grid = window_of(room, px, py)
    if float(grid[1, EGO_RADIUS, EGO_RADIUS + 1]) != 1.0:
        problems.append("2x2 room: hazard one tile right is not at the window's "
                        "centre+1 — the tile mapping does not survive a large room")
    if float(grid[:, EGO_RADIUS, EGO_RADIUS].sum()) != 0.0:
        problems.append("2x2 room: the player's own tile reads as occupied")
    return problems


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--steps", type=int, default=900)
    args = parser.parse_args()

    problems = check_synthetic_large_room()
    print("synthetic 2x2 room: "
          + ("OK" if not problems else "FAILED"))

    config = load_config()
    count = args.instances or config.instances.count
    print(f"\nbringing up {count} instance(s)\n")
    fleet = launcher.bring_up(config, count=count)
    try:
        bridges = fleet.bridges()
        if not bridges:
            raise SystemExit("no instance reached a run")
        env = FloorVecEnv(bridges, config, FloorCurriculum())
        env.reset()

        shapes: Counter = Counter()
        dims: Counter = Counter()
        samples = centred = standable = edge_walled = edge_seen = 0

        for step in range(args.steps):
            _, _, terminated, truncated, _ = env.step(
                np.random.randint(0, 3, size=(env.num_envs, 4)))
            done = terminated | truncated
            if done.any():
                env.reset_done(done)

            for index in range(env.num_envs):
                latest = env._latest[index]
                if not latest or not latest.get("ready", True):
                    continue
                room, player = latest.get("room"), latest.get("player")
                if not room or not player:
                    continue
                info = raw_tile(room, float(player["x"]), float(player["y"]),
                                player.get("grid_index"))
                if info is None:
                    continue
                value, row, col, height, width = info
                shapes[room.get("shape")] += 1
                dims[(height, width)] += 1
                samples += 1

                grid = window_of(room, float(player["x"]), float(player["y"]),
                                 player.get("grid_index"))
                centre = grid[:, EGO_RADIUS, EGO_RADIUS]
                # The payload is a bitmask of tile properties, not a class id.
                # This probe originally decoded it as a class and reported 22
                # false mismatches — every one of them a pit, which moved from
                # value 3 to bit 2 (value 4) and so failed a `1 <= v <= 3` test.
                # `centred` is a self-consistency check either way; `standable`
                # is the one that tests against something outside the code.
                expected = np.array(
                    [1.0 if (value >> bit) & 1 else 0.0
                     for bit in range(GRID_CLASSES)], dtype=np.float32)
                centred += bool(np.array_equal(centre, expected))
                standable += bool(centre[0] == 0.0)

                # Within the window's reach of an edge, the off-map ring must
                # show up as solid on the side facing out.
                if col < EGO_RADIUS:
                    edge_seen += 1
                    edge_walled += bool(grid[0, EGO_RADIUS, 0] == 1.0)
                elif col >= width - EGO_RADIUS:
                    edge_seen += 1
                    edge_walled += bool(grid[0, EGO_RADIUS, EGO_SIZE - 1] == 1.0)

        print(f"{samples} observations")
        print(f"  room shapes seen:     {dict(shapes)}")
        print(f"  raw grid dimensions:  {dict(dims)}")
        print(f"  centre matches raw:   {centred}/{samples}")
        print(f"  player tile standable:{standable}/{samples}")
        print(f"  edge ring walled:     {edge_walled}/{edge_seen}")

        if samples:
            if centred != samples:
                problems.append(f"window centre disagreed with the raw grid on "
                                f"{samples - centred}/{samples} observations")
            if standable < samples * 0.99:
                problems.append(f"player stood in a solid tile on "
                                f"{samples - standable}/{samples} observations")
            if edge_seen and edge_walled < edge_seen:
                problems.append(f"off-map ring missing on {edge_seen - edge_walled}"
                                f"/{edge_seen} edge observations")
            if len(dims) == 1:
                print("\n  note: only one room size was encountered, so the "
                      "large-room path is covered by the synthetic case only")
    finally:
        fleet.shutdown()

    if problems:
        print("\nPROBLEM:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("\negocentric window is aligned to the player on real rooms.")


if __name__ == "__main__":
    main()
