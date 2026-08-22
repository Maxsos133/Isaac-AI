"""Does the policy use the obstacle grid, or only notice that one exists?

The grid is encoded in **room** coordinates — a fixed 15x9 frame indexed by room
tile — and goes through a flat `Linear(405, hidden)`. Everything the agent aims
with is **player-relative**, and the entity branch is pooled before the trunk.
So by the time the grid and the enemies meet, "enemy at offset (+80, -40)" and
"rock at room tile (4, 9)" are two unrelated 128-dim summaries, and there is no
representation left in which "is a rock between me and that enemy" exists.

That is the same defect the entity encoder already had and had fixed — see the
README note on player-relative entity positions, worth 37.5 points of success.

This measures it instead of arguing it. Same enemy, same number of obstacles,
only their *position* changes:

  clear      no obstacles but the walls
  blocking   one rock exactly on the line from player to enemy
  irrelevant one rock in a far corner — same count, different place
  spikes     hazard tiles immediately around the player

`blocking` vs `irrelevant` is the one that matters: it holds the obstacle count
fixed, so any difference is the policy reading *where* the obstacle is. If those
two produce the same action distribution, position is not being used.

    .venv/Scripts/python.exe scripts/diagnose_grid_use.py runs/floor-v10/policy.pt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai.env import GRID_HEIGHT, GRID_WIDTH, encode_observation  # noqa: E402
from isaac_ai.policy import ActorCritic  # noqa: E402
from probe_heads import doors  # noqa: E402

# Room extents used by the synthetic observation, and the tile each point maps
# to under encode_grid's row-major (grid_width) layout.
LEFT, TOP, RIGHT, BOTTOM = 60.0, 140.0, 580.0, 420.0
PLAYER = (320.0, 280.0)
AXES = ("move_x", "move_y", "shoot_x", "shoot_y")


def tile_of(x: float, y: float) -> tuple[int, int]:
    col = min(GRID_WIDTH - 1, max(0, int((x - LEFT) / (RIGHT - LEFT) * GRID_WIDTH)))
    row = min(GRID_HEIGHT - 1, max(0, int((y - TOP) / (BOTTOM - TOP) * GRID_HEIGHT)))
    return row, col


def build_grid(marks: list[tuple[int, int, int]]) -> list[int]:
    """Walled border plus whatever tiles are marked (row, col, class)."""
    cells = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
    for x in range(GRID_WIDTH):
        cells[0][x] = cells[GRID_HEIGHT - 1][x] = 1
    for y in range(GRID_HEIGHT):
        cells[y][0] = cells[y][GRID_WIDTH - 1] = 1
    for row, col, klass in marks:
        if 0 < row < GRID_HEIGHT - 1 and 0 < col < GRID_WIDTH - 1:
            cells[row][col] = klass
    return [v for row in cells for v in row]


def observation(enemy: tuple[float, float], marks: list[tuple[int, int, int]]) -> dict:
    px, py = PLAYER
    return {
        "ready": True,
        "player": {"x": px, "y": py, "vx": 0, "vy": 0, "hearts": 6,
                   "max_hearts": 6, "soul_hearts": 0, "bombs": 1, "keys": 0,
                   "coins": 0, "damage": 3.5, "speed": 1.0, "tear_delay": 10,
                   "range": 260, "can_fly": False, "active_item": 0,
                   "collectibles": 0, "trinket0": 0, "trinket1": 0,
                   "card0": 0, "pill0": 0, "is_dead": False},
        "room": {"index": 84, "type": 1, "shape": 1, "clear": False,
                 "top_left_x": LEFT, "top_left_y": TOP,
                 "bottom_right_x": RIGHT, "bottom_right_y": BOTTOM,
                 "enemies_alive": 1, "doors": doors(),
                 "grid": build_grid(marks), "grid_width": GRID_WIDTH},
        "level": {"stage": 1, "stage_type": 0, "curses": 0,
                  "rooms_total": 12, "rooms_visited": 4},
        "entities": [{"k": "enemy", "t": 10, "v": 0, "s": 0,
                      "x": enemy[0], "y": enemy[1], "vx": 0, "vy": 0,
                      "hp": 10, "mhp": 10, "boss": False,
                      "d": math.dist(enemy, PLAYER),
                      "consumable": False, "pedestal": False, "chest": False,
                      "hostile": False, "flying": False}],
        "events": {},
    }


def distributions(model, obs):
    encoded = encode_observation(obs)
    batch = {k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0)
             for k, v in encoded.items()}
    with torch.no_grad():
        logits, value = model(batch)
    return [torch.softmax(h, dim=-1)[0] for h in logits], float(value[0])


def total_variation(a, b) -> float:
    """Half the L1 distance: 0 = identical policy, 1 = disjoint."""
    return float(sum((x - y).abs().sum() for x, y in zip(a, b)) / (2 * len(a)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()

    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = ActorCritic()
        model.load_state_dict(checkpoint["policy"])
        model.eval()
        print(f"\n{path}  ({checkpoint.get('global_step', 0):,} steps)")

        # Enemy due east, four tiles away. The line between them runs along the
        # player's row, so a rock on that row between the two blocks every tear.
        enemy = (460.0, 280.0)
        prow, pcol = tile_of(*PLAYER)
        erow, ecol = tile_of(*enemy)
        between = (prow, (pcol + ecol) // 2, 1)
        print(f"  player tile {(prow, pcol)}, enemy tile {(erow, ecol)}, "
              f"blocking rock at {between[:2]}")

        cases = {
            "clear": [],
            "blocking": [between],
            "irrelevant": [(1, 2, 1)],
            "spikes": [(prow, pcol + 1, 2), (prow + 1, pcol, 2)],
        }
        result = {}
        for name, marks in cases.items():
            result[name], value = distributions(model, observation(enemy, marks))
            sx, sy = result[name][2], result[name][3]
            print(f"  {name:<11} shoot_x [{sx[0]:.2f} {sx[1]:.2f} {sx[2]:.2f}]  "
                  f"shoot_y [{sy[0]:.2f} {sy[1]:.2f} {sy[2]:.2f}]  V {value:+.2f}")

        print("\n  policy change (total variation, 0 = identical):")
        print(f"    clear    -> blocking    {total_variation(result['clear'], result['blocking']):.4f}")
        print(f"    clear    -> irrelevant  {total_variation(result['clear'], result['irrelevant']):.4f}")
        print(f"    blocking -> irrelevant  {total_variation(result['blocking'], result['irrelevant']):.4f}"
              "   <-- same rock count, different place")
        print(f"    clear    -> spikes      {total_variation(result['clear'], result['spikes']):.4f}")
        print("\n  reading: if blocking->irrelevant is ~0 the policy reacts to how")
        print("  many obstacles exist and not to where any of them is, so line of")
        print("  fire and hazard avoidance are both outside what it can represent.")


if __name__ == "__main__":
    main()
