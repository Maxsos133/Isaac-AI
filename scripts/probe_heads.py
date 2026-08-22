"""Does the policy actually aim, or has an action head died?

combat-v5 through v7 could not shoot horizontally at all. The vertical axis was
frozen on "down" with entropy 0.05 — it fired downward at enemies directly above
it — and the horizontal axis sat at 1.07 against a ln(3)=1.099 ceiling, meaning
a coin flip. The summed `shoot_entropy` those runs logged was ~1.10, which looks
like a policy halfway to confident, so it went unnoticed across three training
runs and two pixel students distilled from them.

The failure is invisible in aggregate and obvious the moment you ask the network
what it would do about an enemy on its left. That is all this does: place one
enemy in each direction and print the action distributions.

No game required — it reads a checkpoint and synthesises the observations, so it
costs seconds and can be run against every checkpoint a run produces.

    .venv/Scripts/python.exe scripts/probe_heads.py runs/combat-v8/policy.pt
    .venv/Scripts/python.exe scripts/probe_heads.py runs/*/policy.pt
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

AXES = ("move_x", "move_y", "shoot_x", "shoot_y")
LABELS = (("left", "none", "right"), ("up", "none", "down"),
          ("left", "none", "right"), ("up", "none", "down"))
UNIFORM = math.log(3)

# An axis below this is committed; above the second is uniform, i.e. abandoned.
COMMITTED, ABANDONED = 0.35, UNIFORM - 0.15


def walled_grid() -> list[int]:
    """A solid border with a couple of rocks — the floor of any real room.

    Omitting the grid entirely is not neutral. `encode_grid` returns all zeros
    for a room it cannot read, and an all-zero grid means "open floor in every
    cell", which never occurs in play: every room is walled. A grid-era policy
    asked about that state is being asked about somewhere it has never been.
    """
    cells = [[0] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
    for x in range(GRID_WIDTH):
        cells[0][x] = cells[GRID_HEIGHT - 1][x] = 1
    for y in range(GRID_HEIGHT):
        cells[y][0] = cells[y][GRID_WIDTH - 1] = 1
    for y, x in ((3, 4), (3, 10), (5, 7)):
        cells[y][x] = 1
    return [value for row in cells for value in row]


def doors() -> list[dict]:
    """One unvisited door on each of the four walls.

    A floor policy's move heads are trained almost entirely on where the doors
    are, so probing them in a doorless room measures nothing — that is why this
    probe read floor-v6's movement as abandoned while the agent was visibly
    walking to doors. Every door is left unvisited so `door_potential` is live
    in all four directions and no single one is the obvious answer.
    """
    return [
        {"slot": 0, "x": 320.0, "y": 140.0, "open": True, "locked": False,
         "visited": 0, "category": "normal"},   # north
        {"slot": 1, "x": 580.0, "y": 280.0, "open": True, "locked": False,
         "visited": 0, "category": "normal"},   # east
        {"slot": 2, "x": 320.0, "y": 420.0, "open": True, "locked": False,
         "visited": 0, "category": "normal"},   # south
        {"slot": 3, "x": 60.0, "y": 280.0, "open": True, "locked": False,
         "visited": 0, "category": "normal"},   # west
    ]


def observation(enemy_x: float, enemy_y: float) -> dict:
    player_x, player_y = 320.0, 280.0
    return {
        "ready": True,
        "player": {"x": player_x, "y": player_y, "vx": 0, "vy": 0, "hearts": 6,
                   "max_hearts": 6, "soul_hearts": 0, "bombs": 1, "keys": 0,
                   "coins": 0, "damage": 3.5, "speed": 1.0, "tear_delay": 10,
                   "range": 260, "can_fly": False, "active_item": 0,
                   "collectibles": 0, "trinket0": 0, "trinket1": 0,
                   "card0": 0, "pill0": 0, "is_dead": False},
        "room": {"index": 84, "type": 1, "shape": 1, "clear": False,
                 "top_left_x": 60, "top_left_y": 140, "bottom_right_x": 580,
                 "bottom_right_y": 420, "enemies_alive": 1,
                 "doors": doors(),
                 "grid": walled_grid(), "grid_width": GRID_WIDTH},
        "level": {"stage": 1, "stage_type": 0, "curses": 0,
                  "rooms_total": 12, "rooms_visited": 4},
        "entities": [{"k": "enemy", "t": 10, "v": 0, "s": 0,
                      "x": enemy_x, "y": enemy_y, "vx": 0, "vy": 0,
                      "hp": 10, "mhp": 10, "boss": False,
                      "d": math.hypot(enemy_x - player_x, enemy_y - player_y)}],
        "events": {},
    }


def distributions(model: ActorCritic, obs: dict) -> list[torch.Tensor]:
    encoded = encode_observation(obs)
    batch = {key: torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)
             for key, value in encoded.items()}
    with torch.no_grad():
        logits, _ = model(batch)
    return [torch.softmax(head, dim=-1)[0] for head in logits]


def inspect(path: Path) -> bool:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ActorCritic()
    try:
        model.load_state_dict(checkpoint["policy"])
    except RuntimeError as exc:
        print(f"{path}: does not fit the current network ({exc.args[0][:60]}...)")
        return True
    model.eval()

    print(f"\n{path}  ({checkpoint.get('global_step', 0):,} steps)")

    # Eight directions at two ranges. Four cardinal points cannot separate a
    # directional bias ("only fires up and right") from a distance effect, and
    # encounters now open at ~100 units so a single far test sits outside the
    # distribution the policy is actually trained on.
    compass = {"W": (-1, 0), "E": (1, 0), "N": (0, -1), "S": (0, 1),
               "NW": (-0.7, -0.7), "NE": (0.7, -0.7),
               "SW": (-0.7, 0.7), "SE": (0.7, 0.7)}
    wanted = {"W": ("left",), "E": ("right",), "N": ("up",), "S": ("down",),
              "NW": ("left", "up"), "NE": ("right", "up"),
              "SW": ("left", "down"), "SE": ("right", "down")}

    print(f"  {'dir':<5}{'range':>7}{'shoot x (l/none/r)':>24}"
          f"{'shoot y (u/none/d)':>24}   fires")
    aimed = total = 0
    for name, (ux, uy) in compass.items():
        for reach in (100, 200):
            probs = distributions(model,
                                  observation(320 + ux * reach, 280 + uy * reach))
            sx, sy = probs[2], probs[3]
            choice = (LABELS[2][int(sx.argmax())], LABELS[3][int(sy.argmax())])
            # Correct if it fires along at least one axis towards the target and
            # none away from it.
            towards = any(c in wanted[name] for c in choice)
            away = any(c != "none" and c not in wanted[name] for c in choice)
            hit = towards and not away
            aimed += hit
            total += 1
            print(f"  {name:<5}{reach:>7}[{sx[0]:.2f} {sx[1]:.2f} {sx[2]:.2f}]"
                  f"{'':>7}[{sy[0]:.2f} {sy[1]:.2f} {sy[2]:.2f}]"
                  f"{'':>7}{choice[0]}/{choice[1]} {'ok' if hit else 'no'}")

    probs = distributions(model, observation(460, 280))
    print(f"\n  per-axis entropy (uniform = {UNIFORM:.3f}):")
    dead = []
    for index, name in enumerate(AXES):
        entropy = float(-(probs[index] * torch.log(probs[index] + 1e-9)).sum())
        note = ""
        if entropy < COMMITTED:
            note = "  committed"
        elif entropy > ABANDONED:
            note = "  ABANDONED — this axis is a coin flip"
            dead.append(name)
        print(f"    {name:<9}{entropy:.3f}{note}")

    print(f"\n  fires towards the target in {aimed}/{total} placements", end="")
    print(f"; dead axes: {', '.join(dead)}" if dead else "; no dead axes")
    # Two thirds is a low bar deliberately: this is meant to catch a head that
    # has stopped working, not to grade good aim.
    return bool(dead) or aimed < total * 2 // 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()

    problems = [path for path in args.checkpoints if inspect(Path(path))]
    print()
    if problems:
        print(f"PROBLEM in {len(problems)}/{len(args.checkpoints)}: "
              f"{', '.join(problems)}")
    else:
        print(f"all {len(args.checkpoints)} checkpoints aim and have no dead axes")


if __name__ == "__main__":
    main()
