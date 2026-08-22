"""Why is one shoot axis dead — decomposition, or abandonment?

README lists this as genuinely unresolved: "Every combat policy trained here
aligns on one axis by *moving* and fires along the other." If that is what is
happening it is a strategy, not a bug — you cannot fire left and right if you
have arranged for every enemy to be above or below you. If it is not happening,
the head has simply been pinned uniform by the entropy bonus and the agent
cannot deal with enemies on the dead axis at all.

The two stories make opposite predictions about the *move* heads, which is what
separates them without needing the game:

  decomposition   movement systematically reduces |dx| — the agent walks itself
                  onto the enemy's column so the live vertical axis can fire.
                  Dead axis is horizontal, alignment should be in x.
  abandonment     movement shows no such alignment; enemies off the live axis
                  are simply never engaged.

Reported per placement and aggregated. No game required.

    .venv/Scripts/python.exe scripts/diagnose_shoot_axis.py runs/floor-v7b/policy.pt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai.policy import ActorCritic  # noqa: E402
from probe_heads import distributions, observation  # noqa: E402

AXIS = (-1, 0, 1)
UNIFORM = math.log(3)


def analyse(path: Path, doors: str = "unvisited", angles: int = 16,
            verbose: bool = True) -> tuple[int, int, int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ActorCritic()
    model.load_state_dict(checkpoint["policy"])
    model.eval()
    if verbose:
        print(f"\n{path}  ({checkpoint.get('global_step', 0):,} steps)"
              f"  doors={doors}\n")

    if verbose:
        print(f"  {'angle':>6}{'dx':>7}{'dy':>7}   {'move':>11}  {'shoot':>11}"
              f"   {'|dx|':>5} {'|dy|':>5}  hit")
    aligns_x = aligns_y = 0
    hits = total = 0
    # How many *distinct* movement responses the enemy's position produces. A
    # policy that answers every placement the same way is not reacting to the
    # enemy at all, and the alignment rates above then just measure how often a
    # fixed direction happens to point the right way (~50% by construction).
    moves: dict[str, int] = {}
    # Placements strictly off both axes are the interesting ones: on a pure
    # cardinal the "aligned" answer and the "chase" answer coincide.
    for step in range(angles):
        theta = 2 * math.pi * step / angles
        dx, dy = math.cos(theta), math.sin(theta)
        for reach in (120, 220):
            ex, ey = 320 + dx * reach, 280 + dy * reach
            probs, _ = _dist(model, ex, ey, doors)
            mx = AXIS[int(probs[0].argmax())]
            my = AXIS[int(probs[1].argmax())]
            sx = AXIS[int(probs[2].argmax())]
            sy = AXIS[int(probs[3].argmax())]

            # Does moving that way shrink the gap on each axis? Positive dx
            # means the enemy is to the right, so mx = +1 closes it.
            closes_x = mx != 0 and (mx > 0) == (dx > 0)
            closes_y = my != 0 and (my > 0) == (dy > 0)
            aligns_x += closes_x
            aligns_y += closes_y
            key = _arrow(mx, my)
            moves[key] = moves.get(key, 0) + 1

            # Would a tear fired this way travel towards the enemy on the axis
            # that matters, without travelling away on the other?
            towards = ((sx != 0 and (sx > 0) == (dx > 0))
                       or (sy != 0 and (sy > 0) == (dy > 0)))
            away = ((sx != 0 and (sx > 0) != (dx > 0))
                    or (sy != 0 and (sy > 0) != (dy > 0)))
            hit = towards and not away
            hits += hit
            total += 1

            if reach == 220 and verbose:
                print(f"  {math.degrees(theta):6.0f}{dx * reach:7.0f}"
                      f"{dy * reach:7.0f}   {_arrow(mx, my):>11}  "
                      f"{_arrow(sx, sy):>11}   "
                      f"{'yes' if closes_x else '  .':>5} "
                      f"{'yes' if closes_y else '  .':>5}  "
                      f"{'ok' if hit else 'no'}")

    if verbose:
        print(f"\n  movement closes the horizontal gap in {aligns_x}/{total} "
              f"placements ({aligns_x / total:.0%})")
        print(f"  movement closes the vertical gap   in {aligns_y}/{total} "
              f"placements ({aligns_y / total:.0%})")
        print(f"  fires towards the target           in {hits}/{total} "
              f"placements ({hits / total:.0%})")
        print(f"  distinct move responses across {total} placements: "
              f"{len(moves)} — {dict(sorted(moves.items(), key=lambda kv: -kv[1]))}")
    return aligns_x, aligns_y, hits, total


def _dist(model, ex: float, ey: float, doors: str):
    obs = observation(ex, ey)
    if doors == "none":
        obs["room"]["doors"] = []
    elif doors == "visited":
        # Doors still there, nothing left to explore — removes the exploration
        # pull without pretending the room has no exits.
        for door in obs["room"]["doors"]:
            door["visited"] = 1
    return distributions(model, obs), None


def _arrow(x: int, y: int) -> str:
    if x == 0 and y == 0:
        return "still"
    parts = []
    if y < 0:
        parts.append("up")
    if y > 0:
        parts.append("down")
    if x < 0:
        parts.append("left")
    if x > 0:
        parts.append("right")
    return "-".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--doors", choices=("unvisited", "visited", "none"),
                        default=None,
                        help="default: sweep all three as a control")
    args = parser.parse_args()

    for path in args.checkpoints:
        if args.doors:
            analyse(Path(path), doors=args.doors)
            continue
        # Door-seeking is a floor policy's dominant drive, so probing enemy
        # response in a room with four unvisited doors measures the doors. The
        # sweep is the control: if the response only varies once the doors stop
        # pulling, the enemy heads are alive and simply outranked.
        analyse(Path(path), doors="unvisited")
        print("\n  control — same placements, exploration pull removed:\n")
        print(f"  {'doors':<10}{'aligns x':>10}{'aligns y':>10}"
              f"{'fires ok':>10}{'move responses':>16}")
        for condition in ("unvisited", "visited", "none"):
            ax, ay, hits, total = analyse(Path(path), doors=condition,
                                          verbose=False)
            moves = _move_variety(Path(path), condition)
            print(f"  {condition:<10}{ax}/{total:<7}{ay}/{total:<7}"
                  f"{hits}/{total:<7}{moves:>10} distinct")


def _move_variety(path: Path, doors: str, angles: int = 16) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ActorCritic()
    model.load_state_dict(checkpoint["policy"])
    model.eval()
    seen = set()
    for step in range(angles):
        theta = 2 * math.pi * step / angles
        for reach in (120, 220):
            probs, _ = _dist(model, 320 + math.cos(theta) * reach,
                             280 + math.sin(theta) * reach, doors)
            seen.add((int(probs[0].argmax()), int(probs[1].argmax())))
    return len(seen)


if __name__ == "__main__":
    main()
