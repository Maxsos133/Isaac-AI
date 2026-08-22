"""Is the corner camping a strategy, or is the agent just pressed against a wall?

`diagnose_shoot_axis.py` shows the policy answering `down` to 30 of 32 enemy
placements — a near-constant, not a response. A constant downward drive does not
look like a decision from outside; it looks like an agent that walks into the
bottom of the room and stays there, and that is exactly what corner camping
would look like too. The two have completely different fixes, so guessing is
expensive.

They separate cleanly on one measurement: **does the movement it asks for
actually happen?** An agent choosing to hold a corner still moves freely when it
wants to. An agent pinned against geometry issues a direction and does not
travel.

Reported alongside where in the room the time is actually spent, because "camps
the bottom-left" should show up as mass in one corner rather than as a policy
that merely leans that way.

    .venv/Scripts/python.exe scripts/diagnose_camping.py runs/floor-v12/policy.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.env import decode_action, tile_at  # noqa: E402
from isaac_ai.floor_curriculum import FloorCurriculum  # noqa: E402
from isaac_ai.floors import FloorVecEnv, door_is_targetable  # noqa: E402
from isaac_ai.policy import ActorCritic, to_tensors  # noqa: E402

# Base Isaac covers several units a tick and action_repeat is 2, so a step that
# moves less than this while asking to move is not slow, it is obstructed.
MOVED = 1.0
SOLID_BIT = 1 << 0


def shaping_target(room: dict, px: float, py: float):
    """The door the potential is pulling towards — its own filter, imported."""
    best, chosen = None, None
    for door in room.get("doors", []):
        if not door_is_targetable(door):
            continue
        distance = np.hypot(float(door["x"]) - px, float(door["y"]) - py)
        if best is None or distance < best:
            best, chosen = distance, door
    return chosen


def line_is_obstructed(room: dict, px: float, py: float, door: dict) -> bool:
    """Does a straight line from the player to that door cross a solid tile?

    `door_potential` measures closeness with `math.dist` — a straight line, in a
    room with walls in it. If something solid sits on that line then walking
    *around* it increases the distance, so the shaping pays a penalty for going
    around and the highest-potential spot is pressed flat against the obstacle.
    This is the check for whether that is actually what is happening.

    The ends of the segment are skipped: the door's own tile is GRID_DOOR, which
    is in the solid set, and the player's tile is where they already stand.
    """
    cells, width = room.get("grid") or [], int(room.get("grid_width") or 0)
    if not cells or width <= 0:
        return False
    here = tile_at(room, px, py)
    if here is None:
        return False
    for step in range(2, 18):          # t = 0.11 .. 0.94, endpoints excluded
        t = step / 19.0
        point = tile_at(room, px + (float(door["x"]) - px) * t,
                        py + (float(door["y"]) - py) * t)
        if point is None or point == here:
            continue
        row, column = point
        if int(cells[row * width + column]) & SOLID_BIT:
            return True
    return False


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--random", action="store_true",
                        help="drive with random actions instead — the control, "
                             "without which the blocked rate means nothing, "
                             "since a random walk bumps into geometry too")
    args = parser.parse_args()

    config = load_config()
    policy = None
    if args.random:
        print("CONTROL: random actions\n")
    else:
        checkpoint = torch.load(config.root / args.checkpoint, map_location="cpu",
                                weights_only=False)
        policy = ActorCritic()
        policy.load_state_dict(checkpoint["policy"])
        policy.eval()
        print(f"{args.checkpoint} ({checkpoint.get('global_step', 0):,} steps)\n")

    count = args.instances or config.instances.count
    fleet = launcher.bring_up(config, count=count)
    try:
        bridges = fleet.bridges()
        if not bridges:
            raise SystemExit("no instance reached a run")
        env = FloorVecEnv(bridges, config, FloorCurriculum())
        obs = env.reset()

        asked = blocked = 0
        still = 0
        blocked_with_target = blocked_obstructed = 0
        moving_with_target = moving_obstructed = 0
        blocked_at_door = moving_at_door = blocked_door_shut = 0
        blocked_closeness: list[float] = []
        moving_closeness: list[float] = []
        quadrants = np.zeros(4, dtype=np.int64)
        positions: list[tuple[float, float]] = []

        def snapshot():
            out = []
            for index in range(env.num_envs):
                latest = env._latest[index] or {}
                player, room = latest.get("player"), latest.get("room")
                if not player or not room:
                    out.append(None)
                    continue
                out.append((float(player["x"]), float(player["y"]), room))
            return out

        for _ in range(args.steps):
            if policy is None:
                actions = np.random.randint(0, 3, size=(env.num_envs, 4))
            else:
                with torch.no_grad():
                    actions, _, _ = policy.act(
                        to_tensors(obs, torch.device("cpu")))
                actions = actions.numpy()
            before = snapshot()
            obs, _, terminated, truncated, _ = env.step(actions)
            after = snapshot()

            for index in range(env.num_envs):
                if before[index] is None or after[index] is None:
                    continue
                mx, my, _, _ = decode_action(actions[index])
                bx, by, _ = before[index]
                ax, ay, room = after[index]
                travelled = np.hypot(ax - bx, ay - by)
                if mx == 0 and my == 0:
                    still += 1
                    continue
                asked += 1
                is_blocked = travelled < MOVED
                blocked += is_blocked

                # The hypothesis under test: when it cannot move, is something
                # solid sitting on the straight line the shaping is pulling it
                # along? If so the reward is holding it against the obstacle.
                door = shaping_target(room, ax, ay)
                if door is not None:
                    obstructed = line_is_obstructed(room, ax, ay, door)
                    span = max(float(room["bottom_right_x"])
                               - float(room["top_left_x"]), 1.0)
                    closeness = 1.0 - min(np.hypot(float(door["x"]) - ax,
                                                   float(door["y"]) - ay) / span,
                                          1.0)
                    if is_blocked:
                        blocked_with_target += 1
                        blocked_obstructed += obstructed
                        blocked_closeness.append(closeness)
                        blocked_at_door += closeness > 0.9
                        blocked_door_shut += not door.get("open")
                    else:
                        moving_with_target += 1
                        moving_obstructed += obstructed
                        moving_closeness.append(closeness)
                        moving_at_door += closeness > 0.9

                left, top = float(room["top_left_x"]), float(room["top_left_y"])
                span_x = max(float(room["bottom_right_x"]) - left, 1.0)
                span_y = max(float(room["bottom_right_y"]) - top, 1.0)
                nx, ny = (ax - left) / span_x, (ay - top) / span_y
                positions.append((nx, ny))
                quadrants[(0 if nx < 0.5 else 1) + (0 if ny < 0.5 else 2)] += 1

            done = terminated | truncated
            if done.any():
                env.reset_done(done)
                obs = env._stack_observations()

        total = asked + still
        print(f"{total} steps observed")
        print(f"  asked to move:        {asked} ({asked / max(total, 1):.0%})")
        print(f"  stood still by choice:{still} ({still / max(total, 1):.0%})")
        print(f"  ASKED BUT DID NOT MOVE: {blocked} "
              f"({blocked / max(asked, 1):.1%} of steps it tried to move)")

        if positions:
            xs = np.array([p[0] for p in positions])
            ys = np.array([p[1] for p in positions])
            print(f"\n  mean position in the room: x {xs.mean():.2f}, "
                  f"y {ys.mean():.2f}   (0.5, 0.5 would be the centre)")
            names = ["top-left", "top-right", "bottom-left", "bottom-right"]
            order = [0, 1, 2, 3]
            share = quadrants / max(quadrants.sum(), 1)
            print("  time by quadrant:")
            for i in order:
                bar = "#" * int(share[i] * 40)
                print(f"    {names[i]:<13}{share[i]:6.1%}  {bar}")

        if blocked_with_target or moving_with_target:
            hit = blocked_obstructed / max(blocked_with_target, 1)
            base = moving_obstructed / max(moving_with_target, 1)
            print("\n  is something solid on the straight line to the shaping "
                  "target?")
            print(f"    while BLOCKED : {blocked_obstructed}/{blocked_with_target}"
                  f"  ({hit:.1%})")
            print(f"    while moving  : {moving_obstructed}/{moving_with_target}"
                  f"  ({base:.1%})   <- baseline")
            print(f"    ratio         : {hit / max(base, 1e-9):.2f}x")
            # The other way a clear line still ends in a wall: the agent has
            # walked all the way to the door and is pressed against the frame.
            # `door_is_targetable` deliberately does not filter on `open`,
            # because doors shut during a fight and excluding them would
            # collapse the potential the moment combat starts. The cost of that
            # choice is that standing at a shut door is the highest-potential
            # place in the room.
            print("\n  where is it when blocked, relative to that door?")
            print(f"    mean closeness  blocked {np.mean(blocked_closeness):.2f}"
                  f"   moving {np.mean(moving_closeness):.2f}   (1.0 = on it)")
            print(f"    right at the door: blocked "
                  f"{blocked_at_door / max(blocked_with_target, 1):.1%}"
                  f"   moving {moving_at_door / max(moving_with_target, 1):.1%}")
            print(f"    of those blocked at a door, it was SHUT: "
                  f"{blocked_door_shut / max(blocked_with_target, 1):.1%}")
            print("    Well above the baseline means the straight-line potential")
            print("    is holding the agent against obstacles: going around one")
            print("    costs reward, so pressing into it is the highest-potential")
            print("    place to stand. Near the baseline means the shaping is not")
            print("    the reason and the constant downward drive needs another.")

        print("\n  reading: a high 'asked but did not move' means the agent is")
        print("  pinned against geometry, and the corner is where a constant")
        print("  downward drive ends up — not a position it is choosing to hold.")
        print("  A low value means it can move freely and is camping on purpose.")
    finally:
        fleet.shutdown()


if __name__ == "__main__":
    main()
