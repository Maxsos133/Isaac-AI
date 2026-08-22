"""Is standing at a wall and trading hits actually wrong?

Every trained policy on this project converges to the same thing: back onto a
wall, fire across the room, and absorb contact damage rather than avoid it. It
has survived closing-speed features being added and the reward being rebalanced,
and it looks wrong to an experienced player who would loop around a charger
instead.

But it might be right. Isaac's base damage is 3.5, so a gaper costs about three
tears and pays damage_dealt 1.0 plus kill 0.5, against -1.0 for a contact hit —
trading a hit for a kill is profitable. With no items, base speed may also not be
enough to outrun anything worth outrunning.

Arguing this from the reward table has not settled it in three attempts, so this
measures it instead, the way aiming and navigation were settled here: hand-written
controllers as baseline arms, run at fixed difficulty on the same fleet.

  teacher   the trained state-based policy, for reference
  kite      orbit whatever is closing on you, shooting the nearest enemy
  wall      go to the nearest wall and shoot, deliberately — the agent's habit
  random    the floor

All three scripted arms read the same privileged state the teacher does, so the
comparison is between *strategies*, not between perception and strategy.

  kite >> wall   the habit is a local optimum, and the reward or the exploration
                 is what keeps the policy in it
  kite ~= wall   movement is not where the wins are, and the policy is right
  kite << wall   the wall is genuinely correct and this line of attack is closed

    .venv/Scripts/python.exe scripts/diagnose_kiting.py --difficulty 0.4,0.6
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.combat import CombatVecEnv  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.curriculum import CombatCurriculum  # noqa: E402
from isaac_ai.distill import load_teacher  # noqa: E402
from isaac_ai.env import ACTION_DIMS, decode_action  # noqa: E402
from isaac_ai.policy import to_tensors  # noqa: E402

ARMS = ("teacher", "kite", "wall", "random")

# Actions are indices into (-1, 0, 1) per axis.
AXIS_TO_INDEX = {-1: 0, 0: 1, 1: 2}


def encode(mx: int, my: int, sx: int, sy: int) -> list[int]:
    return [AXIS_TO_INDEX[mx], AXIS_TO_INDEX[my],
            AXIS_TO_INDEX[sx], AXIS_TO_INDEX[sy]]


def aim(dx: float, dy: float) -> tuple[int, int]:
    """Isaac fires along axes, so take the dominant component."""
    if abs(dx) >= abs(dy):
        return (1 if dx > 0 else -1), 0
    return 0, (1 if dy > 0 else -1)


def enemies_of(obs: dict) -> list[dict]:
    """What can be shot at."""
    if not obs or not obs.get("ready", True):
        return []
    return [e for e in obs.get("entities", []) if e.get("k") == "enemy"]


def hazards_of(obs: dict) -> list[dict]:
    """What has to be avoided — which is not the same list.

    The first version of this controller dodged enemies only. Against a room of
    shooters that meant orbiting a stationary horf, whose closing speed is
    ~zero, while its bullets went unseen: it stood in a corner and got shot.
    A projectile in flight is the most urgent thing on the screen and was the
    one thing the "kiting" baseline could not see.
    """
    if not obs or not obs.get("ready", True):
        return []
    return [e for e in obs.get("entities", [])
            if e.get("k") in ("enemy", "projectile")]


def closing_speed(entity: dict, player: dict) -> float:
    """Positive when the gap is shrinking — the charger/scenery distinction."""
    dx = entity["x"] - player["x"]
    dy = entity["y"] - player["y"]
    distance = math.hypot(dx, dy) or 1.0
    rvx = entity["vx"] - player["vx"]
    rvy = entity["vy"] - player["vy"]
    return -(rvx * dx + rvy * dy) / distance


def kite_action(obs: dict) -> list[int]:
    """Orbit the most threatening enemy while shooting the nearest.

    Moving *directly* away backs into a wall and ends up as the wall strategy by
    another route. Circling sideways keeps distance without surrendering space,
    and the side chosen is whichever heads back towards the middle of the room —
    which is what looping around an enemy actually looks like.
    """
    targets = enemies_of(obs)
    if not targets:
        return encode(0, 0, 0, 0)

    player, room = obs["player"], obs["room"]
    nearest = min(targets, key=lambda e: e["d"])
    sx, sy = aim(nearest["x"] - player["x"], nearest["y"] - player["y"])

    # Dodge over everything that can hurt, projectiles included, and only
    # consider things actually coming closer — orbiting something that is not
    # approaching wastes the movement and is how this stood in corners.
    hazards = [h for h in hazards_of(obs) if closing_speed(h, player) > 0.5]
    if not hazards:
        return encode(0, 0, sx, sy)

    # Closing fast and already close. Distance floors at one tile so a far-off
    # charger does not outrank something already on top of us.
    threat = max(hazards,
                 key=lambda e: closing_speed(e, player) / max(e["d"], 40.0))
    dx = threat["x"] - player["x"]
    dy = threat["y"] - player["y"]
    distance = math.hypot(dx, dy) or 1.0
    ux, uy = dx / distance, dy / distance

    centre_x = (room["top_left_x"] + room["bottom_right_x"]) / 2.0
    centre_y = (room["top_left_y"] + room["bottom_right_y"]) / 2.0
    to_centre = (centre_x - player["x"], centre_y - player["y"])

    # Both perpendiculars circle the threat; take the one heading inward.
    options = ((-uy, ux), (uy, -ux))
    best = max(options, key=lambda p: p[0] * to_centre[0] + p[1] * to_centre[1])
    # Blend in a little retreat so it opens distance rather than orbiting at
    # contact range, where circling still collects the hit.
    move_x = best[0] - 0.5 * ux
    move_y = best[1] - 0.5 * uy

    mx = 0 if abs(move_x) < 0.35 else (1 if move_x > 0 else -1)
    my = 0 if abs(move_y) < 0.35 else (1 if move_y > 0 else -1)
    return encode(mx, my, sx, sy)


def wall_action(obs: dict) -> list[int]:
    """The agent's habit, written out: sit on the nearest wall and shoot."""
    targets = enemies_of(obs)
    if not targets:
        return encode(0, 0, 0, 0)

    player, room = obs["player"], obs["room"]
    nearest = min(targets, key=lambda e: e["d"])
    sx, sy = aim(nearest["x"] - player["x"], nearest["y"] - player["y"])

    # The room is wider than it is tall and base tear range only spans the short
    # axis, so the top wall is the one that keeps every enemy reachable — which
    # is the wall the policies actually pick.
    target_y = room["top_left_y"] + 40.0
    my = 0 if abs(player["y"] - target_y) < 12.0 else (
        1 if target_y > player["y"] else -1)
    # Slide along the wall to line up with the nearest enemy.
    dx = nearest["x"] - player["x"]
    mx = 0 if abs(dx) < 25.0 else (1 if dx > 0 else -1)
    return encode(mx, my, sx, sy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--difficulty", default="0.3,0.45,0.6")
    parser.add_argument("--teacher", default="runs/combat-v6/policy.pt")
    parser.add_argument("--max-steps", type=int, default=250)
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_teacher(config.root / args.teacher, device)

    fleet = launcher.bring_up(config, count=args.instances)
    try:
        bridges = fleet.bridges()
        if not bridges:
            raise SystemExit("no instance reached a run")

        curriculum = CombatCurriculum(max_enemies=config.combat.max_enemies)
        env = CombatVecEnv(bridges, config, curriculum,
                           max_encounter_steps=args.max_steps)
        arm_of = np.array([i % len(ARMS) for i in range(env.num_envs)])
        if min(int((arm_of == i).sum()) for i in range(len(ARMS))) == 0:
            raise SystemExit(f"need at least {len(ARMS)} instances")
        for arm in ARMS:
            print(f"  {arm:<8} on {int((arm_of == ARMS.index(arm)).sum())} instance(s)")

        rng = np.random.default_rng(0)
        results: dict[float, dict[str, float]] = {}
        damage: dict[float, dict[str, list[float]]] = {}

        for difficulty in (float(x) for x in args.difficulty.split(",")):
            curriculum.difficulty = difficulty
            print(f"\n== difficulty {difficulty:.2f} "
                  f"({len(curriculum.available())} enemy types) ==")
            outcomes = {arm: [] for arm in ARMS}
            hits = {arm: [] for arm in ARMS}
            observation = env.reset()

            while min(len(v) for v in outcomes.values()) < args.episodes:
                with torch.no_grad():
                    logits, _ = teacher(to_tensors(observation, device))
                    teacher_actions = torch.stack(
                        [torch.distributions.Categorical(logits=h).sample()
                         for h in logits], dim=-1).cpu().numpy()

                actions = np.zeros((env.num_envs, len(ACTION_DIMS)), dtype=np.int64)
                for index in range(env.num_envs):
                    arm = ARMS[arm_of[index]]
                    raw = env._latest[index]
                    if arm == "teacher":
                        actions[index] = teacher_actions[index]
                    elif arm == "kite":
                        actions[index] = kite_action(raw)
                    elif arm == "wall":
                        actions[index] = wall_action(raw)
                    else:
                        actions[index] = rng.integers(0, 3, size=len(ACTION_DIMS))

                observation, _, terminated, truncated, infos = env.step(actions)
                done = terminated | truncated
                for index, info in enumerate(infos):
                    if "episode" in info:
                        arm = ARMS[arm_of[index]]
                        if len(outcomes[arm]) < args.episodes:
                            outcomes[arm].append(bool(info["episode"]["success"]))
                if done.any():
                    env.reset_done(done)
                    observation = env._stack_observations()

            results[difficulty] = {a: float(np.mean(v)) for a, v in outcomes.items()}
            for arm in ARMS:
                print(f"  {arm:<8} {np.mean(outcomes[arm]):.2f} "
                      f"({sum(outcomes[arm])}/{len(outcomes[arm])})")

        print(f"\n{'difficulty':<12}" + "".join(f"{a:>10}" for a in ARMS))
        for difficulty, row in results.items():
            print(f"{difficulty:<12.2f}" + "".join(f"{row[a]:>10.2f}" for a in ARMS))

        kite = np.mean([r["kite"] for r in results.values()])
        wall = np.mean([r["wall"] for r in results.values()])
        print()
        if kite > wall + 0.10:
            print(f"VERDICT: kiting beats the wall ({kite:.2f} vs {wall:.2f}) — the "
                  f"policy is in a local optimum, and reward or exploration is why")
        elif wall > kite + 0.10:
            print(f"VERDICT: the wall beats kiting ({wall:.2f} vs {kite:.2f}) — the "
                  f"habit is correct and this line of attack is closed")
        else:
            print(f"VERDICT: no real difference ({kite:.2f} vs {wall:.2f}) — movement "
                  f"is not where the wins are; look elsewhere for the ceiling")
    finally:
        fleet.shutdown()


if __name__ == "__main__":
    main()
