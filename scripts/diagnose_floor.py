"""What is the floor policy actually doing?

Two very different failures produce the same near-zero rooms-cleared number:

  passive  - it barely moves, episodes end on the idle timeout
  reckless - it explores and dies

They need opposite fixes, so measure which one it is: distance travelled, how
episodes end, and how far the agent gets before they do.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.floor_curriculum import FloorCurriculum  # noqa: E402
from isaac_ai.floors import FloorVecEnv  # noqa: E402
from isaac_ai.policy import ActorCritic, to_tensors  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--instances", type=int, default=4)
parser.add_argument("--episodes", type=int, default=16)
parser.add_argument("--checkpoint", type=str, default="runs/floor-v1/policy.pt")
args = parser.parse_args()

config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = ActorCritic().to(device)
checkpoint = config.root / args.checkpoint
policy.load_state_dict(torch.load(checkpoint, map_location=device)["policy"])
policy.eval()
print(f"loaded {checkpoint}")

fleet = launcher.bring_up(config, count=args.instances)
try:
    if fleet.ready_count == 0:
        raise SystemExit("no instances reached a run")

    env = FloorVecEnv(fleet.bridges(), config, FloorCurriculum())
    observation = env.reset()

    episodes = 0
    travelled = np.zeros(env.num_envs)
    last_pos: list[tuple[float, float] | None] = [None] * env.num_envs
    stationary_steps = 0
    total_steps = 0
    ends = {"died": 0, "idle_or_timeout": 0, "reached_target": 0}
    seen_rooms, cleared_rooms, lengths = [], [], []

    while episodes < args.episodes:
        with torch.no_grad():
            actions, _, _ = policy.act(to_tensors(observation, device))
        observation, _, terminated, truncated, infos = env.step(actions.cpu().numpy())
        total_steps += env.num_envs

        for index in range(env.num_envs):
            raw = env._latest[index]
            if not raw or not raw.get("ready", True):
                continue
            pos = (raw["player"]["x"], raw["player"]["y"])
            if last_pos[index] is not None:
                delta = math.dist(pos, last_pos[index])
                travelled[index] += delta
                if delta < 1.0:
                    stationary_steps += 1
            last_pos[index] = pos

        for index, info in enumerate(infos):
            if "episode" not in info:
                continue
            episodes += 1
            ep = info["episode"]
            seen_rooms.append(ep["rooms_seen"])
            cleared_rooms.append(ep["rooms_cleared"])
            lengths.append(ep["l"])
            if ep["success"]:
                ends["reached_target"] += 1
            elif terminated[index]:
                ends["died"] += 1
            else:
                ends["idle_or_timeout"] += 1
            travelled[index] = 0.0
            last_pos[index] = None

        done = terminated | truncated
        if done.any():
            env.reset_done(done)
            observation = env._stack_observations()

    print(f"\n=== {episodes} episodes ===")
    print(f"  rooms entered  mean {np.mean(seen_rooms):.2f}  max {max(seen_rooms)}")
    print(f"  rooms cleared  mean {np.mean(cleared_rooms):.2f}  max {max(cleared_rooms)}")
    print(f"  episode length mean {np.mean(lengths):.0f} steps")
    print(f"  stationary     {stationary_steps / max(total_steps,1):.1%} of steps "
          f"(player moved <1px)")
    print("\n  how episodes ended:")
    for name, count in ends.items():
        print(f"    {name:18s} {count:3d}  ({count / max(episodes,1):.0%})")

    print()
    if ends["idle_or_timeout"] > ends["died"]:
        print("VERDICT: passive. It is running out the clock rather than dying —")
        print("  doing nothing scores better than exploring, so the reward balance")
        print("  and the lack of guidance toward doors are the problem.")
    else:
        print("VERDICT: reckless. It explores and dies, so the issue is survival")
        print("  on real floors rather than motivation to move.")
finally:
    fleet.shutdown()
