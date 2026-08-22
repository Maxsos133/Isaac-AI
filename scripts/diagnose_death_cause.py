"""What actually kills the agent?

Deaths are 67% of episodes on floor-v19b and `damage_taken` is the largest
single term in the reward, and nothing in this project has ever reported what
the damage came *from*.

That gap matters now specifically. Every hypothesis a run has been spent on is
fixed or best-ever — dead action axes rescued, enemy response at 5 distinct
placements and 62% horizontal alignment (both project records), stranding
`exhausted` 0.326 -> 0.093, wall-grinding `blocked_rate` 0.114 -> 0.075 — and
`rooms_cleared` has not moved off 0.29. The one measure that has never moved is
obstacle use: `blocking -> irrelevant` sits at 0.0078 against an untrained floor
of 0.0001.

So the next run turns on whether the agent cannot *fight* or cannot *avoid
hazards*, and this measures that instead of arguing it. The mod now classifies
every hit on the player from the game's own `DamageFlag` bits and the source
`EntityRef`; this plays the fleet under a trained policy and reports:

  killing blow    the category of the most recent damaging tick before death,
                  as a share of deaths
  damage split    total damage by category, as a share of all damage taken
  hazard deaths   deaths whose killing blow was environmental rather than an
                  enemy or a projectile — the number the obstacle question
                  actually rests on

**A probe that returns nothing is not a pass.** Zero deaths or zero damage means
the sample is too small or the mapping is broken, and this exits non-zero on
either rather than printing a clean-looking table of zeros.

    .venv/Scripts/python.exe scripts/diagnose_death_cause.py
    .venv/Scripts/python.exe scripts/diagnose_death_cause.py \
        --policy runs/floor-v19b/policy.pt --steps 3000
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.floor_curriculum import FloorCurriculum  # noqa: E402
from isaac_ai.floors import FloorVecEnv  # noqa: E402
from isaac_ai.policy import ActorCritic, to_tensors  # noqa: E402

# Must match DAMAGE_CATEGORIES in mod/isaac_ai/main.lua. Listed here rather than
# derived from the payload so that a category the mod stops sending shows up as
# a missing key instead of silently vanishing from the report.
CATEGORIES = ("enemy", "projectile", "spikes", "fire", "explosion", "creep",
              "other")
# Everything the room does to you rather than something aiming at you. This is
# the split the obstacle question rests on, so it is named once here rather than
# re-derived at each print.
ENVIRONMENTAL = ("spikes", "fire", "explosion", "creep")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--policy", default="runs/floor-v19b/policy.pt")
    parser.add_argument("--random", action="store_true",
                        help="random actions instead of the policy, as a control")
    args = parser.parse_args()

    config = load_config()
    count = args.instances or config.instances.count

    policy = None
    if not args.random:
        checkpoint = torch.load(args.policy, map_location="cpu")
        policy = ActorCritic()
        policy.load_state_dict(checkpoint["policy"])
        policy.eval()
        print(f"policy {args.policy} "
              f"({checkpoint.get('global_step', 0):,} steps)")
    else:
        print("random actions (control)")
    print(f"bringing up {count} instance(s)\n")

    fleet = launcher.bring_up(config, count=count)
    problems: list[str] = []
    try:
        bridges = fleet.bridges()
        if not bridges:
            raise SystemExit("no instance reached a run")
        env = FloorVecEnv(bridges, config, FloorCurriculum())
        observation = env.reset()

        # A category absent from the payload is not the same as one reported
        # zero: the first means the mod never wrote it, which is exactly the
        # failure that made a hand-written chest list read as "no chests exist".
        sample = (env._latest[0] or {}).get("events", {}).get("damage_by")
        if sample is None:
            raise SystemExit(
                "events.damage_by absent from the payload — redeploy the mod "
                "(python -m isaac_ai deploy)")
        missing = [c for c in CATEGORIES if c not in sample]
        if missing:
            problems.append(f"categories absent from the payload: {missing}")
        print(f"payload damage_by keys: {sorted(sample)}\n")

        # Per-instance running totals for the episode currently in progress.
        episode_damage = [Counter() for _ in range(env.num_envs)]
        last_category = [None] * env.num_envs
        # `is_dead` is true for the whole death animation and a restart can take
        # up to MAX_RESET_TICKS, so counting it per tick would multiply every
        # death by however many observations the reset spanned. Count the edge.
        was_dead = [False] * env.num_envs

        killing_blow: Counter = Counter()
        damage_total: Counter = Counter()
        deaths = 0
        deaths_without_attribution = 0

        for step in range(args.steps):
            if policy is not None:
                with torch.no_grad():
                    actions, _, _ = policy.act(
                        to_tensors(observation, torch.device("cpu")))
                actions = actions.numpy()
            else:
                actions = np.random.randint(0, 3, size=(env.num_envs, 4))

            observation, _, terminated, truncated, _ = env.step(actions)

            for index in range(env.num_envs):
                latest = env._latest[index] or {}
                events = latest.get("events", {})
                by = events.get("damage_by") or {}
                tick_total = 0.0
                for category, amount in by.items():
                    amount = float(amount or 0.0)
                    if amount <= 0.0:
                        continue
                    episode_damage[index][category] += amount
                    damage_total[category] += amount
                    tick_total += amount
                if tick_total > 0.0:
                    # The category that took the most of this tick's damage is
                    # the one credited, so a tick that mixes a contact hit and a
                    # spike is attributed to whichever actually did the harm.
                    last_category[index] = max(
                        (c for c, a in by.items() if float(a or 0.0) > 0.0),
                        key=lambda c: float(by[c] or 0.0))

                dead = bool(latest.get("player", {}).get("is_dead"))
                if dead and not was_dead[index]:
                    deaths += 1
                    if last_category[index] is None:
                        deaths_without_attribution += 1
                    else:
                        killing_blow[last_category[index]] += 1
                    episode_damage[index] = Counter()
                    last_category[index] = None
                was_dead[index] = dead

            # Isaac stops running mod callbacks once the game-over screen takes
            # over, so an instance left to reach it can never answer again — and
            # `_receive_all` reads sequentially on a blocking socket, so the
            # whole fleet stalls behind it for the full timeout. Anything that
            # steps this env must reset on the done mask.
            done = terminated | truncated
            if done.any():
                # `reset_done` returns None; the fresh observation has to be
                # restacked afterwards, exactly as the trainer does it.
                env.reset_done(done)
                observation = env._stack_observations()

            if step and step % 500 == 0:
                print(f"  step {step}: {deaths} deaths, "
                      f"{sum(damage_total.values()):.0f} damage attributed")

        total_damage = sum(damage_total.values())
        print(f"\n{deaths} deaths over {args.steps} steps at {env.num_envs} "
              f"instances, {total_damage:.0f} total damage\n")

        if deaths == 0:
            problems.append("no deaths recorded — sample too small to say "
                            "anything; raise --steps")
        if total_damage <= 0.0:
            problems.append("no damage attributed at all — the mapping is "
                            "broken, not the sample")

        if total_damage > 0.0:
            print(f"  {'category':<12}{'damage':>10}{'share':>9}"
                  f"{'killing blow':>14}{'share':>8}")
            for category in CATEGORIES:
                dmg = damage_total.get(category, 0.0)
                kills = killing_blow.get(category, 0)
                print(f"  {category:<12}{dmg:>10.0f}"
                      f"{dmg / total_damage:>8.1%}"
                      f"{kills:>14}"
                      f"{(kills / deaths if deaths else 0):>8.1%}")

            env_damage = sum(damage_total.get(c, 0.0) for c in ENVIRONMENTAL)
            env_kills = sum(killing_blow.get(c, 0) for c in ENVIRONMENTAL)
            print(f"\n  environmental ({'/'.join(ENVIRONMENTAL)}):")
            print(f"    {env_damage / total_damage:.1%} of damage taken")
            if deaths:
                print(f"    {env_kills / deaths:.1%} of deaths")
            print("\n  reading: if environmental is a small share, the agent "
                  "dies to\n  enemies it cannot fight and obstacle perception "
                  "is not the lever.\n  If it is large, the room is killing it "
                  "and the grid is.")

        if deaths_without_attribution:
            # A death with no damage recorded before it is not a tidy zero: it
            # means the fatal hit arrived on a tick this loop never saw, so the
            # killing-blow shares are computed over a smaller base than `deaths`.
            problems.append(
                f"{deaths_without_attribution} of {deaths} deaths had no "
                f"attributed damage beforehand — killing-blow shares are over "
                f"the remainder, not over all deaths")

    finally:
        fleet.shutdown()

    log = Path(config.game.savedata_dir) / "log.txt"
    if log.exists():
        warnings = [line.strip() for line in
                    log.read_text(encoding="utf-8", errors="replace").splitlines()
                    if "unresolved damage flags" in line]
        if warnings:
            problems.append(f"mod reported unresolved flags: {warnings[-1]}")
        else:
            print("\nno unresolved-damage-flag warning in log.txt")

    if problems:
        print("\nPROBLEM:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
