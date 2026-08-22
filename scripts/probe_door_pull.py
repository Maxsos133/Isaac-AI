"""Does door shaping actually pull while a fight is happening?

`diagnose_shoot_axis.py` reports that removing the doors from the observation
takes floor-v21 from 12/32 firing correctly and 3 distinct move responses to
**28/32 and 6** — which reads as the door signal drowning combat behaviour that
is already there. Before spending a run on that, the premise has to hold in real
play, and there is good reason to think it might not.

`door_is_targetable` skips locked doors, and Isaac locks the doors when you enter
a room with live enemies. If combat-barred doors report `locked`, then
`door_potential` already returns 0.0 for the whole fight, there is no pull to
compete with combat, and the probe's state -- an enemy *and* unvisited unlocked
doors -- is a synthetic combination that never occurs. That is precisely the trap
`probe_heads.py` fell into when it synthesised a doorless room and reported a
floor policy's move heads as abandoned.

So this measures, on live data:

  P(targetable door | enemies alive)   the state the shoot-axis probe assumes
  mean door_potential, fighting vs not the pull that actually competes
  share of steps in each regime

If the potential is ~0 whenever enemies are alive, the door-pull hypothesis is a
probe artefact and the shoot-axis control rows say nothing about real play.

Imports `door_is_targetable` and `door_potential` rather than restating them:
this project has already shipped a probe that kept its own copy of that filter
and went on reporting a fixed bug as present.

    .venv/Scripts/python.exe scripts/probe_door_pull.py
    .venv/Scripts/python.exe scripts/probe_door_pull.py --steps 1500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.floor_curriculum import FloorCurriculum  # noqa: E402
from isaac_ai.floors import (FloorVecEnv, door_is_targetable,  # noqa: E402
                             door_potential)
from isaac_ai.policy import ActorCritic, to_tensors  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--policy", default="runs/floor-v21/policy.pt")
    args = parser.parse_args()

    config = load_config()
    count = args.instances or config.instances.count

    checkpoint = torch.load(args.policy, map_location="cpu")
    policy = ActorCritic()
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    print(f"policy {args.policy} ({checkpoint.get('global_step', 0):,} steps)")
    print(f"bringing up {count} instance(s)\n")

    fleet = launcher.bring_up(config, count=count)
    problems: list[str] = []
    try:
        bridges = fleet.bridges()
        if not bridges:
            raise SystemExit("no instance reached a run")
        env = FloorVecEnv(bridges, config, FloorCurriculum())
        observation = env.reset()

        fighting = quiet = 0
        fighting_with_target = quiet_with_target = 0
        fighting_potential = quiet_potential = 0.0
        # Locked is the mechanism under test: if combat bars doors as "locked",
        # the filter already suppresses them and the pull cannot exist.
        doors_seen = doors_locked = doors_shut = 0

        for step in range(args.steps):
            with torch.no_grad():
                actions, _, _ = policy.act(
                    to_tensors(observation, torch.device("cpu")))
            observation, _, terminated, truncated, _ = env.step(actions.numpy())

            for index in range(env.num_envs):
                latest = env._latest[index] or {}
                room = latest.get("room") or {}
                if not latest.get("ready", True) or not room:
                    continue
                enemies = int(room.get("enemies_alive", 0) or 0)
                targetable = any(door_is_targetable(d)
                                 for d in room.get("doors", []))
                potential = door_potential(latest)

                for door in room.get("doors", []):
                    doors_seen += 1
                    if door.get("locked"):
                        doors_locked += 1
                    if not door.get("open", True):
                        doors_shut += 1

                if enemies > 0:
                    fighting += 1
                    fighting_with_target += targetable
                    fighting_potential += potential
                else:
                    quiet += 1
                    quiet_with_target += targetable
                    quiet_potential += potential

            done = terminated | truncated
            if done.any():
                env.reset_done(done)
                observation = env._stack_observations()

            if step and step % 500 == 0:
                print(f"  step {step}: {fighting} fighting, {quiet} quiet")

        total = fighting + quiet
        print(f"\n{total} observations  ({fighting} with live enemies, "
              f"{quiet} without)\n")
        if total == 0:
            raise SystemExit("no observations — nothing to report")

        print(f"  {'regime':<12}{'steps':>8}{'share':>8}"
              f"{'has targetable door':>22}{'mean potential':>16}")
        for name, n, tgt, pot in (
                ("fighting", fighting, fighting_with_target, fighting_potential),
                ("quiet", quiet, quiet_with_target, quiet_potential)):
            if n == 0:
                print(f"  {name:<12}{0:>8}{'-':>8}{'-':>22}{'-':>16}")
                continue
            print(f"  {name:<12}{n:>8}{n / total:>7.1%}"
                  f"{tgt / n:>21.1%}{pot / n:>16.3f}")

        print(f"\n  doors seen {doors_seen}, locked {doors_locked} "
              f"({doors_locked / max(doors_seen, 1):.1%}), "
              f"shut {doors_shut} ({doors_shut / max(doors_seen, 1):.1%})")

        if fighting == 0:
            problems.append("no steps with live enemies — cannot answer the "
                            "question; raise --steps")
        else:
            share = fighting_with_target / fighting
            print(f"\n  reading: P(targetable door | fighting) = {share:.1%}, "
                  f"mean potential {fighting_potential / fighting:.3f}")
            if share < 0.10:
                print("  -> the pull is already suppressed during fights. The "
                      "shoot-axis\n     control rows are a synthetic state and "
                      "say nothing about real play.")
            else:
                print("  -> door shaping is live during fights and does compete "
                      "with combat.\n     The shoot-axis finding stands.")

    finally:
        fleet.shutdown()

    if problems:
        print("\nPROBLEM:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
