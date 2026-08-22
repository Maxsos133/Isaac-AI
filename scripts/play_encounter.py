"""Put a human on the encounters the agent is scored on.

Three learners have now stopped in the same place: combat-v4's curriculum
plateaued around difficulty 0.57, distill-v3's around 0.45, and RL fine-tuning
moved it 0.03 in a million steps. The teacher — which sees exact enemy
positions and velocities and had a million steps of PPO — wins about half its
fights at difficulty 0.5 and half at 0.6.

When every player stops at the same wall, the wall is more likely in the task
than in the players. At difficulty 0.5 the encounter is five mixed enemies
spawned at once at minimum distance, against base Isaac — three hearts, damage
3.5, no items — on a Hard save. Real Isaac never asks for that.

This measures the only baseline that settles it. Same curriculum, same spawner,
same win condition, a person at the keyboard. If an experienced player wins
comfortably, the ceiling belongs to the agent and the agent is what to fix. If
they do not, the task is unfair and no amount of training will show otherwise.

The window must have focus for the keyboard to reach the game. One instance
runs at Isaac's own 30Hz, so it plays at normal speed, and it is resized to
something a person can actually see.

**Do not pause the game.** The mod blocks inside MC_POST_UPDATE waiting for this
script, and a paused game never reaches that callback, so the bridge goes silent
and gives up after 45 seconds. Same for the game-over screen, which is why a
death here is caught and restarted immediately rather than waited on.

    .venv/Scripts/python.exe scripts/play_encounter.py --difficulty 0.5
    .venv/Scripts/python.exe scripts/play_encounter.py --difficulty 0.4 --episodes 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher, windows  # noqa: E402
from isaac_ai.combat import SPAWN_SETTLE_TICKS  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402
from isaac_ai.curriculum import CombatCurriculum  # noqa: E402
from isaac_ai.env import MAX_RESET_TICKS, RESET_REISSUE_TICKS  # noqa: E402

# Matches CombatVecEnv: the encounter is conceded at this many half-hearts
# rather than played to death, so a loss costs a few ticks instead of a floor
# regeneration. The agent is scored the same way, so the human must be too.
DEFEAT_HEARTS = 1

# The training window is 496x309 because twelve of them have to tile. Exactly
# one runs here and a person has to see it.
PLAY_CLIENT = (1280, 720)
# Chrome carried by the outer window beyond the client area, measured by
# probe_capture: 16px wide, 39px of title bar and frame.
CHROME = (16, 39)


def hearts(obs: dict) -> int:
    player = obs["player"]
    return int(player["hearts"]) + int(player["soul_hearts"])


def restart(bridge) -> None:
    """Rebuild the run after a death, before the game-over screen settles.

    A dead player stops running mod callbacks, so the bridge goes silent and
    times out. The reset has to be issued the moment death is seen rather than
    waited on, and re-issued periodically because `restart` can be swallowed if
    it lands on a tick where the console is not accepting commands.
    """
    bridge.send({"t": "reset"})
    for tick in range(MAX_RESET_TICKS):
        obs = bridge.receive()
        if obs.get("events", {}).get("game_started"):
            bridge.send({"t": "human"})
            return
        reissue = tick > 0 and tick % RESET_REISSUE_TICKS == 0
        bridge.send({"t": "reset"} if reissue else {"t": "human"})
    print("  warning: never saw a new run after the restart")


def play_one(bridge, curriculum: CombatCurriculum,
             max_steps: int) -> tuple[bool, int]:
    """Spawn one encounter and hand the keyboard over until it resolves.

    Every receive is answered by exactly one send: the instance is blocked on
    that reply, and skipping one stalls the game rather than the script.
    """
    encounter = curriculum.sample()

    bridge.receive()
    # `heal` is set by to_command, so each encounter starts at full health and
    # they do not compound.
    bridge.send(encounter.to_command())
    for _ in range(SPAWN_SETTLE_TICKS):
        bridge.receive()
        bridge.send({"t": "human"})

    for step in range(max_steps):
        obs = bridge.receive()

        if not obs.get("ready", True):
            bridge.send({"t": "human"})
            continue

        if obs["player"]["is_dead"]:
            restart(bridge)
            return False, step
        if hearts(obs) <= DEFEAT_HEARTS:
            bridge.send({"t": "human"})
            return False, step
        if obs["room"].get("enemies_alive", 0) == 0:
            bridge.send({"t": "human"})
            return True, step

        bridge.send({"t": "human"})

    return False, max_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=900,
                        help="tick cap per encounter (~30s at 30Hz)")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    config = load_config()
    curriculum = CombatCurriculum(max_enemies=config.combat.max_enemies,
                                  difficulty=args.difficulty)

    span = curriculum.max_enemies - curriculum.min_enemies
    count = curriculum.min_enemies + round(span * args.difficulty)
    roster = curriculum.available()

    fleet = launcher.bring_up(config, count=1)
    try:
        bridges = fleet.bridges()
        if not bridges:
            raise SystemExit("instance did not reach a run")
        bridge = bridges[0]

        instance = fleet.instances[0]
        if instance.hwnd:
            # Async, and pumped afterwards: MoveWindow blocks forever against
            # an instance sitting in MC_POST_UPDATE, which this one is.
            windows.move_window_async(instance.hwnd, 40, 40,
                                      PLAY_CLIENT[0] + CHROME[0],
                                      PLAY_CLIENT[1] + CHROME[1])
            for _ in range(30):
                bridge.pump()

        print(f"\ndifficulty {args.difficulty:.2f}: {count} enemies drawn from "
              f"{len(roster)} types")
        print(f"  {', '.join(k.name for k in roster)}")
        print(f"\nsave: {config.save.snapshot.name} (base stats, no items)")
        print(f"an encounter is lost at {DEFEAT_HEARTS} half-heart, exactly as "
              f"the agent's is")
        print("\nCLICK THE ISAAC WINDOW so it has keyboard focus.")
        print("DO NOT PAUSE — a paused game stops answering and the run ends.")
        for remaining in range(5, 0, -1):
            print(f"  starting in {remaining}...")
            # The instance is blocked between ticks, so it has to be kept
            # running while the countdown plays or the window freezes.
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                bridge.pump()

        wins = 0
        for episode in range(1, args.episodes + 1):
            won, steps = play_one(bridge, curriculum, args.max_steps)
            wins += won
            print(f"  encounter {episode:2d}/{args.episodes}: "
                  f"{'WON ' if won else 'lost'} after {steps} ticks "
                  f"({steps / 30:.1f}s)   running {wins}/{episode}")

        rate = wins / args.episodes
        print(f"\nhuman at difficulty {args.difficulty:.2f}: "
              f"{rate:.2f} ({wins}/{args.episodes})")
        print("\ncompare (24 encounters each, same spawner):")
        print(f"{'difficulty':<12}{'teacher':>9}{'student':>9}")
        for level, teacher, student in ((0.30, 0.92, 0.88), (0.40, 0.83, 0.54),
                                        (0.50, 0.54, 0.54), (0.60, 0.50, 0.21)):
            marker = "  <- you played here" if abs(level - args.difficulty) < 0.01 else ""
            print(f"{level:<12.2f}{teacher:>9.2f}{student:>9.2f}{marker}")
    finally:
        fleet.shutdown()


if __name__ == "__main__":
    main()
