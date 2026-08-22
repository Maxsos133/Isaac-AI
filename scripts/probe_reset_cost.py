"""Is `reseed` a cheaper floor reset than `restart`?

Floor episodes end in a full `restart`, which measured at **0.92s each** and
27% of wall clock across floor-v5 — 1.7 restarts per thousand agent steps, and
every one of them stalls the whole fleet because the harness steps in lockstep.

Workshop "fast reset" mods do not help here: they remove the R-key hold delay,
and the harness sends `Isaac.ExecuteCommand("restart")` from Lua, which never
touches that. The cost is the engine tearing down the run, re-initialising the
character and generating a floor.

`reseed` regenerates the *current* floor without restarting the run, so it
should skip the first two of those. Whether it is actually faster, and whether
it leaves a usable state, is what this measures. Both have to hold:

  faster    fewer ticks to a playable floor than `restart`
  clean     a genuinely different layout, rooms_visited back to the start,
            health restorable, and the player somewhere sane

A reseed that silently keeps the old layout would train the agent on one floor
it can memorise, which is worse than the 27%.

    .venv/Scripts/python.exe scripts/probe_reset_cost.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402

TRIALS = 8
SETTLE_CAP = 300


def observe(bridge) -> dict:
    obs = bridge.receive()
    bridge.send({"t": "noop"})
    return obs


def fingerprint(obs: dict) -> tuple:
    """Enough of the floor to tell one layout from another."""
    room = obs.get("room") or {}
    level = obs.get("level") or {}
    return (level.get("stage"), level.get("rooms_total"),
            room.get("index"), len(room.get("doors") or []))


def settle(bridge, before: tuple, heal: bool) -> tuple[int, dict]:
    """Tick until the floor looks different, returning how long it took."""
    for tick in range(SETTLE_CAP):
        obs = bridge.receive()
        if heal and tick == 0:
            # Reseed keeps the player's health, so restore it the way an
            # encounter does rather than pretending the floor is fresh.
            bridge.send({"t": "scenario", "enemies": [], "heal": True,
                         "reposition": False, "new_room": False})
            continue
        bridge.send({"t": "noop"})
        if not obs.get("ready", True):
            continue
        if fingerprint(obs) != before and obs.get("room"):
            return tick + 1, obs
    return SETTLE_CAP, obs


def run(bridge, command: dict, heal: bool) -> tuple[float, int, dict, tuple]:
    before_obs = observe(bridge)
    before = fingerprint(before_obs)
    started = time.perf_counter()
    bridge.receive()
    bridge.send(command)
    ticks, obs = settle(bridge, before, heal)
    return time.perf_counter() - started, ticks, obs, before


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    config = load_config()
    fleet = launcher.bring_up(config, count=1)
    try:
        bridges = fleet.bridges()
        if not bridges:
            raise SystemExit("instance did not reach a run")
        bridge = bridges[0]
        for _ in range(30):
            bridge.pump()

        results: dict[str, list[float]] = {}
        layouts: dict[str, set] = {}
        health: dict[str, list[int]] = {}

        for name, command, heal in (("restart", {"t": "reset"}, False),
                                    ("reseed", {"t": "command",
                                                "value": "reseed"}, True)):
            times, ticks_taken = [], []
            layouts[name] = set()
            health[name] = []
            print(f"\n== {name} ==")
            for trial in range(TRIALS):
                seconds, ticks, obs, before = run(bridge, command, heal)
                times.append(seconds)
                ticks_taken.append(ticks)
                after = fingerprint(obs)
                layouts[name].add(after)
                player = obs.get("player") or {}
                health[name].append(int(player.get("hearts", 0)))
                level = obs.get("level") or {}
                print(f"  trial {trial}: {seconds:5.2f}s  {ticks:3d} ticks  "
                      f"stage {level.get('stage')}  rooms {level.get('rooms_total')}  "
                      f"visited {level.get('rooms_visited')}  "
                      f"hearts {player.get('hearts')}"
                      + ("   LAYOUT UNCHANGED" if after == before else ""))
            results[name] = times
            # Median, not mean. The fingerprint below can collide between two
            # genuinely different floors, and when it does the settle loop runs
            # to its cap and contributes a ten-second outlier that swamps eight
            # real measurements. The failure is in detecting the change, not in
            # the reset being measured.
            print(f"  median {statistics.median(times):.2f}s "
                  f"({statistics.median(ticks_taken):.0f} ticks), "
                  f"{sum(1 for t in ticks_taken if t >= SETTLE_CAP)} undetected")

        print(f"\n{'':<10}{'seconds':>10}{'distinct layouts':>20}{'hearts':>10}")
        for name in results:
            print(f"{name:<10}{statistics.median(results[name]):>10.2f}"
                  f"{len(layouts[name]):>20}{statistics.mean(health[name]):>10.1f}")

        gain = (statistics.median(results["restart"])
                - statistics.median(results["reseed"]))
        print()
        if len(layouts["reseed"]) < TRIALS // 2:
            print("REJECT: reseed is not producing distinct floors — the agent "
                  "would memorise one layout, which costs more than the 27%.")
        elif gain > 0.2:
            print(f"reseed saves {gain:.2f}s a reset. At 1.7 restarts per 1000 "
                  f"steps that is roughly {gain * 1.7 / 10:.0%} of wall clock back.")
        else:
            print(f"reseed saves only {gain:.2f}s — not worth the change; the "
                  f"cost is floor generation either way.")
    finally:
        fleet.shutdown()


if __name__ == "__main__":
    main()
