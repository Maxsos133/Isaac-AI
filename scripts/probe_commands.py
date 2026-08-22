"""Find out which console commands the curriculum can actually rely on.

The whole Phase 2 training distribution is built by issuing console commands to
put the agent in front of real game content. Guessing at command syntax would
mean debugging the curriculum and the learner at the same time, so establish
what works first.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isaac_ai import launcher  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402

config = load_config()
fleet = launcher.bring_up(config, count=1)


def settle(bridge, ticks: int = 20) -> dict:
    """Run a few ticks with neutral input and return the last observation."""
    obs = {}
    for _ in range(ticks):
        obs = bridge.receive()
        bridge.send({"t": "noop"})
    return obs


def describe(obs: dict) -> str:
    if not obs.get("ready", True):
        return "not ready"
    entities = obs.get("entities", [])
    kinds: dict[str, int] = {}
    for entity in entities:
        kinds[entity["k"]] = kinds.get(entity["k"], 0) + 1
    room = obs["room"]
    return (f"stage={obs['level']['stage']} room_type={room['type']} "
            f"clear={room['clear']} entities={len(entities)} {kinds}")


try:
    if fleet.ready_count == 0:
        raise SystemExit("instance did not reach a run")
    bridge = fleet.bridges()[0]

    print("\nbaseline:")
    print("  ", describe(settle(bridge, 30)))

    # Entity type 10 is the generic monster family; 10.0 is Gaper.
    # Syntax under test: `spawn <type>.<variant>.<subtype>`
    commands = [
        ("spawn 10.0", "spawn a Gaper"),
        ("spawn 10.0.0", "spawn a Gaper, fully qualified"),
        ("spawn 212", "spawn a Hopper (type only)"),
        ("goto d.1", "jump to a dedicated room"),
        ("stage 2", "jump to Basement II"),
        ("curse 0", "clear curses"),
        ("debug 3", "toggle infinite HP (would break training if silently on)"),
    ]

    for command, description in commands:
        print(f"\n{command!r} ({description}):")
        bridge.receive()
        bridge.send({"t": "command", "value": command})
        obs = settle(bridge, 40)
        print("  ", describe(obs))

    print("\nspawning a pack of 5 and checking they register as enemies:")
    for _ in range(5):
        bridge.receive()
        bridge.send({"t": "command", "value": "spawn 10.0"})
        bridge.receive()
        bridge.send({"t": "noop"})
    obs = settle(bridge, 40)
    print("  ", describe(obs))
    for entity in obs.get("entities", [])[:6]:
        print(f"     k={entity['k']} type={entity['t']} hp={entity['hp']}/"
              f"{entity['mhp']} dist={entity['d']:.0f}")

    print("\nchecking the room registers as uncleared with enemies present:")
    print("   room clear =", obs["room"]["clear"])
finally:
    fleet.shutdown()
