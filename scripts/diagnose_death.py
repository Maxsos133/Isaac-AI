"""Watch exactly what one instance reports around death and restart.

Death is the most common episode boundary, so if `restart` does not recover
from the game-over screen the whole training loop stalls there.
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
if fleet.ready_count == 0:
    print("instance did not reach a run")
    fleet.shutdown()
    raise SystemExit(1)

bridge = fleet.bridges()[0]
print("\nstepping until death...")

obs = bridge.receive()
ready_flips = 0
last_ready = obs.get("ready", True)
died_at = None

for step in range(4000):
    # Stand still and never shoot: in hard mode this dies quickly.
    bridge.send({"t": "act", "mx": 0, "my": 0, "sx": 0, "sy": 0})
    obs = bridge.receive()

    ready = obs.get("ready", True)
    if ready != last_ready:
        ready_flips += 1
        print(f"  step {step}: ready {last_ready} -> {ready}")
        last_ready = ready

    events = obs.get("events", {})
    if events.get("died"):
        print(f"  step {step}: EVENT died=True")
    if ready and obs.get("player", {}).get("is_dead"):
        if died_at is None:
            died_at = step
            hearts = obs["player"]["hearts"]
            print(f"  step {step}: player.is_dead=True hearts={hearts}")
            break
    if ready and step % 200 == 0:
        print(f"  step {step}: hearts={obs['player']['hearts']} "
              f"entities={len(obs.get('entities', []))}")

if died_at is None:
    print("player never died; skipping restart test")
    fleet.shutdown()
    raise SystemExit(0)

print("\nissuing restart from the death state...")
bridge.send({"t": "reset"})
started = time.perf_counter()
recovered_at = None

for step in range(900):
    obs = bridge.receive()
    events = obs.get("events", {})
    if events.get("game_started"):
        recovered_at = step
        print(f"  recovered after {step} ticks "
              f"({time.perf_counter() - started:.1f}s), "
              f"is_continued={events.get('is_continued')}")
        break
    bridge.send({"t": "noop"})

if recovered_at is None:
    print(f"  NO RECOVERY after 900 ticks ({time.perf_counter() - started:.1f}s)")
    print("  last observation ready flag:", obs.get("ready"))

fleet.shutdown()
