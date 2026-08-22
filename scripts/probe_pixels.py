"""What the production capture path costs, and whether it produces real data.

`probe_capture.py` measured the capture *mechanism* — reading the newest frame
out of a slot, which is nearly free. This measures the path a training step
actually walks: crop to the client area, downsample to the student's input,
roll it into the frame stack, for every instance, every step. That is real work
on the main thread and it was not in the earlier number.

Three things can be wrong here without the throughput looking wrong at all:

  distinct   every grabber could be pointed at the same window, and twelve
             copies of one instance would train perfectly happily
  motion     the stack could hold four copies of one frame, in which case the
             student has no velocity information and nobody would notice
  cost       the downsample could eat the fleet

    .venv/Scripts/python.exe scripts/probe_pixels.py
    .venv/Scripts/python.exe scripts/probe_pixels.py --instances 2 --ticks 60
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from isaac_ai import launcher  # noqa: E402
from isaac_ai.capture import FleetCapture  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "spike" / "capture-probe"

# Every instance starts in an identical room, so they have to be driven apart
# before "are these the same picture" means anything. A fixed direction per
# instance is not enough: with twelve instances and four directions, three of
# them walk the same way and land in the same place, which reads as a capture
# bug that is not there. Each instance walks its own seeded random path instead.
WALK = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1))

# A visible change, in 0-255. Comparing mean absolute difference over the whole
# frame instead would call real movement "identical": the player is about 0.3%
# of a 160x90 frame, so walking across the room moves the frame mean by 0.03.
# What matters is whether *any* pixels changed, not the average over all of them.
VISIBLE = 8


def act(bridges: list, ticks: int, diverge: bool) -> None:
    for tick in range(ticks):
        for bridge in bridges:
            bridge.receive()
        for index, bridge in enumerate(bridges):
            if diverge:
                # Deterministic per instance, so a rerun is comparable.
                rng = np.random.default_rng(index * 977 + tick // 12)
                mx, my = WALK[int(rng.integers(len(WALK)))]
            else:
                mx, my = 0, 0
            bridge.send({"t": "act", "mx": mx, "my": my, "sx": 0, "sy": 0,
                         "bomb": False, "item": False})


def changed_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Share of pixels that visibly differ between two frames."""
    return float((np.abs(a.astype(np.int16) - b.astype(np.int16))
                  > VISIBLE).mean())


def probe_cost(bridges: list, fleet_capture: FleetCapture, ticks: int,
               action_repeat: int) -> None:
    print(f"\n== cost: {len(bridges)} instances, {ticks} ticks ==")

    started = time.perf_counter()
    act(bridges, ticks, diverge=False)
    baseline = len(bridges) * ticks / (time.perf_counter() - started)
    print(f"  no capture:          {baseline:7.1f} game ticks/s")

    observe_ms: list[float] = []
    started = time.perf_counter()
    for tick in range(ticks):
        for bridge in bridges:
            bridge.receive()
        # One observation per agent step, not per game tick — that is the rate
        # a trainer would actually pull frames at.
        if tick % action_repeat == 0:
            at = time.perf_counter()
            fleet_capture.observe()
            observe_ms.append((time.perf_counter() - at) * 1000.0)
        for bridge in bridges:
            bridge.send({"t": "act", "mx": 0, "my": 0, "sx": 0, "sy": 0,
                         "bomb": False, "item": False})
    withcap = len(bridges) * ticks / (time.perf_counter() - started)

    print(f"  full capture path:   {withcap:7.1f} game ticks/s "
          f"({100 * (withcap - baseline) / baseline:+.1f}%)")
    if observe_ms:
        print(f"  observe() per step:  p50 {statistics.median(observe_ms):.2f}ms  "
              f"p95 {sorted(observe_ms)[int(0.95 * len(observe_ms))]:.2f}ms  "
              f"for all {len(bridges)} instances")
    print(f"  => {withcap / action_repeat:.0f} agent steps/s, "
          f"{withcap / action_repeat * 3600:,.0f} per hour")

    # A step that gets a frame it has already seen is not an error, but it is
    # a step whose stack carries no new motion. If this is high, capture is not
    # keeping up with stepping and the student is looking at stale pictures.
    repeats = sum(g.repeats for g in fleet_capture.grabbers if g is not None)
    served = len(observe_ms) * len([g for g in fleet_capture.grabbers if g])
    if served:
        print(f"  repeated frames:     {repeats}/{served} "
              f"({repeats / served:.1%} of reads had no new frame)")


def probe_distinct(bridges: list, fleet_capture: FleetCapture) -> None:
    """Twelve copies of one instance would train, and look fine doing it."""
    print("\n== distinct: each instance is its own window ==")

    hwnds = [g.hwnd for g in fleet_capture.grabbers if g is not None]
    print(f"  distinct hwnds: {len(set(hwnds))}/{len(hwnds)}")

    # Walk them in different directions so identical starting rooms diverge.
    act(bridges, 60, diverge=True)
    observation = fleet_capture.observe()

    newest = observation[:, -fleet_capture.channels:]
    count = len(newest)
    diffs = [changed_fraction(newest[i], newest[j])
             for i in range(count) for j in range(i + 1, count)]
    if not diffs:
        print("  only one instance, nothing to compare")
        return
    identical = sum(1 for d in diffs if d == 0.0)
    print(f"  pairwise changed pixels: min {min(diffs):.3%} "
          f"median {statistics.median(diffs):.3%} max {max(diffs):.3%}")
    if identical:
        print(f"  PROBLEM: {identical}/{len(diffs)} pairs are the same picture")
    else:
        print(f"  all {len(diffs)} pairs differ — separate windows confirmed")


def probe_motion(bridges: list, fleet_capture: FleetCapture,
                 action_repeat: int) -> None:
    """A stack of four identical frames carries no velocity."""
    print("\n== motion: consecutive frames in the stack differ ==")

    fleet_capture.reset()
    stack = fleet_capture.pixels.stack
    observation = None
    for _ in range(stack):
        act(bridges, action_repeat, diverge=True)
        # Every observe advances the stack, so an extra one here with no ticks
        # in between would push the newest frame twice and report it as capture
        # falling behind. Use the last one the loop produced.
        observation = fleet_capture.observe()

    channels = fleet_capture.channels
    stale = 0
    deltas: list[float] = []
    for index in range(observation.shape[0]):
        frames = [observation[index, i * channels:(i + 1) * channels]
                  for i in range(stack)]
        for older, newer in zip(frames, frames[1:]):
            delta = changed_fraction(older, newer)
            deltas.append(delta)
            if delta == 0.0:
                stale += 1

    print(f"  frame-to-frame changed pixels: min {min(deltas):.3%} "
          f"median {statistics.median(deltas):.3%} max {max(deltas):.3%}")
    if stale:
        print(f"  PROBLEM: {stale}/{len(deltas)} adjacent pairs are identical — "
              f"the stack is not seeing new frames")
    else:
        print(f"  all {len(deltas)} adjacent pairs differ — the stack carries "
              f"motion")


def save_stack(fleet_capture: FleetCapture) -> None:
    """Write the stack out as a strip so it can be read left to right."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observation = fleet_capture.observe()
    channels = fleet_capture.channels
    frames = [observation[0, i * channels:(i + 1) * channels].transpose(1, 2, 0)
              for i in range(fleet_capture.pixels.stack)]
    strip = np.concatenate(frames, axis=1)
    if channels == 1:
        strip = np.repeat(strip, 3, axis=2)
    path = OUTPUT_DIR / "stack.png"
    cv2.imwrite(str(path), strip[:, :, ::-1])
    cv2.imwrite(str(OUTPUT_DIR / "stack-zoom.png"),
                cv2.resize(strip[:, :, ::-1],
                           (strip.shape[1] * 3, strip.shape[0] * 3),
                           interpolation=cv2.INTER_NEAREST))
    print(f"\n  wrote {path.name} — {fleet_capture.pixels.stack} frames "
          f"oldest to newest, and a 3x zoom beside it")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--ticks", type=int, default=300)
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    config = load_config()
    count = args.instances or config.instances.count
    fleet = launcher.bring_up(config, count=count)
    fleet_capture = None

    try:
        bridges = fleet.bridges()
        if not bridges:
            raise SystemExit("no instance reached a run")
        print(f"\n{len(bridges)}/{count} instances in a run")

        hwnds = [i.hwnd for i in fleet.instances
                 if i.ready and not i.failed]
        fleet_capture = FleetCapture(hwnds, config.pixels)
        print(f"student observation: {fleet_capture.shape} "
              f"({config.pixels.stack} x "
              f"{'grey' if config.pixels.grayscale else 'RGB'} "
              f"{config.pixels.width}x{config.pixels.height})")

        # Let the sessions fill before anything is measured.
        act(bridges, 30, diverge=False)

        probe_cost(bridges, fleet_capture, args.ticks,
                   config.env.action_repeat)
        probe_distinct(bridges, fleet_capture)
        probe_motion(bridges, fleet_capture, config.env.action_repeat)
        save_stack(fleet_capture)
    finally:
        if fleet_capture is not None:
            fleet_capture.stop()
        fleet.shutdown()


if __name__ == "__main__":
    main()
