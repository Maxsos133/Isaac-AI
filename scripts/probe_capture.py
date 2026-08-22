"""Can we get pixels out of a blocked instance, and what does it cost?

Phase 3 needs a frame for every step the student learns from. `PrintWindow`
cannot supply one: it needs the target to pump Win32 messages, and an instance
blocked inside MC_POST_UPDATE waiting for its action does not pump anything.
Windows Graphics Capture reads the compositor's own surface instead, so it
should not care what the game's message loop is doing.

This measures whether that is actually true, and three things that decide
whether Phase 3 is affordable at all:

  gate       does a frame come back while the game is blocked
  geometry   is every instance the same pixel size, and does it stay that way
  cost       what capture does to the fleet's steps/s, and how stale a frame is
  occlusion  does a window with another window fully on top of it still work
  content    what the frame looks like once downsampled to a policy input

Geometry is not a detail. A capture is whatever size the window happens to be,
so if instances differ from each other, or drift when the grid is re-laid out
for a different fleet size, the student's input distribution moves underneath
it — the same class of non-stationarity the frozen save snapshots exist to
prevent.

Nothing here trains. It brings a fleet up, measures, and shuts down.

Needs `windows-capture` (already installed into .venv; it pulls opencv-python
with it, which is what writes the sample frames):

    .venv/Scripts/python.exe -m pip install windows-capture

    .venv/Scripts/python.exe scripts/probe_capture.py                # 12 instances
    .venv/Scripts/python.exe scripts/probe_capture.py --instances 2  # quick check
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402  (arrives with windows-capture)
from windows_capture import WindowsCapture  # noqa: E402

from isaac_ai import launcher, windows  # noqa: E402
from isaac_ai.config import load_config  # noqa: E402

# Candidate student input sizes, all 16:9 so the downsample is a uniform scale
# rather than a stretch. Against a 480x270 client these are exact divisions by
# 3, 3.75, 5 and 6 — a clean box filter, not a resampling artefact generator.
DOWNSAMPLES = ((160, 90), (128, 72), (96, 54), (80, 45))

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "spike" / "capture-probe"


class LiveCapture:
    """A Graphics Capture session for one window, keeping only its last frame.

    WGC hands over a frame when the window *presents* one. A blocked instance is
    not presenting, so what this holds while blocked is the last frame the game
    drew — which is exactly the thing being measured, not a flaw in the setup.
    """

    def __init__(self, hwnd: int, index: int) -> None:
        self.index = index
        self.hwnd = hwnd
        self.frames = 0
        self.error: str | None = None
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._arrived = 0.0

        session = WindowsCapture(
            cursor_capture=False,
            # The capture highlight is drawn into the frame on Windows 11, so it
            # would end up in the student's observation.
            draw_border=False,
            window_hwnd=hwnd,
        )

        @session.event
        def on_frame_arrived(frame, capture_control):  # noqa: ANN001
            # frame_buffer is a zero-copy view over a native mapping that is
            # released when this callback returns, so it has to be copied here.
            # BGRA -> RGB, matching windows.capture_window's convention.
            pixels = frame.frame_buffer[:, :, [2, 1, 0]].copy()
            with self._lock:
                self._latest = pixels
                self._arrived = time.perf_counter()
                self.frames += 1

        @session.event
        def on_closed():  # noqa: ANN202
            pass

        self._control = session.start_free_threaded()

    def latest(self) -> tuple[np.ndarray | None, float]:
        """The most recent frame and how many milliseconds old it is."""
        with self._lock:
            if self._latest is None:
                return None, 0.0
            return self._latest, (time.perf_counter() - self._arrived) * 1000.0

    def stop(self) -> None:
        try:
            self._control.stop()
        except Exception:  # noqa: BLE001 - shutdown must never mask a result
            pass


def is_blank(frame: np.ndarray) -> bool:
    """A capture that silently returns an empty surface reads as a black image."""
    return float(frame.std()) < 1.0


def describe(frame: np.ndarray) -> str:
    return (f"{frame.shape[1]}x{frame.shape[0]} "
            f"mean={frame.mean():.1f} std={frame.std():.1f} "
            f"{'BLANK' if is_blank(frame) else 'has content'}")


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


# Both return HWND. Left at the ctypes default these come back as c_int, and a
# truncated handle would compare unequal to the real one — which would report
# "not covered" for a window that is in fact covered, quietly inverting the
# occlusion result.
windows.user32.WindowFromPoint.argtypes = [POINT]
windows.user32.WindowFromPoint.restype = wintypes.HWND
windows.user32.GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
windows.user32.GetAncestor.restype = wintypes.HWND

dwmapi = ctypes.WinDLL("dwmapi")
DWMWA_EXTENDED_FRAME_BOUNDS = 9
SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79
SWP_NOZORDER, SWP_NOACTIVATE, SWP_ASYNCWINDOWPOS = 0x0004, 0x0010, 0x4000

# HWND is 64-bit here; left to the ctypes default the handles go in as 32-bit
# ints and the call silently addresses the wrong window, or none.
windows.user32.SetWindowPos.argtypes = [
    wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_uint,
]
windows.user32.SetWindowPos.restype = wintypes.BOOL


def move_window_async(hwnd: int, x: int, y: int,
                      width: int, height: int) -> None:
    """Reposition a window without waiting for its thread to acknowledge.

    `MoveWindow` sends WM_WINDOWPOSCHANGING to the window's own thread and
    blocks until it is handled. An instance blocked in MC_POST_UPDATE is not
    pumping messages, so that call never returns — the same trap
    AttachThreadInput sets, reached through a different function. It is not
    hypothetical: it deadlocked this probe on a 12-instance fleet, while
    slipping through at 2 because the target happened to still be mid-tick.

    SWP_ASYNCWINDOWPOS posts the request instead, so it lands the next time the
    instance runs a tick. Callers must pump afterwards for it to take effect.
    """
    windows.user32.SetWindowPos(hwnd, 0, x, y, width, height,
                                SWP_NOZORDER | SWP_NOACTIVATE | SWP_ASYNCWINDOWPOS)


def root_window_at(x: int, y: int) -> int:
    """Which top-level window owns the pixel at (x, y) — i.e. what is on top."""
    hwnd = windows.user32.WindowFromPoint(POINT(x, y))
    return windows.user32.GetAncestor(hwnd, 2) or 0 if hwnd else 0  # GA_ROOT


def geometry(hwnd: int) -> dict[str, tuple[int, int]]:
    """Sizes that matter for capture, and where the game's picture sits in one.

    Graphics Capture hands back the window as the compositor holds it, which is
    the DWM extended frame — the visible window including its title bar, but
    without the invisible resize border that GetWindowRect counts. The game
    only draws inside the client area, so the title bar is dead pixels that
    have to be cropped off rather than fed to the student.
    """
    outer = windows.RECT()
    windows.user32.GetWindowRect(hwnd, ctypes.byref(outer))
    client = windows.RECT()
    windows.user32.GetClientRect(hwnd, ctypes.byref(client))

    origin = POINT(0, 0)
    windows.user32.ClientToScreen(hwnd, ctypes.byref(origin))

    bounds = windows.RECT()
    dwmapi.DwmGetWindowAttribute(wintypes.HWND(hwnd), DWMWA_EXTENDED_FRAME_BOUNDS,
                                 ctypes.byref(bounds), ctypes.sizeof(bounds))

    return {
        "outer": (outer.right - outer.left, outer.bottom - outer.top),
        "captured": (bounds.right - bounds.left, bounds.bottom - bounds.top),
        "client": (client.right, client.bottom),
        "client_at": (origin.x - bounds.left, origin.y - bounds.top),
    }


def crop_to_client(frame: np.ndarray, geo: dict) -> np.ndarray:
    """Drop the title bar and borders, leaving only what the game drew."""
    x, y = geo["client_at"]
    width, height = geo["client"]
    return frame[y:y + height, x:x + width]


def uniform_margins(frame: np.ndarray) -> tuple[int, int, int, int]:
    """Leading/trailing rows and columns that are flat black — letterbox bars.

    Isaac renders at its own aspect ratio. If the window we impose does not
    match it, part of every frame is bar rather than game, and the student
    spends input resolution on nothing.
    """
    dark = frame.max(axis=2) < 16
    rows, cols = dark.all(axis=1), dark.all(axis=0)

    def leading(flags: np.ndarray) -> int:
        idx = np.flatnonzero(~flags)
        return int(idx[0]) if idx.size else len(flags)

    return (leading(rows), leading(rows[::-1]),
            leading(cols), leading(cols[::-1]))


def probe_gate(bridge, hwnd: int, capture: LiveCapture) -> None:
    """The question the whole phase hangs on: pixels from a blocked game."""
    print("\n== gate: capture while the instance is blocked ==")

    # Received a tick and deliberately not answered, so the game is now sitting
    # inside MC_POST_UPDATE and has stopped pumping messages.
    bridge.receive()

    before = capture.frames
    time.sleep(1.0)
    frame, age_ms = capture.latest()
    delivered = capture.frames - before

    if frame is None:
        print("  WGC:         no frame at all while blocked")
    else:
        print(f"  WGC:         {describe(frame)}, {age_ms:.0f}ms old")
        print(f"               {delivered} new frames arrived during 1s blocked")

    # PrintWindow is expected to hang here. Run it where it can be abandoned.
    outcome: dict[str, object] = {}

    def attempt() -> None:
        started = time.perf_counter()
        image = windows.capture_window(hwnd)
        outcome["ms"] = (time.perf_counter() - started) * 1000.0
        outcome["image"] = image

    thread = threading.Thread(target=attempt, daemon=True)
    thread.start()
    thread.join(timeout=5.0)
    hung = thread.is_alive()
    if hung:
        print("  PrintWindow: no return after 5s — hangs, as documented")
    else:
        image = outcome.get("image")
        note = describe(image) if isinstance(image, np.ndarray) else "returned None"
        print(f"  PrintWindow: returned in {outcome['ms']:.0f}ms — {note}")

    # Let the game run again and see whether the stuck call frees itself. If it
    # does, the failure is precisely "target not pumping" and nothing else.
    bridge.send({"t": "noop"})
    if hung:
        for _ in range(30):
            bridge.pump()
        thread.join(timeout=3.0)
        if thread.is_alive():
            print("               still stuck after the game resumed")
        else:
            print("               unblocked once the game resumed — so the hang "
                  "is the target's message loop, nothing else")


def probe_geometry(config, fleet, captures: list[LiveCapture],
                   count: int) -> dict | None:
    """Every instance must capture at the same size, whatever the fleet size."""
    print("\n== geometry: window and capture sizes ==")

    requested = (config.instances.window_width, config.instances.window_height)
    print(f"  config asks for {requested[0]}x{requested[1]} per window, "
          f"{config.instances.grid_columns} columns")

    live = {c.index: c for c in captures}
    rows: list[tuple[int, dict, tuple[int, int] | None]] = []
    for instance in fleet.instances:
        if instance.index not in live or not instance.hwnd:
            continue
        geo = geometry(instance.hwnd)
        frame, _ = live[instance.index].latest()
        rows.append((instance.index, geo,
                     (frame.shape[1], frame.shape[0]) if frame is not None else None))

    if not rows:
        print("  no instances to measure")
        return None

    print(f"\n  {'inst':<5}{'outer':>11}{'captured':>11}{'client':>11}"
          f"{'client at':>11}{'frame':>11}")
    for index, geo, shape in rows:
        print(f"  {index:<5}{'x'.join(map(str, geo['outer'])):>11}"
              f"{'x'.join(map(str, geo['captured'])):>11}"
              f"{'x'.join(map(str, geo['client'])):>11}"
              f"{'+{},+{}'.format(*geo['client_at']):>11}"
              f"{('x'.join(map(str, shape)) if shape else '-'):>11}")

    problems: list[str] = []
    if len({tuple(g["client"]) for _, g, _ in rows}) > 1:
        problems.append("instances differ in client size")
    if len({s for _, _, s in rows if s}) > 1:
        problems.append("instances differ in captured size")
    for index, geo, shape in rows:
        if shape and tuple(geo["captured"]) != shape:
            problems.append(f"instance {index}: capture {shape} != "
                            f"frame bounds {tuple(geo['captured'])}")

    # Does the grid actually fit? A window pushed past the desktop edge is not
    # clipped by Windows, but it does stop being something a human can check.
    screen = (windows.user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
              windows.user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
    columns = min(count, config.instances.grid_columns)
    grid_rows = (count + config.instances.grid_columns - 1) // config.instances.grid_columns
    span = (config.instances.origin_x + columns * requested[0],
            config.instances.origin_y + grid_rows * requested[1])
    fits = span[0] <= screen[0] and span[1] <= screen[1]
    print(f"\n  grid at {count} instances: {columns}x{grid_rows} = "
          f"{span[0]}x{span[1]} on a {screen[0]}x{screen[1]} desktop "
          f"({'fits' if fits else 'DOES NOT FIT'})")
    if not fits:
        problems.append("the window grid runs off the desktop")

    geo = rows[0][1]
    dead = (geo["captured"][0] - geo["client"][0],
            geo["captured"][1] - geo["client"][1])
    waste = 1 - (geo["client"][0] * geo["client"][1]) / (
        geo["captured"][0] * geo["captured"][1])
    print(f"  dead pixels in each capture: {dead[0]}px wide, {dead[1]}px tall "
          f"(title bar and borders) = {100 * waste:.1f}% of the frame")

    # config.toml drives MoveWindow, which sizes the *outer* rect — that also
    # contains the invisible resize border, which is neither captured nor drawn
    # into. Sizing against the captured rect instead would come out a dozen
    # pixels short, which is the kind of error that only shows up as a policy
    # trained at a resolution nobody chose.
    chrome = (geo["outer"][0] - geo["client"][0],
              geo["outer"][1] - geo["client"][1])
    print(f"  outer window carries {chrome[0]}px / {chrome[1]}px more than the "
          f"client area")

    frame, _ = live[rows[0][0]].latest()
    if frame is not None:
        client = crop_to_client(frame, geo)
        top, bottom, left, right = uniform_margins(client)
        aspect = client.shape[1] / client.shape[0]
        print(f"  game picture: {client.shape[1]}x{client.shape[0]} "
              f"(aspect {aspect:.3f}); black bars "
              f"t={top} b={bottom} l={left} r={right}")

        # Sizing the client to an exact multiple of the student input keeps the
        # downsample a clean box filter instead of a resampling artefact
        # generator, and 16:9 is the aspect options.ini asks Isaac for.
        print("\n  for a clean client area, set config.toml [instances] to:")
        for w, h in ((480, 270), (640, 360)):
            print(f"    {w}x{h} client (16:9) -> window_width = {w + chrome[0]}"
                  f", window_height = {h + chrome[1]}")

    if problems:
        print("\n  PROBLEMS: " + "; ".join(problems))
    else:
        print("\n  all instances identical, grid fits")
    return rows[0][1]


def probe_geometry_stable(fleet, captures: list[LiveCapture],
                          before: dict) -> None:
    """Re-measure after everything else has moved windows around."""
    print("\n== geometry: unchanged after the rest of the probe ==")
    live = {c.index: c for c in captures}
    drifted = []
    for instance in fleet.instances:
        if instance.index not in live or not instance.hwnd:
            continue
        geo = geometry(instance.hwnd)
        if geo["client"] != before["client"] or geo["captured"] != before["captured"]:
            drifted.append(f"instance {instance.index}: {geo}")
    if drifted:
        print("  DRIFTED: " + "; ".join(drifted))
    else:
        print(f"  every instance still {'x'.join(map(str, before['client']))} "
              f"client / {'x'.join(map(str, before['captured']))} captured")


def step_fleet(bridges: list, captures: list[LiveCapture] | None,
               ticks: int) -> tuple[float, list[float], list[float]]:
    """Drive every instance for `ticks`, optionally grabbing a frame each tick.

    Capture happens between receive and send: that is the instant the agent
    holds the observation and has not yet acted, so it is the frame a real step
    would have to pair with an action.
    """
    grab_ms: list[float] = []
    age_ms: list[float] = []

    started = time.perf_counter()
    for _ in range(ticks):
        for bridge in bridges:
            bridge.receive()

        if captures is not None:
            for capture in captures:
                at = time.perf_counter()
                frame, age = capture.latest()
                grab_ms.append((time.perf_counter() - at) * 1000.0)
                if frame is not None:
                    age_ms.append(age)

        for bridge in bridges:
            bridge.send({"t": "noop"})

    return time.perf_counter() - started, grab_ms, age_ms


def probe_cost(bridges: list, captures: list[LiveCapture], ticks: int) -> None:
    print(f"\n== cost: {len(bridges)} instances, {ticks} ticks each ==")

    elapsed, _, _ = step_fleet(bridges, None, ticks)
    baseline = len(bridges) * ticks / elapsed
    print(f"  without capture: {baseline:7.1f} game ticks/s")

    before = [c.frames for c in captures]
    elapsed, grab_ms, age_ms = step_fleet(bridges, captures, ticks)
    withcap = len(bridges) * ticks / elapsed
    delivered = sum(c.frames - b for c, b in zip(captures, before))

    print(f"  with capture:    {withcap:7.1f} game ticks/s "
          f"({100 * (withcap - baseline) / baseline:+.1f}%)")
    if grab_ms:
        print(f"  grab cost:       p50 {statistics.median(grab_ms):.3f}ms  "
              f"p95 {sorted(grab_ms)[int(0.95 * len(grab_ms))]:.3f}ms")
    if age_ms:
        print(f"  frame age:       p50 {statistics.median(age_ms):.1f}ms  "
              f"p95 {sorted(age_ms)[int(0.95 * len(age_ms))]:.1f}ms "
              f"(a tick is 33ms)")
    print(f"  frames delivered: {delivered} over {len(bridges) * ticks} ticks "
          f"= {delivered / max(1, len(bridges) * ticks):.2f} per tick")

    # An agent step is action_repeat ticks, so this is the number that decides
    # whether the fleet can feed a pixel student at its current size.
    print(f"  => {withcap:.0f} ticks/s sustained with capture on")


def probe_occlusion(config, fleet, bridges: list,
                    captures: list[LiveCapture]) -> None:
    """Many-instances-per-host depends on occluded windows still being usable."""
    print("\n== occlusion: a fully covered window ==")
    if len(captures) < 2:
        print("  skipped (needs at least 2 instances)")
        return

    victim, cover = fleet.instances[0], fleet.instances[-1]
    if not victim.hwnd or not cover.hwnd:
        print("  skipped (missing window handle)")
        return

    x, y, width, height = config.instances.window_rect(0)
    # The last instance launched already sits above the first in Z-order, so
    # moving it is enough; no Z-order change is asked for. The move is posted
    # rather than sent, and only lands once the fleet is pumped.
    move_window_async(cover.hwnd, x, y, width, height)
    for _ in range(30):
        for bridge in bridges:
            bridge.pump()

    on_top = root_window_at(x + width // 2, y + height // 2)
    covered = on_top == cover.hwnd
    print(f"  instance 0 covered by instance {cover.index}: {covered}")
    if not covered:
        print("  (cover did not land — occlusion result below is not meaningful)")

    for bridge in bridges:
        bridge.receive()
    frame, age_ms = captures[0].latest()
    for bridge in bridges:
        bridge.send({"t": "noop"})

    if frame is None:
        print("  no frame from the covered window")
    else:
        print(f"  covered window: {describe(frame)}, {age_ms:.0f}ms old")

    move_window_async(cover.hwnd, *config.instances.window_rect(cover.index))
    for _ in range(30):
        for bridge in bridges:
            bridge.pump()


def probe_content(bridges: list, captures: list[LiveCapture],
                  geo: dict | None) -> None:
    """Save what the student would actually see, at each candidate size."""
    print("\n== content: frames written for eyeballing ==")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for _ in range(30):
        for bridge in bridges:
            bridge.pump()

    frame, _ = captures[0].latest()
    if frame is None:
        print("  no frame to write")
        return

    cv2.imwrite(str(OUTPUT_DIR / "raw-window.png"), frame[:, :, ::-1])
    if geo is not None:
        frame = crop_to_client(frame, geo)

    bgr = frame[:, :, ::-1]
    native = OUTPUT_DIR / "native.png"
    cv2.imwrite(str(native), bgr)
    print(f"  raw-window.png   full capture, title bar included")
    print(f"  {native.name:<16} {frame.shape[1]}x{frame.shape[0]} "
          f"(client area only)")

    for width, height in DOWNSAMPLES:
        small = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        path = OUTPUT_DIR / f"down-{width}x{height}.png"
        cv2.imwrite(str(path), small)
        # Written at 4x as well, because a 84x48 thumbnail is unreadable at
        # native size on screen and the question is whether the *content*
        # survives, not whether the file is small.
        cv2.imwrite(str(OUTPUT_DIR / f"down-{width}x{height}-zoom.png"),
                    cv2.resize(small, (width * 4, height * 4),
                               interpolation=cv2.INTER_NEAREST))
        print(f"  {path.name:<16} {width}x{height}  std={small.std():.1f}")

    print(f"\n  open {OUTPUT_DIR} and check the -zoom images: enemies, tears "
          f"and doors all have to stay distinguishable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=None,
                        help="fleet size (default: config.toml)")
    parser.add_argument("--ticks", type=int, default=300,
                        help="ticks per throughput measurement")
    args = parser.parse_args()

    # A 12-instance bring-up is minutes long and this is often watched through a
    # redirected log, where block buffering hides every phase until the process
    # exits — which is worthless precisely when a run wedges and gets killed.
    sys.stdout.reconfigure(line_buffering=True)

    config = load_config()
    count = args.instances or config.instances.count
    fleet = launcher.bring_up(config, count=count)
    captures: list[LiveCapture] = []

    try:
        bridges = fleet.bridges()
        if not bridges:
            raise SystemExit("no instance reached a run")
        print(f"\n{len(bridges)}/{count} instances in a run")

        for instance in fleet.instances:
            if instance.ready and not instance.failed and instance.hwnd:
                captures.append(LiveCapture(instance.hwnd, instance.index))
        if not captures:
            raise SystemExit("no window handles to capture")

        # WGC only fills its frame pool when the window presents, so give the
        # fleet a moment of actual running before asking for a frame.
        for _ in range(30):
            for bridge in bridges:
                bridge.pump()

        started = [c for c in captures if c.frames > 0]
        print(f"capture sessions delivering frames: "
              f"{len(started)}/{len(captures)}")

        probe_gate(bridges[0], captures[0].hwnd, captures[0])
        geo = probe_geometry(config, fleet, captures, count)
        probe_cost(bridges, captures, args.ticks)
        probe_occlusion(config, fleet, bridges, captures)
        if geo is not None:
            probe_geometry_stable(fleet, captures, geo)
        probe_content(bridges, captures, geo)
    finally:
        for capture in captures:
            capture.stop()
        fleet.shutdown()


if __name__ == "__main__":
    main()
