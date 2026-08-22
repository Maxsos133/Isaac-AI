"""Brings up a fleet of Isaac instances and walks each one into a live run.

Startup is the only part of the system that needs window focus. Isaac's menus
ignore synthetic navigation keys, so instead of navigating we rely on the save
snapshot always resuming: from a cold launch, repeated SPACE presses walk
title -> file select -> CONTINUE with no cursor movement required.

The signal that an instance reached gameplay is the bridge connection itself:
the mod only connects from MC_POST_UPDATE, which does not run on menus.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from isaac_ai import windows
from isaac_ai.bridge import BridgeError, InstanceBridge
from isaac_ai.config import AppConfig

SAVE_FILES = (
    "persistentgamedata1.dat",
    "persistentgamedata2.dat",
    "persistentgamedata3.dat",
    "gamestate1.dat",
    "options.ini",
)


@dataclass
class Instance:
    index: int
    port: int
    process: subprocess.Popen
    hwnd: int | None = None
    bridge: InstanceBridge | None = None
    ready: bool = False
    failed: bool = False
    space_taps: int = 0
    # Where this instance's startup time actually went. Bring-up is the one
    # part of the system nobody has measured, and the two phases have entirely
    # different fixes: engine init could in principle be overlapped across
    # instances, whereas the SPACE walk is serialized by the focus requirement
    # and can only be made to tick faster.
    launch_seconds: float = 0.0   # Popen -> window exists
    entry_seconds: float = 0.0    # window exists -> mod connected


@dataclass
class Fleet:
    instances: list[Instance] = field(default_factory=list)

    @property
    def ready_count(self) -> int:
        return sum(1 for inst in self.instances if inst.ready)

    def bridges(self) -> list[InstanceBridge]:
        """Bridges for instances that actually made it into a run."""
        return [inst.bridge for inst in self.instances
                if inst.bridge is not None and inst.ready and not inst.failed]

    def shutdown(self) -> None:
        for inst in self.instances:
            if inst.bridge is not None:
                inst.bridge.close()
            if inst.process.poll() is None:
                inst.process.terminate()
        deadline = time.monotonic() + 10.0
        for inst in self.instances:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                inst.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                inst.process.kill()


def deploy_mod(config: AppConfig) -> Path:
    """Copy the bridge mod into the game's mods folder."""
    source = config.root / "mod" / "isaac_ai"
    target = config.game.mods_dir / "isaac_ai"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return target


def seed_save(config: AppConfig) -> None:
    """Restore the pristine save snapshot so every session starts identical.

    Unlock state determines the item pool, so letting it drift would make the
    environment non-stationary and runs incomparable.
    """
    snapshot = config.save.snapshot
    if not snapshot.is_dir():
        raise FileNotFoundError(f"save snapshot not found: {snapshot}")

    target_dir = config.game.savedata_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in SAVE_FILES:
        source = snapshot / name
        if source.exists():
            shutil.copy2(source, target_dir / name)


def ensure_steam_appid(config: AppConfig) -> None:
    """Let the exe initialise Steam directly instead of via the client.

    Steam's one-copy-at-a-time rule lives in the client UI; the binary itself
    is happy to run many times.
    """
    marker = config.game.executable.parent / "steam_appid.txt"
    if not marker.exists() or marker.read_text().strip() != str(config.game.app_id):
        marker.write_text(str(config.game.app_id))


def _launch_process(config: AppConfig, index: int, async_move: bool = False,
                    verbose: bool = False
                    ) -> tuple[subprocess.Popen, int | None, float]:
    """Start one game on this index's port and place its window.

    Split out of `_spawn` so a relaunch can reuse it without also constructing a
    second `InstanceBridge` for a port the first one still has bound.

    `async_move` is not cosmetic. `move_window` calls `MoveWindow`, which *sends*
    WM_WINDOWPOSCHANGING and waits for it to be handled — safe during bring-up,
    where nothing else exists yet, and a hang during a relaunch, where every
    other instance is blocked inside its socket read and not pumping messages.
    Measured: a relaunch printed its opening line and then never reached the
    SPACE loop at all, which read from outside as "SPACE is not being fired".
    `move_window_async` posts via SWP_ASYNCWINDOWPOS instead and cannot block.
    """
    port = config.instances.port_for(index)
    environment = os.environ.copy()
    environment["ISAAC_AI_PORT"] = str(port)
    environment["ISAAC_AI_INSTANCE"] = f"worker-{index}"

    process = subprocess.Popen(
        [str(config.game.executable), "--luadebug"],
        cwd=str(config.game.executable.parent),
        env=environment,
    )
    started = time.monotonic()
    hwnd = windows.wait_for_window(process.pid, timeout=60.0)
    launch_seconds = time.monotonic() - started
    if verbose:
        print(f"    window after {launch_seconds:.1f}s: hwnd={hwnd}")
    if hwnd:
        x, y, width, height = config.instances.window_rect(index)
        if async_move:
            windows.move_window_async(hwnd, x, y, width, height)
        else:
            windows.move_window(hwnd, x, y, width, height)
    if verbose:
        print(f"    positioned (async={async_move})")
    return process, hwnd, launch_seconds


def _spawn(config: AppConfig, index: int) -> Instance:
    port = config.instances.port_for(index)
    process, hwnd, launch_seconds = _launch_process(config, index)
    instance = Instance(index=index, port=port, process=process)
    instance.bridge = InstanceBridge(port=port, index=index)
    instance.launch_seconds = launch_seconds
    instance.hwnd = hwnd
    return instance


def _pump_ready(fleet: Fleet, verbose: bool) -> None:
    """Keep already-connected instances ticking.

    They are blocked on us otherwise, and a blocked game stops pumping Win32
    messages, which makes its window unfocusable and uncapturable.
    """
    for instance in fleet.instances:
        if instance.ready and not instance.failed and instance.bridge:
            try:
                instance.bridge.pump()
            except (OSError, BridgeError) as exc:
                instance.failed = True
                if verbose:
                    print(f"  instance {instance.index}: lost while idling: {exc}")


def bring_up_sequential(config: AppConfig, count: int | None = None,
                        verbose: bool = True) -> Fleet:
    """Launch and enter runs one instance at a time.

    Sequential startup means the instance being set up is the one the OS just
    gave the foreground to, so SPACE lands on it without any focus-stealing
    tricks. Earlier instances are kept ticking so they stay responsive.
    """
    count = count or config.instances.count
    fleet = Fleet()

    for index in range(count):
        if verbose:
            print(f"  launching instance {index}...")
        instance = _spawn(config, index)
        fleet.instances.append(instance)

        entry_started = time.monotonic()
        deadline = time.monotonic() + config.env.startup_timeout_seconds
        while time.monotonic() < deadline:
            if instance.process.poll() is not None:
                instance.failed = True
                if verbose:
                    print(f"  instance {index}: process exited early")
                break

            assert instance.bridge is not None
            try:
                if instance.bridge.poll_accept(timeout=0.05):
                    instance.ready = True
                    instance.entry_seconds = time.monotonic() - entry_started
                    if verbose:
                        print(f"  instance {index}: in a run "
                              f"({instance.space_taps} SPACE taps, "
                              f"{instance.launch_seconds:.1f}s launch + "
                              f"{instance.entry_seconds:.1f}s entry)")
                    break
            except (OSError, BridgeError) as exc:
                instance.failed = True
                if verbose:
                    print(f"  instance {index}: failed during startup: {exc}")
                break

            _pump_ready(fleet, verbose)

            if instance.hwnd and windows.is_foreground(instance.hwnd):
                windows.send_key(windows.VK_SPACE)
                instance.space_taps += 1
            elif instance.hwnd:
                windows.focus_window(instance.hwnd)

            time.sleep(0.3)

        if not instance.ready and not instance.failed:
            instance.failed = True
            if verbose:
                print(f"  instance {index}: timed out reaching a run")

    if verbose:
        entered = [i for i in fleet.instances if i.ready]
        if entered:
            launch = sum(i.launch_seconds for i in entered)
            entry = sum(i.entry_seconds for i in entered)
            taps = sum(i.space_taps for i in entered)
            print(f"fleet ready: {fleet.ready_count}/{len(fleet.instances)} "
                  f"in {launch + entry:.0f}s "
                  f"({launch:.0f}s engine launch, {entry:.0f}s run entry, "
                  f"{taps} SPACE taps total)")
            print(f"  per instance: {launch / len(entered):.1f}s launch + "
                  f"{entry / len(entered):.1f}s entry")
        else:
            print(f"fleet ready: {fleet.ready_count}/{len(fleet.instances)}")
    return fleet


def enter_runs(config: AppConfig, fleet: Fleet, verbose: bool = True) -> None:
    """Tap SPACE on each instance until its mod connects.

    A connection means MC_POST_UPDATE is running, which only happens in
    gameplay, so it is an exact readiness signal — no screen classification.
    """
    deadline = time.monotonic() + config.env.startup_timeout_seconds

    def try_accept(instance: Instance, timeout: float) -> None:
        """Accept a connection, treating any failure as this instance's alone.

        One instance dying during startup must not take down the fleet.
        """
        assert instance.bridge is not None
        try:
            if instance.bridge.poll_accept(timeout=timeout):
                instance.ready = True
                if verbose:
                    print(f"  instance {instance.index}: in a run "
                          f"(after {instance.space_taps} SPACE taps)")
        except (OSError, BridgeError) as exc:
            instance.failed = True
            if verbose:
                print(f"  instance {instance.index}: failed during startup: {exc}")

    while time.monotonic() < deadline:
        pending = [i for i in fleet.instances if not i.ready and not i.failed]
        if not pending:
            break

        # Instances that are already in are blocked waiting on us. A blocked
        # game does not pump Win32 messages, so its window becomes unfocusable
        # and — worse — AttachThreadInput against it hangs, which would wedge
        # the SPACE taps for every instance still at the menu.
        for instance in fleet.instances:
            if instance.ready and not instance.failed and instance.bridge:
                try:
                    instance.bridge.pump()
                except (OSError, BridgeError) as exc:
                    instance.failed = True
                    if verbose:
                        print(f"  instance {instance.index}: lost while waiting: {exc}")

        for instance in pending:
            if instance.process.poll() is not None:
                instance.failed = True
                if verbose:
                    print(f"  instance {instance.index}: process exited early")
                continue

            try_accept(instance, timeout=0.05)
            if instance.ready or instance.failed:
                continue

            if instance.hwnd and windows.focus_window(instance.hwnd):
                windows.send_key(windows.VK_SPACE)
                instance.space_taps += 1

        time.sleep(0.35)

    for instance in fleet.instances:
        if not instance.ready and not instance.failed:
            try_accept(instance, timeout=0.5)

    if verbose:
        print(f"fleet ready: {fleet.ready_count}/{len(fleet.instances)}")


def bring_up(config: AppConfig, count: int | None = None,
             verbose: bool = True) -> Fleet:
    """Full startup: deploy, seed, launch, and enter runs."""
    if verbose:
        print("deploying mod...")
    deploy_mod(config)

    if verbose:
        print(f"seeding save from {config.save.snapshot.name}...")
    seed_save(config)
    ensure_steam_appid(config)

    n = count or config.instances.count
    if verbose:
        print(f"launching {n} instance(s) sequentially...")

    fleet = Fleet()
    try:
        fleet = bring_up_sequential(config, count=n, verbose=verbose)
    except BaseException:
        # Anything escaping here would leave orphaned game processes spinning
        # on the host with no owner.
        fleet.shutdown()
        raise
    return fleet


def relaunch(config: AppConfig, fleet: Fleet, bridge: InstanceBridge,
             keep_alive: list[InstanceBridge] | None = None,
             verbose: bool = True) -> bool:
    """Bring one crashed instance back on the port it already owns.

    The game faults roughly once an hour across twenty instances (an access
    violation, `0xc0000005`). The bridge fix made that cost one instance instead
    of the whole run, which was the right first step — but floor-v27b lost two of
    twenty over 5M steps, so a long unattended run now finishes 10% short and
    slower than it started.

    Three things make this safe to do mid-training:

    **The port is already ours.** `_failed` never closes the bridge, so its
    listener has been bound since construction. `reset_connection` drops the dead
    socket and leaves the listener accepting, so the new process reconnects to
    the same place with no rebinding.

    **`keep_alive` must be every healthy bridge.** A connected instance sits
    blocked inside its socket read, and a blocked game stops pumping Win32
    messages — which makes it unfocusable and, if anything ever attaches to its
    input queue, hangs. `enter_runs` keeps instances ticking for exactly this
    reason during bring-up, and a relaunch is bring-up with nineteen games
    already blocked. Not pumping them is the difference between this working and
    wedging the fleet.

    **Pumping is not free**, which is why the caller ends every episode
    afterwards: `pump()` sends `noop`, so each healthy instance advances a few
    hundred ticks under neutral control while the SPACE walk happens. Players
    wander and some die. Bounded and rare, but it must not be left to
    contaminate a rollout.

    Returns True if the instance is back in a run.
    """
    keep_alive = keep_alive or []
    instance = next((i for i in fleet.instances if i.index == bridge.index), None)
    if instance is None:
        return False

    if verbose:
        print(f"  relaunching instance {bridge.index} on port {bridge.port}...")

    # Reap first. A faulted Isaac can sit as a zombie holding the window, and a
    # second process on the same port would then race the first to connect.
    if instance.process.poll() is None:
        instance.process.terminate()
        try:
            instance.process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            instance.process.kill()
    bridge.reset_connection()

    try:
        process, hwnd, launch_seconds = _launch_process(
            config, bridge.index, async_move=True, verbose=verbose)
    except OSError as exc:
        if verbose:
            print(f"  instance {bridge.index}: relaunch failed to start: {exc}")
        return False

    instance.process = process
    instance.hwnd = hwnd
    instance.launch_seconds = launch_seconds
    instance.ready = False
    instance.failed = False
    instance.space_taps = 0

    started = time.monotonic()
    deadline = started + config.env.startup_timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            if verbose:
                print(f"  instance {bridge.index}: relaunched process exited")
            instance.failed = True
            return False

        # Every other instance is blocked on us right now. Tick them before
        # touching the foreground, or their windows are unfocusable and the
        # SPACE walk below never lands.
        for other in keep_alive:
            try:
                # `tick`, not `pump`: these instances are blocked waiting for an
                # action, not holding an unread observation. See InstanceBridge.
                other.tick()
            except (OSError, BridgeError):
                # A second instance dying during a relaunch is someone else's
                # problem — the caller sees it on the next step and can relaunch
                # that one too. Losing this attempt as well would make two
                # crashes unrecoverable when one is not.
                pass

        try:
            if bridge.poll_accept(timeout=0.05):
                instance.ready = True
                instance.entry_seconds = time.monotonic() - started
                if verbose:
                    print(f"  instance {bridge.index}: back in a run "
                          f"({instance.space_taps} SPACE taps, "
                          f"{instance.entry_seconds:.1f}s)")
                return True
        except (OSError, BridgeError) as exc:
            if verbose:
                print(f"  instance {bridge.index}: relaunch handshake failed: {exc}")
            instance.failed = True
            return False

        focused = bool(hwnd) and windows.focus_window(hwnd)
        if focused:
            windows.send_key(windows.VK_SPACE)
            instance.space_taps += 1
        # An isolated reproduction of this loop focuses and taps fine, so if a
        # relaunch stalls the reason is somewhere else. Report the state rather
        # than leaving it to be inferred from "SPACE is not being pressed".
        if verbose and instance.space_taps % 10 == 0:
            print(f"    ...{time.monotonic() - started:.0f}s "
                  f"taps={instance.space_taps} focused={focused} "
                  f"hwnd={hwnd} alive={process.poll() is None}")
        time.sleep(0.35)

    if verbose:
        print(f"  instance {bridge.index}: relaunch timed out")
    instance.failed = True
    return False
