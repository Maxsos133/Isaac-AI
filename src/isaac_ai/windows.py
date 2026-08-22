"""Win32 helpers: locating, positioning, focusing and capturing game windows.

Only startup needs focus, for the SPACE taps that walk an instance into a run.
Gameplay input goes through the mod, and frame capture works on background
windows, so nothing here is touched during training except `capture_window`.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import time

import numpy as np

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
dwmapi = ctypes.WinDLL("dwmapi")

DWMWA_EXTENDED_FRAME_BOUNDS = 9
SWP_NOZORDER, SWP_NOACTIVATE, SWP_ASYNCWINDOWPOS = 0x0004, 0x0010, 0x4000

VK_SPACE = 0x20
# ALT. Used only to lift the foreground lock before SetForegroundWindow; it is
# unbound in Isaac, so a stray tap cannot affect a game that is already running.
VK_MENU = 0x12
VK_RETURN = 0x0D
VK_ESCAPE = 0x1B

SW_RESTORE = 9
PW_RENDERFULLCONTENT = 0x00000002
SRCCOPY = 0x00CC0020
BI_RGB = 0

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008


class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class INPUT(ctypes.Structure):
    """Must match the Win32 INPUT layout exactly.

    SendInput silently rejects the call if cbSize does not equal the real
    sizeof(INPUT) — 40 bytes on x64, where the union is sized by MOUSEINPUT
    rather than by the smaller KEYBDINPUT. The trailing padding makes up the
    difference.
    """

    _fields_ = [("type", wintypes.DWORD), ("ki", KEYBDINPUT),
                ("pad1", ctypes.c_uint32), ("pad2", ctypes.c_uint32)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long),
                ("biHeight", ctypes.c_long), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long),
                ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


_ENUM_PROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def find_main_window(pid: int) -> int | None:
    """Return the visible top-level window belonging to `pid`, if any."""
    found: list[int] = []

    def callback(hwnd: int, _lparam: int) -> bool:
        window_pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
        if window_pid.value == pid and user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                found.append(hwnd)
        return True

    user32.EnumWindows(_ENUM_PROC(callback), 0)
    return found[0] if found else None


def wait_for_window(pid: int, timeout: float = 60.0) -> int | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        hwnd = find_main_window(pid)
        if hwnd:
            return hwnd
        time.sleep(0.3)
    return None


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


# HWND is 64-bit; left at the ctypes default these go in truncated and the call
# addresses the wrong window, or none.
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_uint]
user32.SetWindowPos.restype = wintypes.BOOL


def move_window(hwnd: int, x: int, y: int, width: int, height: int) -> None:
    """Only safe before an instance connects. Use `move_window_async` after.

    MoveWindow sends WM_WINDOWPOSCHANGING to the window's own thread and waits
    for it to be handled, so against an instance blocked in MC_POST_UPDATE it
    never returns — the same trap AttachThreadInput sets. Bring-up is fine
    because the instance is still on menus and pumping normally.
    """
    user32.MoveWindow(hwnd, x, y, width, height, True)


def move_window_async(hwnd: int, x: int, y: int,
                      width: int, height: int) -> None:
    """Reposition without waiting for the target thread to acknowledge.

    SWP_ASYNCWINDOWPOS posts the request instead of sending it, so it lands the
    next time that instance runs a tick. The caller has to keep the fleet
    stepping for it to take effect at all.
    """
    user32.SetWindowPos(hwnd, 0, x, y, width, height,
                        SWP_NOZORDER | SWP_NOACTIVATE | SWP_ASYNCWINDOWPOS)


def client_rect_in_capture(hwnd: int) -> tuple[int, int, int, int]:
    """Where the game's picture sits inside a Graphics Capture frame.

    WGC returns the DWM extended frame — the visible window, title bar
    included, but without the invisible resize border GetWindowRect counts. The
    game only draws into the client area, so everything outside it is chrome
    that would otherwise become 11% of the student's observation.

    Returns (x, y, width, height) in capture-frame coordinates.
    """
    client = RECT()
    user32.GetClientRect(hwnd, ctypes.byref(client))

    origin = POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))

    bounds = RECT()
    dwmapi.DwmGetWindowAttribute(wintypes.HWND(hwnd), DWMWA_EXTENDED_FRAME_BOUNDS,
                                 ctypes.byref(bounds), ctypes.sizeof(bounds))

    return (origin.x - bounds.left, origin.y - bounds.top,
            client.right, client.bottom)


def is_foreground(hwnd: int) -> bool:
    return user32.GetForegroundWindow() == hwnd


def focus_window(hwnd: int) -> bool:
    """Ask for the foreground without attaching to another input queue.

    The obvious implementation attaches to the current foreground thread via
    AttachThreadInput so SetForegroundWindow is permitted. Do not do that here:
    Isaac stops pumping messages while it generates a floor and while it is
    blocked on the agent, and AttachThreadInput against a thread in that state
    blocks indefinitely, taking the whole launcher with it.

    Instances are instead brought up one at a time, where a freshly launched
    game takes the foreground on its own and this is only a nudge.

    **That nudge is not enough for a mid-run relaunch.** During bring-up nothing
    else is competing and the new window arrives in the foreground by itself.
    When one instance is relaunched during training, the foreground belongs to
    something else — a blocked Isaac, or the terminal — and Windows refuses
    `SetForegroundWindow` from a process that does not own it. It fails quietly,
    `is_foreground` reports False, and the SPACE walk never sends a key, so the
    relaunched game sits on its cutscene until the startup timeout.

    The fix is to inject a keystroke first. Windows lifts the foreground lock for
    the process that most recently synthesised input, so a bare ALT tap makes the
    subsequent `SetForegroundWindow` legal. It touches nothing but our own input
    queue, which is what makes it safe here where `AttachThreadInput` is not:
    that would attach to the *current* foreground thread, and if that happens to
    be an instance blocked on the agent it hangs forever.

    ALT specifically because it is inert in Isaac — it is not bound to any game
    action, so a stray tap landing on a window that is already in a run cannot
    move a player or fire a tear.
    """
    user32.ShowWindow(hwnd, SW_RESTORE)
    user32.BringWindowToTop(hwnd)
    if not user32.SetForegroundWindow(hwnd):
        _unlock_foreground()
        user32.SetForegroundWindow(hwnd)
    time.sleep(0.1)
    return is_foreground(hwnd)


def _unlock_foreground() -> None:
    """Tap ALT so this process is allowed to set the foreground window."""
    scan = user32.MapVirtualKeyW(VK_MENU, 0)
    for flags in (KEYEVENTF_SCANCODE,
                  KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP):
        event = INPUT(type=INPUT_KEYBOARD)
        event.ki = KEYBDINPUT(wVk=VK_MENU, wScan=scan, dwFlags=flags,
                              time=0, dwExtraInfo=None)
        user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))
    time.sleep(0.02)


def send_key(vk: int, hold_seconds: float = 0.09) -> None:
    """Send a real keystroke to whichever window currently has focus.

    Isaac ignores posted window messages, so this has to go through SendInput
    with scancodes. Only used for menu confirmation during startup.
    """
    scan = user32.MapVirtualKeyW(vk, 0)

    down = INPUT(type=INPUT_KEYBOARD)
    down.ki = KEYBDINPUT(wVk=vk, wScan=scan, dwFlags=KEYEVENTF_SCANCODE,
                         time=0, dwExtraInfo=None)
    sent = user32.SendInput(1, ctypes.byref(down), ctypes.sizeof(INPUT))
    if sent != 1:
        raise OSError(f"SendInput failed: {ctypes.get_last_error()}")

    time.sleep(hold_seconds)

    up = INPUT(type=INPUT_KEYBOARD)
    up.ki = KEYBDINPUT(wVk=vk, wScan=scan,
                       dwFlags=KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP,
                       time=0, dwExtraInfo=None)
    user32.SendInput(1, ctypes.byref(up), ctypes.sizeof(INPUT))


def capture_window(hwnd: int) -> np.ndarray | None:
    """Grab a window's client area as RGB, even when occluded or unfocused."""
    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None

    window_dc = user32.GetDC(hwnd)
    memory_dc = gdi32.CreateCompatibleDC(window_dc)
    bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
    previous = gdi32.SelectObject(memory_dc, bitmap)

    try:
        if not user32.PrintWindow(hwnd, memory_dc, PW_RENDERFULLCONTENT):
            return None

        info = BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        info.bmiHeader.biWidth = width
        # Negative height gives a top-down image, matching array order.
        info.bmiHeader.biHeight = -height
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = BI_RGB

        buffer = ctypes.create_string_buffer(width * height * 4)
        scanlines = gdi32.GetDIBits(memory_dc, bitmap, 0, height, buffer,
                                    ctypes.byref(info), 0)
        if scanlines == 0:
            return None

        pixels = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
        # GDI hands back BGRA; drop alpha and flip to RGB.
        return pixels[:, :, [2, 1, 0]].copy()
    finally:
        gdi32.SelectObject(memory_dc, previous)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(hwnd, window_dc)
