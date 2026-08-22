"""Frame capture for the pixel student.

The teacher reads privileged state from the mod. The student has to learn from
what a human sees, so it needs the game's own picture, one frame per step, for
all twelve instances, without slowing the fleet down.

`PrintWindow` cannot do this: it needs the target to pump Win32 messages, and an
instance blocked inside MC_POST_UPDATE waiting for its action pumps nothing, so
the call hangs. Windows Graphics Capture reads the compositor's surface instead
and does not care what the game's message loop is doing. Measured at 12
instances: 360.0 game ticks/s with capture on against 359.9 with it off.

Two properties of this are worth knowing before reading the code:

  * Capture is *pull* from a *push* source. WGC delivers a frame whenever the
    window presents one, on its own thread. A blocked instance is not
    presenting, so what a step reads is the frame from the previous tick —
    14ms old at the median against a 33ms tick, but never the current one.
  * WGC hands back the whole window, title bar included. That is 11% of the
    frame at the configured window size, so it is cropped to the client area
    before anything else sees it.
"""

from __future__ import annotations

from pathlib import Path
import threading

import cv2
import numpy as np
from windows_capture import WindowsCapture

from isaac_ai.config import PixelConfig
from isaac_ai.windows import client_rect_in_capture

# cv2's INTER_AREA over an exact integer ratio is a plain box filter, and it is
# 22x faster than the equivalent numpy reshape-and-mean (0.112ms against 2.48ms
# per frame; at 12 instances that is 1.3ms against 29.7ms of every step). The
# obvious dependency-free version is the one that would not fit in the budget.
INTERPOLATION = cv2.INTER_AREA


def downsample(frame: np.ndarray, width: int, height: int,
               grayscale: bool) -> np.ndarray:
    """Shrink one client-area frame to the student's input size."""
    small = cv2.resize(frame, (width, height), interpolation=INTERPOLATION)
    if grayscale:
        small = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)[:, :, None]
    return small


def save_stack_image(stack: np.ndarray, channels: int, path: Path) -> None:
    """Write one instance's frame stack as a strip, oldest frame leftmost.

    Worth having because the student's view drifts in ways the privileged
    observation cannot show: rooms fill with blood and gore as encounters run,
    and blood only clears when the player dies. Whether enemies stay legible
    against a red floor at this resolution is a question about pixels, so it
    has to be answered by looking at the pixels.
    """
    frames = [stack[i * channels:(i + 1) * channels].transpose(1, 2, 0)
              for i in range(stack.shape[0] // channels)]
    strip = np.concatenate(frames, axis=1)
    if channels == 1:
        strip = np.repeat(strip, 3, axis=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written at 3x nearest-neighbour: a 160x90 thumbnail is unreadable on
    # screen, and the question is whether the content survives, not the size.
    cv2.imwrite(str(path), cv2.resize(
        strip[:, :, ::-1], (strip.shape[1] * 3, strip.shape[0] * 3),
        interpolation=cv2.INTER_NEAREST))


class FrameGrabber:
    """A capture session for one instance, holding only its most recent frame.

    The session runs on its own thread and overwrites a single slot. Keeping a
    queue would be wrong: a step wants the newest picture, and an older one is
    not worth the memory it sits in.
    """

    def __init__(self, hwnd: int, pixels: PixelConfig) -> None:
        self.hwnd = hwnd
        self.pixels = pixels
        self.frames = 0

        # Resolved once. Windows are laid out at bring-up and never moved during
        # training, and doing it here keeps three Win32 calls off a path that
        # runs 36 times a second per instance.
        self._crop = client_rect_in_capture(hwnd)
        pixels.check_divides(self._crop[2], self._crop[3])

        # How often a step asked for a frame and got one it had already been
        # given. Not an error — the stack is meant to track agent steps, not
        # distinct frames — but it measures whether capture is keeping up, and
        # a high rate means the stack is carrying less motion than it looks.
        self.repeats = 0

        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._latest_id = -1
        self._cached: np.ndarray | None = None
        self._cached_id = -2
        self._served_id = -3

        session = WindowsCapture(
            cursor_capture=False,
            # The capture highlight is composited into the frame on Windows 11,
            # so leaving it on would put a moving yellow border in the student's
            # observation.
            draw_border=False,
            window_hwnd=hwnd,
        )

        @session.event
        def on_frame_arrived(frame, capture_control):  # noqa: ANN001
            # frame_buffer is a zero-copy view over a native mapping released
            # when this returns, so the crop has to be copied out. Fancy
            # indexing on the channel axis (BGRA -> RGB) already produces an
            # owned contiguous array, which is the copy. Cropping here rather
            # than on read means the ~11% of every frame that is title bar is
            # never carried around at all.
            x, y, width, height = self._crop
            client = frame.frame_buffer[y:y + height, x:x + width, [2, 1, 0]]
            with self._lock:
                self._latest = client
                self.frames += 1
                self._latest_id = self.frames

        @session.event
        def on_closed():  # noqa: ANN202
            pass

        self._control = session.start_free_threaded()

    def latest(self) -> np.ndarray | None:
        """The newest frame at the student's input size, or None if none yet.

        The downsample is cached against the frame counter so a step that reads
        twice, or reads faster than the game presents, does the work once.
        """
        with self._lock:
            frame, frame_id = self._latest, self._latest_id
            # Only count a repeat once there is something to repeat, or a
            # session still warming up would report every read as stale.
            if frame is not None and frame_id == self._served_id:
                self.repeats += 1
            self._served_id = frame_id
            if frame_id == self._cached_id:
                return self._cached
        if frame is None:
            return None

        small = downsample(frame, self.pixels.width, self.pixels.height,
                           self.pixels.grayscale)
        with self._lock:
            self._cached, self._cached_id = small, frame_id
        return small

    def stop(self) -> None:
        try:
            self._control.stop()
        except Exception:  # noqa: BLE001 - shutdown must not mask a result
            pass


class FleetCapture:
    """Frames for a whole fleet, stacked, as one array per step.

    A single frame carries no velocity: a tear in flight and a tear sitting
    still look identical, and so do a charger winding up and one at rest. The
    stack is what makes motion visible to a feed-forward encoder.
    """

    def __init__(self, hwnds: list[int | None], pixels: PixelConfig) -> None:
        self.pixels = pixels
        self.num_envs = len(hwnds)
        self.channels = 1 if pixels.grayscale else 3
        self.grabbers: list[FrameGrabber | None] = [
            FrameGrabber(hwnd, pixels) if hwnd else None for hwnd in hwnds
        ]
        self._history = np.zeros(
            (self.num_envs, pixels.stack * self.channels,
             pixels.height, pixels.width), dtype=np.uint8)
        self._needs_fill = np.ones(self.num_envs, dtype=bool)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.pixels.stack * self.channels,
                self.pixels.height, self.pixels.width)

    def observe(self) -> np.ndarray:
        """Pull one frame per instance and roll it into the stack.

        An instance that has no frame yet — a session still warming up, or a
        window that has gone — keeps its previous stack rather than being
        zeroed, so a single dropped frame does not read to the network as the
        screen having gone black.
        """
        for index, grabber in enumerate(self.grabbers):
            if grabber is None:
                continue
            frame = grabber.latest()
            if frame is not None:
                self.push(index, frame)
        return self._history.copy()

    def push(self, index: int, frame: np.ndarray) -> None:
        """Roll one frame into an instance's stack, oldest first.

        Separate from `observe` so the ordering can be tested without a live
        window — a stack that is silently newest-first still trains, just on
        time running backwards.
        """
        if self._needs_fill[index]:
            # A fresh episode has no history, and leaving the empty slots black
            # would teach the network that every encounter is preceded by three
            # frames of darkness — a cue that is perfectly predictive of "an
            # episode just started" and exists nowhere at deployment. Repeating
            # the first frame says the true thing instead: nothing has moved yet.
            self._history[index] = np.tile(frame.transpose(2, 0, 1),
                                           (self.pixels.stack, 1, 1))
            self._needs_fill[index] = False
            return

        self._history[index] = np.roll(
            self._history[index], -self.channels, axis=0)
        self._history[index, -self.channels:] = frame.transpose(2, 0, 1)

    def reset(self, mask: np.ndarray | None = None) -> None:
        """Clear stacked history where an episode has just ended.

        Without this the first observation of a new episode is three frames of
        the previous one, which reads as a scene that teleported.
        """
        if mask is None:
            self._history[:] = 0
            self._needs_fill[:] = True
            return
        for index in np.flatnonzero(mask):
            self._history[int(index)] = 0
            self._needs_fill[int(index)] = True

    def stop(self) -> None:
        for grabber in self.grabbers:
            if grabber is not None:
                grabber.stop()
