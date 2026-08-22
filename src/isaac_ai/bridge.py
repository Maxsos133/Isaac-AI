"""Socket bridge to the Isaac mod.

One TCP listener per instance, so a connection identifies which instance it came
from without any handshake bookkeeping. The mod blocks on its read, so the
game cannot advance until `send_action` is called — the environment is
synchronous by construction.
"""

from __future__ import annotations

import json
import socket
from typing import Any


class BridgeError(RuntimeError):
    """The connection to an instance failed or produced garbage."""


# A tick is ~33ms, and even a floor regeneration is a second or two. Anything
# quiet for this long is not slow, it is stuck — most likely sitting on the
# game-over screen, where Isaac stops running mod callbacks and the instance can
# never speak again. Without a ceiling here a single stuck game blocks the read
# loop forever and takes the whole fleet down with it.
GAMEPLAY_TIMEOUT_SECONDS = 45.0


class InstanceBridge:
    """The agent's end of one Isaac instance's step loop."""

    def __init__(self, port: int, index: int) -> None:
        self.port = port
        self.index = index
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", port))
        self._listener.listen(1)
        self._conn: socket.socket | None = None
        self._reader: Any = None
        self.hello: dict[str, Any] | None = None

    @property
    def connected(self) -> bool:
        return self._conn is not None

    def poll_accept(self, timeout: float) -> bool:
        """Accept a pending connection. Returns True once connected."""
        if self._conn is not None:
            return True
        self._listener.settimeout(timeout)
        try:
            conn, _ = self._listener.accept()
        except (socket.timeout, TimeoutError):
            return False
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Never wait forever on the handshake: a half-open connection here would
        # otherwise wedge the whole startup sequence.
        conn.settimeout(10.0)
        self._conn = conn
        self._reader = conn.makefile("rb")

        # The mod announces itself before the first observation.
        message = self._read_message()
        conn.settimeout(GAMEPLAY_TIMEOUT_SECONDS)
        if message.get("t") != "hello":
            raise BridgeError(f"instance {self.index}: expected hello, got {message!r}")
        self.hello = message
        return True

    def _read_message(self) -> dict[str, Any]:
        if self._reader is None:
            raise BridgeError(f"instance {self.index}: not connected")
        try:
            line = self._reader.readline()
        except (socket.timeout, TimeoutError) as exc:
            # Must stay ahead of the OSError clause below: socket.timeout is
            # TimeoutError, which is itself an OSError.
            raise BridgeError(
                f"instance {self.index}: silent for "
                f"{GAMEPLAY_TIMEOUT_SECONDS:.0f}s — most likely stuck on the "
                f"game-over screen, where mod callbacks stop running"
            ) from exc
        except OSError as exc:
            # A *crashing* game is not a closed connection. Shutting down
            # cleanly ends the stream and `readline` returns b"" below, which is
            # why this went unnoticed for thirty-odd runs — every reset until
            # now was orderly. An access violation instead drops the socket with
            # an RST, and `recv_into` raises ConnectionResetError, which is an
            # OSError and not a BridgeError, so it went straight past
            # `_receive_all`'s handler and killed the whole trainer.
            # `send` already wrapped OSError; only the read path did not, and
            # one instance faulting cost floor-v7 424,000 steps at 57% done.
            raise BridgeError(
                f"instance {self.index}: connection lost mid-read ({exc}) — "
                f"the game process most likely crashed"
            ) from exc
        if not line:
            raise BridgeError(f"instance {self.index}: connection closed by game")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise BridgeError(f"instance {self.index}: bad JSON: {exc}") from exc

    def receive(self) -> dict[str, Any]:
        """Block until the game reports its next tick."""
        return self._read_message()

    def send(self, message: dict[str, Any]) -> None:
        """Release the game to run one more tick."""
        if self._conn is None:
            raise BridgeError(f"instance {self.index}: not connected")
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode()
        try:
            self._conn.sendall(payload)
        except OSError as exc:
            raise BridgeError(f"instance {self.index}: send failed: {exc}") from exc

    def send_action(self, mx: int, my: int, sx: int, sy: int,
                    bomb: bool = False, item: bool = False) -> None:
        self.send({"t": "act", "mx": mx, "my": my, "sx": sx, "sy": sy,
                   "bomb": bomb, "item": item})

    def send_reset(self, seed: str | None = None) -> None:
        message: dict[str, Any] = {"t": "reset"}
        if seed:
            message["seed"] = seed
        self.send(message)

    def send_command(self, value: str) -> None:
        self.send({"t": "command", "value": value})

    def pump(self) -> None:
        """Let the game run one tick with neutral controls.

        A connected instance sits blocked inside its socket read until we
        answer, and a blocked game stops pumping Win32 messages — which makes
        it unfocusable and uncapturable. Anything that holds instances open
        while doing other work must keep them ticking with this.
        """
        self.receive()
        self.send({"t": "noop"})

    def reset_connection(self) -> None:
        """Drop the dead connection but keep listening on the same port.

        `close()` also closes the listener, which is right at shutdown and wrong
        when the plan is to relaunch the game behind this bridge: the port would
        have to be rebound, and on Windows the old listener lingering in
        TIME_WAIT makes that a coin flip even with SO_REUSEADDR.

        The listener has been bound since construction and never accepted more
        than one connection, so leaving it open and calling `poll_accept` again
        is all a relaunched instance needs to find its way back.
        """
        for resource in (self._reader, self._conn):
            if resource is not None:
                try:
                    resource.close()
                except OSError:
                    pass
        self._reader = None
        self._conn = None
        # The handshake is per-connection. Keeping the old one would make a
        # half-relaunched instance look connected to anything that checks it.
        self.hello = None

    def tick(self) -> None:
        """Advance one tick for an instance that is waiting on an *action*.

        `pump` reads first and then answers, which is right immediately after a
        connection: the mod sends its opening observation and blocks, so there is
        always a message waiting. Mid-training the ordering is the other way
        round. `_exchange_all` sends to everyone and then reads from everyone, so
        once a rollout step completes every instance has already handed over its
        observation and is blocked waiting to be told what to do.

        Calling `pump` on one of those waits for a message that will never
        arrive, on a socket with no timeout. That is not a slow path, it is a
        permanent hang — it is what stopped `relaunch` before it sent a single
        SPACE, and why the attempt never even hit its own deadline.
        """
        self.send({"t": "noop"})
        self.receive()

    def close(self) -> None:
        for resource in (self._reader, self._conn, self._listener):
            if resource is not None:
                try:
                    resource.close()
                except OSError:
                    pass
        self._reader = None
        self._conn = None
