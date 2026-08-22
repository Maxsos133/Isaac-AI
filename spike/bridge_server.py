"""Synchronous-stepping probe server.

Accepts one connection per Isaac instance and answers every state line with an
action line. Measures how many closed-loop agent steps per second the game can
actually sustain, which is the number the whole training plan depends on.
"""

from __future__ import annotations

import socket
import sys
import threading
import time

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 900
INSTANCES = int(sys.argv[3]) if len(sys.argv) > 3 else 1

# Walk a square so the movement is obvious on screen, and shoot right the
# whole time so we can confirm both control axes are driven independently.
PATTERN = [(1, 0), (0, 1), (-1, 0), (0, -1)]

results: list[tuple[int, int, float, float]] = []
lock = threading.Lock()


def serve(conn: socket.socket, index: int) -> None:
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    reader = conn.makefile("rb")
    started = time.perf_counter()
    latencies: list[float] = []
    steps = 0
    first_pos: tuple[float, float] | None = None
    last_pos: tuple[float, float] | None = None
    travelled = 0.0

    try:
        while steps < STEPS:
            line = reader.readline()
            if not line:
                break
            t0 = time.perf_counter()

            # state is: step;px;py;hearts;enemies;cleared
            parts = line.decode().strip().split(";")
            if len(parts) >= 3:
                pos = (float(parts[1]), float(parts[2]))
                if first_pos is None:
                    first_pos = pos
                if last_pos is not None:
                    travelled += abs(pos[0] - last_pos[0]) + abs(pos[1] - last_pos[1])
                last_pos = pos

            mx, my = PATTERN[(steps // 25) % len(PATTERN)]
            conn.sendall(f"{mx},{my},1,0\n".encode())
            latencies.append(time.perf_counter() - t0)
            steps += 1
    except OSError as exc:
        print(f"[{index}] connection error after {steps} steps: {exc}")
    finally:
        elapsed = time.perf_counter() - started
        mean_us = (sum(latencies) / len(latencies) * 1e6) if latencies else 0.0
        with lock:
            results.append((index, steps, elapsed, mean_us))
        rate = steps / elapsed if elapsed > 0 else 0.0
        print(
            f"[{index}] {steps} steps in {elapsed:.2f}s = {rate:.1f} steps/s, "
            f"turnaround {mean_us:.0f}us, player travelled {travelled:.0f}px "
            f"({'MOVED' if travelled > 100 else 'DID NOT MOVE'})"
        )
        conn.close()


def main() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", PORT))
    listener.listen(INSTANCES + 4)
    listener.settimeout(90.0)
    print(f"listening on 127.0.0.1:{PORT}, expecting {INSTANCES} instance(s)")

    threads = []
    for index in range(INSTANCES):
        try:
            conn, _ = listener.accept()
        except socket.timeout:
            print("timed out waiting for a connection")
            break
        print(f"[{index}] connected")
        thread = threading.Thread(target=serve, args=(conn, index), daemon=True)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join(timeout=180.0)

    if results:
        total = sum(steps / elapsed for _, steps, elapsed, _ in results if elapsed > 0)
        print(f"\nAGGREGATE across {len(results)} instance(s): {total:.1f} agent steps/s")
        print(f"projected steps/hour: {total * 3600:,.0f}")


if __name__ == "__main__":
    main()
