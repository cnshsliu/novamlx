#!/usr/bin/env python3
"""
Efficiently wait for a worker report file to appear.

Uses kqueue (macOS/BSD) for event-driven detection with <0.5s latency.
Falls back to tight polling (100ms) if kqueue fails.

Usage:
    python3 scripts/wait-for-report.py /tmp/worker-report-20260508-01.md
"""
import os
import select
import sys


def wait_for_file(path: str, check_interval: float = 0.1) -> bool:
    if os.path.exists(path):
        return True

    dir_path = os.path.dirname(path) or "."
    dir_fd = None
    kq = None

    try:
        dir_fd = os.open(dir_path, os.O_RDONLY)
        kq = select.kqueue()
        ev = select.kevent(
            dir_fd,
            filter=select.KQ_FILTER_VNODE,
            flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
            fflags=select.KQ_NOTE_WRITE | select.KQ_NOTE_EXTEND,
        )
        kq.control([ev], 0)

        while not os.path.exists(path):
            # Block up to check_interval; wake early on directory changes
            kq.control(None, 1, check_interval)
            if os.path.exists(path):
                return True
    except Exception:
        # Fallback to pure tight polling
        import time
        while not os.path.exists(path):
            time.sleep(check_interval)
    finally:
        if kq:
            kq.close()
        if dir_fd is not None:
            os.close(dir_fd)

    return os.path.exists(path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path>", file=sys.stderr)
        sys.exit(1)

    target = sys.argv[1]
    if wait_for_file(target):
        print("FOUND")
        sys.exit(0)
    else:
        print("MISSING")
        sys.exit(1)
