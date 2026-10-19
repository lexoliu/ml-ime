"""Release this Colab runtime once its controller stops checking in.

`provision` starts this on the runtime right after creating it. The session
that controls the runtime -- its waiter, a launcher's `status` -- touches
`HEARTBEAT` every time it looks at the runtime. A heartbeat older than the
timeout means nothing is left that would stop the runtime, so it releases
itself, whatever the kernel is doing: a runtime that outlives its controller
only bills.

    python3 watchdog.py [timeout seconds]
"""

import sys
import time
from pathlib import Path

HEARTBEAT = Path("/root/heartbeat")
#: Five of the waiter's two-minute probes.
TIMEOUT_SECONDS = 10 * 60
POLL_SECONDS = 30


def main() -> None:
    timeout = float(sys.argv[1]) if len(sys.argv) > 1 else TIMEOUT_SECONDS
    HEARTBEAT.touch()
    print(f"watching {HEARTBEAT}; releasing after {timeout:.0f}s without a beat", flush=True)
    while True:
        time.sleep(POLL_SECONDS)
        silent = time.time() - HEARTBEAT.stat().st_mtime
        if silent > timeout:
            print(f"no heartbeat for {silent:.0f}s; releasing the runtime", flush=True)
            from google.colab import runtime

            runtime.unassign()
            return


main()
