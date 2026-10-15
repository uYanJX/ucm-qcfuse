"""Stop receipt-owned Linux workers before deleting their generated cache."""

import os
import signal
import time
from pathlib import Path


def worker_is_alive(identity):
    try:
        stat = Path(f"/proc/{identity['pid']}/stat").read_text()
    except FileNotFoundError:
        return False
    fields = stat[stat.rfind(")") + 2 :].split()
    return int(fields[19]) == identity["start_ticks"] and fields[0] != "Z"


def stop_owned_workers(identities, grace_seconds=30, signal_seconds=5):
    """Wait for native shutdown, then stop only matching captured identities."""
    identities = list(identities)
    actions = []

    def wait(seconds):
        deadline = time.monotonic() + seconds
        while any(worker_is_alive(x) for x in identities):
            if time.monotonic() >= deadline:
                return
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))

    wait(grace_seconds)
    for signum in (signal.SIGTERM, signal.SIGKILL):
        for identity in identities:
            if worker_is_alive(identity):
                if identity["pid"] == os.getpid():
                    raise RuntimeError("Refusing to terminate the benchmark controller")
                try:
                    os.kill(identity["pid"], signum)
                    actions.append({"pid": identity["pid"], "signal": int(signum)})
                except ProcessLookupError:
                    pass
        wait(signal_seconds)
    alive = [x for x in identities if worker_is_alive(x)]
    if alive:
        raise RuntimeError(f"Cache writers have not stopped: {alive}")
    return {"tracked_workers": len(identities), "actions": actions, "all_stopped": True}
