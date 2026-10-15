import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks/qcfuse"))
from worker_lifecycle import stop_owned_workers

pytestmark = pytest.mark.skipif(not Path("/proc").exists(), reason="Linux workers")


def identity(pid):
    stat = Path(f"/proc/{pid}/stat").read_text()
    return {"pid": pid, "start_ticks": int(stat[stat.rfind(")") + 2 :].split()[19])}


def test_cache_writer_stops_before_directory_is_removed(tmp_path):
    root = tmp_path / "cache"
    source = """import pathlib,signal,sys,time
signal.signal(signal.SIGTERM,signal.SIG_IGN)
p=pathlib.Path(sys.argv[1])
while True:
    p.mkdir(exist_ok=True)
    (p/'kv.bin').write_bytes(b'kv')
    time.sleep(0.01)
"""
    child = subprocess.Popen([sys.executable, "-c", source, str(root)])
    try:
        deadline = time.monotonic() + 5
        while not (root / "kv.bin").exists():
            assert child.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        result = stop_owned_workers([identity(child.pid)], 0.02, 0.05)
        child.wait(timeout=2)
        assert result["all_stopped"]
        assert any(x["signal"] == 9 for x in result["actions"])
        shutil.rmtree(root)
        time.sleep(0.05)
        assert not root.exists()
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


def test_reused_pid_identity_does_not_stop_another_process():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        captured = identity(child.pid)
        captured["start_ticks"] -= 1
        result = stop_owned_workers([captured], 0, 0)
        assert result["actions"] == [] and child.poll() is None
    finally:
        child.terminate()
        child.wait()


def test_worker_identity_uses_existing_snapshot_rpc():
    from types import SimpleNamespace
    from blend_common import BlendEngineBase

    calls = []
    engine = object.__new__(BlendEngineBase)
    engine.llm = SimpleNamespace(
        collective_rpc=lambda method, **kwargs: calls.append((method, kwargs))
        or [{"pid": os.getpid()}]
    )
    assert engine.worker_process_identities() == [identity(os.getpid())]
    assert calls == [("qcfuse_benchmark_snapshot", {"args": (False,)})]
