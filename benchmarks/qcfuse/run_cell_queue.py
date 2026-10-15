"""Run frozen benchmark cells serially after their recorded prerequisites."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def read_terminal_result(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError:
        # Older cell runners write their terminal receipt directly. Treat an
        # in-progress write as pending, never as completed or failed cleanup.
        return None
    if result.get("status") in (None, "running"):
        return None
    if not result.get("cache_cleanup_verified"):
        raise RuntimeError(f"Cache cleanup is unverified: {path}")
    return result


def gpu_idle(index):
    row = (
        subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=uuid,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .split(",")
    )
    uuid, free, util = (value.strip() for value in row)
    if int(free) < 94000 or int(util) > 5:
        return False
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    return not any(
        line.split(",")[0].strip() == uuid for line in processes.splitlines()
    )


def cell_command(arguments):
    return [sys.executable, str(Path(__file__).with_name("run_frozen_cell.py"))] + [
        value
        for key, argument in arguments.items()
        for value in ("--" + key.replace("_", "-"), str(argument))
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    cases = plan["cases"]
    outputs = [str(Path(case["arguments"]["output"]).resolve()) for case in cases]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Each queued cell must have a unique output")
    if args.dry_run:
        for case in cases:
            print(
                json.dumps(
                    {"case": case["name"], "command": cell_command(case["arguments"])}
                )
            )
        return
    state_path = args.plan.with_suffix(".status.json")
    state = {"protocol": plan["protocol"], "status": "waiting", "cases": []}
    child = None

    def persist(**updates):
        state.update(updates, updated_at=time.time())
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)

    def stop(signum, _frame):
        persist(status="stopping", signal=signum)
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                persist(status="child_still_shutting_down", child_pid=child.pid)
                raise SystemExit(128 + signum)
        persist(status="stopped")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    persist()
    for path in plan.get("prerequisites", []):
        while (result := read_terminal_result(path)) is None:
            persist(status="waiting_for_prerequisite", prerequisite=path)
            time.sleep(30)
        if result["status"] not in ("complete", "partial"):
            persist(status="prerequisite_failed", prerequisite=path)
            return
    workspace = Path(plan["workspace"])
    index = int(plan["gpu"])
    for case in cases:
        result_path = Path(case["arguments"]["output"])
        result = read_terminal_result(result_path)
        if result is not None:
            if any(
                str(result["config"].get(k)) != str(v)
                for k, v in case["arguments"].items()
            ):
                raise ValueError(f"Existing cell configuration differs: {result_path}")
        else:
            lock = (workspace / "tmp" / f"qcfuse_gpu_{index}.lock").open("a")
            with lock:
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        persist(status="waiting_for_gpu_lock", case=case["name"])
                        time.sleep(30)
                        continue
                    if gpu_idle(index):
                        break
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    persist(status="waiting_for_idle_gpu", case=case["name"])
                    time.sleep(30)
                log_path = Path(str(result_path) + ".log")
                result_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("x") as log:
                    child = subprocess.Popen(
                        cell_command(case["arguments"]),
                        env={
                            **os.environ,
                            "CUDA_VISIBLE_DEVICES": str(index),
                            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                        },
                    cwd=Path(__file__).resolve().parents[2],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    persist(status="running", case=case["name"], child_pid=child.pid)
                    code = child.wait()
                result = read_terminal_result(result_path)
                if result is None:
                    persist(status="missing_child_receipt", returncode=code)
                    return
        receipt = {
            "case": case["name"],
            "status": result["status"],
            "result_path": str(result_path),
            "cache_cleanup_verified": True,
        }
        state["cases"].append(receipt)
        capacity_failure = (
            case["arguments"].get("storage") == "gpu"
            and "out of memory"
            in (result.get("error", "") + result.get("traceback", "")).lower()
        )
        if result["status"] not in ("complete", "partial") and not capacity_failure:
            persist(status="cell_failed", case=case["name"])
            return
        persist(status="between_cells", case=case["name"])
    persist(status="finished", child_pid=None)


if __name__ == "__main__":
    main()
