"""Protect experiment ownership and prerequisites in the serial cell queue."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks" / "qcfuse"))
import run_cell_queue as queue


def test_missing_and_running_receipts_do_not_open_the_gate(tmp_path):
    path = tmp_path / "receipt.json"
    assert queue.read_terminal_result(path) is None
    path.write_text(json.dumps({"status": "running"}))
    assert queue.read_terminal_result(path) is None
    path.write_text('{"status": "complete",')
    assert queue.read_terminal_result(path) is None


def test_terminal_receipt_requires_verified_cache_cleanup(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"status": "complete", "cache_cleanup_verified": False}))
    with pytest.raises(RuntimeError, match="cleanup is unverified"):
        queue.read_terminal_result(path)
    path.write_text(json.dumps({"status": "failed", "cache_cleanup_verified": True}))
    assert queue.read_terminal_result(path)["status"] == "failed"


def test_idle_gpu_with_a_foreign_compute_process_is_rejected(monkeypatch):
    def query(command, **_kwargs):
        if "--query-gpu=uuid,memory.free,utilization.gpu" in command:
            return "GPU-target, 95000, 0\n"
        return "GPU-target, 12345\n"

    monkeypatch.setattr(queue.subprocess, "check_output", query)
    assert not queue.gpu_idle(4)


def test_free_target_gpu_accepts_processes_on_other_cards(monkeypatch):
    def query(command, **_kwargs):
        if "--query-gpu=uuid,memory.free,utilization.gpu" in command:
            return "GPU-target, 95000, 0\n"
        return "GPU-other, 12345\n"

    monkeypatch.setattr(queue.subprocess, "check_output", query)
    assert queue.gpu_idle(4)


def test_duplicate_outputs_fail_before_gpu_or_process_work(tmp_path, monkeypatch):
    path = tmp_path / "plan.json"
    case = {"arguments": {"output": str(tmp_path / "same.json")}}
    path.write_text(json.dumps({"cases": [case, case]}))
    monkeypatch.setattr(
        sys, "argv", ["run_cell_queue", "--plan", str(path), "--dry-run"]
    )
    with pytest.raises(ValueError, match="unique output"):
        queue.main()
