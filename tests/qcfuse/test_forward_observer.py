"""Observe true model input sizes without changing forwards or timing cells."""

from types import SimpleNamespace
import sys
from pathlib import Path

import torch
import pytest

from ucm.integration.vllm.qcfuse_worker import QcFuseWorkerExtension


class Model(torch.nn.Module):
    def forward(self, input_ids=None, inputs_embeds=None):
        return input_ids if input_ids is not None else inputs_embeds


def test_observer_captures_positional_and_keyword_rows_then_resets():
    worker = QcFuseWorkerExtension()
    worker.model_runner = SimpleNamespace(model=Model())
    worker.qcfuse_start_forward_observer()
    ids = torch.arange(11)
    assert torch.equal(worker.model_runner.model(ids), ids)
    worker.model_runner.model(input_ids=torch.arange(23))
    worker.model_runner.model(inputs_embeds=torch.ones(3, 4))
    stats = worker._qcfuse_forward_observation
    assert stats["max_input_tokens"] == 23 and stats["forward_calls"] == 3
    assert stats["input_token_histogram"] == {"11": 1, "23": 1, "3": 1}
    worker.qcfuse_start_forward_observer()
    worker.model_runner.model(ids)
    assert worker._qcfuse_forward_observation["forward_calls"] == 1


def test_timing_snapshot_does_not_install_observer():
    sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks" / "qcfuse"))
    from blend_ssd import SSDPipelineEngine

    engine = object.__new__(SSDPipelineEngine)
    engine.worker_snapshots_enabled = True
    engine.forward_observer_enabled = False
    calls = []
    engine.llm = SimpleNamespace(
        collective_rpc=lambda name, **kw: calls.append(name) or []
    )
    engine._worker_snapshot(True)
    assert calls == ["qcfuse_benchmark_snapshot"]
    engine.forward_observer_enabled = True
    engine._worker_snapshot(True)
    assert calls[-2:] == ["qcfuse_start_forward_observer", "qcfuse_benchmark_snapshot"]


def test_snapshot_closes_only_its_observer(monkeypatch):
    from ucm.store.qcfuse import transfer

    worker = QcFuseWorkerExtension()
    worker.model_runner = SimpleNamespace(model=Model())
    monkeypatch.setattr(transfer, "benchmark_snapshot", lambda *args: {})
    worker.qcfuse_start_forward_observer()
    worker.model_runner.model(torch.arange(11))
    snapshot = worker.qcfuse_benchmark_snapshot(False)
    worker.model_runner.model(torch.arange(17))
    assert snapshot["forward_observation"]["max_input_tokens"] == 11
    assert worker._qcfuse_forward_observation["forward_calls"] == 1


def test_resident_prepare_with_observation_uses_resident_rpc(tmp_path):
    sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks" / "qcfuse"))
    from blend_ssd import SSDPipelineEngine

    engine = object.__new__(SSDPipelineEngine)
    engine.storage_mode = "cpu"
    engine.baseline = "ours"
    engine.forward_observer_enabled = True
    (tmp_path / "packed.bin").write_bytes(b"cache")
    calls = []
    engine.llm = SimpleNamespace(
        collective_rpc=lambda name, **kw: calls.append(name) or []
    )
    sample = SimpleNamespace(sample_dir_chunk=tmp_path, sample_dir_query=tmp_path)
    assert engine._prepare_resident_batch([sample])["files"] == 1
    assert calls == ["qcfuse_prepare_resident_batch"]


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_only_gpu_preload_oom_is_tagged_before_online_submission(tmp_path, mode):
    sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks" / "qcfuse"))
    from blend_ssd import GpuResidentPreloadCapacityError, SSDPipelineEngine

    engine = object.__new__(SSDPipelineEngine)
    engine.storage_mode = mode
    engine.baseline = "ours"
    (tmp_path / "cache.bin").write_bytes(b"cache")

    def failed_rpc(*args, **kwargs):
        raise RuntimeError("Worker failed: CUDA out of memory")

    engine.llm = SimpleNamespace(collective_rpc=failed_rpc)
    sample = SimpleNamespace(sample_dir_chunk=tmp_path, sample_dir_query=tmp_path)
    with pytest.raises(RuntimeError) as raised:
        engine._prepare_resident_batch([sample])
    assert isinstance(raised.value, GpuResidentPreloadCapacityError) == (mode == "gpu")


def test_non_capacity_rpc_failure_is_not_tagged_as_unsupported_input(tmp_path):
    sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks" / "qcfuse"))
    from blend_ssd import GpuResidentPreloadCapacityError, SSDPipelineEngine

    engine = object.__new__(SSDPipelineEngine)
    engine.storage_mode = "gpu"
    engine.baseline = "ours"
    (tmp_path / "cache.bin").write_bytes(b"cache")

    def failed_rpc(*args, **kwargs):
        raise RuntimeError("Invalid cache metadata")

    engine.llm = SimpleNamespace(collective_rpc=failed_rpc)
    sample = SimpleNamespace(sample_dir_chunk=tmp_path, sample_dir_query=tmp_path)
    with pytest.raises(RuntimeError) as raised:
        engine._prepare_resident_batch([sample])
    assert not isinstance(raised.value, GpuResidentPreloadCapacityError)


def test_snapshot_adds_cumulative_transfer_metrics_without_gpu_timing(monkeypatch):
    from ucm.store.qcfuse import transfer
    from ucm.sparse import state

    worker = QcFuseWorkerExtension()
    monkeypatch.setattr(
        transfer, "benchmark_snapshot", lambda *args: {"io_scope": "worker"}
    )
    metrics = {"ssd_read_bytes": 1234.0, "ssd_wait_seconds": 0.25}
    sparse = SimpleNamespace(_store=SimpleNamespace(metrics=lambda: dict(metrics)))
    monkeypatch.setattr(state, "has_ucm_sparse", lambda: True)
    monkeypatch.setattr(state, "get_ucm_sparse", lambda: sparse)
    snapshot = worker.qcfuse_benchmark_snapshot(False)
    assert snapshot["cache_transfer_totals"] == metrics
    assert snapshot["cache_transfer_scope"].startswith("cumulative_completed_request")


def test_snapshot_discards_warmup_counts_and_drains_only_measured_requests(monkeypatch):
    from collections import deque
    from ucm.store.qcfuse import transfer
    from ucm.sparse import state
    from ucm.sparse.qcfuse.qcfuse import QcFuse

    sparse = object.__new__(QcFuse)
    sparse._completed_token_work = deque([{"client_request_id": "warmup"}], maxlen=256)
    worker = QcFuseWorkerExtension()
    monkeypatch.setattr(transfer, "benchmark_snapshot", lambda *args: {})
    monkeypatch.setattr(state, "has_ucm_sparse", lambda: True)
    monkeypatch.setattr(state, "get_ucm_sparse", lambda: sparse)
    assert "token_work" not in worker.qcfuse_benchmark_snapshot(True)
    sparse._completed_token_work.extend(
        [{"client_request_id": "query"}, {"client_request_id": "blend"}]
    )
    assert worker.qcfuse_benchmark_snapshot(False)["token_work"] == [
        {"client_request_id": "query"},
        {"client_request_id": "blend"},
    ]
    assert worker.qcfuse_benchmark_snapshot(False)["token_work"] == []
