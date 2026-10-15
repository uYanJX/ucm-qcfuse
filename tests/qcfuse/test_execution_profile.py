from types import SimpleNamespace

import pytest
import torch

from ucm.integration.vllm.qcfuse_worker import QcFuseWorkerExtension


def _worker(model):
    worker = QcFuseWorkerExtension()
    worker.model_runner = SimpleNamespace(model=model)
    return worker


def test_memory_breakdown_deduplicates_parameter_and_native_kv_views(monkeypatch):
    from ucm.store.qcfuse import transfer

    parameters = torch.ones(20)
    cache = torch.ones(40)
    model = SimpleNamespace(
        parameters=lambda: iter([parameters[:10], parameters[10:]]),
        named_modules=lambda: iter(
            [
                ("layer0", SimpleNamespace(kv_cache=cache[:20])),
                ("layer1", SimpleNamespace(kv_cache=[cache[20:]])),
            ]
        ),
    )
    monkeypatch.setattr(transfer, "benchmark_snapshot", lambda worker: {})
    row = _worker(model).qcfuse_memory_breakdown()
    assert row["model_parameter_storage_bytes"] == 80
    assert row["native_kv_storage_bytes"] == 160
    assert sum(item["view_bytes"] for item in row["native_kv_layers"]) == 160


def test_memory_breakdown_marks_unobserved_native_cache_unavailable(monkeypatch):
    from ucm.store.qcfuse import transfer

    monkeypatch.setattr(transfer, "benchmark_snapshot", lambda worker: {})
    row = _worker(torch.nn.Identity()).qcfuse_memory_breakdown()
    assert row["native_kv_storage_bytes"] is None


@pytest.fixture
def profiler_calls(monkeypatch):
    calls = []

    class FakeProfiler:
        def start(self):
            calls.append("start")

        def stop(self):
            calls.append("stop")

        def export_chrome_trace(self, path):
            calls.append(("export", path))

    monkeypatch.setattr(torch.profiler, "profile", lambda **kw: FakeProfiler())
    return calls


def test_profile_stops_at_next_forward_and_never_observes_later_decode(
    tmp_path, profiler_calls
):
    model = torch.nn.Identity()
    worker = _worker(model)
    path = str(tmp_path / "trace.json")
    worker.qcfuse_start_execution_profile(path, max_forwards=3)
    value = torch.ones(2)
    for _ in range(3):
        assert model(value) is value
    assert profiler_calls == ["start"]
    for _ in range(20):
        assert model(value) is value
    assert not model._forward_pre_hooks
    assert profiler_calls == ["start", "stop", ("export", path)]
    assert worker.qcfuse_stop_execution_profile()["forward_calls"] == 3
    assert profiler_calls.count("stop") == 1


def test_profile_finally_stop_handles_short_request_and_rejects_overwrite(
    tmp_path, profiler_calls
):
    model = torch.nn.Identity()
    worker = _worker(model)
    path = str(tmp_path / "trace.json")
    worker.qcfuse_start_execution_profile(path)
    with pytest.raises(RuntimeError, match="already active"):
        worker.qcfuse_start_execution_profile(str(tmp_path / "other.json"))
    model(torch.ones(1))
    row = worker.qcfuse_stop_execution_profile()
    assert row["trace_written"] and row["forward_calls"] == 1
    assert not model._forward_pre_hooks
    assert profiler_calls.count("stop") == 1
    worker.qcfuse_stop_execution_profile()
    assert profiler_calls.count("stop") == 1


def test_failed_observer_install_stops_profiler_and_allows_retry(
    tmp_path, profiler_calls
):
    worker = _worker(SimpleNamespace(register_forward_pre_hook=None))
    with pytest.raises(TypeError):
        worker.qcfuse_start_execution_profile(str(tmp_path / "failed.json"))
    assert profiler_calls == ["start", "stop"]
    worker.model_runner.model = torch.nn.Identity()
    worker.qcfuse_start_execution_profile(str(tmp_path / "retry.json"))
    worker.qcfuse_stop_execution_profile()
    assert profiler_calls.count("stop") == 2


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_profile_rejects_invalid_budget_before_start(tmp_path, profiler_calls, limit):
    with pytest.raises(ValueError, match="positive integer"):
        _worker(torch.nn.Identity()).qcfuse_start_execution_profile(
            str(tmp_path / "invalid.json"), max_forwards=limit
        )
    assert not profiler_calls


def test_profile_preserves_existing_trace(tmp_path, profiler_calls):
    path = tmp_path / "trace.json"
    path.write_text("preserved")
    with pytest.raises(FileExistsError):
        _worker(torch.nn.Identity()).qcfuse_start_execution_profile(str(path))
    assert path.read_text() == "preserved"
    assert not profiler_calls


def test_profile_rank_paths_keep_worker_traces_separate(tmp_path, profiler_calls):
    for rank in (0, 1):
        worker = _worker(torch.nn.Identity())
        worker.rank = rank
        worker.qcfuse_start_execution_profile(str(tmp_path / "trace-{rank}.json"))
        worker.qcfuse_stop_execution_profile()
    paths = [call[1] for call in profiler_calls if isinstance(call, tuple)]
    assert paths == [str(tmp_path / "trace-0.json"), str(tmp_path / "trace-1.json")]


def test_decode_interval_skips_prefill_and_removes_observer(tmp_path, profiler_calls):
    model = torch.nn.Identity()
    worker = _worker(model)
    path = str(tmp_path / "decode.json")
    worker.qcfuse_start_execution_profile(path, max_forwards=4, skip_forwards=2)
    for _ in range(2):
        model(torch.ones(5))
    assert not profiler_calls
    for _ in range(4):
        model(torch.ones(1))
    assert profiler_calls == ["start"]
    model(torch.ones(1))
    assert profiler_calls == ["start", "stop", ("export", path)]
    assert not model._forward_pre_hooks
    for _ in range(8):
        model(torch.ones(1))
    row = worker.qcfuse_stop_execution_profile()
    assert row["forward_calls"] == 4
    assert row["skipped_forward_calls"] == 2
    assert profiler_calls.count("stop") == 1


def test_short_request_never_starts_deferred_profiler(tmp_path, profiler_calls):
    model = torch.nn.Identity()
    worker = _worker(model)
    worker.qcfuse_start_execution_profile(
        str(tmp_path / "unused.json"), skip_forwards=2
    )
    model(torch.ones(5))
    row = worker.qcfuse_stop_execution_profile()
    assert not row["trace_written"] and row["forward_calls"] == 0
    assert row["skipped_forward_calls"] == 1
    assert not profiler_calls and not model._forward_pre_hooks


@pytest.mark.parametrize("skip", [-1, 1.5, True])
def test_profile_rejects_invalid_skip_before_start(tmp_path, profiler_calls, skip):
    with pytest.raises(ValueError, match="non-negative integer"):
        _worker(torch.nn.Identity()).qcfuse_start_execution_profile(
            str(tmp_path / "invalid.json"), skip_forwards=skip
        )
    assert not profiler_calls


def test_deferred_profile_failed_hook_install_does_not_stop_unstarted_profiler(
    tmp_path, profiler_calls
):
    worker = _worker(SimpleNamespace(register_forward_pre_hook=None))
    with pytest.raises(TypeError):
        worker.qcfuse_start_execution_profile(
            str(tmp_path / "failed.json"), skip_forwards=2
        )
    assert not profiler_calls


@pytest.mark.parametrize("tile", [-1, True, 1.5])
def test_ffn_projection_budget_rejects_invalid_values_before_compute(tile):
    from ucm.integration.vllm.latest_sparse import _compact_qwen_mlp_forward

    with pytest.raises(ValueError, match="non-negative integer"):
        _compact_qwen_mlp_forward(None, torch.ones(2, 3), projection_tile_tokens=tile)
