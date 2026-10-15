from types import SimpleNamespace

import pytest
import torch

from ucm.integration.vllm.qcfuse_graph import (
    decode_graph_compilation_config,
    install_v2_graph_hooks,
)
from ucm.integration.vllm.qcfuse_worker import QcFuseWorkerExtension


def _runner(events):
    manager = SimpleNamespace(dispatch=lambda *args, **kwargs: "native-descriptor")
    connector = SimpleNamespace(
        pre_forward=lambda *args, **kwargs: events.append("bind")
    )
    runner = SimpleNamespace(
        kv_connector=connector,
        cudagraph_manager=manager,
        req_states="request-states",
        _ucm_latest_input_batch="input-batch",
        _ucm_latest_sparse_meta_built_for=None,
    )
    return runner


def test_replay_boundary_refreshes_sparse_state_once_after_connector_bind():
    events = []
    runner = _runner(events)
    scheduler = SimpleNamespace()
    sparse = SimpleNamespace(build_sparse_meta=lambda *args: events.append(args))
    install_v2_graph_hooks(runner, lambda: True, lambda: sparse)
    install_v2_graph_hooks(runner, lambda: True, lambda: sparse)
    runner.kv_connector.pre_forward(
        scheduler_output=scheduler, attn_metadata="metadata"
    )
    runner.kv_connector.pre_forward(
        scheduler_output=scheduler, attn_metadata="metadata"
    )
    assert events == [
        "bind",
        (scheduler, "request-states", "input-batch", "metadata"),
        "bind",
    ]
    assert runner._ucm_latest_sparse_meta_built_for is scheduler


def test_next_replay_clears_previous_prefill_state():
    events = []
    runner = _runner(events)
    sparse = SimpleNamespace(active=True)

    def build(scheduler, *args):
        sparse.active = scheduler.is_prefill

    sparse.build_sparse_meta = build
    install_v2_graph_hooks(runner, lambda: True, lambda: sparse)
    for active in (True, False, False):
        runner.kv_connector.pre_forward(
            scheduler_output=SimpleNamespace(is_prefill=active), attn_metadata={}
        )
        assert sparse.active is active


def test_eager_context_metadata_uses_same_boundary(monkeypatch):
    import vllm.forward_context

    events = []
    runner = _runner(events)
    sparse = SimpleNamespace(build_sparse_meta=lambda *args: events.append(args[-1]))
    monkeypatch.setattr(
        vllm.forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata="eager"),
    )
    install_v2_graph_hooks(runner, lambda: True, lambda: sparse)
    runner.kv_connector.pre_forward(scheduler_output=SimpleNamespace())
    assert events == ["bind", "eager"]


@pytest.mark.parametrize("tokens", [1, 2, 4])
def test_sparse_prefill_never_selects_captured_decode(tokens):
    from vllm.config.compilation import CUDAGraphMode

    runner = _runner([])
    runner._ucm_latest_pending_scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"q": tokens},
        kv_connector_metadata=SimpleNamespace(
            request_meta={"q": SimpleNamespace(state=object())}
        ),
    )
    install_v2_graph_hooks(runner, lambda: True, lambda: None)
    desc = runner.cudagraph_manager.dispatch(1, tokens, tokens, max_query_len=None)
    assert desc.cg_mode == CUDAGraphMode.NONE
    assert desc.num_tokens == tokens and desc.num_reqs == 1
    assert (
        runner.cudagraph_manager.dispatch(1, 1, 1, max_query_len=1)
        == "native-descriptor"
    )


def test_non_qcfuse_and_dummy_dispatch_are_preserved():
    runner = _runner([])
    install_v2_graph_hooks(runner, lambda: False, lambda: None)
    assert runner.cudagraph_manager.dispatch(1, 1, 1) == "native-descriptor"
    runner.kv_connector.pre_forward(
        scheduler_output=SimpleNamespace(), attn_metadata={}
    )


def test_execution_observer_reports_actual_replays_and_eager_shapes():
    worker = QcFuseWorkerExtension()
    model = torch.nn.Identity()
    desc = SimpleNamespace(num_tokens=1)
    manager = SimpleNamespace(
        graphs={}, cudagraph_mode="FULL_DECODE_ONLY", run_fullgraph=lambda _: "result"
    )
    worker.model_runner = SimpleNamespace(model=model, cudagraph_manager=manager)
    worker.qcfuse_start_execution_mode_observer()
    worker.qcfuse_start_execution_mode_observer()
    model(torch.ones(17))
    assert manager.run_fullgraph(desc) == "result"
    state = worker._qcfuse_execution_modes
    assert state["eager_model_forwards"] == 1 and state["full_graph_replays"] == 1
    assert state["eager_input_token_histogram"] == {"17": 1}
    assert len(model._forward_pre_hooks) == 1


def test_forward_profile_rejects_replay_without_arming_unbounded_hook(tmp_path):
    worker = QcFuseWorkerExtension()
    model = torch.nn.Identity()
    worker.model_runner = SimpleNamespace(
        model=model, cudagraph_manager=SimpleNamespace(graphs={"captured": object()})
    )
    with pytest.raises(ValueError, match="requires eager"):
        worker.qcfuse_start_execution_profile(str(tmp_path / "unused.json"))
    assert not model._forward_pre_hooks
    assert not (tmp_path / "unused.json").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_public_pipeline_constructor_forwards_graph_option(monkeypatch, enabled):
    from pathlib import Path

    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[2] / "benchmarks" / "qcfuse")
    )
    from blend_common import BlendEngineBase
    from blend_ssd import SSDPipelineEngine

    calls = []
    monkeypatch.setattr(
        BlendEngineBase, "__init__", lambda self, *args, **kwargs: calls.append(kwargs)
    )
    engine = SSDPipelineEngine("model", decode_cudagraph=enabled)
    assert calls[0]["decode_cudagraph"] is enabled
    assert engine.storage_mode == "ssd"


def test_zero_token_cleanup_does_not_require_forward_context(monkeypatch):
    import vllm.forward_context

    def forbidden():
        raise AssertionError("No forward context exists during cleanup")

    monkeypatch.setattr(vllm.forward_context, "get_forward_context", forbidden)
    events = []
    runner = _runner(events)
    sparse = SimpleNamespace(build_sparse_meta=lambda *args: forbidden())
    install_v2_graph_hooks(runner, lambda: True, lambda: sparse)
    scheduler = SimpleNamespace(total_num_scheduled_tokens=0)
    runner.kv_connector.pre_forward(scheduler)
    assert events == ["bind"]
    assert runner._ucm_latest_sparse_meta_built_for is None


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_piecewise_options_keep_compile_disabled_and_bound_capture(batch):
    config = decode_graph_compilation_config("piecewise", batch)
    assert config["mode"] == 0
    assert config["cudagraph_mode"] == "PIECEWISE"
    assert config["cudagraph_capture_sizes"] == list(range(1, batch + 1))
    assert config["max_cudagraph_capture_size"] == batch
    assert (
        decode_graph_compilation_config("full", batch)["cudagraph_mode"]
        == "FULL_DECODE_ONLY"
    )


@pytest.mark.parametrize("mode,batch", [("compiled", 1), ("full", True), ("full", 0)])
def test_invalid_decode_graph_configuration_is_rejected(mode, batch):
    with pytest.raises(ValueError):
        decode_graph_compilation_config(mode, batch)


def test_piecewise_observer_counts_actual_replay_and_lazy_entries():
    events = []
    capture = SimpleNamespace(
        replay=lambda: events.append("real-replay"), _num_graphs=2, _num_eager_breaks=1
    )
    entries = {}
    manager = SimpleNamespace(
        graphs={}, cudagraph_mode="PIECEWISE", run_fullgraph=lambda _: None
    )
    manager.breakable_cg_runner = SimpleNamespace(entries=entries)

    def run_piecewise(*args):
        if not entries:
            entries[1] = SimpleNamespace(
                capture=capture, batch_descriptor=SimpleNamespace(num_tokens=1)
            )
        else:
            capture.replay()

    manager.run_pw_graph = run_piecewise
    worker = QcFuseWorkerExtension()
    worker.model_runner = SimpleNamespace(
        model=torch.nn.Identity(), cudagraph_manager=manager
    )
    worker.qcfuse_start_execution_mode_observer()
    manager.run_pw_graph(None)
    assert worker._qcfuse_execution_modes["piecewise_graph_replays"] == 0
    manager.run_pw_graph(None)
    manager.run_pw_graph(None)
    assert events == ["real-replay", "real-replay"]
    state = worker._qcfuse_execution_modes
    assert state["piecewise_graph_replays"] == 2
    assert state["piecewise_segment_replays"] == 4
    assert state["piecewise_eager_break_calls"] == 2
    assert state["captured_token_counts"] == [1]
