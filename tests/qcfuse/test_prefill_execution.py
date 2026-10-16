from types import SimpleNamespace

import pytest
import torch

from ucm.sparse.qcfuse.qcfuse import QcFuse


@pytest.mark.parametrize("policy", ["", "native", "invalid"])
def test_invalid_prefill_policy_is_rejected_before_model_work(monkeypatch, policy):
    from ucm.sparse.qcfuse.qcfuse import UcmSparseBase

    monkeypatch.setattr(UcmSparseBase, "__init__", lambda *args: None)
    monkeypatch.setenv("QCFUSE_PREFILL_MEMORY_POLICY", policy)
    config = SimpleNamespace(
        device_config=SimpleNamespace(device="cuda"),
        cache_config=SimpleNamespace(block_size=16),
    )
    with pytest.raises(ValueError, match="QCFUSE_PREFILL_MEMORY_POLICY"):
        QcFuse(config, None)


def test_mlp_observer_counts_actual_tiles_without_changing_values():
    from ucm.integration.vllm.latest_sparse import _compact_qwen_mlp_forward

    torch.manual_seed(2816)
    inputs = torch.randn(23, 8)
    up, down = torch.randn(24, 8), torch.randn(8, 12)

    def operation(out, projection):
        out.copy_(torch.nn.functional.silu(projection[:, :12]) * projection[:, 12:])

    mlp = SimpleNamespace(
        gate_up_proj=lambda value: (torch.nn.functional.linear(value, up), None),
        act_fn=SimpleNamespace(op=operation),
        down_proj=SimpleNamespace(weight=down),
    )
    sparse = object.__new__(QcFuse)
    sparse._prefill_memory_policy = "compact"
    before = sparse.prefill_execution_stats()
    expected = _compact_qwen_mlp_forward(mlp, inputs.clone(), projection_tile_tokens=6)
    actual = _compact_qwen_mlp_forward(
        mlp,
        inputs.clone(),
        projection_tile_tokens=6,
        on_projection_tile=lambda rows: sparse.record_compact_prefill("ffn_tile", rows),
    )
    assert torch.equal(actual, expected)
    after = sparse.prefill_execution_stats()
    assert before["counters"]["ffn_tile"] == {"calls": 0, "rows": 0}
    assert not before["ffn_tile_row_histogram"]
    assert after["counters"]["ffn_tile"]["rows"] == len(inputs)
    histogram = after["ffn_tile_row_histogram"]
    assert sum(int(rows) * count for rows, count in histogram.items()) == len(inputs)
    assert max(map(int, histogram)) <= 6
    assert sum(histogram.values()) == after["counters"]["ffn_tile"]["calls"]
    assert after["counters"]["ffn_layer"]["calls"] == 0


def test_worker_exposes_independent_prefill_counter_snapshots(monkeypatch):
    from ucm.integration.vllm.qcfuse_worker import QcFuseWorkerExtension
    from ucm.sparse import state
    from ucm.store.qcfuse import transfer

    sparse = object.__new__(QcFuse)
    wrapper = SimpleNamespace(prefill_execution_stats=sparse.prefill_execution_stats)
    monkeypatch.setattr(transfer, "benchmark_snapshot", lambda *args: {})
    monkeypatch.setattr(state, "has_ucm_sparse", lambda: True)
    monkeypatch.setattr(state, "get_ucm_sparse", lambda: wrapper)
    worker = QcFuseWorkerExtension()
    before = worker.qcfuse_benchmark_snapshot(True)
    sparse.record_compact_prefill("qkv_values", 17)
    after = worker.qcfuse_benchmark_snapshot(False)
    assert before["prefill_execution"]["counters"]["qkv_values"] == {
        "calls": 0,
        "rows": 0,
    }
    assert after["prefill_execution"]["counters"]["qkv_values"] == {
        "calls": 1,
        "rows": 17,
    }
