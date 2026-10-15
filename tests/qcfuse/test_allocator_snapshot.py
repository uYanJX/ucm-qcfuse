"""Allocator counters preserve interval attribution across peak resets."""

import pytest

from ucm.store.qcfuse import transfer


@pytest.mark.parametrize("reset_peak", [False, True])
def test_online_counters_remain_cumulative_across_peak_reset(monkeypatch, reset_peak):
    cuda = transfer.torch.cuda
    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "mem_get_info", lambda: (1024, 8192))
    for name in (
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(cuda, name, lambda: 256)
    calls = []
    monkeypatch.setattr(cuda, "reset_peak_memory_stats", lambda: calls.append("peak"))

    def forbidden_reset_or_sync(*args, **kwargs):
        pytest.fail("snapshot must not reset interval counters or synchronize CUDA")

    monkeypatch.setattr(cuda, "reset_accumulated_memory_stats", forbidden_reset_or_sync)
    monkeypatch.setattr(cuda, "synchronize", forbidden_reset_or_sync)
    counters = iter(
        [
            {"num_alloc_retries": 7, "num_ooms": 2, "num_device_alloc": 31},
            {"num_alloc_retries": 9, "num_ooms": 2, "num_device_alloc": 34},
        ]
    )
    monkeypatch.setattr(cuda, "memory_stats", lambda: next(counters))
    before = transfer.benchmark_snapshot(None, reset_peak)["gpu"]
    after = transfer.benchmark_snapshot(None, False)["gpu"]
    assert (
        after["allocator_counters"]["num_alloc_retries"]
        - before["allocator_counters"]["num_alloc_retries"]
        == 2
    )
    assert (
        after["allocator_counters"]["num_ooms"]
        - before["allocator_counters"]["num_ooms"]
        == 0
    )
    assert before["allocator_counters"]["num_device_free"] is None
    assert calls == (["peak"] if reset_peak else [])
    assert "not_durations" in before["allocator_counters_scope"]


def test_cpu_only_snapshot_does_not_query_cuda_stats(monkeypatch):
    monkeypatch.setattr(transfer.torch.cuda, "is_available", lambda: False)

    def unexpected_driver_call():
        pytest.fail("CPU-only snapshot must not query the CUDA driver")

    monkeypatch.setattr(transfer.torch.cuda, "memory_stats", unexpected_driver_call)
    assert transfer.benchmark_snapshot(None, True)["gpu"] is None
