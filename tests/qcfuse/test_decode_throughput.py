"""Concurrent decode throughput uses a batch window, not summed durations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks" / "qcfuse"))
from blend_ssd import SSDPipelineEngine, _summarize_result


def test_overlapping_decode_windows_have_distinct_aggregate_rate():
    records = [
        {"ttft_seconds": 1.0, "latency_seconds": 9.0},
        {"ttft_seconds": 2.0, "latency_seconds": 7.0},
    ]
    window = SSDPipelineEngine._batch_decode_window(records)
    assert window == 8.0
    result = {
        "metric": [1.0, 1.0],
        "ttft": [1.0, 2.0],
        "first_token": [1.0, 2.0],
        "decode": [8.0, 5.0],
        "generated_tokens": [9, 6],
        "decode_tokens": [8, 5],
        "wall_seconds": 9.0,
        "qcompute_latency": [0.0, 0.0],
        "do_blend_ttft": [1.0, 2.0],
        "latency": [9.0, 7.0],
        "decode_batch_wall_seconds": window,
    }
    summary = _summarize_result(result)
    assert summary["per_sequence_decode_tokens_per_second"] == 1.0
    assert summary["aggregate_decode_tokens_per_second"] == 13.0 / 8.0


def test_single_request_window_matches_its_decode_duration():
    assert (
        SSDPipelineEngine._batch_decode_window(
            [{"ttft_seconds": 3.0, "latency_seconds": 7.0}]
        )
        == 4.0
    )
    assert SSDPipelineEngine._batch_decode_window([]) == 0.0
