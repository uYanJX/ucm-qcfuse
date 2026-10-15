"""Measured matrix reports must preserve cohort and evidence boundaries."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks" / "qcfuse"))
from summarize_matrix import (
    aggregate_decode_rate,
    load_cell,
    summarize_receipt,
    verify_manifest,
)


def receipt():
    return {
        "baseline": "ours",
        "comparison_eligible": True,
        "measurement_kind": "quality",
        "cache_cleanup_verified": True,
        "status": "partial",
        "config": {"dataset": "longbench_v2", "batch_size": 1, "max_model_len": 71680},
        "storage_mode": "ssd",
        "ratio": 0.4,
        "requested_samples": 100,
        "unavailable_samples": [{}] * 69,
        "records": [{"source_id": str(x), "truncated": False} for x in range(31)],
        "summary": {
            "metric_mean": 14 / 31,
            "ttft_mean_seconds": 2.0,
            "ttft_p95_seconds": 3.0,
            "decode_mean_seconds": 1.0,
            "decode_tokens_per_second": 20.0,
            "aggregate_output_tokens_per_second": 15.0,
            "qcompute_mean_seconds": 0.1,
        },
    }


def test_eligible_subset_and_missing_token_counts_remain_explicit():
    row = summarize_receipt(receipt())
    assert row["unique_sources"] == 31 and row["requested_sources"] == 100
    assert row["unavailable_sources"] == 69 and row["score"] == 14 / 31
    assert row["score_scope"] == "eligible_unique_sources_only"
    assert row["actual_token_ranges"]["recomputed_tokens"] is None
    assert row["transfer"]["cache_counters"]["cuda_copy_seconds"] is None


def test_unverified_names_and_missing_quality_rows_are_rejected():
    r = receipt()
    r["baseline"] = "fuserag"
    with pytest.raises(ValueError, match="Unverified"):
        summarize_receipt(r)
    r = receipt()
    r["records"].pop()
    with pytest.raises(ValueError, match="frozen cohort"):
        summarize_receipt(r)


def test_disabled_copy_timing_is_not_reported_as_zero_work():
    r = receipt()
    r["parts"] = [
        {
            "worker_snapshots": [
                {
                    "before": [
                        {
                            "pid": 7,
                            "process_io": {"read_bytes": 100},
                            "cache_transfer_totals": {
                                "ssd_read_bytes": 200,
                                "cuda_copy_seconds": 0.0,
                            },
                        }
                    ],
                    "after": [
                        {
                            "pid": 7,
                            "process_io": {"read_bytes": 300},
                            "cache_transfer_totals": {
                                "ssd_read_bytes": 500,
                                "cuda_copy_seconds": 0.0,
                            },
                        }
                    ],
                }
            ]
        }
    ]
    row = summarize_receipt(r)
    assert row["transfer"]["worker_block_read_bytes"] == 200
    assert row["transfer"]["cache_counters"]["ssd_read_bytes"] == 300
    assert row["transfer"]["cache_counters"]["cuda_copy_seconds"] is None


def test_receipt_prompt_hash_must_match_frozen_manifest(tmp_path):
    import hashlib
    import json

    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "model_config_sha256": "model",
                "samples": [
                    {
                        "source_id": "source",
                        "input_tokens": 100,
                        "prompt_token_ids_sha256": "frozen",
                        "max_new_tokens": 4096,
                    }
                ],
            }
        )
    )
    r = {
        "manifest_path": str(path),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_config_sha256": "model",
        "records": [
            {
                "sample_index": 0,
                "source_id": "source",
                "input_tokens": 100,
                "prompt_token_ids_hash": "different",
                "sampling_max_tokens": 4096,
            }
        ],
    }
    with pytest.raises(ValueError, match="Measured prompt"):
        verify_manifest(r)
    r["records"][0]["prompt_token_ids_hash"] = "frozen"
    verify_manifest(r)


def test_serial_batch1_aggregate_uses_total_tokens_over_total_windows():
    r = receipt()
    r["records"] = [
        {"decode_tokens": 10, "decode_seconds": 1.0},
        {"decode_tokens": 90, "decode_seconds": 3.0},
    ]
    rate, scope = aggregate_decode_rate(r)
    assert rate == 25.0
    assert scope == "recomputed_from_serial_batch1_record_decode_windows"


def test_overlapping_batch2_windows_cannot_reconstruct_aggregate():
    r = receipt()
    r["config"]["batch_size"] = 2
    r["records"] = [
        {"decode_tokens": 10, "decode_seconds": 1.0},
        {"decode_tokens": 90, "decode_seconds": 3.0},
    ]
    assert aggregate_decode_rate(r) == (None, None)
    r["summary"]["aggregate_decode_tokens_per_second"] = 100 / 3
    r["summary"]["aggregate_decode_scope"] = "measured_batch_window"
    assert aggregate_decode_rate(r) == (100 / 3, "measured_batch_window")


def test_missing_batch1_measurements_remain_unavailable():
    r = receipt()
    assert aggregate_decode_rate(r) == (None, None)


@pytest.mark.parametrize(
    "record",
    [
        {"decode_tokens": -1, "decode_seconds": 1.0},
        {"decode_tokens": 1, "decode_seconds": -1.0},
        {"decode_tokens": 1, "decode_seconds": float("nan")},
        {"decode_tokens": 1, "decode_seconds": 0.0},
    ],
)
def test_invalid_batch1_measurements_are_rejected(record):
    r = receipt()
    r["records"] = [record]
    with pytest.raises(ValueError, match="Invalid measured batch1"):
        aggregate_decode_rate(r)


def unavailable_receipt(tmp_path):
    import hashlib
    import json

    samples = [
        {
            "source_id": str(i),
            "input_tokens": 100 + i,
            "prompt_token_ids_sha256": str(i),
            "max_new_tokens": 8,
        }
        for i in range(2)
    ]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"model_config_sha256": "model", "samples": samples}))
    return {
        "manifest_path": str(path),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_config_sha256": "model",
        "measurement_kind": "quality",
        "status": "N/A",
        "requested_samples": 2,
        "records": [],
        "unavailable_samples": [
            {"sample_index": i, "source_id": str(i), "input_tokens": 100 + i}
            for i in range(2)
        ],
    }


def test_all_unavailable_quality_requires_full_original_source_accounting(tmp_path):
    r = unavailable_receipt(tmp_path)
    verify_manifest(r)
    r["unavailable_samples"].pop()
    with pytest.raises(ValueError, match="original frozen cohort"):
        verify_manifest(r)


@pytest.mark.parametrize(
    "field,value", [("source_id", "replacement"), ("input_tokens", 1)]
)
def test_unavailable_source_cannot_replace_or_shorten_original(tmp_path, field, value):
    r = unavailable_receipt(tmp_path)
    r["unavailable_samples"][0][field] = value
    with pytest.raises(ValueError, match="Unavailable source differs"):
        verify_manifest(r)


def test_unavailable_source_cannot_be_duplicated_or_also_measured(tmp_path):
    r = unavailable_receipt(tmp_path)
    r["unavailable_samples"][1] = dict(r["unavailable_samples"][0])
    with pytest.raises(ValueError, match="duplicated or also measured"):
        verify_manifest(r)
    r = unavailable_receipt(tmp_path)
    r["records"] = [
        {
            "sample_index": 0,
            "source_id": "0",
            "input_tokens": 100,
            "prompt_token_ids_hash": "0",
            "sampling_max_tokens": 8,
        }
    ]
    with pytest.raises(ValueError, match="duplicated or also measured"):
        verify_manifest(r)
    r["unavailable_samples"].pop(0)
    r["status"] = "partial"
    verify_manifest(r)


def test_pending_capacity_events_are_not_promoted_to_measured_results(tmp_path):
    path = tmp_path / "cell.json"
    Path(str(path) + ".unavailable.jsonl").write_text('{"status":"N/A"}\n')
    row = load_cell(path)
    assert row["status"] == "partial_unavailable"
    assert row["unavailable_observations"] == 1 and row["measured_observations"] == 0
    assert "score" not in row and "ttft_seconds" not in row
