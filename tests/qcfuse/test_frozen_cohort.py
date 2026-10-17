"""Batch comparisons must preserve the complete fixed measurement cohort."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks" / "qcfuse"))
from run_frozen_cell import gpu_preload_unavailable, measurement_rows


def test_batch_comparison_uses_identical_source_ids():
    rows = [{"source_id": index} for index in range(4)]
    for batch in (1, 2, 4):
        assert measurement_rows(rows, "timing", batch, 4) == rows
    assert measurement_rows(rows, "quality", 1) == rows


@pytest.mark.parametrize("kind,count", [("quality", 4), ("timing", 0), ("timing", 5)])
def test_invalid_timing_cohort_is_rejected(kind, count):
    with pytest.raises(ValueError, match="available cohort"):
        measurement_rows([0, 1, 2, 3], kind, 1, count)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_fixed_audit_cohort_covers_every_batch_and_preserves_default(batch):
    rows = [{"source_id": index} for index in range(4)]
    assert measurement_rows(rows, "audit", batch, audit_samples=4) == rows
    assert measurement_rows(rows, "audit", batch) == rows[:batch]
    assert measurement_rows(rows, "timing", batch) == rows[:batch]


@pytest.mark.parametrize(
    "kind,count", [("quality", 4), ("timing", 4), ("audit", 0), ("audit", 5)]
)
def test_invalid_audit_cohort_is_rejected(kind, count):
    with pytest.raises(ValueError, match="available cohort"):
        measurement_rows([0, 1, 2, 3], kind, 1, audit_samples=count)


def test_audit_and_timing_cohorts_cannot_be_combined():
    with pytest.raises(ValueError, match="mutually exclusive"):
        measurement_rows([0, 1, 2, 3], "audit", 1, 4, 4)


def test_cache_workspace_lease_excludes_other_process_then_releases(tmp_path):
    import subprocess
    from run_frozen_cell import acquire_cache_workspace

    lease, wait = acquire_cache_workspace(tmp_path)
    probe = (
        "import fcntl,sys; f=open(sys.argv[1], 'a'); "
        "fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)"
    )
    command = [
        sys.executable,
        "-c",
        probe,
        str(tmp_path / "qcfuse_cache_workspace.lock"),
    ]
    assert wait >= 0
    assert subprocess.run(command, capture_output=True).returncode != 0
    lease.close()
    assert subprocess.run(command, capture_output=True).returncode == 0


def test_total_prefill_budget_preserves_unchunked_window():
    from run_frozen_cell import resolve_prefill_budget

    assert resolve_prefill_budget(71680) == 71680
    assert resolve_prefill_budget(71680, 80000) == 80000
    # A four-request cohort may have fewer tokens than the model window,
    # but current vLLM still requires the complete unchunked window budget.
    with pytest.raises(ValueError, match="Unchunked"):
        resolve_prefill_budget(71680, 58052)


def test_preload_capacity_n_a_preserves_original_source_without_fake_metrics():
    from types import SimpleNamespace
    from blend_ssd import GpuResidentPreloadCapacityError

    args = SimpleNamespace(kind="quality", storage="gpu", batch_size=1)
    manifest = {"samples": [{"source_id": "original", "input_tokens": 65409}]}
    row = gpu_preload_unavailable(
        args, [(0, {})], manifest, GpuResidentPreloadCapacityError("CUDA out of memory")
    )
    assert row["source_id"] == "original" and row["input_tokens"] == 65409
    assert row["sample_index"] == 0 and row["status"] == "N/A"
    assert row["reason"] == "gpu_resident_preload_capacity_failure"
    assert "metric" not in row and "ttft_seconds" not in row


def test_online_headroom_capacity_is_identified_separately():
    from types import SimpleNamespace
    from blend_ssd import GpuResidentPreloadCapacityError

    row = gpu_preload_unavailable(
        SimpleNamespace(kind="quality", storage="gpu", batch_size=1),
        [(0, {})],
        {"samples": [{"source_id": "original", "input_tokens": 65409}]},
        GpuResidentPreloadCapacityError("CUDA out of memory reserving online headroom"),
    )
    assert row["reason"] == "gpu_resident_online_headroom_failure"
    assert row["source_id"] == "original" and "ttft_seconds" not in row


@pytest.mark.parametrize(
    "kind,mode,batch",
    [("timing", "gpu", 1), ("quality", "cpu", 1), ("quality", "gpu", 2)],
)
def test_capacity_continuation_does_not_skip_timing_or_batch_failures(
    kind, mode, batch
):
    from types import SimpleNamespace
    from blend_ssd import GpuResidentPreloadCapacityError

    error = GpuResidentPreloadCapacityError("CUDA out of memory")
    args = SimpleNamespace(kind=kind, storage=mode, batch_size=batch)
    with pytest.raises(GpuResidentPreloadCapacityError) as raised:
        gpu_preload_unavailable(args, [(0, {})], {"samples": []}, error)
    assert raised.value is error


@pytest.mark.parametrize("second_fits", [True, False])
def test_quality_driver_cleans_failed_cache_then_visits_next_frozen_source(
    tmp_path, monkeypatch, second_fits
):
    import json
    from types import SimpleNamespace
    import run_frozen_cell as runner
    from blend_ssd import GpuResidentPreloadCapacityError

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "num_key_value_heads": 1,
                "head_dim": 1,
                "hidden_size": 1,
                "num_attention_heads": 1,
                "num_hidden_layers": 1,
            }
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "samples": [
            {
                "source_id": name,
                "input_tokens": 10,
                "max_new_tokens": 8,
                "prompt_token_ids_sha256": name,
            }
            for name in ("unsupported", "next")
        ]
    }
    manifest_path.write_text(json.dumps(manifest))
    rows = [{"source_id": x["source_id"]} for x in manifest["samples"]]
    monkeypatch.setattr(runner, "load_dataset", lambda *_: rows)
    monkeypatch.setattr(runner, "validate_dataset_rows", lambda *_: None)
    monkeypatch.setattr(runner, "validate_prompt_settings", lambda *_: None)
    monkeypatch.setattr(
        runner,
        "validate_frozen_manifest",
        lambda *a, **k: {
            "source_authenticity": "verified_official_dataset",
        },
    )
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda *_: "test")
    monkeypatch.setattr(runner, "_summarize_result", lambda *_: {"metric_mean": 1.0})
    visited, released, deleted_before_next = [], [], []

    class Engine:
        def __init__(self, *args, **kwargs):
            self.cache_dir = kwargs["cache_dir"]
            self.llm = SimpleNamespace(
                collective_rpc=lambda *a, **k: [{"resident": {"resident_files": 0}}]
            )

        def configure_baseline(self, *args):
            pass

        def warmup(self, **kwargs):
            pass

        def warmup_blend(self, *args, **kwargs):
            return {}

        def shutdown(self):
            pass

        def release_resident_batch(self):
            released.append(self.cache_dir)

        def _prepare_samples(self, batch, dataset):
            return [
                SimpleNamespace(
                    input_tokens=10, prompt_token_ids_hash=batch[0]["source_id"]
                )
            ]

        def phase1_offline(self, batch, dataset):
            path = Path(self.cache_dir)
            if visited:
                deleted_before_next.append(not Path(visited[-1][1]).exists())
            (path / "cache.bin").write_bytes(b"owned")
            visited.append((batch[0]["source_id"], self.cache_dir))

        def phase2_online(self, batch, dataset, ratio):
            if batch[0]["source_id"] == "unsupported" or not second_fits:
                raise GpuResidentPreloadCapacityError("CUDA out of memory")
            part = {
                k: [1]
                for k in (
                    "metric",
                    "ttft",
                    "first_token",
                    "decode",
                    "decode_tokens",
                    "generated_tokens",
                    "qcompute_latency",
                    "do_blend_ttft",
                    "latency",
                )
            }
            part.update(wall_seconds=1.0, records=[{"ruler_task": None}])
            return part

    monkeypatch.setattr(runner, "SSDPipelineEngine", Engine)
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frozen_cell",
            "--data",
            "unused",
            "--manifest",
            str(manifest_path),
            "--dataset",
            "longbench_v2",
            "--model",
            str(model),
            "--output",
            str(output),
            "--cache-parent",
            str(tmp_path / "caches"),
            "--baseline",
            "ours",
            "--storage",
            "gpu",
            "--kind",
            "quality",
            "--warmup",
            "0",
        ],
    )
    runner.main()
    result = json.loads(output.read_text())
    assert [x[0] for x in visited] == ["unsupported", "next"]
    assert deleted_before_next == [True] and len(released) == 2
    assert result["cache_cleanup_verified"] and not Path(result["cache_root"]).exists()
    assert result["requested_samples"] == 2
    if second_fits:
        assert result["status"] == "partial"
        assert result["records"][0]["sample_index"] == 1
        assert result["records"][0]["source_id"] == "next"
        assert len(result["unavailable_samples"]) == 1
    else:
        assert result["status"] == "N/A" and not result["records"]
        assert len(result["unavailable_samples"]) == 2
        assert "CUDA out of memory" in result["error"]
