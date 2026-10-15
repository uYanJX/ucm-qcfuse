"""Validate native vLLM recovery after a GPU-resident preload capacity failure."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import traceback

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from blend_ssd import SSDPipelineEngine
from run_cell_queue import gpu_idle
from run_frozen_cell import acquire_cache_workspace, cleanup_owned_cache
from utils import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--long-data", type=Path, required=True)
    parser.add_argument("--long-manifest", type=Path, required=True)
    parser.add_argument("--short-data", type=Path, required=True)
    parser.add_argument("--short-manifest", type=Path, required=True)
    parser.add_argument("--short-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve existing probe receipts"
    gpu_lock = (args.workspace / "tmp" / f"qcfuse_gpu_{args.gpu}.lock").open("a")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert gpu_idle(args.gpu), "Probe GPU must be idle"
    lease, waited = acquire_cache_workspace(args.workspace / "tmp")
    engine = None
    root = Path(tempfile.mkdtemp(prefix="qcfuse_cell_", dir=args.workspace / "tmp"))
    (root / ".qcfuse_cell_owner").write_text(str(args.output.resolve()))
    result = {
        "status": "running",
        "scope": "native_single_engine_preload_failure_then_short_request_recovery_not_matrix_timing",
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "pre_init_workspace_wait_seconds": waited,
        "cache_root": str(root),
        "cache_cleanup_verified": False,
        "runtime_protocol": {
            "max_model_len": 71680,
            "max_num_batched_tokens": 71680,
            "gpu_memory_utilization": 0.97,
            "ratio": 0.4,
            "critical_layers": 3,
            "block_size": 512,
            "rope_factor": 4.0,
            "storage": "gpu",
            "dtype": "bfloat16",
        },
        "steps": [],
    }
    error = None
    try:
        assert gpu_idle(args.gpu), "Probe GPU became occupied while waiting"
        if shutil.disk_usage(root).free < 24 * 1024**3:
            raise MemoryError("Insufficient SSD headroom for the single large cache")
        engine = SSDPipelineEngine(
            args.model,
            baseline="ours",
            cache_dir=str(root),
            batch_size=1,
            max_model_len=71680,
            max_num_batched_tokens=71680,
            gpu_memory_utilization=0.97,
            storage_mode="gpu",
            rope_factor=4.0,
            block_size=512,
        )
        engine.offline_kv_device = "cpu"
        engine.token_counts_enabled = True
        engine.worker_snapshots_enabled = True
        engine.configure_baseline("ours", 3)
        engine.warmup(num_warmup=0)
        large = load_dataset(str(args.long_data))[:1]
        lm = json.loads(args.long_manifest.read_text())
        engine.thinking_mode = "off"
        engine.max_new_tokens_override = 128
        engine.cache_dir = str(root / "long")
        sample = engine._prepare_samples(large, "ruler_64k")[0]
        assert (
            sample.prompt_token_ids_hash == lm["samples"][0]["prompt_token_ids_sha256"]
        )
        engine.phase1_offline(large, "ruler_64k")
        prepared = engine._prepare_samples(large, "ruler_64k")
        result["steps"].append(
            {"phase": "large_offline_complete", "input_tokens": sample.input_tokens}
        )
        try:
            engine._prepare_resident_batch(prepared)
        except Exception as exc:
            if "cuda out of memory" not in str(exc).lower():
                raise
            result["large_preload_error"] = repr(exc)
            result["large_preload_oom_observed"] = True
        else:
            raise RuntimeError("Expected large-input preload OOM was not reproduced")
        snapshots = engine.llm.collective_rpc(
            "qcfuse_benchmark_snapshot", args=(True,), timeout=30
        )
        assert all(x["resident"]["resident_files"] == 0 for x in snapshots)
        result["post_failure_worker_snapshots"] = snapshots
        engine.release_resident_batch()
        # The failed producer's owned files are removed before building the
        # next source, so the probe never needs two large caches on disk.
        shutil.rmtree(root / "long")
        result["long_cache_deleted"] = not (root / "long").exists()
        short = load_dataset(str(args.short_data))[:1]
        sm = json.loads(args.short_manifest.read_text())
        engine.thinking_mode = "on"
        engine.max_new_tokens_override = 4096
        engine.cache_dir = str(root / "short")
        sample = engine._prepare_samples(short, "longbench_v2")[0]
        assert (
            sample.prompt_token_ids_hash == sm["samples"][0]["prompt_token_ids_sha256"]
        )
        engine.phase1_offline(short, "longbench_v2")
        result["steps"].append(
            {"phase": "short_offline_complete", "input_tokens": sample.input_tokens}
        )
        measured = engine.phase2_online(short, "longbench_v2", 0.4)
        record = measured["records"][0]
        reference = json.loads(args.short_reference.read_text())
        expected = next(
            r
            for r in reference["records"]
            if r["source_id"] == sm["samples"][0]["source_id"] and r["repeat"] == 0
        )
        fields = ("text", "metric", "prompt_token_ids_hash", "sampling_max_tokens")
        assert all(
            record[k] == expected[k] for k in fields
        ), "Recovered request differs from frozen reference"
        work = record["token_work"]
        assert (
            len(work["layers"]) == 64
            and work["retained_tokens"] == record["input_tokens"]
        )
        assert len(record["qcompute_token_work"]["layers"]) == 51
        result["recovered_request"] = measured
        result["short_reference_exact_pairing"] = True
        result["model_config_sha256"] = hashlib.sha256(
            (Path(args.model) / "config.json").read_bytes()
        ).hexdigest()
        result["status"] = "complete"
    except BaseException as exc:
        error = exc
        result.update(
            status="failed", error=repr(exc), traceback=traceback.format_exc()
        )
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        except BaseException as exc:
            result.update(status="failed", shutdown_error=repr(exc))
            error = error or exc
        finally:
            cleanup_owned_cache(root)
            result["cache_cleanup_verified"] = True
            lease.close()
            gpu_lock.close()
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if error:
        raise error


if __name__ == "__main__":
    main()
