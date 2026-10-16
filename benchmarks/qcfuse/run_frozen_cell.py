"""Measure a frozen cell with bounded cache lifetime and separate timing repeats."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import tempfile
import time
import traceback
from pathlib import Path

from blend_common import set_ours_layers
from blend_ssd import (
    GpuResidentPreloadCapacityError,
    SSDPipelineEngine,
    _summarize_result,
)
from evidence import (
    baseline_identity,
    validate_dataset_rows,
    validate_frozen_manifest,
    validate_prompt_settings,
    index_token_audit,
)
from utils import load_dataset
from cache_state import request_cold_cache
from worker_lifecycle import stop_owned_workers


def cleanup_owned_cache(root):
    root = Path(root)
    if not (root / ".qcfuse_cell_owner").is_file():
        raise RuntimeError(f"Refusing cleanup without ownership marker: {root}")
    shutil.rmtree(root)
    if root.exists():
        raise RuntimeError(f"Generated cache survived cleanup: {root}")


def acquire_cache_workspace(parent):
    """Keep bounded cache batches on this disk from competing across GPUs."""
    lease = (Path(parent) / "qcfuse_cache_workspace.lock").open("a")
    started = time.perf_counter()
    try:
        fcntl.flock(lease, fcntl.LOCK_EX)
    except BaseException:
        lease.close()
        raise
    return lease, time.perf_counter() - started


def measurement_rows(rows, kind, batch_size, timing_samples=None):
    """Keep a fixed cohort when comparing different concurrent batch sizes."""
    if timing_samples is not None:
        if kind != "timing" or not 1 <= timing_samples <= len(rows):
            raise ValueError("timing_samples requires timing and an available cohort")
        return rows[:timing_samples]
    return rows if kind == "quality" else rows[:batch_size]


def resolve_prefill_budget(model_window, override=None):
    budget = model_window if override is None else override
    if budget < model_window:
        raise ValueError("Unchunked prefill requires token budget >= max_model_len")
    return budget


def gpu_preload_unavailable(args, indexed_batch, manifest, error):
    """Continue only an independently tested single-source quality request."""
    if (
        args.kind != "quality"
        or args.storage != "gpu"
        or args.batch_size != 1
        or len(indexed_batch) != 1
        or not isinstance(error, GpuResidentPreloadCapacityError)
    ):
        raise error
    index, _ = indexed_batch[0]
    source = manifest["samples"][index]
    return {
        "sample_index": index,
        "source_id": source["source_id"],
        "status": "N/A",
        "reason": (
            "gpu_resident_online_headroom_failure"
            if "headroom" in str(error).lower()
            else "gpu_resident_preload_capacity_failure"
        ),
        "input_tokens": source["input_tokens"],
        "error": repr(error),
    }


def measurement_repeats(kind, timing_repeats=10, audit_repeats=1):
    if timing_repeats < 1 or audit_repeats < 1:
        raise ValueError("Measurement repeat counts must be positive")
    return (
        timing_repeats if kind == "timing" else audit_repeats if kind == "audit" else 1
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-parent", type=Path, required=True)
    parser.add_argument("--baseline", choices=("fullcomp", "ours"), required=True)
    parser.add_argument("--storage", choices=("ssd", "cpu", "gpu"), default="ssd")
    parser.add_argument("--ratio", type=float, default=0.4)
    parser.add_argument("--kind", choices=("quality", "timing", "audit"), required=True)
    parser.add_argument("--max-model-len", type=int, default=67584)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    parser.add_argument("--gpu-headroom-bytes-per-token", type=int, default=0)
    parser.add_argument("--gpu-headroom-fixed-bytes", type=int, default=0)
    parser.add_argument("--gpu-resident-codec", choices=("ans_f16_bits",), default=None)
    parser.add_argument("--gpu-query-alias", choices=("on", "off"), default="off")
    parser.add_argument(
        "--decode-cudagraph", choices=("on", "off", "piecewise"), default="off"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--critical-layers", type=int, default=3)
    parser.add_argument("--offline-kv-device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--thinking", choices=("on", "off"), default="off")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--ssd-cache-state", choices=("warm", "cold"), default="warm")
    parser.add_argument("--io-group-layers", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--audit-repeats", type=int, default=1)
    parser.add_argument(
        "--timing-samples",
        type=int,
        default=None,
        help="Fixed cohort size for timing; independent of concurrent batch size",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Instrumented diagnostic; adds per-layer logging overhead",
    )
    args = parser.parse_args()
    # This is a total scheduler budget, not a per-request budget.
    args.max_num_batched_tokens = resolve_prefill_budget(
        args.max_model_len, args.max_num_batched_tokens
    )
    os.environ["QCFUSE_IO_GROUP_LAYERS"] = str(args.io_group_layers)
    if args.output.exists():
        raise FileExistsError(f"Existing result is immutable: {args.output}")
    if (
        not 0 < args.ratio < 1
        or args.batch_size < 1
        or args.warmup < 0
        or args.repeats < 1
        or args.audit_repeats < 1
        or (args.max_new_tokens is not None and args.max_new_tokens < 1)
        or (args.kv_cache_memory_bytes is not None and args.kv_cache_memory_bytes < 1)
        or args.gpu_headroom_bytes_per_token < 0
        or args.gpu_headroom_fixed_bytes < 0
    ):
        parser.error("Invalid ratio, batch, warmup or repeats")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_parent.mkdir(parents=True, exist_ok=True)
    rows = load_dataset(str(args.data))
    validate_dataset_rows(rows, args.dataset)
    manifest = json.loads(args.manifest.read_text())
    validate_prompt_settings(manifest, args.thinking, args.max_new_tokens)
    # Verify all source rows before partitioning by capacity. Oversized rows
    # remain explicit N/A entries in the original random sample, never replaced.
    provenance = validate_frozen_manifest(
        args.manifest,
        rows,
        args.model,
        max(
            item["input_tokens"] + int(item.get("max_new_tokens") or 128)
            for item in manifest["samples"]
        ),
    )
    if provenance["source_authenticity"] not in (
        "pinned_nvidia_ruler_generator",
        "verified_official_dataset",
    ):
        raise ValueError("Frozen source provenance is unverified")
    rows = measurement_rows(rows, args.kind, args.batch_size, args.timing_samples)
    eligible, unavailable = [], []
    for index, row in enumerate(rows):
        sample = manifest["samples"][index]
        if (
            sample["input_tokens"]
            + int(sample.get("max_new_tokens") or row.get("max_new_tokens", 48))
            > args.max_model_len
        ):
            unavailable.append(
                {
                    "sample_index": index,
                    "source_id": sample["source_id"],
                    "status": "N/A",
                    "reason": "complete_input_exceeds_model_window",
                    "input_tokens": sample["input_tokens"],
                }
            )
        else:
            eligible.append((index, row))
    root = Path(tempfile.mkdtemp(prefix="qcfuse_cell_", dir=args.cache_parent))
    (root / ".qcfuse_cell_owner").write_text(str(args.output.resolve()))
    profile_path = (
        str(args.output) + ".profile.jsonl"
        if args.profile or args.kind == "audit"
        else None
    )
    audit_path = str(args.output) + ".tokens.jsonl" if args.kind == "audit" else None
    if profile_path:
        os.environ["QCFUSE_PROFILE_PATH"] = profile_path
    else:
        os.environ.pop("QCFUSE_PROFILE_PATH", None)
    os.environ["QCFUSE_TRANSFER_TIMING"] = "1" if args.kind == "audit" else "0"
    result = {
        **baseline_identity(args.baseline),
        **provenance,
        "config": vars(args),
        "status": "running",
        "measurement_kind": args.kind,
        "ratio": args.ratio if args.baseline == "ours" else 1.0,
        "storage_mode": args.storage,
        "requested_samples": len(rows),
        "resident_preload_excluded_from_ttft": True,
        "instrumented": bool(profile_path or audit_path),
        "token_observation_mode": (
            ("cuda_event_audit" if audit_path else "host_shape_counts")
            if args.baseline == "ours"
            else None
        ),
        "unavailable_samples": unavailable,
        "records": [],
        "parts": [],
        "profile_path": profile_path,
        "generation_step_profile_path": (
            str(args.output) + ".steps.jsonl" if args.profile else None
        ),
        "token_audit_path": audit_path,
        "cache_root": str(root),
        "cache_cleanup_verified": False,
        "cache_state_actions": [],
        "offline_cache_batches": [],
        "warmup_batches": [],
        "runtime": {
            name: importlib.metadata.version(name) for name in ("torch", "vllm")
        },
        "allocator_config": os.getenv(
            "PYTORCH_ALLOC_CONF", os.getenv("PYTORCH_CUDA_ALLOC_CONF")
        ),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "runtime_controls": {
            "measurements_per_source": measurement_repeats(
                args.kind, args.repeats, args.audit_repeats
            ),
            "execute_model_rpc_timeout_seconds": int(
                os.getenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "300")
            ),
            "worker_shutdown_timeout_seconds": int(
                os.getenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "5")
            ),
            "offline_build_workspace": os.getenv("QCFUSE_OFFLINE_BUILD_WORKSPACE"),
            "gpu_attention_backend": os.getenv(
                "QCFUSE_GPU_ATTENTION_BACKEND", "native"
            ),
            "flex_key_block": int(os.getenv("QCFUSE_FLEX_KEY_BLOCK", "64")),
            "gpu_ffn_projection_tile_tokens": int(
                os.getenv("QCFUSE_GPU_FFN_TILE_TOKENS", "0")
            ),
            "prefill_memory_policy": os.getenv("QCFUSE_PREFILL_MEMORY_POLICY", "auto"),
            "reuse_attention_metadata": os.getenv("QCFUSE_REUSE_ATTN_METADATA", "0")
            == "1",
        },
    }
    engine = None
    worker_processes = []
    cache_lease = None
    error = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if eligible:
            pre_init_wait = 0.0
            if args.profile and args.baseline == "ours":
                # A diagnostic's model loading must not compete with another
                # cell's measured host-to-device traffic.
                cache_lease, pre_init_wait = acquire_cache_workspace(args.cache_parent)
                result["pre_init_workspace_wait_seconds"] = pre_init_wait
            engine = SSDPipelineEngine(
                args.model,
                baseline=args.baseline,
                cache_dir=str(root),
                batch_size=args.batch_size,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_batched_tokens=args.max_num_batched_tokens,
                storage_mode=args.storage,
                rope_factor=4.0,
                block_size=512,
                kv_cache_memory_bytes=args.kv_cache_memory_bytes,
                gpu_headroom_bytes_per_token=args.gpu_headroom_bytes_per_token,
                gpu_headroom_fixed_bytes=args.gpu_headroom_fixed_bytes,
                gpu_resident_codec=args.gpu_resident_codec,
                gpu_query_alias=args.gpu_query_alias == "on",
                decode_cudagraph=args.decode_cudagraph != "off",
                decode_cudagraph_mode=(
                    "piecewise" if args.decode_cudagraph == "piecewise" else "full"
                ),
            )
            capture_workers = getattr(engine, "worker_process_identities", None)
            if callable(capture_workers):
                worker_processes = capture_workers()
            result["worker_processes"] = worker_processes
            engine.token_audit_path = audit_path
            engine.generation_step_profile_path = result["generation_step_profile_path"]
            engine.offline_kv_device = args.offline_kv_device
            engine.thinking_mode = args.thinking
            engine.max_new_tokens_override = args.max_new_tokens
            engine.worker_snapshots_enabled = True
            engine.token_counts_enabled = args.baseline == "ours"
            engine.forward_observer_enabled = args.kind == "audit"
            engine.configure_baseline(args.baseline, args.critical_layers)
            engine.warmup(num_warmup=0)
            for offset in range(0, len(eligible), args.batch_size):
                workspace_wait = 0.0
                if args.baseline == "ours":
                    if cache_lease is None:
                        cache_lease, workspace_wait = acquire_cache_workspace(
                            args.cache_parent
                        )
                    else:
                        workspace_wait = pre_init_wait
                indexed_batch = eligible[offset : offset + args.batch_size]
                # Sparse eligible subsets can have gaps. A separate batch root
                # makes cache paths unique even when local indices start at 0.
                batch_root = root / f"batch_{offset}"
                batch_root.mkdir()
                engine.cache_dir = str(batch_root)
                batch = [row for _, row in indexed_batch]
                expected = [manifest["samples"][index] for index, _ in indexed_batch]
                samples = engine._prepare_samples(batch, args.dataset)
                for sample, frozen in zip(samples, expected):
                    if (
                        sample.prompt_token_ids_hash
                        != frozen["prompt_token_ids_sha256"]
                    ):
                        raise ValueError(
                            "Runtime prompt IDs differ from the frozen manifest"
                        )
                if args.baseline == "ours":
                    model_config = json.loads(
                        (Path(args.model) / "config.json").read_text()
                    )
                    kv_heads = model_config["num_key_value_heads"]
                    head_dim = model_config.get(
                        "head_dim",
                        model_config["hidden_size"]
                        // model_config["num_attention_heads"],
                    )
                    estimated = (
                        sum(sample.input_tokens for sample in samples)
                        * kv_heads
                        * head_dim
                        * 4
                        * model_config["num_hidden_layers"]
                    )
                    if shutil.disk_usage(root).free < estimated * 1.3 + 2 * 1024**3:
                        raise MemoryError(
                            "Insufficient SSD space for this bounded batch cache"
                        )
                    offline_lease = None
                    offline_wait = 0.0
                    offline_lane = os.getenv("QCFUSE_OFFLINE_BUILD_WORKSPACE")
                    if offline_lane:
                        Path(offline_lane).mkdir(parents=True, exist_ok=True)
                        offline_lease, offline_wait = acquire_cache_workspace(
                            offline_lane
                        )
                    offline_started = time.perf_counter()
                    try:
                        engine.phase1_offline(batch, args.dataset)
                    finally:
                        if offline_lease is not None:
                            offline_lease.close()
                    result["offline_cache_batches"].append(
                        {
                            "batch_offset": offset,
                            "workspace_wait_seconds": workspace_wait,
                            "offline_build_wait_seconds": offline_wait,
                            "construction_seconds": time.perf_counter()
                            - offline_started,
                            "cache_bytes": sum(
                                p.stat().st_size for p in batch_root.rglob("*.bin")
                            ),
                            "excluded_from_online_ttft": True,
                        }
                    )
                try:
                    if offset == 0 or args.kind == "timing":
                        warmup = engine.warmup_blend(
                            batch,
                            args.dataset,
                            args.ratio if args.baseline == "ours" else 1.0,
                            num_warmup=args.warmup,
                        )
                        result["warmup_batches"].append(
                            {
                                "batch_offset": offset,
                                "excluded_from_online_ttft": True,
                                **warmup,
                            }
                        )
                    repeats = measurement_repeats(
                        args.kind, args.repeats, args.audit_repeats
                    )
                    for repeat in range(repeats):
                        if (
                            args.baseline == "ours"
                            and args.storage == "ssd"
                            and args.ssd_cache_state == "cold"
                        ):
                            result["cache_state_actions"].append(
                                request_cold_cache(
                                    root,
                                    [sample.sample_dir_chunk for sample in samples]
                                    + [sample.sample_dir_query for sample in samples],
                                )
                            )
                        part = engine.phase2_online(
                            batch,
                            args.dataset,
                            args.ratio if args.baseline == "ours" else 1.0,
                        )
                        for record, (index, row) in zip(part["records"], indexed_batch):
                            record.update(
                                sample_index=index,
                                repeat=repeat,
                                source_id=manifest["samples"][index]["source_id"],
                            )
                            result["records"].append(record)
                        result["parts"].append(part)
                        # Preserve independently completed observations, even
                        # if a later source cannot fit this GPU backend.
                        with Path(str(args.output) + ".progress.jsonl").open(
                            "a"
                        ) as stream:
                            for record in part["records"]:
                                stream.write(json.dumps(record) + "\n")
                except GpuResidentPreloadCapacityError as exc:
                    unavailable_source = gpu_preload_unavailable(
                        args, indexed_batch, manifest, exc
                    )
                    snapshots = engine.llm.collective_rpc(
                        "qcfuse_benchmark_snapshot", args=(True,), timeout=30
                    )
                    if not snapshots or any(
                        x["resident"]["resident_files"] for x in snapshots
                    ):
                        raise RuntimeError(
                            "GPU resident rollback is unverified"
                        ) from exc
                    result["unavailable_samples"].append(unavailable_source)
                    with Path(str(args.output) + ".unavailable.jsonl").open(
                        "a"
                    ) as stream:
                        stream.write(json.dumps(unavailable_source) + "\n")
                engine.release_resident_batch()
                shutil.rmtree(batch_root)
                if cache_lease is not None:
                    cache_lease.close()
                    cache_lease = None
                print(
                    json.dumps(
                        {
                            "completed": len(result["records"]),
                            "total": len(rows),
                            "batch_cache_deleted": not batch_root.exists(),
                        }
                    ),
                    flush=True,
                )
            result["status"] = (
                "N/A"
                if not result["records"]
                else "partial" if result["unavailable_samples"] else "complete"
            )
            if result["status"] == "N/A":
                capacity_errors = [
                    x["error"]
                    for x in result["unavailable_samples"]
                    if x["reason"]
                    in (
                        "gpu_resident_preload_capacity_failure",
                        "gpu_resident_online_headroom_failure",
                    )
                ]
                if capacity_errors:
                    # Preserve a real capacity error for older live queue
                    # coordinators, which already continue recorded GPU OOMs.
                    result["error"] = capacity_errors[0]
        else:
            result["status"] = "N/A"
        if result["parts"]:
            names = (
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
            merged = {
                name: [v for part in result["parts"] for v in part[name]]
                for name in names
            }
            merged["wall_seconds"] = sum(
                part["wall_seconds"] for part in result["parts"]
            )
            merged["records"] = result["records"]
            if all("decode_batch_wall_seconds" in part for part in result["parts"]):
                merged["decode_batch_wall_seconds"] = sum(
                    part["decode_batch_wall_seconds"] for part in result["parts"]
                )
            result["summary"] = _summarize_result(merged)
            result["summary"]["score_denominator"] = len(result["records"])
            result["summary"]["unique_source_samples"] = len(
                {record["source_id"] for record in result["records"]}
            )
            tasks = {
                record["ruler_task"]
                for record in result["records"]
                if record["ruler_task"]
            }
            if tasks:
                task_scores = {
                    task: [
                        record["metric"]
                        for record in result["records"]
                        if record["ruler_task"] == task
                    ]
                    for task in sorted(tasks)
                }
                result["task_scores"] = {
                    task: {"score": sum(values) / len(values), "count": len(values)}
                    for task, values in task_scores.items()
                }
                result["summary"]["macro_task_score"] = sum(
                    item["score"] for item in result["task_scores"].values()
                ) / len(tasks)
        if audit_path and args.baseline == "ours":
            with Path(audit_path).open() as stream:
                audit_rows = index_token_audit(map(json.loads, stream))
            for record in result["records"]:
                record["token_work"] = audit_rows[record["request_id"]]
                record["qcompute_token_work"] = audit_rows[
                    record["qcompute_request_id"]
                ]
    except BaseException as exc:
        error = exc
        result.update(
            status="failed", error=repr(exc), traceback=traceback.format_exc()
        )
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        except BaseException as shutdown_error:
            result.update(status="failed", shutdown_error=repr(shutdown_error))
            if error is None:
                error = shutdown_error
        finally:
            try:
                result["worker_shutdown"] = stop_owned_workers(worker_processes)
                cleanup_owned_cache(root)
                result["cache_cleanup_verified"] = True
            except BaseException as cleanup_error:
                result.update(
                    status="failed",
                    cache_cleanup_verified=False,
                    cache_cleanup_error=repr(cleanup_error),
                )
                if error is None:
                    error = cleanup_error
            finally:
                if cache_lease is not None:
                    cache_lease.close()
            # Path config values are rendered explicitly, not used to fabricate
            # any absent metrics.
            result["config"] = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result["config"].items()
            }
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if error:
        raise error


if __name__ == "__main__":
    main()
