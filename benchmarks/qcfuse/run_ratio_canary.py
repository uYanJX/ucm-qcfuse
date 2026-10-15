"""Observe five ratios on one frozen prompt and one immutable cache.

This is a diagnostic sweep, not a quality or stable timing cell. Model startup
and KVCOMPUTE run once so a short idle-GPU window can cover the full sweep.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from blend_ssd import SSDPipelineEngine
from evidence import (
    validate_frozen_manifest,
    validate_prompt_settings,
    index_token_audit,
)
from run_frozen_cell import cleanup_owned_cache
from utils import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-parent", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = load_dataset(str(args.data))
    if len(rows) != 1:
        raise ValueError("The ratio canary requires exactly one frozen sample")
    manifest = json.loads(args.manifest.read_text())
    validate_prompt_settings(manifest, "off", None)
    provenance = validate_frozen_manifest(args.manifest, rows, args.model, 67584)
    args.cache_parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="qcfuse_ratio_canary_", dir=args.cache_parent))
    (root / ".qcfuse_cell_owner").write_text(str(args.output.resolve()))
    audit_path = str(args.output) + ".tokens.jsonl"
    os.environ["QCFUSE_PROFILE_PATH"] = str(args.output) + ".profile.jsonl"
    os.environ["QCFUSE_TRANSFER_TIMING"] = "1"
    os.environ["QCFUSE_IO_GROUP_LAYERS"] = "1"
    payload = {
        **provenance,
        "status": "running",
        "measurement_kind": "diagnostic_ratio_sweep",
        "allocator_config": os.getenv("PYTORCH_ALLOC_CONF"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "cache_root": str(root),
        "cache_cleanup_verified": False,
        "cache_policy": "one_read_only_cache_shared_only_within_diagnostic_sweep",
        "io_group_layers": 1,
        "token_audit_path": audit_path,
        "full": None,
        "ratios": {},
    }
    engine = None
    error = None

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    try:
        reference = (
            Path(__file__).resolve().parents[3]
            / "tmp/official_sources/QCFuse-1416459ada5bec0695ec375ada435e407cdea699"
        )
        if not (reference / "srt/utils/triton_attention_score.py").is_file():
            raise FileNotFoundError("Pinned QCFuse differential reference is missing")
        cuda_log = Path(str(args.output) + ".cuda_checks.log")
        payload["cuda_regression_log"] = str(cuda_log)
        with cuda_log.open("a") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    str(
                        Path(__file__).resolve().parents[2]
                        / "tests/qcfuse/test_cuda_kernels.py"
                    ),
                    "--disable-warnings",
                    "--maxfail=1",
                ],
                env={**os.environ, "QCFUSE_REFERENCE_DIR": str(reference)},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        engine = SSDPipelineEngine(
            args.model,
            baseline="fullcomp",
            cache_dir=str(root),
            batch_size=1,
            max_model_len=67584,
            gpu_memory_utilization=0.97,
            storage_mode="ssd",
            block_size=512,
            rope_factor=4.0,
        )
        engine.worker_snapshots_enabled = True
        engine.thinking_mode = "off"
        engine.offline_kv_device = "cpu"
        sample = engine._prepare_samples(rows, "ruler_64k")[0]
        if (
            sample.prompt_token_ids_hash
            != manifest["samples"][0]["prompt_token_ids_sha256"]
        ):
            raise ValueError("Runtime prompt does not match manifest")
        payload["full"] = engine.phase2_online(rows, "ruler_64k", 1.0)
        print(
            json.dumps(
                {"event": "full_completed", "summary": payload["full"]["summary"]}
            ),
            flush=True,
        )
        engine.configure_baseline("ours", 3)
        engine.token_audit_path = audit_path
        engine.phase1_offline(rows, "ruler_64k")
        print(json.dumps({"event": "kvcompute_completed"}), flush=True)
        for ratio in (0.1, 0.2, 0.3, 0.4, 0.5):
            result = engine.phase2_online(rows, "ruler_64k", ratio)
            payload["ratios"][str(ratio)] = result
            args.output.write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n"
            )
            with Path(audit_path).open() as stream:
                audit = index_token_audit(map(json.loads, stream))
            for record in result["records"]:
                record["token_work"] = audit[record["request_id"]]
                record["qcompute_token_work"] = audit[record["qcompute_request_id"]]
            payload["ratios"][str(ratio)] = result
            args.output.write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    {
                        "event": "ratio_completed",
                        "ratio": ratio,
                        "summary": result["summary"],
                    }
                ),
                flush=True,
            )
        payload["status"] = "complete"
    except BaseException as exc:
        error = exc
        payload.update(
            status="failed", error=repr(exc), traceback=traceback.format_exc()
        )
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        finally:
            cleanup_owned_cache(root)
            payload["cache_cleanup_verified"] = True
            args.output.write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n"
            )
    if error:
        raise error


if __name__ == "__main__":
    main()
