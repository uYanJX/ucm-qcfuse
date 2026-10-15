"""Summarize frozen receipts without filling absent measurements with estimates."""

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path


def transfer_deltas(receipt):
    totals, physical_bytes, observations = {}, 0, 0
    for part in receipt.get("parts", []):
        for snapshot in part.get("worker_snapshots", []):
            before, after = snapshot["before"], snapshot["after"]
            if len(before) != len(after):
                raise ValueError("Worker snapshot counts differ")
            for start, end in zip(before, after):
                if start["pid"] != end["pid"]:
                    raise ValueError("Worker identity changed during measurement")
                observations += 1
                physical_bytes += (
                    end["process_io"]["read_bytes"] - start["process_io"]["read_bytes"]
                )
                for key, value in end.get("cache_transfer_totals", {}).items():
                    if key != "ring_depth" and key in start.get(
                        "cache_transfer_totals", {}
                    ):
                        totals[key] = (
                            totals.get(key, 0)
                            + value
                            - start["cache_transfer_totals"][key]
                        )
    # Disabled CUDA timing yields a zero counter, not measured zero-copy work.
    if receipt.get("measurement_kind") != "audit":
        totals["cuda_copy_seconds"] = None
    return {
        "cache_counters": totals,
        "worker_block_read_bytes": physical_bytes if observations else None,
    }


def aggregate_decode_rate(receipt):
    summary = receipt["summary"]
    reported = summary.get("aggregate_decode_tokens_per_second")
    if reported is not None:
        return reported, summary.get(
            "aggregate_decode_scope", "receipt_reported_batch_decode_windows"
        )
    # In a serial batch1 cell every request's decode interval is its batch
    # interval. Multi-request intervals can overlap and cannot be summed.
    if receipt["config"]["batch_size"] != 1:
        return None, None
    tokens, seconds = 0, 0.0
    for record in receipt["records"]:
        count, duration = record.get("decode_tokens"), record.get("decode_seconds")
        if count is None or duration is None:
            return None, None
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            or not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(duration)
            or duration < 0
            or (duration == 0 and count > 0)
        ):
            raise ValueError("Invalid measured batch1 decode count or duration")
        tokens += count
        seconds += duration
    if seconds == 0:
        return None, None
    return tokens / seconds, "recomputed_from_serial_batch1_record_decode_windows"


def summarize_receipt(receipt):
    baseline, kind = receipt.get("baseline"), receipt.get("measurement_kind")
    if baseline not in ("fullcomp", "ours") or not receipt.get("comparison_eligible"):
        raise ValueError("Unverified methods cannot enter the measured matrix")
    if not receipt.get("cache_cleanup_verified"):
        raise ValueError("Terminal cache cleanup is unverified")
    status = receipt.get("status")
    config = receipt["config"]
    row = {
        "dataset": config["dataset"],
        "method": baseline,
        "source": receipt["storage_mode"] if baseline == "ours" else "none",
        "ratio": receipt["ratio"],
        "kind": kind,
        "batch": config["batch_size"],
        "status": status,
        "requested_sources": receipt["requested_samples"],
        "unavailable_sources": len(receipt.get("unavailable_samples", [])),
        "model_window": config["max_model_len"],
        "cache_cleanup_verified": True,
    }
    if status not in ("complete", "partial"):
        row["error"] = receipt.get("error")
        row["measurements"] = None
        return row
    records, summary = receipt["records"], receipt["summary"]
    if not records or any(x.get("truncated") for x in records):
        raise ValueError("Empty or truncated measured cell")
    unique = len({x["source_id"] for x in records})
    if kind == "quality":
        if (
            len(records) != unique
            or unique + row["unavailable_sources"] != row["requested_sources"]
        ):
            raise ValueError("Quality receipt does not account for the frozen cohort")
    elif kind == "timing":
        if config["warmup"] < 5 or config["repeats"] < 10:
            raise ValueError("Timing receipt misses the required warmups/repeats")
        counts = {}
        for record in records:
            counts.setdefault(record["source_id"], set()).add(record["repeat"])
        if any(values != set(range(config["repeats"])) for values in counts.values()):
            raise ValueError("Timing source has missing measured repeats")
        if len(records) != unique * config["repeats"]:
            raise ValueError("Timing receipt contains duplicate observations")
    score = summary.get("macro_task_score", summary["metric_mean"])
    aggregate_rate, aggregate_scope = aggregate_decode_rate(receipt)
    row.update(
        unique_sources=unique,
        measured_observations=len(records),
        score=score if kind == "quality" else None,
        score_scope=(
            "eligible_unique_sources_only"
            if kind == "quality"
            else "not_a_quality_cell"
        ),
        ttft_seconds=summary["ttft_mean_seconds"],
        thinking_ttft_seconds=summary.get("thinking_ttft_mean_seconds"),
        ttft_p95_seconds=summary["ttft_p95_seconds"],
        decode_seconds=summary["decode_mean_seconds"],
        per_sequence_decode_tok_s=summary["decode_tokens_per_second"],
        aggregate_decode_tok_s=aggregate_rate,
        aggregate_decode_scope=aggregate_scope,
        output_tok_s=summary["aggregate_output_tokens_per_second"],
        qcompute_seconds=summary["qcompute_mean_seconds"],
        unfinished_thinking=summary.get("unfinished_thinking_samples"),
    )
    counts = [x["token_work"] for x in records if "token_work" in x]
    row["token_count_observations"] = len(counts)
    row["actual_token_ranges"] = {
        name: (
            [min(x[name] for x in counts), max(x[name] for x in counts)]
            if counts and all(name in x for x in counts)
            else None
        )
        for name in (
            "selected_tokens",
            "recomputed_tokens",
            "retained_tokens",
            "qkv_token_layers",
            "ffn_token_layers",
        )
    }
    row["transfer"] = transfer_deltas(receipt)
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite measured value: {key}")
    return row


def verify_manifest(receipt):
    path = Path(receipt["manifest_path"])
    manifest = json.loads(path.read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["manifest_sha256"]:
        raise ValueError("Receipt manifest hash differs from frozen source")
    if manifest["model_config_sha256"] != receipt["model_config_sha256"]:
        raise ValueError("Receipt checkpoint configuration differs")
    for record in receipt["records"]:
        source = manifest["samples"][record["sample_index"]]
        if (
            record["source_id"],
            record["input_tokens"],
            record["prompt_token_ids_hash"],
        ) != (
            source["source_id"],
            source["input_tokens"],
            source["prompt_token_ids_sha256"],
        ):
            raise ValueError("Measured prompt differs from frozen source")
        if record["sampling_max_tokens"] != source["max_new_tokens"]:
            raise ValueError("Measured output budget differs from frozen source")
    unavailable_indices = set()
    measured_indices = {x["sample_index"] for x in receipt["records"]}
    for record in receipt.get("unavailable_samples", []):
        index = record["sample_index"]
        if not isinstance(index, int) or not 0 <= index < len(manifest["samples"]):
            raise ValueError("Unavailable source index differs from frozen cohort")
        source = manifest["samples"][index]
        if (record["source_id"], record["input_tokens"]) != (
            source["source_id"],
            source["input_tokens"],
        ):
            raise ValueError("Unavailable source differs from frozen cohort")
        if index in unavailable_indices or index in measured_indices:
            raise ValueError("Unavailable source is duplicated or also measured")
        unavailable_indices.add(index)
    if receipt.get("measurement_kind") == "quality" and receipt.get("status") in (
        "complete",
        "partial",
        "N/A",
    ):
        if receipt["requested_samples"] != len(
            manifest["samples"]
        ) or measured_indices | unavailable_indices != set(
            range(len(manifest["samples"]))
        ):
            raise ValueError(
                "Quality receipt does not cover the original frozen cohort"
            )


def load_cell(path, arguments=None):
    path = Path(path)
    if not path.exists():
        progress = Path(str(path) + ".progress.jsonl")
        count = len(progress.read_text().splitlines()) if progress.exists() else 0
        unavailable = Path(str(path) + ".unavailable.jsonl")
        unavailable_count = (
            len(unavailable.read_text().splitlines()) if unavailable.exists() else 0
        )
        return {
            "receipt": str(path),
            "status": (
                "partial_observations"
                if count
                else "partial_unavailable" if unavailable_count else "pending"
            ),
            "measured_observations": count,
            "unavailable_observations": unavailable_count,
            "config": arguments,
        }
    try:
        receipt = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {
            "receipt": str(path),
            "status": "receipt_write_in_progress",
            "config": arguments,
        }
    if arguments and any(
        str(receipt["config"].get(k)) != str(v) for k, v in arguments.items()
    ):
        raise ValueError(f"Receipt differs from queued configuration: {path}")
    if receipt.get("status") in ("complete", "partial", "N/A"):
        verify_manifest(receipt)
    row = summarize_receipt(receipt)
    row["receipt"] = str(path)
    row["receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return row


def render(cells, unavailable_methods=()):
    lines = [
        "# Current frozen matrix",
        "",
        "Only terminal, verified receipts contribute metrics. Partial observations and queued cells remain separate. Quality scores cover eligible unique sources; timing repeats are not independent quality samples. The 71680-window supplement does not validate the original 131072-window protocol.",
        "",
    ]
    for kind in ("quality", "timing"):
        lines.extend(
            [
                f"## {kind.title()}",
                "",
                "| Dataset | Method / source | Ratio | Batch | Unique / requested | Score | First-output TTFT / p95 (s) | Decode mean (s) | Per-sequence / aggregate decode tok/s | Status |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in cells:
            if row.get("kind") != kind:
                continue

            def number(key):
                value = row.get(key)
                return f"{value:.4f}" if value is not None else "N/A"

            score = (
                f"{row['score'] * 100:.2f}%" if row.get("score") is not None else "—"
            )
            lines.append(
                f"|{row['dataset']}|{row['method']} / {row['source']}|{row['ratio']}|{row['batch']}|{row.get('unique_sources', 0)}/{row['requested_sources']}|{score}|{number('ttft_seconds')} / {number('ttft_p95_seconds')}|{number('decode_seconds')}|{number('per_sequence_decode_tok_s')} / {number('aggregate_decode_tok_s')}|{row['status']}|"
            )
        lines.append("")
    pending = [x for x in cells if "kind" not in x]
    if pending:
        lines.extend(
            [
                "## Awaiting terminal receipts",
                "",
                "| Cell | State | Partial observations | Recorded unavailable sources |",
                "|---|---|---:|---:|",
            ]
        )
        lines.extend(
            f"|{Path(x['receipt']).name}|{x['status']}|{x.get('measured_observations', 0)}|{x.get('unavailable_observations', 0)}|"
            for x in pending
        )
    if unavailable_methods:
        lines.extend(
            [
                "",
                "## Unvalidated comparison methods",
                "",
                "| Method | Status | Reason |",
                "|---|---|---|",
            ]
        )
        lines.extend(f"|{x['method']}|N/A|{x['reason']}|" for x in unavailable_methods)
    lines.extend(
        [
            "",
            "Transfer counters, actual token ranges, source receipt hashes and unavailable-row counts are preserved in the accompanying JSON. For serial batch1 receipts without a reported aggregate decode rate, it is recomputed from recorded decode tokens and request decode windows; missing counts or multi-request windows remain N/A. Absent token counts and disabled CUDA copy timing remain N/A. Partial-observation status does not by itself establish that a process is live; process liveness is verified separately.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--extra-receipt", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--baseline-catalog", type=Path)
    parser.add_argument("--watch-queue-pid", type=int)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    expected = {
        str(Path(x["arguments"]["output"]).resolve()): x["arguments"]
        for x in plan["cases"]
    }
    for path in args.extra_receipt:
        expected.setdefault(str(path.resolve()), None)
    unavailable_methods = []
    if args.baseline_catalog:
        unavailable_methods = [
            method
            for method in json.loads(args.baseline_catalog.read_text())["methods"]
            if method["status"] == "N/A"
        ]
    previous = None
    while True:
        # Refresh only after a receipt/progress file changes; never poll a GPU.
        fingerprint = tuple(
            (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for name in expected
            for path in (
                Path(name),
                Path(name + ".progress.jsonl"),
                Path(name + ".unavailable.jsonl"),
            )
            if path.exists()
        )
        if fingerprint != previous:
            cells = [load_cell(path, arguments) for path, arguments in expected.items()]
            generated_at = datetime.now().astimezone().isoformat()
            output = {
                "protocol": plan["protocol"],
                "scope": "current_receipts_not_completed_matrix",
                "generated_at": generated_at,
                "cells": cells,
                "unavailable_methods": unavailable_methods,
            }
            for path, content in (
                (
                    args.results_json,
                    json.dumps(output, indent=2, allow_nan=False) + "\n",
                ),
                (
                    args.output,
                    render(cells, unavailable_methods)
                    + f"\nGenerated: {generated_at}\n",
                ),
            ):
                temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
                temporary.write_text(content)
                temporary.replace(path)
            previous = fingerprint
        if args.watch_queue_pid is None:
            return
        command = Path(f"/proc/{args.watch_queue_pid}/cmdline")
        if not command.exists():
            return
        argv = command.read_bytes().split(b"\0")
        if not any(value.endswith(b"run_cell_queue.py") for value in argv):
            return
        if os.fsencode(str(args.plan)) not in argv:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
