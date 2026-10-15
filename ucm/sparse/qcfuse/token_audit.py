"""Opt-in token-work observations; never used to choose or prune tokens."""

import json
import os
import time


def new_audit(path, phase, ratio, start, locs, *, counts_only=False):
    if not path and not counts_only:
        return None
    return {
        "path": os.fspath(path) if path else None,
        "phase": phase,
        "requested_ratio": ratio,
        "start_layer": start,
        "prompt_tokens": locs[-1] - locs[0],
        "prefix_tokens": locs[1] - locs[0] if len(locs) > 2 else 0,
        "context_tokens": locs[-2] - locs[1] if len(locs) > 3 else 0,
        "query_tokens": locs[-1] - locs[-2],
        "layers": [],
    }


def observe_layer(state, layer, qkv_tokens, attention_q_tokens, attention_kv_tokens):
    audit = state.token_audit
    if audit is None:
        return
    audit["layers"].append(
        {
            "layer": int(layer),
            "qkv_tokens": int(qkv_tokens),
            "attention_query_tokens": int(attention_q_tokens),
            "attention_kv_tokens": int(attention_kv_tokens),
        }
    )


def observe_prefill_batch(step):
    """Record scheduler co-residency rather than the requested batch limit."""
    if not step.active or not any(
        entry.state.token_audit is not None for entry in step.entries
    ):
        return
    batch = {
        "request_ids": [entry.request_id for entry in step.entries],
        "client_request_ids": [entry.state.client_request_id for entry in step.entries],
        "request_count": len(step.entries),
        "scheduled_tokens": [entry.scheduled_tokens for entry in step.entries],
    }
    for entry in step.entries:
        if entry.state.token_audit is not None:
            entry.state.token_audit["scheduled_prefill_batch"] = batch


def observe_ffn(state, tokens):
    if state.token_audit is not None and state.token_audit["layers"]:
        state.token_audit["layers"][-1]["ffn_tokens"] = int(tokens)


def begin_gpu_span(state, name, layer, tensor):
    if (
        state.token_audit is None
        or not state.token_audit.get("path")
        or not tensor.is_cuda
    ):
        return None
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(torch.cuda.current_stream(tensor.device))
    return (name, int(layer), start, end, time.perf_counter(), tensor.device)


def end_gpu_span(state, span):
    if span is not None:
        import torch

        name, layer, start, end, host_start, device = span
        end.record(torch.cuda.current_stream(device))
        state.token_audit.setdefault("cuda_spans", []).append(
            (name, layer, start, end, time.perf_counter() - host_start)
        )


def finish_audit(state):
    audit = state.token_audit
    if audit is None or audit.get("emitted"):
        return
    row = {k: v for k, v in audit.items() if k not in ("path", "emitted", "cuda_spans")}
    measured_spans = []
    for name, layer, start, end, host_seconds in audit.get("cuda_spans", []):
        # Diagnostic-only synchronization, after all layer intervals have been
        # recorded. Timing runs never enable this observer.
        end.synchronize()
        measured_spans.append(
            {
                "name": name,
                "layer": layer,
                "cuda_stream_seconds": start.elapsed_time(end) / 1000.0,
                "host_seconds": host_seconds,
            }
        )
    row["measured_spans"] = measured_spans
    row["span_scope"] = (
        "stream_intervals_include_dependencies_and_idle_gaps_not_exclusive_kernel_time"
    )
    row.update(
        event="token_work",
        request_id=state.request_id,
        client_request_id=state.client_request_id,
        query_session_id=state.query_session_id,
        ratio_semantics="document_token_recompute_budget_after_selection",
    )
    if state.blend_top_indices is not None:
        selected = int(state.blend_top_indices.numel())
        # Selection always includes the complete query.  The ratio=1 path
        # additionally includes the prefix; partial selection excludes it.
        prefix_selected = row["prefix_tokens"] if state.ratio >= 1 else 0
        context_selected = selected - row["query_tokens"] - prefix_selected
        row.update(
            selected_tokens=selected,
            recomputed_tokens=selected,
            recomputed_tokens_scope="rows_after_selection_per_layer_not_total_flops",
            selected_context_tokens=context_selected,
            retained_tokens=row["prompt_tokens"],
            retained_tokens_scope="full_attention_kv_tokens",
            reused_tokens=row["prompt_tokens"] - selected,
            actual_context_recompute_ratio=(
                context_selected / row["context_tokens"]
                if row["context_tokens"]
                else None
            ),
        )
    row["qkv_token_layers"] = sum(x["qkv_tokens"] for x in row["layers"])
    row["ffn_token_layers"] = sum(x.get("ffn_tokens", 0) for x in row["layers"])
    path = audit["path"]
    row["observation_mode"] = "cuda_event_audit" if path else "host_shape_counts"
    if path:
        # Explicit diagnostics only: hash actual order and positions. Formal
        # counts-only timing must never synchronize or copy these tensors.
        import hashlib
        import torch

        for name, value in (
            ("selected_indices", state.blend_top_indices),
            ("positions", state.positions),
            ("attention_positions", state.attention_positions),
        ):
            if value is not None:
                data = value.detach().to(device="cpu", dtype=torch.int64)
                payload = data.contiguous().numpy().astype("<i8", copy=False).tobytes()
                row[name + "_int64_le_sha256"] = hashlib.sha256(payload).hexdigest()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    audit["emitted"] = True
    return row
