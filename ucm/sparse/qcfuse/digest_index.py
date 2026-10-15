# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""Digest-index math for QCFuse (kvzip compressed-view scoring + ranking).

Faithful port of SGLang's ``digest_index_manager.py``, operating on a
:class:`~ucm.sparse.qcfuse.blend_info.QcFuseRequestState` instead of the global
``DigestIndexManager`` / ``ContextBlendPool`` class singletons.

The one mathematical core is :func:`accumulate_kvzip_layer_score`: for each
document chunk, the offline augmented prompt carries a trailing "zip" segment
(``zip = "Repeat the previous context exactly."``).  The zip query attends over
``[sink | doc | zip]``; the softmax mass landing on the *doc* positions is the
token's importance, reduced over query positions and heads with ``amax`` and
accumulated across layers with ``torch.maximum``.
"""

from __future__ import annotations

import copy
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ucm.sparse.qcfuse.blend_info import (
    DEFAULT_DIGEST_RATIO,
    BlendStyle,
    QcFuseRequestState,
)

DIGEST_INDEX_VERSION = 8
DIGEST_INDEX_METHOD = "kvzip"


def normalize_method(method: Optional[str]) -> str:
    method = (method or DIGEST_INDEX_METHOD).lower()
    if method != DIGEST_INDEX_METHOD:
        raise ValueError(
            f"Unsupported digest_index_method={method!r}; expected {DIGEST_INDEX_METHOD!r}"
        )
    return method


def index_filename(method: str) -> str:
    return f"index_{normalize_method(method)}.json"


def _cumsum_lens(lengths: Sequence[int]) -> List[int]:
    out = [0]
    total = 0
    for length in lengths:
        total += int(length)
        out.append(total)
    return out


def ensure_forward_positions(state: QcFuseRequestState) -> torch.Tensor:
    """Return (and cache) the forward positions of the offline prefill.

    KVCOMPUTE treats each augmented chunk as an independent sequence. Positions
    therefore restart at zero for every chunk, matching SGLang's
    ``compute_position`` call with one zero-length prefix per chunk.
    """
    positions = state.positions
    if positions is not None:
        return positions
    if state.chunk_loc_list is None:
        raise ValueError("chunk_loc_list is required to compute forward positions")
    chunk_lens = [int(length) for length in state.chunk_lens.tolist()]
    positions = torch.cat(
        [
            torch.arange(length, dtype=torch.long, device=state.chunk_loc_list.device)
            for length in chunk_lens
        ]
    )
    state.positions = positions
    return positions


def prepare_augmented_locs_for_request(raw_locs: Sequence[int]) -> Optional[Dict]:
    """Detect the ``[sys][doc][zip][doc][zip]...[query]`` layout of an augmented
    offline prompt and build the forward/original loc maps.

    Returns ``None`` when the layout is not an augmented prompt.
    """
    raw_locs = [int(x) for x in raw_locs]
    num_raw_chunks = len(raw_locs) - 1
    if num_raw_chunks < 4 or num_raw_chunks % 2 != 0:
        return None

    doc_chunk_indices = list(range(1, num_raw_chunks - 1, 2))
    zip_chunk_indices = list(range(2, num_raw_chunks - 1, 2))
    if len(doc_chunk_indices) != len(zip_chunk_indices):
        return None

    sys_start, sys_end = raw_locs[0], raw_locs[1]
    query_start = raw_locs[-2]
    query_end = raw_locs[-1]

    forward_lens = [sys_end - sys_start]
    original_lens = [sys_end - sys_start]
    keep_indices = list(range(sys_start, sys_end))
    aug_doc_ranges: List[Tuple[int, int]] = []
    aug_zip_ranges: List[Tuple[int, int]] = []

    for doc_idx, zip_idx in zip(doc_chunk_indices, zip_chunk_indices):
        doc_start = raw_locs[doc_idx]
        doc_end = raw_locs[doc_idx + 1]
        zip_start = raw_locs[zip_idx]
        zip_end = raw_locs[zip_idx + 1]

        forward_lens.append(zip_end - doc_start)
        original_lens.append(doc_end - doc_start)
        keep_indices.extend(range(doc_start, doc_end))
        aug_doc_ranges.append((doc_start, doc_end))
        aug_zip_ranges.append((zip_start, zip_end))

    forward_lens.append(query_end - query_start)
    original_lens.append(query_end - query_start)
    keep_indices.extend(range(query_start, query_end))

    return {
        "forward_locs": _cumsum_lens(forward_lens),
        "original_locs": _cumsum_lens(original_lens),
        "keep_indices": keep_indices,
        "aug_sys_range": (sys_start, sys_end),
        "aug_doc_ranges": aug_doc_ranges,
        "aug_zip_ranges": aug_zip_ranges,
    }


def _rank_from_scores(scores: torch.Tensor, chunk_len: int, n_sink: int) -> List[int]:
    if chunk_len <= 0:
        return []
    scores = scores.detach()
    sink_count = min(max(int(n_sink), 0), int(chunk_len))
    sink_indices = torch.arange(sink_count, device=scores.device)
    if sink_count < chunk_len:
        tail_scores = scores[sink_count:]
        tail_order = torch.argsort(tail_scores, descending=True) + sink_count
        ranked = torch.cat([sink_indices, tail_order])
    else:
        ranked = sink_indices
    return [int(x) for x in ranked.cpu().tolist()]


def _rank_scores_by_layer(
    scores_by_layer_chunk: Sequence[Sequence[torch.Tensor]],
    doc_chunk_lengths: Sequence[int],
    n_sink: int,
) -> List[List[List[int]]]:
    ranked_by_layer = []
    for layer_scores in scores_by_layer_chunk:
        ranked = [[]]
        for idx, chunk_len in enumerate(doc_chunk_lengths):
            chunk_len = int(chunk_len)
            if chunk_len <= 0:
                ranked.append([])
                continue
            if idx < len(layer_scores) and layer_scores[idx].numel() == chunk_len:
                scores = layer_scores[idx]
            else:
                scores = torch.zeros(chunk_len)
            ranked.append(_rank_from_scores(scores, chunk_len, n_sink))
        ranked_by_layer.append(ranked)
    return ranked_by_layer


def _scores_from_kvzip_dict(
    doc_chunk_lengths: Sequence[int],
    scores_by_layer_chunk: Dict[int, List[Optional[torch.Tensor]]],
    num_layers: int,
) -> List[List[torch.Tensor]]:
    out = []
    for layer_id in range(max(0, int(num_layers))):
        raw_layer_scores = scores_by_layer_chunk.get(layer_id, [])
        layer_scores = []
        for idx, chunk_len in enumerate(doc_chunk_lengths):
            chunk_len = int(chunk_len)
            if (
                chunk_len > 0
                and idx < len(raw_layer_scores)
                and raw_layer_scores[idx] is not None
                and raw_layer_scores[idx].numel() == chunk_len
            ):
                layer_scores.append(raw_layer_scores[idx].detach().float())
            else:
                layer_scores.append(torch.zeros(max(0, chunk_len)))
        out.append(layer_scores)
    return out


def build_all_indices(state: QcFuseRequestState) -> None:
    """Build per-layer kvzip rankings and materialized context positions."""
    full_num_layers = (
        int(state.att_params.num_layers)
        if state.att_params is not None
        else len(state.k_buffer)
    )
    qcompute_end = int(
        state.qcompute_end if state.qcompute_end is not None else full_num_layers
    )
    num_layers = max(0, min(qcompute_end, full_num_layers))
    digest_ratio = float(
        DEFAULT_DIGEST_RATIO if state.digest_ratio is None else state.digest_ratio
    )
    digest_method = normalize_method(state.digest_index_method)
    critical_layers = [int(x) for x in (state.critical_layers or [])]
    original_locs = (
        state.digest_original_chunk_loc_list
        if state.digest_original_chunk_loc_list is not None
        else state.chunk_loc_list
    )

    original_ranges = (
        [
            (int(original_locs[i]), int(original_locs[i + 1]))
            for i in range(len(original_locs) - 1)
        ]
        if original_locs is not None
        else []
    )
    num_chunks = len(original_ranges)

    empty_payload = {
        "digest_index_version": DIGEST_INDEX_VERSION,
        "orig_chunk_ranges": [],
        "total_tokens": 0,
        "available_methods": [digest_method],
        "layer_wise": True,
        "num_layers": num_layers,
        "digest_ratio": digest_ratio,
        "digest_index_method": digest_method,
        "critical_layers": critical_layers,
        "qcompute_end": num_layers,
        "materialized_digest": True,
    }
    if num_chunks == 0:
        state.metadata = {**empty_payload, "context_positions_by_layer": []}
        state.indices_by_method = {
            digest_method: {
                "digest_index_version": DIGEST_INDEX_VERSION,
                "method": digest_method,
                "ranked_indices_by_chunk": [],
                "ranked_indices_by_layer_chunk": [],
            }
        }
        state.set_index_metadata(
            ranked_indices_by_chunk=[],
            ranked_indices_by_layer_chunk=[],
            orig_chunk_ranges=[],
            total_tokens=0,
            num_layers=num_layers,
        )
        return

    doc_original_ranges = original_ranges[1:-1]
    doc_chunk_lengths = [end - start for start, end in doc_original_ranges]

    total_tokens = int(original_ranges[-1][0])
    n_sink = max(0, int(state.context_n_sink or 0))

    kvzip_scores = _scores_from_kvzip_dict(
        doc_chunk_lengths, state.kvzip_scores_by_layer_doc_chunk, num_layers
    )
    selected_index = _rank_scores_by_layer(kvzip_scores, doc_chunk_lengths, n_sink)
    ranked_shared = selected_index[0] if selected_index else []
    selected_payload = {"context_n_sink": n_sink, "score_reduce": "max_head_query"}

    state.metadata = {
        "digest_index_version": DIGEST_INDEX_VERSION,
        "orig_chunk_ranges": [[int(s), int(e)] for s, e in original_ranges],
        "total_tokens": total_tokens,
        "available_methods": [digest_method],
        "layer_wise": True,
        "num_layers": num_layers,
        "digest_ratio": digest_ratio,
        "digest_index_method": digest_method,
        "critical_layers": critical_layers,
        "qcompute_end": num_layers,
        "materialized_digest": True,
    }
    state.indices_by_method = {
        digest_method: {
            "digest_index_version": DIGEST_INDEX_VERSION,
            "method": digest_method,
            "ranked_indices_by_chunk": ranked_shared,
            "ranked_indices_by_layer_chunk": selected_index,
            "layer_wise": True,
            **selected_payload,
        }
    }

    state.set_index_metadata(
        ranked_indices_by_chunk=ranked_shared,
        ranked_indices_by_layer_chunk=selected_index,
        orig_chunk_ranges=original_ranges[:-1],
        total_tokens=total_tokens,
        num_layers=num_layers,
    )
    state.build_context_positions(digest_ratio=digest_ratio)
    state.metadata["context_positions_by_layer"] = [
        [int(x) for x in layer_positions]
        for layer_positions in state.context_positions_by_layer
    ]
    state.kvzip_scores_by_layer_doc_chunk = {}


def export_payload(state: QcFuseRequestState, method: Optional[str] = None):
    if not state.metadata or not state.indices_by_method:
        raise ValueError("Digest indices have not been built")
    method = normalize_method(method or state.metadata.get("digest_index_method"))
    if method not in state.indices_by_method:
        available = sorted(state.indices_by_method)
        raise ValueError(
            f"Digest method {method!r} not available; available={available}"
        )
    return (
        copy.deepcopy(state.metadata),
        {method: copy.deepcopy(state.indices_by_method[method])},
    )


def save(state: QcFuseRequestState, sample_dir: str) -> None:
    if not state.metadata or not state.indices_by_method:
        raise ValueError("Digest indices have not been built")
    os.makedirs(sample_dir, exist_ok=True)
    with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
        json.dump(state.metadata, f, indent=2)
    for method, payload in state.indices_by_method.items():
        with open(os.path.join(sample_dir, index_filename(method)), "w") as f:
            json.dump(payload, f, indent=2)


def load(sample_dir: str, method: Optional[str] = None):
    method = normalize_method(method)
    meta_path = os.path.join(sample_dir, "metadata.json")
    index_path = os.path.join(sample_dir, index_filename(method))
    with open(meta_path, "r") as f:
        meta = json.load(f)
    if meta.get("digest_index_version") != DIGEST_INDEX_VERSION:
        raise ValueError(
            f"Unsupported digest metadata version in {meta_path}: "
            f"{meta.get('digest_index_version')}"
        )
    available = meta.get("available_methods", [])
    if method not in available:
        raise ValueError(
            f"Digest method {method!r} not available in {meta_path}; available={available}"
        )
    with open(index_path, "r") as f:
        index = json.load(f)
    if index.get("method") != method:
        raise ValueError(
            f"Digest index method mismatch in {index_path}: {index.get('method')} != {method}"
        )
    return meta, index


@torch.no_grad()
def accumulate_kvzip_layer_score(
    state: QcFuseRequestState,
    layer_id: int,
    q: torch.Tensor,
    k: torch.Tensor,
    rotary_emb=None,
) -> None:
    """Accumulate per-layer kvzip doc-token scores into ``state``.

    ``q`` / ``k`` are the *un-rotated* layer activations with a merged head dim:
    ``q`` is ``[seq, num_heads * head_dim]`` and ``k`` is
    ``[seq, num_kv_heads * head_dim]`` (matching the original SGLang call site,
    which passes the pre-reshape attention inputs).
    """
    if state.blend_style != BlendStyle.KVCOMPUTE or not state.is_contextblend:
        return
    qcompute_end = state.qcompute_end
    if qcompute_end is not None and int(layer_id) >= int(qcompute_end):
        return

    doc_ranges = state.digest_aug_doc_ranges
    zip_ranges = state.digest_aug_zip_ranges
    if not doc_ranges or not zip_ranges:
        return

    params = state.att_params
    if params is None:
        return

    head_dim = int(params.head_dim)
    num_heads = int(params.num_heads)
    num_kv_heads = int(params.num_kv_heads)
    # vLLM hands q/k as [tokens, heads, head_dim] (3D); SGLang's original used
    # the flattened [tokens, heads*head_dim] (2D) layout, so branch on dim.
    is_3d = q.dim() == 3
    if is_3d:
        num_heads = q.shape[1]
        num_kv_heads = k.shape[1]
    else:
        if q.shape[-1] != num_heads * head_dim:
            num_heads = q.shape[-1] // head_dim
        if k.shape[-1] != num_kv_heads * head_dim:
            num_kv_heads = k.shape[-1] // head_dim
    if num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        return

    kvzip_layer_scores = state.kvzip_scores_by_layer_doc_chunk.setdefault(
        int(layer_id), [None] * len(doc_ranges)
    )
    if len(kvzip_layer_scores) != len(doc_ranges):
        kvzip_layer_scores = [None] * len(doc_ranges)
        state.kvzip_scores_by_layer_doc_chunk[int(layer_id)] = kvzip_layer_scores

    positions = ensure_forward_positions(state).to(device=q.device)
    if rotary_emb is not None:
        # rotary may mutate q/k in-place; never perturb the real attention path
        q_for_score, k_for_score = rotary_emb(positions, q.clone(), k.clone())
    else:
        q_for_score, k_for_score = q, k

    sys_start, sys_end = state.digest_aug_sys_range or (0, 0)
    n_sink = max(0, int(state.context_n_sink or 0))
    sink_end = min(int(sys_end), int(sys_start) + n_sink)
    sink_indices = list(range(int(sys_start), sink_end))
    scale = 1.0 / math.sqrt(float(head_dim))
    num_groups = num_heads // num_kv_heads

    for chunk_idx, ((doc_start, doc_end), (zip_start, zip_end)) in enumerate(
        zip(doc_ranges, zip_ranges)
    ):
        doc_start = int(doc_start)
        doc_end = int(doc_end)
        zip_start = int(zip_start)
        zip_end = int(zip_end)
        doc_len = doc_end - doc_start
        zip_len = zip_end - zip_start
        if doc_len <= 0 or zip_len <= 0:
            continue

        key_indices = (
            sink_indices
            + list(range(doc_start, doc_end))
            + list(range(zip_start, zip_end))
        )
        key_index_t = torch.tensor(key_indices, dtype=torch.long, device=q.device)
        if is_3d:
            q_zip = q_for_score[zip_start:zip_end]
            k_sub = k_for_score.index_select(0, key_index_t)
        else:
            q_zip = q_for_score[zip_start:zip_end].view(zip_len, num_heads, head_dim)
            k_sub = k_for_score.index_select(0, key_index_t).view(
                len(key_indices), num_kv_heads, head_dim
            )

        q_grouped = (
            q_zip.permute(1, 0, 2)
            .contiguous()
            .view(num_kv_heads, num_groups, zip_len, head_dim)
        )
        k_grouped = k_sub.permute(1, 0, 2).contiguous()
        logits = (
            torch.matmul(q_grouped, k_grouped.unsqueeze(1).transpose(-1, -2)) * scale
        )

        sink_len = len(sink_indices)
        zip_col_start = sink_len + doc_len
        if zip_len > 1:
            causal_mask = torch.ones(
                zip_len, zip_len, dtype=torch.bool, device=q.device
            ).triu(1)
            logits[..., zip_col_start:] = logits[..., zip_col_start:].masked_fill(
                causal_mask.view(1, 1, zip_len, zip_len),
                torch.finfo(logits.dtype).min,
            )

        attn = torch.softmax(logits.float(), dim=-1)
        attn_doc = attn[..., sink_len : sink_len + doc_len]
        kvzip_score_cpu = attn_doc.amax(dim=(0, 1, 2)).detach().float().cpu()

        prev = kvzip_layer_scores[chunk_idx]
        if prev is None:
            kvzip_layer_scores[chunk_idx] = kvzip_score_cpu
        else:
            kvzip_layer_scores[chunk_idx] = torch.maximum(prev, kvzip_score_cpu)
