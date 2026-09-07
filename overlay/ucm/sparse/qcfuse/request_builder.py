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
"""Populate a :class:`QcFuseRequestState` from per-request blend args.

This is the vLLM-side port of SGLang's ``schedule_batch.py`` blend-info builder
(the ~180 lines that turn ``blend_loc_list`` + the per-request ``blend_style`` /
``ratio`` / cache-path fields into the fields the three-pass pipeline consumes).

It is deliberately import-light (``torch`` + the two pure modules below) so the
CPU unit tests can exercise the field-mapping and separator-splitting logic
without importing vLLM.  The connector calls :func:`build_request_state` once
per request at admission time; the result is the object that
``QcFuse.request_begin`` stores and ``build_sparse_meta`` ships to the worker.

Each returned object describes one request. The worker combines these objects
into a step-level batch without merging their cache paths or transient state.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ucm.sparse.qcfuse.blend_info import (
    DEFAULT_DIGEST_RATIO,
    AttParams,
    BlendStyle,
    QcFuseRequestState,
    SelectMode,
    split_tokens,
)
from ucm.sparse.qcfuse import digest_index


def _as_int(value, default: int = 0) -> int:
    return default if value is None else int(value)


def build_request_state(
    *,
    blend_args: Dict,
    prompt_token_ids: List[int],
    sep_token: List[int],
    att_params: AttParams,
    device: str = "cpu",
) -> QcFuseRequestState:
    """Build a populated ``QcFuseRequestState`` for one request.

    Parameters
    ----------
    blend_args:
        The per-request fields carried by the connector, exactly as the eval
        runner's ``_blend_args`` + ``_ssd_args`` produce them: ``blend_style``,
        ``start``, ``ratio``, ``method``, (``attn_start``/``attn_end`` when
        ``method == "attn"``), ``is_contextblend``, ``context_cache_source``,
        ``digest_ratio``, ``digest_index_method``, ``critical_layers``,
        ``context_n_sink``, and the cache paths.
    prompt_token_ids:
        The request's tokenized prompt *before* separator removal.
    sep_token:
        The tokenized form of ``BLEND_SEP`` (the frontend delimiter).
    att_params:
        Model attention geometry (heads / kv-heads / head-dim / layers).
    device:
        Target device string for the tensor fields.
    """
    state = QcFuseRequestState()
    num_layers = int(att_params.num_layers)

    # ------------------------------------------------------------------ #
    #  scalar blend fields (schedule_batch.py:1358-1396)
    # ------------------------------------------------------------------ #
    state.blend_style = BlendStyle.parse(blend_args.get("blend_style"))
    if state.blend_style is None:
        raise ValueError(
            f"Invalid QCFuse blend_style: {blend_args.get('blend_style')!r}"
        )
    state.att_params = att_params
    state.start = _as_int(blend_args.get("start"))
    state.ratio = float(blend_args.get("ratio", 0.3))
    if not 0.0 <= state.ratio <= 1.0:
        raise ValueError(f"QCFuse ratio must be in [0, 1], got {state.ratio}")
    state.attn_start = _as_int(blend_args.get("attn_start"))
    attn_end = blend_args.get("attn_end", -1)
    state.attn_end = (
        num_layers if (attn_end is None or int(attn_end) == -1) else int(attn_end)
    )

    is_contextblend = bool(blend_args.get("is_contextblend", False))
    state.is_contextblend = is_contextblend
    state.context_cache_source = (
        (blend_args.get("context_cache_source") or "query")
        if is_contextblend
        else "none"
    )
    state.context_n_sink = _as_int(blend_args.get("context_n_sink"), 4)
    digest_ratio = blend_args.get("digest_ratio")
    state.digest_ratio = (
        DEFAULT_DIGEST_RATIO if digest_ratio is None else float(digest_ratio)
    )
    state.digest_index_method = blend_args.get("digest_index_method") or "kvzip"
    state.sample_dir_chunk = blend_args.get("ssd_cache_path_chunk")
    state.sample_dir_query = blend_args.get("ssd_cache_path_query")
    state.query_session_id = blend_args.get("query_session_id")
    if state.query_session_id is None:
        state.query_session_id = state.sample_dir_query
    critical_layers = [int(x) for x in (blend_args.get("critical_layers") or [])]
    state.critical_layers = critical_layers
    state.critical_layers_set = set(critical_layers)
    state.qcompute_end = max(critical_layers) + 1 if critical_layers else None
    if state.qcompute_end is not None and state.qcompute_end > num_layers:
        raise ValueError(
            f"QCFuse critical layer exceeds model depth {num_layers}: {critical_layers}"
        )

    # ------------------------------------------------------------------ #
    #  separator split -> chunk_loc_list / req_len_list
    # ------------------------------------------------------------------ #
    separator = blend_args.get("separator", "<|blendsep|>")
    precomputed_locs = blend_args.get("blend_loc_list")
    if precomputed_locs is not None:
        # The eval runner already stripped the separators from the prompt and
        # computed the chunk boundaries in the stripped space.  Use them
        # directly (the prompt arriving here is already stripped, so split_tokens
        # would find no separator and collapse everything into one chunk).
        blend_loc_list = [int(x) for x in precomputed_locs]
    else:
        _, new_ids, blend_loc_list = split_tokens(
            None, list(prompt_token_ids), separator, list(sep_token)
        )
        if blend_loc_list is None:
            # No separator tokens present: the whole prompt is one chunk.
            blend_loc_list = [0, len(prompt_token_ids)]

    has_ssd_paths = (
        blend_args.get("ssd_cache_path_chunk") is not None
        or blend_args.get("ssd_cache_path_query") is not None
    )
    is_query_offline = (
        state.blend_style == BlendStyle.KVCOMPUTE
        and is_contextblend
        and has_ssd_paths
        and blend_args.get("ssd_cache_path_query") is not None
    )

    forward_locs = list(blend_loc_list)
    if is_query_offline:
        transformed = digest_index.prepare_augmented_locs_for_request(forward_locs)
        if transformed is None:
            raise ValueError(
                "ContextBlend query KVCOMPUTE expects augmented chunks "
                "with layout sys, doc, zipprompt, ..., query."
            )
        forward_locs = transformed["forward_locs"]
        original_locs = transformed["original_locs"]
        state.digest_original_chunk_loc_list = torch.tensor(
            original_locs, dtype=torch.int64, device=device
        )
        state.digest_keep_indices = list(transformed["keep_indices"])
        state.digest_aug_sys_range = tuple(transformed["aug_sys_range"])
        state.digest_aug_doc_ranges = [tuple(r) for r in transformed["aug_doc_ranges"]]
        state.digest_aug_zip_ranges = [tuple(r) for r in transformed["aug_zip_ranges"]]

    chunk_locs_t = torch.tensor(forward_locs, dtype=torch.int64, device=device)
    state.chunk_loc_list = chunk_locs_t
    state.chunk_lens = chunk_locs_t.diff()
    state.req_len_list = torch.tensor(
        [len(forward_locs) - 1], dtype=torch.int32, device=device
    )

    # ------------------------------------------------------------------ #
    #  style-specific geometry
    # ------------------------------------------------------------------ #
    if state.blend_style in (BlendStyle.QCOMPUTE, BlendStyle.KVCOMPUTE):
        if state.blend_style == BlendStyle.QCOMPUTE:
            state.init_attmeta = True
            # q_lens / q_offsets / query_k_lens + quest/query indices.
            # Single request: req_boundaries = [0, num_chunks].
            num_chunks = len(forward_locs) - 1
            if num_chunks < 2:
                raise ValueError(
                    "QCOMPUTE prompt must contain at least two chunks "
                    "(query-prefix and question separated by the blend separator)"
                )
            # second chunk is the question; [0, second_chunk_end) is the query.
            starts = chunk_locs_t[1].item()
            ends = chunk_locs_t[2].item()
            lengths = ends - starts
            req_starts = chunk_locs_t[0].item()
            q_offset = starts - req_starts
            query_length = q_offset + lengths

            state.q_lens = [lengths]
            state.q_offsets = [q_offset]
            state.query_k_lens = [query_length]
            state.quest_indices = torch.arange(
                starts, ends, dtype=torch.long, device=device
            )
            state.query_indices = torch.arange(
                req_starts, ends, dtype=torch.long, device=device
            )
    else:
        # DO_BLEND / DO_BLEND_FINISH
        state.select_mode = SelectMode(blend_args.get("method", "attn"))
        # keep_layers_set: layers preloaded before DO_BLEND that survive across
        # ratio rounds (schedule_batch.py:1463-1481).
        if state.select_mode == SelectMode.ATTN:
            if critical_layers:
                state.keep_layers_set = set(critical_layers)
            elif state.attn_start == 0 and state.attn_end == num_layers:
                state.keep_layers_set = set(range(num_layers))
            else:
                state.keep_layers_set = {state.attn_start}
        else:
            state.keep_layers_set = set()

    return state
