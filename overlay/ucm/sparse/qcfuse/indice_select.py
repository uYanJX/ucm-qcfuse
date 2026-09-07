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
"""Token selection for QCFuse DO_BLEND.

Faithful port of SGLang's ``indice_select.py``.  The query-only path and the
request-region geometry are pure torch and CPU-testable here; the attention
based ``_compute_layer_fusion`` path calls the Triton importance kernel, which
is implemented in ``ucm.sparse.qcfuse.triton_attention_score``.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple

import torch

from ucm.sparse.qcfuse.blend_info import QcFuseRequestState, SelectMode


class _ReqRegion(NamedTuple):
    req_start: int
    quest_start: int
    quest_end: int
    rag_start: int
    rag_len: int


def _compute_request_boundaries(info: QcFuseRequestState) -> Tuple[int, torch.Tensor]:
    chunk_loc = info.chunk_loc_list
    req_len_list = info.req_len_list
    device = chunk_loc.device
    num_reqs = len(req_len_list)
    req_boundaries = torch.zeros(num_reqs + 1, dtype=torch.long, device=device)
    req_boundaries[1:] = torch.cumsum(req_len_list, dim=0)
    return num_reqs, req_boundaries


def _get_request_region(
    chunk_loc: torch.Tensor, req_boundaries: torch.Tensor, req_idx: int
) -> _ReqRegion:
    req_start_chunk = req_boundaries[req_idx].item()
    req_end_chunk = req_boundaries[req_idx + 1].item()

    prefix_idx = req_start_chunk
    quest_idx = req_end_chunk - 1

    req_start = chunk_loc[prefix_idx].item()
    quest_start = chunk_loc[quest_idx].item()
    quest_end = chunk_loc[quest_idx + 1].item()

    if quest_idx > prefix_idx + 1:
        rag_start = chunk_loc[prefix_idx + 1].item()
        rag_len = quest_start - rag_start
    else:
        rag_start = quest_start
        rag_len = 0

    return _ReqRegion(req_start, quest_start, quest_end, rag_start, rag_len)


def _build_final_output(
    final_indices_list: List[torch.Tensor], req_lens_list: List[int], device
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.cat(final_indices_list), torch.tensor(req_lens_list, device=device)


def _compute_budget(length: int, ratio: float) -> int:
    if ratio <= 0:
        return 0
    return min(length, max(1, int(length * ratio)))


class IndiceSelector:
    @staticmethod
    def select(
        info: QcFuseRequestState,
        old_k: Optional[List[torch.Tensor]] = None,
        old_q: Optional[List[torch.Tensor]] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if info.ratio <= 0:
            return IndiceSelector._compute_query_only(info)
        if info.ratio >= 1:
            return IndiceSelector._compute_full(info)
        if info.select_mode != SelectMode.ATTN:
            raise ValueError(f"Unsupported select mode for release: {info.select_mode}")
        if old_k is None or old_q is None:
            raise ValueError("ATTN mode requires old_k and old_q")
        return IndiceSelector._compute_layer_fusion(info, old_k, old_q, positions)

    @staticmethod
    def _compute_full(
        info: QcFuseRequestState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk_loc = info.chunk_loc_list
        device = chunk_loc.device
        num_reqs, req_boundaries = _compute_request_boundaries(info)
        selections: List[torch.Tensor] = []
        lengths: List[int] = []
        for req_idx in range(num_reqs):
            region = _get_request_region(chunk_loc, req_boundaries, req_idx)
            indices = torch.arange(region.req_start, region.quest_end, device=device)
            selections.append(indices)
            lengths.append(indices.numel())
        return _build_final_output(selections, lengths, device)

    @staticmethod
    def _compute_query_only(
        info: QcFuseRequestState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk_loc = info.chunk_loc_list
        device = chunk_loc.device

        num_reqs, req_boundaries = _compute_request_boundaries(info)
        final_indices_list: List[torch.Tensor] = []
        req_lens_list: List[int] = []

        for req_idx in range(num_reqs):
            r = _get_request_region(chunk_loc, req_boundaries, req_idx)
            quest_indices = torch.arange(r.quest_start, r.quest_end, device=device)
            final_indices_list.append(quest_indices)
            req_lens_list.append(quest_indices.numel())

        return _build_final_output(final_indices_list, req_lens_list, device)

    @staticmethod
    def _compute_layer_fusion(
        info: QcFuseRequestState,
        old_k: List[torch.Tensor],
        old_q: List[torch.Tensor],
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attention-based fusion selection.

        The scoring kernel lives in ``ucm.sparse.qcfuse.triton_attention_score``.
        """
        from ucm.sparse.qcfuse.triton_attention_score import (
            compute_att_full_softmax_importance,
        )

        params = info.att_params
        chunk_loc = info.chunk_loc_list
        device = chunk_loc.device
        ratio = info.ratio
        layer_ids = [int(x) for x in (info.critical_layers or [])]
        if not layer_ids:
            layer_ids = list(range(int(info.attn_start), int(info.attn_end)))
        num_layers = len(layer_ids)
        rotary_emb = getattr(info, "rotary_emb", None)

        num_reqs, req_boundaries = _compute_request_boundaries(info)
        final_indices_list: List[torch.Tensor] = []
        req_lens_list: List[int] = []

        old_k_stacked = torch.stack(old_k, dim=0)
        old_q_stacked = torch.stack(old_q, dim=0)
        if not old_q_stacked.is_cuda:
            raise RuntimeError("ATTN selection requires CUDA.")

        if info.critical_layers:
            query_k_layers = info.get_query_k_layers(layer_ids)
        else:
            query_k_layers = info.get_all_query_k(info.attn_start, info.attn_end)
        query_k_stacked = torch.stack(query_k_layers, dim=0)

        q_lens = info.q_lens
        q_offsets = info.q_offsets
        query_k_lens = info.query_k_lens

        q_loc = 0
        query_k_loc = 0
        for req_idx in range(num_reqs):
            r = _get_request_region(chunk_loc, req_boundaries, req_idx)
            q_len = int(q_lens[req_idx])
            q_offset = int(q_offsets[req_idx])
            query_k_len = int(query_k_lens[req_idx])
            q_pos_start = r.quest_start + q_offset
            q_pos_end = q_pos_start + q_len
            k_abs_end = q_pos_end
            quest_indices = torch.arange(r.quest_start, r.quest_end, device=device)

            if r.rag_len > 0:
                prefix_len = r.quest_start - r.req_start
                target_start = r.rag_start - r.req_start
                full_q_start = prefix_len + q_offset

                q_chunk = old_q_stacked[:, q_loc : q_loc + q_len].reshape(
                    num_layers, q_len, params.num_heads, params.head_dim
                )
                prefix_k = old_k_stacked[:, r.req_start : r.quest_start]
                query_k = query_k_stacked[:, query_k_loc : query_k_loc + query_k_len]
                k_full = torch.cat([prefix_k, query_k], dim=1).reshape(
                    num_layers,
                    prefix_len + query_k_len,
                    params.num_kv_heads,
                    params.head_dim,
                )

                if rotary_emb is not None and positions is not None:
                    q_chunk = IndiceSelector._rotate_stacked(
                        rotary_emb,
                        positions[q_pos_start:q_pos_end],
                        q_chunk,
                        num_layers,
                        params.num_heads,
                        params.head_dim,
                        is_query=True,
                    )
                    k_full = IndiceSelector._rotate_stacked(
                        rotary_emb,
                        positions[r.req_start : k_abs_end],
                        k_full,
                        num_layers,
                        params.num_kv_heads,
                        params.head_dim,
                        is_query=False,
                    )

                importance = compute_att_full_softmax_importance(
                    q_chunk,
                    k_full,
                    target_start=target_start,
                    target_len=r.rag_len,
                    q_start=full_q_start,
                )

                k_budget = _compute_budget(r.rag_len, ratio)
                if k_budget > 0:
                    _, top_idx = torch.topk(
                        importance, k=min(k_budget, importance.numel())
                    )
                    selected_rag, _ = torch.sort(top_idx + r.rag_start)
                    req_selection = torch.cat([selected_rag, quest_indices])
                else:
                    req_selection = quest_indices
            else:
                req_selection = quest_indices

            q_loc += q_len
            query_k_loc += query_k_len
            final_indices_list.append(req_selection)
            req_lens_list.append(req_selection.numel())

        return _build_final_output(final_indices_list, req_lens_list, device)

    @staticmethod
    def _rotate_stacked(
        rotary_emb,
        token_positions: torch.Tensor,
        tensor: torch.Tensor,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        *,
        is_query: bool,
    ) -> torch.Tensor:
        pos = token_positions.repeat(num_layers)
        # vLLM's rotary_emb (``ops.rotary_embedding``) expects 3D
        # ``[tokens, heads, head_dim]``; SGLang's original passed the flattened
        # 2D ``[tokens, heads*head_dim]`` layout, so reshape here first.
        three_d = tensor.reshape(-1, num_heads, head_dim).contiguous()
        # SGLang's rotary implementation accepts aliased Q/K inputs, while
        # vLLM's CUDA op updates both tensors in place.  Keep the unused input
        # separate so the tensor being scored is rotated exactly once.
        scratch = torch.empty_like(three_d)
        if is_query:
            q_rot, _ = rotary_emb(pos, three_d, scratch)
            rotated = q_rot
        else:
            _, k_rot = rotary_emb(pos, scratch, three_d)
            rotated = k_rot
        return rotated.reshape(num_layers, -1, num_heads, head_dim)
