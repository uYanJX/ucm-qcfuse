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
"""Per-request state and small enums for the QCFuse sparse algorithm.

This is a faithful port of SGLang's ``cache_blender_info.py`` with one
deliberate change: the module-level singletons ``HackBlendKVPool`` /
``ContextBlendPool`` / ``BatchBlendInfo`` are merged into a single
:class:`QcFuseRequestState` *instance*.  QCFuse assumed a single-process,
serial, single-request world; UCM's scheduler/worker model forbids that, so
every request now owns its state and it travels in ``QcFuseMetadata``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

DEFAULT_DIGEST_RATIO = 0.1


class BlendStyle(Enum):
    """Which pass of the three-pass QCFuse pipeline a request is in."""

    KVCOMPUTE = 0  # offline: full prefill, build digest, write SSD cache
    QCOMPUTE = 1  # online: query attends over compressed view to score tokens
    DO_BLEND = 2  # online: fuse selected RAG tokens into cached K/V
    DO_BLEND_FINISH = 3  # online: final ratio pass, direct reference

    @classmethod
    def parse(cls, value):
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            name = value.upper()
            if name in cls.__members__:
                return cls[name]
            return None
        if isinstance(value, int):
            try:
                return cls(value)
            except ValueError:
                return None
        return None


class SelectMode(Enum):
    """Selection strategy for cache blending."""

    ATTN = "attn"  # Attention based


@dataclass
class AttParams:
    """Attention geometry needed for selection / digest scoring."""

    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    num_layers: int = 32


@dataclass
class QcFuseRequestState:
    """Per-request state for the QCFuse three-pass pipeline.

    Subsumes three SGLang globals into one instance so concurrent requests can
    never share state:

    * ``BatchBlendInfo``  -> the scalar/tensor ``blend_info`` fields below,
    * ``HackBlendKVPool`` -> ``k_buffer`` / ``v_buffer`` / ``q_buffer`` / ``query_k_buffer``,
    * ``ContextBlendPool`` -> ``ctx_k_buffer`` / ``ctx_v_buffer`` / digest rankings.
    """

    # ---- request identity and blend configuration --------------------- #
    request_id: Optional[str] = None
    query_session_id: Optional[str] = None
    blend_style: Optional[BlendStyle] = None
    select_mode: SelectMode = SelectMode.ATTN
    ratio: float = 0.3
    att_params: Optional[AttParams] = None
    start: int = 0
    attn_start: int = 0
    attn_end: int = -1
    chunk_lens: Optional[torch.Tensor] = None
    chunk_loc_list: Optional[torch.Tensor] = None
    req_len_list: Optional[torch.Tensor] = None
    blend_top_indices: Optional[torch.Tensor] = None
    blend_top_lens: Optional[torch.Tensor] = None
    # number of tokens the model actually processes for this request (the full,
    # un-narrowed prefill length).  Captured in ``narrow_stream`` at the DO_BLEND
    # start layer, then used by ``QcFuse.model_finished`` to scatter the narrowed
    # residual stream back into a full-length buffer before the final RMSNorm /
    # logits.  Without this, vLLM samples logits at the last full token position
    # but only gets the narrowed [n_sel] rows -> index out of bounds.
    full_token_count: Optional[int] = None
    fake_q: Optional[torch.Tensor] = None
    quest_indices: Optional[torch.Tensor] = None
    query_indices: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None
    attention_positions: Optional[torch.Tensor] = None
    init_attmeta: bool = False
    is_contextblend: bool = False
    context_cache_source: str = "query"
    context_n_sink: int = 4
    digest_index_method: str = "kvzip"
    digest_ratio: Optional[float] = None
    critical_layers: Optional[List[int]] = None
    critical_layers_set: Optional[set] = None
    qcompute_end: Optional[int] = None
    digest_keep_indices: Optional[List[int]] = None
    digest_original_chunk_loc_list: Optional[torch.Tensor] = None
    digest_aug_sys_range: Optional[Tuple[int, int]] = None
    digest_aug_doc_ranges: Optional[List[Tuple[int, int]]] = None
    digest_aug_zip_ranges: Optional[List[Tuple[int, int]]] = None
    keep_layers_set: Optional[set] = None
    # per-request SSD sample dirs (was KVSSDManager's per-request configure args).
    # Carried on the state so the worker-side hooks can pass them to the store
    # without touching connector instance state.
    sample_dir_chunk: Optional[str] = None
    sample_dir_query: Optional[str] = None
    # Rotary embedding used while aligning selected cache positions.
    rotary_emb: object = None

    # ---- transient caches for the QCOMPUTE / DO_BLEND passes ----------- #
    # runtime metadata the attention kernel needs (was scattered on blend_info)
    attn_meta: Dict = field(default_factory=dict)
    _context_fake_q: Optional[torch.Tensor] = None
    _context_positions_t_by_layer: Dict[int, torch.Tensor] = field(default_factory=dict)

    # ---- KV pools (was HackBlendKVPool) ------------------------------- #
    k_buffer: List[Optional[torch.Tensor]] = field(default_factory=list)
    v_buffer: List[Optional[torch.Tensor]] = field(default_factory=list)
    q_buffer: List[Optional[torch.Tensor]] = field(default_factory=list)
    query_k_buffer: List[Optional[torch.Tensor]] = field(default_factory=list)
    q_lens: List[int] = field(default_factory=list)
    q_offsets: List[int] = field(default_factory=list)
    query_k_lens: List[int] = field(default_factory=list)

    # ---- context pool (was ContextBlendPool) -------------------------- #
    ctx_k_buffer: List[Optional[torch.Tensor]] = field(default_factory=list)
    ctx_v_buffer: List[Optional[torch.Tensor]] = field(default_factory=list)
    ranked_indices_by_chunk: List[List[int]] = field(default_factory=list)
    ranked_indices_by_layer_chunk: List[List[List[int]]] = field(default_factory=list)
    orig_chunk_ranges: List[Tuple[int, int]] = field(default_factory=list)
    context_positions: List[int] = field(default_factory=list)
    context_positions_by_layer: List[List[int]] = field(default_factory=list)
    total_tokens: int = 0

    # ---- digest index manager state (was DigestIndexManager) ----------- #
    metadata: Dict = field(default_factory=dict)
    indices_by_method: Dict = field(default_factory=dict)
    kvzip_scores_by_layer_doc_chunk: Dict[int, List[Optional[torch.Tensor]]] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------ #
    #  device materialisation
    # ------------------------------------------------------------------ #
    def to(self, device: torch.device) -> "QcFuseRequestState":
        """Move the scheduler-built CPU index tensors onto the worker device.

        ``build_request_state`` runs in the scheduler process (which has no CUDA
        device), so ``chunk_loc_list``/``req_len_list``/``quest_indices``/
        ``query_indices``/``digest_original_chunk_loc_list`` are created on CPU
        and arrive here through the scheduler-output pickle.  The worker-side
        attention hooks index on-device q/k/v with these, so materialise them
        once on the worker GPU (idempotent: on-device tensors are left alone).
        """
        for name in (
            "chunk_lens",
            "chunk_loc_list",
            "req_len_list",
            "quest_indices",
            "query_indices",
            "digest_original_chunk_loc_list",
        ):
            t = getattr(self, name)
            if t is not None and t.device != device:
                setattr(self, name, t.to(device))
        return self

    # ------------------------------------------------------------------ #
    #  KV pool helpers
    # ------------------------------------------------------------------ #
    def init_buffers(self, num_layers: int) -> None:
        self.k_buffer = [None] * num_layers
        self.v_buffer = [None] * num_layers
        self.q_buffer = []
        self.query_k_buffer = []
        self.q_lens = []
        self.q_offsets = []
        self.query_k_lens = []

    def has_kv(self, layer_id: int) -> bool:
        return (
            0 <= layer_id < len(self.k_buffer)
            and self.k_buffer[layer_id] is not None
            and isinstance(self.k_buffer[layer_id], torch.Tensor)
            and self.k_buffer[layer_id].numel() > 0
        )

    def get_kv(
        self, layer_id: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.k_buffer[layer_id], self.v_buffer[layer_id]

    def get_kv_layers(self, layer_ids: Sequence[int]):
        return (
            [self.k_buffer[int(i)] for i in layer_ids],
            [self.v_buffer[int(i)] for i in layer_ids],
        )

    def get_all_kv(self, start: int, end: int):
        return self.k_buffer[start:end], self.v_buffer[start:end]

    def put_kv(self, k: torch.Tensor, v: torch.Tensor, layer_id: int) -> None:
        k = k.clone()
        while len(self.k_buffer) <= layer_id:
            self.k_buffer.append(None)
            self.v_buffer.append(None)
        if self.k_buffer[layer_id] is None:
            self.k_buffer[layer_id] = k
        else:
            self.k_buffer[layer_id] = torch.cat([self.k_buffer[layer_id], k], dim=0)
        if self.v_buffer[layer_id] is None:
            self.v_buffer[layer_id] = v
        else:
            self.v_buffer[layer_id] = torch.cat([self.v_buffer[layer_id], v], dim=0)

    def get_q(self, layer_id: int):
        return self.q_buffer[layer_id]

    def put_q(self, q: torch.Tensor, layer_id: int) -> None:
        while len(self.q_buffer) <= layer_id:
            self.q_buffer.append(None)
        if self.q_buffer[layer_id] is None:
            self.q_buffer[layer_id] = q
        else:
            self.q_buffer[layer_id] = torch.cat([self.q_buffer[layer_id], q], dim=0)

    def get_all_q(self, start: int, end: int):
        return self.q_buffer[start:end]

    def get_q_layers(self, layer_ids: Sequence[int]):
        return [self.q_buffer[int(i)] for i in layer_ids]

    def put_query_k(self, k: torch.Tensor, layer_id: int) -> None:
        while len(self.query_k_buffer) <= layer_id:
            self.query_k_buffer.append(None)
        if self.query_k_buffer[layer_id] is None:
            self.query_k_buffer[layer_id] = k
        else:
            self.query_k_buffer[layer_id] = torch.cat(
                [self.query_k_buffer[layer_id], k], dim=0
            )

    def get_all_query_k(self, start: int, end: int):
        return self.query_k_buffer[start:end]

    def get_query_k_layers(self, layer_ids: Sequence[int]):
        return [self.query_k_buffer[int(i)] for i in layer_ids]

    # ------------------------------------------------------------------ #
    #  Context pool helpers
    # ------------------------------------------------------------------ #
    def set_index_metadata(
        self,
        *,
        ranked_indices_by_chunk=None,
        ranked_indices_by_layer_chunk=None,
        orig_chunk_ranges: Sequence[Sequence[int]],
        total_tokens: int,
        num_layers: Optional[int] = None,
    ) -> None:
        self.ranked_indices_by_chunk = [
            list(x) for x in (ranked_indices_by_chunk or [])
        ]
        self.ranked_indices_by_layer_chunk = [
            [list(chunk) for chunk in layer]
            for layer in (ranked_indices_by_layer_chunk or [])
        ]
        self.orig_chunk_ranges = [tuple(x) for x in orig_chunk_ranges]
        self.total_tokens = int(total_tokens)
        self.context_positions = []
        self.context_positions_by_layer = []

    @staticmethod
    def _coalesce_sorted_positions(
        positions: Sequence[int],
    ) -> List[Tuple[int, int, int]]:
        if not positions:
            return []
        spans = []
        run_start = int(positions[0])
        prev = run_start
        out_start = 0
        for raw_pos in positions[1:]:
            pos = int(raw_pos)
            if pos == prev + 1:
                prev = pos
                continue
            spans.append((run_start, prev + 1, out_start))
            out_start += prev - run_start + 1
            run_start = pos
            prev = pos
        spans.append((run_start, prev + 1, out_start))
        return spans

    def _ranked_indices_for_layer(self, layer_id: int):
        if self.ranked_indices_by_layer_chunk and 0 <= int(layer_id) < len(
            self.ranked_indices_by_layer_chunk
        ):
            return self.ranked_indices_by_layer_chunk[int(layer_id)]
        return self.ranked_indices_by_chunk

    def _build_context_positions_for_layer(
        self, layer_id: int = 0, digest_ratio: float = DEFAULT_DIGEST_RATIO
    ) -> List[int]:
        positions: List[int] = []
        ranked_by_chunk = self._ranked_indices_for_layer(layer_id)
        for chunk_idx, (orig_start, orig_end) in enumerate(self.orig_chunk_ranges):
            orig_start = int(orig_start)
            orig_end = int(orig_end)
            chunk_len = orig_end - orig_start
            if chunk_len <= 0:
                continue
            if chunk_idx == 0:  # system prompt always kept
                positions.extend(range(orig_start, orig_end))
                continue
            ratio = min(1.0, max(0.0, float(digest_ratio)))
            n_left = min(chunk_len, int(math.ceil(chunk_len * ratio)))
            local_selected: List[int] = []
            seen = set()
            ranked = (
                ranked_by_chunk[chunk_idx] if chunk_idx < len(ranked_by_chunk) else []
            )
            for raw_idx in ranked[:n_left]:
                idx = int(raw_idx)
                if idx in seen or idx < 0 or idx >= chunk_len:
                    continue
                seen.add(idx)
                local_selected.append(idx)
            local_selected.sort()
            positions.extend(orig_start + idx for idx in local_selected)
        return positions

    def build_context_positions(
        self, digest_ratio: float = DEFAULT_DIGEST_RATIO
    ) -> List[int]:
        num_layers = max(
            len(self.ranked_indices_by_layer_chunk),
            len(self.ctx_k_buffer),
            1,
        )
        self.context_positions_by_layer = []
        for layer_id in range(num_layers):
            self.context_positions_by_layer.append(
                self._build_context_positions_for_layer(
                    layer_id=layer_id, digest_ratio=digest_ratio
                )
            )
        self.context_positions = (
            self.context_positions_by_layer[0]
            if self.context_positions_by_layer
            else []
        )
        return self.context_positions

    def get_context_positions(self, layer_id: int) -> List[int]:
        if self.context_positions_by_layer and 0 <= int(layer_id) < len(
            self.context_positions_by_layer
        ):
            return self.context_positions_by_layer[int(layer_id)]
        return self.context_positions

    def init_ctx_buffers(self, num_layers: int) -> None:
        self.ctx_k_buffer = [None] * num_layers
        self.ctx_v_buffer = [None] * num_layers

    def get_ctx(
        self, layer_id: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return compressed context K/V for one layer (was ContextBlendPool.get)."""
        if not (0 <= int(layer_id) < len(self.ctx_k_buffer)):
            return None, None
        return self.ctx_k_buffer[int(layer_id)], self.ctx_v_buffer[int(layer_id)]

    def put_ctx(self, k: torch.Tensor, v: torch.Tensor, layer_id: int) -> None:
        while len(self.ctx_k_buffer) <= int(layer_id):
            self.ctx_k_buffer.append(None)
            self.ctx_v_buffer.append(None)
        self.ctx_k_buffer[int(layer_id)] = k
        self.ctx_v_buffer[int(layer_id)] = v

    # ------------------------------------------------------------------ #
    #  Blend-info helper (was BatchBlendInfo.should_collect_q)
    # ------------------------------------------------------------------ #
    def should_collect_q(self, layer_id: int) -> bool:
        if self.blend_style != BlendStyle.QCOMPUTE:
            return False
        if self.critical_layers_set:
            return int(layer_id) in self.critical_layers_set
        attn_start = int(self.attn_start or 0)
        if self.attn_end is None:
            return layer_id >= attn_start
        attn_end = int(self.attn_end)
        if attn_end < 0:
            return layer_id >= attn_start
        return attn_start <= layer_id < attn_end


# --------------------------------------------------------------------------- #
#  Separator token splitting (was CacheBlender.split_tokens / split_text_tokens)
# --------------------------------------------------------------------------- #
def _find_pattern_matches_numpy(
    input_ids: np.ndarray, sep_token: np.ndarray
) -> np.ndarray:
    sep_len = len(sep_token)
    n = len(input_ids)
    if sep_len > n:
        return np.array([], dtype=np.int64)
    if sep_len == 1:
        return np.where(input_ids == sep_token[0])[0]
    windows = np.lib.stride_tricks.sliding_window_view(input_ids, sep_len)
    matches = np.all(windows == sep_token, axis=1)
    return np.where(matches)[0]


def split_tokens(
    input_text: Optional[str],
    input_ids: List[int],
    separator: str,
    sep_token: List[int],
) -> Tuple[Optional[str], List[int], Optional[List[int]]]:
    """Split token ids by a separator pattern; return (text, ids, loc_list)."""
    sep_len = len(sep_token)
    n = len(input_ids)
    if sep_len == 0 or sep_len > n:
        return input_text, input_ids, None

    input_arr = np.asarray(input_ids, dtype=np.int64)
    sep_arr = np.asarray(sep_token, dtype=np.int64)
    matches = _find_pattern_matches_numpy(input_arr, sep_arr)
    num_matches = len(matches)
    if num_matches == 0:
        return input_text, input_ids, [0, len(input_ids)]

    keep_mask = np.ones(n, dtype=np.bool_)
    for match_idx in matches:
        keep_mask[match_idx : match_idx + sep_len] = False

    new_input_ids = input_arr[keep_mask].tolist()
    blend_loc_list = []
    current_new_pos = 0
    prev_end = 0
    for match_idx in matches:
        # match_idx 是 np.int64，直接参与运算会把 blend_loc_list 的元素
        # 提升成 np.int64，导致后续塞进 extra_args 走 msgpack 序列化时
        # 报 "Object of type numpy.int64 is not serializable"。转成 Python int。
        match_idx = int(match_idx)
        segment_len = match_idx - prev_end
        blend_loc_list.append(current_new_pos)
        current_new_pos += segment_len
        prev_end = match_idx + sep_len
    blend_loc_list.append(current_new_pos)
    blend_loc_list.append(len(new_input_ids))

    new_text = input_text.replace(separator, "") if input_text is not None else None
    return new_text, new_input_ids, blend_loc_list
