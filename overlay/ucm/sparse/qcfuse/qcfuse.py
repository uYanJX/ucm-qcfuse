"""QCFuse sparse attention integration for vLLM 0.11 Qwen3 models."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
from vllm.forward_context import ForwardContext

from ucm.sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseMetadata,
    UcmSparseRole,
)
from ucm.sparse.qcfuse import blender, digest_index
from ucm.sparse.qcfuse.batch import QcFuseStepBatch, cumulative_starts
from ucm.sparse.qcfuse.blend_info import BlendStyle, QcFuseRequestState

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request


@dataclass
class QcFuseMetadata(UcmSparseMetadata):
    states: Dict[str, QcFuseRequestState] = field(default_factory=dict)


@dataclass
class _PreparedAttention:
    layer_id: int
    q_lens: List[int]
    kv_lens: List[int]
    q_positions: torch.Tensor
    write_kv_cache: bool


class QcFuse(UcmSparseBase):
    """Per-request QCFuse state with homogeneous online batching."""

    def __init__(self, vllm_config: "VllmConfig", role: UcmSparseRole):
        super().__init__(vllm_config, role)
        self.device = vllm_config.device_config.device
        self.block_size = vllm_config.cache_config.block_size
        self._scheduler_states: Dict[str, QcFuseRequestState] = {}
        self._worker_states: Dict[str, QcFuseRequestState] = {}
        self._step = QcFuseStepBatch()
        self._store = None
        self._prepared: Optional[_PreparedAttention] = None
        self._current_layer_id: Optional[int] = None
        self._skip_next_impl = False
        self._qcompute_stash: Dict[str, dict] = {}

    def attach_store(self, store, **_kwargs) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Scheduler and worker lifecycle
    # ------------------------------------------------------------------
    def request_begin(self, request_id: Union[int, str], prompt_token_ids: List[int]):
        self._scheduler_states[str(request_id)] = QcFuseRequestState(
            request_id=str(request_id)
        )

    def get_state(self, request_id: Union[int, str]) -> QcFuseRequestState:
        return self._scheduler_states[str(request_id)]

    def request_finished_in_scheduler(self, request_id: Union[int, str]):
        self._scheduler_states.pop(str(request_id), None)

    def request_finished_in_worker(self, request_id: Union[int, str]):
        state = self._worker_states.pop(str(request_id), None)
        if state is not None and self._store is not None:
            finish = getattr(self._store, "finish_request", None)
            if callable(finish):
                finish(str(request_id))

    def estimate_num_slots_sparsed(self, request: "Request") -> int:
        return INVALID_SLOT

    def update_states(self, scheduler_output) -> None:
        for request_id in getattr(scheduler_output, "finished_req_ids", ()):
            self.request_finished_in_worker(request_id)

    def build_sparse_meta(
        self, scheduler_output, requests, input_batch, attn_metadata
    ) -> UcmSparseMetadata:
        connector_meta = getattr(scheduler_output, "kv_connector_metadata", None)
        request_meta = getattr(connector_meta, "request_meta", {}) or {}
        for request_id, item in request_meta.items():
            state = getattr(item, "state", None)
            if state is not None:
                state.request_id = str(request_id)
                self._worker_states[str(request_id)] = state

        request_ids = [
            str(request_id)
            for request_id in input_batch.req_ids[: input_batch.num_reqs]
        ]
        computed = input_batch.num_computed_tokens_cpu[: input_batch.num_reqs]
        prompts = input_batch.num_prompt_tokens[: input_batch.num_reqs]
        self._step = QcFuseStepBatch.build(
            request_ids,
            scheduler_output.num_scheduled_tokens,
            self._worker_states,
            computed,
            prompts,
        )
        self._prepared = None

        if self._step.active:
            for entry in self._step.entries:
                entry.state.to(torch.device(self.device))
            if self._store is not None:
                begin = getattr(self._store, "begin_requests", None)
                if callable(begin):
                    begin(self._step.entries)
        return QcFuseMetadata(
            states={entry.request_id: entry.state for entry in self._step.entries}
        )

    def effective_layer_end(self, start_layer: int, end_layer: int) -> int:
        if not self._step.active or self._step.phase != BlendStyle.QCOMPUTE:
            return end_layer
        layer_ends = {
            int(entry.state.qcompute_end or end_layer) for entry in self._step.entries
        }
        if len(layer_ends) != 1:
            raise RuntimeError(
                "All QCOMPUTE requests in a batch need the same end layer"
            )
        return min(end_layer, layer_ends.pop())

    # ------------------------------------------------------------------
    # Qwen3 pre-RoPE integration
    # ------------------------------------------------------------------
    def _stash_qcompute(self, state: QcFuseRequestState) -> None:
        session_id = state.query_session_id
        if not session_id:
            raise ValueError("QCOMPUTE requires query_session_id")
        self._qcompute_stash[session_id] = {
            "q_buffer": list(state.q_buffer),
            "query_k_buffer": list(state.query_k_buffer),
            "q_lens": list(state.q_lens),
            "q_offsets": list(state.q_offsets),
            "query_k_lens": list(state.query_k_lens),
        }
        if len(self._qcompute_stash) > 256:
            raise RuntimeError("QCFuse QCOMPUTE hand-off limit exceeded")

    def _materialize_blend_inputs(self, state: QcFuseRequestState) -> None:
        if self._store is not None and state.sample_dir_query is not None:
            for layer_id in state.critical_layers or []:
                layer_id = int(layer_id)
                if not state.has_kv(layer_id):
                    key, value = self._store.load_query_layer(
                        state.sample_dir_query, "critical", layer_id
                    )
                    state.put_kv(key, value, layer_id)

        session_id = state.query_session_id
        stash = self._qcompute_stash.pop(session_id, None)
        if stash is None:
            raise RuntimeError(
                f"DO_BLEND has no matching QCOMPUTE state for {session_id!r}"
            )
        state.q_buffer = stash["q_buffer"]
        state.query_k_buffer = stash["query_k_buffer"]
        state.q_lens = stash["q_lens"]
        state.q_offsets = stash["q_offsets"]
        state.query_k_lens = stash["query_k_lens"]

    def qwen3_pre_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        layer_id: int,
        rotary_emb,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        self._current_layer_id = int(layer_id)
        self._prepared = None
        if not self._step.active:
            return query, key, value, False

        expected = sum(
            entry.layer_input_tokens(layer_id) for entry in self._step.entries
        )
        if query.shape[0] != expected or positions.shape[-1] != expected:
            raise RuntimeError(
                f"QCFuse layer {layer_id} input layout mismatch: "
                f"q={query.shape[0]}, positions={positions.shape[-1]}, expected={expected}"
            )

        query_parts = []
        key_parts = []
        value_parts = []
        q_lens = []
        kv_lens = []
        attention_positions = []
        offset = 0
        phase = self._step.phase

        for entry in self._step.entries:
            state = entry.state
            count = entry.layer_input_tokens(layer_id)
            q_part = query[offset : offset + count]
            k_part = key[offset : offset + count]
            v_part = value[offset : offset + count]
            pos_part = positions[offset : offset + count]
            offset += count

            if phase == BlendStyle.KVCOMPUTE:
                state.put_kv(k_part, v_part, layer_id)
                digest_index.accumulate_kvzip_layer_score(
                    state, layer_id, q_part, k_part, rotary_emb
                )

            if state.should_collect_q(layer_id):
                state.put_q(q_part.index_select(0, state.quest_indices), layer_id)
                state.put_query_k(k_part.index_select(0, state.query_indices), layer_id)

            if (
                phase in (BlendStyle.DO_BLEND, BlendStyle.DO_BLEND_FINISH)
                and layer_id == state.start
            ):
                self._materialize_blend_inputs(state)

            q_part, k_part, v_part = blender.blend_layer(
                state,
                layer_id,
                q_part,
                k_part,
                v_part,
                pos_part,
                rotary_emb,
                store=self._store,
                sample_dir_chunk=state.sample_dir_chunk,
                sample_dir_query=state.sample_dir_query,
            )

            critical = [int(item) for item in state.critical_layers or []]
            if (
                state.should_collect_q(layer_id)
                and critical
                and layer_id == max(critical)
            ):
                self._stash_qcompute(state)

            query_parts.append(q_part)
            key_parts.append(k_part)
            value_parts.append(v_part)
            if phase == BlendStyle.KVCOMPUTE:
                chunk_lens = [int(length) for length in state.chunk_lens.tolist()]
                if sum(chunk_lens) != int(q_part.shape[0]):
                    raise RuntimeError(
                        f"QCFuse KVCOMPUTE chunk layout has {sum(chunk_lens)} "
                        f"tokens, expected {q_part.shape[0]}"
                    )
                q_lens.extend(chunk_lens)
                kv_lens.extend(chunk_lens)
                attention_positions.append(
                    torch.cat(
                        [
                            torch.arange(
                                length, dtype=torch.int32, device=q_part.device
                            )
                            for length in chunk_lens
                        ]
                    )
                )
            else:
                q_lens.append(int(q_part.shape[0]))
                kv_lens.append(int(k_part.shape[0]))
                if state.attention_positions is not None:
                    attention_positions.append(state.attention_positions)

        custom_attention = phase in (BlendStyle.KVCOMPUTE, BlendStyle.QCOMPUTE) or (
            phase in (BlendStyle.DO_BLEND, BlendStyle.DO_BLEND_FINISH)
            and layer_id >= self._step.entries[0].state.start
        )
        if custom_attention:
            if len(attention_positions) != len(self._step.entries):
                raise RuntimeError("QCFuse did not prepare all attention positions")
            self._prepared = _PreparedAttention(
                layer_id=layer_id,
                q_lens=q_lens,
                kv_lens=kv_lens,
                q_positions=torch.cat(attention_positions).to(torch.int32),
                write_kv_cache=phase
                in (
                    BlendStyle.KVCOMPUTE,
                    BlendStyle.DO_BLEND,
                    BlendStyle.DO_BLEND_FINISH,
                ),
            )

        return (
            torch.cat(query_parts),
            torch.cat(key_parts),
            torch.cat(value_parts),
            True,
        )

    # ------------------------------------------------------------------
    # Attention, residual-stream and logits hooks
    # ------------------------------------------------------------------
    @staticmethod
    def _layer_id_from_name(layer_name: str, attn=None) -> Optional[int]:
        layer_index = getattr(attn, "layer_idx", None)
        if layer_index is not None:
            return int(layer_index)
        match = re.search(r"layers\.(\d+)\.", layer_name or "")
        return int(match.group(1)) if match else None

    def should_skip_impl_forward(self) -> bool:
        return self._skip_next_impl

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        output: Optional[torch.Tensor] = None,
        phase: Optional[str] = None,
        k_hash: Optional[torch.Tensor] = None,
        decode_ql_nope: Optional[torch.Tensor] = None,
        decode_q_pe: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._skip_next_impl = False
        prepared = self._prepared
        if prepared is None:
            return query, key, value, output

        layer = forward_context.no_compile_layers[layer_name]
        layer_id = self._layer_id_from_name(layer_name, layer)
        if layer_id != prepared.layer_id:
            raise RuntimeError(
                f"QCFuse prepared layer {prepared.layer_id}, got {layer_id}"
            )
        self._run_sparse_attention(query, key, value, output, layer, prepared)
        if prepared.write_kv_cache:
            self._write_kv_cache(forward_context, layer_name, key, value)
        self._skip_next_impl = True
        return query, key, value, output

    def _run_sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        layer,
        prepared: _PreparedAttention,
    ) -> None:
        from ucm.sparse.qcfuse.ragged_attention import ragged_positions_attention_fwd

        if output is None or output.shape != query.shape:
            raise RuntimeError("QCFuse requires vLLM attention's shaped output buffer")
        device = query.device
        q_lens = torch.tensor(prepared.q_lens, dtype=torch.int32, device=device)
        kv_lens = torch.tensor(prepared.kv_lens, dtype=torch.int32, device=device)
        impl = layer.impl
        window = getattr(impl, "sliding_window", (-1, -1))
        sliding_window = -1 if not window or window[0] < 0 else int(window[0]) + 1
        ragged_positions_attention_fwd(
            query,
            key,
            value,
            output,
            cumulative_starts(prepared.q_lens, device),
            q_lens,
            cumulative_starts(prepared.kv_lens, device),
            kv_lens,
            prepared.q_positions,
            max(prepared.q_lens),
            is_causal=True,
            sm_scale=float(getattr(impl, "scale", query.shape[-1] ** -0.5)),
            logit_cap=float(getattr(impl, "logits_soft_cap", 0.0)),
            sliding_window_size=sliding_window,
        )

    @staticmethod
    def _write_kv_cache(
        forward_context: ForwardContext,
        layer_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        from vllm._custom_ops import reshape_and_cache_flash

        layer = forward_context.no_compile_layers[layer_name]
        key_cache, value_cache = layer.kv_cache[forward_context.virtual_engine].unbind(
            0
        )
        metadata = forward_context.attn_metadata
        if isinstance(metadata, dict):
            metadata = metadata[layer_name]
        if metadata.slot_mapping.numel() != key.shape[0]:
            raise RuntimeError(
                f"QCFuse KV cache layout mismatch: {metadata.slot_mapping.numel()} "
                f"slots for {key.shape[0]} tokens"
            )
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            metadata.slot_mapping,
            layer.impl.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def layer_begin(self, positions, hidden_states, residual):
        if not self._step.active:
            return positions, hidden_states, residual
        phase = self._step.phase
        if phase not in (BlendStyle.DO_BLEND, BlendStyle.DO_BLEND_FINISH):
            return positions, hidden_states, residual
        states = [entry.state for entry in self._step.entries]
        if all(state.blend_top_indices is not None for state in states):
            selected_positions = torch.cat([state.positions for state in states])
            if positions.shape[0] != selected_positions.shape[0]:
                positions = selected_positions
        return positions, hidden_states, residual

    def ffn_begin(self, hidden_states, residual):
        if not self._step.active or self._current_layer_id is None:
            return hidden_states, residual
        if self._step.phase not in (BlendStyle.DO_BLEND, BlendStyle.DO_BLEND_FINISH):
            return hidden_states, residual
        if self._current_layer_id != self._step.entries[0].state.start:
            return hidden_states, residual

        indices = self._step.global_selected_indices(hidden_states.device)
        for entry in self._step.entries:
            entry.state.full_token_count = entry.scheduled_tokens
        if hidden_states.shape[0] != indices.shape[0]:
            hidden_states = hidden_states.index_select(0, indices)
        if residual is not None and residual.shape[0] != indices.shape[0]:
            residual = residual.index_select(0, indices)
        return hidden_states, residual

    def model_finished(self, hidden_states, residual):
        if not self._step.active:
            return hidden_states, residual
        if self._step.phase not in (BlendStyle.DO_BLEND, BlendStyle.DO_BLEND_FINISH):
            return hidden_states, residual
        indices = self._step.global_selected_indices(hidden_states.device)
        full_count = self._step.original_token_count
        if hidden_states.shape[0] == full_count:
            return hidden_states, residual
        if hidden_states.shape[0] != indices.shape[0]:
            raise RuntimeError("QCFuse cannot restore the narrowed model output")

        full_hidden = hidden_states.new_zeros((full_count, *hidden_states.shape[1:]))
        full_hidden.index_copy_(0, indices, hidden_states)
        full_residual = None
        if residual is not None:
            full_residual = residual.new_zeros((full_count, *residual.shape[1:]))
            full_residual.index_copy_(0, indices, residual)
        return full_hidden, full_residual
