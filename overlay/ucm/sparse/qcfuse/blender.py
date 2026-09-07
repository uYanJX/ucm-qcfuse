"""QCFuse's three phase, per-request Q/K/V transformation.

Inputs are Qwen3 Q/K after q/k normalization and before RoPE. Keys written to
disk therefore stay raw and are placed on the final request position grid once.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ucm.sparse.qcfuse import digest_index, indice_select
from ucm.sparse.qcfuse.blend_info import (
    DEFAULT_DIGEST_RATIO,
    BlendStyle,
    QcFuseRequestState,
    SelectMode,
)


def _apply_rotary(rotary_emb, positions, q, k):
    if rotary_emb is None:
        return q, k
    return rotary_emb(positions, q, k)


def _rotate_key(rotary_emb, positions: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    if rotary_emb is None or key.numel() == 0:
        return key
    fake_q = torch.empty_like(key)
    _, rotated = rotary_emb(positions, fake_q, key)
    return rotated


def build_context_pool(state: QcFuseRequestState) -> None:
    digest_index.build_all_indices(state)


def narrow_stream(
    state: QcFuseRequestState,
    layer_id: int,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if (
        state.blend_style in (BlendStyle.DO_BLEND, BlendStyle.DO_BLEND_FINISH)
        and layer_id == state.start
        and state.blend_top_indices is not None
        and hidden_states.shape[0] > 1
    ):
        indices = state.blend_top_indices
        state.full_token_count = int(hidden_states.shape[0])
        return hidden_states[indices], residual[indices]
    return hidden_states, residual


def _context_positions_t(
    state: QcFuseRequestState, layer_id: int, positions, device
) -> torch.Tensor:
    cached = state._context_positions_t_by_layer.get(int(layer_id))
    if cached is None or cached.device != device or cached.numel() != len(positions):
        cached = torch.tensor(positions, dtype=torch.long, device=device)
        state._context_positions_t_by_layer[int(layer_id)] = cached
    return cached


def _save_offline_cache(
    state: QcFuseRequestState,
    store,
    sample_dir_chunk: str,
    sample_dir_query: Optional[str],
) -> None:
    num_layers = int(state.att_params.num_layers)
    if state.is_contextblend and not state.metadata:
        build_context_pool(state)
    kv_by_layer = {layer: state.get_kv(layer) for layer in range(num_layers)}
    store.dump_full_cache(
        sample_dir_chunk,
        kv_by_layer,
        num_layers,
        token_indices=state.digest_keep_indices,
    )
    if not state.is_contextblend or sample_dir_query is None:
        return

    qcompute_end = max(0, min(int(state.qcompute_end or num_layers), num_layers))
    keep = state.digest_keep_indices
    digest_indices = {}
    for layer_id, positions in enumerate(
        state.context_positions_by_layer[:qcompute_end]
    ):
        if keep is None:
            indices = positions
        else:
            indices = [int(keep[int(position)]) for position in positions]
        digest_indices[layer_id] = torch.as_tensor(indices, dtype=torch.long)

    digest_meta, indices_by_method = digest_index.export_payload(state)
    critical_layers = [int(layer) for layer in state.critical_layers or []]
    store.dump_query_cache(
        sample_dir_query,
        qcompute_end,
        {layer: state.get_kv(layer) for layer in range(qcompute_end)},
        digest_indices,
        critical_layers,
        {layer: state.get_kv(layer) for layer in critical_layers},
        digest_meta,
        indices_by_method,
        digest_ratio=state.digest_ratio,
        digest_index_method=state.digest_index_method,
        qcompute_end=qcompute_end,
        critical_token_indices=keep,
    )


def _query_context(
    state: QcFuseRequestState,
    layer_id: int,
    store,
    sample_dir_query: Optional[str],
):
    if store is not None and sample_dir_query is not None:
        ctx_k, ctx_v = store.load_query_layer(sample_dir_query, "digest", layer_id)
        meta = store.load_query_meta(sample_dir_query)["metadata"]
        context_positions = meta["context_positions_by_layer"][layer_id]
        state.total_tokens = int(meta["total_tokens"])
        return ctx_k, ctx_v, [int(position) for position in context_positions]

    context_positions = state.get_context_positions(layer_id)
    if not context_positions:
        state.build_context_positions(
            digest_ratio=state.digest_ratio or DEFAULT_DIGEST_RATIO
        )
        context_positions = state.get_context_positions(layer_id)
    old_k, old_v = state.get_kv(layer_id)
    indices = _context_positions_t(state, layer_id, context_positions, old_k.device)
    return (
        old_k.index_select(0, indices),
        old_v.index_select(0, indices),
        context_positions,
    )


def blend_layer(
    state: QcFuseRequestState,
    layer_id: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb=None,
    store=None,
    sample_dir_chunk: Optional[str] = None,
    sample_dir_query: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return RoPE-applied tensors for one request and one attention layer."""
    state.rotary_emb = rotary_emb

    if state.blend_style == BlendStyle.KVCOMPUTE:
        forward_positions = digest_index.ensure_forward_positions(state)
        q_rot, k_rot = _apply_rotary(rotary_emb, forward_positions, q, k)
        if layer_id == int(state.att_params.num_layers) - 1:
            if state.is_contextblend:
                build_context_pool(state)
            if store is not None and sample_dir_chunk is not None:
                _save_offline_cache(state, store, sample_dir_chunk, sample_dir_query)
        return q_rot, k_rot, v

    if state.blend_style == BlendStyle.QCOMPUTE and state.is_contextblend:
        ctx_k, ctx_v, context_positions = _query_context(
            state, layer_id, store, sample_dir_query
        )
        if len(context_positions) != int(ctx_k.shape[0]):
            raise ValueError(
                f"QCFuse digest position/KV mismatch at layer {layer_id}: "
                f"{len(context_positions)} != {ctx_k.shape[0]}"
            )

        context_pos = _context_positions_t(
            state, layer_id, context_positions, ctx_k.device
        )
        ctx_k = _rotate_key(rotary_emb, context_pos, ctx_k)
        query_positions = torch.arange(
            state.total_tokens,
            state.total_tokens + q.shape[0],
            dtype=torch.long,
            device=q.device,
        )
        q, k = _apply_rotary(rotary_emb, query_positions, q, k)
        state.positions = query_positions
        state.attention_positions = torch.arange(
            ctx_k.shape[0],
            ctx_k.shape[0] + q.shape[0],
            dtype=torch.int32,
            device=q.device,
        )
        return q, torch.cat((ctx_k, k), dim=0), torch.cat((ctx_v, v), dim=0)

    if state.blend_style in (BlendStyle.DO_BLEND, BlendStyle.DO_BLEND_FINISH):
        if layer_id < state.start:
            q, k = _apply_rotary(rotary_emb, positions, q, k)
            return q, k, v

        if layer_id == state.start:
            state.init_attmeta = True
            if state.ratio <= 0:
                indices, lens = indice_select.IndiceSelector.select(state)
            elif state.select_mode == SelectMode.ATTN:
                layers = state.critical_layers or list(
                    range(state.attn_start, state.attn_end)
                )
                old_k, _ = state.get_kv_layers(layers)
                old_q = state.get_q_layers(layers)
                indices, lens = indice_select.IndiceSelector.select(
                    state, old_k=old_k, old_q=old_q, positions=positions
                )
            else:
                raise ValueError(f"Unsupported QCFuse selection: {state.select_mode}")

            state.blend_top_indices = indices
            state.blend_top_lens = lens
            state.positions = positions.index_select(0, indices)
            state.attention_positions = state.positions.to(torch.int32)
            q, k = _apply_rotary(rotary_emb, positions, q, k)
            return q.index_select(0, indices), k, v

        if store is not None and sample_dir_chunk is not None:
            old_k, old_v = store.load_chunk_layer(sample_dir_chunk, layer_id)
        else:
            old_k, old_v = state.get_kv(layer_id)
            old_k, old_v = old_k.clone(), old_v.clone()

        indices = state.blend_top_indices
        old_k.index_copy_(0, indices, k)
        old_v.index_copy_(0, indices, v)
        q, _ = _apply_rotary(rotary_emb, state.positions, q, k)
        full_positions = torch.arange(old_k.shape[0], device=old_k.device)
        old_k = _rotate_key(rotary_emb, full_positions, old_k)
        state.attention_positions = state.positions.to(torch.int32)
        return q, old_k, old_v

    q, k = _apply_rotary(rotary_emb, positions, q, k)
    return q, k, v
