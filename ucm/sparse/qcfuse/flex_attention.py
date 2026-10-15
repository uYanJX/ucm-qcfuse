"""Opt-in position-based attention; request masks retain every cached KV row.

This backend changes floating-point kernel arithmetic. Keep it experimental
until matched model-quality regressions pass. QCOMPUTE and decode do not use it.
"""

from dataclasses import dataclass
from typing import Any

import torch

_compiled_attention = None


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        # vLLM creates selection tensors under inference_mode. Their positions
        # remain immutable for the request, but they have no version counter.
        return None


@dataclass
class FlexAttentionPlan:
    source: torch.Tensor
    version: Any
    kv_length: int
    key_block: int
    positions: torch.Tensor
    block_mask: Any

    def matches(self, positions, kv_length, key_block):
        return (
            self.source is positions
            and self.version == _version(positions)
            and self.kv_length == kv_length
            and self.key_block == key_block
        )


def make_position_plan(positions, kv_length, key_block=64):
    """Build O(query_blocks * key_blocks) metadata, never a dense Q-by-K mask."""
    from torch.nn.attention.flex_attention import BlockMask

    query_length = positions.numel()
    if query_length == 0 or kv_length <= 0 or key_block not in (64, 128):
        raise ValueError("Flex attention requires nonempty rows and key blocks 64/128")
    stable_positions = positions.to(dtype=torch.int32).clone()
    query_block = 128
    blocks = (query_length + query_block - 1) // query_block
    key_blocks = (kv_length + key_block - 1) // key_block
    padding = blocks * query_block - query_length
    minimum = (
        torch.nn.functional.pad(stable_positions, (0, padding), value=kv_length)
        .view(blocks, query_block)
        .amin(dim=1)
    )
    maximum = (
        torch.nn.functional.pad(stable_positions, (0, padding), value=-1)
        .view(blocks, query_block)
        .amax(dim=1)
    )
    full_count = ((minimum + 1) // key_block).clamp(0, key_blocks).to(torch.int32)
    total_count = (
        ((maximum + key_block) // key_block).clamp(0, key_blocks).to(torch.int32)
    )
    partial_count = total_count - full_count
    columns = torch.arange(key_blocks, dtype=torch.int32, device=positions.device)
    full_indices = columns.expand(blocks, -1).contiguous()
    partial_indices = full_count[:, None] + columns[None, :]
    partial_indices = torch.where(
        columns[None, :] < partial_count[:, None], partial_indices, 0
    )

    def position_mask(batch, head, query, key):
        return (
            (query < query_length)
            & (key < kv_length)
            & (key <= stable_positions[query.clamp(max=query_length - 1)])
        )

    block_mask = BlockMask.from_kv_blocks(
        partial_count[None, None],
        partial_indices[None, None],
        full_kv_num_blocks=full_count[None, None],
        full_kv_indices=full_indices[None, None],
        BLOCK_SIZE=(query_block, key_block),
        mask_mod=position_mask,
        seq_lengths=(query_length, kv_length),
        compute_q_blocks=False,
    )
    return FlexAttentionPlan(
        positions,
        _version(positions),
        kv_length,
        key_block,
        stable_positions,
        block_mask,
    )


def flex_positions_attention_fwd(
    query, key, value, output, states, q_lens, kv_lens, scale, key_block=64
):
    """Run isolated request slices with immutable positions and layer-reused masks."""
    global _compiled_attention
    if _compiled_attention is None:
        from torch.nn.attention.flex_attention import flex_attention

        # Frozen quality suites contain many different sequence lengths. A
        # static compile per source can exhaust Dynamo's recompilation limit.
        _compiled_attention = torch.compile(flex_attention, dynamic=True)
    if len(states) != len(q_lens) or len(states) != len(kv_lens):
        raise ValueError("Flex attention request layout does not match lengths")
    if (
        sum(q_lens) != len(query)
        or sum(kv_lens) != len(key)
        or key.shape != value.shape
    ):
        raise ValueError("Flex attention request lengths do not cover Q/K/V")
    query_offset = key_offset = 0
    for state, query_length, kv_length in zip(states, q_lens, kv_lens):
        positions = state.positions
        if positions is None or positions.numel() != query_length:
            raise ValueError(
                "Flex attention requires original positions for every Q row"
            )
        plan = state.attn_meta.get("flex_position_plan")
        if plan is None or not plan.matches(positions, kv_length, key_block):
            plan = make_position_plan(positions, kv_length, key_block)
            state.attn_meta["flex_position_plan"] = plan
        q_slice = slice(query_offset, query_offset + query_length)
        kv_slice = slice(key_offset, key_offset + kv_length)
        result = _compiled_attention(
            query[q_slice].transpose(0, 1)[None],
            key[kv_slice].transpose(0, 1)[None],
            value[kv_slice].transpose(0, 1)[None],
            block_mask=plan.block_mask,
            scale=scale,
            enable_gqa=True,
            kernel_options={"BLOCK_M": 128, "BLOCK_N": key_block, "USE_TMA": False},
        )
        output[q_slice].copy_(result[0].transpose(0, 1))
        query_offset += query_length
        key_offset += kv_length
