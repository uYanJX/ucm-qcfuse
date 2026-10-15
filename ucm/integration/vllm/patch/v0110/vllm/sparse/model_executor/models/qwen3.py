import re
from typing import Optional

import torch
from torch import nn

from ucm.sparse.state import (
    maybe_execute_sparse_ffn_begin,
    maybe_execute_sparse_ffn_finished,
    maybe_execute_sparse_qwen3_pre_rope,
)


class Qwen3Attention(nn.Module):

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_shape = (*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        k_shape = (*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        q = self.q_norm(q.view(q_shape)).view(q.shape)
        k = self.k_norm(k.view(k_shape)).view(k.shape)

        layer_id = getattr(self.attn, "layer_idx", None)
        if layer_id is None:
            layer_id = getattr(self.attn, "layer_id", None)
        if layer_id is None:
            match = re.search(r"layers\.(\d+)\.", getattr(self.attn, "layer_name", ""))
            layer_id = int(match.group(1)) if match else None
        if layer_id is None:
            raise RuntimeError("QCFuse requires a layer index on Qwen3 attention")

        q, k, v, rope_applied = maybe_execute_sparse_qwen3_pre_rope(
            q, k, v, positions, int(layer_id), self.rotary_emb
        )
        if not rope_applied:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = maybe_execute_sparse_ffn_begin(
            hidden_states, residual
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        hidden_states, residual = maybe_execute_sparse_ffn_finished(
            hidden_states, residual
        )

        return hidden_states, residual
