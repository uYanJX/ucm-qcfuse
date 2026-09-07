import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="QCFuse kernel tests require CUDA"
)


def test_attention_importance_matches_query_head_layer_mean():
    from ucm.sparse.qcfuse.triton_attention_score import (
        compute_att_full_softmax_importance,
    )

    torch.manual_seed(7)
    layers, seq_q, seq_k = 2, 5, 11
    heads_q, heads_k, head_dim = 4, 2, 32
    target_start, target_len, q_start = 2, 6, 6
    query = torch.randn(
        layers, seq_q, heads_q, head_dim, device="cuda", dtype=torch.float16
    )
    key = torch.randn(
        layers, seq_k, heads_k, head_dim, device="cuda", dtype=torch.float16
    )

    expanded_key = key.repeat_interleave(heads_q // heads_k, dim=2)
    logits = torch.einsum(
        "lqhd,lkhd->lhqk", query.float(), expanded_key.float()
    ) / math.sqrt(head_dim)
    q_positions = q_start + torch.arange(seq_q, device="cuda")
    k_positions = torch.arange(seq_k, device="cuda")
    logits.masked_fill_(
        ~(k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)).view(
            1, 1, seq_q, seq_k
        ),
        float("-inf"),
    )
    expected = torch.softmax(logits, dim=-1)[
        ..., target_start : target_start + target_len
    ].mean(dim=(0, 1, 2))

    actual = compute_att_full_softmax_importance(
        query,
        key,
        target_start=target_start,
        target_len=target_len,
        q_start=q_start,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)


def test_attention_importance_matches_upstream_qcfuse():
    reference_root = os.environ.get("QCFUSE_REFERENCE_DIR")
    if not reference_root:
        pytest.skip("set QCFUSE_REFERENCE_DIR to run the upstream differential")
    module_path = Path(reference_root) / "srt/utils/triton_attention_score.py"
    if not module_path.exists():
        pytest.fail(f"QCFuse reference module is missing: {module_path}")

    spec = importlib.util.spec_from_file_location(
        "qcfuse_upstream_triton_attention_score", module_path
    )
    upstream = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(upstream)

    from ucm.sparse.qcfuse.triton_attention_score import (
        compute_att_full_softmax_importance,
    )

    torch.manual_seed(31)
    query = torch.randn(3, 7, 8, 64, device="cuda", dtype=torch.float16)
    key = torch.randn(3, 19, 2, 64, device="cuda", dtype=torch.float16)
    kwargs = {
        "target_start": 3,
        "target_len": 11,
        "q_start": 12,
        "causal": True,
    }
    actual = compute_att_full_softmax_importance(query, key, **kwargs)
    expected = upstream.compute_att_full_softmax_importance(query, key, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("batch_size", [2, 4])
def test_ragged_attention_matches_torch_for_unequal_batches(batch_size):
    from ucm.sparse.qcfuse.batch import cumulative_starts
    from ucm.sparse.qcfuse.ragged_attention import ragged_positions_attention_fwd

    torch.manual_seed(17 + batch_size)
    q_lens = [2 + index for index in range(batch_size)]
    kv_lens = [length + 4 for length in q_lens]
    heads_q, heads_k, head_dim = 4, 2, 32
    query = torch.randn(
        sum(q_lens), heads_q, head_dim, device="cuda", dtype=torch.float16
    )
    key = torch.randn(
        sum(kv_lens), heads_k, head_dim, device="cuda", dtype=torch.float16
    )
    value = torch.randn_like(key)
    q_positions = torch.cat(
        [
            torch.arange(kv_len - q_len, kv_len, device="cuda")
            for q_len, kv_len in zip(q_lens, kv_lens)
        ]
    ).to(torch.int32)
    output = torch.empty_like(query)
    q_lens_t = torch.tensor(q_lens, dtype=torch.int32, device="cuda")
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")

    ragged_positions_attention_fwd(
        query,
        key,
        value,
        output,
        cumulative_starts(q_lens, query.device),
        q_lens_t,
        cumulative_starts(kv_lens, query.device),
        kv_lens_t,
        q_positions,
        max(q_lens),
        sm_scale=head_dim**-0.5,
    )

    expected_parts = []
    q_offset = kv_offset = position_offset = 0
    for q_len, kv_len in zip(q_lens, kv_lens):
        q_part = query[q_offset : q_offset + q_len].float()
        k_part = key[kv_offset : kv_offset + kv_len].float()
        v_part = value[kv_offset : kv_offset + kv_len].float()
        k_part = k_part.repeat_interleave(heads_q // heads_k, dim=1)
        v_part = v_part.repeat_interleave(heads_q // heads_k, dim=1)
        logits = torch.einsum("qhd,khd->hqk", q_part, k_part) / math.sqrt(head_dim)
        positions = q_positions[position_offset : position_offset + q_len]
        mask = torch.arange(kv_len, device="cuda").unsqueeze(0) <= positions.unsqueeze(
            1
        )
        logits.masked_fill_(~mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        expected_parts.append(torch.einsum("hqk,khd->qhd", probs, v_part))
        q_offset += q_len
        kv_offset += kv_len
        position_offset += q_len

    expected = torch.cat(expected_parts).to(output.dtype)
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-3)
