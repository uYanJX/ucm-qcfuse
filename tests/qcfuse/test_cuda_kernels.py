import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="QCFuse kernel tests require CUDA"
)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_tma_native_dispatch_cuda_preserves_bits_positions_and_reuse(batch):
    from types import SimpleNamespace
    from ucm.sparse.qcfuse.batch import cumulative_starts
    from ucm.sparse.qcfuse.blend_info import BlendStyle
    from ucm.sparse.qcfuse.qcfuse import QcFuse
    from ucm.sparse.qcfuse.ragged_attention import ragged_positions_attention_fwd
    from triton.runtime import _allocation

    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Descriptor attention requires Hopper")
    torch.manual_seed(1816 + batch)
    lengths = [5, 99, 139, 211][:batch]
    keys = [11, 193, 277, 419][:batch]
    query = torch.randn(sum(lengths) * 2, 64, 128, device="cuda", dtype=torch.bfloat16)[
        ::2
    ]
    key = torch.randn(sum(keys) * 2, 8, 128, device="cuda", dtype=torch.bfloat16)[::2]
    value = torch.randn_like(key)
    originals = [t.clone() for t in (query, key, value)]
    output = torch.empty_like(query)
    fuser = object.__new__(QcFuse)
    fuser._attention_backend = "tma"
    fuser._attention_backend_counts = {"native": 0, "flex": 0}
    fuser._flex_key_block = 64
    fuser._step = SimpleNamespace(phase=BlendStyle.DO_BLEND)
    layer = SimpleNamespace(impl=SimpleNamespace(sliding_window=(-1, -1)))
    consumer = torch.cuda.Stream()
    previous_allocator = _allocation._allocator.get()
    for reuse in range(2):
        positions = torch.cat(
            [
                torch.arange(n - length, n, dtype=torch.int32, device="cuda").flip(0)
                for length, n in zip(lengths, keys)
            ]
        )
        if reuse:
            positions = positions.flip(0).clone()
            # Keep each request's positions within its own full KV range.
            offset = 0
            for length, n in zip(lengths, keys):
                positions[offset : offset + length].remainder_(n)
                offset += length
        prepared = SimpleNamespace(q_lens=lengths, kv_lens=keys, q_positions=positions)
        expected = torch.empty_like(query)
        metadata = (
            cumulative_starts(lengths, query.device),
            torch.tensor(lengths, dtype=torch.int32, device=query.device),
            cumulative_starts(keys, query.device),
            torch.tensor(keys, dtype=torch.int32, device=query.device),
        )
        ragged_positions_attention_fwd(
            query, key, value, expected, *metadata, positions, max(lengths)
        )
        consumer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(consumer):
            fuser._run_sparse_attention(query, key, value, output, layer, prepared)
        assert _allocation._allocator.get() is previous_allocator
        consumer.synchronize()
        assert torch.equal(expected.view(torch.int16), output.view(torch.int16))
        assert all(
            torch.equal(old, new) for old, new in zip(originals, (query, key, value))
        )
    fuser._step.phase = BlendStyle.QCOMPUTE
    fuser._run_sparse_attention(query, key, value, output, layer, prepared)
    torch.cuda.synchronize()
    assert torch.equal(expected.view(torch.int16), output.view(torch.int16))
    assert fuser.attention_backend_stats()["layer_calls"] == {
        "native": 1,
        "flex": 0,
        "tma": 2,
    }


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_attention_layout_cuda_reuse_preserves_ragged_output(batch):
    from ucm.sparse.qcfuse.batch import QcFuseStepBatch, cumulative_starts
    from ucm.sparse.qcfuse.ragged_attention import ragged_positions_attention_fwd

    torch.manual_seed(1616 + batch)
    step = QcFuseStepBatch()
    keys = [11 + 3 * i for i in range(batch)]
    k = torch.randn(sum(keys), 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    originals = [t.clone() for t in (k, v)]
    for queries in (
        [5 + i for i in range(batch)],
        [3 + i for i in range(batch)],
        [5 + i for i in range(batch)],
    ):
        q = torch.randn(sum(queries), 64, 128, device="cuda", dtype=torch.bfloat16)
        positions = torch.cat(
            [
                torch.arange(n - length, n, device="cuda", dtype=torch.int32).flip(0)
                for length, n in zip(queries, keys)
            ]
        )
        expected, actual = torch.empty_like(q), torch.empty_like(q)
        baseline = (
            cumulative_starts(queries, q.device),
            torch.tensor(queries, dtype=torch.int32, device=q.device),
            cumulative_starts(keys, q.device),
            torch.tensor(keys, dtype=torch.int32, device=q.device),
        )
        ragged_positions_attention_fwd(
            q, k, v, expected, *baseline, positions, max(queries)
        )
        for _ in range(2):
            metadata = step.attention_layout(queries, keys, q.device)
            ragged_positions_attention_fwd(
                q, k, v, actual, *metadata, positions, max(queries)
            )
            torch.cuda.synchronize()
            assert torch.equal(expected.view(torch.int16), actual.view(torch.int16))
        assert all(torch.equal(a, b) for a, b in zip((k, v), originals))
    assert step.attention_layout_builds == 2 and step.attention_layout_hits == 4


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_attention_layout_cuda_consumer_survives_owner_release(batch):
    from ucm.sparse.qcfuse.batch import QcFuseStepBatch

    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    lengths = [3 + i for i in range(batch)]
    keys = [11 + i for i in range(batch)]
    step = QcFuseStepBatch()
    with torch.cuda.stream(producer):
        initial = step.attention_layout(lengths, keys, "cuda")
    with torch.cuda.stream(consumer):
        metadata = step.attention_layout(lengths, keys, "cuda")
        torch.cuda._sleep(10_000_000)
        copies = [t.clone() for t in metadata]
    del step, initial, metadata
    with torch.cuda.stream(producer):
        pressure = [
            torch.full((batch,), 999, dtype=torch.int32, device="cuda")
            for _ in range(128)
        ]
    consumer.synchronize()
    assert copies[0].tolist() == [sum(lengths[:i]) for i in range(batch)]
    assert copies[1].tolist() == lengths
    assert copies[2].tolist() == [sum(keys[:i]) for i in range(batch)]
    assert copies[3].tolist() == keys
    producer.synchronize()
    del pressure


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_native_breakable_decode_keeps_eager_metadata_and_cuda_output(
    batch, monkeypatch
):
    from types import SimpleNamespace
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import get_forward_context, set_forward_context
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture,
        eager_break_during_capture,
    )

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    torch.manual_seed(1610 + batch)
    first = torch.nn.Linear(64, 64, bias=False).cuda().to(torch.bfloat16)
    second = torch.nn.Linear(64, 32, bias=False).cuda().to(torch.bfloat16)
    inputs = torch.randn(batch, 64, device="cuda", dtype=torch.bfloat16)
    scratch = torch.empty_like(inputs)

    @eager_break_during_capture
    def attention_break(value, output):
        bound = get_forward_context().attn_metadata["toy"].max_seq_len
        output.copy_(value * float(bound))

    def forward():
        attention_break(first(inputs), scratch)
        return second(scratch)

    config = VllmConfig()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with set_current_vllm_config(config):
        with (
            torch.cuda.stream(capture_stream),
            set_forward_context(
                {"toy": SimpleNamespace(max_seq_len=3)},
                config,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            ),
        ):
            for _ in range(3):
                forward()
            torch.cuda.synchronize()
            capture = BreakableCUDAGraphCapture()
            with capture:
                actual = forward()
        torch.cuda.current_stream().wait_stream(capture_stream)
        assert capture._num_graphs == 2 and capture._num_eager_breaks == 1
        for bound in (3, 7, 3):
            with set_forward_context(
                {"toy": SimpleNamespace(max_seq_len=bound)},
                config,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            ):
                capture.replay()
            expected = second(first(inputs) * float(bound))
            torch.cuda.synchronize()
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("key_block", [64, 128])
def test_flex_attention_cuda_preserves_masks_and_repeated_request_ownership(
    batch_size, key_block
):
    from types import SimpleNamespace
    from ucm.sparse.qcfuse.blend_info import BlendStyle, QcFuseRequestState
    from ucm.sparse.qcfuse.qcfuse import QcFuse
    from ucm.sparse.qcfuse.flex_attention import flex_positions_attention_fwd

    torch.manual_seed(300 + batch_size)
    q_lens = [97, 139, 155, 211][:batch_size]
    kv_lens = [193, 277, 311, 419][:batch_size]
    query = torch.randn(sum(q_lens) * 2, 8, 128, device="cuda", dtype=torch.bfloat16)[
        ::2
    ]
    key = torch.randn(sum(kv_lens) * 2, 2, 128, device="cuda", dtype=torch.bfloat16)[
        ::2
    ]
    value = torch.randn_like(key)
    originals = [item.clone() for item in (query, key, value)]
    output = torch.empty_like(query)
    states = [
        QcFuseRequestState(positions=torch.randint(kv_len, (q_len,), device="cuda"))
        for q_len, kv_len in zip(q_lens, kv_lens)
    ]
    for reuse in range(2):
        if reuse:
            states = [
                QcFuseRequestState(positions=state.positions.flip(0))
                for state in states
            ]
        fuser = object.__new__(QcFuse)
        fuser._attention_backend = "flex"
        fuser._flex_key_block = key_block
        fuser._attention_backend_counts = {"native": 0, "flex": 0}
        fuser._step = SimpleNamespace(
            phase=BlendStyle.DO_BLEND,
            entries=[SimpleNamespace(state=state) for state in states],
        )
        prepared = SimpleNamespace(
            q_lens=q_lens,
            kv_lens=kv_lens,
            q_positions=torch.cat([state.positions for state in states]).to(
                torch.int32
            ),
        )
        layer = SimpleNamespace(impl=SimpleNamespace(sliding_window=(-1, -1)))
        fuser._run_sparse_attention(query, key, value, output, layer, prepared)
        assert fuser.attention_backend_stats()["layer_calls"] == {
            "native": 0,
            "flex": 1,
        }
        plans = [state.attn_meta["flex_position_plan"] for state in states]
        first = output.clone()
        flex_positions_attention_fwd(
            query, key, value, output, states, q_lens, kv_lens, 128**-0.5, key_block
        )
        assert torch.equal(first, output)
        assert all(
            state.attn_meta["flex_position_plan"] is plan
            for state, plan in zip(states, plans)
        )
        qo = ko = 0
        for state, q_len, kv_len in zip(states, q_lens, kv_lens):
            q = query[qo : qo + q_len].float().transpose(0, 1)
            k = key[ko : ko + kv_len].float().transpose(0, 1).repeat_interleave(4, 0)
            v = value[ko : ko + kv_len].float().transpose(0, 1).repeat_interleave(4, 0)
            logits = (q @ k.transpose(-1, -2)) * (128**-0.5)
            mask = torch.arange(kv_len, device="cuda")[None] <= state.positions[:, None]
            expected = (
                logits.masked_fill(~mask[None], -float("inf")).softmax(-1) @ v
            ).transpose(0, 1)
            torch.testing.assert_close(
                output[qo : qo + q_len].float(), expected, atol=0.015625, rtol=0.03
            )
            qo += q_len
            ko += kv_len
        assert all(
            torch.equal(old, current)
            for old, current in zip(originals, (query, key, value))
        )
        fuser._step.phase = BlendStyle.QCOMPUTE
        fuser._run_sparse_attention(query, key, value, output, layer, prepared)
        assert fuser.attention_backend_stats()["layer_calls"] == {
            "native": 1,
            "flex": 1,
        }


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("ratio", [0.4, 0.5])
def test_selection_workspace_keeps_native_rope_and_selected_indices_bit_exact(
    batch, ratio, monkeypatch
):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from ucm.sparse.qcfuse.blend_info import AttParams, QcFuseRequestState
    from ucm.sparse.qcfuse import indice_select

    with set_current_vllm_config(VllmConfig()):
        rotary = get_rope(head_size=128, max_position=2048, dtype=torch.bfloat16).cuda()
    chunks, lengths, query_lengths = [0], [], []
    positions = []
    for index in range(batch):
        document, query = 257 + index * 31, 17 + index
        total = 8 + document + query
        start = chunks[-1]
        chunks.extend([start + 8, start + 8 + document, start + total])
        positions.append(torch.arange(total, device="cuda"))
        lengths.append(total)
        query_lengths.append(query)
    keys = [
        torch.randn(sum(lengths), 1024, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    queries = [
        torch.randn(sum(query_lengths), 8192, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    query_keys = [
        torch.randn(sum(query_lengths), 1024, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    originals = [value.clone() for value in keys + queries + query_keys]
    state = QcFuseRequestState(
        ratio=ratio,
        att_params=AttParams(64, 8, 128, 3),
        chunk_loc_list=torch.tensor(chunks, device="cuda"),
        req_len_list=torch.tensor([3] * batch, device="cuda"),
        critical_layers=[0, 1, 2],
        query_k_buffer=query_keys,
        q_lens=query_lengths,
        q_offsets=[0] * batch,
        query_k_lens=query_lengths,
        rotary_emb=rotary,
    )
    positions = torch.cat(positions)
    actual, actual_lengths = indice_select.IndiceSelector.select(
        state, keys, queries, positions
    )

    def original_assembly(prefix, query):
        return torch.cat([torch.stack(prefix), torch.stack(query)], dim=1)

    def original_rotation(
        rotary_emb,
        token_positions,
        tensor,
        num_layers,
        num_heads,
        head_dim,
        *,
        is_query,
    ):
        flat = tensor.reshape(-1, num_heads, head_dim).contiguous()
        scratch = torch.empty_like(flat)
        if is_query:
            rotated, _ = rotary_emb(token_positions.repeat(num_layers), flat, scratch)
        else:
            _, rotated = rotary_emb(token_positions.repeat(num_layers), scratch, flat)
        return rotated.reshape(num_layers, -1, num_heads, head_dim)

    # Compare each actual native RoPE result, not only a possibly stable top-k.
    for heads, is_query in ((64, True), (8, False)):
        raw = torch.randn(
            3, 19 + batch, heads, 128, device="cuda", dtype=torch.bfloat16
        )
        pos = torch.arange(19 + batch, device="cuda").flip(0)
        expected = original_rotation(
            rotary, pos, raw.clone(), 3, heads, 128, is_query=is_query
        )
        rotated = indice_select.IndiceSelector._rotate_stacked(
            rotary, pos, raw.clone(), 3, heads, 128, is_query=is_query
        )
        torch.testing.assert_close(rotated, expected, rtol=0, atol=0)
    with monkeypatch.context() as reference:
        reference.setattr(indice_select, "_assemble_selection_keys", original_assembly)
        reference.setattr(
            indice_select.IndiceSelector,
            "_rotate_stacked",
            staticmethod(original_rotation),
        )
        expected, expected_lengths = indice_select.IndiceSelector.select(
            state, keys, queries, positions
        )
    assert torch.equal(actual, expected) and torch.equal(
        actual_lengths, expected_lengths
    )
    for value, original in zip(keys + queries + query_keys, originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_compact_projection_values_preserve_normalized_qkv(batch):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm

    head_dim, q_heads, kv_heads = 128, 8, 1
    q_size, kv_size = head_dim * q_heads, head_dim * kv_heads
    fused = torch.randn(
        batch * 17, q_size + 2 * kv_size, device="cuda", dtype=torch.bfloat16
    )
    with set_current_vllm_config(VllmConfig()):
        q_norm = RMSNorm(head_dim, eps=1e-6).cuda().to(torch.bfloat16)
        k_norm = RMSNorm(head_dim, eps=1e-6).cuda().to(torch.bfloat16)
    q, k, v = fused.split([q_size, kv_size, kv_size], dim=-1)
    expected_q = q_norm(q.view(-1, q_heads, head_dim)).view(q.shape)
    expected_k = k_norm(k.view(-1, kv_heads, head_dim)).view(k.shape)
    expected_v = v.clone()
    compact_v = v.clone()
    compact_q = q_norm(q.view(-1, q_heads, head_dim)).view(q.shape)
    compact_k = k_norm(k.view(-1, kv_heads, head_dim)).view(k.shape)
    original_pointer = fused.untyped_storage().data_ptr()
    assert all(
        t.untyped_storage().data_ptr() != original_pointer
        for t in (compact_q, compact_k, compact_v)
    )
    assert (
        compact_v.untyped_storage().nbytes()
        == compact_v.numel() * compact_v.element_size()
    )
    for actual, expected in zip(
        (compact_q, compact_k, compact_v), (expected_q, expected_k, expected_v)
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.fixture(autouse=True)
def _delete_generated_test_kv(request):
    yield
    directory = request.node.funcargs.get("tmp_path")
    if directory is not None:
        for path in directory.rglob("*.bin"):
            if not path.is_symlink():
                path.unlink()


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_compact_native_silu_and_full_gemm_are_bit_exact(batch):
    from types import SimpleNamespace
    from vllm import _custom_ops  # noqa: F401 - registers native CUDA operators

    operation = torch.ops._C.silu_and_mul
    from ucm.integration.vllm.latest_sparse import (
        _compact_silu_and_mul,
        _compact_qwen_mlp_forward,
    )

    # Actual Qwen3-32B FFN dimensions, uneven tile ends and several requests.
    tokens, hidden, intermediate = batch * 513, 5120, 25600
    gate = torch.randn(tokens, intermediate * 2, device="cuda", dtype=torch.bfloat16)
    expected_activation = gate.new_empty((tokens, intermediate))
    operation(expected_activation, gate)
    actual_activation = _compact_silu_and_mul(gate, operation)
    assert (
        actual_activation.untyped_storage().data_ptr()
        == gate.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(actual_activation, expected_activation, rtol=0, atol=0)
    del gate, actual_activation, expected_activation

    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    up_weight = (
        torch.randn(intermediate * 2, hidden, device="cuda", dtype=torch.bfloat16)
        * 0.01
    )
    down_weight = (
        torch.randn(hidden, intermediate, device="cuda", dtype=torch.bfloat16) * 0.01
    )
    up = torch.nn.functional.linear(x, up_weight)
    activated = up.new_empty((tokens, intermediate))
    operation(activated, up)
    expected = torch.nn.functional.linear(activated, down_weight)
    del up, activated
    mlp = SimpleNamespace(
        gate_up_proj=lambda t: (torch.nn.functional.linear(t, up_weight), None),
        act_fn=SimpleNamespace(op=operation),
        down_proj=SimpleNamespace(weight=down_weight),
    )
    actual = _compact_qwen_mlp_forward(mlp, x)
    assert actual is x
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_gpu_ffn_row_tiles_keep_all_rows_and_native_output(batch):
    from types import SimpleNamespace
    from vllm import _custom_ops  # noqa: F401 - registers native CUDA operators

    operation = torch.ops._C.silu_and_mul
    from ucm.integration.vllm.latest_sparse import _compact_qwen_mlp_forward

    # A raw 8192-row split would leave one row: balanced tiles avoid this
    # boundary for the actual Qwen3-32B BF16 projection dimensions.
    tokens, hidden, intermediate = batch * 8193, 5120, 25600
    source = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    up = (
        torch.randn(intermediate * 2, hidden, device="cuda", dtype=torch.bfloat16)
        * 0.01
    )
    down = torch.randn(hidden, intermediate, device="cuda", dtype=torch.bfloat16) * 0.01
    rows_seen = []

    def project(value):
        rows_seen.append(value.shape[0])
        return torch.nn.functional.linear(value, up), None

    mlp = SimpleNamespace(
        gate_up_proj=project,
        act_fn=SimpleNamespace(op=operation),
        down_proj=SimpleNamespace(weight=down),
    )
    expected = _compact_qwen_mlp_forward(mlp, source.clone())
    rows_seen.clear()
    work = source.clone()
    observed_tiles = []
    actual = _compact_qwen_mlp_forward(
        mlp, work, projection_tile_tokens=8192, on_projection_tile=observed_tiles.append
    )
    assert actual is work and sum(rows_seen) == tokens
    assert observed_tiles == rows_seen
    assert max(rows_seen) <= 8192 and min(rows_seen) >= 4096
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_transfer_event_orders_a_different_consumer_stream():
    from ucm.store.qcfuse.transfer import _attach_ready_event, wait_on_current_stream

    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    tensors = torch.empty(1024, device="cuda"), torch.empty(1024, device="cuda")
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        torch.cuda._sleep(10_000_000)
        tensors[0].fill_(17)
        tensors[1].fill_(23)
        ready = torch.cuda.Event()
        ready.record(producer)
    _attach_ready_event(tensors, ready)
    with torch.cuda.stream(consumer):
        wait_on_current_stream(tensors)
        copied = tensors[0] + tensors[1]
    consumer.synchronize()
    torch.testing.assert_close(copied, torch.full_like(copied, 40), rtol=0, atol=0)


def test_native_vllm_rope_reuse_preserves_key_bits_for_mixed_lengths():
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from ucm.sparse.qcfuse.blender import _rotate_key

    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        rotary = get_rope(head_size=32, max_position=128, dtype=torch.bfloat16).cuda()
        pointers = []
        for length in (3, 7, 4, 5):
            key = torch.randn(length, 64, device="cuda", dtype=torch.bfloat16)
            positions = torch.arange(length, device="cuda")
            _, expected = rotary(positions, torch.empty_like(key), key.clone())
            actual = _rotate_key(rotary, positions, key.clone())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            pointers.append(rotary._cached_fake_q.data_ptr())
        assert pointers[1] == pointers[2] == pointers[3]


@pytest.mark.parametrize("mode", ["ssd", "cpu", "gpu"])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_store_reports_exact_h2d_payload_across_reused_request_pools(
    tmp_path, monkeypatch, mode, batch
):
    from ucm.store.qcfuse import transfer, packed_format as pf
    from ucm.store.qcfuse.qcfuse_store import QcFusePrefetchTask, UcmQcFuseStore

    monkeypatch.setenv("QCFUSE_IO_GROUP_LAYERS", "2")
    monkeypatch.setenv("QCFUSE_TRANSFER_TIMING", "0")
    key = torch.full((257, 1, 32), 17, dtype=torch.bfloat16)
    value = torch.full_like(key, 23)
    path = tmp_path / "kv_packed.bin"
    pf.save_packed_kv(str(tmp_path), {i: (key + i, value) for i in range(2)}, 2)
    meta = pf.load_packed_meta(str(tmp_path))
    store = UcmQcFuseStore({"storage_mode": mode})
    payload_per_layer = (key.numel() + value.numel()) * key.element_size()
    operations = [
        (kind, "", layer, str(tmp_path))
        for kind, layer in (("query", 0), ("chunk", 0), ("chunk", 1))
    ]
    if mode != "ssd":
        transfer.preload_resident_file(str(path), mode)
    expected = 0
    try:
        for reuse in range(2):
            for item in range(batch):
                request_id = f"payload-{reuse}-{item}"

                def load(operation, request):
                    kind, _, layer, directory = operation
                    return transfer.load_packed_layer(
                        directory,
                        meta,
                        layer,
                        "cuda:0",
                        kind=f"request:{request}:{kind}",
                        wait=False,
                    )

                task = QcFusePrefetchTask(request_id, operations, load, depth=2)
                store._tasks[request_id] = task
                for operation in operations:
                    actual = transfer.wait_on_current_stream(task.take(operation))
                    assert torch.equal(actual[0].cpu(), key + operation[2])
                    assert torch.equal(actual[1].cpu(), value)
                    actual[0].fill_(-99)
                store.finish_request(request_id)
                if mode != "gpu":
                    expected += len(operations) * payload_per_layer
                assert store.metrics()["h2d_bytes"] == expected
                store.finish_request(request_id)
                assert store.metrics()["h2d_bytes"] == expected
                assert not any(
                    kind.startswith(f"request:{request_id}:")
                    for kind, _ in transfer._POOLS
                )
    finally:
        torch.cuda.synchronize()
        store.close()


@pytest.mark.parametrize("mode", ["ssd", "cpu", "gpu"])
def test_actual_cache_load_is_ready_on_another_thread_stream(tmp_path, mode):
    from concurrent.futures import ThreadPoolExecutor
    from ucm.store.qcfuse import transfer, packed_format as pf

    key = torch.full((8192, 1, 32), 17, dtype=torch.bfloat16)
    value = torch.full_like(key, 23)
    pf.save_packed_kv(str(tmp_path), {0: (key, value)}, 1)
    meta = pf.load_packed_meta(str(tmp_path))
    kind = f"cuda_test:{mode}"
    if mode != "ssd":
        transfer.preload_resident_file(str(tmp_path / "kv_packed.bin"), mode)
    pool = transfer.get_transfer_pool(kind, "cuda:0")
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    try:

        def load():
            with torch.cuda.stream(producer):
                torch.cuda._sleep(10_000_000)
                delay = torch.cuda.Event()
                delay.record(producer)
                pool.stream.wait_event(delay)
                return transfer.load_packed_layer(
                    str(tmp_path), meta, 0, "cuda:0", kind=kind, wait=False
                )

        with ThreadPoolExecutor(max_workers=1) as executor:
            tensors = executor.submit(load).result()
        with torch.cuda.stream(consumer):
            transfer.wait_on_current_stream(tensors)
            result = tensors[0] + tensors[1]
        consumer.synchronize()
        torch.testing.assert_close(result.cpu(), key + value, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        transfer.release_transfer_pools(kind)
        transfer.release_resident_files()


@pytest.mark.parametrize("registered", [False, True])
def test_resident_pin_failure_keeps_bytes_off_ssd(tmp_path, monkeypatch, registered):
    from ucm.store.qcfuse import transfer, packed_format as pf

    key = torch.full((4096, 1, 32), 17, dtype=torch.bfloat16)
    value = torch.full_like(key, 23)
    pf.save_packed_kv(
        str(tmp_path), {layer: (key + layer, value) for layer in range(3)}, 3
    )
    meta = pf.load_packed_meta(str(tmp_path))
    original_empty = torch.empty
    failed_once = False

    def fail_first_pin(*args, **kwargs):
        nonlocal failed_once
        if kwargs.get("pin_memory") and not failed_once:
            failed_once = True
            raise RuntimeError("Forced whole-resident pin allocation failure")
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fail_first_pin)
    if not registered:
        monkeypatch.setattr(transfer, "_try_register_host_memory", lambda tensor: False)
    path = tmp_path / "kv_packed.bin"
    kind = f"resident_fallback:{registered}"
    try:
        transfer.preload_resident_file(str(path), "cpu")
        resident = transfer.resident_reader(str(path))
        assert failed_once and resident is not None
        if registered and not resident.registered:
            pytest.skip(
                "Host registration unavailable; pageable fallback tested separately"
            )
        assert resident.registered == registered
        assert resident.pinned == registered
        # Unlink alone leaves existing fds and mappings readable. Change the
        # backing bytes first so a file-backed fallback cannot silently pass.
        with path.open("r+b", buffering=0) as stream:
            stream.write(bytes(path.stat().st_size))
        path.unlink()
        consumer = torch.cuda.Stream()
        for layer in range(3):
            tensors = transfer.load_packed_layer(
                str(tmp_path), meta, layer, "cuda:0", kind=kind, wait=False
            )
            with torch.cuda.stream(consumer):
                transfer.wait_on_current_stream(tensors)
                actual_key = tensors[0].clone()
                actual_value = tensors[1].clone()
            consumer.synchronize()
            assert torch.equal(
                actual_key.cpu().view(torch.uint8), (key + layer).view(torch.uint8)
            )
            assert torch.equal(
                actual_value.cpu().view(torch.uint8), value.view(torch.uint8)
            )
        metrics = transfer.get_transfer_pool(kind, "cuda:0").drain_metrics()
        assert metrics["ssd_read_bytes"] == 0
        if registered:
            assert metrics["resident_cpu_direct_layers"] == 3
        else:
            assert (
                metrics["resident_read_bytes"] == key.numel() * key.element_size() * 6
            )
    finally:
        torch.cuda.synchronize()
        transfer.release_transfer_pools(kind)
        transfer.release_resident_files()


def test_failed_host_registration_does_not_poison_next_cuda_kernel():
    from types import SimpleNamespace
    from ucm.store.qcfuse.transfer import _try_register_host_memory

    torch.cuda.init()
    invalid_host_pointer = SimpleNamespace(numel=lambda: 1, data_ptr=lambda: 0)
    assert _try_register_host_memory(invalid_host_pointer) is False
    result = torch.empty(8, device="cuda")
    result.fill_(7)
    torch.cuda.synchronize()
    torch.testing.assert_close(result, torch.full_like(result, 7), rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_failed_pinned_staging_falls_back_with_exact_cache_bytes(
    tmp_path, monkeypatch, batch
):
    from ucm.store.qcfuse import transfer, packed_format as pf

    key = torch.full((512, 1, 32), 17, dtype=torch.bfloat16)
    value = torch.full_like(key, 23)
    pf.save_packed_kv(str(tmp_path), {i: (key + i, value) for i in range(3)}, 3)
    meta = pf.load_packed_meta(str(tmp_path))
    original_empty = torch.empty

    def fail_pinning(*args, **kwargs):
        if kwargs.get("pin_memory"):
            # Leave a real handled CUDA last-error, then inject the observed
            # allocator failure without exhausting machine-wide host memory.
            status = torch.cuda.cudart().cudaHostRegister(0, 1, 0)
            assert int(status[0] if isinstance(status, (tuple, list)) else status) != 0
            raise RuntimeError("CUDA error: invalid argument from cuMemHostAlloc")
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fail_pinning)
    kind = f"pageable_staging_test:{batch}"
    consumer = torch.cuda.Stream()
    try:
        for _ in range(batch):
            for layer in range(3):
                with torch.cuda.stream(consumer):
                    tensors = transfer.load_packed_layer(
                        str(tmp_path), meta, layer, "cuda:0", kind=kind
                    )
                    copied = (tensors[0].clone(), tensors[1].clone())
                consumer.synchronize()
                torch.testing.assert_close(copied[0].cpu(), key + layer, rtol=0, atol=0)
                torch.testing.assert_close(copied[1].cpu(), value, rtol=0, atol=0)
        pool = transfer.get_transfer_pool(kind, "cuda:0")
        assert pool.drain_metrics()["pinned_allocation_fallbacks"] > 0
        assert pool.last_operation_metrics()["pageable_staging"] == 1.0
        # A later ordinary CUDA operation must remain healthy.
        healthy = torch.zeros(8, device="cuda") + 7
        torch.cuda.synchronize()
        torch.testing.assert_close(healthy, torch.full_like(healthy, 7), rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        transfer.release_transfer_pools(kind)


def test_cpu_offline_stash_preserves_gpu_cache_bytes(tmp_path):
    from ucm.sparse.qcfuse.blend_info import BlendStyle, QcFuseRequestState
    from ucm.store.qcfuse.packed_format import save_packed_kv

    fused = torch.arange(120, device="cuda", dtype=torch.bfloat16).reshape(3, 40)
    _, key, value = fused.split([32, 4, 4], dim=-1)
    gpu = QcFuseRequestState(blend_style=BlendStyle.KVCOMPUTE)
    cpu = QcFuseRequestState(blend_style=BlendStyle.KVCOMPUTE, offline_kv_device="cpu")
    gpu.put_kv(key, value, 0)
    cpu.put_kv(key, value, 0)
    torch.testing.assert_close(cpu.get_kv(0)[0], gpu.get_kv(0)[0].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(cpu.get_kv(0)[1], gpu.get_kv(0)[1].cpu(), rtol=0, atol=0)
    indices = torch.tensor([0, 2], device="cuda")
    save_packed_kv(str(tmp_path / "gpu"), {0: gpu.get_kv(0)}, 1, token_indices=indices)
    save_packed_kv(str(tmp_path / "cpu"), {0: cpu.get_kv(0)}, 1, token_indices=indices)
    assert (tmp_path / "gpu/kv_packed.bin").read_bytes() == (
        tmp_path / "cpu/kv_packed.bin"
    ).read_bytes()
    assert (tmp_path / "gpu/kv_packed_meta.json").read_bytes() == (
        tmp_path / "cpu/kv_packed_meta.json"
    ).read_bytes()


@pytest.mark.parametrize("group", [1, 2, 4])
def test_grouped_pinned_reads_preserve_layers_when_ring_wraps(
    tmp_path, monkeypatch, group
):
    import math
    from ucm.store.qcfuse import transfer, packed_format as pf

    monkeypatch.setenv("QCFUSE_IO_GROUP_LAYERS", str(group))
    monkeypatch.setenv("QCFUSE_TRANSFER_RING_DEPTH", "2")
    monkeypatch.setenv("QCFUSE_TRANSFER_RING_MAX_DEPTH", "2")
    monkeypatch.setenv("QCFUSE_TRANSFER_TIMING", "1")
    layers = {
        layer: (
            torch.full((128, 2, 32), layer, dtype=torch.bfloat16),
            torch.full((128, 2, 32), layer + 100, dtype=torch.bfloat16),
        )
        for layer in range(7)
    }
    pf.save_packed_kv(str(tmp_path), layers, 7)
    meta = pf.load_packed_meta(str(tmp_path))
    kind = f"grouped_test:{group}"
    pool = transfer.get_transfer_pool(kind, "cuda:0")
    try:
        with torch.cuda.stream(pool.stream):
            torch.cuda._sleep(10_000_000)
        loaded = [
            transfer.load_packed_layer(
                str(tmp_path), meta, layer, "cuda:0", kind=kind, wait=False
            )
            for layer in range(7)
        ]
        consumer = torch.cuda.Stream()
        with torch.cuda.stream(consumer):
            copies = [
                (key.clone(), value.clone())
                for key, value in (
                    transfer.wait_on_current_stream(pair) for pair in loaded
                )
            ]
        consumer.synchronize()
        for layer, (key, value) in enumerate(copies):
            torch.testing.assert_close(key.cpu(), layers[layer][0], rtol=0, atol=0)
            torch.testing.assert_close(value.cpu(), layers[layer][1], rtol=0, atol=0)
        metrics = pool.drain_metrics()
        assert metrics["read_blocks"] == math.ceil(7 / group)
        assert metrics["completed_copies"] == 7
        assert metrics["ssd_read_bytes"] == (tmp_path / "kv_packed.bin").stat().st_size
    finally:
        torch.cuda.synchronize()
        transfer.release_transfer_pools(kind)


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


def test_failed_resident_gpu_batch_releases_partial_allocation_and_can_retry(
    tmp_path, monkeypatch
):
    from ucm.store.qcfuse import transfer

    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(bytes(range(32)))
    second.write_bytes(bytes(range(64)))
    transfer.release_resident_files()
    torch.cuda.synchronize()
    initial_allocated = torch.cuda.memory_allocated()
    original_to = torch.Tensor.to

    def fail_second_copy(tensor, *args, **kwargs):
        if (
            tensor.device.type == "cpu"
            and tensor.numel() == 64
            and kwargs.get("device") == "cuda:0"
        ):
            raise RuntimeError("Injected CUDA out of memory during resident preload")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fail_second_copy)
    try:
        with pytest.raises(RuntimeError, match="Injected CUDA out of memory"):
            transfer.prepare_resident_batch(None, [str(first), str(second)], "gpu")
        torch.cuda.synchronize()
        assert transfer.resident_metrics()["resident_files"] == 0
        assert transfer.resident_metrics()["resident_gpu_bytes"] == 0
        assert torch.cuda.memory_allocated() == initial_allocated

        retried = transfer.prepare_resident_batch(None, [str(first)], "gpu")
        assert retried["resident_files"] == 1
        assert retried["resident_gpu_bytes"] == 32
        assert not retried["resident_batch_reused"]
        reader = transfer.resident_reader(str(first))
        assert bytes(reader.tensor.cpu().tolist()) == first.read_bytes()
    finally:
        transfer.release_resident_files()


def test_gpu_headroom_failure_rolls_back_before_online_submission(
    tmp_path, monkeypatch
):
    from ucm.store.qcfuse import transfer

    path = tmp_path / "headroom.bin"
    path.write_bytes(bytes(range(32)))
    transfer.release_resident_files()
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    original_empty = torch.empty
    probe_bytes = 4 * 1024**2

    def fail_probe(*args, **kwargs):
        if args == (probe_bytes,) and kwargs.get("dtype") == torch.uint8:
            raise RuntimeError("Injected CUDA out of memory probing online headroom")
        return original_empty(*args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", fail_probe)
            with pytest.raises(RuntimeError, match="probing online headroom"):
                transfer.prepare_resident_batch(None, [str(path)], "gpu", probe_bytes)
        torch.cuda.synchronize()
        assert transfer.resident_metrics()["resident_files"] == 0
        assert torch.cuda.memory_allocated() == initial
        result = transfer.prepare_resident_batch(None, [str(path)], "gpu", probe_bytes)
        assert result["online_headroom_bytes"] == probe_bytes
        assert result["resident_files"] == 1
        result = transfer.prepare_resident_batch(None, [str(path)], "gpu", probe_bytes)
        assert result["resident_batch_reused"]
        assert torch.equal(
            transfer.resident_reader(str(path)).tensor.cpu(),
            torch.arange(32, dtype=torch.uint8),
        )
    finally:
        transfer.release_resident_files()


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_gpu_resident_lazy_load_keeps_source_bytes_immutable(tmp_path, batch):
    from types import SimpleNamespace
    from ucm.store.qcfuse import transfer, packed_format as pf
    from ucm.store.qcfuse.qcfuse_store import UcmQcFuseStore

    key = torch.full((512, 1, 32), 17, dtype=torch.bfloat16)
    value = torch.full_like(key, 23)
    pf.save_packed_kv(str(tmp_path), {i: (key, value) for i in range(3)}, 3)
    path = str(tmp_path / "kv_packed.bin")
    transfer.prepare_resident_batch(None, [path], "gpu")
    store = UcmQcFuseStore({"storage_mode": "gpu", "resident_preload_policy": "batch"})
    original = transfer.resident_reader(path).tensor.clone()
    consumer = torch.cuda.Stream()
    try:
        store.begin_requests([SimpleNamespace(request_id=str(i)) for i in range(batch)])
        assert not store._tasks
        for _request in range(batch):
            with torch.cuda.stream(consumer):
                loaded_k, loaded_v = store.load_chunk_layer(str(tmp_path), 1)
                loaded_k.fill_(99)
                loaded_v.zero_()
                fresh_k, fresh_v = store.load_chunk_layer(str(tmp_path), 1)
            consumer.synchronize()
            torch.testing.assert_close(fresh_k.cpu(), key, rtol=0, atol=0)
            torch.testing.assert_close(fresh_v.cpu(), value, rtol=0, atol=0)
            assert torch.equal(transfer.resident_reader(path).tensor, original)
    finally:
        torch.cuda.synchronize()
        store.close()
        transfer.release_transfer_pools("")
        transfer.release_resident_files()


@pytest.mark.parametrize("skip", [0, 2])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_bounded_worker_profile_keeps_cuda_outputs_and_removes_decode_hook(
    tmp_path, batch, skip
):
    import json
    from types import SimpleNamespace
    from ucm.integration.vllm.qcfuse_worker import QcFuseWorkerExtension

    torch.manual_seed(91 + batch)
    model = torch.nn.Linear(64, 32, bias=False).cuda().to(torch.bfloat16)
    value = torch.randn(batch * 7, 64, device="cuda", dtype=torch.bfloat16)
    expected = model(value)
    worker = QcFuseWorkerExtension()
    worker.model_runner = SimpleNamespace(model=model)
    path = tmp_path / "execution.json"
    worker.qcfuse_start_execution_profile(str(path), max_forwards=2, skip_forwards=skip)
    for _ in range(4 + skip):
        torch.testing.assert_close(model(value), expected, rtol=0, atol=0)
    row = worker.qcfuse_stop_execution_profile()
    assert row["trace_written"] and row["forward_calls"] == 2
    assert row["skipped_forward_calls"] == skip
    assert not model._forward_pre_hooks
    trace = json.loads(path.read_text())
    assert any(event.get("cat") == "kernel" for event in trace["traceEvents"])


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_native_graph_boundary_reuse_preserves_cuda_outputs(batch):
    from types import SimpleNamespace
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from ucm.integration.vllm.qcfuse_graph import install_v2_graph_hooks
    from ucm.integration.vllm.qcfuse_worker import QcFuseWorkerExtension

    torch.manual_seed(501 + batch)
    model = torch.nn.Linear(64, 32, bias=False).cuda().to(torch.bfloat16)
    source = torch.randn(batch, 64, device="cuda", dtype=torch.bfloat16)
    static = source.clone()
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            model(static)
    torch.cuda.current_stream().wait_stream(warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = model(static)
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=batch,
        num_reqs=batch,
        uniform_token_count=1,
        max_query_len=1,
    )

    def replay(_desc):
        graph.replay()
        return output

    manager = SimpleNamespace(
        graphs={desc: graph},
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        dispatch=lambda *args, **kwargs: desc,
        run_fullgraph=replay,
    )
    sparse = SimpleNamespace(prepared="stale-prefill")
    builds = []

    def build(scheduler, states, input_batch, metadata):
        sparse.prepared = None
        builds.append(scheduler)

    sparse.build_sparse_meta = build
    runner = SimpleNamespace(
        model=model,
        cudagraph_manager=manager,
        kv_connector=SimpleNamespace(pre_forward=lambda **kwargs: None),
        req_states=None,
        _ucm_latest_input_batch=None,
        _ucm_latest_sparse_meta_built_for=None,
    )
    install_v2_graph_hooks(runner, lambda: True, lambda: sparse)
    worker = QcFuseWorkerExtension()
    worker.model_runner = runner
    worker.qcfuse_start_execution_mode_observer()
    try:
        for offset in (0.0, 0.5, -0.25):
            scheduler = SimpleNamespace()
            sparse.prepared = "stale-prefill"
            value = source + offset
            saved = value.clone()
            static.copy_(value)
            runner.kv_connector.pre_forward(
                scheduler_output=scheduler, attn_metadata={}
            )
            assert sparse.prepared is None
            actual = manager.run_fullgraph(desc)
            torch.testing.assert_close(actual, model(value), rtol=0, atol=0)
            torch.testing.assert_close(value, saved, rtol=0, atol=0)
        assert len(builds) == 3
        assert worker._qcfuse_execution_modes["full_graph_replays"] == 3
    finally:
        worker._qcfuse_execution_mode_hook.remove()
        graph.reset()
