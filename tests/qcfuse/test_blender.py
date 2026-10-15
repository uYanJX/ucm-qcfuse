import pytest
import torch

from ucm.sparse.qcfuse import blender
from ucm.sparse.qcfuse import digest_index
from ucm.sparse.qcfuse.blend_info import AttParams, BlendStyle, QcFuseRequestState


def _rotary(positions, query, key):
    shape = (positions.numel(),) + (1,) * (query.ndim - 1)
    delta = positions.to(query.dtype).view(shape)
    return query + delta, key + delta


def test_rope_buffer_grows_then_reuses_capacity_for_mixed_lengths():
    class Rotary:
        def __call__(self, positions, query, key):
            assert query.shape == key.shape
            return _rotary(positions, query, key)

    rotary = Rotary()
    pointers = []
    for length in (3, 7, 4, 5):
        key = torch.zeros(length, 4)
        positions = torch.arange(length)
        rotated = blender._rotate_key(rotary, positions, key)
        torch.testing.assert_close(
            rotated, positions[:, None].expand_as(key).float(), rtol=0, atol=0
        )
        pointers.append(rotary._cached_fake_q.data_ptr())
    assert pointers[1] == pointers[2] == pointers[3]
    assert rotary._cached_fake_q.shape == (7, 4)


class _QueryStore:
    def load_query_layer(self, sample_dir, section, layer_id):
        assert section == "digest"
        return torch.zeros(2, 4), torch.ones(2, 4)

    def load_query_meta(self, sample_dir):
        return {
            "metadata": {
                "total_tokens": 6,
                "context_positions_by_layer": [[1, 4]],
            }
        }


def test_qcompute_rotates_raw_context_at_original_positions():
    state = QcFuseRequestState(
        blend_style=BlendStyle.QCOMPUTE,
        is_contextblend=True,
        sample_dir_query="/query/a",
    )
    query = torch.zeros(2, 4)
    key = torch.zeros(2, 4)
    value = torch.full((2, 4), 2.0)

    q_out, k_out, v_out = blender.blend_layer(
        state,
        0,
        query,
        key,
        value,
        torch.arange(2),
        _rotary,
        store=_QueryStore(),
        sample_dir_query="/query/a",
    )

    torch.testing.assert_close(q_out[:, 0], torch.tensor([6.0, 7.0]))
    torch.testing.assert_close(k_out[:, 0], torch.tensor([1.0, 4.0, 6.0, 7.0]))
    assert state.attention_positions.tolist() == [2, 3]
    assert v_out.shape == (4, 4)


class _ChunkStore:
    def load_chunk_layer(self, sample_dir, layer_id):
        return torch.zeros(5, 4), torch.zeros(5, 4)


def test_later_blend_layer_inserts_raw_key_before_rope():
    state = QcFuseRequestState(
        blend_style=BlendStyle.DO_BLEND,
        att_params=AttParams(num_layers=4),
        start=0,
        blend_top_indices=torch.tensor([1, 3]),
        positions=torch.tensor([1, 3]),
        sample_dir_chunk="/chunk/a",
    )
    query = torch.zeros(2, 4)
    key = torch.full((2, 4), 10.0)
    value = torch.full((2, 4), 20.0)

    q_out, k_out, v_out = blender.blend_layer(
        state,
        1,
        query,
        key,
        value,
        state.positions,
        _rotary,
        store=_ChunkStore(),
        sample_dir_chunk="/chunk/a",
    )

    torch.testing.assert_close(q_out[:, 0], torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(k_out[:, 0], torch.tensor([0.0, 11.0, 2.0, 13.0, 4.0]))
    torch.testing.assert_close(v_out[[1, 3], 0], torch.tensor([20.0, 20.0]))


def test_kvcompute_positions_restart_for_each_chunk():
    state = QcFuseRequestState(
        chunk_lens=torch.tensor([2, 3, 1]),
        chunk_loc_list=torch.tensor([0, 2, 5, 6]),
    )

    positions = digest_index.ensure_forward_positions(state)

    assert positions.tolist() == [0, 1, 0, 1, 2, 0]


def test_offline_kv_does_not_retain_fused_qkv_backing_storage():
    fused = torch.arange(120, dtype=torch.float32).reshape(3, 40)
    _, key, value = fused.split([32, 4, 4], dim=-1)
    original_value = value.clone()
    state = QcFuseRequestState()
    state.put_kv(key, value, 0)
    saved_key, saved_value = state.get_kv(0)
    assert saved_value.untyped_storage().nbytes() == saved_value.numel() * 4
    fused.zero_()
    torch.testing.assert_close(saved_value, original_value, rtol=0, atol=0)
    assert saved_key.untyped_storage().nbytes() == saved_key.numel() * 4


def test_offline_cpu_stash_is_an_independent_exact_copy():
    fused = torch.arange(120, dtype=torch.float32).reshape(3, 40)
    _, key, value = fused.split([32, 4, 4], dim=-1)
    state = QcFuseRequestState(
        blend_style=BlendStyle.KVCOMPUTE, offline_kv_device="cpu"
    )
    state.put_kv(key, value, 0)
    saved_key, saved_value = state.get_kv(0)
    torch.testing.assert_close(saved_key, key, rtol=0, atol=0)
    torch.testing.assert_close(saved_value, value, rtol=0, atol=0)
    assert saved_key.data_ptr() != key.data_ptr()
    assert (
        saved_value.untyped_storage().nbytes()
        == saved_value.numel() * saved_value.element_size()
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_blend_adopts_owned_loads_then_releases_only_selection_kv(
    monkeypatch, batch_size
):
    from ucm.sparse.qcfuse.blend_info import SelectMode

    indices = torch.tensor([1, 3])
    for _ in range(batch_size):
        state = QcFuseRequestState(
            blend_style=BlendStyle.DO_BLEND_FINISH,
            select_mode=SelectMode.ATTN,
            start=0,
            ratio=0.4,
            critical_layers=[1, 2],
        )
        owned = [(torch.zeros(5, 4), torch.ones(5, 4)) for _ in range(3)]
        for layer_id, (key, value) in enumerate(owned):
            state.put_kv(key, value, layer_id, take_ownership=True)
            assert state.get_kv(layer_id)[0] is key
            assert state.get_kv(layer_id)[1] is value

        def select(info, *, old_k, old_q, positions):
            assert old_k == [owned[1][0], owned[2][0]]
            return indices, torch.tensor([2])

        monkeypatch.setattr(blender.indice_select.IndiceSelector, "select", select)
        state.q_buffer = [None, torch.zeros(1, 4), torch.zeros(1, 4)]
        query = torch.arange(20).reshape(5, 4).float()
        key, value = torch.zeros_like(query), torch.ones_like(query)
        result = blender.blend_layer(
            state,
            0,
            query,
            key,
            value,
            torch.arange(5),
            None,
            store=_ChunkStore(),
            sample_dir_chunk="/chunk/a",
        )
        torch.testing.assert_close(
            result[0], query.index_select(0, indices), rtol=0, atol=0
        )
        assert result[1] is key and result[2] is value
        assert state.get_kv(0) == owned[0]
        assert state.get_kv(1) == (None, None)
        assert state.get_kv(2) == (None, None)


def test_offline_kv_cannot_adopt_projection_storage():
    state = QcFuseRequestState(blend_style=BlendStyle.KVCOMPUTE)
    with pytest.raises(ValueError, match="Only online"):
        state.put_kv(torch.zeros(2, 4), torch.zeros(2, 4), 0, take_ownership=True)
