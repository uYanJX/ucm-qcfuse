import torch

from ucm.sparse.qcfuse import blender
from ucm.sparse.qcfuse import digest_index
from ucm.sparse.qcfuse.blend_info import AttParams, BlendStyle, QcFuseRequestState


def _rotary(positions, query, key):
    shape = (positions.numel(),) + (1,) * (query.ndim - 1)
    delta = positions.to(query.dtype).view(shape)
    return query + delta, key + delta


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
