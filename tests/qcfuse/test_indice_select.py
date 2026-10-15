import torch
import pytest

from ucm.sparse.qcfuse.blend_info import QcFuseRequestState
from ucm.sparse.qcfuse.indice_select import IndiceSelector, _assemble_selection_keys


def test_full_selection_keeps_every_request_token():
    state = QcFuseRequestState(
        ratio=1.0,
        chunk_loc_list=torch.tensor([0, 2, 10, 13]),
        req_len_list=torch.tensor([3]),
    )

    indices, lengths = IndiceSelector.select(state)

    assert indices.tolist() == list(range(13))
    assert lengths.tolist() == [13]


def test_rotate_stacked_does_not_alias_in_place_q_and_k():
    def in_place_rotary(_positions, query, key):
        query.add_(1)
        key.add_(2)
        return query, key

    tensor = torch.zeros(2, 3, 1, 4)
    positions = torch.arange(3)

    query = IndiceSelector._rotate_stacked(
        in_place_rotary,
        positions,
        tensor.clone(),
        num_layers=2,
        num_heads=1,
        head_dim=4,
        is_query=True,
    )
    key = IndiceSelector._rotate_stacked(
        in_place_rotary,
        positions,
        tensor.clone(),
        num_layers=2,
        num_heads=1,
        head_dim=4,
        is_query=False,
    )

    assert torch.equal(query, torch.ones_like(query))
    assert torch.equal(key, torch.full_like(key, 2))


@pytest.mark.parametrize("query_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_selection_key_assembly_preserves_cat_order_dtype_and_source(query_dtype):
    prefixes = [
        torch.arange(80).reshape(10, 8).to(torch.float16)[1:8:2] + layer
        for layer in range(3)
    ]
    queries = [
        torch.arange(24).reshape(3, 8).to(query_dtype) + layer for layer in range(3)
    ]
    expected = torch.cat([torch.stack(prefixes), torch.stack(queries)], dim=1)
    actual = _assemble_selection_keys(prefixes, queries)
    assert actual.dtype == expected.dtype and actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.zero_()
    torch.testing.assert_close(
        torch.cat([torch.stack(prefixes), torch.stack(queries)], dim=1),
        expected,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("is_query", [False, True])
def test_selection_rotation_uses_one_layer_scratch_and_keeps_position_order(is_query):
    positions = torch.tensor([3, 1, 7, 4])
    calls = []

    def rotary(pos, query, key):
        calls.append((pos.clone(), query.shape, key.shape))
        return query + pos[:, None, None], key - pos[:, None, None]

    source = torch.arange(3 * 4 * 2 * 8).reshape(3, 4, 2, 8).float()
    expected = source + (1 if is_query else -1) * positions[None, :, None, None]
    actual = IndiceSelector._rotate_stacked(
        rotary, positions, source.clone(), 3, 2, 8, is_query=is_query
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(calls) == 3
    for pos, query_shape, key_shape in calls:
        assert torch.equal(pos, positions)
        assert query_shape == key_shape == (4, 2, 8)
