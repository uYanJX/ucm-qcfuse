import torch

from ucm.sparse.qcfuse.blend_info import QcFuseRequestState
from ucm.sparse.qcfuse.indice_select import IndiceSelector


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
