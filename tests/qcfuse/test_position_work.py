import pytest
import torch

from ucm.sparse.qcfuse.token_audit import position_mask_work


def test_equal_row_counts_can_have_different_causal_work():
    early = position_mask_work(torch.arange(5), 10, query_block=2, key_block=2)
    late = position_mask_work(torch.arange(5, 10), 10, query_block=2, key_block=2)
    assert early["query_rows"] == late["query_rows"] == 5
    assert early["valid_causal_qk_pairs_per_head"] == 15
    assert late["valid_causal_qk_pairs_per_head"] == 40
    assert early["full_causal_qk_pairs_per_head"] == 55
    assert early["reference_issued_dot_slots_per_head"] == 24
    assert late["reference_issued_dot_slots_per_head"] == 56


def test_full_rows_match_full_reference_with_padded_final_query_tile():
    row = position_mask_work(torch.arange(9), 9, query_block=4, key_block=4)
    assert row["valid_causal_qk_pairs_per_head"] == 45
    assert row["valid_pair_fraction_vs_full_causal"] == 1
    assert row["reference_issued_dot_slots_per_head"] == 96
    assert row["reference_tile_work_fraction"] == 1


@pytest.mark.parametrize("positions", [torch.tensor([-1]), torch.tensor([10])])
def test_out_of_retained_kv_positions_are_rejected(positions):
    with pytest.raises(ValueError, match="outside"):
        position_mask_work(positions, 10)


def test_empty_query_is_zero_work():
    row = position_mask_work(torch.tensor([], dtype=torch.int64), 10)
    assert row["valid_causal_qk_pairs_per_head"] == 0
    assert row["reference_issued_dot_slots_per_head"] == 0
