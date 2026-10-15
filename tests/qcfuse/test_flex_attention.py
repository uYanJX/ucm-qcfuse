import pytest
import torch

from ucm.sparse.qcfuse.flex_attention import make_position_plan


@pytest.mark.parametrize("key_block", [64, 128])
@pytest.mark.parametrize("kv_length", [1, 63, 64, 65, 193, 277])
def test_block_plan_exactly_covers_original_causal_positions(key_block, kv_length):
    torch.manual_seed(kv_length)
    positions = torch.randint(kv_length, (139,))
    positions[0], positions[-1] = 0, kv_length - 1
    original = positions.clone()
    plan = make_position_plan(positions, kv_length, key_block)
    mask = plan.block_mask
    actual = torch.zeros(len(positions), kv_length, dtype=torch.bool)
    expected = torch.arange(kv_length)[None] <= positions[:, None]
    for row_block in range(2):
        rows = slice(row_block * 128, min((row_block + 1) * 128, len(positions)))
        full = mask.full_kv_num_blocks[0, 0, row_block].item()
        partial = mask.kv_num_blocks[0, 0, row_block].item()
        for col in mask.full_kv_indices[0, 0, row_block, :full].tolist():
            columns = slice(col * key_block, min((col + 1) * key_block, kv_length))
            actual[rows, columns] = True
        for col in mask.kv_indices[0, 0, row_block, :partial].tolist():
            columns = torch.arange(
                col * key_block, min((col + 1) * key_block, kv_length)
            )
            query_rows = torch.arange(rows.start, rows.stop)
            actual[rows, columns] = mask.mask_mod(
                torch.tensor(0), torch.tensor(0), query_rows[:, None], columns[None]
            )
    assert torch.equal(actual, expected)
    assert torch.equal(positions, original)
    assert plan.matches(positions, kv_length, key_block)
    positions[0] = kv_length - 1
    assert not plan.matches(positions, kv_length, key_block)
    assert torch.equal(plan.positions, original.to(torch.int32))


def test_plan_accepts_immutable_inference_selection_without_version_counter():
    with torch.inference_mode():
        positions = torch.arange(139)
        plan = make_position_plan(positions, 193)
        assert plan.matches(positions, 193, 64)
        assert not plan.matches(positions.clone(), 193, 64)
        assert not plan.matches(positions, 194, 64)
