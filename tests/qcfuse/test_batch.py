import pytest
import torch

from ucm.sparse.qcfuse.batch import QcFuseStepBatch, cumulative_starts
from ucm.sparse.qcfuse.blend_info import BlendStyle, QcFuseRequestState


def _state(phase, session, start=2, qcompute_end=4):
    return QcFuseRequestState(
        blend_style=phase,
        query_session_id=session,
        sample_dir_chunk=f"/chunk/{session}",
        sample_dir_query=f"/query/{session}",
        start=start,
        qcompute_end=qcompute_end,
    )


def test_builds_two_request_prefill_layout():
    states = {
        "a": _state(BlendStyle.QCOMPUTE, "a"),
        "b": _state(BlendStyle.QCOMPUTE, "b"),
    }
    batch = QcFuseStepBatch.build(["a", "b"], {"a": 3, "b": 5}, states, [0, 0], [3, 5])

    assert batch.active
    assert batch.phase is BlendStyle.QCOMPUTE
    assert [(item.input_start, item.input_end) for item in batch.entries] == [
        (0, 3),
        (3, 8),
    ]
    assert cumulative_starts([3, 5], torch.device("cpu")).tolist() == [0, 3]


def test_decode_step_is_not_rewritten():
    state = _state(BlendStyle.DO_BLEND, "a")
    batch = QcFuseStepBatch.build(["a"], {"a": 1}, {"a": state}, [10], [8])
    assert not batch.active
    assert batch.entries[0].is_decode


@pytest.mark.parametrize(
    "states,error",
    [
        (
            {
                "a": _state(BlendStyle.QCOMPUTE, "a"),
                "b": _state(BlendStyle.DO_BLEND, "b"),
            },
            "one phase",
        ),
        (
            {
                "a": _state(BlendStyle.QCOMPUTE, "same"),
                "b": _state(BlendStyle.QCOMPUTE, "same"),
            },
            "distinct query sessions",
        ),
    ],
)
def test_rejects_unsafe_batches(states, error):
    with pytest.raises(RuntimeError, match=error):
        QcFuseStepBatch.build(["a", "b"], {"a": 3, "b": 3}, states, [0, 0], [3, 3])


def test_rejects_chunked_prefill():
    with pytest.raises(RuntimeError, match="unchunked"):
        QcFuseStepBatch.build(
            ["a"],
            {"a": 3},
            {"a": _state(BlendStyle.QCOMPUTE, "a")},
            [0],
            [6],
        )
