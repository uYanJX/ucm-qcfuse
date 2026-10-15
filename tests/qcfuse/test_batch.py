import pytest
import torch
from types import SimpleNamespace

from ucm.sparse.qcfuse.batch import QcFuseStepBatch, cumulative_starts
from ucm.sparse.qcfuse.blend_info import BlendStyle, QcFuseRequestState
from ucm.sparse.qcfuse.qcfuse import QcFuse


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


def test_do_blend_reuses_qcompute_handoff():
    fuser = object.__new__(QcFuse)
    fuser._store = None
    fuser._qcompute_reuse_hits = 0
    fuser._qcompute_reuse_misses = 0
    fuser._qcompute_stash = {
        "session": {
            "q_buffer": ["q"],
            "query_k_buffer": ["k"],
            "q_lens": [1],
            "q_offsets": [0],
            "query_k_lens": [1],
        }
    }
    state = QcFuseRequestState(query_session_id="session")

    fuser._materialize_blend_inputs(state)

    assert state.qcompute_reused
    assert state.q_buffer == ["q"]
    assert fuser.qcompute_reuse_stats() == {
        "qcompute_reuse_hits": 1,
        "qcompute_reuse_misses": 0,
        "qcompute_stash_size": 0,
    }


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


@pytest.mark.parametrize("storage", ["gpu", "cpu", "ssd"])
@pytest.mark.parametrize("phase", list(BlendStyle))
def test_projection_compaction_is_limited_to_gpu_blend(storage, phase):
    fuser = object.__new__(QcFuse)
    fuser._store = SimpleNamespace(storage_mode=storage)
    fuser._step = SimpleNamespace(active=True, phase=phase)
    assert fuser.compact_projection_values() == (
        storage == "gpu" and phase in (BlendStyle.DO_BLEND, BlendStyle.DO_BLEND_FINISH)
    )
    fuser._step.active = False
    assert not fuser.compact_projection_values()
