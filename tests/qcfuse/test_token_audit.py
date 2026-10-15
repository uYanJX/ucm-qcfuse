import json
from types import SimpleNamespace

import pytest
import torch

from ucm.sparse.qcfuse.blend_info import QcFuseRequestState, BlendStyle
from ucm.sparse.qcfuse.qcfuse import QcFuse
from ucm.sparse.qcfuse.indice_select import _compute_budget
from ucm.sparse.qcfuse.token_audit import (
    new_audit,
    observe_layer,
    finish_audit,
    observe_prefill_batch,
    begin_gpu_span,
)


@pytest.mark.parametrize("ratio", [0.1, 0.2, 0.3, 0.4, 0.5])
def test_observations_follow_actual_rows_not_requested_ratio(tmp_path, ratio):
    # Tiny context deliberately exposes the floor/minimum budget rounding.
    path = tmp_path / "audit.jsonl"
    budget = _compute_budget(17, ratio)
    selected = torch.tensor(list(range(2, 2 + budget)) + [19, 20, 21])
    state = QcFuseRequestState(
        request_id="r",
        query_session_id="s",
        ratio=ratio,
        start=0,
        blend_style=BlendStyle.DO_BLEND_FINISH,
        blend_top_indices=selected,
        token_audit=new_audit(path, "DO_BLEND_FINISH", ratio, 0, [0, 2, 19, 22]),
    )
    fuser = object.__new__(QcFuse)
    entry = SimpleNamespace(state=state, scheduled_tokens=22)
    fuser._step = SimpleNamespace(
        active=True,
        entries=[entry],
        phase=state.blend_style,
        global_selected_indices=lambda _device: selected,
    )
    fuser._current_layer_id = 0
    observe_layer(state, 0, 22, len(selected), 22)
    hidden = torch.arange(22, dtype=torch.float32).view(22, 1)
    output, _ = fuser.ffn_begin(hidden, hidden)
    assert output[:, 0].tolist() == selected.tolist()
    finish_audit(state)
    finish_audit(state)  # one record per completed forward, not per decode step
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["selected_context_tokens"] == budget
    assert row["recomputed_tokens"] == budget + 3
    assert row["retained_tokens"] == 22
    assert row["qkv_token_layers"] == 22  # selection layer projects full prompt
    assert row["ffn_token_layers"] == len(selected)
    assert row["actual_context_recompute_ratio"] == budget / 17


@pytest.mark.parametrize("scheduled", [[11], [11, 17], [11, 17, 19, 23]])
def test_batch_audit_records_actual_scheduled_requests(scheduled):
    entries = [
        SimpleNamespace(
            request_id=f"internal-{index}",
            scheduled_tokens=tokens,
            state=QcFuseRequestState(
                client_request_id=f"client-{index}", token_audit={}
            ),
        )
        for index, tokens in enumerate(scheduled)
    ]
    step = SimpleNamespace(active=True, entries=entries)
    observe_prefill_batch(step)
    for entry in entries:
        batch = entry.state.token_audit["scheduled_prefill_batch"]
        assert batch["request_count"] == len(scheduled)
        assert batch["scheduled_tokens"] == scheduled
        assert batch["client_request_ids"] == [
            f"client-{i}" for i in range(len(scheduled))
        ]


def test_host_counts_keep_actual_work_without_events_or_files(monkeypatch):
    import os

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Counts-only observation must not time CUDA or write files"
        )

    monkeypatch.setattr(torch.cuda, "Event", forbidden)
    monkeypatch.setattr(os, "makedirs", forbidden)
    state = QcFuseRequestState(
        request_id="worker-request",
        client_request_id="client-request",
        ratio=0.4,
        blend_top_indices=torch.tensor([2, 3, 4, 5, 6, 7, 19, 20, 21]),
        token_audit=new_audit(
            None, "DO_BLEND_FINISH", 0.4, 0, [0, 2, 19, 22], counts_only=True
        ),
    )
    assert begin_gpu_span(state, "blender", 0, SimpleNamespace(is_cuda=True)) is None
    observe_layer(state, 0, 22, 9, 22)
    from ucm.sparse.qcfuse.token_audit import observe_ffn

    observe_ffn(state, 9)
    observe_layer(state, 1, 9, 9, 22)
    observe_ffn(state, 9)
    row = finish_audit(state)
    assert row["client_request_id"] == "client-request"
    assert row["selected_tokens"] == row["recomputed_tokens"] == 9
    assert row["retained_tokens"] == 22
    assert row["qkv_token_layers"] == 31
    assert row["ffn_token_layers"] == 18
    assert row["measured_spans"] == []
    assert row["observation_mode"] == "host_shape_counts"
    assert finish_audit(state) is None


def test_token_observer_remains_disabled_by_default():
    assert new_audit(None, "DO_BLEND_FINISH", 0.4, 0, [0, 2, 19, 22]) is None


def test_diagnostic_hashes_actual_selection_order_and_positions(tmp_path):
    import hashlib

    selected = torch.tensor([5, 2, 19, 20, 21])
    state = QcFuseRequestState(
        request_id="audit",
        ratio=0.4,
        blend_top_indices=selected,
        positions=selected.clone(),
        attention_positions=selected.to(torch.int32),
        token_audit=new_audit(
            tmp_path / "selection.jsonl", "DO_BLEND_FINISH", 0.4, 0, [0, 2, 19, 22]
        ),
    )
    row = finish_audit(state)
    expected = hashlib.sha256(selected.numpy().astype("<i8").tobytes()).hexdigest()
    for name in ("selected_indices", "positions", "attention_positions"):
        assert row[name + "_int64_le_sha256"] == expected
    assert (
        expected
        != hashlib.sha256(
            selected.sort().values.numpy().astype("<i8").tobytes()
        ).hexdigest()
    )


def test_counts_only_does_not_copy_selection_to_host(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Formal counts-only timing copied selection to host")

    state = QcFuseRequestState(
        request_id="counts",
        ratio=0.4,
        blend_top_indices=torch.tensor([2, 3, 19, 20, 21]),
        token_audit=new_audit(
            None, "DO_BLEND_FINISH", 0.4, 0, [0, 2, 19, 22], counts_only=True
        ),
    )
    monkeypatch.setattr(torch.Tensor, "to", forbidden)
    assert not any("sha256" in key for key in finish_audit(state))
