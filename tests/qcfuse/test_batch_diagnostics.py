"""Keep scheduling comparisons attributable and payload audits ownership bounded."""

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks/qcfuse"))
from batch_diagnostics import execution_context, hash_owned_payloads, query_batches


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_query_groups_preserve_order_tokens_and_sampling(batch):
    requests = [
        ([index, 42], {"max_new_tokens": 1}, {"blend_style": "QCOMPUTE"})
        for index in range(batch)
    ]
    assert query_batches(requests) == [requests]
    for size in range(1, batch + 1):
        groups = query_batches(iter(requests), size)
        restored = [item for group in groups for item in group]
        assert [id(item) for item in restored] == [id(item) for item in requests]
        assert all(1 <= len(group) <= size for group in groups)


@pytest.mark.parametrize("size", [0, -1, 1.5])
def test_query_group_size_rejected_before_generation(size):
    with pytest.raises(ValueError, match="positive integer"):
        query_batches([], size)


def test_query_grouping_cannot_split_blend_or_mixed_phases():
    query = ("query", {}, {"blend_style": "QCOMPUTE"})
    blend = ("context", {}, {"blend_style": "DO_BLEND_FINISH"})
    for requests in [[blend], [query, blend], [("full", {}, None)]]:
        with pytest.raises(ValueError, match="only QCOMPUTE"):
            query_batches(requests, 1)


def test_execution_context_records_actual_client_without_guessing(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    engine = SimpleNamespace(
        _engine=SimpleNamespace(engine_core=object()), qcompute_batch_size=1
    )
    observed = execution_context(engine)
    assert observed["engine_core_client_class"] == "builtins.object"
    assert observed["multiprocessing_environment"] == "0"
    assert observed["configured_qcompute_batch_size"] == 1
    assert execution_context(SimpleNamespace())["engine_core_client_class"] is None


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_warmup_splits_only_queries_and_keeps_blend_budget(batch):
    from blend_ssd import SSDPipelineEngine

    engine = object.__new__(SSDPipelineEngine)
    engine.batch_size = batch
    engine.qcompute_batch_size = 1
    engine.baseline = "ours"
    engine.method = "attn"
    samples = [
        SimpleNamespace(
            has_cache=True, query_sep=f"q{n}", prompt=f"full{n}", max_new_tokens=4096
        )
        for n in range(batch)
    ]
    engine._prepare_samples = lambda *_: samples
    engine._prepare_resident_batch = lambda *_: None
    engine._blend_args = lambda style, *_: {"blend_style": style}
    engine._ssd_args = lambda *_: {}
    calls = []
    engine._drain_generate_batch = calls.append
    engine.warmup_blend([{}] * batch, "longbench_v2", 0.5, num_warmup=2)
    assert len(calls) == 2 * (batch + 1)
    for offset in (0, batch + 1):
        assert [group[0][0] for group in calls[offset : offset + batch]] == [
            f"q{n}" for n in range(batch)
        ]
        blend = calls[offset + batch]
        assert len(blend) == batch
        assert all(item[2]["blend_style"] == "DO_BLEND_FINISH" for item in blend)
        assert all(item[1]["max_new_tokens"] == 4096 for item in blend)


@pytest.mark.parametrize("size", [0, 3, 1.5])
def test_invalid_engine_query_size_rejected_before_model_initialization(size):
    from blend_ssd import SSDPipelineEngine

    with pytest.raises(ValueError, match="between 1 and batch_size"):
        SSDPipelineEngine("unused", batch_size=2, qcompute_batch_size=size)


def test_hashes_compare_payload_contents_and_reject_unowned_targets(tmp_path):
    root = tmp_path / "owned"
    root.mkdir()
    (root / ".qcfuse_cell_owner").write_text("test owner")
    cache = root / "batch_0" / "source"
    cache.mkdir(parents=True)
    payload = cache / "chunk_packed.bin"
    payload.write_bytes(b"immutable KV")
    (cache / "metadata.json").write_text("metadata is not a KV payload")
    before = hash_owned_payloads(root, [cache, cache])
    assert len(before["files"]) == 1
    assert before["files"][0] == {
        "relative_path": "batch_0/source/chunk_packed.bin",
        "bytes": len(b"immutable KV"),
        "sha256": hashlib.sha256(b"immutable KV").hexdigest(),
    }
    assert payload.read_bytes() == b"immutable KV"
    assert (
        before["excluded_from_online_ttft"] and before["reads_payloads_before_online"]
    )
    payload.write_bytes(b"changed KV")
    assert hash_owned_payloads(root, [cache])["files"] != before["files"]
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="owned directories"):
        hash_owned_payloads(root, [outside])
    (cache / "external.bin").symlink_to(outside / "unread.bin")
    with pytest.raises(ValueError, match="owned files"):
        hash_owned_payloads(root, [cache])
    with pytest.raises(ValueError, match="ownership marker"):
        hash_owned_payloads(tmp_path, [cache])


@pytest.mark.parametrize(
    "baseline,storage,kind,option",
    [
        ("ours", "ssd", "audit", ["--audit-cache-payloads"]),
        ("ours", "cpu", "timing", ["--audit-cache-payloads"]),
        ("ours", "cpu", "quality", ["--audit-cache-payloads"]),
        ("fullcomp", "cpu", "audit", ["--audit-cache-payloads"]),
        ("fullcomp", "cpu", "audit", ["--qcompute-batch-size", "1"]),
        ("ours", "cpu", "audit", ["--qcompute-batch-size", "3"]),
    ],
)
def test_invalid_audit_options_fail_before_reading_data_or_creating_cache(
    monkeypatch, tmp_path, baseline, storage, kind, option
):
    import run_frozen_cell

    output, cache = tmp_path / "result.json", tmp_path / "cache"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frozen_cell",
            "--data",
            "unused",
            "--manifest",
            "unused",
            "--dataset",
            "longbench_v2",
            "--model",
            "unused",
            "--output",
            str(output),
            "--cache-parent",
            str(cache),
            "--baseline",
            baseline,
            "--storage",
            storage,
            "--kind",
            kind,
            "--batch-size",
            "2",
            *option,
        ],
    )
    with pytest.raises(SystemExit) as raised:
        run_frozen_cell.main()
    assert raised.value.code == 2
    assert not cache.exists() and not output.exists()
