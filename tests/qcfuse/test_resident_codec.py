import builtins
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from ucm.store.qcfuse import packed_format as pf, transfer
from ucm.store.qcfuse.resident_codec import QueryAliasReader, validate_query_alias


@pytest.fixture(autouse=True)
def clean_registry(tmp_path):
    transfer.release_resident_files()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    transfer.release_transfer_pools("codec_test")
    transfer.release_resident_files()
    for path in tmp_path.rglob("*.bin"):
        path.unlink()


def pair(root, tokens=11, query_tokens=0):
    tensors = {}
    for layer in range(3):
        key = torch.arange(tokens * 8).reshape(tokens, 2, 4).to(torch.bfloat16)
        tensors[layer] = key + layer * 100, -key - layer * 100
    context_end = tokens - query_tokens
    positions = [[context_end - 1, 1, 4], [0, context_end - 2], [3]]
    chunk, query = root / "chunk", root / "query"
    pf.save_packed_kv(str(chunk), tensors, 3)
    pf.save_query_cache(
        str(query),
        3,
        tensors,
        {layer: torch.tensor(indices) for layer, indices in enumerate(positions)},
        [1, 2],
        tensors,
        {
            "total_tokens": context_end,
            "orig_chunk_ranges": [[0, 1], [1, context_end], [context_end, tokens]],
            "context_positions_by_layer": positions,
        },
        {},
    )
    return chunk / "kv_packed.bin", query / "query_packed.bin", tensors, positions


def query_items(path):
    meta = pf.load_query_meta(str(path.parent))
    return [
        (
            section,
            int(layer),
            name,
            pf.section_tensor_meta(meta, section, int(layer), name),
        )
        for section in ("digest", "critical")
        for layer in meta[section]["layers"]
        for name in ("k", "v")
    ]


def test_query_alias_proves_ordered_byte_mapping_and_owns_outputs(tmp_path):
    source, query, tensors, positions = pair(tmp_path)
    bindings = validate_query_alias(str(query), str(source))
    raw = torch.tensor(list(source.read_bytes()), dtype=torch.uint8)
    reader = transfer._ResidentReader(raw, "gpu")
    alias = QueryAliasReader(str(query), str(source), reader, bindings, "cpu")
    try:
        for section, layer, name, item in query_items(query):
            expected = tensors[layer][name == "v"]
            if section == "digest":
                expected = expected[positions[layer]]
            actual = alias.owned_tensor(item)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.fill_(99)
            torch.testing.assert_close(
                alias.owned_tensor(item), expected, rtol=0, atol=0
            )
    finally:
        alias.close()
        reader.close()


def test_query_alias_preserves_full_query_tail_and_context_only_digest(tmp_path):
    source, query, tensors, positions = pair(tmp_path, query_tokens=2)
    reader = transfer._ResidentReader(
        torch.tensor(list(source.read_bytes()), dtype=torch.uint8), "gpu"
    )
    alias = QueryAliasReader(
        str(query),
        str(source),
        reader,
        validate_query_alias(str(query), str(source)),
        "cpu",
    )
    try:
        for section, layer, name, item in query_items(query):
            expected = tensors[layer][name == "v"]
            if section == "digest":
                expected = expected[positions[layer]]
            actual = alias.owned_tensor(item)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert actual.shape[0] == (
                len(positions[layer]) if section == "digest" else 11
            )
    finally:
        alias.close()
        reader.close()


@pytest.mark.parametrize(
    "section,layer,name", [("digest", 2, "k"), ("critical", 1, "v")]
)
def test_query_alias_rejects_wrong_source_bytes(tmp_path, section, layer, name):
    source, query, _, _ = pair(tmp_path)
    item = pf.section_tensor_meta(
        pf.load_query_meta(str(query.parent)), section, layer, name
    )
    raw = bytearray(query.read_bytes())
    raw[item["offset"]] ^= 1
    query.write_bytes(raw)
    with pytest.raises(ValueError, match="bytes do not match"):
        validate_query_alias(str(query), str(source))


@pytest.mark.parametrize("bad", [-1, 11, True, 2.5])
def test_query_alias_rejects_invalid_positions(tmp_path, bad):
    source, query, _, _ = pair(tmp_path)
    path = query.parent / "query_packed_meta.json"
    meta = json.loads(path.read_text())
    meta["metadata"]["context_positions_by_layer"][0][0] = bad
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="outside the full cache"):
        validate_query_alias(str(query), str(source))


def test_alias_mismatch_rolls_back_before_any_gpu_preload(tmp_path, monkeypatch):
    source, query, _, _ = pair(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    transfer.preload_resident_file(str(source), "cpu")
    path = query.parent / "query_packed_meta.json"
    meta = json.loads(path.read_text())
    meta["metadata"]["total_tokens"] += 1
    path.write_text(json.dumps(meta))

    def forbidden(*args, **kwargs):
        pytest.fail("GPU allocation attempted before validating query bytes")

    monkeypatch.setattr(transfer, "preload_resident_file", forbidden)
    with pytest.raises(ValueError, match="different full token count"):
        transfer.prepare_resident_batch(
            None,
            [str(source), str(query)],
            "gpu",
            query_sources={str(query): str(source)},
        )
    assert transfer.resident_metrics()["resident_files"] == 0


def test_query_alias_requires_gpu_and_distinct_explicit_sources(tmp_path):
    source, query, _, _ = pair(tmp_path)
    with pytest.raises(ValueError, match="GPU mode"):
        transfer.prepare_resident_batch(
            None,
            [str(source), str(query)],
            "cpu",
            query_sources={str(query): str(source)},
        )
    with pytest.raises(ValueError, match="distinct explicit sources"):
        transfer.prepare_resident_batch(
            None, [str(source)], "gpu", query_sources={str(source): str(source)}
        )


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
@pytest.mark.parametrize("codec", [None, "ans_f16_bits"])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_cuda_alias_reuse_is_bit_exact_without_files(
    tmp_path, monkeypatch, codec, batch
):
    if codec:
        pytest.importorskip("nvidia.nvcomp")
    cases = [pair(tmp_path / str(i), 11 + i, query_tokens=2) for i in range(batch)]
    paths = [str(p) for source, query, _, _ in cases for p in (source, query)]
    aliases = {str(query): str(source) for source, query, _, _ in cases}
    items = {str(query): query_items(query) for _, query, _, _ in cases}
    full_meta = {
        str(source): pf.load_packed_meta(str(source.parent))
        for source, _, _, _ in cases
    }
    prepared = transfer.prepare_resident_batch(
        None, paths, "gpu", gpu_codec=codec, query_sources=aliases
    )
    assert prepared["resident_gpu_query_aliases"] == batch
    assert prepared["resident_gpu_compressed_files"] == (batch if codec else 0)
    sources = [transfer.resident_reader(str(source)) for source, _, _, _ in cases]
    payloads = [
        [p.clone() for p in r._entries.values()] if codec else [r.tensor.clone()]
        for r in sources
    ]
    for source, query, _, _ in cases:
        source.unlink()
        query.unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("Online GPU residency read a file")

    monkeypatch.setattr(builtins, "open", forbidden)
    assert transfer.prepare_resident_batch(
        None, paths, "gpu", gpu_codec=codec, query_sources=aliases
    )["resident_batch_reused"]
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    for _ in range(2):
        with ThreadPoolExecutor(max_workers=1) as executor:

            def load():
                with torch.cuda.stream(producer):
                    results = []
                    for source, query, tensors, positions in cases:
                        for section, layer, name, item in items[str(query)]:
                            result = transfer.resident_reader(str(query)).owned_tensor(
                                item
                            )
                            expected = tensors[layer][name == "v"]
                            if section == "digest":
                                expected = expected[positions[layer]]
                            results.append((result, expected))
                        for layer in range(3):
                            kv = transfer.load_packed_layer(
                                str(source.parent),
                                full_meta[str(source)],
                                layer,
                                "cuda:0",
                                kind="codec_test",
                                wait=False,
                            )
                            for result, expected in zip(kv, tensors[layer]):
                                transfer.wait_on_current_stream(kv)
                                results.append((result, expected))
                    ready = torch.cuda.Event()
                    ready.record(producer)
                    return results, ready

            results, ready = executor.submit(load).result()
        with torch.cuda.stream(consumer):
            consumer.wait_event(ready)
            for actual, expected in results:
                actual.record_stream(consumer)
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                actual.fill_(99)
        consumer.synchronize()
    for reader, saved in zip(sources, payloads):
        actual = list(reader._entries.values()) if codec else [reader.tensor]
        for value, expected in zip(actual, saved):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)


@cuda
def test_cuda_compressed_residency_preserves_all_bf16_patterns(tmp_path):
    pytest.importorskip("nvidia.nvcomp")
    raw = (
        torch.arange(65536, dtype=torch.int32)
        .to(torch.uint16)
        .view(torch.bfloat16)
        .reshape(256, 1, 256)
    )
    pf.save_packed_kv(str(tmp_path), {0: (raw, raw.flip(0))}, 1)
    path = str(tmp_path / "kv_packed.bin")
    transfer.prepare_resident_batch(None, [path], "gpu", gpu_codec="ans_f16_bits")
    meta = pf.load_packed_meta(str(tmp_path))
    values = transfer.load_packed_layer(
        str(tmp_path), meta, 0, "cuda:0", kind="codec_test"
    )
    for actual, expected in zip(values, (raw, raw.flip(0))):
        assert torch.equal(actual.cpu().view(torch.uint8), expected.view(torch.uint8))


@cuda
def test_cuda_partial_compressed_preload_can_retry(tmp_path, monkeypatch):
    pytest.importorskip("nvidia.nvcomp")
    source, query, _, _ = pair(tmp_path)
    original = transfer.preload_resident_file

    def fail_second(path, *args, **kwargs):
        if path == str(query):
            raise RuntimeError("forced second file failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(transfer, "preload_resident_file", fail_second)
    with pytest.raises(RuntimeError, match="forced second file failure"):
        transfer.prepare_resident_batch(
            None, [str(source), str(query)], "gpu", gpu_codec="ans_f16_bits"
        )
    assert transfer.resident_metrics()["resident_files"] == 0
    monkeypatch.setattr(transfer, "preload_resident_file", original)
    assert (
        transfer.prepare_resident_batch(
            None, [str(source), str(query)], "gpu", gpu_codec="ans_f16_bits"
        )["resident_files"]
        == 2
    )
