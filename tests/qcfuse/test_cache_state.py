import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks/qcfuse"))
from cache_state import observe_cache_residency, request_cold_cache
from cache_residency import file_residency, inspect_owned_cache


def test_cold_advice_targets_only_owned_cache_files(tmp_path, monkeypatch):
    root = tmp_path / "cell"
    directory = root / "batch"
    directory.mkdir(parents=True)
    (root / ".qcfuse_cell_owner").write_text("our result")
    (directory / "kv.bin").write_bytes(b"owned bytes")
    observed = []
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(
        os,
        "posix_fadvise",
        lambda fd, *args: observed.append(os.read(fd, 20)),
        raising=False,
    )
    result = request_cold_cache(root, [directory])
    assert observed == [b"owned bytes"]
    assert result["files"] == 1 and result["advised_bytes"] == 11
    assert result["eviction_is_advisory"] is True
    with pytest.raises(ValueError, match="owned cell"):
        request_cold_cache(root, [tmp_path])


def test_cold_advice_rejects_cache_symlinks_outside_cell(tmp_path):
    root = tmp_path / "cell"
    directory = root / "batch"
    directory.mkdir(parents=True)
    (root / ".qcfuse_cell_owner").write_text("our result")
    outside = tmp_path / "unrelated.bin"
    outside.write_bytes(b"keep")
    (directory / "link.bin").symlink_to(outside)
    if not hasattr(os, "posix_fadvise"):
        pytest.skip("requires per-file POSIX advice")
    with pytest.raises(ValueError, match="outside"):
        request_cold_cache(root, [directory])
    assert outside.read_bytes() == b"keep"


@pytest.mark.skipif(sys.platform != "linux", reason="mincore requires Linux")
def test_residency_observation_does_not_read_or_warm_payload(tmp_path):
    import hashlib

    root = tmp_path / "cell"
    directory = root / "batch"
    directory.mkdir(parents=True)
    (root / ".qcfuse_cell_owner").write_text("our result")
    path = directory / "kv.bin"
    payload = bytes(range(256)) * 8192 + b"partial final page"
    path.write_bytes(payload)
    expected_hash = hashlib.sha256(payload).hexdigest()
    warm = inspect_owned_cache(root, [directory])
    assert warm["resident_pages"] == warm["pages"] > 0
    action = request_cold_cache(root, [directory], verify_residency=True)
    snapshot = action["residency_before_online"]
    assert snapshot["file_bytes"] == action["advised_bytes"] == len(payload)
    assert action["residency_check_excluded_from_online_ttft"]
    assert action["eviction_is_advisory"]
    assert snapshot["all_payload_pages_observed_nonresident"] == (
        snapshot["resident_pages"] == 0
    )
    assert not snapshot["payload_contents_read"] and not snapshot["files_changed"]
    assert not snapshot["storage_medium_verified"]

    def reads():
        rows = dict(
            line.split(":", 1)
            for line in Path("/proc/self/io").read_text().splitlines()
        )
        return int(rows["read_bytes"])

    before_reads = reads()
    page_size = os.sysconf("SC_PAGESIZE")
    repeated = inspect_owned_cache(root, [directory], chunk_bytes=page_size)
    assert reads() == before_reads
    assert repeated["resident_pages"] == snapshot["resident_pages"]
    assert repeated["pages"] == (len(payload) + page_size - 1) // page_size
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
    rewarmed = inspect_owned_cache(root, [directory])
    assert rewarmed["resident_pages"] == rewarmed["pages"]


def test_residency_rejects_missing_owner_outside_directories_and_symlinks(tmp_path):
    root = tmp_path / "cell"
    directory = root / "batch"
    directory.mkdir(parents=True)
    path = directory / "kv.bin"
    path.write_bytes(b"owned payload")
    with pytest.raises(ValueError, match="ownership marker"):
        inspect_owned_cache(root, [directory])
    (root / ".qcfuse_cell_owner").write_text("our result")
    with pytest.raises(ValueError, match="inside the owned cell"):
        inspect_owned_cache(root, [tmp_path])
    (directory / "alias.bin").symlink_to(path)
    with pytest.raises(ValueError, match="symlinks"):
        inspect_owned_cache(root, [directory])
    assert path.read_bytes() == b"owned payload"


@pytest.mark.skipif(sys.platform != "linux", reason="mincore requires Linux")
def test_residency_empty_payload_and_chunk_validation(tmp_path):
    root = tmp_path / "cell"
    directory = root / "batch"
    directory.mkdir(parents=True)
    (root / ".qcfuse_cell_owner").write_text("our result")
    with pytest.raises(ValueError, match="No owned KV"):
        inspect_owned_cache(root, [directory])
    empty = directory / "empty.bin"
    empty.write_bytes(b"")
    assert file_residency(empty)["pages"] == 0
    with pytest.raises(ValueError, match="nonempty"):
        inspect_owned_cache(root, [directory])
    for chunk in (0, 1, os.sysconf("SC_PAGESIZE") + 1):
        with pytest.raises(ValueError, match="whole pages"):
            file_residency(empty, chunk_bytes=chunk)


@pytest.mark.skipif(sys.platform != "linux", reason="mincore requires Linux")
def test_warm_state_observation_does_not_prepare_or_read_payload(tmp_path, monkeypatch):
    root = tmp_path / "cell"
    directory = root / "batch"
    directory.mkdir(parents=True)
    (root / ".qcfuse_cell_owner").write_text("our result")
    path = directory / "kv.bin"
    payload = bytes(range(256)) * 8192 + b"partial page"
    path.write_bytes(payload)
    warm = observe_cache_residency(root, [directory])
    assert (
        warm["residency_before_online"]["resident_pages"]
        == warm["residency_before_online"]["pages"]
    )
    cold = request_cold_cache(root, [directory], verify_residency=True)[
        "residency_before_online"
    ]

    def forbidden(*args, **kwargs):
        raise AssertionError("Observation must not advise, read or prepare payload")

    with monkeypatch.context() as patch:
        patch.setattr(os, "posix_fadvise", forbidden)
        patch.setattr(os, "read", forbidden)
        patch.setattr(Path, "read_bytes", forbidden)
        observed = observe_cache_residency(root, [directory])
    snapshot = observed["residency_before_online"]
    assert snapshot["resident_pages"] == cold["resident_pages"]
    assert snapshot["file_bytes"] == len(payload)
    assert observed["method"] == "mincore_without_cache_preparation"
    assert observed["cache_preparation_performed"] is False
    assert observed["residency_check_excluded_from_online_ttft"] is True
    assert not snapshot["payload_contents_read"] and not snapshot["files_changed"]
    assert path.read_bytes() == payload
