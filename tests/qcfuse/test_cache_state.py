import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks/qcfuse"))
from cache_state import request_cold_cache


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
