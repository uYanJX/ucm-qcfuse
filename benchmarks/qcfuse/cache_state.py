"""Request cold filesystem cache only for files owned by this benchmark cell."""

import os
import time
from pathlib import Path


def observe_cache_residency(root, directories):
    """Observe requested warm state without preparing or reading KV payloads."""
    from cache_residency import inspect_owned_cache

    started = time.perf_counter()
    snapshot = inspect_owned_cache(root, directories)
    return {
        "method": "mincore_without_cache_preparation",
        "residency_before_online": snapshot,
        "residency_check_seconds": time.perf_counter() - started,
        "residency_check_excluded_from_online_ttft": True,
        "cache_preparation_performed": False,
    }


def request_cold_cache(root, directories, *, verify_residency=False):
    directories = list(directories)
    root = Path(root).resolve()
    if not (root / ".qcfuse_cell_owner").is_file():
        raise ValueError("Cold-cache request needs a cell ownership marker")
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("Per-file cold-cache advice is unavailable on this host")
    paths = []
    for directory in directories:
        directory = Path(directory).resolve()
        if root not in directory.parents:
            raise ValueError("Cold-cache advice is restricted to the owned cell")
        paths.extend(directory.rglob("*.bin"))
    paths = sorted(set(paths))
    for path in paths:
        if root not in path.resolve().parents:
            raise ValueError("A cache symlink points outside the owned cell")
    started = time.perf_counter()
    advised_bytes = 0
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            advised_bytes += path.stat().st_size
        finally:
            os.close(fd)
    result = {
        "files": len(paths),
        "advised_bytes": advised_bytes,
        "seconds": time.perf_counter() - started,
        "eviction_is_advisory": True,
        "method": "fsync_then_per_file_POSIX_FADV_DONTNEED",
    }
    if verify_residency:
        from cache_residency import inspect_owned_cache

        started = time.perf_counter()
        snapshot = inspect_owned_cache(root, directories)
        assert len(snapshot["files"]) == len(paths)
        assert snapshot["file_bytes"] == advised_bytes
        result["residency_before_online"] = snapshot
        result["residency_check_seconds"] = time.perf_counter() - started
        result["residency_check_excluded_from_online_ttft"] = True
    return result
