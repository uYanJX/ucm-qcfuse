"""Bound query scheduling and record owned-cache evidence outside online timing."""

import hashlib
import os
from pathlib import Path
import time


def query_batches(requests, size=None):
    requests = list(requests)
    if size is not None and (not isinstance(size, int) or size < 1):
        raise ValueError("QCOMPUTE batch size must be a positive integer")
    if any(
        not args or args.get("blend_style") != "QCOMPUTE" for _, _, args in requests
    ):
        raise ValueError("Query grouping accepts only QCOMPUTE requests")
    if not requests:
        return []
    size = len(requests) if size is None else size
    return [requests[start : start + size] for start in range(0, len(requests), size)]


def execution_context(engine):
    client = getattr(getattr(engine, "_engine", None), "engine_core", None)
    return {
        "runner_pid": os.getpid(),
        "engine_core_client_class": (
            f"{type(client).__module__}.{type(client).__qualname__}"
            if client is not None
            else None
        ),
        "multiprocessing_environment": os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING"),
        "configured_qcompute_batch_size": getattr(engine, "qcompute_batch_size", None),
    }


def hash_owned_payloads(root, directories):
    root = Path(root).resolve()
    if not (root / ".qcfuse_cell_owner").is_file():
        raise ValueError("Cache hashing requires a cell ownership marker")
    paths = set()
    for directory in directories:
        directory = Path(directory)
        if directory.is_symlink() or root not in directory.resolve().parents:
            raise ValueError("Cache hashing is restricted to owned directories")
        paths.update(directory.rglob("*.bin"))
    if not paths:
        raise ValueError("No owned KV payloads to hash")
    for path in paths:
        if path.is_symlink() or root not in path.resolve().parents:
            raise ValueError("Cache payload points outside owned files")
    started = time.perf_counter()
    files = []
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("KV payload changed during audit hashing")
        files.append(
            {
                "relative_path": str(path.relative_to(root)),
                "bytes": before.st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return {
        "files": files,
        "seconds": time.perf_counter() - started,
        "excluded_from_online_ttft": True,
        "reads_payloads_before_online": True,
    }
