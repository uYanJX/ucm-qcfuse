"""Read-only Linux page-cache snapshot for an owned QCFuse benchmark cell.

The Linux mincore API reports only a momentary residency bitmap. It does not
identify the storage medium or prove which later I/O belongs to KV or TTFT:
https://man7.org/linux/man-pages/man2/mincore.2.html
"""

import ctypes
from datetime import datetime, timezone
import json
import mmap
import os
from pathlib import Path
import stat
import sys


def owned_files(root, directories):
    root = Path(root).resolve(strict=True)
    marker = root / ".qcfuse_cell_owner"
    if not marker.is_file() or marker.is_symlink():
        raise ValueError("Expected a real benchmark ownership marker")
    if marker.stat().st_uid != os.geteuid():
        raise ValueError("Ownership marker belongs to a different user")
    paths = set()
    for directory in directories:
        directory = Path(directory).resolve(strict=True)
        if root not in directory.parents:
            raise ValueError("Inspect only directories inside the owned cell")
        for path in directory.rglob("*.bin"):
            if path.is_symlink() or root not in path.resolve(strict=True).parents:
                raise ValueError("Reject cache symlinks or paths outside the cell")
            paths.add(path)
    if not paths:
        raise ValueError("No owned KV payload files found")
    return root, sorted(paths)


def file_residency(path, *, chunk_bytes=64 * 1024 * 1024):
    if sys.platform != "linux":
        raise RuntimeError("This verifier uses the Linux mincore API")
    page = os.sysconf("SC_PAGESIZE")
    if chunk_bytes < page or chunk_bytes % page:
        raise ValueError("Inspection chunks must be positive whole pages")
    libc = ctypes.CDLL(None, use_errno=True)
    mincore = libc.mincore
    mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    mincore.restype = ctypes.c_int
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        original = os.fstat(fd)
        if not stat.S_ISREG(original.st_mode) or original.st_uid != os.geteuid():
            # Modern kernels can obscure residency for files the caller cannot
            # modify or does not own. Our generated files must be owned by us.
            raise ValueError("Expected a regular cache file owned by this user")
        resident = 0
        for offset in range(0, original.st_size, chunk_bytes):
            length = min(chunk_bytes, original.st_size - offset)
            count = (length + page - 1) // page
            # MAP_PRIVATE allows obtaining an address without writing the
            # read-only file. Never read or write mapping contents: that would
            # fault pages in and invalidate the observation.
            with mmap.mmap(
                fd,
                length,
                flags=mmap.MAP_PRIVATE,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset,
            ) as mapping:
                address_ref = ctypes.c_char.from_buffer(mapping)
                vector = (ctypes.c_ubyte * count)()
                try:
                    code = mincore(ctypes.addressof(address_ref), length, vector)
                    if code:
                        error = ctypes.get_errno()
                        raise OSError(error, os.strerror(error), str(path))
                    resident += sum(value & 1 for value in vector)
                finally:
                    del address_ref
        after = os.fstat(fd)
        assert (
            original.st_dev,
            original.st_ino,
            original.st_size,
            original.st_mtime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ), "Cache file changed during inspection"
        return {
            "path": str(path),
            "device": original.st_dev,
            "inode": original.st_ino,
            "file_bytes": original.st_size,
            "pages": (original.st_size + page - 1) // page,
            "resident_pages": resident,
            "page_size": page,
        }
    finally:
        os.close(fd)


def inspect_owned_cache(root, directories, *, chunk_bytes=64 * 1024 * 1024):
    root, paths = owned_files(root, directories)
    files = [file_residency(p, chunk_bytes=chunk_bytes) for p in paths]
    if not any(f["file_bytes"] for f in files):
        raise ValueError("No nonempty owned KV payload files found")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "momentary_Linux_page_cache_snapshot_not_SSD_or_physical_KV_read_proof",
        "cache_root": str(root),
        "files": files,
        "file_bytes": sum(f["file_bytes"] for f in files),
        "pages": sum(f["pages"] for f in files),
        "resident_pages": sum(f["resident_pages"] for f in files),
        "all_payload_pages_observed_nonresident": all(
            f["resident_pages"] == 0 for f in files
        ),
        "payload_contents_read": False,
        "files_changed": False,
        "GPU_queried_or_initialized": False,
        "storage_medium_verified": False,
    }
