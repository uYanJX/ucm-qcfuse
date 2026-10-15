# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
"""Persistent SSD readers and asynchronous CUDA transfer pools for QCFuse.

The pool is request-scoped by :mod:`qcfuse_store`. Each pool owns its own
pinned-host ring, CUDA stream, and file readers, so independent requests never
overwrite one another's staging buffers. A CUDA event protects every ring
slot until the corresponding H2D copy has completed.
"""

from __future__ import annotations

import mmap
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from ucm.store.qcfuse import packed_format as pf


def _attach_ready_event(tensors, event):
    for tensor in tensors:
        tensor._qcfuse_ready_event = event
    return tensors


def wait_on_current_stream(tensors):
    """Order model consumption after transfer, then protect allocator lifetime."""
    key, value = tensors
    if key.is_cuda:
        stream = torch.cuda.current_stream(key.device)
        events = []
        for tensor in tensors:
            event = getattr(tensor, "_qcfuse_ready_event", None)
            if event is not None and all(event is not other for other in events):
                stream.wait_event(event)
                events.append(event)
            tensor.record_stream(stream)
    return tensors


class _ResidentReader:
    """Read packed bytes from a CPU-resident tensor without touching SSD."""

    def __init__(self, tensor: torch.Tensor, mode: str, pinned: bool = False) -> None:
        self.tensor = tensor
        self.mode = mode
        self.pinned = bool(pinned)
        self.registered = False
        self.size = int(tensor.numel())
        self.logical_size = self.size
        self.codec = None
        # Keep one byte view for the lifetime of a CPU resident reader.  The
        # previous path rebuilt a NumPy/memoryview wrapper for every layer,
        # which added Python overhead to the CPU-resident pipeline.  GPU
        # resident readers never call read_into and must not call tensor.numpy().
        self._byte_view = (
            memoryview(tensor.numpy()).cast("B") if mode == "cpu" else None
        )

    def advise(self, _offset: int, _length: int) -> None:
        return

    def owned_tensor(self, item):
        return (
            self.tensor[item["offset"] : item["offset"] + item["nbytes"]]
            .view(item["dtype"])
            .view(item["shape"])
            .clone()
        )

    def read_into(
        self,
        destination: torch.Tensor,
        offset: int,
        nbytes: int,
        destination_view: Optional[memoryview] = None,
    ) -> None:
        if offset < 0 or nbytes < 0 or offset + nbytes > self.size:
            raise ValueError(
                f"Resident KV read outside buffer: offset={offset}, "
                f"nbytes={nbytes}, size={self.size}"
            )
        if nbytes:
            # A byte copy avoids launching the CPU tensor copy thread pool
            # for each layer on many-core serving hosts.
            source_view = self._byte_view
            temporary_source = False
            if source_view is None:
                source_view = memoryview(self.tensor.numpy()).cast("B")
                temporary_source = True
            temporary_destination = destination_view is None
            if temporary_destination:
                destination_view = memoryview(destination[:nbytes].numpy()).cast("B")
            try:
                destination_view[:nbytes] = source_view[offset : offset + nbytes]
            finally:
                if temporary_destination:
                    destination_view.release()
                if temporary_source:
                    source_view.release()

    def close(self) -> None:
        if self._byte_view is not None:
            self._byte_view.release()
            self._byte_view = None
        if self.registered:
            try:
                torch.cuda.cudart().cudaHostUnregister(self.tensor.data_ptr())
            except (AttributeError, RuntimeError, TypeError, ValueError, OSError):
                pass
            self.registered = False


_RESIDENT_FILES: Dict[str, _ResidentReader] = {}
_RESIDENT_LOCK = threading.Lock()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in ("0", "false", "no", "off", "")


def _clear_handled_cuda_registration_error() -> None:
    """Consume a handled host-registration error before the next kernel launch."""
    cudart = torch.cuda.cudart()
    getter = getattr(cudart, "cudaGetLastError", None)
    if getter is not None:
        getter()
        return
    # Some PyTorch builds expose cudaHostRegister but not cudaGetLastError.
    # Resolve the already-loaded runtime, rather than loading another ABI.
    import ctypes

    major = str(torch.version.cuda).split(".", 1)[0]
    prefix = f"libcudart.so.{major}"
    paths = [
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if prefix in line and line.split()[-1].startswith("/")
    ]
    if not paths:
        raise RuntimeError(
            "Cannot locate loaded CUDA runtime to clear registration error"
        )
    runtime = ctypes.CDLL(paths[0])
    getter = runtime.cudaGetLastError
    getter.argtypes = []
    getter.restype = ctypes.c_int
    getter()


def _try_register_host_memory(tensor: torch.Tensor) -> bool:
    """Register a large pageable resident buffer for direct H2D DMA.

    Pinning the complete CPU resident cache with ``pin_memory=True`` can fail
    because the process memlock limit is smaller than the cache.  CUDA can
    sometimes register an already allocated buffer even in that situation;
    when it cannot, the caller keeps the bounded pinned-ring fallback.
    """
    if tensor.numel() == 0 or not _env_bool("QCFUSE_REGISTER_RESIDENT_CPU", True):
        return False
    attempted = False
    try:
        cudart = torch.cuda.cudart()
        register = cudart.cudaHostRegister
        attempted = True
        result = register(tensor.data_ptr(), tensor.numel(), 0)
        error = result[0] if isinstance(result, (tuple, list)) else result
        error = int(error)
    except (AttributeError, RuntimeError, TypeError, ValueError, OSError):
        if attempted:
            _clear_handled_cuda_registration_error()
        return False
    if error != 0:
        _clear_handled_cuda_registration_error()
        return False
    return True


def preload_resident_file(
    path: str, mode: str, device: str = "cuda:0", gpu_codec=None
) -> int:
    """Load one packed file into CPU pinned memory or GPU memory.

    This is intentionally an explicit preload operation.  Benchmark runners
    must load resident bytes before the online TTFT timer starts, either at
    worker initialization or through prepare_resident_batch between requests.
    """
    path = os.fspath(path)
    mode = str(mode).strip().lower()
    if mode not in ("cpu", "gpu"):
        raise ValueError(f"resident mode must be cpu or gpu, got {mode!r}")
    with _RESIDENT_LOCK:
        if path in _RESIDENT_FILES:
            return _RESIDENT_FILES[path].size
    if gpu_codec is not None:
        if mode != "gpu" or gpu_codec != "ans_f16_bits":
            raise ValueError("Unsupported GPU resident codec")
        from ucm.store.qcfuse.resident_codec import CompressedResidentReader

        reader = CompressedResidentReader(path, device)
        with _RESIDENT_LOCK:
            existing = _RESIDENT_FILES.setdefault(path, reader)
        if existing is not reader:
            reader.close()
        return existing.size
    size = os.path.getsize(path)
    registered = False
    pinned = False
    if size <= 0:
        tensor = torch.empty(0, dtype=torch.uint8, device="cpu")
    else:
        pinned = mode == "cpu"
        try:
            tensor = torch.empty(
                size,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=pinned,
            )
        except (RuntimeError, torch.AcceleratorError):
            # Large packed samples can exceed the host pin budget.  Keep the
            # bytes resident in pageable RAM and let TransferPool copy them
            # into its bounded pinned ring before H2D.
            if pinned and torch.cuda.is_available():
                _clear_handled_cuda_registration_error()
            pinned = False
            tensor = torch.empty(size, dtype=torch.uint8, device="cpu")
        with open(path, "rb", buffering=0) as stream:
            view = memoryview(tensor.numpy()).cast("B")
            try:
                filled = 0
                while filled < size:
                    count = stream.readinto(view[filled:])
                    if not count:
                        raise EOFError(f"Incomplete resident KV preload: {path}")
                    filled += count
            finally:
                view.release()
        if mode == "cpu" and not pinned:
            registered = _try_register_host_memory(tensor)
    if mode == "gpu":
        tensor = tensor.to(device=device, non_blocking=False)
    reader = _ResidentReader(
        tensor,
        mode,
        pinned=(pinned or registered) if mode == "cpu" else False,
    )
    reader.registered = registered
    with _RESIDENT_LOCK:
        existing = _RESIDENT_FILES.setdefault(path, reader)
        if existing is not reader:
            reader.close()
            del tensor
        return existing.size


def resident_reader(path: str) -> Optional[_ResidentReader]:
    with _RESIDENT_LOCK:
        return _RESIDENT_FILES.get(os.fspath(path))


def prepare_resident_batch(
    worker, paths, mode, online_headroom_bytes=0, gpu_codec=None, query_sources=None
):
    """vLLM worker control RPC; load only the active batch outside its timer.

    Call between completed requests. Synchronization makes host unregister and
    GPU buffer release safe after the previous batch's asynchronous transfers.
    No tensor data travels through RPC.
    """
    if mode not in ("cpu", "gpu"):
        raise ValueError(f"Unsupported resident mode: {mode}")
    if online_headroom_bytes < 0:
        raise ValueError("Online GPU headroom must be nonnegative")
    paths = [os.fspath(path) for path in paths]
    query_sources = {
        os.fspath(query): os.fspath(source)
        for query, source in (query_sources or {}).items()
    }
    if gpu_codec not in (None, "ans_f16_bits"):
        raise ValueError("Unsupported GPU resident codec")
    if mode != "gpu" and (gpu_codec or query_sources):
        raise ValueError("Compressed residency and query aliases require GPU mode")
    if any(
        query not in paths or source not in paths or source in query_sources
        for query, source in query_sources.items()
    ):
        raise ValueError("Query aliases require distinct explicit sources in the batch")
    started = time.perf_counter()
    with _RESIDENT_LOCK:
        unchanged = set(paths) == set(_RESIDENT_FILES) and all(
            reader.mode == mode
            and (
                getattr(reader, "source_path", None) == query_sources[path]
                if path in query_sources
                else getattr(reader, "codec", None) == gpu_codec
                and not hasattr(reader, "source_path")
            )
            for path, reader in _RESIDENT_FILES.items()
        )
    if not unchanged:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        release_resident_files()
    try:
        if not unchanged:
            bindings = {}
            if query_sources:
                from ucm.store.qcfuse.resident_codec import (
                    QueryAliasReader,
                    validate_query_alias,
                )

                # Prove equivalence before committing any GPU payloads.
                bindings = {
                    query: validate_query_alias(query, source)
                    for query, source in query_sources.items()
                }
            for path in paths:
                if path not in query_sources:
                    if gpu_codec:
                        preload_resident_file(path, mode, gpu_codec=gpu_codec)
                    else:
                        preload_resident_file(path, mode)
            for query, source in query_sources.items():
                reader = QueryAliasReader(
                    query, source, resident_reader(source), bindings[query], "cuda:0"
                )
                with _RESIDENT_LOCK:
                    _RESIDENT_FILES[query] = reader
            if mode == "gpu" and paths:
                # Include index uploads in explicit preload, before online TTFT.
                torch.cuda.synchronize()
        if mode == "gpu" and paths and online_headroom_bytes:
            # A raw cache that fits can still starve selection/FFN workspace.
            # Probe capacity before request submission; a failure rolls back
            # resident bytes while the worker is healthy. No cache values move.
            try:
                headroom = torch.empty(
                    online_headroom_bytes, dtype=torch.uint8, device="cuda:0"
                )
            except torch.OutOfMemoryError as exc:
                raise RuntimeError(
                    f"CUDA out of memory reserving online headroom "
                    f"({online_headroom_bytes} bytes): {exc}"
                ) from exc
            del headroom
    except BaseException:
        release_resident_files()
        raise
    return {
        **resident_metrics(),
        "resident_prepare_seconds": time.perf_counter() - started,
        "resident_batch_reused": unchanged,
        "online_headroom_bytes": online_headroom_bytes if mode == "gpu" else 0,
        "gpu_resident_codec": gpu_codec,
        "gpu_query_aliases": len(query_sources),
    }


def release_resident_files() -> None:
    """Release registered host buffers after all transfer users have stopped."""
    with _RESIDENT_LOCK:
        readers = list(_RESIDENT_FILES.values())
        _RESIDENT_FILES.clear()
    for reader in readers:
        reader.close()


def resident_metrics() -> Dict[str, int]:
    with _RESIDENT_LOCK:
        cpu_bytes = sum(
            item.size for item in _RESIDENT_FILES.values() if item.mode == "cpu"
        )
        gpu_bytes = sum(
            item.size for item in _RESIDENT_FILES.values() if item.mode == "gpu"
        )
        return {
            "resident_files": len(_RESIDENT_FILES),
            "resident_cpu_bytes": cpu_bytes,
            "resident_gpu_bytes": gpu_bytes,
            "resident_gpu_logical_bytes": sum(
                item.logical_size
                for item in _RESIDENT_FILES.values()
                if item.mode == "gpu"
            ),
            "resident_gpu_compressed_files": sum(
                getattr(item, "codec", None) is not None
                for item in _RESIDENT_FILES.values()
            ),
            "resident_gpu_query_aliases": sum(
                hasattr(item, "source_path") for item in _RESIDENT_FILES.values()
            ),
            "resident_cpu_pinned": sum(
                1
                for item in _RESIDENT_FILES.values()
                if item.mode == "cpu" and item.pinned
            ),
            "resident_cpu_registered": sum(
                1
                for item in _RESIDENT_FILES.values()
                if item.mode == "cpu" and item.registered
            ),
        }


def benchmark_snapshot(worker, reset_peak=False):
    """Small worker-side control RPC; distinguish OS block reads from read calls."""
    process_io = {}
    try:
        with open("/proc/self/io") as stream:
            process_io = {
                name: int(value)
                for name, value in (line.split(":", 1) for line in stream)
            }
    except OSError:
        pass
    gpu = None
    if torch.cuda.is_available():
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        allocator_stats = torch.cuda.memory_stats()
        gpu = {
            "free_bytes": free,
            "total_bytes": total,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "allocator_counters": {
                name: allocator_stats.get(name)
                for name in (
                    "num_alloc_retries",
                    "num_ooms",
                    "num_device_alloc",
                    "num_device_free",
                )
            },
            "allocator_counters_scope": (
                "cumulative_worker_allocator_counts_not_durations;"
                "subtract_before_from_after_for_online_interval;"
                "reset_peak_does_not_reset_counters"
            ),
        }
    return {
        "pid": os.getpid(),
        "process_io": process_io,
        "gpu": gpu,
        "resident": resident_metrics(),
        "io_scope": "all_worker_process_io_not_cache_exclusive",
    }


def _env_int(name: str, default: int, lower: int, upper: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(lower, min(value, upper))


def _nvtx_push(label: str) -> bool:
    if not _env_bool("QCFUSE_NVTX", bool(os.getenv("QCFUSE_PROFILE_PATH"))):
        return False
    try:
        torch.cuda.nvtx.range_push(label)
        return True
    except (AttributeError, RuntimeError):
        return False


def _nvtx_pop(active: bool) -> None:
    if not active:
        return
    try:
        torch.cuda.nvtx.range_pop()
    except (AttributeError, RuntimeError):
        pass


class _PersistentReader:
    """One long-lived fd/mmap pair for a packed cache file."""

    def __init__(self, path: str) -> None:
        self.path = os.fspath(path)
        self.fd = os.open(self.path, os.O_RDONLY)
        self.size = os.fstat(self.fd).st_size
        if self.size and hasattr(os, "posix_fadvise"):
            try:
                # Packed KV layers are consumed in file order by the bounded
                # prefetch producer. Tell the kernel this explicitly so its
                # readahead window can cover the next layer without changing
                # the read path or allocating an unbounded userspace cache.
                os.posix_fadvise(
                    self.fd,
                    0,
                    self.size,
                    os.POSIX_FADV_SEQUENTIAL,
                )
            except (AttributeError, OSError, ValueError):
                pass
        # Direct preadv is the default because it fills pinned memory without
        # an intermediate Python bytes/tensor copy. ``mmap`` remains available
        # for filesystems where page-fault driven reads benchmark better.
        mode = os.getenv("QCFUSE_IO_MODE", "pread").strip().lower()
        self.mode = "pread" if mode == "pread" else "mmap"
        self.mapping: Optional[mmap.mmap] = None
        # Read-ahead windows for consecutive layers overlap heavily. Keep the
        # furthest advised byte so the producer only asks the kernel for the
        # newly uncovered tail instead of repeatedly advising the same bytes.
        self._advised_until = 0
        if self.mode == "mmap" and self.size:
            self.mapping = mmap.mmap(self.fd, self.size, access=mmap.ACCESS_READ)
            if hasattr(self.mapping, "madvise") and hasattr(mmap, "MADV_SEQUENTIAL"):
                try:
                    # Packed KV layers are consumed in file order.  This is a
                    # one-time kernel hint; range-specific WILLNEED advice is
                    # still used below for bounded read-ahead.
                    self.mapping.madvise(mmap.MADV_SEQUENTIAL)
                except (OSError, TypeError, ValueError):
                    pass

    def advise(self, offset: int, length: int) -> None:
        if length <= 0:
            return
        start = max(0, int(offset))
        end = min(self.size, start + int(length))
        if end <= start:
            return
        advice_start = max(start, self._advised_until)
        if end <= advice_start:
            return
        advice_length = end - advice_start
        try:
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(
                    self.fd,
                    advice_start,
                    advice_length,
                    os.POSIX_FADV_WILLNEED,
                )
        except (AttributeError, OSError, ValueError):
            pass
        if self.mapping is not None and hasattr(self.mapping, "madvise"):
            try:
                # Restrict mmap read-ahead to the same layer window advised to
                # the file descriptor. Advising the entire multi-GB mapping
                # on every layer defeats bounded prefetch and can evict useful
                # pages on a multi-request worker.
                page_size = getattr(mmap, "PAGESIZE", 4096)
                mapping_offset = advice_start - (advice_start % page_size)
                mapping_end = min(
                    self.size,
                    ((end + page_size - 1) // page_size) * page_size,
                )
                mapping_length = mapping_end - mapping_offset
                if mapping_length > 0:
                    self.mapping.madvise(
                        mmap.MADV_WILLNEED,
                        mapping_offset,
                        mapping_length,
                    )
            except (AttributeError, OSError, TypeError, ValueError):
                pass
        self._advised_until = max(self._advised_until, end)

    def read_into(
        self,
        destination: torch.Tensor,
        offset: int,
        nbytes: int,
        destination_view: Optional[memoryview] = None,
    ) -> None:
        if offset < 0 or nbytes < 0 or offset + nbytes > self.size:
            raise ValueError(
                f"Packed KV read outside file {self.path}: "
                f"offset={offset}, nbytes={nbytes}, size={self.size}"
            )
        if nbytes == 0:
            return
        if self.mapping is not None:
            view = memoryview(self.mapping)[offset : offset + nbytes]
            temporary_destination = destination_view is None
            if temporary_destination:
                destination_view = memoryview(destination[:nbytes].numpy()).cast("B")
            try:
                destination_view[:nbytes] = view
            finally:
                view.release()
                if temporary_destination:
                    destination_view.release()
            return

        # ``preadv`` may return short reads. Keep the fd offset-independent so
        # the reader remains safe if a caller later uses several reader tasks.
        # The destination is pinned host memory, so the kernel can DMA into it
        # without the extra mmap -> torch copy that the first implementation
        # accidentally introduced.
        position = int(offset)
        remaining = int(nbytes)
        destination_offset = 0
        temporary_destination = destination_view is None
        if temporary_destination:
            destination_view = memoryview(destination.numpy()).cast("B")
        try:
            while remaining:
                target = destination_view[
                    destination_offset : destination_offset + remaining
                ]
                if hasattr(os, "preadv"):
                    consumed = os.preadv(self.fd, [target], position)
                else:
                    data = os.pread(self.fd, remaining, position)
                    consumed = len(data)
                    target[:consumed] = data
                if not consumed:
                    raise EOFError("Unexpected EOF while reading packed KV payload")
                position += consumed
                destination_offset += consumed
                remaining -= consumed
        finally:
            if temporary_destination:
                destination_view.release()

    def close(self) -> None:
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        os.close(self.fd)


def _layer_read_window(layer_id, resolver, count, max_bytes):
    key, value = resolver(layer_id, "k"), resolver(layer_id, "v")
    start, end = key["offset"], value["end"]
    if key["end"] != value["offset"]:
        raise ValueError("K/V bytes are not adjacent in packed layer")
    layers = 1
    for next_layer in range(layer_id + 1, layer_id + count):
        try:
            next_key, next_value = resolver(next_layer, "k"), resolver(next_layer, "v")
        except (KeyError, IndexError):
            break
        if next_key["offset"] != end or next_key["end"] != next_value["offset"]:
            break
        if next_value["end"] - start > max_bytes:
            break
        end = next_value["end"]
        layers += 1
    return start, end - start, layers


class TransferPool:
    """Reusable pinned buffers and an asynchronous CUDA stream for one pool.

    The ring starts at ``QCFUSE_TRANSFER_RING_DEPTH`` (default four) and can
    grow up to ``QCFUSE_TRANSFER_RING_MAX_DEPTH`` (default eight) when the next
    slot's completion event is still pending. Growth is demand-driven, which
    avoids allocating eight large pinned buffers for short or serial requests.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self.lock = threading.Lock()
        self.stream = torch.cuda.Stream(device=device)
        initial = _env_int("QCFUSE_TRANSFER_RING_DEPTH", 4, 2, 16)
        self.max_ring_depth = _env_int(
            "QCFUSE_TRANSFER_RING_MAX_DEPTH", max(initial, 8), initial, 32
        )
        self._slots = [{"event": None} for _ in range(initial)]
        self._next_slot = 0
        self._read_window = None
        self._group_layers = _env_int("QCFUSE_IO_GROUP_LAYERS", 1, 1, 8)
        self._group_max_bytes = _env_int(
            "QCFUSE_IO_GROUP_MAX_BYTES", 1024**3, 4096, 4 * 1024**3
        )
        self._files: Dict[str, _PersistentReader] = {}
        self._aggregate = {
            "ssd_read_seconds": 0.0,
            "ssd_read_bytes": 0.0,
            "resident_read_seconds": 0.0,
            "resident_read_bytes": 0.0,
            "resident_cpu_direct_layers": 0.0,
            "h2d_enqueue_seconds": 0.0,
            "h2d_bytes": 0.0,
            "cuda_wait_seconds": 0.0,
            "cuda_copy_seconds": 0.0,
            "completed_copies": 0,
            "read_blocks": 0,
            "read_window_hits": 0,
            "pinned_allocation_fallbacks": 0.0,
        }
        self._last_operation_metrics: Dict[str, float] = {}
        self._last_completed_metrics: Dict[str, float] = {
            "cuda_event_wait_seconds": 0.0,
            "cuda_event_copy_seconds": 0.0,
        }
        self._measure_timing = _env_bool(
            "QCFUSE_TRANSFER_TIMING", bool(os.getenv("QCFUSE_PROFILE_PATH"))
        )
        self._direct_copy_events = []

    def _reader(self, bin_path: str) -> _PersistentReader:
        resident = resident_reader(bin_path)
        if resident is not None:
            return resident  # type: ignore[return-value]
        reader = self._files.get(bin_path)
        if reader is None:
            reader = _PersistentReader(bin_path)
            self._files[bin_path] = reader
        return reader

    def _complete_slot(self, slot: Dict) -> float:
        event = slot.get("event")
        if event is None:
            return 0.0
        started = time.perf_counter()
        try:
            event.synchronize()
            wait_seconds = time.perf_counter() - started
            copy_seconds = 0.0
            copies = slot.get(
                "copies", [(slot.get("copy_start"), slot.get("copy_end"))]
            )
            for start_event, end_event in copies:
                if start_event is not None and end_event is not None:
                    copy_seconds += max(
                        0.0, start_event.elapsed_time(end_event) / 1000.0
                    )
            self._aggregate["cuda_wait_seconds"] += wait_seconds
            self._aggregate["cuda_copy_seconds"] += copy_seconds
            self._aggregate["completed_copies"] += len(copies)
            self._last_completed_metrics = {
                "cuda_event_wait_seconds": wait_seconds,
                "cuda_event_copy_seconds": copy_seconds,
            }
            return wait_seconds
        finally:
            slot["event"] = None
            slot["copies"] = []

    def _acquire_slot(self) -> Tuple[Dict, float]:
        self._last_completed_metrics = {
            "cuda_event_wait_seconds": 0.0,
            "cuda_event_copy_seconds": 0.0,
        }
        slot = self._slots[self._next_slot]
        self._next_slot = (self._next_slot + 1) % len(self._slots)
        event = slot.get("event")
        if event is not None:
            # Avoid a host stall by adding a slot when the ring is allowed to
            # grow. Once at the cap, wait for the oldest slot as a backstop.
            try:
                pending = not event.query()
            except RuntimeError:
                pending = True
            if pending and len(self._slots) < self.max_ring_depth:
                slot = {"event": None}
                self._slots.append(slot)
            else:
                wait_seconds = self._complete_slot(slot)
                self._clear_slot_metadata(slot)
                return slot, wait_seconds
        self._clear_slot_metadata(slot)
        return slot, 0.0

    def _ensure_byte_buf(self, slot: Dict, name: str, nbytes: int) -> None:
        buf_name = f"pin_buf_{name}"
        view_name = f"pin_view_{name}"
        cap_name = f"byte_capacity_{name}"
        if slot.get(buf_name) is not None and slot.get(cap_name, 0) >= nbytes:
            return
        old_view = slot.get(view_name)
        if old_view is not None:
            old_view.release()
        capacity = max(nbytes, 16384)
        try:
            buffer = torch.empty(capacity, dtype=torch.uint8, pin_memory=True)
            pinned = True
        except (RuntimeError, torch.AcceleratorError) as exc:
            message = str(exc).lower()
            if not any(
                word in message
                for word in ("cumemhostalloc", "out of memory", "invalid argument")
            ):
                raise
            _clear_handled_cuda_registration_error()
            buffer = torch.empty(capacity, dtype=torch.uint8)
            pinned = False
            self._aggregate.setdefault("pinned_allocation_fallbacks", 0.0)
            self._aggregate["pinned_allocation_fallbacks"] += 1.0
        slot[buf_name] = buffer
        slot[f"pin_buf_pinned_{name}"] = pinned
        slot[view_name] = memoryview(slot[buf_name].numpy()).cast("B")
        slot[cap_name] = capacity

    @staticmethod
    def _clear_slot_metadata(slot: Dict) -> None:
        for key in list(slot.keys()):
            if (
                key == "event"
                or key.startswith("pin_buf_")
                or key.startswith("pin_view_")
                or key.startswith("byte_capacity_")
            ):
                continue
            del slot[key]

    def _read_block(
        self,
        slot: Dict,
        bin_path: str,
        block_offset: int,
        block_nbytes: int,
    ) -> float:
        self._ensure_byte_buf(slot, "block", block_nbytes)
        reader = self._reader(bin_path)
        slot["resident_mode"] = (
            reader.mode if isinstance(reader, _ResidentReader) else "ssd"
        )
        active = _nvtx_push(f"qcfuse.ssd_read:{block_nbytes}")
        started = time.perf_counter()
        try:
            reader.read_into(
                slot["pin_buf_block"],
                block_offset,
                block_nbytes,
                slot.get("pin_view_block"),
            )
        finally:
            _nvtx_pop(active)
        elapsed = time.perf_counter() - started
        self._aggregate["read_blocks"] += 1
        if slot["resident_mode"] == "ssd":
            self._aggregate["ssd_read_seconds"] += elapsed
            self._aggregate["ssd_read_bytes"] += float(block_nbytes)
        else:
            self._aggregate.setdefault("resident_read_seconds", 0.0)
            self._aggregate["resident_read_seconds"] += elapsed
            self._aggregate.setdefault("resident_read_bytes", 0.0)
            self._aggregate["resident_read_bytes"] += float(block_nbytes)
        slot["block_offset"] = block_offset
        slot["block_nbytes"] = block_nbytes
        return elapsed

    def _read_layer_block(self, bin_path, layer_id, resolver):
        key, value = resolver(layer_id, "k"), resolver(layer_id, "v")
        window = self._read_window
        if self._group_layers > 1 and window is not None:
            path, slot = window
            if (
                path == bin_path
                and slot.get("block_offset") is not None
                and key["offset"] >= slot["block_offset"]
                and value["end"] <= slot["block_offset"] + slot["block_nbytes"]
            ):
                self._aggregate["read_window_hits"] += 1
                self._last_completed_metrics = {
                    "cuda_event_wait_seconds": 0.0,
                    "cuda_event_copy_seconds": 0.0,
                }
                return slot, 0.0, 0.0, 0
        slot, wait_seconds = self._acquire_slot()
        offset, nbytes, layers = _layer_read_window(
            layer_id, resolver, self._group_layers, self._group_max_bytes
        )
        read_seconds = self._read_block(slot, bin_path, offset, nbytes)
        slot["group_layers"] = layers
        self._read_window = (bin_path, slot) if self._group_layers > 1 else None
        return slot, wait_seconds, read_seconds, nbytes

    def _copy_layer_from_block(
        self, slot: Dict, layer_id: int, meta_item_resolver
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        block_offset = slot["block_offset"]

        def _view(name: str):
            item = meta_item_resolver(layer_id, name)
            rel = item["offset"] - block_offset
            if rel < 0 or rel + item["nbytes"] > slot["block_nbytes"]:
                raise ValueError(
                    f"Packed KV layer {layer_id}.{name} is outside pinned block"
                )
            return (
                slot["pin_buf_block"][rel : rel + item["nbytes"]]
                .view(item["dtype"])
                .view(item["shape"])
            ), item

        k_pin, k_item = _view("k")
        v_pin, v_item = _view("v")
        self._aggregate["h2d_bytes"] += float(k_item["nbytes"] + v_item["nbytes"])
        copy_start = (
            torch.cuda.Event(enable_timing=True) if self._measure_timing else None
        )
        copy_end = (
            torch.cuda.Event(enable_timing=True) if self._measure_timing else None
        )
        # The completion event is required to protect a ring slot, but timing
        # support is unnecessary on the normal path.  Keeping timing disabled
        # unless explicitly requested avoids per-layer CUDA event overhead in
        # production CPU/SSD pipelines.
        completion = torch.cuda.Event(enable_timing=self._measure_timing)
        active = _nvtx_push(f"qcfuse.h2d_enqueue:layer={layer_id}")
        started = time.perf_counter()
        try:
            with torch.cuda.stream(self.stream):
                if copy_start is not None:
                    copy_start.record(self.stream)
                k_gpu = torch.empty(
                    k_item["shape"], dtype=k_item["dtype"], device=self.device
                )
                v_gpu = torch.empty(
                    v_item["shape"], dtype=v_item["dtype"], device=self.device
                )
                asynchronous = bool(slot.get("pin_buf_pinned_block", True))
                k_gpu.copy_(k_pin, non_blocking=asynchronous)
                v_gpu.copy_(v_pin, non_blocking=asynchronous)
                if copy_end is not None:
                    copy_end.record(self.stream)
                # A prefetch thread's current stream is not the model's stream.
                # Carry this exact completion event to the actual consumer.
                completion.record(self.stream)
        finally:
            _nvtx_pop(active)
        enqueue_seconds = time.perf_counter() - started
        slot["copy_start"] = copy_start
        slot["copy_end"] = copy_end
        slot.setdefault("copies", []).append((copy_start, copy_end))
        slot["event"] = completion
        _attach_ready_event((k_gpu, v_gpu), completion)
        return k_gpu, v_gpu, enqueue_seconds

    def _read_ahead(
        self, reader: _PersistentReader, layer_id: int, meta_item_resolver
    ) -> None:
        read_ahead_layers = _env_int("QCFUSE_READ_AHEAD_LAYERS", 4, 0, 32)
        if read_ahead_layers <= 0:
            return
        try:
            next_k = meta_item_resolver(layer_id + 1, "k")
        except (KeyError, IndexError, TypeError, ValueError):
            return
        last_end = next_k["offset"]
        for offset_layer in range(1, read_ahead_layers + 1):
            try:
                next_v = meta_item_resolver(layer_id + offset_layer, "v")
            except (KeyError, IndexError, TypeError, ValueError):
                break
            last_end = next_v["end"]
        reader.advise(next_k["offset"], last_end - next_k["offset"])

    def load_layer(
        self,
        bin_path: str,
        meta: Dict,
        layer_id: int,
        meta_item_resolver,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read and enqueue one packed layer without blocking the consumer."""
        del meta  # Metadata is retained by the store; resolver is authoritative.
        with self.lock:
            resident = resident_reader(bin_path)
            if resident is not None and resident.mode == "gpu":
                k_item = meta_item_resolver(layer_id, "k")
                v_item = meta_item_resolver(layer_id, "v")
                self._last_operation_metrics = {
                    "ssd_read_seconds": 0.0,
                    "ssd_read_bytes": 0.0,
                    "resident_read_bytes": 0.0,
                    "h2d_enqueue_seconds": 0.0,
                    "cuda_wait_seconds": 0.0,
                    "cuda_copy_seconds": 0.0,
                    "ring_depth": float(len(self._slots)),
                    "resident_gpu": 1.0,
                }
                # RoPE and index_copy_ in blender mutate the returned tensors.
                # Each request must own its repair buffers; exposing registry
                # views corrupts the next warmup/request's pre-RoPE cache.
                result = resident.owned_tensor(k_item), resident.owned_tensor(v_item)
                ready = torch.cuda.Event(enable_timing=False)
                ready.record(torch.cuda.current_stream(device=self.device))
                return _attach_ready_event(result, ready)
            if resident is not None and resident.mode == "cpu" and resident.pinned:
                k_item = meta_item_resolver(layer_id, "k")
                v_item = meta_item_resolver(layer_id, "v")
                k_pin = (
                    resident.tensor[
                        k_item["offset"] : k_item["offset"] + k_item["nbytes"]
                    ]
                    .view(k_item["dtype"])
                    .view(k_item["shape"])
                )
                v_pin = (
                    resident.tensor[
                        v_item["offset"] : v_item["offset"] + v_item["nbytes"]
                    ]
                    .view(v_item["dtype"])
                    .view(v_item["shape"])
                )
                copy_start = (
                    torch.cuda.Event(enable_timing=True)
                    if self._measure_timing
                    else None
                )
                copy_end = torch.cuda.Event(enable_timing=self._measure_timing)
                started = time.perf_counter()
                with torch.cuda.stream(self.stream):
                    k_gpu = torch.empty(
                        k_item["shape"], dtype=k_item["dtype"], device=self.device
                    )
                    v_gpu = torch.empty(
                        v_item["shape"], dtype=v_item["dtype"], device=self.device
                    )
                    if copy_start is not None:
                        copy_start.record(self.stream)
                    k_gpu.copy_(k_pin, non_blocking=True)
                    v_gpu.copy_(v_pin, non_blocking=True)
                    copy_end.record(self.stream)
                enqueue_seconds = time.perf_counter() - started
                self._aggregate["h2d_enqueue_seconds"] += enqueue_seconds
                self._aggregate["h2d_bytes"] += float(
                    k_item["nbytes"] + v_item["nbytes"]
                )
                if copy_start is not None:
                    self._direct_copy_events.append((copy_start, copy_end))
                self._aggregate["resident_cpu_direct_layers"] += 1.0
                self._last_operation_metrics = {
                    "ssd_read_seconds": 0.0,
                    "resident_read_seconds": 0.0,
                    "ssd_read_bytes": 0.0,
                    "resident_read_bytes": 0.0,
                    "resident_cpu": 1.0,
                    "resident_cpu_direct": 1.0,
                    "h2d_enqueue_seconds": enqueue_seconds,
                    "cuda_wait_seconds": 0.0,
                    "cuda_copy_seconds": 0.0,
                    "ring_depth": float(len(self._slots)),
                }
                # The resident tensor itself remains alive in the registry;
                # the stream dependency protects the destination tensors.
                return _attach_ready_event((k_gpu, v_gpu), copy_end)
            slot, wait_seconds, read_seconds, block_nbytes = self._read_layer_block(
                bin_path, layer_id, meta_item_resolver
            )
            reader = self._reader(bin_path)
            self._read_ahead(reader, layer_id, meta_item_resolver)
            k_gpu, v_gpu, enqueue_seconds = self._copy_layer_from_block(
                slot, layer_id, meta_item_resolver
            )
            self._aggregate["h2d_enqueue_seconds"] += enqueue_seconds
            self._last_operation_metrics = {
                "ssd_read_seconds": (
                    read_seconds if slot.get("resident_mode") == "ssd" else 0.0
                ),
                "resident_read_seconds": (
                    read_seconds if slot.get("resident_mode") != "ssd" else 0.0
                ),
                "ssd_read_bytes": (
                    float(block_nbytes) if slot.get("resident_mode") == "ssd" else 0.0
                ),
                "resident_read_bytes": (
                    float(block_nbytes) if slot.get("resident_mode") != "ssd" else 0.0
                ),
                "resident_cpu": float(slot.get("resident_mode") == "cpu"),
                "h2d_enqueue_seconds": enqueue_seconds,
                "cuda_wait_seconds": wait_seconds,
                "cuda_copy_seconds": 0.0,
                "ring_depth": float(len(self._slots)),
                "group_layers": float(slot.get("group_layers", 1)),
                "pageable_staging": float(not slot.get("pin_buf_pinned_block", True)),
                **self._last_completed_metrics,
            }
            return k_gpu, v_gpu

    def last_operation_metrics(self) -> Dict[str, float]:
        with self.lock:
            return dict(self._last_operation_metrics)

    def drain_metrics(self) -> Dict[str, float]:
        with self.lock:
            for slot in self._slots:
                if slot.get("event") is not None:
                    self._complete_slot(slot)
            self._drain_direct_copy_events()
            return {**self._aggregate, "ring_depth": float(len(self._slots))}

    def _drain_direct_copy_events(self) -> None:
        for start, end in self._direct_copy_events:
            end.synchronize()
            self._aggregate["cuda_copy_seconds"] += max(
                0.0, start.elapsed_time(end) / 1000.0
            )
            self._aggregate["completed_copies"] += 1
        self._direct_copy_events.clear()

    def close(self) -> None:
        with self.lock:
            self._read_window = None
            for slot in self._slots:
                if slot.get("event") is not None:
                    self._complete_slot(slot)
            self._drain_direct_copy_events()
            for reader in self._files.values():
                reader.close()
            self._files.clear()
            for slot in self._slots:
                for key, value in list(slot.items()):
                    if key.startswith("pin_view_") and value is not None:
                        value.release()
                        slot[key] = None


# Pool cache keyed by (kind, device). QCFuse uses request-specific ``kind``
# values, so there is no cross-request staging-buffer sharing.
_POOLS: Dict = {}
_POOLS_LOCK = threading.Lock()


def get_transfer_pool(kind: str, device: str) -> TransferPool:
    """Return (creating if needed) a reusable transfer pool for ``kind``."""
    with _POOLS_LOCK:
        pool = _POOLS.get((kind, device))
        if pool is None:
            pool = TransferPool(device)
            _POOLS[(kind, device)] = pool
        return pool


def drain_transfer_pool(kind: str, device: str) -> Dict[str, float]:
    with _POOLS_LOCK:
        pool = _POOLS.get((kind, device))
    if pool is None:
        return {}
    return pool.drain_metrics()


def last_transfer_metrics(kind: str, device: str) -> Dict[str, float]:
    with _POOLS_LOCK:
        pool = _POOLS.get((kind, device))
    if pool is None:
        return {}
    return pool.last_operation_metrics()


def release_transfer_pools(kind_prefix: str) -> None:
    """Close and drop reusable buffers/readers for completed requests."""
    with _POOLS_LOCK:
        pools = [
            _POOLS.pop(key)
            for key in list(_POOLS)
            if str(key[0]).startswith(kind_prefix)
        ]
    for pool in pools:
        pool.close()


def load_packed_layer(
    sample_dir: str,
    meta: Dict,
    layer_id: int,
    device: str,
    kind: str = "chunk",
    wait: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a full-context packed KV layer to GPU (``kv_packed.bin``)."""
    pool = get_transfer_pool(kind, device)
    bin_path = _bin_path(sample_dir, pf._PACKED_BIN_NAME)
    resolver = lambda lid, name: pf._tensor_meta(meta, lid, name)
    tensors = pool.load_layer(bin_path, meta, layer_id, resolver)
    return wait_on_current_stream(tensors) if wait else tensors


def load_query_layer(
    sample_dir: str,
    query_meta: Dict,
    section: str,
    layer_id: int,
    device: str,
    kind: str = "query",
    wait: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load one query-cache layer (digest or critical) to GPU."""
    pool = get_transfer_pool(kind, device)
    bin_path = _bin_path(sample_dir, pf._QUERY_BIN_NAME)
    resolver = lambda lid, name: pf.section_tensor_meta(query_meta, section, lid, name)
    tensors = pool.load_layer(bin_path, query_meta, layer_id, resolver)
    return wait_on_current_stream(tensors) if wait else tensors


def _bin_path(sample_dir: str, name: str) -> str:
    return os.path.join(sample_dir, name)
