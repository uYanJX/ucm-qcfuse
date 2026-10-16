"""Sample-addressed storage and bounded layer prefetch for QCFuse."""

from __future__ import annotations

import threading
import time
import os
import json
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from ucm.sparse.qcfuse.blend_info import BlendStyle
from ucm.store.qcfuse import packed_format as pf
from ucm.store.qcfuse import transfer
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1

KVByLayer = Mapping[int, Tuple[torch.Tensor, torch.Tensor]]
_Op = Tuple[str, str, int, str]


def _record_on_current_stream(
    tensors: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transfer CUDA tensor ownership to the model's current stream."""
    return transfer.wait_on_current_stream(tensors)


@dataclass
class QcFuseTransferMetrics:
    read_seconds: float = 0.0
    wait_seconds: float = 0.0
    layers: int = 0
    ssd_read_seconds: float = 0.0
    ssd_read_bytes: float = 0.0
    resident_read_seconds: float = 0.0
    resident_read_bytes: float = 0.0
    resident_cpu_direct_layers: float = 0.0
    pinned_allocation_fallbacks: float = 0.0
    h2d_enqueue_seconds: float = 0.0
    h2d_bytes: float = 0.0
    cuda_wait_seconds: float = 0.0
    cuda_copy_seconds: float = 0.0
    max_ring_depth: float = 0.0


class QcFusePrefetchTask(Task):
    """A bounded producer that overlaps SSD/H2D work with model layers."""

    def __init__(self, request_id: str, operations: Sequence[_Op], loader, depth: int):
        self.request_id = request_id
        self.operations = list(operations)
        self._loader = loader
        self._slots = threading.Semaphore(max(1, int(depth)))
        self._condition = threading.Condition()
        self._results: Dict[_Op, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._error: Optional[BaseException] = None
        self._done = False
        self._cancelled = False
        self.metrics = QcFuseTransferMetrics()
        # ``wait_seconds`` is periodically moved to the store aggregate when
        # a layer is consumed. Keep a per-task copy for profiling.
        self.profile_wait_seconds = 0.0
        self._thread = threading.Thread(
            target=self._run,
            name=f"qcfuse-prefetch-{request_id}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            for operation in self.operations:
                self._slots.acquire()
                with self._condition:
                    if self._cancelled:
                        self._slots.release()
                        break
                started = time.perf_counter()
                result = self._loader(operation, self.request_id)
                elapsed = time.perf_counter() - started
                with self._condition:
                    self.metrics.read_seconds += elapsed
                    self.metrics.layers += 1
                    self._results[operation] = result
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()
        finally:
            with self._condition:
                self._done = True
                self._condition.notify_all()

    def take(self, operation: _Op) -> Tuple[torch.Tensor, torch.Tensor]:
        started = time.perf_counter()
        with self._condition:
            while (
                operation not in self._results
                and self._error is None
                and not self._done
            ):
                self._condition.wait()
            self.metrics.wait_seconds += time.perf_counter() - started
            if operation in self._results:
                result = self._results.pop(operation)
            elif self._error is not None:
                raise RuntimeError(
                    f"QCFuse SSD prefetch failed for request {self.request_id}"
                ) from self._error
            else:
                raise RuntimeError(
                    f"QCFuse prefetch ended before {operation[:3]} was ready"
                )
        self._slots.release()
        return result

    def cancel(self) -> None:
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()
        for _ in range(max(1, len(self.operations))):
            self._slots.release()

    def check(self) -> bool:
        with self._condition:
            return self._done and not self._results

    def wait(self) -> None:
        self._thread.join()
        if self._error is not None:
            raise RuntimeError(
                f"QCFuse SSD prefetch failed for request {self.request_id}"
            ) from self._error


class UcmQcFuseStore(UcmKVStoreBaseV1):
    """QCFuse cache store; block-addressed UCM operations are unsupported."""

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.device = str(config.get("device", "cuda:0"))
        self.cache_root = config.get("cache_root")
        self._is_worker = "device_id" in config
        self.storage_mode = (
            str(config.get("storage_mode", os.getenv("QCFUSE_STORAGE_MODE", "ssd")))
            .strip()
            .lower()
        )
        self.resident_preload_policy = config.get("resident_preload_policy", "all")
        if self.resident_preload_policy not in ("all", "batch"):
            raise ValueError("resident_preload_policy must be all or batch")
        if self.storage_mode not in ("ssd", "cpu", "gpu"):
            raise ValueError(
                f"QCFuse storage_mode must be ssd, cpu, or gpu, got {self.storage_mode!r}"
            )
        self._meta_lock = threading.Lock()
        self._packed_meta: Dict[str, Dict] = {}
        self._query_meta: Dict[str, Dict] = {}
        self._tasks: Dict[str, QcFusePrefetchTask] = {}
        self._query_tasks: Dict[str, QcFusePrefetchTask] = {}
        self._chunk_tasks: Dict[str, QcFusePrefetchTask] = {}
        self._metrics = QcFuseTransferMetrics()
        self._closed = False
        self._profile_path = config.get("profile_path") or os.getenv(
            "QCFUSE_PROFILE_PATH"
        )
        self._profile_lock = threading.Lock()
        self._write_profile(
            {
                "event": "store_init",
                "pid": os.getpid(),
                "device": self.device,
                "role": "worker" if self._is_worker else "scheduler",
                "storage_mode": self.storage_mode,
                "timestamp": time.time(),
            }
        )
        if (
            self._is_worker
            and self.storage_mode in ("cpu", "gpu")
            and self.resident_preload_policy == "all"
        ):
            self._preload_resident_cache()
        # Each request uses a distinct TransferPool, whose own lock protects
        # its ring. Keep the old process-wide lock as an opt-in diagnostic
        # switch only; the default path allows independent requests to overlap.
        self._transfer_lock = (
            threading.Lock()
            if os.getenv("QCFUSE_GLOBAL_TRANSFER_LOCK") == "1"
            else None
        )

    def _preload_resident_cache(self) -> None:
        """Preload existing packed files before online request timing starts."""
        if not self.cache_root:
            return
        root = os.fspath(self.cache_root)
        if not os.path.isdir(root):
            return
        paths = sorted(
            os.path.join(directory, filename)
            for directory, _dirs, filenames in os.walk(root)
            for filename in filenames
            if filename.endswith(".bin")
        )
        max_bytes_raw = os.getenv("QCFUSE_RESIDENT_MAX_BYTES")
        max_bytes = (
            int(max_bytes_raw) if max_bytes_raw and max_bytes_raw.isdigit() else 0
        )
        total = 0
        loaded = 0
        for path in paths:
            size = os.path.getsize(path)
            if max_bytes and total + size > max_bytes:
                raise MemoryError(
                    f"resident QCFuse preload exceeds QCFUSE_RESIDENT_MAX_BYTES="
                    f"{max_bytes}: {path}"
                )
            total += transfer.preload_resident_file(
                path, self.storage_mode, self.device
            )
            loaded += 1
        self._write_profile(
            {
                "event": "storage_preload",
                "storage_mode": self.storage_mode,
                "files": loaded,
                "bytes": total,
                **transfer.resident_metrics(),
                "timestamp": time.time(),
            }
        )

    def cc_store(self) -> int:
        return 0

    def lookup(self, block_ids: List[bytes]) -> List[bool]:
        return [False] * len(block_ids)

    def lookup_on_prefix(self, block_ids: List[bytes]) -> int:
        return -1

    def lookup_on_reverse(self, block_ids: List[bytes]) -> int:
        # QCFuse uses sample-addressed cache metadata and never participates in
        # UCM's block-addressed reverse lookup protocol.
        return -1

    def prefetch(self, block_ids: List[bytes]) -> None:
        raise NotImplementedError("QCFuse uses sample-addressed prefetch")

    @staticmethod
    def _unsupported(*_args, **_kwargs):
        raise NotImplementedError("QCFuse does not implement block-addressed I/O")

    load = _unsupported
    dump = _unsupported
    load_data = _unsupported
    dump_data = _unsupported

    def wait(self, task: Task) -> None:
        if isinstance(task, QcFusePrefetchTask):
            task.wait()

    def record_qcompute_reuse(self, request_id: str, reused: bool) -> None:
        """Record the QCOMPUTE -> DO_BLEND handoff without changing tensors."""
        self._write_profile(
            {
                "event": "qcompute_reuse",
                "request_id": str(request_id),
                "reused": bool(reused),
                "timestamp": time.time(),
            }
        )

    def check(self, task: Task) -> bool:
        return isinstance(task, QcFusePrefetchTask) and task.check()

    def dump_full_cache(
        self,
        sample_dir: str,
        kv_by_layer: KVByLayer,
        num_layers: int,
        token_indices: Optional[torch.Tensor] = None,
        token_indices_by_layer: Optional[Mapping[int, torch.Tensor]] = None,
    ) -> None:
        pf.save_packed_kv(
            sample_dir,
            kv_by_layer,
            num_layers,
            token_indices=token_indices,
            token_indices_by_layer=token_indices_by_layer,
        )
        with self._meta_lock:
            self._packed_meta.pop(sample_dir, None)

    def dump_query_cache(
        self,
        sample_dir: str,
        num_digest_layers: int,
        digest_kv_by_layer: KVByLayer,
        digest_token_indices_by_layer: Mapping[int, torch.Tensor],
        critical_layers: Sequence[int],
        critical_kv_by_layer: KVByLayer,
        digest_meta: Dict,
        indices_by_method: Dict,
        digest_ratio: Optional[float] = None,
        digest_index_method: str = "kvzip",
        qcompute_end: Optional[int] = None,
        critical_token_indices: Optional[torch.Tensor] = None,
    ) -> None:
        pf.save_query_cache(
            sample_dir,
            num_digest_layers,
            digest_kv_by_layer,
            digest_token_indices_by_layer,
            critical_layers,
            critical_kv_by_layer,
            digest_meta,
            indices_by_method,
            digest_ratio=digest_ratio,
            digest_index_method=digest_index_method,
            qcompute_end=qcompute_end,
            critical_token_indices=critical_token_indices,
        )
        with self._meta_lock:
            self._query_meta.pop(sample_dir, None)

    def load_packed_meta(
        self, sample_dir: str, expected_num_layers: Optional[int] = None
    ) -> Dict:
        with self._meta_lock:
            meta = self._packed_meta.get(sample_dir)
            if meta is None:
                meta = pf.load_packed_meta(sample_dir, expected_num_layers)
                self._packed_meta[sample_dir] = meta
            elif (
                expected_num_layers is not None
                and meta["num_layers"] != expected_num_layers
            ):
                raise ValueError(
                    f"Packed KV num_layers mismatch: {meta['num_layers']} "
                    f"!= {expected_num_layers}"
                )
            return meta

    def load_query_meta(self, sample_dir: str) -> Dict:
        with self._meta_lock:
            meta = self._query_meta.get(sample_dir)
            if meta is None:
                meta = pf.load_query_meta(sample_dir)
                self._query_meta[sample_dir] = meta
            return meta

    def _load_operation(
        self, operation: _Op, request_id: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kind, section, layer_id, sample_dir = operation
        if (
            self.storage_mode in ("cpu", "gpu")
            and self.resident_preload_policy == "batch"
        ):
            filename = pf._PACKED_BIN_NAME if kind == "chunk" else pf._QUERY_BIN_NAME
            path = os.path.join(sample_dir, filename)
            reader = transfer.resident_reader(path)
            if reader is None or reader.mode != self.storage_mode:
                raise RuntimeError(
                    f"{self.storage_mode} batch was not prepared before loading {path}"
                )
        pool_key = f"request:{request_id}:{kind}"
        started = time.perf_counter()
        guard = self._transfer_lock or nullcontext()
        with guard:
            with torch.cuda.device(self.device):
                if kind == "chunk":
                    result = transfer.load_packed_layer(
                        sample_dir,
                        self.load_packed_meta(sample_dir),
                        layer_id,
                        self.device,
                        kind=pool_key,
                        wait=False,
                    )
                else:
                    result = transfer.load_query_layer(
                        sample_dir,
                        self.load_query_meta(sample_dir),
                        section,
                        layer_id,
                        self.device,
                        kind=pool_key,
                        wait=False,
                    )
        transfer_metrics = transfer.last_transfer_metrics(pool_key, self.device)
        self._write_profile(
            {
                "event": "load",
                "request_id": str(request_id),
                "kind": kind,
                "section": section,
                "layer": int(layer_id),
                "seconds": time.perf_counter() - started,
                **transfer_metrics,
                "timestamp": time.time(),
            }
        )
        return result

    def begin_requests(self, entries) -> None:
        # GPU-resident bytes need only a request-owned repair copy. Reading
        # ahead materializes up to eight full-layer clones before consumption,
        # competing with attention/selection workspace on nearly full cards.
        # Clone on the consumer stream instead; keep resident sources immutable.
        if self.storage_mode == "gpu":
            return
        # Diagnostic/compatibility escape hatch for validating batched reads.
        # The native vLLM path can use synchronous layer loads when a storage
        # backend is not safe for several concurrent CUDA transfer workers.
        disable_prefetch = os.getenv("QCFUSE_DISABLE_PREFETCH_BATCH") == "1"
        for entry in entries:
            request_id = entry.request_id
            if request_id in self._tasks:
                continue
            state = entry.state
            operations: List[_Op] = []
            critical = [int(layer) for layer in state.critical_layers or []]
            if state.blend_style == BlendStyle.QCOMPUTE:
                operations = [
                    ("query", "digest", layer, state.sample_dir_query)
                    for layer in range(int(state.qcompute_end or 0))
                ]
            elif state.blend_style in (BlendStyle.DO_BLEND, BlendStyle.DO_BLEND_FINISH):
                operations.extend(
                    ("query", "critical", layer, state.sample_dir_query)
                    for layer in critical
                )
                operations.extend(
                    ("chunk", "", layer, state.sample_dir_chunk)
                    for layer in range(state.start + 1, state.att_params.num_layers)
                )
            if not operations:
                continue
            if disable_prefetch:
                continue
            if any(operation[3] is None for operation in operations):
                raise ValueError(f"QCFuse request {request_id} has no cache path")
            try:
                configured_depth = int(os.getenv("QCFUSE_PREFETCH_DEPTH", "8"))
            except ValueError:
                configured_depth = 8
            depth = max(2, min(configured_depth, len(operations)))
            task = QcFusePrefetchTask(
                request_id, operations, self._load_operation, depth
            )
            self._tasks[request_id] = task
            if state.sample_dir_query:
                self._query_tasks[state.sample_dir_query] = task
            if state.sample_dir_chunk:
                self._chunk_tasks[state.sample_dir_chunk] = task

    def _take_prefetched(
        self, task: Optional[QcFusePrefetchTask], operation: _Op
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if task is None or operation not in task.operations:
            return None
        result = task.take(operation)
        wait_seconds = task.metrics.wait_seconds
        self._metrics.wait_seconds += wait_seconds
        task.profile_wait_seconds += wait_seconds
        task.metrics.wait_seconds = 0.0
        self._write_profile(
            {
                "event": "consume",
                "request_id": str(task.request_id),
                "kind": operation[0],
                "section": operation[1],
                "layer": int(operation[2]),
                "wait_seconds": wait_seconds,
                "timestamp": time.time(),
            }
        )
        return result

    def _write_profile(self, payload: Dict) -> None:
        if not self._profile_path:
            return
        path = os.fspath(self._profile_path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with self._profile_lock:
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def load_chunk_layer(
        self, sample_dir: str, layer_id: int, device: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        operation = ("chunk", "", int(layer_id), sample_dir)
        result = self._take_prefetched(self._chunk_tasks.get(sample_dir), operation)
        if result is not None:
            return _record_on_current_stream(result)
        device = device or self.device
        return _record_on_current_stream(
            transfer.load_packed_layer(
                sample_dir,
                self.load_packed_meta(sample_dir),
                layer_id,
                device,
            )
        )

    def load_query_layer(
        self,
        sample_dir: str,
        section: str,
        layer_id: int,
        device: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        operation = ("query", section, int(layer_id), sample_dir)
        result = self._take_prefetched(self._query_tasks.get(sample_dir), operation)
        if result is not None:
            return _record_on_current_stream(result)
        device = device or self.device
        return _record_on_current_stream(
            transfer.load_query_layer(
                sample_dir,
                self.load_query_meta(sample_dir),
                section,
                layer_id,
                device,
            )
        )

    def finish_request(self, request_id: str) -> None:
        task = self._tasks.pop(str(request_id), None)
        if task is None:
            return
        task.cancel()
        try:
            task.wait()
        finally:
            transfer_metrics = {}
            for kind in ("query", "chunk"):
                transfer_metrics_for_kind = transfer.drain_transfer_pool(
                    f"request:{request_id}:{kind}", self.device
                )
                for name, value in transfer_metrics_for_kind.items():
                    if isinstance(value, (int, float)):
                        transfer_metrics[name] = transfer_metrics.get(
                            name, 0.0
                        ) + float(value)
            self._write_profile(
                {
                    "event": "request_finish",
                    "request_id": str(request_id),
                    "read_seconds": task.metrics.read_seconds,
                    "wait_seconds": task.profile_wait_seconds
                    + task.metrics.wait_seconds,
                    "layers": int(task.metrics.layers),
                    "operations": len(task.operations),
                    **transfer_metrics,
                    "finished_at": time.time(),
                }
            )
            self._metrics.read_seconds += task.metrics.read_seconds
            self._metrics.layers += task.metrics.layers
            self._metrics.ssd_read_seconds += transfer_metrics.get(
                "ssd_read_seconds", 0.0
            )
            self._metrics.ssd_read_bytes += transfer_metrics.get("ssd_read_bytes", 0.0)
            self._metrics.resident_read_seconds += transfer_metrics.get(
                "resident_read_seconds", 0.0
            )
            self._metrics.resident_read_bytes += transfer_metrics.get(
                "resident_read_bytes", 0.0
            )
            self._metrics.resident_cpu_direct_layers += transfer_metrics.get(
                "resident_cpu_direct_layers", 0.0
            )
            self._metrics.pinned_allocation_fallbacks += transfer_metrics.get(
                "pinned_allocation_fallbacks", 0.0
            )
            self._metrics.h2d_enqueue_seconds += transfer_metrics.get(
                "h2d_enqueue_seconds", 0.0
            )
            self._metrics.h2d_bytes += transfer_metrics.get("h2d_bytes", 0.0)
            self._metrics.cuda_wait_seconds += transfer_metrics.get(
                "cuda_wait_seconds", 0.0
            )
            self._metrics.cuda_copy_seconds += transfer_metrics.get(
                "cuda_copy_seconds", 0.0
            )
            self._metrics.max_ring_depth = max(
                self._metrics.max_ring_depth,
                transfer_metrics.get("ring_depth", 0.0),
            )
            for sample_dir, candidate in list(self._query_tasks.items()):
                if candidate is task:
                    self._query_tasks.pop(sample_dir, None)
            for sample_dir, candidate in list(self._chunk_tasks.items()):
                if candidate is task:
                    self._chunk_tasks.pop(sample_dir, None)
            transfer.release_transfer_pools(f"request:{request_id}:")

    def metrics(self) -> Dict[str, float]:
        return {
            "ssd_h2d_seconds": self._metrics.read_seconds,
            "ssd_wait_seconds": self._metrics.wait_seconds,
            "ssd_layers": float(self._metrics.layers),
            "ssd_read_seconds": self._metrics.ssd_read_seconds,
            "ssd_read_bytes": self._metrics.ssd_read_bytes,
            "resident_read_seconds": self._metrics.resident_read_seconds,
            "resident_read_bytes": self._metrics.resident_read_bytes,
            "resident_cpu_direct_layers": self._metrics.resident_cpu_direct_layers,
            "pinned_allocation_fallbacks": self._metrics.pinned_allocation_fallbacks,
            "h2d_enqueue_seconds": self._metrics.h2d_enqueue_seconds,
            "h2d_bytes": self._metrics.h2d_bytes,
            "cuda_wait_seconds": self._metrics.cuda_wait_seconds,
            "cuda_copy_seconds": self._metrics.cuda_copy_seconds,
            "ring_depth": self._metrics.max_ring_depth,
        }

    def close(self) -> None:
        """Stop outstanding prefetches and release process-local resources.

        QCFuse resident files and transfer pools are process-local registries,
        so relying on interpreter teardown leaks pinned host memory and CUDA
        streams in a long-lived vLLM worker.  The method is deliberately
        idempotent because connector shutdown hooks may be called more than
        once during normal and error teardown.
        """
        if self._closed:
            return
        self._closed = True
        first_error: Optional[BaseException] = None
        for request_id in list(self._tasks):
            try:
                self.finish_request(request_id)
            except BaseException as error:  # pragma: no cover - defensive teardown
                if first_error is None:
                    first_error = error
        try:
            transfer.release_transfer_pools("request:")
            transfer.release_resident_files()
        finally:
            self._write_profile(
                {
                    "event": "store_close",
                    "storage_mode": self.storage_mode,
                    "timestamp": time.time(),
                }
            )
        if first_error is not None:
            raise first_error

    def read_chunk_layer_cpu(
        self, sample_dir: str, layer_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        meta = self.load_packed_meta(sample_dir)
        bin_path = f"{sample_dir}/{pf._PACKED_BIN_NAME}"
        key = pf.read_tensor_bytes(bin_path, pf._tensor_meta(meta, layer_id, "k"))
        value = pf.read_tensor_bytes(bin_path, pf._tensor_meta(meta, layer_id, "v"))
        return key, value
