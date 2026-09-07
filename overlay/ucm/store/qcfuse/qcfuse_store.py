"""Sample-addressed storage and bounded layer prefetch for QCFuse."""

from __future__ import annotations

import threading
import time
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
    key, value = tensors
    if key.is_cuda:
        stream = torch.cuda.current_stream(key.device)
        key.record_stream(stream)
        value.record_stream(stream)
    return key, value


@dataclass
class QcFuseTransferMetrics:
    read_seconds: float = 0.0
    wait_seconds: float = 0.0
    layers: int = 0


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
        self._meta_lock = threading.Lock()
        self._packed_meta: Dict[str, Dict] = {}
        self._query_meta: Dict[str, Dict] = {}
        self._tasks: Dict[str, QcFusePrefetchTask] = {}
        self._query_tasks: Dict[str, QcFusePrefetchTask] = {}
        self._chunk_tasks: Dict[str, QcFusePrefetchTask] = {}
        self._metrics = QcFuseTransferMetrics()

    def cc_store(self) -> int:
        return 0

    def lookup(self, block_ids: List[bytes]) -> List[bool]:
        return [False] * len(block_ids)

    def lookup_on_prefix(self, block_ids: List[bytes]) -> int:
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
        pool_key = f"request:{request_id}:{kind}"
        with torch.cuda.device(self.device):
            if kind == "chunk":
                return transfer.load_packed_layer(
                    sample_dir,
                    self.load_packed_meta(sample_dir),
                    layer_id,
                    self.device,
                    kind=pool_key,
                )
            return transfer.load_query_layer(
                sample_dir,
                self.load_query_meta(sample_dir),
                section,
                layer_id,
                self.device,
                kind=pool_key,
            )

    def begin_requests(self, entries) -> None:
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
            if any(operation[3] is None for operation in operations):
                raise ValueError(f"QCFuse request {request_id} has no cache path")
            depth = max(2, len(critical))
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
        self._metrics.wait_seconds += task.metrics.wait_seconds
        task.metrics.wait_seconds = 0.0
        return result

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
            self._metrics.read_seconds += task.metrics.read_seconds
            self._metrics.layers += task.metrics.layers
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
        }

    def read_chunk_layer_cpu(
        self, sample_dir: str, layer_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        meta = self.load_packed_meta(sample_dir)
        bin_path = f"{sample_dir}/{pf._PACKED_BIN_NAME}"
        key = pf.read_tensor_bytes(bin_path, pf._tensor_meta(meta, layer_id, "k"))
        value = pf.read_tensor_bytes(bin_path, pf._tensor_meta(meta, layer_id, "v"))
        return key, value
