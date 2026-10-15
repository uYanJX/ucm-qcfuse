import time

import pytest
import torch
from types import SimpleNamespace

from ucm.sparse.qcfuse.blend_info import BlendStyle

from ucm.store.qcfuse.qcfuse_store import (
    QcFusePrefetchTask,
    UcmQcFuseStore,
    _record_on_current_stream,
)


def test_prefetch_returns_each_layer_and_propagates_failure():
    operations = [("query", "digest", layer, "/sample") for layer in range(3)]

    def loader(operation, request_id):
        if operation[2] == 2:
            raise OSError("read failed")
        time.sleep(0.01)
        tensor = torch.tensor([operation[2]])
        return tensor, tensor

    task = QcFusePrefetchTask("request", operations, loader, depth=2)
    assert task.take(operations[0])[0].item() == 0
    assert task.take(operations[1])[0].item() == 1
    with pytest.raises(RuntimeError, match="prefetch failed"):
        task.take(operations[2])


def test_prefetched_cuda_tensors_are_adopted_by_current_stream(monkeypatch):
    stream = object()

    class FakeCudaTensor:
        is_cuda = True
        device = "cuda:0"

        def __init__(self):
            self.streams = []

        def record_stream(self, value):
            self.streams.append(value)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)
    key = FakeCudaTensor()
    value = FakeCudaTensor()

    assert _record_on_current_stream((key, value)) == (key, value)
    assert key.streams == [stream]
    assert value.streams == [stream]


def test_prefetched_readiness_is_waited_on_actual_consumer_once(monkeypatch):
    from types import SimpleNamespace

    observed = []
    event = object()
    stream = SimpleNamespace(wait_event=lambda e: observed.append(("wait", e)))

    class Tensor:
        is_cuda = True
        device = "cuda:0"
        _qcfuse_ready_event = event

        def record_stream(self, target):
            observed.append(("record", target))

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    tensors = Tensor(), Tensor()
    _record_on_current_stream(tensors)
    assert observed == [("wait", event), ("record", stream), ("record", stream)]


def test_qcfuse_store_close_releases_process_resources(monkeypatch):
    store = UcmQcFuseStore({"storage_mode": "ssd"})
    released = []
    monkeypatch.setattr(
        "ucm.store.qcfuse.qcfuse_store.transfer.release_transfer_pools",
        lambda prefix: released.append(("pools", prefix)),
    )
    monkeypatch.setattr(
        "ucm.store.qcfuse.qcfuse_store.transfer.release_resident_files",
        lambda: released.append(("resident", None)),
    )

    store.close()
    store.close()

    assert released == [("pools", "request:"), ("resident", None)]


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("phase", [BlendStyle.QCOMPUTE, BlendStyle.DO_BLEND_FINISH])
def test_gpu_resident_does_not_materialize_future_layer_clones(
    monkeypatch, batch, phase
):
    store = UcmQcFuseStore({"storage_mode": "gpu", "resident_preload_policy": "batch"})

    def unexpected_producer(*args, **kwargs):
        pytest.fail("GPU-resident cache must be copied only at consumer demand")

    monkeypatch.setattr(
        "ucm.store.qcfuse.qcfuse_store.QcFusePrefetchTask", unexpected_producer
    )
    entries = [
        SimpleNamespace(request_id=str(i), state=SimpleNamespace(blend_style=phase))
        for i in range(batch)
    ]
    store.begin_requests(entries)
    assert not store._tasks and not store._query_tasks and not store._chunk_tasks
    store.close()
