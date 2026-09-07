import time

import pytest
import torch

from ucm.store.qcfuse.qcfuse_store import (
    QcFusePrefetchTask,
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
