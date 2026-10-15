import torch

from ucm.store.qcfuse.transfer import _PersistentReader, _ResidentReader


def test_grouped_window_reads_same_bytes_with_fewer_io_calls(tmp_path):
    from ucm.store.qcfuse.transfer import TransferPool

    payload = bytes(range(128))
    path = tmp_path / "packed.bin"
    path.write_bytes(payload)

    def resolver(layer, name):
        if layer >= 7:
            raise KeyError(layer)
        offset = layer * 16 + (8 if name == "v" else 0)
        return {"offset": offset, "end": offset + 8, "nbytes": 8}

    pool = object.__new__(TransferPool)
    pool._group_layers, pool._group_max_bytes = 4, 1024
    pool._read_window = None
    pool._aggregate = {"read_window_hits": 0}
    slots, calls = [], []

    def acquire():
        slot = {}
        slots.append(slot)
        return slot, 0.0

    def read(slot, filename, offset, nbytes):
        calls.append((offset, nbytes))
        slot.update(
            block_offset=offset,
            block_nbytes=nbytes,
            bytes=Path(filename).read_bytes()[offset : offset + nbytes],
        )
        return 0.1

    from pathlib import Path

    pool._acquire_slot, pool._read_block = acquire, read
    for layer in range(7):
        slot, _, _, _ = pool._read_layer_block(str(path), layer, resolver)
        key = resolver(layer, "k")
        relative = key["offset"] - slot["block_offset"]
        assert (
            slot["bytes"][relative : relative + 16]
            == payload[layer * 16 : (layer + 1) * 16]
        )
    assert calls == [(0, 64), (64, 48)]
    assert pool._aggregate["read_window_hits"] == 5


def test_grouped_window_stops_at_byte_cap_and_nonadjacent_layers():
    from ucm.store.qcfuse.transfer import _layer_read_window

    def resolver(layer, name):
        offset = layer * 16 + (8 if name == "v" else 0)
        if layer >= 2:
            offset += 16
        return {"offset": offset, "end": offset + 8}

    assert _layer_read_window(0, resolver, 4, 1024) == (0, 32, 2)
    assert _layer_read_window(0, resolver, 4, 16) == (0, 16, 1)


def test_grouped_slot_completion_protects_last_copy_and_counts_all_copies():
    from ucm.store.qcfuse.transfer import TransferPool

    synchronized = []

    class Event:
        def __init__(self, milliseconds):
            self.milliseconds = milliseconds

        def synchronize(self):
            synchronized.append(self)

        def elapsed_time(self, end):
            return self.milliseconds

    last = Event(0)
    slot = {"event": last, "copies": [(Event(1), Event(0)), (Event(2), Event(0))]}
    pool = object.__new__(TransferPool)
    pool._aggregate = {
        "cuda_wait_seconds": 0.0,
        "cuda_copy_seconds": 0.0,
        "completed_copies": 0,
    }
    pool._complete_slot(slot)
    assert synchronized == [last]
    assert pool._aggregate["cuda_copy_seconds"] == 0.003
    assert pool._aggregate["completed_copies"] == 2
    assert slot["event"] is None and slot["copies"] == []


def test_persistent_reader_pread_fills_destination(tmp_path, monkeypatch):
    payload = bytes(range(256)) * 32
    path = tmp_path / "packed.bin"
    path.write_bytes(payload)
    monkeypatch.setenv("QCFUSE_IO_MODE", "pread")

    reader = _PersistentReader(str(path))
    try:
        destination = torch.empty(len(payload), dtype=torch.uint8)
        reader.read_into(destination, 0, len(payload))
        assert bytes(destination.tolist()) == payload
    finally:
        reader.close()


def test_persistent_reader_accepts_reused_larger_destination_view(
    tmp_path, monkeypatch
):
    payload = bytes(range(128)) * 4
    path = tmp_path / "packed.bin"
    path.write_bytes(payload)
    monkeypatch.setenv("QCFUSE_IO_MODE", "pread")

    reader = _PersistentReader(str(path))
    try:
        destination = torch.empty(len(payload) + 64, dtype=torch.uint8)
        destination_view = memoryview(destination.numpy()).cast("B")
        nbytes = len(payload) - 17
        reader.read_into(destination, 17, nbytes, destination_view)
        assert bytes(destination[:nbytes].tolist()) == payload[17:]
    finally:
        destination_view.release()
        reader.close()


def test_persistent_reader_mmap_fills_subrange(tmp_path, monkeypatch):
    payload = bytes(range(128)) * 16
    path = tmp_path / "packed.bin"
    path.write_bytes(payload)
    monkeypatch.setenv("QCFUSE_IO_MODE", "mmap")

    reader = _PersistentReader(str(path))
    try:
        destination = torch.empty(97, dtype=torch.uint8)
        reader.advise(31, destination.numel())
        reader.read_into(destination, 31, destination.numel())
        assert bytes(destination.tolist()) == payload[31 : 31 + destination.numel()]
    finally:
        reader.close()


def test_persistent_reader_coalesces_overlapping_read_ahead(tmp_path, monkeypatch):
    path = tmp_path / "packed.bin"
    path.write_bytes(bytes(256))
    monkeypatch.setenv("QCFUSE_IO_MODE", "pread")

    reader = _PersistentReader(str(path))
    try:
        reader.advise(16, 32)
        assert reader._advised_until == 48
        # The second window overlaps the first one; only its uncovered tail
        # should advance the advice cursor.
        reader.advise(32, 32)
        assert reader._advised_until == 64
        reader.advise(20, 8)
        assert reader._advised_until == 64
    finally:
        reader.close()


def test_resident_reader_reuses_cpu_byte_view():
    source = torch.arange(256, dtype=torch.uint8)
    reader = _ResidentReader(source, "cpu", pinned=False)
    try:
        view_id = id(reader._byte_view)
        destination = torch.empty(32, dtype=torch.uint8)
        reader.read_into(destination, 17, destination.numel())
        assert bytes(destination.tolist()) == bytes(range(17, 49))
        assert id(reader._byte_view) == view_id
    finally:
        reader.close()
    assert reader._byte_view is None


def test_resident_batch_replaces_old_bytes_and_reuses_same_batch(tmp_path, monkeypatch):
    from ucm.store.qcfuse import transfer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(bytes(range(32)))
    second.write_bytes(bytes(range(64)))
    transfer.release_resident_files()
    try:
        one = transfer.prepare_resident_batch(None, [str(first)], "cpu")
        old = transfer.resident_reader(str(first))
        assert one["resident_cpu_bytes"] == 32
        assert transfer.prepare_resident_batch(None, [str(first)], "cpu")[
            "resident_batch_reused"
        ]
        assert transfer.resident_reader(str(first)) is old
        two = transfer.prepare_resident_batch(None, [str(second)], "cpu")
        assert two["resident_files"] == 1 and two["resident_cpu_bytes"] == 64
        assert transfer.resident_reader(str(first)) is None and old._byte_view is None
        assert (
            bytes(transfer.resident_reader(str(second)).tensor.tolist())
            == second.read_bytes()
        )
        assert transfer.prepare_resident_batch(None, [], "cpu")["resident_files"] == 0
    finally:
        transfer.release_resident_files()


def test_handled_registration_failure_clears_runtime_error(monkeypatch):
    from types import SimpleNamespace
    from ucm.store.qcfuse import transfer

    calls = []
    cudart = SimpleNamespace(
        cudaHostRegister=lambda *_: (1,),
        cudaGetLastError=lambda: calls.append("clear") or 1,
    )
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    assert (
        transfer._try_register_host_memory(torch.empty(16, dtype=torch.uint8)) is False
    )
    assert calls == ["clear"]
    cudart.cudaHostRegister = lambda *_: (0,)
    assert (
        transfer._try_register_host_memory(torch.empty(16, dtype=torch.uint8)) is True
    )
    assert calls == ["clear"]
