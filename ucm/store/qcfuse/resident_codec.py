"""Optional, lossless GPU residency for the unchanged packed cache format."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import torch

from ucm.store.qcfuse import packed_format as pf


def item_key(item):
    return (item["offset"], item["nbytes"], tuple(item["shape"]), str(item["dtype"]))


def _nvcomp_module():
    try:
        from nvidia import nvcomp
    except ImportError as exc:
        raise RuntimeError(
            "Lossless GPU residency requires the optional nvidia-nvcomp package "
            "matching the CUDA runtime (nvCOMP 6 or newer)"
        ) from exc
    return nvcomp


def _codec(device, stream):
    return _nvcomp_module().Codec(
        algorithm="ANS",
        data_type="<f2",
        device_id=device.index,
        cuda_stream=stream.cuda_stream,
    )


class CompressedResidentReader:
    """Store immutable byte payloads; decode only into request-owned tensors.

    BF16 words are reinterpreted as 16-bit symbols for the lossless codec.
    No floating-point conversion or quantization is performed. Each caller
    thread/stream gets its own mutable codec and preconfigured scratch state.
    """

    mode = "gpu"
    pinned = False
    registered = False
    codec = "ans_f16_bits"

    def __init__(self, path, device):
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.logical_size = os.path.getsize(path)
        self._entries = {}
        self._local = threading.local()
        self.size = 0
        parent = os.fspath(Path(path).parent)
        if Path(path).name == pf._PACKED_BIN_NAME:
            meta = pf.load_packed_meta(parent)
            items = [
                pf._tensor_meta(meta, layer, name)
                for layer in range(meta["num_layers"])
                for name in ("k", "v")
            ]
        elif Path(path).name == pf._QUERY_BIN_NAME:
            meta = pf.load_query_meta(parent)
            items = [
                pf.section_tensor_meta(meta, section, int(layer), name)
                for section in ("digest", "critical")
                for layer in meta[section]["layers"]
                for name in ("k", "v")
            ]
        else:
            raise ValueError("GPU compression requires a validated packed KV file")
        stream = torch.cuda.current_stream(self.device)
        encoder = _codec(self.device, stream)
        configs = {}
        nvcomp = _nvcomp_module()
        try:
            with open(path, "rb", buffering=0) as source:
                for item in items:
                    if item["dtype"] != torch.bfloat16:
                        raise ValueError("ans_f16_bits residency requires BF16 KV")
                    if not item["nbytes"]:
                        continue
                    source.seek(item["offset"])
                    data = source.read(item["nbytes"])
                    if len(data) != item["nbytes"]:
                        raise EOFError("Incomplete lossless GPU preload")
                    host = torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy())
                    raw = host.to(self.device)
                    config = configs.get(item["nbytes"])
                    if config is None:
                        config = encoder.compression_config(item["nbytes"])
                        configs[item["nbytes"]] = config
                    encoded = encoder.encode(
                        nvcomp.as_array(
                            raw.view(torch.float16), cuda_stream=stream.cuda_stream
                        ),
                        compression_config=config,
                    )
                    # Preload runs outside online timing. Compact foreign
                    # worst-case capacity before retaining the resident bytes.
                    stream.synchronize()
                    encoded_view = torch.from_dlpack(encoded).view(torch.uint8)
                    payload = encoded_view.clone()
                    stream.synchronize()
                    self._entries[item_key(item)] = payload
                    self.size += payload.numel()
                    del encoded_view, encoded, raw, host, data
            # Prepare the model thread's decode state before online timing.
            for nbytes in configs:
                self._decoder(nbytes)
        except BaseException:
            self.close()
            raise

    def _decoder(self, nbytes):
        stream = torch.cuda.current_stream(self.device)
        decoders = getattr(self._local, "decoders", None)
        if decoders is None:
            decoders = self._local.decoders = {}
        decoder = decoders.get(stream.cuda_stream)
        if decoder is None:
            decoder = (_codec(self.device, stream), {})
            decoders[stream.cuda_stream] = decoder
        codec, configs = decoder
        config = configs.get(nbytes)
        if config is None:
            config = codec.decompression_config(codec.compression_config(nbytes))
            configs[nbytes] = config
        return codec, config, stream

    def owned_tensor(self, item):
        if not item["nbytes"]:
            return torch.empty(item["shape"], dtype=item["dtype"], device=self.device)
        payload = self._entries[item_key(item)]
        output = torch.empty(item["nbytes"], dtype=torch.uint8, device=self.device)
        codec, config, stream = self._decoder(item["nbytes"])
        nvcomp = _nvcomp_module()
        codec.decode(
            nvcomp.as_array(payload, cuda_stream=stream.cuda_stream),
            out=nvcomp.as_array(output, cuda_stream=stream.cuda_stream),
            decompression_config=config,
        )
        return output.view(item["dtype"]).view(item["shape"])

    def close(self):
        self._entries.clear()
        self._local = threading.local()


def validate_query_alias(query_path, source_path):
    """Prove the query file is exactly a view of this source before preload.

    Pairing is explicit. Positions refer to the stripped full-cache rows;
    compare their bytes as integers so all BF16 bit patterns are preserved.
    No source or query payload is read during subsequent online requests.
    """
    source_meta = pf.load_packed_meta(os.fspath(Path(source_path).parent))
    query_meta = pf.load_query_meta(os.fspath(Path(query_path).parent))
    positions_by_layer = query_meta["metadata"]["context_positions_by_layer"]
    total_tokens = query_meta["metadata"]["total_tokens"]
    # Digest positions cover prefix/context only; the full source and critical
    # section also include the original query. Augmented zip rows have already
    # been stripped using the source's token_indices.
    full_tokens = total_tokens
    ranges = query_meta["metadata"].get("orig_chunk_ranges")
    if ranges:
        previous = 0
        for span in ranges:
            if (
                not isinstance(span, (list, tuple))
                or len(span) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) for x in span)
                or span[0] != previous
                or span[1] < span[0]
            ):
                raise ValueError("Query alias has invalid original chunk ranges")
            previous = span[1]
        if ranges[-1][0] != total_tokens:
            raise ValueError(
                "Query alias has a different full token count or context boundary"
            )
        full_tokens = ranges[-1][1]
    bindings = []
    with open(source_path, "rb") as source, open(query_path, "rb") as query:
        for section in ("digest", "critical"):
            for layer in query_meta[section]["layers"]:
                layer = int(layer)
                positions = positions_by_layer[layer] if section == "digest" else None
                if positions is not None and any(
                    isinstance(p, bool)
                    or not isinstance(p, int)
                    or p < 0
                    or p >= total_tokens
                    for p in positions
                ):
                    raise ValueError("Query alias positions are outside the full cache")
                for name in ("k", "v"):
                    full = pf._tensor_meta(source_meta, layer, name)
                    part = pf.section_tensor_meta(query_meta, section, layer, name)
                    if (
                        full["dtype"] != torch.bfloat16
                        or part["dtype"] != full["dtype"]
                    ):
                        raise ValueError(
                            "GPU query alias requires matching BF16 tensors"
                        )
                    if full["shape"][0] != full_tokens:
                        raise ValueError("Query alias has a different full token count")
                    expected_shape = (
                        [len(positions), *full["shape"][1:]]
                        if positions is not None
                        else list(full["shape"])
                    )
                    if list(part["shape"]) != expected_shape:
                        raise ValueError("Query alias has an incompatible tensor shape")
                    source.seek(full["offset"])
                    raw = source.read(full["nbytes"])
                    if len(raw) != full["nbytes"]:
                        raise EOFError("Incomplete query alias source")
                    expected = np.frombuffer(raw, dtype=np.uint16).reshape(
                        full["shape"]
                    )
                    if positions is not None:
                        expected = expected[np.asarray(positions, dtype=np.int64)]
                    query.seek(part["offset"])
                    actual = query.read(part["nbytes"])
                    if expected.tobytes() != actual:
                        raise ValueError(
                            "Query KV bytes do not match the explicit source"
                        )
                    bindings.append((part, full, positions))
    return bindings


class QueryAliasReader:
    """Materialize query KV entirely from an immutable GPU source."""

    mode = "gpu"
    pinned = False
    registered = False
    codec = None

    def __init__(self, path, source_path, source_reader, bindings, device):
        self.source_path = source_path
        self.source = source_reader
        self.logical_size = os.path.getsize(path)
        self._entries = {}
        index_cache = {}
        for part, full, positions in bindings:
            indices = None
            if positions is not None:
                key = tuple(positions)
                indices = index_cache.get(key)
                if indices is None:
                    indices = torch.tensor(positions, dtype=torch.long, device=device)
                    index_cache[key] = indices
            self._entries[item_key(part)] = full, indices
        self.size = sum(t.numel() * t.element_size() for t in index_cache.values())

    def owned_tensor(self, item):
        full, indices = self._entries[item_key(item)]
        tensor = self.source.owned_tensor(full)
        return tensor.index_select(0, indices) if indices is not None else tensor

    def close(self):
        self._entries.clear()
        self.source = None
