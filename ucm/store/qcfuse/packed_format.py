# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""Self-describing packed KV file format for QCFuse (pure Python + torch).

This is the on-disk contract shared between the offline KVCOMPUTE pass and the
online QCOMPUTE / DO_BLEND passes.  It is a faithful port of the file layout in
``sglang/srt/utils/kv_ssd_manager.py``, decoupled from SGLang's global pools so
it can be unit-tested on CPU.

Layout
------
Two self-describing file pairs live under one sample directory:

* ``kv_packed.bin`` / ``kv_packed_meta.json``      -- full-context K/V, one
  layer after another, each layer writing ``k`` then ``v``.
* ``query_packed.bin`` / ``query_packed_meta.json`` -- the QCOMPUTE cache:
  materialized digest K/V for every digest layer, then the raw critical-layer
  K/V, in that exact order.

Each ``*.bin`` is a flat byte stream.  The companion ``*.json`` records, per
tensor, its ``offset`` (bytes from file start), ``nbytes``, ``shape`` and
``dtype`` (a compact name like ``"F16"``).  Loaders validate that offsets are
contiguous and that ``sum(nbytes) == filesize`` before reading.
"""

import json
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

# --------------------------------------------------------------------------- #
#  Constants / format tags
# --------------------------------------------------------------------------- #
PACKED_KV_FORMAT = "ucm_qcfuse_kv_packed_v4"
QUERY_CACHE_FORMAT = "ucm_qcfuse_query_cache_v4"
KEY_ENCODING = "raw_pre_rope"
OFFLINE_ATTENTION_LAYOUT = "independent_chunks_local_positions"
_PACKED_BIN_NAME = "kv_packed.bin"
_PACKED_META_NAME = "kv_packed_meta.json"
_QUERY_BIN_NAME = "query_packed.bin"
_QUERY_META_NAME = "query_packed_meta.json"

_PACKED_DTYPE_MAP: Dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}
for _packed_name, _torch_name in (
    ("U16", "uint16"),
    ("U32", "uint32"),
    ("U64", "uint64"),
):
    if hasattr(torch, _torch_name):
        _PACKED_DTYPE_MAP[_packed_name] = getattr(torch, _torch_name)

_PACKED_DTYPE_NAMES: Dict[torch.dtype, str] = {
    v: k for k, v in _PACKED_DTYPE_MAP.items()
}

# A layer's KV tensors, keyed by layer id: ``{layer_id: (k, v)}``.
KVByLayer = Mapping[int, Tuple[torch.Tensor, torch.Tensor]]


# --------------------------------------------------------------------------- #
#  dtype helpers
# --------------------------------------------------------------------------- #
def dtype_to_packed_name(dtype: torch.dtype) -> str:
    if dtype not in _PACKED_DTYPE_NAMES:
        raise ValueError(f"Unsupported packed KV dtype: {dtype}")
    return _PACKED_DTYPE_NAMES[dtype]


def packed_name_to_dtype(name: str) -> torch.dtype:
    if name not in _PACKED_DTYPE_MAP:
        raise ValueError(f"Unsupported packed KV dtype name: {name}")
    return _PACKED_DTYPE_MAP[name]


def _shape_numel(shape: Sequence[int]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def tensor_nbytes(shape: Sequence[int], dtype: torch.dtype) -> int:
    return _shape_numel(shape) * torch.empty((), dtype=dtype).element_size()


def _write_tensor_bytes(f, tensor: torch.Tensor) -> None:
    """Write a tensor's raw bytes (device-agnostic; callers pass CPU tensors)."""
    byte_view = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    if byte_view.numel() == 0:
        return
    f.write(memoryview(byte_view.numpy()))


def _read_exact_into(f, tensor: torch.Tensor, nbytes: int) -> None:
    view = memoryview(tensor[:nbytes].numpy())
    total = 0
    while total < nbytes:
        n = f.readinto(view[total:])
        if n is None:
            continue
        if n == 0:
            raise EOFError("Unexpected EOF while reading packed KV payload")
        total += n


def _select_tokens(
    tensor: torch.Tensor, token_indices: Optional[torch.Tensor]
) -> torch.Tensor:
    """Apply ``index_select(0, token_indices)`` when indices are given."""
    if token_indices is None:
        return tensor
    idx = torch.as_tensor(token_indices, dtype=torch.long, device=tensor.device)
    return tensor.index_select(0, idx)


# --------------------------------------------------------------------------- #
#  Offline: write
# --------------------------------------------------------------------------- #
def _write_layer(
    f, offset: int, layer_id: int, k: torch.Tensor, v: torch.Tensor
) -> Tuple[Dict, int]:
    if k is None or v is None:
        raise ValueError(f"Missing KV tensor for layer {layer_id}")
    layer_meta: Dict = {}
    for name, tensor in (("k", k), ("v", v)):
        tensor_cpu = tensor.detach().contiguous().cpu()
        shape = list(tensor_cpu.shape)
        dtype_name = dtype_to_packed_name(tensor_cpu.dtype)
        nbytes = tensor_nbytes(shape, tensor_cpu.dtype)
        layer_meta[name] = {
            "offset": offset,
            "nbytes": nbytes,
            "shape": shape,
            "dtype": dtype_name,
        }
        _write_tensor_bytes(f, tensor_cpu)
        offset += nbytes
        del tensor_cpu
    return layer_meta, offset


def save_packed_kv(
    sample_dir: str,
    kv_by_layer: KVByLayer,
    num_layers: int,
    token_indices: Optional[torch.Tensor] = None,
    token_indices_by_layer: Optional[Mapping[int, torch.Tensor]] = None,
) -> None:
    """Save full-context K/V as one ``kv_packed.bin`` + meta file per sample.

    Args:
        sample_dir: directory to write the sample's cache into (created if needed).
        kv_by_layer: ``{layer_id: (k, v)}`` for every layer ``0 .. num_layers-1``.
        num_layers: total layer count.
        token_indices: optional global token selection applied to every layer.
        token_indices_by_layer: optional per-layer token selection, overrides
            ``token_indices`` for the layers it names.
    """
    os.makedirs(sample_dir, exist_ok=True)
    bin_path = os.path.join(sample_dir, _PACKED_BIN_NAME)
    meta_path = os.path.join(sample_dir, _PACKED_META_NAME)
    meta = {
        "format": PACKED_KV_FORMAT,
        "key_encoding": KEY_ENCODING,
        "offline_attention_layout": OFFLINE_ATTENTION_LAYOUT,
        "num_layers": int(num_layers),
        "layers": {},
    }
    if token_indices is not None:
        # Persist the token selection so the DO_BLEND pass can reconstruct the
        # augmented->stripped RoPE delta (see blender._apply_rotary_delta).
        meta["token_indices"] = [
            int(x) for x in torch.as_tensor(token_indices).tolist()
        ]

    offset = 0
    with open(bin_path, "wb") as f:
        for layer_id in range(int(num_layers)):
            full_k, full_v = kv_by_layer[layer_id]

            layer_indices = None
            if (
                token_indices_by_layer is not None
                and layer_id in token_indices_by_layer
            ):
                layer_indices = token_indices_by_layer[layer_id]
            elif token_indices is not None:
                layer_indices = token_indices

            full_k = _select_tokens(full_k, layer_indices)
            full_v = _select_tokens(full_v, layer_indices)

            layer_meta, offset = _write_layer(f, offset, layer_id, full_k, full_v)
            meta["layers"][str(layer_id)] = layer_meta

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def save_query_cache(
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
    """Save the QCOMPUTE cache: digest K/V then critical-layer K/V.

    Args:
        sample_dir: directory to write the sample's query cache into.
        num_digest_layers: number of layers whose digest K/V is materialized.
        digest_kv_by_layer: ``{layer_id: (k, v)}`` full context per digest layer;
            only ``digest_token_indices_by_layer[layer_id]`` tokens are kept.
        digest_token_indices_by_layer: per-layer selection for the digest.
        critical_layers: layer ids whose *raw* chunk K/V is stored.
        critical_kv_by_layer: ``{layer_id: (k, v)}`` full context per critical layer.
        digest_meta / indices_by_method: JSON payloads produced by the digest
            index manager (``metadata`` + ``indices_by_method``).
        critical_token_indices: optional token selection applied to critical K/V.
    """
    os.makedirs(sample_dir, exist_ok=True)
    bin_path = os.path.join(sample_dir, _QUERY_BIN_NAME)
    meta_path = os.path.join(sample_dir, _QUERY_META_NAME)

    critical_layers = [int(x) for x in (critical_layers or [])]
    qcompute_end = int(qcompute_end if qcompute_end is not None else num_digest_layers)

    query_meta = {
        "format": QUERY_CACHE_FORMAT,
        "key_encoding": KEY_ENCODING,
        "offline_attention_layout": OFFLINE_ATTENTION_LAYOUT,
        "digest_index_version": digest_meta.get("digest_index_version"),
        "available_methods": digest_meta.get(
            "available_methods", [digest_index_method]
        ),
        "num_layers": int(num_digest_layers),
        "digest_ratio": (
            digest_ratio
            if digest_ratio is not None
            else digest_meta.get("digest_ratio")
        ),
        "digest_index_method": digest_index_method,
        "critical_layers": critical_layers,
        "qcompute_end": qcompute_end,
        "materialized_digest": True,
        "metadata": digest_meta,
        "indices_by_method": indices_by_method,
        "digest": {"num_layers": int(num_digest_layers), "layers": {}},
        "critical": {"layer_ids": critical_layers, "layers": {}},
    }

    offset = 0
    with open(bin_path, "wb") as f:
        for layer_id in range(int(num_digest_layers)):
            full_k, full_v = digest_kv_by_layer[layer_id]
            indices = digest_token_indices_by_layer[layer_id]
            full_k = _select_tokens(full_k, indices)
            full_v = _select_tokens(full_v, indices)
            layer_meta, offset = _write_layer(f, offset, layer_id, full_k, full_v)
            query_meta["digest"]["layers"][str(layer_id)] = layer_meta

        for layer_id in critical_layers:
            full_k, full_v = critical_kv_by_layer[layer_id]
            full_k = _select_tokens(full_k, critical_token_indices)
            full_v = _select_tokens(full_v, critical_token_indices)
            layer_meta, offset = _write_layer(f, offset, layer_id, full_k, full_v)
            query_meta["critical"]["layers"][str(layer_id)] = layer_meta

    with open(meta_path, "w") as f:
        json.dump(query_meta, f, indent=2)


# --------------------------------------------------------------------------- #
#  Online: meta load + validation
# --------------------------------------------------------------------------- #
def _validate_tensor_item(item, file_size, meta_path, label, expected_offset) -> int:
    if not isinstance(item, dict):
        raise ValueError(f"Missing packed KV metadata for {label}")
    offset = item.get("offset")
    nbytes = item.get("nbytes")
    shape = item.get("shape")
    dtype_name = item.get("dtype")
    if (
        not isinstance(offset, int)
        or not isinstance(nbytes, int)
        or offset < 0
        or nbytes < 0
        or offset + nbytes > file_size
    ):
        raise ValueError(f"Bad byte range for {label} in {meta_path}")
    if offset != expected_offset:
        raise ValueError(
            f"Non-contiguous offset for {label}: {offset} != {expected_offset}"
        )
    if not isinstance(shape, list) or not all(
        isinstance(dim, int) and dim >= 0 for dim in shape
    ):
        raise ValueError(f"Bad shape for {label}: {shape}")
    dtype = packed_name_to_dtype(dtype_name)
    expected_nbytes = tensor_nbytes(shape, dtype)
    if nbytes != expected_nbytes:
        raise ValueError(
            f"Bad nbytes for {label} in {meta_path}: {nbytes} != {expected_nbytes}"
        )
    return expected_offset + nbytes


def _validate_layer(layers, layer_id, file_size, meta_path, expected_offset) -> int:
    """Validate one layer inside an already-resolved ``{layer_id: {k,v}}`` dict."""
    if not isinstance(layers, dict):
        raise ValueError(f"Bad layers metadata in {meta_path}")
    layer = layers.get(str(layer_id))
    if not isinstance(layer, dict):
        raise ValueError(f"Missing metadata for layer {layer_id}")
    for name in ("k", "v"):
        expected_offset = _validate_tensor_item(
            layer.get(name),
            file_size,
            meta_path,
            f"layer {layer_id}.{name}",
            expected_offset,
        )
    return expected_offset


def load_packed_meta(
    sample_dir: str, expected_num_layers: Optional[int] = None
) -> Dict:
    """Load and validate ``kv_packed_meta.json`` (full-context cache)."""
    meta_path = os.path.join(sample_dir, _PACKED_META_NAME)
    bin_path = os.path.join(sample_dir, _PACKED_BIN_NAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing packed KV metadata: {meta_path}")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Missing packed KV data: {bin_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)
    if meta.get("format") != PACKED_KV_FORMAT:
        raise ValueError(
            f"Unsupported packed KV format: {meta.get('format')}; "
            "regenerate the QCFuse cache"
        )
    if meta.get("key_encoding") != KEY_ENCODING:
        raise ValueError("QCFuse packed cache does not contain raw pre-RoPE keys")
    if meta.get("offline_attention_layout") != OFFLINE_ATTENTION_LAYOUT:
        raise ValueError("QCFuse packed cache has an incompatible attention layout")

    num_layers = meta.get("num_layers")
    if not isinstance(num_layers, int) or num_layers < 0:
        raise ValueError(f"Bad packed KV num_layers: {num_layers}")
    if expected_num_layers is not None and num_layers != expected_num_layers:
        raise ValueError(
            f"Packed KV num_layers mismatch: {num_layers} != {expected_num_layers}"
        )

    layers = meta.get("layers")
    if not isinstance(layers, dict):
        raise ValueError(f"Bad packed KV layers metadata")

    file_size = os.path.getsize(bin_path)
    expected_offset = 0
    for layer_id in range(num_layers):
        expected_offset = _validate_layer(
            layers, layer_id, file_size, meta_path, expected_offset
        )
    if expected_offset != file_size:
        raise ValueError(f"Packed KV size mismatch: {file_size} != {expected_offset}")
    return meta


def load_query_meta(sample_dir: str) -> Dict:
    """Load and validate ``query_packed_meta.json`` (QCOMPUTE cache)."""
    meta_path = os.path.join(sample_dir, _QUERY_META_NAME)
    bin_path = os.path.join(sample_dir, _QUERY_BIN_NAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing query cache metadata: {meta_path}")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Missing query cache data: {bin_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)
    if meta.get("format") != QUERY_CACHE_FORMAT:
        raise ValueError(
            f"Unsupported query cache format: {meta.get('format')}; "
            "regenerate the QCFuse cache"
        )
    if meta.get("key_encoding") != KEY_ENCODING:
        raise ValueError("QCFuse query cache does not contain raw pre-RoPE keys")
    if meta.get("offline_attention_layout") != OFFLINE_ATTENTION_LAYOUT:
        raise ValueError("QCFuse query cache has an incompatible attention layout")

    digest = meta.get("digest")
    critical = meta.get("critical")
    if not isinstance(digest, dict) or not isinstance(critical, dict):
        raise ValueError(f"Bad query cache sections")
    num_digest_layers = digest.get("num_layers")
    if not isinstance(num_digest_layers, int) or num_digest_layers < 0:
        raise ValueError(f"Bad query cache digest num_layers: {num_digest_layers}")
    critical_layer_ids = critical.get("layer_ids", [])
    if not isinstance(critical_layer_ids, list):
        raise ValueError(f"Bad query cache critical layer_ids")

    file_size = os.path.getsize(bin_path)
    expected_offset = 0
    for layer_id in range(num_digest_layers):
        expected_offset = _validate_layer(
            digest.get("layers"), layer_id, file_size, meta_path, expected_offset
        )
    for layer_id in critical_layer_ids:
        expected_offset = _validate_layer(
            critical.get("layers"), int(layer_id), file_size, meta_path, expected_offset
        )
    if expected_offset != file_size:
        raise ValueError(f"Query cache size mismatch: {file_size} != {expected_offset}")
    return meta


# --------------------------------------------------------------------------- #
#  Online: byte-range read to a CPU tensor (device transfer lives in transfer.py)
# --------------------------------------------------------------------------- #
def _tensor_meta(meta: Dict, layer_id: int, name: str) -> Dict:
    item = meta["layers"][str(layer_id)][name]
    return {
        "offset": item["offset"],
        "end": item["offset"] + item["nbytes"],
        "nbytes": item["nbytes"],
        "shape": tuple(item["shape"]),
        "dtype": packed_name_to_dtype(item["dtype"]),
    }


def read_tensor_bytes(bin_path: str, item: Dict) -> torch.Tensor:
    """Read one tensor's raw bytes from ``bin_path`` into a fresh CPU tensor.

    ``item`` must be a *resolved* metadata dict (as returned by
    :func:`_tensor_meta` / :func:`section_tensor_meta`), i.e. its ``dtype`` is
    already a ``torch.dtype``.
    """
    shape = tuple(item["shape"])
    dtype = item["dtype"]
    nbytes = item["nbytes"]
    buf = torch.empty(nbytes, dtype=torch.uint8)
    with open(bin_path, "rb", buffering=0) as f:
        f.seek(item["offset"])
        _read_exact_into(f, buf, nbytes)
    return buf.view(dtype).view(shape)


def section_tensor_meta(meta: Dict, section: str, layer_id: int, name: str) -> Dict:
    """Resolve a tensor's byte-range metadata inside a query-cache section."""
    item = meta[section]["layers"][str(layer_id)][name]
    return {
        "offset": item["offset"],
        "end": item["offset"] + item["nbytes"],
        "nbytes": item["nbytes"],
        "shape": tuple(item["shape"]),
        "dtype": packed_name_to_dtype(item["dtype"]),
    }
