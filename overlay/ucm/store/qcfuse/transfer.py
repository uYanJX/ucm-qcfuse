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
"""CUDA transfer helper for QCFuse packed KV: pinned host buffer + stream copy.

Faithful port of ``KVSSDManager``'s online load path (``_ensure_transfer_pool``,
``_read_packed_block_to_slot``, ``_copy_packed_layer_from_block_to_gpu``), with
the SGLang global pools removed.  Each pool is keyed by a caller-chosen ``kind``
so query and chunk prefetch never share a mutable pinned source buffer.

This module is CUDA-only: importing it on a CPU-only build is fine, but any call
to :func:`TransferPool.load_layer` will raise because ``torch.cuda`` is absent.
"""

import threading
from typing import Dict

import torch

from ucm.store.qcfuse import packed_format as pf


class TransferPool:
    """Reusable pinned host buffer + CUDA stream for one transfer kind."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.lock = threading.Lock()
        self.stream = torch.cuda.Stream(device=device)
        self._slot: Dict = {}

    def _ensure_byte_buf(self, name: str, nbytes: int) -> None:
        buf_name = f"pin_buf_{name}"
        cap_name = f"byte_capacity_{name}"
        if (
            self._slot.get(buf_name) is not None
            and self._slot.get(cap_name, 0) >= nbytes
        ):
            return
        capacity = max(nbytes, 16384)
        self._slot[buf_name] = torch.empty(capacity, dtype=torch.uint8, pin_memory=True)
        self._slot[cap_name] = capacity

    @staticmethod
    def _clear_slot_metadata(slot: Dict) -> None:
        for key in list(slot.keys()):
            if key.startswith("pin_buf_") or key.startswith("byte_capacity_"):
                continue
            del slot[key]

    def _read_block(self, bin_path: str, block_offset: int, block_nbytes: int) -> None:
        self._ensure_byte_buf("block", block_nbytes)
        with open(bin_path, "rb", buffering=0) as f:
            f.seek(block_offset)
            pf._read_exact_into(f, self._slot["pin_buf_block"], block_nbytes)
        self._slot["block_offset"] = block_offset
        self._slot["block_nbytes"] = block_nbytes

    def _copy_layer_from_block(
        self, layer_id: int, meta_item_resolver
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        block_offset = self._slot["block_offset"]

        def _view(name: str):
            item = meta_item_resolver(layer_id, name)
            rel = item["offset"] - block_offset
            if rel < 0 or rel + item["nbytes"] > self._slot["block_nbytes"]:
                raise ValueError(
                    f"Packed KV layer {layer_id}.{name} is outside pinned block"
                )
            return (
                self._slot["pin_buf_block"][rel : rel + item["nbytes"]]
                .view(item["dtype"])
                .view(item["shape"])
            ), item

        k_pin, k_item = _view("k")
        v_pin, v_item = _view("v")

        with torch.cuda.stream(self.stream):
            k_gpu = torch.empty(
                k_item["shape"], dtype=k_item["dtype"], device=self.device
            )
            v_gpu = torch.empty(
                v_item["shape"], dtype=v_item["dtype"], device=self.device
            )
            k_gpu.copy_(k_pin, non_blocking=True)
            v_gpu.copy_(v_pin, non_blocking=True)
        self.stream.synchronize()
        return k_gpu, v_gpu

    def load_layer(
        self,
        bin_path: str,
        meta: Dict,
        layer_id: int,
        meta_item_resolver,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Read one layer's contiguous K/V block and copy it to GPU."""
        with self.lock:
            self._clear_slot_metadata(self._slot)
            first_k = meta_item_resolver(layer_id, "k")
            last_v = meta_item_resolver(layer_id, "v")
            block_offset = first_k["offset"]
            block_end = last_v["end"]
            self._read_block(bin_path, block_offset, block_end - block_offset)
            return self._copy_layer_from_block(layer_id, meta_item_resolver)


# Pool cache keyed by (kind, device), mirroring KVSSDManager._transfer_pools.
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


def release_transfer_pools(kind_prefix: str) -> None:
    """Drop reusable buffers belonging to a completed request."""
    with _POOLS_LOCK:
        for key in list(_POOLS):
            if str(key[0]).startswith(kind_prefix):
                _POOLS.pop(key, None)


def load_packed_layer(
    sample_dir: str,
    meta: Dict,
    layer_id: int,
    device: str,
    kind: str = "chunk",
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Load a full-context packed KV layer to GPU (``kv_packed.bin``)."""
    pool = get_transfer_pool(kind, device)
    bin_path = _bin_path(sample_dir, pf._PACKED_BIN_NAME)
    resolver = lambda lid, name: pf._tensor_meta(meta, lid, name)
    return pool.load_layer(bin_path, meta, layer_id, resolver)


def load_query_layer(
    sample_dir: str,
    query_meta: Dict,
    section: str,
    layer_id: int,
    device: str,
    kind: str = "query",
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Load one query-cache layer (digest or critical) to GPU."""
    pool = get_transfer_pool(kind, device)
    bin_path = _bin_path(sample_dir, pf._QUERY_BIN_NAME)
    resolver = lambda lid, name: pf.section_tensor_meta(query_meta, section, lid, name)
    return pool.load_layer(bin_path, query_meta, layer_id, resolver)


def _bin_path(sample_dir: str, name: str) -> str:
    import os

    return os.path.join(sample_dir, name)
