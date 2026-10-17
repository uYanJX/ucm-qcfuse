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
"""Guarded Hopper attention dispatch; imports the CUDA kernel only on use."""

import torch


def _unsupported_reason(q, k, v, out, positions, q_lengths, kv_lengths, window, cap):
    if q.device.type != "cuda" or getattr(torch.version, "hip", None) is not None:
        return "device"
    if torch.cuda.get_device_capability(q.device)[0] != 9:
        return "device_capability"
    if window != -1 or cap != 0.0:
        return "mask_or_logit_cap"
    if q.shape[1:] != (64, 128) or k.shape[1:] != (8, 128):
        return "head_shape"
    if v.shape != k.shape or out.shape != q.shape:
        return "tensor_shape"
    if any(t.dtype != torch.bfloat16 or t.device != q.device for t in (q, k, v, out)):
        return "dtype_or_device"
    if (
        not q_lengths
        or len(q_lengths) != len(kv_lengths)
        or any(n < 0 for n in q_lengths)
        or max(q_lengths) <= 0
        or any(n <= 0 for n in kv_lengths)
        or sum(q_lengths) > q.shape[0]
        or sum(kv_lengths) > k.shape[0]
    ):
        return "empty_or_unknown_KV_layout"
    if (
        positions.device != q.device
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.ndim != 1
        or positions.stride(0) != 1
        or positions.numel() < sum(q_lengths)
    ):
        return "query_positions"
    if (
        any(
            t.stride(-1) != 1
            or t.stride(1) != 128
            or t.stride(0) < t.shape[1] * 128
            or t.data_ptr() % 16
            or (t.stride(0) * t.element_size()) % 16
            for t in (q, k, v)
        )
        or out.stride(-1) != 1
    ):
        return "descriptor_layout_or_alignment"
    return None


def try_tma_positions_attention(
    q,
    k,
    v,
    out,
    q_starts,
    q_lens,
    kv_starts,
    kv_lens,
    positions,
    q_lengths,
    kv_lengths,
    scale,
    sliding_window=-1,
    logit_cap=0.0,
    automatic=False,
):
    """Return None after a launch, otherwise the reason to use native attention.

    This entry point is only for DO_BLEND. The caller retains QCOMPUTE/decode
    dispatch and native fallback. No process-global allocator setting persists
    beyond this launch; scratch belongs to the actual CUDA consumption stream.
    """
    reason = _unsupported_reason(
        q, k, v, out, positions, q_lengths, kv_lengths, sliding_window, logit_cap
    )
    if reason is not None:
        return reason
    if automatic and "H20" not in torch.cuda.get_device_name(q.device).split():
        return "automatic_device_model"
    try:
        import triton
        import triton.language as tl
        from triton.runtime import _allocation
    except ImportError:
        return "descriptor_runtime_unavailable"
    allocator = getattr(_allocation, "_allocator", None)
    if (
        not hasattr(tl, "make_tensor_descriptor")
        or not hasattr(allocator, "set")
        or not hasattr(allocator, "reset")
    ):
        return "descriptor_runtime_unavailable"
    from ucm.sparse.qcfuse.tma_kernel import position_attention_tma

    def allocate(size, alignment, stream):
        current = torch.cuda.current_stream(q.device)
        if stream is not None and stream != current.cuda_stream:
            with torch.cuda.stream(torch.cuda.ExternalStream(stream, device=q.device)):
                return torch.empty(size, device=q.device, dtype=torch.uint8)
        return torch.empty(size, device=q.device, dtype=torch.uint8)

    token = allocator.set(allocate)
    try:
        with torch.cuda.device(q.device):
            position_attention_tma[
                (len(q_lengths), 64, triton.cdiv(max(q_lengths), 128))
            ](
                q,
                k,
                v,
                positions,
                q_starts,
                q_lens,
                kv_starts,
                kv_lens,
                out,
                q.stride(0),
                k.stride(0),
                v.stride(0),
                out.stride(0),
                out.stride(1),
                scale,
                HQ=64,
                HK=8,
                D=128,
                M=128,
                N=64,
                SPECIALIZE=False,
                num_warps=8,
                num_stages=2,
                maxnreg=128,
            )
    finally:
        allocator.reset(token)
    return None
