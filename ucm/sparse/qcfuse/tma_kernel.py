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
"""Hopper descriptor loads with the reference position-attention arithmetic.

Keep tile sizes and floating-point reduction order aligned with the native
QCFuse kernel; sparse query positions determine the causal mask.
"""

import triton
import triton.language as tl


@triton.jit
def position_attention_tma(
    Q,
    K,
    V,
    POS,
    QS,
    QN,
    KS,
    KN,
    OUT,
    sqt,
    skt,
    svt,
    sot,
    soh,
    scale,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    SPECIALIZE: tl.constexpr,
):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qn, kn = tl.load(QN + batch), tl.load(KN + batch)
    qs, ks = tl.load(QS + batch), tl.load(KS + batch)
    if block * M < qn:
        qdesc = tl.make_tensor_descriptor(
            Q + qs * sqt, shape=[qn, HQ * D], strides=[sqt, 1], block_shape=[M, D]
        )
        kdesc = tl.make_tensor_descriptor(
            K + ks * skt, shape=[kn, HK * D], strides=[skt, 1], block_shape=[N, D]
        )
        vdesc = tl.make_tensor_descriptor(
            V + ks * svt, shape=[kn, HK * D], strides=[svt, 1], block_shape=[N, D]
        )
        rows = block * M + tl.arange(0, M)
        cols, dims = tl.arange(0, N), tl.arange(0, D)
        q = qdesc.load([block * M, head * D])
        positions = tl.load(POS + qs + rows, mask=rows < qn, other=0)
        stop = tl.minimum(kn, tl.max(tl.where(rows < qn, positions, -1)) + 1)
        kv_head = head // (HQ // HK)
        maximum = tl.zeros([M], dtype=tl.float32) - float("inf")
        denominator = tl.zeros([M], dtype=tl.float32)
        accumulator = tl.zeros([M, D], dtype=tl.float32)
        for start in tl.range(0, stop, N, warp_specialize=SPECIALIZE):
            indices = start + cols
            key = kdesc.load([start, kv_head * D]).T
            scores = tl.dot(q, key)
            scores *= scale
            scores += tl.where(
                positions[:, None] >= indices[None, :], 0.0, float("-inf")
            )
            local_max = tl.max(scores, 1)
            safe_max = tl.where(local_max == float("-inf"), 0.0, local_max)
            probabilities = tl.exp(scores - safe_max[:, None])
            local_sum = tl.sum(probabilities, 1)
            next_max = tl.maximum(maximum, local_max)
            alpha = tl.exp(
                tl.where(maximum == float("-inf"), float("-inf"), maximum - next_max)
            )
            beta = tl.exp(
                tl.where(
                    local_max == float("-inf"), float("-inf"), local_max - next_max
                )
            )
            next_sum = alpha * denominator + beta * local_sum
            p_scale = tl.where(next_sum == 0.0, 0.0, beta / next_sum)
            acc_scale = tl.where(next_sum == 0.0, 0.0, denominator / next_sum * alpha)
            probabilities *= p_scale[:, None]
            accumulator *= acc_scale[:, None]
            value = vdesc.load([start, kv_head * D])
            probabilities = probabilities.to(value.dtype)
            accumulator += tl.dot(probabilities, value)
            denominator, maximum = next_sum, next_max
        tl.store(
            OUT + (qs + rows[:, None]) * sot + head * soh + dims[None, :],
            accumulator,
            mask=rows[:, None] < qn,
        )
