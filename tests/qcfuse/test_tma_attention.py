from types import SimpleNamespace

import pytest
import torch

from ucm.sparse.qcfuse.tma_attention import _unsupported_reason


class CudaHeader:
    """Inspect real tensor shapes/strides without allocating a CUDA tensor."""

    device = torch.device("cuda:0")

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)


def headers():
    return [
        CudaHeader(torch.empty(3, 64, 128, dtype=torch.bfloat16)),
        CudaHeader(torch.empty(5, 8, 128, dtype=torch.bfloat16)),
        CudaHeader(torch.empty(5, 8, 128, dtype=torch.bfloat16)),
        CudaHeader(torch.empty(3, 64, 128, dtype=torch.bfloat16)),
        CudaHeader(torch.arange(3)),
    ]


def test_tma_cpu_fallback_never_initializes_device(monkeypatch):
    def unexpected(*args):
        raise AssertionError("CPU fallback must not query a CUDA device")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected)
    args = [torch.empty(3, 64, 128)] * 4 + [torch.arange(3)]
    assert _unsupported_reason(*args, [3], [5], -1, 0.0) == "device"


@pytest.mark.parametrize(
    "case,expected",
    [
        ("ampere", "device_capability"),
        ("window", "mask_or_logit_cap"),
        ("cap", "mask_or_logit_cap"),
        ("fp16", "dtype_or_device"),
        ("heads", "head_shape"),
        ("value_shape", "tensor_shape"),
        ("query_stride", "descriptor_layout_or_alignment"),
        ("position_stride", "query_positions"),
        ("empty_kv", "empty_or_unknown_KV_layout"),
        ("oversized_kv", "empty_or_unknown_KV_layout"),
    ],
)
def test_tma_unsupported_headers_choose_native(monkeypatch, case, expected):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    args = headers()
    window, cap, kv_lengths = -1, 0.0, [5]
    assert _unsupported_reason(*args, [3], kv_lengths, window, cap) is None
    if case == "ampere":
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    elif case == "window":
        window = 16
    elif case == "cap":
        cap = 30.0
    elif case == "fp16":
        args[0] = CudaHeader(args[0].tensor.to(torch.float16))
    elif case == "heads":
        args[0] = CudaHeader(torch.empty(3, 32, 128, dtype=torch.bfloat16))
    elif case == "value_shape":
        args[2] = CudaHeader(torch.empty(4, 8, 128, dtype=torch.bfloat16))
    elif case == "query_stride":
        args[0] = CudaHeader(torch.empty(3, 64, 129, dtype=torch.bfloat16)[..., :128])
    elif case == "position_stride":
        args[4] = CudaHeader(torch.arange(6)[::2])
    elif case == "empty_kv":
        kv_lengths = [0]
    elif case == "oversized_kv":
        kv_lengths = [6]
    assert _unsupported_reason(*args, [3], kv_lengths, window, cap) == expected


def test_qcompute_tma_request_keeps_original_native_attention(monkeypatch):
    from ucm.sparse.qcfuse.blend_info import BlendStyle
    from ucm.sparse.qcfuse.qcfuse import QcFuse
    from ucm.sparse.qcfuse import ragged_attention

    calls = []

    def native(q, k, v, out, *args, **kwargs):
        calls.append(args[-2].clone())
        out.copy_(q)

    monkeypatch.setattr(ragged_attention, "ragged_positions_attention_fwd", native)
    fuser = object.__new__(QcFuse)
    fuser._attention_backend = "tma"
    fuser._attention_backend_counts = {"native": 0, "flex": 0}
    fuser._flex_key_block = 64
    fuser._step = SimpleNamespace(phase=BlendStyle.QCOMPUTE)
    q = torch.randn(3, 64, 128)
    k = v = torch.randn(5, 8, 128)
    out = torch.empty_like(q)
    positions = torch.tensor([4, 0, 2], dtype=torch.int32)
    prepared = SimpleNamespace(q_lens=[3], kv_lens=[5], q_positions=positions)
    layer = SimpleNamespace(impl=SimpleNamespace(sliding_window=(-1, -1)))
    fuser._run_sparse_attention(q, k, v, out, layer, prepared)
    assert torch.equal(out, q) and torch.equal(calls[0], positions)
    stats = fuser.attention_backend_stats()
    assert stats["layer_calls"] == {"native": 1, "flex": 0}
    assert stats["tma_fallbacks"] == {"phase_keeps_original_kernel": 1}
