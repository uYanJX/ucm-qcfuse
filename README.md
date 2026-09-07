# UCM QCFuse

This repository is a focused port of [QCFuse](https://github.com/uYanJX/QCFuse)
to Unified Cache Management (UCM). It contains only the QCFuse implementation,
the required UCM/vLLM integration patch, targeted tests, and a benchmark runner.
Model weights, datasets, generated caches, and raw logs are intentionally
excluded.

The port preserves QCFuse's three-pass design:

1. `KVCOMPUTE` builds full KV, digest, and critical-layer caches.
2. `QCOMPUTE` scores context tokens from the compressed view.
3. `DO_BLEND` recomputes selected tokens and reconstructs attention with cached KV.

The implementation uses raw pre-RoPE Q/K/V, mean importance aggregation,
position-aware ragged attention, bounded SSD/H2D layer prefetch, and per-request
state. Homogeneous online batches are supported and have been smoke-tested with
batch sizes 2 and 4. Offline `KVCOMPUTE` remains serial by design.

## Compatibility

- UCM base revision: `61c8387518592c545e88d205e58a7fc3edd8d119`
- QCFuse reference revision: `1416459ada5bec0695ec375ada435e407cdea699`
- vLLM 0.11, Qwen3, CUDA, and Triton
- Tensor parallel size 1, pipeline parallel size 1, eager execution
- Prefix caching disabled; each online batch must use one phase and an unchunked prefill

## Apply to UCM

Start from the compatible UCM revision, then apply the tracked-file patch and
copy the new modules:

```bash
git -C /path/to/unified-cache-management checkout 61c8387518592c545e88d205e58a7fc3edd8d119
git -C /path/to/unified-cache-management apply "$PWD/patches/ucm-vllm-v0110-integration.patch"
cp -a overlay/. /path/to/unified-cache-management/
```

Install UCM and its patched vLLM environment as usual. Existing post-RoPE
QCFuse cache files are incompatible and must be rebuilt.

## Validate

```bash
PYTHONPATH=/path/to/unified-cache-management pytest -q tests/qcfuse -k "not cuda"

CUDA_VISIBLE_DEVICES=0 \
QCFUSE_REFERENCE_DIR=/path/to/QCFuse \
PYTHONPATH=/path/to/unified-cache-management \
pytest -q tests/qcfuse/test_cuda_kernels.py
```

See [benchmarks/qcfuse/README.md](benchmarks/qcfuse/README.md) for evaluation
commands and [RESULTS.md](RESULTS.md) for the retained 30-sample smoke results.

## Repository Layout

- `overlay/`: new files copied into a UCM checkout.
- `patches/`: minimal changes to existing UCM and vLLM patch files.
- `tests/qcfuse/`: CPU and CUDA regression tests.
- `benchmarks/qcfuse/`: LongBench/RULER evaluation driver.

Licensing and upstream attribution are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
