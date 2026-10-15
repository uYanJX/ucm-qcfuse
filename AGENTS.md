# Repository guidance

This branch is a source distribution. Edit `ucm/` directly; do not introduce
an overlay directory or an external patch-file installation step.

QCFuse computation is in `ucm/sparse/qcfuse/`; cache transport is in
`ucm/store/qcfuse/`; current vLLM integration is in
`ucm/integration/vllm/latest_sparse.py` and the QCFuse connector/worker modules.
Use the supported runtime in README.md rather than historical adapters.

Preserve pre-RoPE cache semantics, request-local mutation, selected token
budgets, explicit position-based attention masks and CUDA stream ownership.
Algorithm changes require differential validation against the pinned reference.
Kernel, transfer and batching changes require CUDA checks for batch sizes
1, 2 and 4, including repeated cache reuse and cleanup.

Run `PYTHONPATH="$PWD" python -m pytest -q tests/qcfuse -k 'not cuda'` for CPU
regressions, and run `tests/qcfuse/test_cuda_kernels.py` on an idle CUDA device.
Format changed Python with Black targeting Python 3.10.

Never commit models, datasets, cache payloads, generated predictions, raw
experiment records, local host configuration or credentials. Use generic paths
in documentation and preserve upstream licenses and attribution.
