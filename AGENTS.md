# Repository Guidelines

## Project Structure & Module Organization

This repository is a focused overlay, not a standalone UCM distribution.
`overlay/ucm/sparse/qcfuse/` contains request state, batching, selection, and
attention logic. `overlay/ucm/store/qcfuse/` owns packed cache files and bounded
SSD-to-GPU prefetch. vLLM integration files live under
`overlay/ucm/integration/vllm/`; edits to existing UCM files are captured in
`patches/ucm-vllm-v0110-integration.patch`. Tests and evaluation tools are in
`tests/qcfuse/` and `benchmarks/qcfuse/`.

## Build, Test, and Development Commands

Apply the patch and overlay to the UCM revision listed in `README.md` before
running code. Use `pytest -q tests/qcfuse -k "not cuda"` for CPU regressions.
Run CUDA checks with `CUDA_VISIBLE_DEVICES=0 pytest -q tests/qcfuse`; set
`QCFUSE_REFERENCE_DIR=/path/to/QCFuse` to enable the upstream differential.
Format Python with `black --target-version py310 overlay tests benchmarks` and
validate the integration patch with `git apply --check patches/*.patch` from a
clean compatible UCM checkout.

## Coding Style & Naming Conventions

Use Python 3.10+, four-space indentation, Black formatting, and type hints on
public boundaries. Name modules and functions in `snake_case`, classes in
`PascalCase`, and tests `test_*.py`. Keep comments for non-obvious invariants,
especially stream ownership, tensor layouts, cache positions, and batching.
Do not retain development-phase notes or commented-out experiments.

## Testing Guidelines

Add focused CPU coverage for metadata and layout changes. Kernel or stream
changes require CUDA tests, and algorithm changes require a differential check
against the pinned upstream revision. Exercise batch sizes 1, 2, and 4 when
changing request slicing or ragged attention. Never commit model outputs,
generated cache files, or benchmark logs.

## Commit & Pull Request Guidelines

Use concise imperative subjects such as `[Fix] Preserve CUDA stream ownership`.
Keep overlay code and its integration-patch update in the same commit. Pull
requests should state the compatible UCM/vLLM versions, describe behavioral
changes, list exact test commands, and report quality or timing impact for
performance-sensitive changes.
