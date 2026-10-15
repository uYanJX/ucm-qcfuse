# Third-party notices

## Unified Cache Management

The UCM source in this branch derives from
[ModelEngine-Group/unified-cache-management](https://github.com/ModelEngine-Group/unified-cache-management)
at revision `e7375fcdbf3c39f7cb97e38f40fffba7aa4dea67`, with QCFuse integration,
vLLM compatibility and cache lifecycle/transport changes applied as source.
UCM is Copyright 2025 Huawei Technologies Co., Ltd. and distributed under the
MIT license in `LICENSE`. Existing file-level notices are preserved.

## QCFuse and SGLang

Parts of `ucm/sparse/qcfuse/`, `ucm/store/qcfuse/` and `benchmarks/qcfuse/`
derive from [QCFuse](https://github.com/uYanJX/QCFuse) revision
`1416459ada5bec0695ec375ada435e407cdea699`. QCFuse is Copyright 2026 QCFuse
Contributors and licensed under Apache-2.0, included in
`LICENSES/Apache-2.0.txt`. QCFuse includes software from
[SGLang](https://github.com/sgl-project/sglang).

This adaptation changes runtime integration, cache representation, RoPE
handling, transfer scheduling and batching. The position-aware attention mask
and token-selection semantics remain explicit in the source.

Upstream UCM native dependencies retain their own notices and build-time
license requirements. Model weights and datasets are not distributed here;
users must obtain them under their respective licenses.
