# QCFuse on UCM

This branch contains UCM source with QCFuse integrated for vLLM 0.31 and
Qwen3. Install this checkout directly; there is no overlay-copy or patch-file
application step, and installed vLLM source files are not edited.

QCFuse reuses full KV caches while recomputing selected document tokens:

1. **KVCOMPUTE** creates raw pre-RoPE KV, digest, and critical-layer caches.
2. **QCOMPUTE** runs the query against the digest and retains selection inputs.
3. **DO_BLEND** selects tokens, repairs cached KV, runs position-aware attention,
   and writes the resulting full KV to vLLM for decoding.

The recomputation ratio controls document-token selection. It is not a KV
compression ratio or a promise that total model work scales by that fraction.

## Compatibility

The QCFuse path is validated with Python 3.10, PyTorch 2.14.1+cu130,
vLLM 0.31.0, Transformers 5.19.0, Qwen3 BF16, CUDA and Triton. Use tensor and
pipeline parallel sizes of one, eager execution, disabled prefix caching and
unchunked prefill. Homogeneous online batches are supported; offline cache
construction is serial. Required GPU memory depends on total input tokens,
model weights, native KV pages, source-cache residency and workspace.

GPU-resident mode maintains an immutable source cache alongside the online KV
pool. A model/context fitting full prefill does not guarantee that GPU-resident
fusion fits. CPU/SSD modes provide bounded layer staging. The runner reports
capacity failures rather than silently truncating inputs or changing precision.

Old post-RoPE caches must be rebuilt. Other UCM integrations are retained as
upstream source, but this branch's QCFuse support statement covers the runtime
above; historical external patch bundles are not distributed.

## Install

Use a Linux CUDA environment with the versions above, a matching CUDA toolkit,
a C++17 compiler and CMake. The vLLM wheel must match the installed PyTorch/CUDA
runtime. Then install this source tree:

```bash
git clone --branch ucm-adapt git@github.com:uYanJX/ucm-qcfuse.git
cd ucm-qcfuse
python -m pip install -r requirements-build.txt
PLATFORM=cuda python -m pip install --no-build-isolation -e .
python -m pip install -r requirements-qcfuse.txt
```

Enable the UCM runtime integration when running inference:

```bash
export ENABLE_UCM_PATCH=1
export ENABLE_SPARSE=1
```

`ENABLE_UCM_PATCH` is the upstream name for runtime hook activation; it does
not mean applying an external patch file. Benchmark runners configure the
QCFuse connector and worker extension automatically.

## Run and test

See [the evaluation guide](benchmarks/qcfuse/README.md) for frozen manifests,
quality runs, repeated timing, cold/warm storage and cleanup behavior.
Models, datasets and experiment outputs are supplied locally.

```bash
PYTHONPATH="$PWD" python -m pytest -q tests/qcfuse -k 'not cuda'
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  python -m pytest -q tests/qcfuse/test_cuda_kernels.py
```

Set `QCFUSE_REFERENCE_DIR` to a local checkout of the pinned QCFuse reference
to enable optional upstream differential checks. Benchmarks recognize
`fullcomp` and `ours` as verified implementation identities; unvalidated
variants are not reported as CacheBlend or ProphetKV.

## Source map

| Path | Purpose |
|---|---|
| `ucm/sparse/qcfuse/` | Selection, request state, pre-RoPE fusion and position-aware attention |
| `ucm/store/qcfuse/` | Packed cache format, CPU/GPU residency and bounded SSD/H2D prefetch |
| `ucm/integration/vllm/latest_sparse.py` | Native vLLM 0.31 Qwen3 hooks |
| `ucm/integration/vllm/qcfuse_connector.py` | Scheduler/worker request metadata |
| `ucm/integration/vllm/qcfuse_worker.py` | Residency control, memory attribution and bounded profiling |

For diagnostics, the configured worker extension exposes named RPCs that work
with the current vLLM serializer:

```python
memory = llm.collective_rpc("qcfuse_memory_breakdown")
llm.collective_rpc(
    "qcfuse_start_execution_profile",
    args=("/path/to/new-trace.json", 3),
)
try:
    # Submit the fixed diagnostic requests through the normal generation path.
    ...
finally:
    profile = llm.collective_rpc("qcfuse_stop_execution_profile")
```

Use a trace path containing `{rank}` when running more than one rank. The
profiler records tensor shapes and stops before the forward following its
budget; the final RPC also handles requests that end earlier. It removes its
model callback before later decode. Detailed profiles include instrumentation
overhead and must be disabled for formal timing. Memory attribution counts
unique parameter and observed native KV storage; allocator reservations must
not be added again as live tensor memory.
| `benchmarks/qcfuse/` | Dataset preparation, measured runs, queue and result validation |
| `tests/qcfuse/` | CPU, CUDA, lifecycle and benchmark-contract tests |
| `ucm/`, `kv_semantics/`, `test/` | UCM Python/native source and upstream tests |

Private experiment logs, generated answers, host configuration, model files,
datasets and caches are excluded. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for upstream revisions and licenses.
