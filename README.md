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

Optional GPU residency can store BF16 KV as losslessly encoded bytes and
materialize query KV from the same explicitly paired source. The packed
pre-RoPE disk format, token selection and attention computation are unchanged.
Decoding returns request-owned tensors; fusion never mutates resident sources.

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

For a matching CUDA 13 runtime, optional lossless residency was tested with
`nvidia-nvcomp-cu13==6.0.0.12` (and its matching library dependency). Install it
separately, then add `--storage gpu --gpu-resident-codec ans_f16_bits
--gpu-query-alias on` to the frozen runner. The codec reinterprets BF16 words as
16-bit symbols without a numerical conversion. Query/source byte equivalence
is verified during preload; online loads decode and gather exclusively on
GPU. Preload is reported separately, and online decoding remains inside TTFT.
These switches are opt-in; uncompressed GPU, CPU and SSD defaults are retained.
Capacity and end-to-end speed must be measured for the actual model and inputs.

```bash
PYTHONPATH="$PWD" python -m pytest -q tests/qcfuse -k 'not cuda'
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  python -m pytest -q tests/qcfuse/test_cuda_kernels.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  python -m pytest -q tests/qcfuse/test_resident_codec.py
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
| `benchmarks/qcfuse/` | Dataset preparation, measured runs, queue and result validation |
| `tests/qcfuse/` | CPU, CUDA, lifecycle and benchmark-contract tests |
| `ucm/`, `kv_semantics/`, `test/` | UCM Python/native source and upstream tests |

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
model callback before later decode. To profile decode alone, pass a third
argument with the number of forwards to skip; for a QCOMPUTE request followed
by sparse prefill, `args=("/path/to/new-decode-trace.json", 32, 2)` captures up
to 32 subsequent forwards. Skipped forwards are not traced. Detailed profiles include instrumentation
overhead and must be disabled for formal timing. Memory attribution counts
unique parameter and observed native KV storage; allocator reservations must
not be added again as live tensor memory.

Private experiment logs, generated answers, host configuration, model files,
datasets and caches are excluded. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for upstream revisions and licenses.

## GPU FFN workspace experiment

`QCFUSE_GPU_FFN_TILE_TOKENS=8192` optionally bounds projection rows for the
existing BF16, single-rank, unquantized GPU-resident sparse-prefill fast path.
The default is `0` (whole-matrix projections). Tiling balances uneven ends and
keeps every selected row, both weights, native SiLU and the original position
mask. CPU/SSD, ordinary prefill and decode use their existing paths.

This controls transient memory rather than promising a latency gain. GEMM
shapes change, so verify selection, token outputs and quality against the
whole-matrix configuration on the same inputs before adopting it. Frozen-run
receipts record the configured tile budget separately.

## Position-based attention experiment

`QCFUSE_GPU_ATTENTION_BACKEND=flex` enables a compiled FlexAttention backend
only for the DO_BLEND prefill pass. The default is `native`. It preserves
original query positions, full cached KV lengths and `key_position <=
query_position`, including unequal request lengths. It builds compact block
metadata once per request and reuses the mask across layers; masks and source
positions are request-owned. QCOMPUTE selection, offline cache generation and
decode keep their existing kernels. Unsupported head dimensions, sliding
windows and logit caps keep native attention.

This opt-in requires the current tested PyTorch FlexAttention API. BF16
arithmetic differs from the reference kernel, so kernel tolerances alone do
not prove model quality. Complete matched output/quality regressions before
adopting it. `QCFUSE_FLEX_KEY_BLOCK=64` (default) or `128` controls key tiling.
Worker benchmark snapshots report actual native/flex layer-call counts;
frozen receipts record the backend and block size.

Audit defaults to one observation. Use `--audit-repeats 2` to verify two
continuous requests against the same resident source. `--repeats` controls
timing observations only; quality retains one observation per source. Receipts
record the effective measurement count separately from CLI options.

Explicit audit also records causal-pair work from the actual query positions
and padded dot-product slots for the reference 128-by-64 kernel. Selecting half
the rows need not select half the causal pairs. These are work counters rather
than latency estimates; native FA3 uses its own tiling. Counts-only timing never
copies position tensors or computes these diagnostic counters.

## Native decode graph experiment

The frozen runner accepts `--decode-cudagraph on` (default: `off`). It uses
native vLLM V2 `FULL_DECODE_ONLY`, disables compilation and captures decode
batch sizes from 1 through the configured batch size. QCFuse prefill,
including a one-token QCOMPUTE, remains eager. The connector refreshes sparse
request state before replay, since replay bypasses Python model callbacks.

Worker snapshots report captured token counts and actual graph replay/eager
forward counters. Verify that replays occurred, compare all output tokens and
selection/position hashes, and include capture memory in capacity checks.
Graph capture cost belongs to startup; this option makes no unmeasured speed
claim. Use separate eager diagnostics for the bounded model-forward profiler;
a replay does not execute its Python hook.
