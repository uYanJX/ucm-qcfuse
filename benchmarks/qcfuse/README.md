# QCFuse Evaluation Runner

`blend_ssd.py` evaluates QCFuse through UCM and vLLM. It creates per-sample
caches with serial `KVCOMPUTE`, then runs batched `QCOMPUTE` and `DO_BLEND`.

## Frozen quality and timing

Run these commands from this source checkout after installation. Obtain a
dataset under its own license and verify its provenance before freezing it.
The example uses LongBench v2; replace paths with your own files.

```bash
python benchmarks/qcfuse/prepare_manifest.py \
  --source /path/to/longbench_v2.jsonl --dataset longbench_v2 \
  --model /path/to/Qwen3-32B --size 100 --seed 42 \
  --thinking on --max-new-tokens 4096 \
  --output-dir datasets/frozen_longbench

ENABLE_UCM_PATCH=1 ENABLE_SPARSE=1 CUDA_VISIBLE_DEVICES=0 \
python benchmarks/qcfuse/run_frozen_cell.py \
  --data datasets/frozen_longbench/longbench_v2.jsonl \
  --manifest datasets/frozen_longbench/manifest.json \
  --dataset longbench_v2 --model /path/to/Qwen3-32B \
  --baseline ours --storage ssd --ratio 0.4 \
  --kind quality --warmup 0 --repeats 1 --batch-size 1 \
  --thinking on --max-new-tokens 4096 --max-model-len 71680 \
  --cache-parent /path/to/local/cache \
  --output results/qcfuse_ssd_quality.json
```

Use a fresh output path for each run. For the full-prefill baseline, use
`--baseline fullcomp`. For steady-state latency, use `--kind timing
--warmup 5 --repeats 10 --timing-samples 1`. CPU/GPU resident modes are selected
with `--storage cpu` or `--storage gpu`; `--ssd-cache-state cold` requests a
filesystem cold-cache control. The model window is an explicit protocol
parameter, not a guarantee that every input or resident cache fits a GPU.

On Linux, add `--verify-cold-residency` with `--baseline ours --storage ssd
--ssd-cache-state cold` to record each owned KV file's page residency after
fsync and per-file eviction advice, immediately before online execution.
The read-only check does not read payload contents, and its duration is
recorded outside TTFT. Inspect `cache_state_actions[].residency_before_online`:
only `all_payload_pages_observed_nonresident=true` establishes an observed
cold snapshot. Eviction advice alone does not establish this, and a snapshot
does not identify the backing storage or guarantee later pages stay cold.
Keep the remaining resident-page counts if advice does not fully evict them;
do not label that observation a fully cold SSD run.

Use `--verify-ssd-residency` to record the same snapshot for either warm or
cold SSD requests. Warm mode only observes residency: it does not read,
preload or warm KV payloads. A warm observation qualifies as fully resident
only when `resident_pages == pages`; retain partial residency as observed.
In cold mode the snapshot follows the existing eviction advice. Both options
are disabled by default, and neither verifies the backing storage medium.

The runner keeps oversized sources as unavailable entries, checks immutable
manifest identity and removes its uniquely owned cache root after worker
shutdown. Keep result JSON, generated predictions and manifests outside Git.

For a query-scheduling diagnostic, set an explicit
`VLLM_ENABLE_V1_MULTIPROCESSING=0` for both control and candidate and add
`--qcompute-batch-size 1` to split only QCOMPUTE. DO_BLEND and decode retain
`--batch-size`; warmup follows the same query grouping. The default keeps the
existing query batching. Inspect `execution_context` for the actual EngineCore
client class and environment, and `parts[].phase_timings[].qcompute_group_sizes`
for observed query submission sizes. This option does not establish output
equivalence or accuracy by itself.

Add `--audit-cache-payloads` to an `ours` CPU/GPU resident audit to record owned
KV file sizes and SHA-256 under `offline_cache_batches[].payload_audit`.
Hashing reads payloads before online execution and is excluded from TTFT;
it is rejected for SSD, quality and timing runs to avoid hidden cache warming
or formal timing overhead. Compare hashes before attributing a result to
scheduling alone. No KV contents are included in the receipt.

## Interactive dataset runner

```bash
ENABLE_UCM_PATCH=1 ENABLE_SPARSE=1 \
python benchmarks/qcfuse/blend_ssd.py \
  --model qwen3-8b \
  --model_dir /path/to/models \
  --dataset hotpotqa \
  --data_dir /path/to/datasets \
  --cache_dir /path/to/ssd/qcfuse \
  --baseline ours \
  --size 30 \
  --batch-size 4 \
  --blend-ratio 0.5 \
  --results-json logs/qcfuse/hotpotqa-ours.json
```

Supported datasets are `hotpotqa`, `2wikimqa`, `musique`, `ruler_vt`,
`ruler_mq`, `ruler_mv`, `ruler_64k`, and `longbench_v2`; each requires a matching
JSONL file under `--data_dir`. Verified baselines are `fullcomp` and `ours`.
`fuserag` is not CacheBlend, and the runner's `prophetkv` variant has not been
validated against ProphetKV. They require an explicit diagnostic override and
are excluded from the four-method comparison. Real CacheBlend and ProphetKV
remain N/A until a compatible Qwen3-32B implementation is verified.
Use a fast local NVMe path for `--cache_dir`.

Run the resumable matrix with:

```bash
PY=/path/to/python \
DATA=/path/to/QCFuse/data \
MODELS=/path/to/models \
SIZE=30 BATCH_SIZE=4 BLEND_RATIO=0.5 \
bash benchmarks/qcfuse/run_full_benchmark.sh
```

The script stores raw outputs under `logs/qcfuse/`, which is ignored by Git.
Set `DATASETS`, `BASELINES`, or `MAX_NUM_BATCHED_TOKENS` to narrow the run.

Use `run_frozen_cell.py` for measured comparisons against a frozen manifest.
It checks source identity, exact prompt token hashes, checkpoint configuration,
Thinking mode and output budget before measurement. Quality, repeated timing
and CUDA-event audit are distinct modes. Oversized inputs remain N/A with their
original source IDs; eligible-subset accuracy is not full-cohort accuracy.

Frozen QCFuse cells record actual selected/recomputed/retained token counts and
QKV/FFN token-layers through host shape observations. Normal quality/timing
cells use no token-log file or CUDA timing events. A bounded worker buffer is
cleared before the measured batch and drained afterwards through the named
control RPC; QCOMPUTE and DO_BLEND records are associated by client request ID.
`--kind audit` additionally records CUDA stream intervals and layer diagnostics.
For comparisons across batch sizes, use `--audit-samples N --audit-repeats 2`
to check the same first N frozen sources, then `--kind timing --timing-samples N`
for measurements without instrumentation. Both cohort options are independent
of `--batch-size` and cannot be combined. Without a cohort option, audit and
timing keep their first-batch default; quality visits all frozen sources.
The ratio remains a document selection budget, rather than a full-model FLOPs
or retained-KV fraction. Results generated before these observations were
enabled retain their original fields and are not backfilled with formula data.

Caches belong to a unique cell root and are removed per batch and on shutdown.
Offline construction, shared cache-workspace waits and CPU/GPU preload are
reported separately from online TTFT. Transfer counters in worker snapshots
are cumulative: subtract before from after. `cuda_wait_seconds` measures host
waiting for a staging-slot completion event; it is not exclusive GPU stall
time. `cuda_copy_seconds` is available only when CUDA-event timing is enabled;
a disabled-event zero must not be interpreted as zero transfer work.
`h2d_bytes` counts actual key/value payload copied by completed requests,
including CPU resident loads; grouped file-read padding is excluded. GPU
resident copies do not add H2D bytes. Host enqueue time and overlapped CUDA
copy duration are separate measurements, neither an exclusive TTFT cost.
