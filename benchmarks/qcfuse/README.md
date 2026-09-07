# QCFuse Evaluation Runner

`blend_ssd.py` evaluates QCFuse through UCM and vLLM. It creates per-sample
caches with serial `KVCOMPUTE`, then runs batched `QCOMPUTE` and `DO_BLEND`.

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
`ruler_mq`, and `ruler_mv`; each requires a matching JSONL file under
`--data_dir`. Baselines are `fullcomp`, `ours`, `fuserag`, and `prophetkv`.
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
