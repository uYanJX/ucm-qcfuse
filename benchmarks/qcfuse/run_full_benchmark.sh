#!/bin/bash
# Resumable QCFuse benchmark matrix. Successful cells get a .done marker.
#
# Usage:
#   bash benchmarks/qcfuse/run_full_benchmark.sh
#   DATASETS="musique 2wikimqa" bash benchmarks/qcfuse/run_full_benchmark.sh
set -uo pipefail

export ENABLE_UCM_PATCH=1
export ENABLE_SPARSE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${PY:-python}
DATA=${DATA:-$ROOT/benchmarks/qcfuse}
MODELS=${MODELS:-$ROOT/models}
MODEL=${MODEL:-qwen3-8b}
LOG=${LOG:-$ROOT/logs/qcfuse}
CACHE=${CACHE:-$ROOT/cache/qcfuse}
SIZE=${SIZE:-30}
BATCH_SIZE=${BATCH_SIZE:-1}
BLEND_RATIO=${BLEND_RATIO:-0.5}
NUM_WARMUP=${NUM_WARMUP:-3}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-}
RUN_TAG=${RUN_TAG:-v4_n${SIZE}_b${BATCH_SIZE}_r${BLEND_RATIO}_w${NUM_WARMUP}}

DATASETS=${DATASETS:-"hotpotqa 2wikimqa musique ruler_mq ruler_mv ruler_vt"}
BASELINES=${BASELINES:-"fullcomp ours"}

cd "$ROOT"
mkdir -p "$LOG/results"
failed=0
for ds in $DATASETS; do
    for bl in $BASELINES; do
        marker=$LOG/.done_${RUN_TAG}_${ds}_${bl}
        logfile=$LOG/${RUN_TAG}_${ds}_${bl}.log
        result_json=$LOG/results/${RUN_TAG}_${ds}_${bl}.json
        if [ -f "$marker" ]; then
            echo "[skip]   $ds/$bl (already done)"
            continue
        fi
        echo "[START]  $ds/$bl $(date)"
        extra_args=()
        if [ -n "$MAX_NUM_BATCHED_TOKENS" ]; then
            extra_args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
        fi
        $PY benchmarks/qcfuse/blend_ssd.py \
            --model "$MODEL" --model_dir "$MODELS" \
            --data_dir "$DATA" --dataset "$ds" --baseline "$bl" \
            --size "$SIZE" --batch-size "$BATCH_SIZE" --cache_dir "$CACHE" \
            --blend-ratio "$BLEND_RATIO" --num-warmup "$NUM_WARMUP" \
            --results-json "$result_json" "${extra_args[@]}" \
            > "$logfile" 2>&1
        rc=$?
        if [ $rc -eq 0 ]; then
            echo "[DONE]   $ds/$bl $(date)" | tee -a "$LOG/${RUN_TAG}_summary.log"
            grep -h "^$(basename "$MODEL")" "$logfile" \
                >> "$LOG/${RUN_TAG}_summary.log"
            touch "$marker"
        else
            failed=1
            echo "[FAIL]   $ds/$bl exit=$rc $(date)" \
                | tee -a "$LOG/${RUN_TAG}_summary.log"
        fi
    done
done
echo "[ALLDONE] $(date)" | tee -a "$LOG/${RUN_TAG}_summary.log"
exit "$failed"
