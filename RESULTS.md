# Smoke Test Results

Tests ran on 2026-09-07 with Qwen3-8B, one RTX A6000, vLLM 0.11,
`batch_size=4`, blend ratio `0.5`, three warmups, and 30 examples per dataset.
QCFuse TTFT is `QCOMPUTE + DO_BLEND`; offline cache construction is excluded.

| Dataset | Metric | Full | QCFuse | Full TTFT | QCFuse TTFT | Full throughput | QCFuse throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | F1 | 0.6625 | 0.6456 | 5.734 s | 9.025 s | 0.612/s | 0.414/s |
| 2WikiMQA | F1 | 0.3602 | 0.3426 | 4.364 s | 4.628 s | 0.786/s | 0.785/s |
| Musique | F1 | 0.3288 | 0.3789 | 6.338 s | 8.479 s | 0.557/s | 0.421/s |
| RULER multi-query | string match | 0.9917 | 0.9917 | 6.627 s | 4.685 s | 0.492/s | 0.650/s |

Across the four completed paired runs, aggregate throughput was 0.594 samples/s
for full computation and 0.526 samples/s for QCFuse, so this host did not show
an overall speedup. RULER multi-query did improve: TTFT was 1.41x faster and
throughput 1.32x higher. The other workloads were neutral or slower because two
online passes and SSD reads outweighed the saved attention work. The cache
volume was on SATA-class storage measured at about 525 MB/s sequential read;
NVMe is recommended for performance evaluation.

Quality remained close at this smoke-test scale: two F1 datasets changed by
less than 0.018, Musique improved by 0.050, and RULER was unchanged. Thirty
examples are useful for regression detection, not a statistically strong model
quality claim.

Additional checks completed successfully:

- 14 CPU tests passed in the clean overlay verification.
- 4 CUDA tests passed, including a differential check against upstream QCFuse.
- End-to-end online smoke runs passed with batch sizes 2 and 4.

The run was stopped before paired RULER multi-value and variable-tracking
results completed, so partial results are not reported here.
