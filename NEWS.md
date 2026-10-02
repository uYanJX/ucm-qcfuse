# 🔥 News

- **2026.10.02** 🚀 QCFuse is compatible with Huawei's [Unified Cache Management (UCM)](https://github.com/uYanJX/ucm-qcfuse) framework and supports multi-batch inference.
- **2026.10.02** 🚀 Qwen3-32B reasoning evaluation on HotpotQA and 2WikiMQA compares FullComp, Ours, and ProphetKV at blend ratios 0.4 and 0.5.

## Qwen3-32B results

Percentages are relative ROUGE-L changes from FullComp within the same dataset and thinking setting.

| Dataset | Thinking | FullComp | Ours @ 0.4 | Ours @ 0.5 | ProphetKV @ 0.4 | ProphetKV @ 0.5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | Disabled | 0.6549 <small>(baseline)</small> | 0.6454 <small>(−1.45%)</small> | 0.6510 <small>(−0.60%)</small> | 0.6398 <small>(−2.31%)</small> | 0.6493 <small>(−0.86%)</small> |
| HotpotQA | Enabled | 0.6956 <small>(baseline)</small> | 0.6946 <small>(−0.14%)</small> | 0.6863 <small>(−1.34%)</small> | 0.6911 <small>(−0.65%)</small> | 0.6829 <small>(−1.83%)</small> |
| 2WikiMQA | Disabled | 0.5153 <small>(baseline)</small> | 0.5105 <small>(−0.93%)</small> | 0.5210 <small>(+1.11%)</small> | 0.5130 <small>(−0.45%)</small> | 0.5120 <small>(−0.64%)</small> |
| 2WikiMQA | Enabled | 0.6174 <small>(baseline)</small> | 0.5906 <small>(−4.34%)</small> | 0.6211 <small>(+0.60%)</small> | 0.5886 <small>(−4.66%)</small> | 0.5983 <small>(−3.09%)</small> |

All entries use 500 valid examples.
