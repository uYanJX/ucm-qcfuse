# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""
SSD-backed QCFuse blend runner (vLLM-via-UCM).

Usage (flat imports resolve because the script directory is prepended to
``sys.path``; run from anywhere, or ``cd benchmarks/qcfuse`` first):
    python benchmarks/qcfuse/blend_ssd.py --model qwen3-8b \
        --dataset hotpotqa --model_dir models --baseline ours
"""

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from utils import (
    load_dataset,
    build_prompt_for_dataset,
    evaluate_sample,
    get_metric_name,
    get_system_prompt,
    get_max_new_tokens,
)

from blend_common import (
    DEFAULT_DATA_DIR,
    BLEND_SEP,
    get_critical_layers,
    set_fuserag_layers,
    set_ours_layers,
    set_prophetkv_layers,
    BlendEngineBase,
)
from qcfuse_config import (
    BASELINE_DIGEST_RATIOS,
    BLEND_BASELINES,
    DEFAULT_BLEND_RATIO,
    DEFAULT_CONTEXT_N_SINK,
    DEFAULT_CRITICAL_LAYERS,
    DIGEST_INDEX_METHOD,
    DIGEST_RATIO,
    SUPPORTED_BASELINES,
)

DIGEST_ZIP_PROMPT = "\n\nRepeat the previous context exactly."


def _packed_format_module():
    """Lazy import of the on-disk format tags (needs ucm + torch)."""
    from ucm.store.qcfuse import packed_format

    return packed_format


def _digest_index_module():
    """Lazy import of the digest-index version/method helpers (needs ucm + torch)."""
    from ucm.sparse.qcfuse import digest_index

    return digest_index


def print_final_metrics(
    model_name: str,
    dataset_name: str,
    baseline: str,
    result: dict,
    metric_name: str,
) -> None:
    summary = result["summary"]
    score_text = f"{summary['metric_mean']:.4f}"
    ttft_text = f"{summary['ttft_mean_seconds']:.4f}s"
    print(
        f"{model_name}\t{dataset_name}\t{baseline}\t"
        f"{metric_name}={score_text}\tavg_ttft={ttft_text}\t"
        f"p50={summary['ttft_p50_seconds']:.4f}s\t"
        f"p95={summary['ttft_p95_seconds']:.4f}s\t"
        f"throughput={summary['throughput_samples_per_second']:.4f} samples/s"
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _chunks(values: Sequence, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _summarize_result(result: dict) -> dict:
    metrics = result["metric"]
    ttfts = result["ttft"]
    wall_seconds = float(result["wall_seconds"])
    completed = len(metrics)
    return {
        "completed_samples": completed,
        "metric_mean": statistics.fmean(metrics) if metrics else float("nan"),
        "ttft_mean_seconds": statistics.fmean(ttfts) if ttfts else float("nan"),
        "ttft_p50_seconds": _percentile(ttfts, 0.50),
        "ttft_p95_seconds": _percentile(ttfts, 0.95),
        "qcompute_mean_seconds": (
            statistics.fmean(result["qcompute_latency"])
            if result["qcompute_latency"]
            else 0.0
        ),
        "do_blend_ttft_mean_seconds": (
            statistics.fmean(result["do_blend_ttft"])
            if result["do_blend_ttft"]
            else float("nan")
        ),
        "latency_mean_seconds": (
            statistics.fmean(result["latency"]) if result["latency"] else float("nan")
        ),
        "wall_seconds": wall_seconds,
        "throughput_samples_per_second": (
            completed / wall_seconds if wall_seconds > 0 else float("nan")
        ),
    }


@dataclass
class SSDSample:
    idx: int
    answers: List[str] = field(default_factory=list)
    prompt: str = ""
    plain_prompt: str = ""
    offline_prompt: str = ""
    query_sep: str = ""
    sample_dir_chunk: str = ""
    # query_cache stores materialized digest context KV plus critical-layer KV.
    sample_dir_query: str = ""
    query_session_id: str = ""
    has_cache: bool = False


class SSDPipelineEngine(BlendEngineBase):
    """Two-phase SSD pipeline runner."""

    def __init__(
        self,
        model_path: str,
        baseline: str = "ours",
        context_enhance: bool = False,
        cache_dir: str = "cache/qcfuse",
        digest_index_method: str = DIGEST_INDEX_METHOD,
        digest_ratio: float = DIGEST_RATIO,
        context_cache_source: str = "none",
        batch_size: int = 1,
        max_num_batched_tokens: Optional[int] = None,
    ):
        # The base constructor reads this through _kv_connector_extra_config().
        self.cache_dir = cache_dir
        super().__init__(
            model_path,
            baseline,
            kv_connector_extra_config=self._kv_connector_extra_config(),
            batch_size=batch_size,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        self.context_enhance = context_enhance
        self.context_n_sink = DEFAULT_CONTEXT_N_SINK
        self.digest_index_method = digest_index_method
        self.digest_ratio = digest_ratio
        self.context_cache_source = context_cache_source
        self.critical_layer_topk = DEFAULT_CRITICAL_LAYERS

    def _kv_connector_extra_config(self) -> dict:
        """The ``kv_connector_extra_config`` that selects the QcFuse path.

        ``UCMConnector`` dispatches to ``UCMQcFuseConnector`` when ``QcFuse`` is
        present in ``ucm_sparse_config``; the single ``UcmQcFuseStore`` entry is
        the sample-addressed SSD store. Per-request cache paths arrive later in
        vLLM's serialized ``kv_transfer_params`` request metadata.
        """
        return {
            "ucm_connectors": [
                {
                    "ucm_connector_name": "UcmQcFuseStore",
                    "ucm_connector_config": {
                        "cache_root": self.cache_dir,
                    },
                }
            ],
            "ucm_sparse_config": {
                "QcFuse": {},
            },
        }

    def _cache_paths(self, dataset_name: str, sample_idx: int) -> tuple[str, str]:
        chunk_dir, query_dir = self._cache_dataset_dirs(dataset_name)
        return (
            str(chunk_dir / f"sample_{sample_idx}"),
            str(query_dir / f"sample_{sample_idx}"),
        )

    def _cache_dataset_dirs(self, dataset_name: str) -> tuple[Path, Path]:
        base = Path(self.cache_dir)
        chunk_dir = base / "chunk_cache" / self.model_name / dataset_name
        query_dir = base / "query_cache" / self.model_name / dataset_name
        return chunk_dir, query_dir

    def _requires_query_cache(self) -> bool:
        return bool(self.context_enhance and self.context_cache_source == "query")

    def _offline_query_critical_layers(self) -> List[int]:
        if self.critical_layers:
            return [int(layer) for layer in self.critical_layers]
        return get_critical_layers(
            self.model_name,
            self._get_model_config()["num_layers"],
            critical_layers=self.critical_layer_topk,
        )

    @staticmethod
    def _has_packed_cache(sample_dir: Path) -> bool:
        meta_path = sample_dir / "kv_packed_meta.json"
        if not ((sample_dir / "kv_packed.bin").exists() and meta_path.exists()):
            return False
        meta = json.loads(meta_path.read_text())
        return meta.get("format") == _packed_format_module().PACKED_KV_FORMAT

    @staticmethod
    def _has_query_cache(sample_dir: Path) -> bool:
        meta_path = sample_dir / "query_packed_meta.json"
        if not ((sample_dir / "query_packed.bin").exists() and meta_path.exists()):
            return False
        meta = json.loads(meta_path.read_text())
        return meta.get("format") == _packed_format_module().QUERY_CACHE_FORMAT

    def _has_cache(
        self,
        sample_dir_chunk: str,
        sample_dir_query: str,
        require_query: bool = True,
        query_critical_layers: Optional[List[int]] = None,
    ) -> bool:
        chunk_dir = Path(sample_dir_chunk)
        query_dir = Path(sample_dir_query)
        if not self._has_packed_cache(chunk_dir):
            return False
        if not require_query:
            return True

        critical_layers = query_critical_layers
        if critical_layers is None:
            critical_layers = self.critical_layers
        if not critical_layers:
            return True
        if not self._has_query_cache(query_dir):
            return False

        digest_index = _digest_index_module()
        query_meta = json.loads((query_dir / "query_packed_meta.json").read_text())
        meta = query_meta.get("metadata", {})
        method = digest_index.normalize_method(self.digest_index_method)
        digest_ratio = meta.get("digest_ratio")
        try:
            digest_ratio_matches = abs(float(digest_ratio) - self.digest_ratio) < 1e-9
        except (TypeError, ValueError):
            digest_ratio_matches = False
        critical_layers = [int(x) for x in critical_layers]
        expected_qcompute_end = max(critical_layers) + 1
        critical_layers_match = [
            int(x) for x in meta.get("critical_layers", [])
        ] == critical_layers
        qcompute_end_match = meta.get("qcompute_end") == expected_qcompute_end
        index_payload = query_meta.get("indices_by_method", {}).get(method, {})
        try:
            context_n_sink_match = (
                int(index_payload.get("context_n_sink", -1)) == self.context_n_sink
            )
        except (TypeError, ValueError):
            context_n_sink_match = False
        return (
            meta.get("digest_index_version") == digest_index.DIGEST_INDEX_VERSION
            and meta.get("materialized_digest") is True
            and digest_ratio_matches
            and critical_layers_match
            and qcompute_end_match
            and query_meta.get("digest", {}).get("num_layers") == expected_qcompute_end
            and context_n_sink_match
            and set(critical_layers).issubset(
                set(int(x) for x in query_meta.get("critical", {}).get("layer_ids", []))
            )
        )

    def _ssd_args(self, sample: SSDSample) -> dict:
        return {
            "ssd_cache_path_chunk": sample.sample_dir_chunk,
            "ssd_cache_path_query": sample.sample_dir_query,
            "query_session_id": sample.query_session_id,
        }

    def _blend_args(
        self,
        blend_style: str,
        ratio: float,
        *,
        save_query_cache: bool = False,
        query_critical_layers: Optional[List[int]] = None,
    ) -> dict:
        args = {
            "blend_style": blend_style,
            "separator": BLEND_SEP,
            "start": self.start,
            "ratio": ratio,
            "method": self.method,
        }
        if self.method == "attn":
            args["attn_start"] = self.attn_start
            args["attn_end"] = self.attn_end
        uses_contextblend = save_query_cache or (
            self.context_enhance and blend_style != "KVCOMPUTE"
        )
        if uses_contextblend:
            args["is_contextblend"] = True
            if save_query_cache:
                args["context_cache_source"] = "query"
                args["digest_ratio"] = self.digest_ratio
                args["digest_index_method"] = self.digest_index_method
            else:
                args["context_cache_source"] = self.context_cache_source
            if not save_query_cache and self.context_cache_source == "query":
                args["digest_ratio"] = self.digest_ratio
                args["digest_index_method"] = self.digest_index_method
        critical_layers = (
            query_critical_layers
            if save_query_cache and query_critical_layers is not None
            else self.critical_layers
        )
        if critical_layers:
            args["critical_layers"] = [int(x) for x in critical_layers]
        if save_query_cache or blend_style == "KVCOMPUTE":
            args["context_n_sink"] = self.context_n_sink
        return args

    def _build_augmented_prompt(
        self, system_prompt: str, docs: List[str], q_prompt: List[str]
    ) -> str:
        sys_h, sys_e, asst_h = self._get_template()
        prefix = sys_h + system_prompt + sys_e
        suffix = "".join(q_prompt) + "\n\n## Answer\n" + asst_h
        parts = [prefix]
        for doc in docs:
            parts.extend([doc, DIGEST_ZIP_PROMPT])
        parts.append(suffix)
        return BLEND_SEP.join(parts)

    def _prepare_sample(
        self,
        example: Dict,
        dataset_name: str,
        sample_idx: int,
        system_prompt: str,
    ) -> SSDSample:
        answers = example.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]

        sample_dir_chunk, sample_dir_query = self._cache_paths(dataset_name, sample_idx)
        docs, q_prompt = build_prompt_for_dataset(example, dataset_name)
        prompt, query_sep = self._build_prompt(
            system_prompt, docs, q_prompt, use_sep=True
        )
        plain_prompt, _ = self._build_prompt(
            system_prompt, docs, q_prompt, use_sep=False
        )
        offline_prompt = self._build_augmented_prompt(system_prompt, docs, q_prompt)

        return SSDSample(
            idx=sample_idx,
            answers=answers,
            prompt=prompt,
            plain_prompt=plain_prompt,
            offline_prompt=offline_prompt,
            query_sep=query_sep,
            sample_dir_chunk=sample_dir_chunk,
            sample_dir_query=sample_dir_query,
            query_session_id=f"{self.model_name}:{dataset_name}:{sample_idx}",
            has_cache=self._has_cache(
                sample_dir_chunk,
                sample_dir_query,
                require_query=self._requires_query_cache(),
            ),
        )

    def _prepare_samples(
        self,
        dataset: List[Dict],
        dataset_name: str,
    ) -> List[SSDSample]:
        system_prompt = get_system_prompt(dataset_name)
        return [
            self._prepare_sample(example, dataset_name, idx, system_prompt)
            for idx, example in enumerate(dataset)
        ]

    @staticmethod
    def _qcompute_params() -> dict:
        return {"temperature": 0, "max_new_tokens": 0}

    def _append_result(
        self,
        bucket: dict,
        result: dict,
        sample: SSDSample,
        dataset_name: str,
    ) -> float:
        score = evaluate_sample(
            result["text"],
            sample.answers,
            dataset_name,
        )
        bucket["ttft"].append(result["ttft"])
        bucket["qcompute_latency"].append(result["qcompute_latency"])
        bucket["do_blend_ttft"].append(result["do_blend_ttft"])
        bucket["latency"].append(result["latency"])
        bucket["metric"].append(score)
        bucket["records"].append(
            {
                "sample_index": sample.idx,
                "metric": score,
                "ttft_seconds": result["ttft"],
                "qcompute_seconds": result["qcompute_latency"],
                "do_blend_ttft_seconds": result["do_blend_ttft"],
                "latency_seconds": result["latency"],
                "generated_tokens": result["generated_tokens"],
                "text": result["text"],
            }
        )
        return score

    def warmup_blend(
        self,
        dataset: List[Dict],
        dataset_name: str,
        ratio: float,
        num_warmup: int = 3,
    ):
        if not dataset:
            return

        sample_data = self._prepare_samples(dataset, dataset_name)
        has_qcompute = self.method == "attn"
        is_fullcomp = self.baseline == "fullcomp"
        start_idx = max(0, len(sample_data) - num_warmup)

        warmup_samples = sample_data[start_idx:]
        for batch in _chunks(warmup_samples, self.batch_size):
            if is_fullcomp:
                self._drain_generate_batch(
                    [
                        (
                            sample.plain_prompt,
                            {"temperature": 0, "max_new_tokens": 1},
                            None,
                        )
                        for sample in batch
                    ]
                )
                continue
            batch = [sample for sample in batch if sample.has_cache]
            if not batch:
                continue
            if has_qcompute:
                self._drain_generate_batch(
                    [
                        (
                            sample.query_sep,
                            self._qcompute_params(),
                            {
                                **self._blend_args("QCOMPUTE", ratio),
                                **self._ssd_args(sample),
                            },
                        )
                        for sample in batch
                    ]
                )
            self._drain_generate_batch(
                [
                    (
                        sample.prompt,
                        {"temperature": 0, "max_new_tokens": 1},
                        {
                            **self._blend_args("DO_BLEND_FINISH", ratio),
                            **self._ssd_args(sample),
                        },
                    )
                    for sample in batch
                ]
            )

    def phase1_offline(
        self,
        dataset: List[Dict],
        dataset_name: str,
    ):
        """Run KVCOMPUTE for all samples, serialize to SSD. Skip if cache exists."""
        if self.first_style != "KVCOMPUTE":
            return
        system_prompt = get_system_prompt(dataset_name)
        query_critical_layers = self._offline_query_critical_layers()

        for idx, example in enumerate(dataset):
            sample_dir_chunk, sample_dir_query = self._cache_paths(dataset_name, idx)

            has_cache = self._has_cache(
                sample_dir_chunk,
                sample_dir_query,
                require_query=True,
                query_critical_layers=query_critical_layers,
            )
            if has_cache:
                continue

            sample = self._prepare_sample(example, dataset_name, idx, system_prompt)

            self._drain_generate(
                sample.offline_prompt,
                {"temperature": 0, "max_new_tokens": 1},
                **self._blend_args(
                    self.first_style,
                    0.0,
                    save_query_cache=True,
                    query_critical_layers=query_critical_layers,
                ),
                **self._ssd_args(sample),
            )

            if self._has_cache(
                sample_dir_chunk,
                sample_dir_query,
                require_query=True,
                query_critical_layers=query_critical_layers,
            ):
                continue
            if self._has_cache(sample_dir_chunk, sample_dir_query, require_query=False):
                raise RuntimeError(
                    f"sample_{idx} query cache is incomplete: {sample_dir_query}"
                )
            raise RuntimeError(
                f"sample_{idx} SSD cache was not generated: {sample_dir_chunk}"
            )

    def phase2_online(
        self,
        dataset: List[Dict],
        dataset_name: str,
        ratio: float,
    ) -> dict:
        result_bucket = {
            "ttft": [],
            "qcompute_latency": [],
            "do_blend_ttft": [],
            "latency": [],
            "metric": [],
            "records": [],
        }

        has_qcompute = self.method == "attn"
        is_fullcomp = self.baseline == "fullcomp"

        sample_data = self._prepare_samples(dataset, dataset_name)

        max_tokens = get_max_new_tokens(dataset_name)
        params = {"temperature": 0, "max_new_tokens": max_tokens}

        started = time.perf_counter()
        if not is_fullcomp:
            missing = [sample.idx for sample in sample_data if not sample.has_cache]
            if missing:
                raise RuntimeError(
                    f"{len(missing)} samples have no complete QCFuse cache: {missing[:8]}"
                )

        total_batches = (len(sample_data) + self.batch_size - 1) // self.batch_size
        for batch_number, batch in enumerate(
            _chunks(sample_data, self.batch_size), start=1
        ):
            if is_fullcomp:
                outputs = self._timed_generate_batch(
                    [(sample.plain_prompt, params, None) for sample in batch]
                )
                for sample, output in zip(batch, outputs):
                    output["qcompute_latency"] = 0.0
                    output["do_blend_ttft"] = output["ttft"]
                    self._append_result(result_bucket, output, sample, dataset_name)
            else:
                if has_qcompute:
                    query_outputs = self._timed_generate_batch(
                        [
                            (
                                sample.query_sep,
                                self._qcompute_params(),
                                {
                                    **self._blend_args("QCOMPUTE", ratio),
                                    **self._ssd_args(sample),
                                },
                            )
                            for sample in batch
                        ]
                    )
                else:
                    query_outputs = [{"latency": 0.0} for _sample in batch]

                outputs = self._timed_generate_batch(
                    [
                        (
                            sample.prompt,
                            params,
                            {
                                **self._blend_args("DO_BLEND_FINISH", ratio),
                                **self._ssd_args(sample),
                            },
                        )
                        for sample in batch
                    ]
                )
                for sample, query_output, output in zip(batch, query_outputs, outputs):
                    q_latency = float(query_output["latency"])
                    output["qcompute_latency"] = q_latency
                    output["do_blend_ttft"] = output["ttft"]
                    output["ttft"] += q_latency
                    output["latency"] += q_latency
                    self._append_result(result_bucket, output, sample, dataset_name)
            print(
                f"[online] {dataset_name}/{self.baseline} "
                f"batch={batch_number}/{total_batches} size={len(batch)} "
                f"completed={len(result_bucket['metric'])}/{len(sample_data)}",
                flush=True,
            )

        result_bucket["wall_seconds"] = time.perf_counter() - started
        result_bucket["summary"] = _summarize_result(result_bucket)
        return result_bucket


def main():
    parser = argparse.ArgumentParser(description="SSD-backed QCFuse blend runner")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--baseline",
        default="ours",
        choices=SUPPORTED_BASELINES,
        help="Baseline to run: fullcomp, ours, fuserag, or prophetkv",
    )
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument(
        "--dataset",
        type=str,
        default="hotpotqa",
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-8b",
        help="Model name under --model_dir, or a full model path",
    )
    parser.add_argument(
        "--model_dir", type=str, default="", help="Base model directory"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache/qcfuse",
        help="Base SSD directory for KV cache storage",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=1,
        help="Number of homogeneous online requests submitted together",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        "--max_num_batched_tokens",
        type=int,
        default=None,
        help="vLLM scheduler token budget (default: 16000 * batch size)",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default="",
        help="Optional path for per-sample timings, scores, and summary",
    )
    parser.add_argument(
        "--blend-ratio",
        type=float,
        default=DEFAULT_BLEND_RATIO,
        help="Fraction of context tokens retained by online blending",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=3,
        help="Number of model and blend warmup requests (0 disables warmup)",
    )
    args = parser.parse_args()
    if not 0.0 <= args.blend_ratio <= 1.0:
        parser.error("--blend-ratio must be in [0, 1]")
    if args.num_warmup < 0:
        parser.error("--num-warmup must be non-negative")

    data_dir = Path(args.data_dir)
    model_arg = Path(args.model)
    model_path = str(
        model_arg if model_arg.is_absolute() else Path(args.model_dir) / args.model
    )
    model_name = Path(model_path).name
    dataset_name = args.dataset
    dataset_path = data_dir / f"{dataset_name}.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    origin_dataset = load_dataset(str(dataset_path))
    dataset = origin_dataset[: min(args.size, len(origin_dataset))]
    metric_name = get_metric_name(dataset_name)

    with SSDPipelineEngine(
        model_path,
        baseline=args.baseline,
        context_enhance=False,
        cache_dir=args.cache_dir,
        digest_index_method=DIGEST_INDEX_METHOD,
        digest_ratio=DIGEST_RATIO,
        context_cache_source="none",
        batch_size=args.batch_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
    ) as engine:
        engine.warmup(num_warmup=args.num_warmup)
        engine.set_baseline(args.baseline)
        if args.baseline in BLEND_BASELINES:
            engine.context_enhance = True
            engine.context_cache_source = "query"
            engine.digest_ratio = BASELINE_DIGEST_RATIOS[args.baseline]
            if args.baseline == "ours":
                set_ours_layers(engine, model_name)
            elif args.baseline == "fuserag":
                set_fuserag_layers(engine, model_name)
            elif args.baseline == "prophetkv":
                set_prophetkv_layers(engine, model_name)
        else:
            engine.context_enhance = False
            engine.context_cache_source = "none"
            engine.critical_layers = None

        if args.baseline in BLEND_BASELINES:
            engine.phase1_offline(dataset, dataset_name)
            ratio = args.blend_ratio
        else:
            ratio = 1.0

        engine.warmup_blend(dataset, dataset_name, ratio, num_warmup=args.num_warmup)
        result = engine.phase2_online(
            dataset,
            dataset_name,
            ratio,
        )
        print_final_metrics(
            model_name,
            dataset_name,
            args.baseline,
            result,
            metric_name,
        )
        if args.results_json:
            output_path = Path(args.results_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model": model_name,
                "dataset": dataset_name,
                "baseline": args.baseline,
                "batch_size": args.batch_size,
                "blend_ratio": ratio,
                "requested_samples": args.size,
                "dataset_samples": len(dataset),
                "metric_name": metric_name,
                **result,
            }
            output_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
