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
import hashlib
import json
import os
import statistics
import time
from batch_diagnostics import query_batches
from evidence import (
    RATIO_SEMANTICS,
    baseline_identity,
    validate_dataset_rows,
    validate_frozen_manifest,
    validate_prompt_settings,
    index_token_audit,
)
from output_protocol import split_output
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
    sample_prompt_parts,
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


class GpuResidentPreloadCapacityError(RuntimeError):
    """A GPU source did not fit before submitting any online request."""


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
        f"first_token={summary['first_token_mean_seconds']:.4f}s\t"
        f"avg_decode={summary['decode_mean_seconds']:.4f}s\t"
        f"decode_total={summary['decode_total_seconds']:.4f}s\t"
        f"decode_tok_s={summary['decode_tokens_per_second']:.2f}\t"
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
    first_tokens = result["first_token"]
    decodes = result["decode"]
    generated_tokens = result["generated_tokens"]
    wall_seconds = float(result["wall_seconds"])
    completed = len(metrics)
    summary = {
        "completed_samples": completed,
        "metric_mean": statistics.fmean(metrics) if metrics else float("nan"),
        "ttft_mean_seconds": statistics.fmean(ttfts) if ttfts else float("nan"),
        "ttft_p50_seconds": _percentile(ttfts, 0.50),
        "ttft_p95_seconds": _percentile(ttfts, 0.95),
        "first_token_mean_seconds": (
            statistics.fmean(first_tokens) if first_tokens else float("nan")
        ),
        "first_token_p50_seconds": _percentile(first_tokens, 0.50),
        "first_token_p95_seconds": _percentile(first_tokens, 0.95),
        "decode_mean_seconds": statistics.fmean(decodes) if decodes else float("nan"),
        "decode_total_seconds": sum(decodes),
        "decode_p50_seconds": _percentile(decodes, 0.50),
        "decode_p95_seconds": _percentile(decodes, 0.95),
        "decode_generated_tokens": sum(result["decode_tokens"]),
        "aggregate_output_tokens_per_second": (
            sum(generated_tokens) / wall_seconds if wall_seconds > 0 else 0.0
        ),
        "decode_tokens_per_second": (
            sum(result["decode_tokens"]) / sum(decodes)
            if decodes and sum(decodes) > 0
            else 0.0
        ),
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
    records = result.get("records", [])
    summary["per_sequence_decode_tokens_per_second"] = summary[
        "decode_tokens_per_second"
    ]
    summary["decode_rate_scope"] = (
        "tokens_divided_by_sum_of_per_request_decode_durations"
    )
    if "decode_batch_wall_seconds" in result:
        span = float(result["decode_batch_wall_seconds"])
        summary["decode_batch_wall_seconds"] = span
        summary["aggregate_decode_tokens_per_second"] = (
            sum(result["decode_tokens"]) / span if span > 0 else 0.0
        )
        summary["aggregate_decode_scope"] = (
            "sum_of_batch_first_output_to_last_completion_windows"
        )
    if records:
        for kind in ("thinking", "answer", "control", "total_output"):
            summary[f"{kind}_tokens_total"] = sum(
                record[f"{kind}_tokens"] for record in records
            )
        summary["unfinished_thinking_samples"] = sum(
            record["thinking_status"] == "unfinished" for record in records
        )
        for name in ("thinking", "answer"):
            values = [
                record[f"{name}_ttft_seconds"]
                for record in records
                if record[f"{name}_ttft_seconds"] is not None
            ]
            summary[f"{name}_ttft_mean_seconds"] = (
                statistics.fmean(values) if values else None
            )
            summary[f"{name}_ttft_observed_samples"] = len(values)
    return summary


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
    input_tokens: int = 0
    context_tokens: int = 0
    prompt_hash: str = ""
    category: str = ""
    truncated: bool = False
    prompt_token_ids: List[int] = field(default_factory=list)
    prompt_token_ids_hash: str = ""
    ruler_task: Optional[str] = None
    max_new_tokens: int = 48


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
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 131072,
        block_size: int = 512,
        rope_factor: Optional[float] = 4.0,
        storage_mode: str = "ssd",
        kv_cache_memory_bytes: Optional[int] = None,
        gpu_headroom_bytes_per_token: int = 0,
        gpu_headroom_fixed_bytes: int = 0,
        gpu_resident_codec: Optional[str] = None,
        gpu_query_alias: bool = False,
        decode_cudagraph: bool = False,
        decode_cudagraph_mode: str = "full",
        qcompute_batch_size: Optional[int] = None,
    ):
        if qcompute_batch_size is not None and (
            not isinstance(qcompute_batch_size, int)
            or not 1 <= qcompute_batch_size <= batch_size
        ):
            raise ValueError("QCOMPUTE batch size must be between 1 and batch_size")
        self.qcompute_batch_size = qcompute_batch_size
        # The base constructor reads this through _kv_connector_extra_config().
        self.cache_dir = cache_dir
        self.storage_mode = storage_mode
        if (gpu_resident_codec or gpu_query_alias) and (
            storage_mode != "gpu" or baseline == "fullcomp"
        ):
            raise ValueError("Lossless residency options require a GPU cached method")
        self.gpu_resident_codec = gpu_resident_codec
        self.gpu_query_alias = gpu_query_alias
        if gpu_headroom_bytes_per_token < 0 or gpu_headroom_fixed_bytes < 0:
            raise ValueError("GPU headroom per token must be nonnegative")
        self.gpu_headroom_bytes_per_token = gpu_headroom_bytes_per_token
        self.gpu_headroom_fixed_bytes = gpu_headroom_fixed_bytes
        super().__init__(
            model_path,
            baseline,
            kv_connector_extra_config=self._kv_connector_extra_config(),
            batch_size=batch_size,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            block_size=block_size,
            rope_factor=rope_factor,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            decode_cudagraph=decode_cudagraph,
            decode_cudagraph_mode=decode_cudagraph_mode,
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
        connector_config = {
            "cache_root": self.cache_dir,
            "storage_mode": self.storage_mode,
            "resident_preload_policy": "batch",
        }
        profile_path = os.getenv("QCFUSE_PROFILE_PATH")
        if profile_path:
            connector_config["profile_path"] = profile_path
        return {
            "ucm_connectors": [
                {
                    "ucm_connector_name": "UcmQcFuseStore",
                    "ucm_connector_config": connector_config,
                }
            ],
            "ucm_sparse_config": {
                "QcFuse": {},
            },
        }

    def configure_baseline(self, baseline, critical_layers=DEFAULT_CRITICAL_LAYERS):
        self.set_baseline(baseline)
        if baseline in BLEND_BASELINES:
            self.context_enhance = True
            self.context_cache_source = "query"
            self.digest_ratio = BASELINE_DIGEST_RATIOS[baseline]
            if baseline == "ours":
                self.critical_layer_topk = critical_layers
                set_ours_layers(self, self.model_name, critical_layers)
            elif baseline == "fuserag":
                set_fuserag_layers(self, self.model_name)
            else:
                set_prophetkv_layers(self, self.model_name)
        else:
            self.context_enhance = False
            self.context_cache_source = "none"
            self.critical_layers = None

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
            "offline_kv_device": getattr(self, "offline_kv_device", "gpu"),
        }
        if getattr(self, "token_audit_path", None):
            args["token_audit_path"] = self.token_audit_path
        if getattr(self, "token_counts_enabled", False) and blend_style != "KVCOMPUTE":
            args["token_counts"] = True
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
        self,
        system_prompt: str,
        docs: List[str],
        q_prompt: List[str],
        answer_suffix: str = "\n\n## Answer\n",
        answer_prefix: str = "",
    ) -> str:
        sys_h, sys_e, asst_h = self._get_template()
        prefix = sys_h + system_prompt + sys_e
        suffix = self._query_suffix(q_prompt, answer_suffix, asst_h, answer_prefix)
        parts = [prefix]
        for doc in docs:
            parts.extend([doc, DIGEST_ZIP_PROMPT])
        parts.append(suffix)
        return BLEND_SEP.join(parts)

    def _fit_context_to_model_window(
        self,
        system_prompt: str,
        docs: List[str],
        q_prompt: List[str],
        **prompt_options,
    ) -> tuple[List[str], bool]:
        """Reject overlong inputs; never silently alter benchmark evidence."""
        prompt, _ = self._build_prompt(
            system_prompt, docs, q_prompt, use_sep=True, **prompt_options
        )
        input_ids, _, _ = self._prepare_generation(prompt, {}, {})
        if len(input_ids) > self.context_length:
            raise ValueError(
                f"Complete input has {len(input_ids)} tokens, exceeding "
                f"max_model_len={self.context_length}; truncation is disabled"
            )
        return docs, False

    def _prepare_sample(
        self,
        example: Dict,
        dataset_name: str,
        sample_idx: int,
        system_prompt: str,
    ) -> SSDSample:
        answers = example.get("answers", [])
        if not answers and dataset_name == "longbench_v2":
            answer = example.get("answer", "")
            answers = [answer] if answer != "" else []
        if not answers and example.get("answer", "") != "":
            answers = [example["answer"]]
        if isinstance(answers, str):
            answers = [answers]

        sample_dir_chunk, sample_dir_query = self._cache_paths(dataset_name, sample_idx)
        system_prompt, docs, q_prompt, prompt_options = sample_prompt_parts(
            example, dataset_name
        )
        docs, truncated = self._fit_context_to_model_window(
            system_prompt, docs, q_prompt, **prompt_options
        )
        prompt, query_sep = self._build_prompt(
            system_prompt, docs, q_prompt, use_sep=True, **prompt_options
        )
        plain_prompt, _ = self._build_prompt(
            system_prompt, docs, q_prompt, use_sep=False, **prompt_options
        )
        offline_prompt = self._build_augmented_prompt(
            system_prompt, docs, q_prompt, **prompt_options
        )
        # Full Prefill must receive exactly the same token IDs as DO_BLEND.
        # Concatenating text before BPE can otherwise merge across boundaries.
        input_ids, _, _ = self._prepare_generation(prompt, {}, {})
        context_text = "".join(docs)
        context_ids = self.tokenizer(
            context_text,
            add_special_tokens=False,
            return_attention_mask=False,
            verbose=False,
        )["input_ids"]
        prompt_hash = hashlib.sha256(plain_prompt.encode("utf-8")).hexdigest()
        category = str(
            example.get("sub_domain")
            or example.get("domain")
            or example.get("category")
            or dataset_name
        )
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
            input_tokens=len(input_ids),
            context_tokens=len(context_ids),
            prompt_hash=prompt_hash,
            category=category,
            ruler_task=example.get("ruler_task"),
            max_new_tokens=int(
                getattr(self, "max_new_tokens_override", None)
                or example.get("max_new_tokens", get_max_new_tokens(dataset_name))
            ),
            truncated=truncated,
            prompt_token_ids=input_ids,
            prompt_token_ids_hash=hashlib.sha256(
                json.dumps(input_ids, separators=(",", ":")).encode()
            ).hexdigest(),
        )

    def _prepare_samples(
        self,
        dataset: List[Dict],
        dataset_name: str,
    ) -> List[SSDSample]:
        system_prompt = get_system_prompt(dataset_name)
        return [
            self._prepare_sample(example, dataset_name, idx, system_prompt)
            for idx, example in enumerate(
                dataset, start=int(getattr(self, "sample_index_offset", 0))
            )
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
        output_parts = split_output(
            result["output_token_ids"],
            self.tokenizer,
            getattr(self, "thinking_mode", "off") == "on",
        )
        actual_budget = result.get("sampling_max_tokens")
        if actual_budget is not None and actual_budget != sample.max_new_tokens:
            raise ValueError(
                "Effective sampling output budget differs from frozen sample"
            )
        if output_parts["total_output_tokens"] != result["generated_tokens"]:
            raise ValueError("Observed output token IDs disagree with generated count")
        score = evaluate_sample(
            output_parts["answer_text"],
            sample.answers,
            dataset_name,
            ruler_task=sample.ruler_task,
        )
        bucket["ttft"].append(result["ttft"])
        bucket["first_token"].append(result.get("first_token_seconds", result["ttft"]))
        bucket["decode"].append(result["decode"])
        bucket["generated_tokens"].append(result["generated_tokens"])
        bucket["decode_tokens"].append(result["decode_tokens"])
        bucket["qcompute_latency"].append(result["qcompute_latency"])
        bucket["do_blend_ttft"].append(result["do_blend_ttft"])
        bucket["latency"].append(result["latency"])
        bucket["metric"].append(score)
        bucket["records"].append(
            {
                "sample_index": sample.idx,
                "request_id": result.get("request_id"),
                "qcompute_request_id": result.get("qcompute_request_id"),
                "metric": score,
                "ttft_seconds": result["ttft"],
                "engine_phase_ttft_seconds": result["engine_phase_ttft"],
                "ttft_scope": "batch_submission_to_first_token_including_qcompute_barrier",
                "first_token_seconds": result.get(
                    "first_token_seconds", result["ttft"]
                ),
                "decode_seconds": result["decode"],
                "decode_tokens_per_second": (
                    result["decode_tokens"] / result["decode"]
                    if result["decode"] > 0
                    else 0.0
                ),
                "qcompute_seconds": result["qcompute_latency"],
                "do_blend_ttft_seconds": result["do_blend_ttft"],
                "latency_seconds": result["latency"],
                "generated_tokens": result["generated_tokens"],
                "output_token_ids_sha256": hashlib.sha256(
                    json.dumps(
                        result["output_token_ids"], separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "decode_tokens": result["decode_tokens"],
                "first_token_output_tokens": result["first_token_output_tokens"],
                "ruler_task": sample.ruler_task,
                "text": result["text"],
                **output_parts,
                "finish_reason": result.get("finish_reason"),
                "max_new_tokens": sample.max_new_tokens,
                "sampling_max_tokens": actual_budget,
                "thinking_ttft_seconds": result.get("thinking_ttft"),
                "answer_ttft_seconds": result.get("answer_ttft"),
                "input_tokens": sample.input_tokens,
                "context_tokens": sample.context_tokens,
                "prompt_hash": sample.prompt_hash,
                "prompt_token_ids_hash": sample.prompt_token_ids_hash,
                "category": sample.category,
                "truncated": sample.truncated,
            }
        )
        return score

    @staticmethod
    def _batch_decode_window(records):
        if not records:
            return 0.0
        return max(
            0.0,
            max(r["latency_seconds"] for r in records)
            - min(r["ttft_seconds"] for r in records),
        )

    def _prepare_resident_batch(self, samples):
        if self.storage_mode == "ssd" or self.baseline == "fullcomp":
            return None

        paths = sorted(
            {
                str(path)
                for sample in samples
                for directory in (sample.sample_dir_chunk, sample.sample_dir_query)
                for path in Path(directory).rglob("*.bin")
            }
        )
        if not paths:
            raise RuntimeError("Resident batch has no generated KV files")
        started = time.perf_counter()
        headroom_bytes = (
            sum(sample.input_tokens for sample in samples)
            * getattr(self, "gpu_headroom_bytes_per_token", 0)
            + getattr(self, "gpu_headroom_fixed_bytes", 0)
            if self.storage_mode == "gpu"
            else 0
        )
        try:
            args = (paths, self.storage_mode, headroom_bytes)
            gpu_codec = getattr(self, "gpu_resident_codec", None)
            gpu_alias = getattr(self, "gpu_query_alias", False)
            if gpu_codec or gpu_alias:
                pairs = (
                    {
                        str(Path(sample.sample_dir_query) / "query_packed.bin"): str(
                            Path(sample.sample_dir_chunk) / "kv_packed.bin"
                        )
                        for sample in samples
                    }
                    if gpu_alias
                    else {}
                )
                args += (gpu_codec, pairs)
            workers = self.llm.collective_rpc(
                "qcfuse_prepare_resident_batch",
                args=args,
            )
        except Exception as exc:
            if self.storage_mode == "gpu" and "cuda out of memory" in str(exc).lower():
                raise GpuResidentPreloadCapacityError(str(exc)) from exc
            raise
        return {
            "wall_seconds": time.perf_counter() - started,
            "workers": workers,
            "files": len(paths),
            "excluded_from_ttft": True,
            "online_headroom_bytes": headroom_bytes,
        }

    def release_resident_batch(self):
        if self.storage_mode in ("cpu", "gpu") and self.baseline != "fullcomp":
            self.llm.collective_rpc(
                "qcfuse_prepare_resident_batch", args=([], self.storage_mode)
            )

    def _worker_snapshot(self, reset_peak):
        if not getattr(self, "worker_snapshots_enabled", False):
            return None, 0.0
        started = time.perf_counter()
        if reset_peak and getattr(self, "forward_observer_enabled", False):
            self.llm.collective_rpc("qcfuse_start_forward_observer")
        workers = self.llm.collective_rpc(
            "qcfuse_benchmark_snapshot", args=(reset_peak,)
        )
        return workers, time.perf_counter() - started

    def warmup_blend(
        self,
        dataset: List[Dict],
        dataset_name: str,
        ratio: float,
        num_warmup: int = 3,
    ):
        warmup_started = time.perf_counter()
        prepares = []
        if not dataset:
            return {"warmup_wall_seconds": 0.0, "resident_prepares": [], "repeats": 0}

        sample_data = self._prepare_samples(dataset, dataset_name)
        has_qcompute = self.method == "attn"
        is_fullcomp = self.baseline == "fullcomp"
        for repeat in range(num_warmup):
            size = min(self.batch_size, len(sample_data))
            batch = [
                sample_data[(repeat * size + i) % len(sample_data)] for i in range(size)
            ]
            if is_fullcomp:
                self._drain_generate_batch(
                    [
                        (
                            sample.prompt_token_ids,
                            {"temperature": 0, "max_new_tokens": sample.max_new_tokens},
                            None,
                        )
                        for sample in batch
                    ]
                )
                continue
            batch = [sample for sample in batch if sample.has_cache]
            if not batch:
                continue
            prepare = self._prepare_resident_batch(batch)
            if prepare:
                prepares.append({"repeat": repeat, **prepare})
            if has_qcompute:
                groups = query_batches(
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
                    ],
                    getattr(self, "qcompute_batch_size", None),
                )
                for group in groups:
                    self._drain_generate_batch(group)
            self._drain_generate_batch(
                [
                    (
                        sample.prompt,
                        {"temperature": 0, "max_new_tokens": sample.max_new_tokens},
                        {
                            **self._blend_args("DO_BLEND_FINISH", ratio),
                            **self._ssd_args(sample),
                        },
                    )
                    for sample in batch
                ]
            )
        return {
            "warmup_wall_seconds": time.perf_counter() - warmup_started,
            "resident_prepares": prepares,
            "repeats": num_warmup,
        }

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

        for idx, example in enumerate(
            dataset, start=int(getattr(self, "sample_index_offset", 0))
        ):
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
            "first_token": [],
            "decode": [],
            "generated_tokens": [],
            "decode_tokens": [],
            "qcompute_latency": [],
            "do_blend_ttft": [],
            "latency": [],
            "metric": [],
            "records": [],
            "phase_timings": [],
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
        resident_prepares = []
        worker_snapshots = []
        result_bucket["decode_batch_wall_seconds"] = 0.0
        control_wall = 0.0
        for batch_number, batch in enumerate(
            _chunks(sample_data, self.batch_size), start=1
        ):
            prepare = self._prepare_resident_batch(batch)
            if prepare:
                resident_prepares.append(prepare)
            before, elapsed = self._worker_snapshot(True)
            control_wall += elapsed
            if is_fullcomp:
                phase_started = time.perf_counter()
                outputs = self._timed_generate_batch(
                    [
                        (
                            sample.prompt_token_ids,
                            {**params, "max_new_tokens": sample.max_new_tokens},
                            None,
                        )
                        for sample in batch
                    ]
                )
                full_wall = time.perf_counter() - phase_started
                result_bucket["phase_timings"].append(
                    {
                        "batch": batch_number,
                        "size": len(batch),
                        "full_prefill_wall_seconds": full_wall,
                    }
                )
                for sample, output in zip(batch, outputs):
                    output["engine_phase_ttft"] = output["ttft"]
                    output["qcompute_latency"] = 0.0
                    output["do_blend_ttft"] = output["ttft"]
                    output["ttft"] = (
                        output["first_token_time"] or output["finished_time"]
                    ) - phase_started
                    output["first_token_seconds"] = output["ttft"]
                    output["latency"] = output["finished_time"] - phase_started
                    for name in ("thinking", "answer"):
                        timestamp = output.get(f"first_{name}_time")
                        output[f"{name}_ttft"] = (
                            timestamp - phase_started if timestamp is not None else None
                        )
                    self._append_result(result_bucket, output, sample, dataset_name)
            else:
                if has_qcompute:
                    q_started = time.perf_counter()
                    groups = query_batches(
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
                        ],
                        getattr(self, "qcompute_batch_size", None),
                    )
                    query_outputs = []
                    for group in groups:
                        group_outputs = self._timed_generate_batch(group)
                        if len(group_outputs) != len(group):
                            raise RuntimeError("QCOMPUTE response count differs")
                        query_outputs.extend(group_outputs)
                    q_wall = time.perf_counter() - q_started
                else:
                    groups = []
                    query_outputs = [{"latency": 0.0} for _sample in batch]
                    q_wall = 0.0

                blend_started = time.perf_counter()
                outputs = self._timed_generate_batch(
                    [
                        (
                            sample.prompt,
                            {**params, "max_new_tokens": sample.max_new_tokens},
                            {
                                **self._blend_args("DO_BLEND_FINISH", ratio),
                                **self._ssd_args(sample),
                            },
                        )
                        for sample in batch
                    ]
                )
                blend_wall = time.perf_counter() - blend_started
                result_bucket["phase_timings"].append(
                    {
                        "batch": batch_number,
                        "size": len(batch),
                        "qcompute_wall_seconds": q_wall,
                        "qcompute_group_sizes": [len(group) for group in groups],
                        "do_blend_wall_seconds": blend_wall,
                        "qcompute_output_latency_seconds": [
                            float(item.get("latency", 0.0)) for item in query_outputs
                        ],
                        "do_blend_first_token_seconds": [
                            float(
                                item.get("first_token_seconds", item.get("ttft", 0.0))
                            )
                            for item in outputs
                        ],
                    }
                )
                for sample, query_output, output in zip(batch, query_outputs, outputs):
                    output["qcompute_request_id"] = query_output.get("request_id")
                    q_latency = float(query_output["latency"])
                    output["qcompute_latency"] = q_latency
                    output["do_blend_ttft"] = output["ttft"]
                    output["engine_phase_ttft"] = output["ttft"] + q_latency
                    pipeline_started = q_started if has_qcompute else blend_started
                    output["ttft"] = (
                        output["first_token_time"] or output["finished_time"]
                    ) - pipeline_started
                    output["first_token_seconds"] = output["ttft"]
                    output["latency"] = output["finished_time"] - pipeline_started
                    for name in ("thinking", "answer"):
                        timestamp = output.get(f"first_{name}_time")
                        output[f"{name}_ttft"] = (
                            timestamp - pipeline_started
                            if timestamp is not None
                            else None
                        )
                    self._append_result(result_bucket, output, sample, dataset_name)
            after, elapsed = self._worker_snapshot(False)
            if not is_fullcomp and getattr(self, "token_counts_enabled", False):
                observed = index_token_audit(
                    row
                    for worker in after or []
                    for row in worker.get("token_work", [])
                )
                for record in result_bucket["records"][-len(batch) :]:
                    record["token_work"] = observed[record["request_id"]]
                    if has_qcompute:
                        record["qcompute_token_work"] = observed[
                            record["qcompute_request_id"]
                        ]
            result_bucket["decode_batch_wall_seconds"] += self._batch_decode_window(
                result_bucket["records"][-len(batch) :]
            )
            control_wall += elapsed
            if before is not None:
                worker_snapshots.append(
                    {
                        "batch": batch_number,
                        "before": before,
                        "after": after,
                        "expected_full_prefill_input_tokens": sum(
                            sample.input_tokens for sample in batch
                        ),
                    }
                )
            print(
                f"[online] {dataset_name}/{self.baseline} "
                f"batch={batch_number}/{total_batches} size={len(batch)} "
                f"completed={len(result_bucket['metric'])}/{len(sample_data)}",
                flush=True,
            )

        physical_wall = time.perf_counter() - started
        prepare_wall = sum(item["wall_seconds"] for item in resident_prepares)
        result_bucket["online_including_resident_preload_wall_seconds"] = physical_wall
        result_bucket["resident_prepare_wall_seconds"] = prepare_wall
        result_bucket["resident_prepares"] = resident_prepares
        result_bucket["worker_snapshots"] = worker_snapshots
        result_bucket["measurement_control_wall_seconds"] = control_wall
        result_bucket["wall_seconds"] = physical_wall - prepare_wall - control_wall
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
        help="vLLM scheduler token budget (default: max_model_len * batch size)",
    )
    parser.add_argument(
        "--max-model-len",
        "--max_model_len",
        type=int,
        default=131072,
        help="vLLM maximum model length (default: 131072)",
    )
    parser.add_argument(
        "--block-size",
        "--block_size",
        type=int,
        default=512,
        help="vLLM KV block size (default: 512)",
    )
    parser.add_argument(
        "--rope-factor",
        "--rope_factor",
        type=float,
        default=4.0,
        help="YaRN rope scaling factor; use 0 to keep the checkpoint config",
    )
    parser.add_argument(
        "--storage-mode",
        choices=("ssd", "cpu", "gpu"),
        default="ssd",
        help="KV cache source; CPU/GPU preload from an offline-generated cache before online timing",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of one GPU available to vLLM (default: 0.9)",
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
        help="Document-token recompute selection budget; query and earlier layers add work",
    )
    parser.add_argument(
        "--critical-layers",
        type=int,
        default=DEFAULT_CRITICAL_LAYERS,
        help="QCFuse attention signal layers; -1 selects every layer",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=3,
        help="Number of model and blend warmup requests (0 disables warmup)",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "kvcompute", "online"),
        default="all",
        help=(
            "Run both phases, only generate KVCOMPUTE SSD cache, "
            "or only run online DO_BLEND"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Frozen sample/model manifest; verified before engine startup",
    )
    parser.add_argument("--allow-unverified-baseline", action="store_true")
    parser.add_argument("--thinking", choices=("on", "off"), default="off")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--token-audit",
        action="store_true",
        help="Record per-layer token work; diagnostic run, not timing baseline",
    )
    args = parser.parse_args()
    try:
        identity = baseline_identity(args.baseline, args.allow_unverified_baseline)
    except ValueError as error:
        parser.error(str(error))
    if args.token_audit and not args.results_json:
        parser.error("--token-audit requires --results-json")
    if not 0.0 <= args.blend_ratio <= 1.0:
        parser.error("--blend-ratio must be in [0, 1]")
    if args.num_warmup < 0:
        parser.error("--num-warmup must be non-negative")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.max_model_len < 1 or args.block_size < 1:
        parser.error("--max-model-len and --block-size must be positive")
    if args.phase == "kvcompute" and args.baseline not in BLEND_BASELINES:
        parser.error("--phase kvcompute requires a QCFuse baseline")

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

    dataset = load_dataset(str(dataset_path), limit=args.size)
    validate_dataset_rows(dataset, dataset_name)
    if args.manifest:
        validate_prompt_settings(
            json.loads(Path(args.manifest).read_text()),
            args.thinking,
            args.max_new_tokens,
        )
    provenance = (
        validate_frozen_manifest(args.manifest, dataset, model_path, args.max_model_len)
        if args.manifest
        else {}
    )
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
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        rope_factor=args.rope_factor if args.rope_factor > 0 else None,
        storage_mode=args.storage_mode,
    ) as engine:
        engine.thinking_mode = args.thinking
        engine.max_new_tokens_override = args.max_new_tokens
        audit_path = None
        if args.token_audit:
            import uuid

            audit_path = (
                str(Path(args.results_json).resolve())
                + f".{uuid.uuid4().hex}.tokens.jsonl"
            )
            engine.token_audit_path = audit_path
        engine.warmup(num_warmup=args.num_warmup)
        engine.configure_baseline(args.baseline, args.critical_layers)

        ratio = args.blend_ratio if args.baseline in BLEND_BASELINES else 1.0
        if args.phase in ("all", "kvcompute") and args.baseline in BLEND_BASELINES:
            engine.phase1_offline(dataset, dataset_name)

        if args.phase == "kvcompute":
            cache_status = []
            for idx in range(len(dataset)):
                chunk_dir, query_dir = (
                    Path(path) for path in engine._cache_paths(dataset_name, idx)
                )
                cache_status.append(
                    {
                        "sample_index": idx,
                        "chunk_dir": str(chunk_dir),
                        "query_dir": str(query_dir),
                        "chunk_exists": chunk_dir.exists(),
                        "query_exists": query_dir.exists(),
                        "chunk_bytes": (
                            sum(
                                path.stat().st_size
                                for path in chunk_dir.rglob("*")
                                if path.is_file()
                            )
                            if chunk_dir.exists()
                            else 0
                        ),
                        "query_bytes": (
                            sum(
                                path.stat().st_size
                                for path in query_dir.rglob("*")
                                if path.is_file()
                            )
                            if query_dir.exists()
                            else 0
                        ),
                    }
                )
            payload = {
                "model": model_name,
                "dataset": dataset_name,
                **identity,
                "phase": args.phase,
                "ratio_semantics": RATIO_SEMANTICS,
                "requested_samples": args.size,
                "dataset_samples": len(dataset),
                "blend_ratio": ratio,
                "storage_mode": args.storage_mode,
                "max_model_len": args.max_model_len,
                "block_size": args.block_size,
                "rope_factor": args.rope_factor,
                "rope_original_max_position_embeddings": (
                    engine._original_max_position_embeddings
                ),
                "cache_status": cache_status,
                "memory": engine.gpu_memory_snapshot(),
            }
            print(
                f"[kvcompute] generated {len(cache_status)} sample cache entries "
                f"under {args.cache_dir}"
            )
            if args.results_json:
                output_path = Path(args.results_json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(payload, indent=2) + "\n")
            return

        if args.phase in ("all", "online"):
            engine.warmup_blend(
                dataset, dataset_name, ratio, num_warmup=args.num_warmup
            )
        result = engine.phase2_online(dataset, dataset_name, ratio)
        result["memory"] = engine.gpu_memory_snapshot()
        print_final_metrics(
            model_name,
            dataset_name,
            identity["baseline"],
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
                "storage_mode": args.storage_mode,
                "max_model_len": args.max_model_len,
                "block_size": args.block_size,
                "rope_factor": args.rope_factor,
                "rope_original_max_position_embeddings": (
                    engine._original_max_position_embeddings
                ),
                "requested_samples": args.size,
                "dataset_samples": len(dataset),
                "metric_name": metric_name,
                **result,
            }
            payload.update(provenance)
            payload["prompt_settings"] = {
                "thinking": args.thinking,
                "max_new_tokens_override": args.max_new_tokens,
            }
            payload.update(identity)
            payload["ratio_semantics"] = (
                RATIO_SEMANTICS if args.baseline != "fullcomp" else "full_prefill"
            )
            payload["measurement_kind"] = (
                "token_audit" if args.token_audit else "benchmark"
            )
            payload["token_audit_path"] = audit_path
            if audit_path and args.baseline in BLEND_BASELINES:
                rows = [
                    json.loads(line)
                    for line in Path(audit_path).read_text().splitlines()
                ]
                by_id = index_token_audit(rows)
                for record in payload["records"]:
                    # Missing observations are an error, never estimated from ratio.
                    record["token_work"] = by_id[record["request_id"]]
                    record["qcompute_token_work"] = by_id[record["qcompute_request_id"]]
            output_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
