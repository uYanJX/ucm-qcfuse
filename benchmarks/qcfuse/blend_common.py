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
Shared constants, utilities, and base class for blend test scripts.

The pure helpers at the top (templates, critical layers, prompt tokenization)
are import-light on purpose: they run on CPU in the unit tests.  ``transformers``
and ``vllm`` are imported lazily inside :class:`BlendEngineBase.__init__` so the
module itself can be imported without a GPU host.
"""

import copy
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from qcfuse_config import (
    DEFAULT_CRITICAL_LAYERS,
    MODEL_TOP10_CRITICAL_LAYERS,
    SUPPORTED_BASELINES,
)

DEFAULT_DATA_DIR = Path(__file__).parent
# Frontend-only delimiter. The tokenizer path splits on this string before
# tokenization, so it must not rely on whitespace to avoid token merges.
BLEND_SEP = "<|blendsep|>"

# Model template configurations
TEMPLATES = {
    "llama": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    "mistral": ("<s>[INST]", "", "[/INST]"),
    "qwen": (
        "<|im_start|>system\n",
        "<|im_end|>\n<|im_start|>user\n",
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ),
}

# Blend baseline configurations: (style, start, method)
BLEND_CONFIG = {
    "fullcomp": ("FULLCOMPUTE", 0, "none"),
    "ours": ("KVCOMPUTE", 0, "attn"),
    "fuserag": ("KVCOMPUTE", 0, "attn"),
    "prophetkv": ("KVCOMPUTE", 0, "attn"),
}


def _critical_model_key(model_name: str) -> str:
    name_lower = model_name.lower()
    if name_lower.startswith("qwen3-8b"):
        return "qwen3-8b"
    elif name_lower.startswith("qwen3-14b"):
        return "qwen3-14b"
    elif name_lower.startswith("qwen3-32b"):
        return "qwen3-32b"
    elif name_lower.startswith("llama"):
        return "llama3.1-8b"
    elif name_lower.startswith("mistral"):
        return "mistral-7b"
    raise ValueError(f"critical layers are not configured for model {model_name}")


def get_critical_layers(
    model_name: str, num_layers: int, critical_layers: int = DEFAULT_CRITICAL_LAYERS
) -> List[int]:
    """Return Top-K critical layers as 0-based indices.

    critical_layers=-1 selects every layer.
    """
    critical_layers = int(critical_layers)
    if critical_layers == -1:
        return list(range(num_layers))

    model_key = _critical_model_key(model_name)
    top_layers = MODEL_TOP10_CRITICAL_LAYERS[model_key]
    if critical_layers < 1 or critical_layers > len(top_layers):
        raise ValueError(
            f"critical_layers must be -1 or an integer in [1, {len(top_layers)}], "
            f"got {critical_layers}"
        )

    return _validate_explicit_layers(
        model_name,
        num_layers,
        top_layers[:critical_layers],
    )


def _validate_explicit_layers(
    model_name: str, num_layers: int, layers: List[int]
) -> List[int]:
    invalid_layers = [layer for layer in layers if layer < 0 or layer >= num_layers]
    if invalid_layers:
        raise ValueError(
            f"critical layers {invalid_layers} are out of range for "
            f"model {model_name} with {num_layers} layers"
        )
    return layers


def _tokenize_segment(tokenizer, text: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        return tokenizer.encode(text, add_special_tokens=False)
    encoded = tokenizer(text, add_special_tokens=False)
    return encoded["input_ids"]


def tokenize_with_sep(tokenizer, text: str, sep_token: List[int]) -> List[int]:
    """Tokenize ``text`` segment-wise, inserting ``sep_token`` at each delimiter.

    SGLang's tokenizer manager splits the prompt on the frontend delimiter
    *before* tokenization (``split_text_tokens``) so BPE/SentencePiece cannot
    merge the delimiter with adjacent text.  The UCM connector recovers chunk
    boundaries by matching ``sep_token`` inside the token ids
    (``blend_info.split_tokens``), so we additionally insert the literal
    ``sep_token`` at every boundary, making each boundary an unambiguous
    subsequence.
    """
    ids: List[int] = []
    for i, part in enumerate(text.split(BLEND_SEP)):
        if i:
            ids.extend(sep_token)
        ids.extend(_tokenize_segment(tokenizer, part))
    return ids


def _set_critical_layers(engine, model_name: str, layers: List[int]) -> None:
    layers = _validate_explicit_layers(
        model_name,
        engine._get_model_config()["num_layers"],
        [int(layer) for layer in layers],
    )
    engine.critical_layers = layers
    engine.attn_start, engine.attn_end = 0, max(layers) + 1


def set_ours_layers(
    engine, model_name: str, critical_layers: int = DEFAULT_CRITICAL_LAYERS
):
    """Set the critical layer Top-K used by the ours baseline."""
    num_layers = engine._get_model_config()["num_layers"]
    layers = get_critical_layers(
        model_name, num_layers, critical_layers=critical_layers
    )
    _set_critical_layers(engine, model_name, layers)


def set_fuserag_layers(engine, model_name: str):
    """Set the last-layer signal used by the FUSE-RAG baseline."""
    num_layers = engine._get_model_config()["num_layers"]
    _set_critical_layers(engine, model_name, [num_layers - 1])


def set_prophetkv_layers(engine, model_name: str):
    """Set all-layer signals used by the ProphetKV baseline."""
    num_layers = engine._get_model_config()["num_layers"]
    _set_critical_layers(engine, model_name, list(range(num_layers)))


def _unwrap_engine(llm):
    """Return the underlying vLLM engine from an ``LLM`` wrapper."""
    for attr in ("llm_engine", "engine"):
        engine = getattr(llm, attr, None)
        if engine is not None:
            return engine
    raise RuntimeError("Could not locate the vLLM engine on the LLM wrapper")


class BlendEngineBase:
    """Base class with shared blend engine functionality (vLLM-via-UCM)."""

    # UCM connector entry point (the QcFuse dispatch lives inside ``UCMConnector``).
    kv_connector_name = "UCMConnector"
    kv_connector_module_path = "ucm.integration.vllm.ucm_connector"

    def __init__(
        self,
        model_path: str,
        baseline: str = "ours",
        kv_connector_extra_config: dict = None,
        batch_size: int = 1,
        max_num_batched_tokens: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 131072,
        block_size: int = 512,
        rope_factor: Optional[float] = 4.0,
        decode_cudagraph: bool = False,
        decode_cudagraph_mode: str = "full",
        **engine_kwargs,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if max_model_len < 1:
            raise ValueError(f"max_model_len must be positive, got {max_model_len}")
        if block_size < 1:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.model_name = Path(model_path).name.lower()
        self.model_path = model_path
        self.context_length = int(max_model_len)
        self.block_size = int(block_size)
        self.rope_factor = rope_factor
        self.batch_size = int(batch_size)
        self.attn_start = 0
        self.attn_end = -1
        self.critical_layers = None
        self._model_config = None
        if decode_cudagraph_mode not in ("full", "piecewise"):
            raise ValueError("decode_cudagraph_mode must be full or piecewise")
        if decode_cudagraph and decode_cudagraph_mode == "piecewise":
            os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
        from ucm.integration.vllm.qcfuse_graph import decode_graph_compilation_config

        graph_config = decode_graph_compilation_config(
            decode_cudagraph_mode, self.batch_size
        )
        # Enable before importing vLLM: native attention break decorators are
        # selected at module import. Fresh model workers inherit the switch.

        # The multiprocess EngineCore can begin the first large request while
        # the next request is still crossing IPC. Inproc queues the complete
        # batch before the first step; model execution can still use MP.
        if self.batch_size > 1:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        # Lazy import: vLLM + UCM + transformers live only on the GPU host.
        from transformers import AutoConfig, AutoTokenizer
        from vllm import LLM
        from vllm.config import KVTransferConfig
        from vllm.inputs import TokensPrompt

        self._hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Qwen3 checkpoints do not all share the same native context length.
        # Use the value declared by the official checkpoint when constructing
        # a YaRN override; a fixed 32768 silently changes the RoPE mapping for
        # Qwen3-32B, whose native max position is 40960.
        self._original_max_position_embeddings = int(
            getattr(self._hf_config, "max_position_embeddings", 32768)
        )
        model_type = getattr(self._hf_config, "model_type", None)
        if baseline != "fullcomp" and model_type != "qwen3":
            raise ValueError(
                "UCM QCFuse requires a Qwen3 checkpoint; "
                f"model={model_path!r} has model_type={model_type!r}."
            )

        self._TokensPrompt = TokensPrompt

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        # The tokenized form of BLEND_SEP is both inserted into every prompt and
        # carried in SamplingParams.extra_args["kv_transfer_params"]["sep_token"],
        # so the connector can find the chunk boundaries.
        self._sep_token = _tokenize_segment(self.tokenizer, BLEND_SEP)

        ktc = KVTransferConfig(
            kv_connector=self.kv_connector_name,
            kv_connector_module_path=self.kv_connector_module_path,
            kv_role="kv_both",
            kv_connector_extra_config=kv_connector_extra_config or {},
        )
        batched_tokens = max_num_batched_tokens
        if batched_tokens is None:
            batched_tokens = self.context_length * self.batch_size
        if batched_tokens < 1:
            raise ValueError(
                "max_num_batched_tokens must be positive, "
                f"got {max_num_batched_tokens}"
            )
        engine_options = {
            "enforce_eager": True,
            "kv_transfer_config": ktc,
            "max_model_len": self.context_length,
            "block_size": self.block_size,
            "max_num_batched_tokens": int(batched_tokens),
            "max_num_seqs": self.batch_size,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
            "tensor_parallel_size": 1,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "distributed_executor_backend": "mp",
            "worker_extension_cls": "ucm.integration.vllm.qcfuse_worker.QcFuseWorkerExtension",
        }
        if decode_cudagraph:
            engine_options.update(
                enforce_eager=False,
                compilation_config=graph_config,
            )
        if os.getenv("QCFUSE_DISABLE_FLASHINFER_AUTOTUNE", "0") == "1":
            # FlashInfer's startup autotune allocates a transient workspace
            # on top of Qwen3-32B BF16 weights and KV cache.  On an 80GB A800
            # this can OOM before the first request even though the serving
            # path itself fits; keep the backend available and skip only the
            # optional autotune sweep.
            engine_options["enable_flashinfer_autotune"] = False
        if self.rope_factor is not None and float(self.rope_factor) > 0:
            engine_options["hf_overrides"] = {
                "rope_scaling": {
                    "factor": float(self.rope_factor),
                    "original_max_position_embeddings": (
                        self._original_max_position_embeddings
                    ),
                    # vLLM 0.18+ reads the explicit rope_type field while
                    # older transformers/vLLM paths still consume type.
                    "rope_type": "yarn",
                    "type": "yarn",
                }
            }
        engine_options.update(engine_kwargs)
        # Pass the public options directly to LLM.  Converting EngineArgs with
        # ``asdict`` is not API-stable: vLLM 0.31 serializes nested compilation
        # fields with ``None`` values that fail its pydantic validation, while
        # the same options are accepted by the public constructor.
        self.llm = LLM(model=model_path, **engine_options)
        self._engine = _unwrap_engine(self.llm)
        if decode_cudagraph:
            self.llm.collective_rpc("qcfuse_start_execution_mode_observer")

        self._rid_counter = 0

        self.set_baseline(baseline)

    def worker_process_identities(self):
        """Capture worker identities before any offline cache producer runs."""
        # This established RPC also works with workers initialized while the
        # latest overlay was synchronizing. No new worker method is required.
        snapshots = self.llm.collective_rpc("qcfuse_benchmark_snapshot", args=(False,))
        identities = []
        for snapshot in snapshots:
            pid = int(snapshot["pid"])
            stat = Path(f"/proc/{pid}/stat").read_text()
            fields = stat[stat.rfind(")") + 2 :].split()
            identities.append({"pid": pid, "start_ticks": int(fields[19])})
        return identities

    @staticmethod
    def gpu_memory_snapshot() -> dict:
        """Return device-wide and process-local CUDA memory counters."""
        try:
            import torch

            if not torch.cuda.is_available():
                return {"cuda_available": False}
            torch.cuda.synchronize()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return {
                "cuda_available": True,
                "gpu_index": int(torch.cuda.current_device()),
                "gpu_memory_total_bytes": int(total_bytes),
                "gpu_memory_free_bytes": int(free_bytes),
                "gpu_memory_used_bytes": int(total_bytes - free_bytes),
                "torch_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
                "torch_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
                "torch_peak_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated()
                ),
                "torch_peak_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved()
                ),
            }
        except Exception as exc:
            return {"cuda_available": False, "error": str(exc)}

    def set_baseline(self, baseline: str):
        """Switch baseline configuration."""
        if baseline not in SUPPORTED_BASELINES:
            raise ValueError(
                f"Unsupported baseline={baseline!r}; expected one of "
                f"{SUPPORTED_BASELINES}"
            )
        self.baseline = baseline
        self.critical_layers = None
        cfg = BLEND_CONFIG[baseline]
        self.first_style, self.start, self.method = cfg

    def _get_model_config(self) -> dict:
        """Get model architecture parameters (cached)."""
        if self._model_config is not None:
            return self._model_config

        from transformers import AutoConfig

        config = self._hf_config
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        if getattr(config, "multi_query_attention", False):
            num_kv_heads = getattr(config, "multi_query_group_num", 1)
        else:
            num_kv_heads = getattr(
                config,
                "num_key_value_heads",
                getattr(
                    config,
                    "multi_query_group_num",
                    config.num_attention_heads,
                ),
            )

        self._model_config = {
            "head_dim": head_dim,
            "num_layers": getattr(config, "num_hidden_layers", 32),
            "num_heads": getattr(config, "num_attention_heads", 32),
            "num_kv_heads": num_kv_heads,
        }
        return self._model_config

    def _get_template(self) -> Tuple[str, str, str]:
        """Get model template based on model name."""
        for prefix, template in TEMPLATES.items():
            if self.model_name.startswith(prefix):
                if prefix == "qwen" and getattr(self, "thinking_mode", "off") == "on":
                    # The checkpoint's enable_thinking=True generation prompt
                    # ends at the assistant header; the model emits <think>.
                    return (
                        template[0],
                        template[1],
                        "<|im_end|>\n<|im_start|>assistant\n",
                    )
                return template
        return ("", "", "")

    def _build_prompt(
        self,
        system_prompt: str,
        docs: List[str],
        q_prompt: List[str],
        use_sep: bool,
        answer_suffix: str = "\n\n## Answer\n",
        answer_prefix: str = "",
    ) -> Tuple[str, str]:
        """Build complete prompt from components."""
        sys_h, sys_e, asst_h = self._get_template()
        prefix = sys_h + system_prompt + sys_e
        suffix = self._query_suffix(q_prompt, answer_suffix, asst_h, answer_prefix)

        if use_sep:
            query_sep = BLEND_SEP.join(q_prompt)
            return BLEND_SEP.join([prefix] + docs + [suffix]), query_sep
        return prefix + "".join(docs) + suffix, suffix

    def _query_suffix(self, q_prompt, answer_suffix, assistant_header, answer_prefix):
        if getattr(self, "thinking_mode", "off") == "on":
            # Keep the official RULER completion hint inside the user turn.
            # An assistant prefill would otherwise bypass native reasoning.
            return "".join(q_prompt) + answer_prefix + answer_suffix + assistant_header
        return "".join(q_prompt) + answer_suffix + assistant_header + answer_prefix

    # ------------------------------------------------------------------ #
    #  vLLM-via-UCM request driving (replaces sgl.Engine.generate)
    # ------------------------------------------------------------------ #
    def _next_request_id(self) -> str:
        self._rid_counter += 1
        return f"qcfuse-{self._rid_counter}"

    def _build_sampling_params(
        self,
        sampling_params,
        blend_args: Optional[dict],
    ):
        from vllm import SamplingParams

        if isinstance(sampling_params, dict):
            params = SamplingParams(
                temperature=sampling_params.get("temperature", 0),
                max_tokens=max(1, int(sampling_params.get("max_new_tokens", 1))),
            )
        else:
            params = copy.deepcopy(sampling_params)
        if blend_args is not None:
            params.extra_args = {
                "kv_transfer_params": {
                    "blend_args": dict(blend_args),
                    "sep_token": list(self._sep_token),
                }
            }
        return params

    def _submit_batch_and_collect(
        self,
        requests: Sequence[Tuple[List[int], object, Optional[dict]]],
    ) -> List[dict]:
        """Submit a homogeneous request batch and preserve input order."""
        if not requests:
            return []
        if len(requests) > self.batch_size:
            raise ValueError(
                f"request batch has {len(requests)} entries, engine capacity is "
                f"{self.batch_size}"
            )

        request_ids = []
        started = {}
        results = {}
        unfinished = set()
        last_token_ids = {}
        output_progress = {}
        step_profile_path = getattr(self, "generation_step_profile_path", None)
        step_profile = []
        for prompt_token_ids, sampling_params, blend_args in requests:
            rid = self._next_request_id()
            if hasattr(self, "tokenizer"):
                from output_protocol import OutputProgress

                output_progress[rid] = OutputProgress(
                    self.tokenizer, getattr(self, "thinking_mode", "off") == "on"
                )
            request_ids.append(rid)
            started[rid] = time.perf_counter()
            unfinished.add(rid)
            if blend_args is not None:
                blend_args = {**blend_args, "client_request_id": rid}
            params = self._build_sampling_params(sampling_params, blend_args)
            self._engine.add_request(
                rid,
                self._TokensPrompt(prompt_token_ids=prompt_token_ids),
                params,
            )
            results[rid] = {
                "request_id": rid,
                "text": "",
                "ttft": None,
                "latency": None,
                "decode": None,
                "first_token_seconds": None,
                "first_token_time": None,
                "generated_tokens": 0,
                "first_token_output_tokens": 0,
                "decode_tokens": 0,
                "output_token_ids": [],
                "sampling_max_tokens": getattr(params, "max_tokens", None),
            }

        while unfinished:
            if step_profile_path:
                step_started = time.perf_counter()
                active_before = len(unfinished)
            outputs = self._engine.step()
            if step_profile_path:
                step_finished = time.perf_counter()
                generated_before = sum(x["generated_tokens"] for x in results.values())
            for out in outputs:
                rid = str(getattr(out, "request_id", ""))
                if rid not in unfinished:
                    continue
                now = time.perf_counter()
                candidate = out.outputs[0] if out.outputs else None
                if candidate is not None:
                    token_ids = getattr(candidate, "token_ids", ()) or ()
                    if results[rid]["ttft"] is None and token_ids:
                        results[rid]["ttft"] = now - started[rid]
                        results[rid]["first_token_seconds"] = now - started[rid]
                        results[rid]["first_token_time"] = now
                        results[rid]["first_token_output_tokens"] = len(token_ids)
                    results[rid]["text"] = candidate.text
                    results[rid]["generated_tokens"] = len(token_ids)
                    last_token_ids[rid] = token_ids
                    if rid in output_progress:
                        output_progress[rid].update(token_ids, now)
                if bool(getattr(out, "finished", False)):
                    results[rid]["output_token_ids"] = list(last_token_ids.get(rid, ()))
                    results[rid]["finish_reason"] = getattr(
                        candidate, "finish_reason", None
                    )
                    if rid in output_progress:
                        progress = output_progress[rid]
                        results[rid][
                            "first_thinking_time"
                        ] = progress.first_thinking_time
                        results[rid]["first_answer_time"] = progress.first_answer_time
                    results[rid]["finished_time"] = now
                    results[rid]["latency"] = now - started[rid]
                    if results[rid]["ttft"] is None:
                        results[rid]["ttft"] = results[rid]["latency"]
                    if results[rid]["first_token_seconds"] is None:
                        results[rid]["first_token_seconds"] = results[rid]["latency"]
                    first_token_time = results[rid]["first_token_time"]
                    results[rid]["decode"] = (
                        max(0.0, now - first_token_time)
                        if first_token_time is not None
                        else 0.0
                    )
                    results[rid]["decode_tokens"] = max(
                        0,
                        results[rid]["generated_tokens"]
                        - results[rid]["first_token_output_tokens"],
                    )
                    unfinished.remove(rid)

            if step_profile_path:
                step_profile.append(
                    {
                        "step": len(step_profile),
                        "active_before": active_before,
                        "active_after": len(unfinished),
                        "engine_step_seconds": step_finished - step_started,
                        "frontend_output_seconds": time.perf_counter() - step_finished,
                        "new_output_tokens": sum(
                            x["generated_tokens"] for x in results.values()
                        )
                        - generated_before,
                    }
                )

        if step_profile_path:
            path = Path(step_profile_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            phase = (requests[0][2] or {}).get("blend_style", "full_prefill")
            with path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "phase": phase,
                            "request_ids": request_ids,
                            "scope": "instrumented_parent_engine_step_and_output_processing_not_exclusive_GPU_time",
                            "steps": step_profile,
                        }
                    )
                    + "\n"
                )

        return [results[rid] for rid in request_ids]

    def _submit_and_collect(
        self,
        prompt_token_ids: List[int],
        sampling_params,
        blend_args: dict = None,
    ) -> dict:
        """Submit one request and drive the engine to completion.

        ``blend_args`` (the merged ``_blend_args`` + ``_ssd_args`` dict, plus the
        ``blend_loc_list`` added by :meth:`_generate`) travels on
        ``SamplingParams.extra_args["kv_transfer_params"]``.  The engine process
        and the scheduler/EngineCore process are separate, so this serialized
        per-request channel is what ``build_connector_meta`` reads.
        """
        return self._submit_batch_and_collect(
            [(prompt_token_ids, sampling_params, blend_args)]
        )[0]

    def _tokenize(self, prompt: str) -> List[int]:
        return tokenize_with_sep(self.tokenizer, prompt, self._sep_token)

    def _prepare_generation(
        self, prompt: str, params, blend_args: Optional[dict]
    ) -> Tuple[List[int], object, Optional[dict]]:
        """Tokenize one request and add separator-removed chunk boundaries."""
        if isinstance(prompt, list):
            if blend_args is not None:
                raise ValueError(
                    "Pre-tokenized baseline prompts cannot carry blend args"
                )
            return prompt, params, None
        cache = getattr(self, "_prompt_token_cache", None)
        if cache is None:
            cache = self._prompt_token_cache = OrderedDict()
        cache_key = (prompt, blend_args is not None)
        if cache_key in cache:
            stripped_ids, locs = cache.pop(cache_key)
        else:
            ids = self._tokenize(prompt)
            if blend_args is None:
                stripped_ids, locs = ids, None
            else:
                from ucm.sparse.qcfuse.blend_info import split_tokens

                _, stripped_ids, locs = split_tokens(
                    None, ids, BLEND_SEP, self._sep_token
                )
                if locs is None:
                    locs = [0, len(stripped_ids)]
        cache[cache_key] = (stripped_ids, locs)
        while len(cache) > 8:
            cache.popitem(last=False)
        if blend_args is None:
            return stripped_ids, params, None
        prepared_args = dict(blend_args)
        prepared_args["blend_loc_list"] = [int(x) for x in locs]
        return stripped_ids, params, prepared_args

    def _generate_batch(
        self,
        requests: Sequence[Tuple[str, object, Optional[dict]]],
    ) -> List[dict]:
        requests = list(requests)
        if self.baseline in ("ours", "fuserag", "prophetkv"):
            # QCFuse's position-aware ragged path is validated through four
            # simultaneous requests. Split a larger user batch before it
            # reaches vLLM; letting the scheduler mix later prefill with decode
            # requests would violate QCFuse's homogeneous-step invariant.
            try:
                safe_batch = int(os.getenv("QCFUSE_MAX_SAFE_BATCH_SIZE", "4"))
            except ValueError:
                safe_batch = 4
            if safe_batch > 0 and len(requests) > safe_batch:
                outputs: List[dict] = []
                for start in range(0, len(requests), safe_batch):
                    outputs.extend(
                        self._generate_batch(requests[start : start + safe_batch])
                    )
                return outputs
        prepared = [
            self._prepare_generation(prompt, params, blend_args)
            for prompt, params, blend_args in requests
        ]
        return self._submit_batch_and_collect(prepared)

    def _generate(self, prompt: str, params: dict, blend_args: dict = None) -> dict:
        """Tokenize, submit, and collect one request."""
        return self._generate_batch([(prompt, params, blend_args)])[0]

    def _timed_generate(self, prompt: str, params: dict, **kwargs) -> dict:
        """Run generation and return generated text plus TTFT."""
        return self._generate(prompt, params, blend_args=(kwargs or None))

    def _drain_generate(self, prompt: str, params: dict, **kwargs) -> None:
        self._generate(prompt, params, blend_args=(kwargs or None))

    def _timed_generate_batch(
        self, requests: Sequence[Tuple[str, object, Optional[dict]]]
    ) -> List[dict]:
        return self._generate_batch(requests)

    def _drain_generate_batch(
        self, requests: Sequence[Tuple[str, object, Optional[dict]]]
    ) -> None:
        self._generate_batch(requests)

    def warmup(self, num_warmup: int = 3):
        # Warm model execution; sparse Triton kernels compile on first use.
        sys_h, sys_e, asst_h = self._get_template()
        warmup_prompt = (
            sys_h
            + "You are a helpful assistant."
            + sys_e
            + "Hello, how are you?"
            + asst_h
        )
        for _ in range(num_warmup):
            self._generate(warmup_prompt, {"temperature": 0, "max_new_tokens": 1})

    def shutdown(self):
        engine_core = getattr(self._engine, "engine_core", None)
        if engine_core is not None:
            engine_core.shutdown()
        self._engine = None
        self.llm = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False
