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
import os
import time
from dataclasses import asdict
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
        **engine_kwargs,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.model_name = Path(model_path).name.lower()
        self.model_path = model_path
        self.context_length = 16000
        self.batch_size = int(batch_size)
        self.attn_start = 0
        self.attn_end = -1
        self.critical_layers = None
        self._model_config = None

        # The multiprocess EngineCore can begin the first large request while
        # the next request is still crossing IPC. Inproc queues the complete
        # batch before the first step; model execution can still use MP.
        if self.batch_size > 1:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        # Lazy import: vLLM + UCM + transformers live only on the GPU host.
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.config import KVTransferConfig
        from vllm.engine.arg_utils import EngineArgs
        from vllm.inputs import TokensPrompt

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
            "max_num_batched_tokens": int(batched_tokens),
            "max_num_seqs": self.batch_size,
            "gpu_memory_utilization": 0.6,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
            "tensor_parallel_size": 1,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "distributed_executor_backend": "mp",
        }
        engine_options.update(engine_kwargs)
        llm_args = EngineArgs(model=model_path, **engine_options)
        self.llm = LLM(**asdict(llm_args))
        self._engine = _unwrap_engine(self.llm)

        self._rid_counter = 0

        self.set_baseline(baseline)

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

        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
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
                return template
        return ("", "", "")

    def _build_prompt(
        self, system_prompt: str, docs: List[str], q_prompt: List[str], use_sep: bool
    ) -> Tuple[str, str]:
        """Build complete prompt from components."""
        sys_h, sys_e, asst_h = self._get_template()
        prefix = sys_h + system_prompt + sys_e
        suffix = "".join(q_prompt) + "\n\n## Answer\n" + asst_h

        if use_sep:
            query_sep = BLEND_SEP.join(q_prompt)
            return BLEND_SEP.join([prefix] + docs + [suffix]), query_sep
        return prefix + "".join(docs) + suffix, suffix

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
        for prompt_token_ids, sampling_params, blend_args in requests:
            rid = self._next_request_id()
            request_ids.append(rid)
            started[rid] = time.perf_counter()
            unfinished.add(rid)
            params = self._build_sampling_params(sampling_params, blend_args)
            self._engine.add_request(
                rid,
                self._TokensPrompt(prompt_token_ids=prompt_token_ids),
                params,
            )
            results[rid] = {
                "text": "",
                "ttft": None,
                "latency": None,
                "generated_tokens": 0,
            }

        while unfinished:
            for out in self._engine.step():
                rid = str(getattr(out, "request_id", ""))
                if rid not in unfinished:
                    continue
                now = time.perf_counter()
                candidate = out.outputs[0] if out.outputs else None
                if candidate is not None:
                    token_ids = getattr(candidate, "token_ids", ()) or ()
                    if results[rid]["ttft"] is None and token_ids:
                        results[rid]["ttft"] = now - started[rid]
                    results[rid]["text"] = candidate.text
                    results[rid]["generated_tokens"] = len(token_ids)
                if bool(getattr(out, "finished", False)):
                    results[rid]["latency"] = now - started[rid]
                    if results[rid]["ttft"] is None:
                        results[rid]["ttft"] = results[rid]["latency"]
                    unfinished.remove(rid)

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
        if blend_args is None:
            return self._tokenize(prompt), params, None

        from ucm.sparse.qcfuse.blend_info import split_tokens

        ids = self._tokenize(prompt)
        _, stripped_ids, locs = split_tokens(None, ids, BLEND_SEP, self._sep_token)
        if locs is None:
            locs = [0, len(stripped_ids)]
        prepared_args = dict(blend_args)
        prepared_args["blend_loc_list"] = [int(x) for x in locs]
        return stripped_ids, params, prepared_args

    def _generate_batch(
        self,
        requests: Sequence[Tuple[str, object, Optional[dict]]],
    ) -> List[dict]:
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
