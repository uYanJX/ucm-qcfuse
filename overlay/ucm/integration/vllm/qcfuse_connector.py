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
"""Pass per-request QCFuse state from the vLLM scheduler to UCM hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from ucm.integration.vllm.ucm_connector import UCMDirectConnector
from ucm.sparse.qcfuse.blend_info import AttParams, QcFuseRequestState
from ucm.sparse.qcfuse.request_builder import build_request_state
from ucm.sparse.state import get_ucm_sparse

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass
class QcFuseRequestMeta:
    state: QcFuseRequestState


@dataclass
class QcFuseConnectorMetadata(KVConnectorMetadata):
    request_meta: Dict[str, QcFuseRequestMeta] = field(default_factory=dict)


class UCMQcFuseConnector(UCMDirectConnector):
    """Connector for QCFuse's sample-addressed KV and digest caches."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config=None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        ucm_sparse_config = self.launch_config.get("ucm_sparse_config", {}) or {}
        if "QcFuse" not in ucm_sparse_config:
            raise ValueError(
                "UCMQcFuseConnector requires 'QcFuse' in kv_connector_extra_config"
                "['ucm_sparse_config']; please check your kv_transfer config."
            )
        self.qcfuse_config = ucm_sparse_config["QcFuse"]

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        model_type = getattr(model_config.hf_config, "model_type", None)
        if model_type != "qwen3":
            raise ValueError(
                f"UCM QCFuse currently supports Qwen3, got model_type={model_type!r}"
            )
        if (
            parallel_config.tensor_parallel_size != 1
            or parallel_config.pipeline_parallel_size != 1
        ):
            raise ValueError("UCM QCFuse currently requires TP=1 and PP=1")
        num_heads = model_config.get_num_attention_heads(parallel_config)
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_dim = model_config.get_head_size()
        num_layers = model_config.get_num_layers(parallel_config)
        self.att_params = AttParams(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_layers=num_layers,
        )

    def setup_model(self, _model) -> None:
        """Bind the worker-side store after sparse initialization."""
        sparse = get_ucm_sparse()
        if sparse is not None and hasattr(sparse, "attach_store"):
            sparse.attach_store(self.store)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        # QCFuse owns prompt-cache reuse instead of vLLM's block connector.
        return 0, False

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = QcFuseConnectorMetadata()
        for request in scheduler_output.scheduled_new_reqs:
            req_id = str(request.req_id)
            _sp = request.sampling_params
            kvt = (
                (_sp.extra_args or {}).get("kv_transfer_params", {})
                if _sp is not None and _sp.extra_args
                else {}
            )
            blend_args = kvt.get("blend_args")
            if not blend_args:
                continue
            sep_token = kvt.get("sep_token", [])
            state = build_request_state(
                blend_args=blend_args,
                prompt_token_ids=list(request.prompt_token_ids),
                sep_token=sep_token,
                att_params=self.att_params,
                device="cpu",
            )
            state.request_id = req_id
            meta.request_meta[req_id] = QcFuseRequestMeta(state=state)

        return meta

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        self.connector_metadata = connector_metadata

    def start_load_kv(self, forward_context, **kwargs) -> None:
        return None

    def wait_for_save(self) -> None:
        return None

    def clear_connector_metadata(self) -> None:
        self.connector_metadata = None
