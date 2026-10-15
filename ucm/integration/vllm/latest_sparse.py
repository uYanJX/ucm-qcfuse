"""Native vLLM 0.31+ QCFuse integration.

This module deliberately hooks the public Qwen3 model path and the vLLM
``unified_attention_with_output`` entry point.  It is kept separate from the
legacy versioned patch tree so upgrades do not silently select the v0.11 code.
"""

from __future__ import annotations

import re
from typing import Any

import torch

_INSTALLED = False


def _layer_id(name: str) -> int:
    m = re.search(r"layers\.(\d+)\.", str(name))
    return int(m.group(1)) if m else -1


def _layer_name_value(name) -> str:
    return str(getattr(name, "value", name))


def _forward_with_layer_end(model, forward, sparse, *args, **kwargs):
    original_end = model.end_layer
    model.end_layer = sparse.effective_layer_end(model.start_layer, original_end)
    try:
        return forward(model, *args, **kwargs)
    finally:
        model.end_layer = original_end


def _compact_silu_and_mul(gate_up, operation, tile_tokens=512):
    """Compact a request-owned projection using the unchanged native SiLU op."""
    if gate_up.ndim != 2 or not gate_up.is_contiguous():
        raise ValueError("Compact SwiGLU requires a contiguous token matrix")
    tokens, width = gate_up.shape
    if width % 2 or tile_tokens < 1:
        raise ValueError("Invalid compact SwiGLU width or tile size")
    intermediate = width // 2
    flat = gate_up.view(-1)
    scratch = gate_up.new_empty((min(tile_tokens, tokens), intermediate))
    for start in range(0, tokens, tile_tokens):
        stop = min(start + tile_tokens, tokens)
        operation(scratch[: stop - start], gate_up[start:stop])
        # The native operation completes first on this stream. Destination
        # ends before any future input row, so compaction cannot destroy it.
        flat[start * intermediate : stop * intermediate].copy_(
            scratch[: stop - start].reshape(-1)
        )
    return flat[: tokens * intermediate].view(tokens, intermediate)


def _compact_qwen_mlp_forward(mlp, hidden_states, projection_tile_tokens=0):
    """Reuse BF16 FFN buffers, optionally bounding projection rows.

    Row tiling is an opt-in experiment. Balance the rows so an uneven end
    does not force a singleton GEMM, and keep all selected rows and weights.
    Each tile's projection storage dies before the next tile is allocated.
    """
    if (
        isinstance(projection_tile_tokens, bool)
        or not isinstance(projection_tile_tokens, int)
        or projection_tile_tokens < 0
    ):
        raise ValueError("projection_tile_tokens must be a non-negative integer")
    tokens = hidden_states.shape[0]
    if projection_tile_tokens and tokens > projection_tile_tokens:
        parts = (tokens + projection_tile_tokens - 1) // projection_tile_tokens
        rows = (tokens + parts - 1) // parts
        for start in range(0, tokens, rows):
            _compact_qwen_mlp_forward(mlp, hidden_states[start : start + rows])
        return hidden_states
    gate_up, _ = mlp.gate_up_proj(hidden_states)
    activation = _compact_silu_and_mul(gate_up, mlp.act_fn.op)
    # Input projection has consumed hidden_states. Native BF16 F.linear and
    # its out overload use the same matrix dimensions for this projection.
    return torch.ops.aten.linear.out(
        activation, mlp.down_proj.weight, None, out=hidden_states
    )


def _can_compact_qwen_mlp(mlp, hidden_states, residual):
    from vllm import envs
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    down = mlp.down_proj
    method = down.quant_method
    return (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.ndim == 2
        and hidden_states.is_contiguous()
        and down.tp_size == 1
        and down.bias is None
        and isinstance(method, UnquantizedLinearMethod)
        and method._gemm_impl is default_unquantized_gemm
        and not envs.VLLM_BATCH_INVARIANT
        and isinstance(mlp.act_fn, SiluAndMul)
        and down.weight.shape[0] == hidden_states.shape[1]
        and hidden_states.shape[0] * down.weight.shape[1] * 2 > 128 * 1024**2
        and (
            residual is None
            or hidden_states.untyped_storage().data_ptr()
            != residual.untyped_storage().data_ptr()
        )
    )


def install_vllm_latest_sparse_hooks() -> None:
    """Install idempotent hooks for the current vLLM Qwen3 implementation."""
    global _INSTALLED
    if _INSTALLED:
        return
    import os
    import vllm
    from packaging.version import Version

    if Version(vllm.__version__) < Version("0.31.0"):
        raise RuntimeError(
            f"latest_sparse adapter requires vLLM >= 0.31, got {vllm.__version__}"
        )
    projection_tile_tokens = int(os.getenv("QCFUSE_GPU_FFN_TILE_TOKENS", "0"))
    if projection_tile_tokens < 0:
        raise ValueError("QCFUSE_GPU_FFN_TILE_TOKENS must be non-negative")

    from ucm.sparse.state import (
        get_ucm_sparse,
        has_ucm_sparse,
        maybe_execute_sparse_ffn_begin,
        maybe_execute_sparse_ffn_finished,
        maybe_execute_sparse_layer_begin,
        maybe_execute_sparse_layer_finished,
        maybe_execute_sparse_model_finished,
        maybe_execute_sparse_attention_begin,
        maybe_execute_sparse_attention_should_skip,
        maybe_execute_sparse_qwen3_pre_rope,
    )

    # Qwen3Attention.forward is small and stable in vLLM 0.31.  Keeping the
    # implementation here avoids touching vLLM's installed files.
    from vllm.model_executor.models.qwen3 import (
        Qwen3Attention,
        Qwen3DecoderLayer,
        Qwen3Model,
    )

    if not getattr(Qwen3Attention, "_ucm_latest_wrapped", False):

        def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            sparse = get_ucm_sparse() if has_ucm_sparse() else None
            compact_values = getattr(sparse, "compact_projection_values", None)
            if callable(compact_values) and compact_values():
                # V otherwise pins the entire fused QKV allocation through
                # attention. Q/K norms create their own storage; retain only
                # an exact V copy while loading GPU-resident cached layers.
                v = v.clone()
            q = self.q_norm(q.view(*q.shape[:-1], -1, self.head_dim)).view(q.shape)
            k = self.k_norm(k.view(*k.shape[:-1], -1, self.head_dim)).view(k.shape)
            del qkv
            layer_name = getattr(self.attn, "layer_name", "")
            q, k, v, handled = maybe_execute_sparse_qwen3_pre_rope(
                q, k, v, positions, _layer_id(layer_name), self.rotary_emb
            )
            if not handled:
                q, k = self.rotary_emb(positions, q, k)
            return self.o_proj(self.attn(q, k, v))[0]

        Qwen3Attention.forward = forward
        Qwen3Attention._ucm_latest_wrapped = True

    # QCFuse may narrow the token stream after the first attention layer.  The
    # native vLLM Qwen3 layer has no sparse lifecycle callbacks, so install the
    # same callbacks around its existing layer body.  This keeps positions,
    # residuals, and the final model output aligned with the selected tokens.
    if not getattr(Qwen3DecoderLayer, "_ucm_latest_wrapped", False):

        def decoder_layer_forward(self, positions, hidden_states, residual):
            positions, hidden_states, residual = maybe_execute_sparse_layer_begin(
                positions, hidden_states, residual
            )
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states
            )
            hidden_states, residual = maybe_execute_sparse_ffn_begin(
                hidden_states, residual
            )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            sparse = get_ucm_sparse() if has_ucm_sparse() else None
            compact_values = getattr(sparse, "compact_projection_values", None)
            if (
                callable(compact_values)
                and compact_values()
                and _can_compact_qwen_mlp(self.mlp, hidden_states, residual)
            ):
                hidden_states = _compact_qwen_mlp_forward(
                    self.mlp, hidden_states, projection_tile_tokens
                )
            else:
                hidden_states = self.mlp(hidden_states)
            hidden_states, residual = maybe_execute_sparse_ffn_finished(
                hidden_states, residual
            )
            return maybe_execute_sparse_layer_finished(
                positions, hidden_states, residual
            )[1:]

        Qwen3DecoderLayer.forward = decoder_layer_forward
        Qwen3DecoderLayer._ucm_latest_wrapped = True

    if not getattr(Qwen3Model, "_ucm_latest_wrapped", False):
        original_qwen3_model_forward = Qwen3Model.forward

        def model_forward(self, *args, **kwargs):
            sparse = get_ucm_sparse() if has_ucm_sparse() else None
            if callable(getattr(sparse, "effective_layer_end", None)):
                result = _forward_with_layer_end(
                    self, original_qwen3_model_forward, sparse, *args, **kwargs
                )
            else:
                result = original_qwen3_model_forward(self, *args, **kwargs)
            # Qwen3Model normally returns a tensor on the single-rank setup used
            # by QCFuse.  Restore the full token stream after the final norm so
            # vLLM's sampler still sees the scheduler's original token count.
            if torch.is_tensor(result):
                result, _ = maybe_execute_sparse_model_finished(result, None)
            elif isinstance(result, tuple) and result and torch.is_tensor(result[0]):
                restored, _ = maybe_execute_sparse_model_finished(result[0], None)
                result = (restored, *result[1:])
            return result

        Qwen3Model.forward = model_forward
        Qwen3Model._ucm_latest_wrapped = True

    # Direct-call attention is used by eager mode and is the safest hook point.
    import vllm.model_executor.layers.attention.attention as attention_mod

    def sparse_owns_kv_update():
        if not has_ucm_sparse():
            return False
        skip = getattr(get_ucm_sparse(), "should_skip_kv_cache_update", None)
        return bool(skip()) if callable(skip) else False

    if not getattr(attention_mod, "_ucm_latest_kv_update_wrapped", False):
        original_kv_update = attention_mod.unified_kv_cache_update

        def unified_kv_cache_update(key, value, layer_name):
            if sparse_owns_kv_update():
                return key.new_empty(0)
            return original_kv_update(key, value, layer_name)

        attention_mod.unified_kv_cache_update = unified_kv_cache_update
        attention_mod._ucm_latest_kv_update_wrapped = True

    # H20's current vLLM path uses the registered opaque torch operator, whose
    # handle is not replaceable through ``torch.ops`` assignment.  Wrap the
    # public Attention.forward method instead and run QCFuse's prepared ragged
    # attention directly, while leaving ordinary vLLM attention untouched.
    Attention = attention_mod.Attention
    if not getattr(Attention, "_ucm_latest_wrapped", False):
        original_attention_forward = Attention.forward

        def attention_forward(
            self, query, key, value, output_shape=None, output_dtype=None
        ):
            sparse = get_ucm_sparse() if has_ucm_sparse() else None
            if sparse is not None and getattr(sparse, "_prepared", None) is not None:
                if output_dtype is None:
                    output_dtype = query.dtype
                if output_shape is None:
                    output_shape = torch.Size(
                        (query.shape[0], self.num_heads * self.head_size_v)
                    )
                output = torch.empty(
                    output_shape, dtype=output_dtype, device=query.device
                )
                hidden_size = output_shape[-1]
                query_view = query.view(-1, self.num_heads, self.head_size)
                output_view = output.view(-1, self.num_heads, self.head_size_v)
                key_view = (
                    key.view(-1, self.num_kv_heads, self.head_size)
                    if key is not None
                    else key
                )
                value_view = (
                    value.view(-1, self.num_kv_heads, self.head_size_v)
                    if value is not None
                    else value
                )
                from vllm.forward_context import get_forward_context

                query_view, key_view, value_view, output_view = (
                    maybe_execute_sparse_attention_begin(
                        query_view,
                        key_view,
                        value_view,
                        _layer_name_value(self.layer_name),
                        get_forward_context(),
                        output=output_view,
                    )
                )
                if maybe_execute_sparse_attention_should_skip():
                    return output_view.view(-1, hidden_size)
            return original_attention_forward(
                self, query, key, value, output_shape, output_dtype
            )

        Attention.forward = attention_forward
        Attention._ucm_latest_wrapped = True
    if not getattr(attention_mod, "_ucm_latest_wrapped", False):
        original = attention_mod.unified_attention_with_output

        def unified_attention_with_output(
            query,
            key,
            value,
            output,
            layer_name,
            output_scale=None,
            output_block_scale=None,
            kv_cache_dummy_dep=None,
        ):
            from vllm.forward_context import get_forward_context

            query, key, value, output = maybe_execute_sparse_attention_begin(
                query,
                key,
                value,
                _layer_name_value(layer_name),
                get_forward_context(),
                output=output,
            )
            if maybe_execute_sparse_attention_should_skip():
                return None
            return original(
                query,
                key,
                value,
                output,
                layer_name,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )

        attention_mod.unified_attention_with_output = unified_attention_with_output
        attention_mod._ucm_latest_wrapped = True

    # V2 model runner normally calls the registered torch custom op rather
    # than the Python symbol above.  Replace it with the same narrow wrapper;
    # PatchOpProxy preserves fake_impl so torch compilation/introspection keeps
    # working.
    try:
        from ucm.integration.vllm.patch.utils import PatchOpProxy

        original_kv_update_op = torch.ops.vllm.unified_kv_cache_update
        if not getattr(original_kv_update_op, "_ucm_latest_wrapped", False):

            def kv_update_op_wrapper(key, value, layer_name):
                if sparse_owns_kv_update():
                    return key.new_empty(0)
                return original_kv_update_op(key, value, layer_name)

            kv_update_proxy = PatchOpProxy(original_kv_update_op, kv_update_op_wrapper)
            kv_update_proxy._self_ucm_latest_wrapped = True
            torch.ops.vllm.unified_kv_cache_update = kv_update_proxy
        original_op = torch.ops.vllm.unified_attention_with_output
        if not getattr(original_op, "_ucm_latest_wrapped", False):

            def op_wrapper(
                query,
                key,
                value,
                output,
                layer_name,
                output_scale=None,
                output_block_scale=None,
                kv_cache_dummy_dep=None,
            ):
                from vllm.forward_context import get_forward_context

                q, k, v, out = maybe_execute_sparse_attention_begin(
                    query,
                    key,
                    value,
                    _layer_name_value(layer_name),
                    get_forward_context(),
                    output=output,
                )
                if maybe_execute_sparse_attention_should_skip():
                    return None
                return original_op(
                    q,
                    k,
                    v,
                    out,
                    layer_name,
                    output_scale=output_scale,
                    output_block_scale=output_block_scale,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )

            proxy = PatchOpProxy(original_op, op_wrapper)
            proxy._self_ucm_latest_wrapped = True
            torch.ops.vllm.unified_attention_with_output = proxy
    except (AttributeError, RuntimeError):
        # Some CPU/import-only contexts do not register vLLM custom ops yet.
        pass

    # vLLM 0.31's V2 runner builds attention metadata inside execute_model but
    # has no UCM sparse callback.  Keep the scheduler output on the runner for
    # the duration of that call and build QCFuse's per-batch state immediately
    # after metadata is available.  This is the lifecycle point required by
    # qwen3_pre_rope; older patch files are intentionally not used.
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if not getattr(GPUModelRunner, "_ucm_latest_lifecycle_wrapped", False):
        original_execute_model = GPUModelRunner.execute_model
        original_build_attention_metadata = GPUModelRunner._build_attention_metadata
        original_model_forward = GPUModelRunner._model_forward

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            self._ucm_latest_pending_scheduler_output = scheduler_output
            self._ucm_latest_sparse_meta_built_for = None
            try:
                return original_execute_model(
                    self, scheduler_output, intermediate_tensors
                )
            finally:
                self._ucm_latest_pending_scheduler_output = None
                self._ucm_latest_sparse_meta_built_for = None

        def build_attention_metadata(self, *args, **kwargs):
            result = original_build_attention_metadata(self, *args, **kwargs)
            scheduler_output = getattr(
                self, "_ucm_latest_pending_scheduler_output", None
            )
            if (
                scheduler_output is not None
                and getattr(self, "_ucm_latest_sparse_meta_built_for", None)
                is not scheduler_output
                and has_ucm_sparse()
            ):
                sparse = get_ucm_sparse()
                sparse.build_sparse_meta(
                    scheduler_output,
                    self.requests,
                    self.input_batch,
                    result[0],
                )
                self._ucm_latest_sparse_meta_built_for = scheduler_output
            return result

        def model_forward(self, *args, **kwargs):
            scheduler_output = getattr(
                self, "_ucm_latest_pending_scheduler_output", None
            )
            if (
                scheduler_output is not None
                and getattr(self, "_ucm_latest_sparse_meta_built_for", None)
                is not scheduler_output
                and has_ucm_sparse()
            ):
                from vllm.forward_context import get_forward_context

                sparse = get_ucm_sparse()
                sparse.build_sparse_meta(
                    scheduler_output,
                    self.requests,
                    self.input_batch,
                    get_forward_context().attn_metadata,
                )
                self._ucm_latest_sparse_meta_built_for = scheduler_output
            return original_model_forward(self, *args, **kwargs)

        GPUModelRunner.execute_model = execute_model
        GPUModelRunner._build_attention_metadata = build_attention_metadata
        GPUModelRunner._model_forward = model_forward
        GPUModelRunner._ucm_latest_lifecycle_wrapped = True

    # In V2 the worker owns the call boundary and may hold a runner class
    # imported through a different module alias.  Mark the concrete runner
    # through Worker.execute_model so _model_forward always sees the current
    # SchedulerOutput regardless of that alias.
    from vllm.v1.worker.gpu_worker import Worker

    if not getattr(Worker, "_ucm_latest_lifecycle_wrapped", False):
        original_worker_execute_model = Worker.execute_model

        def worker_execute_model(self, scheduler_output):
            runner = getattr(self, "model_runner", None)
            # vLLM V1 reports completed request ids on SchedulerOutput.  The
            # QCFuse producer and consumer are separate requests, so lifecycle
            # cleanup must run at the worker boundary before metadata for the
            # next step is built; QCFuse itself preserves a live QCOMPUTE
            # hand-off until its matching DO_BLEND consumes it.
            if has_ucm_sparse():
                get_ucm_sparse().update_states(scheduler_output)
            if runner is not None:
                connector = getattr(runner, "kv_connector", None)
                # vLLM 0.31 V2 wraps the configured UCM connector in its
                # ActiveKVConnector.  The public runner object therefore does
                # not expose UCM's setup_model directly; unwrap it here while
                # retaining compatibility with a direct connector object.
                ucm_connector = getattr(connector, "kv_connector", connector)
                setup_model = getattr(ucm_connector, "setup_model", None)
                if callable(setup_model) and not getattr(
                    runner, "_ucm_latest_store_attached", False
                ):
                    setup_model(runner.model)
                    runner._ucm_latest_store_attached = True
                if not getattr(runner, "_ucm_latest_instance_forward_wrapped", False):
                    import types
                    from ucm.integration.vllm.qcfuse_graph import install_v2_graph_hooks

                    install_v2_graph_hooks(runner, has_ucm_sparse, get_ucm_sparse)

                    original_prepare_inputs = runner.prepare_inputs
                    original_model_forward = runner.model.forward

                    def instance_prepare_inputs(runner_self, *args, **kwargs):
                        batch = original_prepare_inputs(*args, **kwargs)
                        runner_self._ucm_latest_input_batch = batch
                        return batch

                    runner.prepare_inputs = types.MethodType(
                        instance_prepare_inputs, runner
                    )

                    def instance_model_forward(model_self, *args, **kwargs):
                        scheduler = getattr(
                            runner, "_ucm_latest_pending_scheduler_output", None
                        )
                        if (
                            scheduler is not None
                            and getattr(
                                runner, "_ucm_latest_sparse_meta_built_for", None
                            )
                            is not scheduler
                            and has_ucm_sparse()
                        ):
                            from vllm.forward_context import get_forward_context

                            get_ucm_sparse().build_sparse_meta(
                                scheduler,
                                runner.req_states,
                                runner._ucm_latest_input_batch,
                                get_forward_context().attn_metadata,
                            )
                            runner._ucm_latest_sparse_meta_built_for = scheduler
                        return original_model_forward(*args, **kwargs)

                    runner.model.forward = types.MethodType(
                        instance_model_forward, runner.model
                    )
                    runner._ucm_latest_instance_forward_wrapped = True
                runner._ucm_latest_pending_scheduler_output = scheduler_output
                runner._ucm_latest_sparse_meta_built_for = None
            try:
                return original_worker_execute_model(self, scheduler_output)
            finally:
                if runner is not None:
                    runner._ucm_latest_pending_scheduler_output = None
                    runner._ucm_latest_sparse_meta_built_for = None

        Worker.execute_model = worker_execute_model
        Worker._ucm_latest_lifecycle_wrapped = True
    _INSTALLED = True
