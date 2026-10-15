"""Keep QCFuse prefill eager and refresh V2 state before native graph replay."""


def build_v2_sparse_meta(runner, scheduler, attn_metadata, has_sparse, get_sparse):
    if (
        scheduler is not None
        and getattr(runner, "_ucm_latest_sparse_meta_built_for", None) is not scheduler
        and has_sparse()
    ):
        get_sparse().build_sparse_meta(
            scheduler,
            runner.req_states,
            runner._ucm_latest_input_batch,
            attn_metadata,
        )
        runner._ucm_latest_sparse_meta_built_for = scheduler


def install_v2_graph_hooks(runner, has_sparse, get_sparse):
    connector = runner.kv_connector
    if not getattr(connector, "_ucm_latest_pre_forward_wrapped", False):
        original_pre_forward = connector.pre_forward

        def pre_forward(*args, **kwargs):
            result = original_pre_forward(*args, **kwargs)
            scheduler = kwargs.get(
                "scheduler_output",
                (
                    args[0]
                    if args
                    else getattr(runner, "_ucm_latest_pending_scheduler_output", None)
                ),
            )
            # V2 also calls pre_forward for a zero-token cleanup/heartbeat.
            # That path has no input batch or forward context to inspect.
            if getattr(scheduler, "total_num_scheduled_tokens", None) == 0:
                return result
            if scheduler is not None and has_sparse():
                if "attn_metadata" in kwargs:
                    attn_metadata = kwargs["attn_metadata"]
                else:
                    from vllm.forward_context import get_forward_context

                    attn_metadata = get_forward_context().attn_metadata
                build_v2_sparse_meta(
                    runner, scheduler, attn_metadata, has_sparse, get_sparse
                )
            return result

        connector.pre_forward = pre_forward
        connector._ucm_latest_pre_forward_wrapped = True

    manager = getattr(runner, "cudagraph_manager", None)
    if manager is not None and not getattr(
        manager, "_ucm_qcfuse_dispatch_wrapped", False
    ):
        original_dispatch = manager.dispatch

        def dispatch(
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras=0,
            max_query_len=None,
            num_ubatches=1,
        ):
            scheduler = getattr(runner, "_ucm_latest_pending_scheduler_output", None)
            metadata = getattr(scheduler, "kv_connector_metadata", None)
            request_meta = getattr(metadata, "request_meta", {}) or {}
            scheduled = getattr(scheduler, "num_scheduled_tokens", {})
            # V2 sets this bound to None for a batch containing prefill.
            # A one-token QCOMPUTE must not be mistaken for ordinary decode.
            if max_query_len is None and any(
                req_id in scheduled and getattr(item, "state", None) is not None
                for req_id, item in request_meta.items()
            ):
                from vllm.config.compilation import CUDAGraphMode
                from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

                return BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.NONE,
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                    num_active_loras=num_active_loras,
                    num_ubatches=num_ubatches,
                )
            return original_dispatch(
                num_reqs,
                num_tokens,
                uniform_token_count,
                num_active_loras=num_active_loras,
                max_query_len=max_query_len,
                num_ubatches=num_ubatches,
            )

        manager.dispatch = dispatch
        manager._ucm_qcfuse_dispatch_wrapped = True
