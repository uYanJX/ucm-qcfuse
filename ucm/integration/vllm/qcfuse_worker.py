"""Named worker control methods for current vLLM's collective RPC API."""


class QcFuseWorkerExtension:
    def qcfuse_prepare_resident_batch(
        self, paths, mode, online_headroom_bytes=0, gpu_codec=None, query_sources=None
    ):
        from ucm.store.qcfuse.transfer import prepare_resident_batch

        return prepare_resident_batch(
            self, paths, mode, online_headroom_bytes, gpu_codec, query_sources
        )

    def qcfuse_benchmark_snapshot(self, reset_peak=False):
        from ucm.store.qcfuse.transfer import benchmark_snapshot

        result = benchmark_snapshot(self, reset_peak)
        from ucm.sparse.state import has_ucm_sparse, get_ucm_sparse

        if has_ucm_sparse():
            sparse = get_ucm_sparse()
            take_token_work = getattr(sparse, "take_token_work", None)
            if callable(take_token_work):
                rows = take_token_work()
                # The before snapshot discards offline/warmup observations.
                if not reset_peak:
                    result["token_work"] = rows
            store = getattr(sparse, "_store", None)
            metrics = getattr(store, "metrics", None)
            if callable(metrics):
                result["cache_transfer_totals"] = metrics()
                result["cache_transfer_scope"] = (
                    "cumulative_completed_request_metrics_subtract_before_from_after"
                )
        observation = getattr(self, "_qcfuse_forward_observation", None)
        if observation is not None:
            result["forward_observation"] = dict(observation)
        execution = getattr(self, "_qcfuse_execution_modes", None)
        if execution is not None:
            result["native_execution_modes"] = {
                **execution,
                "eager_input_token_histogram": dict(
                    execution["eager_input_token_histogram"]
                ),
            }
        if not reset_peak:
            handle = getattr(self, "_qcfuse_forward_observer_handle", None)
            if handle is not None:
                handle.remove()
                self._qcfuse_forward_observer_handle = None
        return result

    def qcfuse_start_execution_mode_observer(self):
        """Count actual eager forwards and native replays without CUDA sync."""
        if getattr(self, "_qcfuse_execution_modes", None) is not None:
            return dict(self._qcfuse_execution_modes)
        runner = self.model_runner
        manager = getattr(runner, "cudagraph_manager", None)
        graphs = getattr(manager, "graphs", {})
        state = self._qcfuse_execution_modes = {
            "available": bool(graphs),
            "configured_mode": str(getattr(manager, "cudagraph_mode", None)),
            "captured_token_counts": sorted({int(desc.num_tokens) for desc in graphs}),
            "eager_model_forwards": 0,
            "full_graph_replays": 0,
            "eager_input_token_histogram": {},
        }

        def eager_forward(_module, args, kwargs):
            value = kwargs.get("input_ids")
            if value is None:
                value = kwargs.get("inputs_embeds")
            if value is None and args:
                value = args[0]
            state["eager_model_forwards"] += 1
            if value is not None and hasattr(value, "shape"):
                key = str(int(value.shape[0]))
                histogram = state["eager_input_token_histogram"]
                histogram[key] = histogram.get(key, 0) + 1

        self._qcfuse_execution_mode_hook = runner.model.register_forward_pre_hook(
            eager_forward, with_kwargs=True
        )
        if manager is not None:
            original_replay = manager.run_fullgraph

            def replay(*args, **kwargs):
                output = original_replay(*args, **kwargs)
                state["full_graph_replays"] += 1
                return output

            manager.run_fullgraph = replay
        return dict(state)

    def qcfuse_memory_breakdown(self):
        """Attribute live model and native KV storage without copying tensors."""
        import torch
        from ucm.store.qcfuse.transfer import benchmark_snapshot

        model = self.model_runner.model
        parameters = {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for tensor in model.parameters()
            if tensor.numel()
        }
        cache_storages = {}
        cache_layers = []
        for name, module in model.named_modules():
            cache = getattr(module, "kv_cache", None)
            values = cache if isinstance(cache, (tuple, list)) else (cache,)
            for tensor in values:
                if not isinstance(tensor, torch.Tensor) or not tensor.numel():
                    continue
                storage = tensor.untyped_storage()
                cache_storages[storage.data_ptr()] = storage.nbytes()
                cache_layers.append(
                    {
                        "module": name,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "view_bytes": tensor.numel() * tensor.element_size(),
                    }
                )
        return {
            **benchmark_snapshot(self),
            "model_parameter_storage_bytes": sum(parameters.values()),
            "native_kv_storage_bytes": (
                sum(cache_storages.values()) if cache_layers else None
            ),
            "native_kv_layers": cache_layers,
            "memory_scope": (
                "unique_parameter_and_observed_layer_kv_storage;"
                "allocator_reserved_bytes_are_not_additional_live_tensors"
            ),
        }

    def qcfuse_start_execution_profile(self, path, max_forwards=3, skip_forwards=0):
        """Profile a bounded eager interval through a named, serializable RPC.

        The next model forward stops the profiler and removes the observer.
        Call qcfuse_stop_execution_profile in a finally block as well, since
        skip_forwards can exclude QCOMPUTE/prefill to observe decode alone.
        Generation may finish before reaching that boundary. Detailed profiles
        are diagnostic and must never supply the formal latency measurement.
        """
        import os
        import torch

        if isinstance(max_forwards, bool) or not isinstance(max_forwards, int):
            raise ValueError("max_forwards must be a positive integer")
        if max_forwards < 1:
            raise ValueError("max_forwards must be a positive integer")
        if (
            isinstance(skip_forwards, bool)
            or not isinstance(skip_forwards, int)
            or skip_forwards < 0
        ):
            raise ValueError("skip_forwards must be a non-negative integer")
        manager = getattr(self.model_runner, "cudagraph_manager", None)
        if getattr(manager, "graphs", None):
            raise ValueError("Bounded model-forward profiling requires eager execution")
        previous = getattr(self, "_qcfuse_execution_profile", None)
        if previous is not None and previous["profiler"] is not None:
            raise RuntimeError("An execution profile is already active")
        path = os.fspath(path).replace("{rank}", str(getattr(self, "rank", 0)))
        if os.path.exists(path):
            raise FileExistsError(path)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(activities=activities, record_shapes=True)
        state = {
            "path": path,
            "max_forwards": max_forwards,
            "forward_calls": 0,
            "skip_forwards": skip_forwards,
            "skipped_forward_calls": 0,
            "profiler": profiler,
            "profiler_started": skip_forwards == 0,
            "hook": None,
            "trace_written": False,
        }
        if state["profiler_started"]:
            profiler.start()
        self._qcfuse_execution_profile = state

        def before_forward(_model, _args):
            if state["skipped_forward_calls"] < skip_forwards:
                state["skipped_forward_calls"] += 1
                return
            if not state["profiler_started"]:
                profiler.start()
                state["profiler_started"] = True
            if state["forward_calls"] == max_forwards:
                self.qcfuse_stop_execution_profile()
            else:
                state["forward_calls"] += 1

        try:
            state["hook"] = self.model_runner.model.register_forward_pre_hook(
                before_forward
            )
        except BaseException:
            state["profiler"] = None
            if state["profiler_started"]:
                profiler.stop()
            raise
        return {
            "path": path,
            "max_forwards": max_forwards,
            "skip_forwards": skip_forwards,
            "active": True,
        }

    def qcfuse_stop_execution_profile(self):
        """Stop and export once; leave no profiler callback on later decode."""
        state = getattr(self, "_qcfuse_execution_profile", None)
        if state is None:
            return {"active": False, "trace_written": False, "forward_calls": 0}
        hook = state["hook"]
        state["hook"] = None
        if hook is not None:
            hook.remove()
        profiler = state["profiler"]
        state["profiler"] = None
        if profiler is not None and state["profiler_started"]:
            profiler.stop()
            profiler.export_chrome_trace(state["path"])
            state["trace_written"] = True
        return {
            "active": False,
            "path": state["path"],
            "forward_calls": state["forward_calls"],
            "skipped_forward_calls": state["skipped_forward_calls"],
            "trace_written": state["trace_written"],
        }

    def qcfuse_start_forward_observer(self):
        """Observe actual eager model input sizes in explicitly audited runs."""
        previous = getattr(self, "_qcfuse_forward_observer_handle", None)
        if previous is not None:
            previous.remove()
        self._qcfuse_forward_observation = {
            "forward_calls": 0,
            "max_input_tokens": 0,
            "input_token_histogram": {},
        }

        def observe(_module, args, kwargs):
            value = kwargs.get("input_ids")
            if value is None:
                value = kwargs.get("inputs_embeds")
            if value is None and args:
                value = args[0]
            if value is None or not hasattr(value, "shape"):
                return
            tokens = int(value.shape[0])
            counts = self._qcfuse_forward_observation
            counts["forward_calls"] += 1
            counts["max_input_tokens"] = max(counts["max_input_tokens"], tokens)
            histogram = counts["input_token_histogram"]
            histogram[str(tokens)] = histogram.get(str(tokens), 0) + 1

        self._qcfuse_forward_observer_handle = (
            self.model_runner.model.register_forward_pre_hook(observe, with_kwargs=True)
        )
        return {"observer": "eager_model_input_shapes", "active": True}
