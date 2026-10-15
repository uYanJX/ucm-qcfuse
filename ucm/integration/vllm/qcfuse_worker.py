"""Named worker control methods for current vLLM's collective RPC API."""


class QcFuseWorkerExtension:
    def qcfuse_prepare_resident_batch(self, paths, mode, online_headroom_bytes=0):
        from ucm.store.qcfuse.transfer import prepare_resident_batch

        return prepare_resident_batch(self, paths, mode, online_headroom_bytes)

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
        if not reset_peak:
            handle = getattr(self, "_qcfuse_forward_observer_handle", None)
            if handle is not None:
                handle.remove()
                self._qcfuse_forward_observer_handle = None
        return result

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
