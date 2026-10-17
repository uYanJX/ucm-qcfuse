import sys
from pathlib import Path
import pytest

from ucm.sparse.qcfuse.blend_info import BlendStyle, QcFuseRequestState
from ucm.sparse.qcfuse.qcfuse import QcFuse

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks" / "qcfuse"))

from utils import build_prompt_for_dataset, evaluate_sample


def test_warmup_reports_initial_preload_without_changing_generation(monkeypatch):
    from types import SimpleNamespace
    import blend_ssd

    engine = object.__new__(blend_ssd.SSDPipelineEngine)
    engine.batch_size = 1
    engine.baseline = "ours"
    engine.method = "attn"
    sample = SimpleNamespace(
        has_cache=True, query_sep="query", prompt="full", max_new_tokens=4096
    )
    engine._prepare_samples = lambda *_: [sample]
    calls = []
    engine._drain_generate_batch = lambda requests: calls.append(requests)
    engine._blend_args = lambda style, ratio: {"blend_style": style, "ratio": ratio}
    engine._ssd_args = lambda _: {}
    prepares = iter(
        [
            {"wall_seconds": 7.5, "workers": [{"resident_batch_reused": False}]},
            {"wall_seconds": 0.001, "workers": [{"resident_batch_reused": True}]},
        ]
    )
    engine._prepare_resident_batch = lambda _: next(prepares)
    clock = iter([0.0, 20.0])
    monkeypatch.setattr(blend_ssd.time, "perf_counter", lambda: next(clock))
    result = engine.warmup_blend([{}], "longbench_v2", 0.4, num_warmup=2)
    assert result["resident_prepares"][0]["wall_seconds"] == 7.5
    assert (
        result["resident_prepares"][0]["workers"][0]["resident_batch_reused"] is False
    )
    assert result["resident_prepares"][1]["workers"][0]["resident_batch_reused"] is True
    assert result["warmup_wall_seconds"] == 20.0 and result["repeats"] == 2
    assert len(calls) == 4
    assert [batch[0][2]["blend_style"] for batch in calls] == [
        "QCOMPUTE",
        "DO_BLEND_FINISH",
        "QCOMPUTE",
        "DO_BLEND_FINISH",
    ]
    assert calls[1][0][1]["max_new_tokens"] == calls[3][0][1]["max_new_tokens"] == 4096


def test_official_ruler_keeps_text_and_task_metric():
    from prepare_ruler import adapt_row
    from utils import sample_prompt_parts
    from blend_common import BlendEngineBase, BLEND_SEP

    raw = {
        "input": "Official instruction\nneedle=17\nWhat is the value?",
        "outputs": ["17", "seventeen"],
        "index": 0,
        "length": 65500,
        "answer_prefix": " The value is",
    }
    row = adapt_row(
        raw, "niah_single_1", {"task": "niah", "tokens_to_generate": 128}, "pinned"
    )
    assert row["context"] + row["input"] == raw["input"]
    engine = object.__new__(BlendEngineBase)
    engine.model_name = "qwen3-32b"
    system, docs, query, options = sample_prompt_parts(row, "ruler_64k")
    prompt, _ = engine._build_prompt(system, docs, query, use_sep=True, **options)
    assert raw["input"] in prompt.replace(BLEND_SEP, "")
    assert "Passage:" not in prompt and "## Answer" not in prompt
    assert prompt.endswith(raw["answer_prefix"])
    assert evaluate_sample("17", row["answers"], "ruler_64k", "niah_single_1") == 0.5
    assert evaluate_sample("17", row["answers"], "ruler_64k", "qa_1") == 1.0


def test_decode_counts_tokens_after_first_output(monkeypatch):
    from types import SimpleNamespace
    import blend_common

    engine = object.__new__(blend_common.BlendEngineBase)
    engine.batch_size = 1
    engine._rid_counter = 0
    engine._TokensPrompt = dict
    engine._build_sampling_params = lambda params, blend_args: params
    outputs = iter(
        [
            [
                SimpleNamespace(
                    request_id="qcfuse-1",
                    finished=False,
                    outputs=[SimpleNamespace(token_ids=[1, 2], text="first")],
                )
            ],
            [
                SimpleNamespace(
                    request_id="qcfuse-1",
                    finished=True,
                    outputs=[SimpleNamespace(token_ids=[1, 2, 3], text="done")],
                )
            ],
        ]
    )
    engine._engine = SimpleNamespace(
        add_request=lambda *a: None, step=lambda: next(outputs)
    )
    times = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr(blend_common.time, "perf_counter", lambda: next(times))
    record = engine._submit_batch_and_collect([([42], {}, None)])[0]
    assert record["ttft"] == 1.0 and record["decode"] == 1.0
    assert record["generated_tokens"] == 3
    assert record["first_token_output_tokens"] == 2 and record["decode_tokens"] == 1


def test_step_profile_separates_engine_wait_and_output_processing(
    tmp_path, monkeypatch
):
    import json
    from types import SimpleNamespace
    import blend_common

    engine = object.__new__(blend_common.BlendEngineBase)
    engine.batch_size = 1
    engine._rid_counter = 0
    engine._TokensPrompt = dict
    engine._build_sampling_params = lambda params, blend_args: params
    engine.generation_step_profile_path = tmp_path / "steps.jsonl"
    responses = iter(
        [
            [
                SimpleNamespace(
                    request_id="qcfuse-1",
                    finished=False,
                    outputs=[SimpleNamespace(token_ids=[1], text="first")],
                )
            ],
            [
                SimpleNamespace(
                    request_id="qcfuse-1",
                    finished=True,
                    outputs=[SimpleNamespace(token_ids=[1, 2], text="done")],
                )
            ],
        ]
    )
    engine._engine = SimpleNamespace(
        add_request=lambda *args: None, step=lambda: next(responses)
    )
    clock = iter(range(9))
    monkeypatch.setattr(blend_common.time, "perf_counter", lambda: next(clock))
    result = engine._submit_batch_and_collect([([42], {}, None)])[0]
    assert result["text"] == "done" and result["output_token_ids"] == [1, 2]
    assert result["ttft"] == 3 and result["decode"] == 4
    row = json.loads(engine.generation_step_profile_path.read_text())
    assert [s["engine_step_seconds"] for s in row["steps"]] == [1, 1]
    assert [s["frontend_output_seconds"] for s in row["steps"]] == [2, 2]
    assert [s["active_after"] for s in row["steps"]] == [1, 0]
    assert [s["new_output_tokens"] for s in row["steps"]] == [1, 1]


def test_overlong_input_is_rejected_instead_of_truncated():
    import pytest
    from blend_ssd import SSDPipelineEngine

    engine = object.__new__(SSDPipelineEngine)
    engine.context_length = 10
    engine._build_prompt = lambda *a, **kw: ("complete prompt", "")
    engine._prepare_generation = lambda *a: (list(range(11)), {}, {})
    with pytest.raises(ValueError, match="truncation is disabled"):
        engine._fit_context_to_model_window("", ["all context"], ["", "question"])


def test_cell_cleanup_requires_ownership_and_preserves_sibling(tmp_path):
    import pytest
    from run_frozen_cell import cleanup_owned_cache

    unrelated = tmp_path / "existing_cache"
    unrelated.mkdir()
    with pytest.raises(RuntimeError, match="ownership marker"):
        cleanup_owned_cache(unrelated)
    generated = tmp_path / "our_cell"
    generated.mkdir()
    (generated / ".qcfuse_cell_owner").write_text("our result")
    (generated / "kv.bin").write_bytes(b"generated cache")
    cleanup_owned_cache(generated)
    assert not generated.exists() and unrelated.exists()


def test_prompt_cache_reuses_only_tokens_and_keeps_request_args_separate():
    from blend_common import BlendEngineBase

    engine = object.__new__(BlendEngineBase)
    engine._sep_token = [999]
    calls = []

    def tokenize(prompt):
        calls.append(prompt)
        return [10, 999, 11, 999, 12]

    engine._tokenize = tokenize
    one = engine._prepare_generation(
        "same prompt", {}, {"ratio": 0.4, "session": "one"}
    )
    two = engine._prepare_generation(
        "same prompt", {}, {"ratio": 0.5, "session": "two"}
    )
    assert calls == ["same prompt"]
    assert one[0] == two[0] == [10, 11, 12]
    assert one[2]["ratio"] == 0.4 and two[2]["ratio"] == 0.5
    assert one[2]["session"] != two[2]["session"]
    assert one[2]["blend_loc_list"] == two[2]["blend_loc_list"]
    for index in range(10):
        engine._prepare_generation(f"other {index}", {}, {})
    assert len(engine._prompt_token_cache) == 8


def test_frozen_qcfuse_uses_compressed_query_context_like_main_runner():
    from blend_ssd import SSDPipelineEngine

    engine = object.__new__(SSDPipelineEngine)
    engine.model_name = "qwen3-32b"
    engine._get_model_config = lambda: {"num_layers": 64}
    engine.digest_index_method = "kvzip"
    engine.context_n_sink = 4
    engine.configure_baseline("ours", 3)
    args = engine._blend_args("QCOMPUTE", 0.4)
    assert args["is_contextblend"] is True
    assert args["context_cache_source"] == "query"
    assert args["digest_ratio"] == 0.1
    assert args["critical_layers"] == [48, 45, 50]


def test_worker_controls_use_named_rpc_without_function_serialization():
    from types import SimpleNamespace
    from blend_ssd import SSDPipelineEngine

    calls = []
    engine = object.__new__(SSDPipelineEngine)
    engine.baseline = "ours"
    engine.storage_mode = "cpu"
    engine.worker_snapshots_enabled = True
    engine.llm = SimpleNamespace(
        collective_rpc=lambda name, **kwargs: calls.append((name, kwargs)) or []
    )
    engine.release_resident_batch()
    engine._worker_snapshot(False)
    assert [name for name, _ in calls] == [
        "qcfuse_prepare_resident_batch",
        "qcfuse_benchmark_snapshot",
    ]


def test_qcompute_model_loop_stops_at_cache_end_and_restores_native_depth():
    import pytest
    from types import SimpleNamespace
    from ucm.integration.vllm.latest_sparse import _forward_with_layer_end

    model = SimpleNamespace(start_layer=0, end_layer=64)
    sparse = SimpleNamespace(effective_layer_end=lambda start, end: 51)
    seen = []

    def forward(model):
        seen.extend(range(model.start_layer, model.end_layer))
        return "query output"

    assert _forward_with_layer_end(model, forward, sparse) == "query output"
    assert seen[-1] == 50 and len(seen) == 51
    assert model.end_layer == 64

    def failure(model):
        assert model.end_layer == 51
        raise RuntimeError("request failure")

    with pytest.raises(RuntimeError, match="request failure"):
        _forward_with_layer_end(model, failure, sparse)
    assert model.end_layer == 64


@pytest.mark.parametrize("query_group_size", [None, 1, 2])
def test_pipeline_ttft_includes_qcompute_batch_barrier(monkeypatch, query_group_size):
    import blend_ssd
    from types import SimpleNamespace

    engine = object.__new__(blend_ssd.SSDPipelineEngine)
    engine.method = "attn"
    engine.baseline = "ours"
    engine.batch_size = 2
    engine.qcompute_batch_size = query_group_size
    engine.tokenizer = SimpleNamespace(
        all_special_ids=[99],
        convert_tokens_to_ids=lambda text: {"<think>": 1, "</think>": 2}[text],
        decode=lambda ids, **kw: "A" if ids else "",
    )
    samples = [
        blend_ssd.SSDSample(
            idx=index, answers=["A"], has_cache=True, max_new_tokens=4096
        )
        for index in range(2)
    ]
    engine._prepare_samples = lambda *args: samples
    engine._prepare_resident_batch = lambda batch: None
    engine._blend_args = lambda style, *args: {"blend_style": style}
    engine._ssd_args = lambda sample: {}
    query_outputs = [
        {"latency": 1.0, "request_id": "q0"},
        {"latency": 4.0, "request_id": "q1"},
    ]
    outputs = iter(
        (
            [query_outputs[:1], query_outputs[1:]]
            if query_group_size == 1
            else [query_outputs]
        )
        + [
            [
                {
                    "request_id": f"b{index}",
                    "text": "A",
                    "ttft": 0.5 + index * 0.3,
                    "first_token_time": 7.5 + index * 0.3,
                    "finished_time": 8.5 + index * 0.3,
                    "decode": 1.0,
                    "latency": 1.5,
                    "generated_tokens": 3,
                    "output_token_ids": [3, 3, 99],
                    "first_token_output_tokens": 1,
                    "decode_tokens": 2,
                }
                for index in range(2)
            ],
        ]
    )
    submitted = []

    def timed(requests):
        submitted.append(requests)
        return next(outputs)

    engine._timed_generate_batch = timed
    times = iter([0.0, 1.0, 5.0, 6.0, 9.0, 10.0])
    monkeypatch.setattr(blend_ssd.time, "perf_counter", lambda: next(times))
    result = engine.phase2_online([{}, {}], "longbench_v2", 0.4)
    assert result["ttft"] == [6.5, 6.8]
    assert result["records"][0]["engine_phase_ttft_seconds"] == 1.5
    assert result["do_blend_ttft"] == [0.5, 0.8]

    assert len(submitted[-1]) == 2
    assert all(request[1]["max_new_tokens"] == 4096 for request in submitted[-1])
    assert [request[2]["blend_style"] for request in submitted[-1]] == [
        "DO_BLEND_FINISH"
    ] * 2
    assert [len(group) for group in submitted[:-1]] == (
        [1, 1] if query_group_size == 1 else [2]
    )
    assert [r["qcompute_request_id"] for r in result["records"]] == ["q0", "q1"]
    assert result["phase_timings"][0]["qcompute_group_sizes"] == (
        [1, 1] if query_group_size == 1 else [2]
    )


def _stash():
    return {
        "q_buffer": ["q"],
        "query_k_buffer": ["k"],
        "q_lens": [1],
        "q_offsets": [0],
        "query_k_lens": [1],
        "created_at": 0.0,
    }


def test_longbench_context_is_one_complete_document():
    context = "full-context-" * 100
    docs, query = build_prompt_for_dataset(
        {
            "context": context,
            "question": "Which option is correct?",
            "choice_A": "first",
            "choice_B": "second",
            "choice_C": "third",
            "choice_D": "fourth",
        },
        "longbench_v2",
    )

    assert len(docs) == 1
    assert context in docs[0]
    assert "A. first" in query[1]
    assert "D. fourth" in query[1]


def test_ruler_context_list_preserves_all_passages():
    docs, _ = build_prompt_for_dataset(
        {"context": ["passage one", "passage two"], "input": "find it"},
        "ruler_qa_1",
    )
    assert len(docs) == 2
    assert "passage one" in docs[0]
    assert "passage two" in docs[1]


def test_ruler_64k_uses_ruler_prompt_and_answer_metric():
    docs, query = build_prompt_for_dataset(
        {"context": "long context", "input": "find the value"},
        "ruler_64k",
    )
    assert len(docs) == 1
    assert "## Question" in query[0]
    assert evaluate_sample("needle", ["needle"], "ruler_64k") == 1.0


def test_qcompute_stash_survives_producer_finish_until_consumer():
    producer = QcFuseRequestState(
        request_id="producer",
        blend_style=BlendStyle.QCOMPUTE,
        query_session_id="session",
    )
    consumer = QcFuseRequestState(
        request_id="consumer",
        blend_style=BlendStyle.DO_BLEND_FINISH,
        query_session_id="session",
    )
    fuser = object.__new__(QcFuse)
    fuser._worker_states = {"producer": producer, "consumer": consumer}
    fuser._qcompute_stash = {"session": _stash()}
    fuser._store = None

    fuser.request_finished_in_worker("producer")
    assert "session" in fuser._qcompute_stash

    fuser.request_finished_in_worker("consumer")
    assert "session" not in fuser._qcompute_stash
