import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks/qcfuse"))

from output_protocol import split_output
from evidence import validate_prompt_settings
from utils import evaluate_sample


class Tokenizer:
    all_special_ids = [99]

    def convert_tokens_to_ids(self, text):
        return {"<think>": 1, "</think>": 2}[text]

    def decode(self, ids, **kwargs):
        return "".join({3: "A", 4: "B"}.get(token, "") for token in ids)


def test_reasoning_does_not_leak_into_choice_accuracy():
    output = split_output([1, 3, 2, 4, 99], Tokenizer(), True)
    assert output["thinking_text"] == "A" and output["answer_text"] == "B"
    assert evaluate_sample(output["answer_text"], ["A"], "longbench_v2") == 0.0
    assert output["thinking_tokens"] == output["answer_tokens"] == 1
    assert output["control_tokens"] == 3 and output["total_output_tokens"] == 5


def test_unfinished_reasoning_has_no_scored_answer():
    output = split_output([1, 3, 99], Tokenizer(), True)
    assert output["thinking_status"] == "unfinished"
    assert output["answer_text"] == "" and output["answer_tokens"] == 0
    assert evaluate_sample(output["answer_text"], ["A"], "ruler_64k") == 0.0


def test_nonthinking_counts_exclude_end_of_turn_control_token():
    output = split_output([4, 99], Tokenizer(), False)
    assert output["thinking_status"] == "disabled"
    assert output["thinking_tokens"] == 0 and output["answer_tokens"] == 1
    assert (
        output["total_output_tokens"]
        == output["answer_tokens"] + output["control_tokens"]
    )


def test_formal_output_budget_cannot_silently_use_old_protocol():
    manifest = {"prompt_settings": {"thinking": "on", "max_new_tokens_override": 4096}}
    validate_prompt_settings(manifest, "on", 4096)
    with pytest.raises(ValueError, match="differs"):
        validate_prompt_settings(manifest, "off", None)
    with pytest.raises(ValueError, match="differs"):
        validate_prompt_settings({}, "on", 4096)


def test_thinking_prompt_leaves_checkpoint_assistant_prefix_empty():
    from blend_common import BlendEngineBase, BLEND_SEP

    engine = object.__new__(BlendEngineBase)
    engine.model_name = "qwen3-32b"
    engine.thinking_mode = "on"
    prompt, _ = engine._build_prompt(
        "",
        ["context"],
        ["", "question"],
        True,
        answer_suffix="",
        answer_prefix=" Answer:",
    )
    plain = prompt.replace(BLEND_SEP, "")
    assert plain.endswith("question Answer:<|im_end|>\n<|im_start|>assistant\n")
    assert "<think>" not in plain


@pytest.mark.parametrize(
    "answer",
    ["A", "(A)", "A.", "The correct answer is (A)", "The correct answer is **A**"],
)
def test_choice_metric_accepts_final_answer_formats(answer):
    assert evaluate_sample(answer, ["A"], "longbench_v2") == 1.0


@pytest.mark.parametrize("answer", ["AB", "A or B", "", "The correct answer is (B)"])
def test_choice_metric_rejects_ambiguous_or_wrong_answer(answer):
    assert evaluate_sample(answer, ["A"], "longbench_v2") == 0.0


def test_first_thinking_content_and_answer_have_distinct_observed_times():
    from output_protocol import OutputProgress

    progress = OutputProgress(Tokenizer(), True)
    progress.update([1], 1.0)
    assert progress.first_thinking_time is None
    progress.update([1, 3], 2.0)
    progress.update([1, 3, 2], 3.0)
    progress.update([1, 3, 2, 4, 99], 4.0)
    assert progress.first_thinking_time == 2.0
    assert progress.first_answer_time == 4.0
