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
LongBench / RULER utilities for Blend evaluation.

Supports: hotpotqa, 2wikimqa, musique, ruler_vt, ruler_mq, ruler_mv
Metrics : F-Measure, string-match-all
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple

RULER_DATASETS = ("ruler_vt", "ruler_mq", "ruler_mv")

SYSTEM_PROMPT = (
    "You are a highly precise question-answering assistant.\n\n"
    "## Task\n"
    "Read the provided passages and answer the user's question based strictly "
    "on the information within.\n\n"
    "## Output Rules\n"
    "- Direct Answer ONLY: Output nothing but the final exact answer.\n"
    "- No Explanations: Do not provide reasoning, context, conversational "
    "fillers, or extra words.\n\n"
    "## Passages\n"
)

QUERY_PREFIX = (
    "Remember to answer the question based strictly on the passages above. "
    "Output ONLY the answer and no other words.\n\n## Question\n"
)

RULER_VT_FEWSHOT = (
    "Example:\n"
    "Text:\n"
    "The maintenance report begins with routine notes about lighting, archived boxes, "
    "and a schedule change for the west corridor. A coordinator wrote that several "
    "labels had been moved after the weekly inspection, but most of the paragraph is "
    "ordinary filler that should not be treated as an assignment. Near the inventory "
    "table, the report states VAR ABCDE = 12345 before describing spare cables and "
    "a delayed delivery. Later, after a note about temperature readings, it says "
    "VAR FGHIJ = VAR ABCDE. The next page mentions visitor badges, old invoices, "
    "and two unrelated serial numbers. Hidden between those details is "
    "VAR KLMNO = VAR FGHIJ. The report then discusses a broken cart, a missing "
    "clipboard, and yesterday's storage request. After that, the chain continues: "
    "VAR PQRST = VAR KLMNO. The closing paragraph talks about cleaning supplies, "
    "meeting times, and duplicate copies of the same memo, then finally records "
    "VAR UVWXY = VAR PQRST. A later appendix lists department codes, desk numbers, "
    "and several dates from a training calendar. Those details are only background "
    "text, even when they contain digits or capitalized words. Another section says "
    "that the archive door was repaired, the finance folder was renamed, and the "
    "morning checklist should be reviewed before the next shift. The example also "
    "mentions that the copied forms were sorted by color, that the hallway map was "
    "reprinted, and that temporary notes should be discarded after confirmation. "
    "None of these surrounding statements changes the assignment chain above. "
    "No other variable in this example is assigned the "
    "value 12345 through this chain.\n"
    "Question: Find all variables that are assigned the value 12345 in the text above.\n"
    "Answer: ABCDE, FGHIJ, KLMNO, PQRST, UVWXY\n\n"
)

RULER_SYS_INSTRUCT = {
    "ruler_mq": (
        "Some special magic numbers are hidden within the following text. "
        "Make sure to memorize it. I will quiz you about the numbers afterwards.\n\n"
        "Return only the requested magic numbers. If there are multiple numbers, "
        "separate them with commas and do not explain.\n\n"
        "## Text\n"
    ),
    "ruler_mv": (
        "Some special magic numbers are hidden within the following text. "
        "Make sure to memorize it. I will quiz you about the numbers afterwards.\n\n"
        "Return only the requested magic numbers. If there are multiple numbers, "
        "separate them with commas and do not explain.\n\n"
        "## Text\n"
    ),
    "ruler_vt": (
        "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
        f"{RULER_VT_FEWSHOT}"
        "For the actual text below, return only the variable names assigned to the queried value. "
        "If there are multiple variable names, separate them with commas and do not explain.\n\n"
        "## Text\n"
    ),
}

RULER_QUERY_PREFIX = {
    "ruler_mq": (
        "Answer the question using only the provided text. "
        "Return only the requested magic number or numbers, separated by commas.\n\n## Question\n"
    ),
    "ruler_mv": (
        "Answer the question using only the provided text. "
        "Return only the requested magic number or numbers, separated by commas.\n\n## Question\n"
    ),
    "ruler_vt": (
        "Answer the question using only the provided text. "
        "Return only the requested variable name or names, separated by commas.\n\n## Question\n"
    ),
}

MAX_NEW_TOKENS = 48

MAX_NEW_TOKENS_BY_DATASET = {
    "ruler_vt": 30,
    "ruler_mq": 128,
    "ruler_mv": 128,
}

DEFAULT_CHUNK_TOPK = 20


def load_dataset(dataset_path: str) -> List[Dict]:
    """Load a JSONL dataset file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_question(question: str) -> str:
    """Lowercase first letter and ensure trailing '?'."""
    if not question:
        return ""
    if not question.endswith("?"):
        question += "?"
    return question[0].lower() + question[1:]


def build_prompt_for_dataset(
    example: Dict, dataset_name: str
) -> Tuple[List[str], List[str]]:
    """Build document list and question prompt.

    Returns:
        docs: list of passage strings
        q_prompt: [query_prefix, input_text]
    """
    context = example.get("context", "")
    if dataset_name not in RULER_DATASETS:
        context = context[: min(len(context), DEFAULT_CHUNK_TOPK)]
    docs = [f"Passage:\n{ctx}\n\n" for ctx in context]

    input_text = example.get("input", "")
    if dataset_name in RULER_DATASETS:
        return docs, [RULER_QUERY_PREFIX[dataset_name], input_text]

    return docs, [QUERY_PREFIX, normalize_question(input_text)]


def scorer_f1(prediction: str, ground_truth: str) -> float:
    """F1 score from rouge_scorer."""
    if not prediction.strip() or not ground_truth.strip():
        return 0.0
    # Lazy import: rouge_score is only needed for the F1 LongBench tasks, so the
    # RULER-only / CPU unit tests can import this module without it installed.
    from rouge_score import rouge_scorer as _rouge_scorer

    scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(ground_truth, prediction)["rougeL"].fmeasure


def scorer_string_match_all(prediction: str, ground_truths: List[str]) -> float:
    """RULER-style score: fraction of references found in the prediction."""
    if not ground_truths:
        return 0.0
    prediction = prediction.lower()
    hits = sum(
        1.0 if str(truth).lower() in prediction else 0.0 for truth in ground_truths
    )
    return hits / len(ground_truths)


TASK_METRICS = {
    "hotpotqa": "f1",
    "2wikimqa": "f1",
    "musique": "f1",
    "ruler_vt": "string_match_all",
    "ruler_mq": "string_match_all",
    "ruler_mv": "string_match_all",
}

METRIC_DISPLAY = {
    "f1": "F1",
    "string_match_all": "sm",
}


def evaluate_sample(
    prediction: str,
    ground_truths: List[str],
    dataset_name: str,
) -> float:
    """Evaluate prediction against ground truths; returns max score."""
    if not ground_truths:
        return 0.0
    metric_type = TASK_METRICS.get(dataset_name, "f1")
    if metric_type == "string_match_all":
        return scorer_string_match_all(prediction, ground_truths)

    best = 0.0
    for truth in ground_truths:
        score = scorer_f1(prediction, truth)
        if score > best:
            best = score
    return best


def get_system_prompt(_dataset_name: str) -> str:
    return RULER_SYS_INSTRUCT.get(_dataset_name, SYSTEM_PROMPT)


def get_max_new_tokens(_dataset_name: str) -> int:
    return MAX_NEW_TOKENS_BY_DATASET.get(_dataset_name, MAX_NEW_TOKENS)


def get_metric_name(dataset_name: str) -> str:
    metric_type = TASK_METRICS.get(dataset_name, "f1")
    return METRIC_DISPLAY.get(metric_type, "Score")
