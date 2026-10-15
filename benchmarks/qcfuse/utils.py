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

The benchmark runner must preserve the complete context.  In particular,
LongBench v2 stores context as one large string while HotpotQA stores a list of
passages; treating both values as an iterable of passages silently truncated or
split the LongBench input in the old implementation.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

RULER_DATASETS = (
    "ruler_64k",
    "ruler_cwe",
    "ruler_fwe",
    "ruler_niah_multikey_1",
    "ruler_niah_multikey_2",
    "ruler_niah_multikey_3",
    "ruler_niah_multiquery",
    "ruler_niah_multivalue",
    "ruler_niah_single_1",
    "ruler_niah_single_2",
    "ruler_niah_single_3",
    "ruler_qa_1",
    "ruler_qa_2",
    "ruler_vt",
    "ruler_mq",
    "ruler_mv",
)

LONG_BENCH_DATASETS = ("longbench_v2",)

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

GENERIC_RULER_SYSTEM = (
    "Read the text below and answer the question using only information from it. "
    "Return only the answer without an explanation.\n\n## Text\n"
)
GENERIC_RULER_QUERY = (
    "Answer the question using only the provided text. "
    "Return only the answer without an explanation.\n\n## Question\n"
)

LONGBENCH_QUERY_PREFIX = (
    "Answer the multiple-choice question using only the provided context. "
    "Return only the option letter (A, B, C, or D).\n\n## Question\n"
)

MAX_NEW_TOKENS = 48

MAX_NEW_TOKENS_BY_DATASET = {
    "ruler_vt": 30,
    "ruler_mq": 128,
    "ruler_mv": 128,
}


def load_dataset(dataset_path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load up to ``limit`` records from a JSONL dataset file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    with open(path, "r", encoding="utf-8") as f:
        records = []
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
        return records


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
    if example.get("prompt_format") == "ruler_official_chat":
        # The generator's instruction and context stay verbatim. Only chat
        # role wrappers and QCFuse document boundaries are added.
        return [example["context"]], ["", example["input"]]
    context = example.get("context", "")
    if isinstance(context, str):
        context_parts = [context] if context else []
    elif context is None:
        context_parts = []
    else:
        # LongBench uses a string, while the other long-context datasets may
        # provide a list of passages.  Preserve every supplied passage.
        context_parts = [
            item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            for item in context
        ]
    docs = [f"Passage:\n{ctx}\n\n" for ctx in context_parts]

    input_text = example.get("input", "")
    if dataset_name in RULER_DATASETS:
        return docs, [
            RULER_QUERY_PREFIX.get(dataset_name, GENERIC_RULER_QUERY),
            input_text,
        ]

    if dataset_name in LONG_BENCH_DATASETS:
        question = example.get("question", input_text)
        choices = []
        for label in ("A", "B", "C", "D"):
            if f"choice_{label}" in example:
                choices.append(f"{label}. {example[f'choice_{label}']}")
        choice_text = "\n".join(choices)
        query = str(question)
        if choice_text:
            query = f"{query}\n\n{choice_text}"
        return docs, [LONGBENCH_QUERY_PREFIX, query]

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
    "ruler_64k": "string_match_all",
    "longbench_v2": "accuracy",
}

METRIC_DISPLAY = {
    "f1": "F1",
    "string_match_all": "sm",
    "accuracy": "Accuracy",
}


def scorer_accuracy(prediction: str, ground_truth: str) -> float:
    """Score a LongBench v2 multiple-choice answer."""
    expected = str(ground_truth).strip().upper()
    if not expected:
        return 0.0
    response = str(prediction).replace("*", "").strip()
    # Official LongBench v2 pred.py extracts these final-answer phrases.
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match is None:
        match = re.search(r"The correct answer is ([A-D])", response)
    if match is None:
        # This runner also requests a bare option letter. Reject ambiguous
        # multi-letter strings rather than accepting the first character.
        match = re.fullmatch(r"\(?([A-D])\)?[.)]?", response.upper())
    return float(match is not None and match.group(1) == expected)


def evaluate_sample(
    prediction: str,
    ground_truths: List[str],
    dataset_name: str,
    ruler_task: Optional[str] = None,
) -> float:
    """Evaluate prediction against ground truths; returns max score."""
    if not ground_truths:
        return 0.0
    if ruler_task or dataset_name.startswith("ruler"):
        if ruler_task in ("qa_1", "qa_2") or dataset_name in (
            "ruler_qa_1",
            "ruler_qa_2",
        ):
            return max(
                float(str(truth).lower() in prediction.lower())
                for truth in ground_truths
            )
        return scorer_string_match_all(prediction, ground_truths)
    metric_type = TASK_METRICS.get(dataset_name, "f1")
    if metric_type == "string_match_all":
        return scorer_string_match_all(prediction, ground_truths)
    if metric_type == "accuracy":
        return max(scorer_accuracy(prediction, truth) for truth in ground_truths)

    best = 0.0
    for truth in ground_truths:
        score = scorer_f1(prediction, truth)
        if score > best:
            best = score
    return best


def get_system_prompt(_dataset_name: str) -> str:
    if _dataset_name in RULER_DATASETS:
        return RULER_SYS_INSTRUCT.get(_dataset_name, GENERIC_RULER_SYSTEM)
    return SYSTEM_PROMPT


def get_max_new_tokens(_dataset_name: str) -> int:
    if _dataset_name.startswith("ruler"):
        return MAX_NEW_TOKENS_BY_DATASET.get(_dataset_name, 128)
    return MAX_NEW_TOKENS_BY_DATASET.get(_dataset_name, MAX_NEW_TOKENS)


def get_metric_name(dataset_name: str) -> str:
    if dataset_name.startswith("ruler"):
        return "sm"
    metric_type = TASK_METRICS.get(dataset_name, "f1")
    return METRIC_DISPLAY.get(metric_type, "Score")


def sample_prompt_parts(example: Dict, dataset_name: str):
    docs, query = build_prompt_for_dataset(example, dataset_name)
    if example.get("prompt_format") == "ruler_official_chat":
        return (
            "",
            docs,
            query,
            {"answer_suffix": "", "answer_prefix": example.get("answer_prefix", "")},
        )
    return get_system_prompt(dataset_name), docs, query, {}
