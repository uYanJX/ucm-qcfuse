"""Freeze a seeded subset without truncation or a vLLM/CUDA engine."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from blend_common import BlendEngineBase, BLEND_SEP, _tokenize_segment
from evidence import validate_dataset_rows
from utils import sample_prompt_parts, get_max_new_tokens


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thinking", choices=("on", "off"), default="off")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()
    if args.size < 1 or (args.max_new_tokens is not None and args.max_new_tokens < 1):
        parser.error("Sample count and explicit output budget must be positive")
    # JSON strings may contain Unicode line separators; only newline-delimited
    # file iteration respects JSONL record boundaries.
    with args.source.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    validate_dataset_rows(rows, args.dataset)
    indices = random.Random(args.seed).sample(range(len(rows)), args.size)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    builder = object.__new__(BlendEngineBase)
    builder.model_name = args.model.name.lower()
    builder.thinking_mode = args.thinking
    entries, subset = [], []
    for index in indices:
        example = rows[index]
        system, docs, query, options = sample_prompt_parts(example, args.dataset)
        prompt, _ = builder._build_prompt(system, docs, query, use_sep=True, **options)
        ids = [
            token
            for part in prompt.split(BLEND_SEP)
            for token in _tokenize_segment(tokenizer, part)
        ]
        entries.append(
            {
                "source_index": index,
                "source_id": example.get("_id", example.get("id", index)),
                "sample_sha256": hashlib.sha256(
                    json.dumps(example, sort_keys=True).encode()
                ).hexdigest(),
                "prompt_sha256": hashlib.sha256(
                    prompt.replace(BLEND_SEP, "").encode()
                ).hexdigest(),
                "prompt_token_ids_sha256": hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()
                ).hexdigest(),
                "input_tokens": len(ids),
                "context_tokens": len(_tokenize_segment(tokenizer, "".join(docs))),
                "task": example.get("ruler_task"),
                "max_new_tokens": args.max_new_tokens
                or example.get("max_new_tokens", get_max_new_tokens(args.dataset)),
                "truncated": False,
            }
        )
        subset.append(example)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Refuse accidental overwrite of a frozen subset.
    with (args.output_dir / f"{args.dataset}.jsonl").open("x") as stream:
        for row in subset:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "count": len(entries),
        "source": str(args.source),
        "source_sha256": file_hash(args.source),
        "source_authenticity": "requires_provenance_verification",
        "model": str(args.model),
        "model_config_sha256": file_hash(args.model / "config.json"),
        "tokenizer_sha256": file_hash(args.model / "tokenizer.json"),
        "samples": entries,
        "prompt_settings": {
            "thinking": args.thinking,
            "max_new_tokens_override": args.max_new_tokens,
            "ruler_answer_prefix_role": (
                "user" if args.thinking == "on" else "assistant"
            ),
        },
        "tokenizer_config_sha256": file_hash(args.model / "tokenizer_config.json"),
    }
    receipt_path = args.source.parent / "generation_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        payload.update(
            source_authenticity="pinned_nvidia_ruler_generator",
            generator_revision=receipt["revision"],
            generation_receipt_sha256=file_hash(receipt_path),
            prompt_protocol=receipt["protocol"],
        )
    payload["dataset_sha256"] = file_hash(args.output_dir / f"{args.dataset}.jsonl")
    with (args.output_dir / "manifest.json").open("x") as stream:
        json.dump(payload, stream, indent=2)
    print(
        json.dumps(
            {
                "count": len(entries),
                "min_tokens": min(x["input_tokens"] for x in entries),
                "max_tokens": max(x["input_tokens"] for x in entries),
                "output": str(args.output_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
