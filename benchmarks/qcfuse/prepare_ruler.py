"""Run pinned NVIDIA generators and adapt their verbatim text for cache reuse."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from prepare_manifest import file_hash

QUERY_MARKERS = {
    "niah": "\nWhat ",
    "variable_tracking": "\nQuestion:",
    "common_words_extraction": "\nQuestion:",
    "freq_words_extraction": "\nQuestion:",
    "qa": "\n\nAnswer the question based on the given documents.",
}


def adapt_row(row, task, config, revision):
    raw = row["input"]
    split = raw.rfind(QUERY_MARKERS[config["task"]])
    if split < 1:
        raise ValueError(f"Cannot locate official query boundary for {task}")
    adapted = {
        "context": raw[:split],
        "input": raw[split:],
        "answers": row["outputs"],
        "answer_prefix": row.get("answer_prefix", ""),
        "prompt_format": "ruler_official_chat",
        "ruler_task": task,
        "max_new_tokens": config["tokens_to_generate"],
        "id": f"{task}:{row['index']}",
        "category": task,
        "generator_revision": revision,
        "raw_sample_sha256": hashlib.sha256(
            json.dumps(row, sort_keys=True).encode()
        ).hexdigest(),
        "generator_length": row["length"],
    }
    assert adapted["context"] + adapted["input"] == raw
    return adapted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(1, 50), required=True)
    parser.add_argument("--max-seq-length", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    import yaml

    scripts = args.source_tree / "scripts"
    constants = scripts / "data/synthetic/constants.py"
    spec = importlib.util.spec_from_file_location("official_ruler_constants", constants)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    customized = yaml.safe_load((scripts / "synthetic.yaml").read_text())
    tasks = list(customized) if args.size == 50 else ["niah_single_1"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output_dir / "generation_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"Frozen generation already exists: {receipt_path}")
    receipt, adapted = [], []
    for index, task in enumerate(tasks):
        count = 1 if args.size == 1 else (4 if index < 11 else 3)
        config = {**customized[task], **module.TASKS[customized[task]["task"]]}
        raw_path = args.output_dir / "raw" / task / "validation.jsonl"
        command = [
            sys.executable,
            str(scripts / f"data/synthetic/{config['task']}.py"),
            "--save_dir",
            str(args.output_dir / "raw"),
            "--save_name",
            task,
            "--tokenizer_path",
            str(args.model),
            "--tokenizer_type",
            "hf",
            "--max_seq_length",
            str(args.max_seq_length),
            "--tokens_to_generate",
            str(config["tokens_to_generate"]),
            "--num_samples",
            str(count),
            "--random_seed",
            str(args.seed),
            "--template",
            config["template"] + config.get("answer_prefix", ""),
        ]
        for key, value in config["args"].items():
            command.extend([f"--{key}", str(value)])
        if not raw_path.exists():
            print(json.dumps({"task": task, "command": command}), flush=True)
            subprocess.run(command, check=True)
        with raw_path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if len(rows) != count:
            raise ValueError(f"{task}: expected {count}, found {len(rows)}")
        adapted.extend(adapt_row(row, task, config, args.revision) for row in rows)
        receipt.append(
            {
                "task": task,
                "count": count,
                "command": command,
                "raw_sha256": file_hash(raw_path),
                "generator_sha256": file_hash(Path(command[1])),
            }
        )
    assert len(adapted) == args.size
    with (args.output_dir / "ruler_64k.jsonl").open("x") as stream:
        for row in adapted:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with receipt_path.open("x") as stream:
        json.dump(
            {
                "revision": args.revision,
                "seed": args.seed,
                "protocol": "official_generator_base_text_in_qwen_chat_wrapper",
                "counts": receipt,
                "total": len(adapted),
                "yaml_sha256": file_hash(scripts / "synthetic.yaml"),
                "constants_sha256": file_hash(constants),
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
