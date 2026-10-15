"""Fail closed on unverified baseline names and known placeholder datasets."""

import hashlib
import json
from pathlib import Path

RATIO_SEMANTICS = "document_token_recompute_budget_after_selection"


def index_token_audit(rows):
    indexed = {}
    for row in rows:
        key = row.get("client_request_id") or row["request_id"]
        if key in indexed:
            raise ValueError(f"Duplicate token audit for client request {key}")
        indexed[key] = row
    return indexed


def baseline_identity(name, allow_unverified=False):
    if name in ("fullcomp", "ours"):
        return {
            "baseline": name,
            "requested_baseline": name,
            "implementation_status": (
                "native_full_prefill" if name == "fullcomp" else "ucm_qcfuse_port"
            ),
            "comparison_eligible": True,
        }
    if name == "cacheblend":
        raise ValueError(
            "CacheBlend is N/A in this runner: no validated latest-vLLM adapter"
        )
    if name not in ("prophetkv", "fuserag"):
        raise ValueError(f"Unknown baseline: {name}")
    if not allow_unverified:
        raise ValueError(
            f"{name} is an unverified QCFuse-runner variant. Use "
            "--allow-unverified-baseline only for diagnostics; it is not eligible "
            "for the four-method comparison. fuserag is not CacheBlend."
        )
    return {
        "baseline": f"{name}_unverified",
        "requested_baseline": name,
        "implementation_status": "unverified_qcfuse_variant",
        "comparison_eligible": False,
    }


def validate_dataset_rows(rows, dataset_name):
    if not dataset_name.startswith("ruler"):
        return
    for index, row in enumerate(rows):
        context = str(row.get("context", ""))
        answers = row.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        if "Official RULER ruler_" in context or any(
            str(a).startswith("answer_ruler_") for a in answers
        ):
            raise ValueError(
                f"{dataset_name} row {index} is a known placeholder fixture, "
                "not NVIDIA RULER generator output; refusing benchmark."
            )


def validate_frozen_manifest(path, rows, model_path, max_model_len):
    """Verify exact sample order and checkpoint before allocating a GPU engine."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    samples = manifest["samples"]
    if len(rows) > len(samples):
        raise ValueError("Dataset is larger than its frozen manifest")
    config_hash = hashlib.sha256(
        (Path(model_path) / "config.json").read_bytes()
    ).hexdigest()
    if config_hash != manifest["model_config_sha256"]:
        raise ValueError("Checkpoint config differs from the frozen manifest")
    for key, filename in (
        ("tokenizer_sha256", "tokenizer.json"),
        ("tokenizer_config_sha256", "tokenizer_config.json"),
    ):
        if key in manifest:
            actual = hashlib.sha256(
                (Path(model_path) / filename).read_bytes()
            ).hexdigest()
            if actual != manifest[key]:
                raise ValueError(f"{filename} differs from frozen manifest")
    for index, (row, sample) in enumerate(zip(rows, samples)):
        digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if digest != sample["sample_sha256"]:
            raise ValueError(
                f"Dataset row {index} differs from frozen sample order/content"
            )
    overlong = [
        index
        for index, sample in enumerate(samples[: len(rows)])
        if sample["input_tokens"] + int(sample.get("max_new_tokens") or 48)
        > max_model_len
    ]
    if overlong:
        raise ValueError(
            f"{len(overlong)} complete inputs exceed max_model_len={max_model_len}; "
            f"frozen row indices {overlong[:8]}; truncation is disabled"
        )
    provenance = {
        "manifest_path": str(path.resolve()),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_config_sha256": config_hash,
        "dataset_seed": manifest["seed"],
        "source_authenticity": manifest.get("source_authenticity"),
        "prompt_settings": manifest.get(
            "prompt_settings", {"thinking": "off", "max_new_tokens_override": None}
        ),
    }
    verification_path = path.parent / "source_verification.json"
    if verification_path.exists():
        verification = json.loads(verification_path.read_text())
        if (
            verification["status"] != "verified"
            or verification["manifest_sha256"] != provenance["manifest_sha256"]
            or verification["source_sha256"] != manifest["source_sha256"]
        ):
            raise ValueError("Source verification does not match frozen manifest")
        provenance.update(
            source_authenticity="verified_official_dataset",
            source_verification=verification,
        )
    return provenance


def validate_prompt_settings(manifest, thinking, max_new_tokens):
    settings = manifest.get(
        "prompt_settings", {"thinking": "off", "max_new_tokens_override": None}
    )
    if (
        thinking != settings["thinking"]
        or max_new_tokens != settings["max_new_tokens_override"]
    ):
        raise ValueError("Thinking/output budget differs from frozen prompt settings")
