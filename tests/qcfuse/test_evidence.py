import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks" / "qcfuse"))
from evidence import baseline_identity, validate_dataset_rows


def test_audit_uses_explicit_client_id_without_guessing_internal_suffix():
    from evidence import index_token_audit

    row = {
        "request_id": "qcfuse-4-aee46d12",
        "client_request_id": "qcfuse-4",
        "selected_tokens": 6581,
    }
    assert index_token_audit([row])["qcfuse-4"] is row
    with pytest.raises(ValueError, match="Duplicate"):
        index_token_audit([row, row])


def test_manifest_preflight_checks_content_and_capacity(tmp_path):
    import hashlib
    import json
    from evidence import validate_frozen_manifest

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}')
    row = {"context": "whole source", "answers": ["17"]}
    manifest = {
        "seed": 20261013,
        "model_config_sha256": hashlib.sha256(
            (model / "config.json").read_bytes()
        ).hexdigest(),
        "samples": [
            {
                "sample_sha256": hashlib.sha256(
                    json.dumps(row, sort_keys=True).encode()
                ).hexdigest(),
                "input_tokens": 65500,
                "max_new_tokens": 128,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="truncation is disabled"):
        validate_frozen_manifest(path, [row], model, 65536)
    assert (
        validate_frozen_manifest(path, [row], model, 131072)["dataset_seed"] == 20261013
    )
    with pytest.raises(ValueError, match="order/content"):
        validate_frozen_manifest(path, [{**row, "context": "changed"}], model, 131072)


@pytest.mark.parametrize("name", ["prophetkv", "fuserag", "cacheblend"])
def test_unverified_baselines_fail_closed(name):
    with pytest.raises(ValueError):
        baseline_identity(name)


def test_diagnostic_variant_cannot_report_under_official_name():
    row = baseline_identity("prophetkv", allow_unverified=True)
    assert row["baseline"] == "prophetkv_unverified"
    assert row["comparison_eligible"] is False
    assert baseline_identity("fullcomp")["comparison_eligible"]


def test_known_ruler_fixtures_are_rejected():
    with pytest.raises(ValueError, match="placeholder"):
        validate_dataset_rows([{"answers": ["answer_ruler_vt_0"]}], "ruler_vt")
    validate_dataset_rows([{"input": "Find number", "answers": ["174"]}], "ruler_vt")
