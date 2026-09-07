import json

import pytest
import torch

from ucm.store.qcfuse import packed_format as pf


def test_v4_round_trip_records_raw_chunk_local_keys(tmp_path):
    key = torch.arange(12, dtype=torch.float32).view(3, 4)
    value = key + 100
    pf.save_packed_kv(str(tmp_path), {0: (key, value)}, 1)

    meta = pf.load_packed_meta(str(tmp_path), expected_num_layers=1)
    assert meta["format"] == pf.PACKED_KV_FORMAT
    assert meta["key_encoding"] == pf.KEY_ENCODING
    assert meta["offline_attention_layout"] == pf.OFFLINE_ATTENTION_LAYOUT
    loaded = pf.read_tensor_bytes(
        str(tmp_path / pf._PACKED_BIN_NAME), pf._tensor_meta(meta, 0, "k")
    )
    torch.testing.assert_close(loaded, key)


def test_rejects_legacy_post_rope_cache(tmp_path):
    (tmp_path / pf._PACKED_BIN_NAME).write_bytes(b"")
    (tmp_path / pf._PACKED_META_NAME).write_text(
        json.dumps({"format": "sgblend_kv_packed_v1", "num_layers": 0, "layers": {}})
    )

    with pytest.raises(ValueError, match="regenerate"):
        pf.load_packed_meta(str(tmp_path))
