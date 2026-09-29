from __future__ import annotations

import json
from pathlib import Path

import pytest

from multilingual_safety.agents.backends.base import BackendError
from multilingual_safety.agents.backends.cache import (
    ModelNotCachedError,
    ModelRevisionError,
    ModelSnapshotError,
    assert_pinned_revision,
    resolve_model_snapshot,
)

REVISION = "a" * 40
MODEL_ID = "test-org/test-model"


def make_snapshot(
    tmp_path: Path,
    *,
    config: dict | None = None,
    shards: tuple[str, ...] = (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ),
) -> Path:
    snapshot = (
        tmp_path
        / "models--test-org--test-model"
        / "snapshots"
        / REVISION
    )
    snapshot.mkdir(parents=True)

    payload = config or {
        "architectures": ["TestModelForCausalLM"],
        "num_hidden_layers": 4,
        "hidden_size": 256,
        "vocab_size": 1000,
        "torch_dtype": "bfloat16",
    }
    (snapshot / "config.json").write_text(json.dumps(payload), encoding="utf-8")

    weight_map = {
        f"layer.{index}.weight": shard
        for index, shard in enumerate(shards)
    }
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )

    for shard in shards:
        (snapshot / shard).touch()

    return snapshot


@pytest.mark.parametrize("revision", [None, "main", "abcdef", "A" * 40, "z" * 40])
def test_pinned_revision_rejects_non_sha(revision: str | None) -> None:
    with pytest.raises(ModelRevisionError):
        assert_pinned_revision(MODEL_ID, revision)


def test_resolves_complete_fake_sharded_snapshot(tmp_path: Path) -> None:
    snapshot = make_snapshot(tmp_path)

    resolved = resolve_model_snapshot(
        MODEL_ID,
        REVISION,
        cache_root=tmp_path,
        expected={
            "architecture": "TestModelForCausalLM",
            "num_hidden_layers": 4,
            "hidden_size": 256,
            "vocab_size": 1000,
        },
    )

    assert resolved.snapshot_dir == snapshot
    assert resolved.architecture == "TestModelForCausalLM"
    assert resolved.num_hidden_layers == 4
    assert resolved.hidden_size == 256
    assert resolved.vocab_size == 1000
    assert resolved.dtype == "bfloat16"
    assert resolved.n_shards == 2
    assert resolved.config_sha256
    assert resolved.index_sha256
    assert resolved.to_dict()["snapshot_dir"] == str(snapshot)


def test_missing_snapshot_fails_without_download(tmp_path: Path) -> None:
    with pytest.raises(ModelNotCachedError):
        resolve_model_snapshot(MODEL_ID, REVISION, cache_root=tmp_path)


def test_missing_declared_shard_fails(tmp_path: Path) -> None:
    snapshot = make_snapshot(tmp_path)
    (snapshot / "model-00002-of-00002.safetensors").unlink()

    with pytest.raises(ModelSnapshotError, match="missing declared weight shard"):
        resolve_model_snapshot(MODEL_ID, REVISION, cache_root=tmp_path)


def test_expected_metadata_mismatch_fails(tmp_path: Path) -> None:
    make_snapshot(tmp_path)

    with pytest.raises(ModelSnapshotError, match="expected hidden_size"):
        resolve_model_snapshot(
            MODEL_ID,
            REVISION,
            cache_root=tmp_path,
            expected={"hidden_size": 999},
        )
