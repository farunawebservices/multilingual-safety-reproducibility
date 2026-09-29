"""Offline resolution and validation of pinned Hugging Face model snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .base import BackendError

_SHA40 = re.compile(r"^[0-9a-f]{40}$")


class ModelRevisionError(BackendError):
    """Raised when a model revision is not an immutable full commit SHA."""


class ModelNotCachedError(BackendError):
    """Raised when a pinned model snapshot is absent from the local cache."""


class ModelSnapshotError(BackendError):
    """Raised when cached snapshot metadata or shards are invalid."""


def assert_pinned_revision(model_id: str, revision: str | None) -> str:
    """Require an explicit lowercase 40-hex Hugging Face commit SHA."""
    if not isinstance(revision, str) or not _SHA40.fullmatch(revision):
        raise ModelRevisionError(
            f"{model_id}: revision must be a lowercase 40-hex commit SHA; "
            f"got {revision!r}"
        )
    return revision


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """Text-free metadata describing a locally cached, pinned model snapshot."""

    model_id: str
    revision: str
    snapshot_dir: Path
    architecture: str
    num_hidden_layers: int
    hidden_size: int
    vocab_size: int
    dtype: str | None
    index_sha256: str | None
    config_sha256: str
    n_shards: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["snapshot_dir"] = str(self.snapshot_dir)
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_root() -> Path:
    """Return the local Hugging Face hub cache path without contacting a server."""
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _snapshot_dir(model_id: str, revision: str, cache_root: Path) -> Path:
    repo_dir = "models--" + model_id.replace("/", "--")
    return cache_root / repo_dir / "snapshots" / revision


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelSnapshotError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelSnapshotError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ModelSnapshotError(f"{label} must contain a JSON object: {path}")
    return payload


def _positive_int(config: Mapping[str, Any], key: str, model_id: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelSnapshotError(f"{model_id}: config.json has invalid {key!r}: {value!r}")
    return value


def _validate_expected(
    model_id: str,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any] | None,
) -> None:
    if not expected:
        return
    for key in ("architecture", "num_hidden_layers", "hidden_size", "vocab_size"):
        if key in expected and expected[key] != actual[key]:
            raise ModelSnapshotError(
                f"{model_id}: expected {key}={expected[key]!r}, got {actual[key]!r}"
            )


def _weight_metadata(snapshot_dir: Path, model_id: str) -> tuple[str | None, int]:
    """Validate all declared safetensors shards, or a single supported weight file."""
    index = snapshot_dir / "model.safetensors.index.json"
    if index.is_file():
        payload = _read_json(index, "model.safetensors.index.json")
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ModelSnapshotError(f"{model_id}: safetensors index has no weight_map")
        shard_names = set(weight_map.values())
        if not all(isinstance(name, str) and name for name in shard_names):
            raise ModelSnapshotError(f"{model_id}: safetensors index contains invalid shard names")
        missing = sorted(name for name in shard_names if not (snapshot_dir / name).is_file())
        if missing:
            raise ModelSnapshotError(
                f"{model_id}: snapshot is missing declared weight shard(s): {', '.join(missing)}"
            )
        return _sha256(index), len(shard_names)

    single = snapshot_dir / "model.safetensors"
    if single.is_file():
        return None, 1

    raise ModelSnapshotError(
        f"{model_id}: no model.safetensors.index.json or model.safetensors in {snapshot_dir}"
    )


def resolve_model_snapshot(
    model_id: str,
    revision: str | None,
    *,
    expected: Mapping[str, Any] | None = None,
    cache_root: Path | None = None,
) -> ResolvedModel:
    """Resolve a pinned local snapshot only; never downloads or loads weights."""
    revision = assert_pinned_revision(model_id, revision)
    root = cache_root if cache_root is not None else _cache_root()
    snapshot_dir = _snapshot_dir(model_id, revision, root)

    if not snapshot_dir.is_dir():
        raise ModelNotCachedError(
            f"{model_id}: pinned snapshot is not cached locally: {snapshot_dir}"
        )

    config_path = snapshot_dir / "config.json"
    config = _read_json(config_path, "config.json")

    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures or not isinstance(architectures[0], str):
        raise ModelSnapshotError(f"{model_id}: config.json has no usable architectures[0]")

    actual = {
        "architecture": architectures[0],
        "num_hidden_layers": _positive_int(config, "num_hidden_layers", model_id),
        "hidden_size": _positive_int(config, "hidden_size", model_id),
        "vocab_size": _positive_int(config, "vocab_size", model_id),
    }
    _validate_expected(model_id, actual, expected)
    index_sha256, n_shards = _weight_metadata(snapshot_dir, model_id)

    dtype = config.get("torch_dtype")
    if dtype is not None and not isinstance(dtype, str):
        raise ModelSnapshotError(f"{model_id}: config.json has invalid torch_dtype")

    return ResolvedModel(
        model_id=model_id,
        revision=revision,
        snapshot_dir=snapshot_dir,
        architecture=actual["architecture"],
        num_hidden_layers=actual["num_hidden_layers"],
        hidden_size=actual["hidden_size"],
        vocab_size=actual["vocab_size"],
        dtype=dtype,
        index_sha256=index_sha256,
        config_sha256=_sha256(config_path),
        n_shards=n_shards,
    )
