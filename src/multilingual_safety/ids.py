"""Stable identifiers and content hashes.

Design constraint: identifiers must be derivable from metadata alone, so that a
manifest can be tracked in git while the prompt text stays out of it
(THIRD_PARTY_DATASETS.md). Content hashes let a collaborator *prove* they built
the same corpus without either party sending the text to the other.

Identifier shapes
-----------------
behavior  ``beh_<12 hex>``   stable across languages: one behavior, seven renderings
record    ``rec_<lang>_<12 hex>``  one behavior in one language
run       ``run_<UTC ts>_<8 hex>``  one execution of one resolved config
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

BEHAVIOR_ID_PREFIX = "beh_"
RECORD_ID_PREFIX = "rec_"
RUN_ID_PREFIX = "run_"

_BEHAVIOR_HEX = 12
_RECORD_HEX = 12
_RUN_HEX = 8


def content_hash(text: str, *, normalize_form: str = "NFC") -> str:
    """SHA-256 of Unicode-normalised text, as full-length hex.

    Normalising before hashing means the same Yoruba string hashes identically
    whether it arrived NFC or NFD -- otherwise a round-trip through a tool that
    decomposes combining marks would look like corpus corruption. The
    normalisation form used is part of the contract and is recorded in
    manifests alongside the hash.
    """
    normalized = unicodedata.normalize(normalize_form, text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace, UTF-8 kept."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable_hash(payload: Mapping[str, Any]) -> str:
    """SHA-256 over a canonical JSON encoding of ``payload``."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def behavior_id(*, source_id: str, source_item_id: str, arm: str) -> str:
    """Stable ID for a behavior, shared by all seven language renderings.

    Derived from provenance only -- never from prompt text -- so the ID is safe
    to commit and remains valid if the text is re-fetched.

    ``arm`` is part of the key because harmful and benign items may share a
    source item id across upstream splits, and collapsing them would silently
    merge the two arms.
    """
    if arm not in {"harmful", "benign"}:
        raise ValueError(f"arm must be 'harmful' or 'benign', got {arm!r}")
    for name, value in (("source_id", source_id), ("source_item_id", source_item_id)):
        if not value or not str(value).strip():
            raise ValueError(f"{name} must be a non-empty string")
    digest = stable_hash(
        {"source_id": source_id, "source_item_id": str(source_item_id), "arm": arm}
    )
    return f"{BEHAVIOR_ID_PREFIX}{digest[:_BEHAVIOR_HEX]}"


def record_id(*, behavior: str, language: str, variant: str = "base") -> str:
    """Stable ID for one behavior rendered in one language.

    ``variant`` distinguishes alternative renderings of the same behavior in the
    same language (e.g. a re-translation, a dialectal form). Variants are
    additional records, never in-place edits: overwriting a rendering would
    silently change what a prior run measured.
    """
    if not behavior.startswith(BEHAVIOR_ID_PREFIX):
        raise ValueError(f"expected a behavior id starting with {BEHAVIOR_ID_PREFIX!r}, got {behavior!r}")
    digest = stable_hash({"behavior": behavior, "language": language, "variant": variant})
    return f"{RECORD_ID_PREFIX}{language}_{digest[:_RECORD_HEX]}"


def utc_timestamp(now: datetime | None = None) -> str:
    """Compact UTC timestamp, ``YYYYmmddTHHMMSSZ``."""
    moment = now.astimezone(timezone.utc) if now else datetime.now(timezone.utc)
    return moment.strftime("%Y%m%dT%H%M%SZ")


def run_id(
    *,
    config_hash: str,
    code_hash: str = "",
    now: datetime | None = None,
) -> str:
    """Mint a run ID: ``run_<UTC timestamp>_<8 hex of config+code>``.

    Two runs of the identical configuration produce *different* run IDs by
    design. Reruns never overwrite; the difference between them is data (it
    measures nondeterminism), not a collision to be avoided.
    """
    if not config_hash:
        raise ValueError("config_hash is required; an unhashed run is untraceable")
    digest = stable_hash({"config_hash": config_hash, "code_hash": code_hash})
    return f"{RUN_ID_PREFIX}{utc_timestamp(now)}_{digest[:_RUN_HEX]}"


def is_behavior_id(value: str) -> bool:
    return _matches(value, BEHAVIOR_ID_PREFIX, _BEHAVIOR_HEX)


def is_run_id(value: str) -> bool:
    """True for a well-formed run ID."""
    if not value.startswith(RUN_ID_PREFIX):
        return False
    rest = value[len(RUN_ID_PREFIX) :]
    parts = rest.split("_")
    if len(parts) != 2:
        return False
    stamp, digest = parts
    if len(stamp) != 16 or not stamp.endswith("Z") or stamp[8] != "T":
        return False
    return _is_hex(digest, _RUN_HEX)


def _matches(value: str, prefix: str, hex_len: int) -> bool:
    if not value.startswith(prefix):
        return False
    return _is_hex(value[len(prefix) :], hex_len)


def _is_hex(value: str, length: int) -> bool:
    if len(value) != length:
        return False
    return all(ch in "0123456789abcdef" for ch in value)
