"""Behavior manifest: the tracked, text-free description of the corpus.

A manifest row describes one behavior rendered in one language. It carries
identifiers, a content hash, provenance, and translation metadata -- and no
prompt text. This is what makes it safe to commit while the corpus itself stays
in git-ignored ``data/`` (README §7, THIRD_PARTY_DATASETS.md).

The manifest is the single source of truth for row counts. Every stage compares
its own output count against it, and a mismatch is a stop condition
(CLAUDE.md §10).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from multilingual_safety.ids import is_behavior_id
from multilingual_safety.languages import LANGUAGE_CODES, validate_language_code

ARMS: tuple[str, ...] = ("harmful", "benign")
SPLITS: tuple[str, ...] = ("dev", "test")
TRANSLATION_METHODS: tuple[str, ...] = (
    "original",
    "human",
    "professional_service",
    "machine_reviewed",
    "machine_unreviewed",
)
REVIEW_STATUSES: tuple[str, ...] = ("none", "spot_checked", "fully_reviewed")

#: Field names that must never appear in a manifest row. Enforced, not merely
#: documented -- a helpfully-added "prompt" column is the exact way harmful text
#: ends up in git.
FORBIDDEN_FIELDS: frozenset[str] = frozenset(
    {
        "prompt",
        "text",
        "prompt_text",
        "content",
        "body",
        "completion",
        "response",
        "output",
        "transcript",
        "plan",
        "rationale",
        "api_key",
        "token",
        "secret",
        "password",
    }
)


class ManifestError(ValueError):
    """Raised when a manifest row violates a structural or safety invariant."""


@dataclass(frozen=True, slots=True)
class BehaviorRecord:
    """One behavior in one language. Metadata only -- never prompt text."""

    record_id: str
    behavior_id: str
    language: str
    arm: str
    behavior_family: str
    source_id: str
    source_item_id: str
    content_sha256: str
    normalization: str = "NFC"
    variant: str = "base"
    translation_method: str = "original"
    translation_model: str | None = None
    review_status: str = "none"
    backtranslation_similarity: float | None = None
    char_length: int | None = None
    orthography: Mapping[str, Any] = field(default_factory=dict)
    license_id: str | None = None
    redistribution_allowed: bool | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not is_behavior_id(self.behavior_id):
            raise ManifestError(f"malformed behavior_id {self.behavior_id!r}")
        validate_language_code(self.language)
        if self.arm not in ARMS:
            raise ManifestError(f"arm must be one of {ARMS}, got {self.arm!r}")
        if self.translation_method not in TRANSLATION_METHODS:
            raise ManifestError(f"unknown translation_method {self.translation_method!r}")
        if self.review_status not in REVIEW_STATUSES:
            raise ManifestError(f"unknown review_status {self.review_status!r}")
        if len(self.content_sha256) != 64 or not all(
            c in "0123456789abcdef" for c in self.content_sha256
        ):
            raise ManifestError(f"content_sha256 must be 64 lowercase hex chars, got {self.content_sha256!r}")
        if self.language != "en" and self.translation_method == "original":
            # Not fatal for `en`-sourced corpora only; flagged because unrecorded
            # translation provenance makes a cross-lingual comparison
            # uninterpretable (CLAUDE.md §6).
            raise ManifestError(
                f"record {self.record_id}: non-pivot language {self.language!r} declares "
                "translation_method='original'; record real provenance instead"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assert_no_forbidden_fields(row: Mapping[str, Any]) -> None:
    """Raise if a row carries a field that could smuggle text or secrets."""
    present = sorted(FORBIDDEN_FIELDS.intersection({k.lower() for k in row}))
    if present:
        raise ManifestError(
            f"manifest row contains forbidden field(s) {present}. Manifests hold IDs, "
            "hashes, and metadata only -- never prompt text or credentials "
            "(THIRD_PARTY_DATASETS.md)."
        )


def write_manifest(records: Iterable[BehaviorRecord], path: str | Path) -> int:
    """Write records as JSONL. Returns the row count written.

    Refuses to write zero rows: an empty manifest is a degenerate output and a
    stop condition (CLAUDE.md §10).
    """
    path = Path(path)
    rows = [r.to_dict() for r in records]
    if not rows:
        raise ManifestError("refusing to write an empty manifest (stop condition)")
    seen: set[str] = set()
    for row in rows:
        assert_no_forbidden_fields(row)
        rid = row["record_id"]
        if rid in seen:
            raise ManifestError(f"duplicate record_id {rid!r} (stop condition)")
        seen.add(rid)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return len(rows)


def read_manifest(path: str | Path) -> list[BehaviorRecord]:
    """Read and validate a JSONL manifest.

    A malformed line raises rather than being skipped: skipping bad rows is the
    silent-discard failure this project explicitly forbids (CLAUDE.md §2).
    """
    return list(iter_manifest(path))


def iter_manifest(path: str | Path) -> Iterator[BehaviorRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{lineno}: malformed JSON: {exc}") from exc
            assert_no_forbidden_fields(row)
            try:
                yield BehaviorRecord(**row)
            except TypeError as exc:
                raise ManifestError(f"{path}:{lineno}: unexpected or missing fields: {exc}") from exc


def summarize(records: Iterable[BehaviorRecord]) -> dict[str, Any]:
    """Row counts per language and per arm, plus coverage checks.

    This is the payload of the post-stage summary required by CLAUDE.md §9.
    """
    records = list(records)
    per_language: dict[str, int] = dict.fromkeys(LANGUAGE_CODES, 0)
    per_arm: dict[str, int] = dict.fromkeys(ARMS, 0)
    behaviors: dict[str, set[str]] = {}

    for rec in records:
        per_language[rec.language] += 1
        per_arm[rec.arm] += 1
        behaviors.setdefault(rec.behavior_id, set()).add(rec.language)

    incomplete = {
        bid: sorted(set(LANGUAGE_CODES) - langs)
        for bid, langs in behaviors.items()
        if langs != set(LANGUAGE_CODES)
    }
    return {
        "n_rows": len(records),
        "n_behaviors": len(behaviors),
        "rows_per_language": per_language,
        "rows_per_arm": per_arm,
        "n_behaviors_incomplete": len(incomplete),
        # Behaviors missing a language break the paired cross-lingual design;
        # they are surfaced, never dropped.
        "incomplete_behaviors": dict(sorted(incomplete.items())[:20]),
        "languages_with_zero_rows": [c for c, n in per_language.items() if n == 0],
        "arms_with_zero_rows": [a for a, n in per_arm.items() if n == 0],
    }
