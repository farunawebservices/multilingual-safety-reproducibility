"""Translation pilot row schema, statuses, and backend resolution.

One row = one behavior in one language. The schema is deliberately strict about
three things:

1. **``behavior_id`` is invariant across languages.** All seven rows for a
   behavior carry the same ID. Enforced in :func:`validate_row_group`.
2. **``review_required`` cannot be cleared by a machine.** Only
   ``HUMAN_REVIEWED`` status clears it, and only a real reviewer sets that.
   Heuristics passing is not review (:mod:`multilingual_safety.langcheck`).
3. **``split_placeholder`` is a placeholder.** It holds a sentinel, not a split.
   Assignment happens later, at behavior level, so no behavior can land in both
   development and final evaluation.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from multilingual_safety.hashing import text_sha256
from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE, validate_language_code

#: Sentinel for `split_placeholder`. Never a real split name.
SPLIT_UNASSIGNED = "UNASSIGNED"

HARM_LABELS: tuple[str, ...] = ("harmful", "benign")


class TranslationStatus(str, Enum):
    """Lifecycle of one translated row. No default, no silent transitions."""

    #: English: canonical source text, used verbatim. Not a translation.
    CANONICAL_SOURCE = "CANONICAL_SOURCE"
    #: Requested but not yet produced (e.g. no translation backend configured).
    PENDING_TRANSLATION = "PENDING_TRANSLATION"
    #: Produced by a machine, not yet checked.
    MACHINE_TRANSLATED_UNCHECKED = "MACHINE_TRANSLATED_UNCHECKED"
    #: Machine output that passed the heuristics. STILL NOT VALIDATED.
    HEURISTICS_PASSED = "HEURISTICS_PASSED"
    #: Machine output that failed at least one heuristic.
    HEURISTICS_FAILED = "HEURISTICS_FAILED"
    #: A named human reviewer accepted this row. The only status that is
    #: evidence of correctness, and the only one that clears review_required.
    HUMAN_REVIEWED = "HUMAN_REVIEWED"
    #: A human or a hard failure rejected this row.
    REJECTED = "REJECTED"
    #: The backend errored.
    FAILED = "FAILED"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: Statuses that may carry usable text downstream.
USABLE_STATUSES: frozenset[TranslationStatus] = frozenset(
    {TranslationStatus.CANONICAL_SOURCE, TranslationStatus.HUMAN_REVIEWED}
)

#: The only status permitted to set review_required=False for a non-pivot row.
CLEARS_REVIEW: frozenset[TranslationStatus] = frozenset({TranslationStatus.HUMAN_REVIEWED})


class TranslationMethod(str, Enum):
    """How the text for this row was produced."""

    #: English pivot: the upstream text, unmodified.
    ORIGINAL = "original"
    #: No translation has been produced yet.
    PENDING = "pending"
    MACHINE_UNREVIEWED = "machine_unreviewed"
    MACHINE_REVIEWED = "machine_reviewed"
    HUMAN = "human"
    PROFESSIONAL_SERVICE = "professional_service"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


REQUIRED_FIELDS: tuple[str, ...] = (
    "behavior_id",
    "source",
    "category",
    "harm_label",
    "source_text",
    "language",
    "translated_text",
    "translation_method",
    "translation_model",
    "translation_timestamp",
    "translation_status",
    "review_required",
    "prompt_hash",
    "split_placeholder",
)

#: Fields that carry prompt text. Stripped before anything is written to a
#: tracked manifest (THIRD_PARTY_DATASETS.md).
TEXT_FIELDS: frozenset[str] = frozenset({"source_text", "translated_text"})


class TranslationSchemaError(ValueError):
    """Raised when a pilot row violates a schema or policy invariant."""


def compute_prompt_hash(text: str | None) -> str | None:
    """SHA-256 of the NFC-normalised prompt for this row, or None if absent."""
    if text is None or not text.strip():
        return None
    return text_sha256(text)


@dataclass(frozen=True, slots=True)
class TranslationRow:
    """One behavior in one language."""

    behavior_id: str
    source: str
    category: str
    harm_label: str
    source_text: str
    language: str
    translated_text: str | None
    translation_method: str
    translation_model: str | None
    translation_timestamp: str | None
    translation_status: str
    review_required: bool
    prompt_hash: str | None
    split_placeholder: str = SPLIT_UNASSIGNED
    # --- provenance beyond the required minimum ---------------------------
    source_item_id: str | None = None
    reviewer_id: str | None = None
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_language_code(self.language)
        if self.harm_label not in HARM_LABELS:
            raise TranslationSchemaError(
                f"harm_label must be one of {HARM_LABELS}, got {self.harm_label!r}"
            )
        try:
            status = TranslationStatus(self.translation_status)
        except ValueError as exc:
            raise TranslationSchemaError(
                f"unknown translation_status {self.translation_status!r}"
            ) from exc
        try:
            TranslationMethod(self.translation_method)
        except ValueError as exc:
            raise TranslationSchemaError(
                f"unknown translation_method {self.translation_method!r}"
            ) from exc
        if self.split_placeholder != SPLIT_UNASSIGNED:
            raise TranslationSchemaError(
                f"split_placeholder must be {SPLIT_UNASSIGNED!r} at the pilot stage, got "
                f"{self.split_placeholder!r}. Splits are assigned later, at behavior level."
            )
        if not self.source_text.strip():
            raise TranslationSchemaError(f"{self.behavior_id}/{self.language}: empty source_text")

        # --- pivot-language invariants ------------------------------------
        if self.language == PIVOT_LANGUAGE:
            if status is not TranslationStatus.CANONICAL_SOURCE:
                raise TranslationSchemaError(
                    f"English rows must be {TranslationStatus.CANONICAL_SOURCE.value}, "
                    f"got {self.translation_status}. English is not translated."
                )
            if self.translation_method != TranslationMethod.ORIGINAL.value:
                raise TranslationSchemaError(
                    f"English rows must use method 'original', got {self.translation_method!r}"
                )
            if self.translated_text != self.source_text:
                raise TranslationSchemaError(
                    "English rows must carry the canonical source text verbatim"
                )
        else:
            if status is TranslationStatus.CANONICAL_SOURCE:
                raise TranslationSchemaError(
                    f"{self.language!r} is not the pivot language; it cannot be CANONICAL_SOURCE"
                )
            if self.translation_method == TranslationMethod.ORIGINAL.value:
                raise TranslationSchemaError(
                    f"{self.language!r} cannot declare method 'original'; record real provenance"
                )
            # review_required may only be cleared by an actual human review.
            if not self.review_required and status not in CLEARS_REVIEW:
                raise TranslationSchemaError(
                    f"{self.behavior_id}/{self.language}: review_required=False requires status "
                    f"{TranslationStatus.HUMAN_REVIEWED.value}, got {self.translation_status}. "
                    "Heuristics passing is not review, and no machine may clear this flag."
                )
            if status is TranslationStatus.HUMAN_REVIEWED and not self.reviewer_id:
                raise TranslationSchemaError(
                    f"{self.behavior_id}/{self.language}: HUMAN_REVIEWED requires a reviewer_id. "
                    "Native-speaker validation is not claimed without a named reviewer."
                )

        # --- text/hash consistency ----------------------------------------
        expected_hash = compute_prompt_hash(self.translated_text)
        if self.prompt_hash != expected_hash:
            raise TranslationSchemaError(
                f"{self.behavior_id}/{self.language}: prompt_hash does not match translated_text"
            )
        if status is TranslationStatus.PENDING_TRANSLATION and self.translated_text is not None:
            raise TranslationSchemaError(
                f"{self.behavior_id}/{self.language}: PENDING_TRANSLATION must have "
                "translated_text=null. Do not invent text for a pending row."
            )

    # --- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = list(self.issues)
        return payload

    def to_manifest_dict(self) -> dict[str, Any]:
        """Text-free projection, safe for a tracked manifest.

        Every field carrying prompt text is dropped and replaced by a hash and a
        length, so the manifest still supports verification and row counting.
        """
        payload = {k: v for k, v in self.to_dict().items() if k not in TEXT_FIELDS}
        payload["source_text_sha256"] = text_sha256(self.source_text)
        payload["source_text_n_chars"] = len(self.source_text)
        payload["translated_text_n_chars"] = (
            None if self.translated_text is None else len(self.translated_text)
        )
        return dict(sorted(payload.items()))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TranslationRow:
        missing = [f for f in REQUIRED_FIELDS if f not in payload]
        if missing:
            raise TranslationSchemaError(f"row is missing required field(s): {missing}")
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unexpected = sorted(set(payload) - known)
        if unexpected:
            raise TranslationSchemaError(f"row has unexpected field(s): {unexpected}")
        data = dict(payload)
        data["issues"] = tuple(data.get("issues") or ())
        return cls(**data)


def make_pending_row(
    *,
    behavior_id: str,
    source: str,
    category: str,
    harm_label: str,
    source_text: str,
    language: str,
    source_item_id: str | None = None,
) -> TranslationRow:
    """A row awaiting translation. Carries no invented text."""
    if language == PIVOT_LANGUAGE:
        raise TranslationSchemaError("English is never pending; it is canonical source text")
    return TranslationRow(
        behavior_id=behavior_id,
        source=source,
        category=category,
        harm_label=harm_label,
        source_text=source_text,
        language=language,
        translated_text=None,
        translation_method=TranslationMethod.PENDING.value,
        translation_model=None,
        translation_timestamp=None,
        translation_status=TranslationStatus.PENDING_TRANSLATION.value,
        review_required=True,
        prompt_hash=None,
        source_item_id=source_item_id,
        issues=(),
    )


def make_english_row(
    *,
    behavior_id: str,
    source: str,
    category: str,
    harm_label: str,
    source_text: str,
    source_item_id: str | None = None,
) -> TranslationRow:
    """The English pivot row: canonical source text, verbatim, not translated."""
    text = unicodedata.normalize("NFC", source_text)
    return TranslationRow(
        behavior_id=behavior_id,
        source=source,
        category=category,
        harm_label=harm_label,
        source_text=text,
        language=PIVOT_LANGUAGE,
        translated_text=text,
        translation_method=TranslationMethod.ORIGINAL.value,
        translation_model=None,
        translation_timestamp=None,
        translation_status=TranslationStatus.CANONICAL_SOURCE.value,
        review_required=False,
        prompt_hash=compute_prompt_hash(text),
        source_item_id=source_item_id,
        issues=(),
    )


# --- group-level invariants ------------------------------------------------
def validate_row_group(rows: Sequence[TranslationRow]) -> list[str]:
    """Invariants across all rows sharing a behavior_id."""
    if not rows:
        return ["empty row group"]
    violations: list[str] = []

    ids = {r.behavior_id for r in rows}
    if len(ids) != 1:
        violations.append(f"behavior_id is not invariant across languages: {sorted(ids)}")

    for field_name in ("harm_label", "category", "source", "source_text"):
        values = {getattr(r, field_name) for r in rows}
        if len(values) != 1:
            label = "source_text" if field_name == "source_text" else field_name
            violations.append(f"{label} differs across languages within one behavior")

    languages = [r.language for r in rows]
    if len(languages) != len(set(languages)):
        violations.append("duplicate language rows within one behavior")

    missing = sorted(set(LANGUAGE_CODES) - set(languages))
    if missing:
        violations.append(f"missing language rows: {missing}")

    if PIVOT_LANGUAGE not in languages:
        violations.append("no English pivot row")

    splits = {r.split_placeholder for r in rows}
    if splits != {SPLIT_UNASSIGNED}:
        violations.append(f"split_placeholder is not uniformly unassigned: {sorted(splits)}")

    return violations


def find_duplicate_prompt_hashes(rows: Iterable[TranslationRow]) -> dict[str, list[str]]:
    """Map ``language:prompt_hash -> behavior_ids`` for hashes used more than once.

    Two different behaviors that translate to identical text in one language is
    a real defect: it collapses two conditions into one.
    """
    seen: dict[str, list[str]] = {}
    for row in rows:
        if row.prompt_hash is None:
            continue
        seen.setdefault(f"{row.language}:{row.prompt_hash}", []).append(row.behavior_id)
    return {k: sorted(v) for k, v in sorted(seen.items()) if len(set(v)) > 1}
