"""Local language identification (LID) as a *diagnostic*, not validation.

Backend: **GlotLID** (``cis-lmu/glotlid``), a fastText classifier covering ~2100
language labels including Hausa, Yoruba and Swahili. Apache-2.0, runs locally on
CPU, no API and no credential. It is independent of every translation engine,
which is the whole point: a translator grading its own output is not evidence.

What LID can and cannot do
--------------------------
LID answers *"what language is this text?"* It does **not** answer *"is this a
faithful translation?"* It cannot detect a fluent Yoruba sentence that means
something different from the English source, and it cannot detect added
operational detail -- which is the constraint that actually matters for this
corpus.

Therefore:

* LID **never** clears ``review_required``. :data:`ALWAYS_HUMAN_REVIEW` names
  ``ha``, ``yo`` and ``sw`` explicitly, and
  :func:`requires_human_review` returns True for *every* non-English language
  regardless of LID outcome.
* A LID **disagreement** is a strong negative signal (wrong language returned).
  A LID **agreement** is a weak positive signal (right language, unknown
  meaning). The asymmetry is deliberate and is recorded in the report.
* If the model is unavailable, LID is reported as ``unavailable`` and nothing is
  inferred. Absence of a check is never recorded as a passed check.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE, validate_language_code

#: Hugging Face repo and file for GlotLID.
GLOTLID_REPO = "cis-lmu/glotlid"
GLOTLID_FILE = "model.bin"
GLOTLID_LICENSE = "Apache-2.0"
GLOTLID_URL = "https://huggingface.co/cis-lmu/glotlid"

#: GlotLID emits ``<iso639-3>_<Script>`` labels. Map ours onto them.
#: Verified against the model's own label set at load time (never assumed).
EXPECTED_LID_LABELS: dict[str, tuple[str, ...]] = {
    "en": ("eng_Latn",),
    "ar": ("arb_Arab", "ara_Arab"),
    "ha": ("hau_Latn",),
    "yo": ("yor_Latn",),
    "sw": ("swh_Latn", "swa_Latn"),
    "hi": ("hin_Deva",),
    "ko": ("kor_Hang",),
}

#: Languages that stay under human review no matter what LID says. These are the
#: lower-resource targets where both LID and our heuristics are least reliable.
ALWAYS_HUMAN_REVIEW: frozenset[str] = frozenset({"ha", "yo", "sw"})

#: Below this probability the identification is treated as inconclusive.
MIN_CONFIDENCE = 0.50

STATUS_MATCH = "MATCH"
STATUS_MISMATCH = "MISMATCH"
STATUS_INCONCLUSIVE = "INCONCLUSIVE"
STATUS_UNAVAILABLE = "UNAVAILABLE"


class LIDUnavailableError(RuntimeError):
    """The LID model could not be loaded. Nothing is inferred from its absence."""


@dataclass(frozen=True, slots=True)
class LIDResult:
    """One LID observation. A diagnostic record, never a verdict."""

    text_sha256: str
    expected_language: str
    predicted_label: str | None
    predicted_confidence: float | None
    status: str
    model: str = GLOTLID_REPO
    model_revision: str | None = None
    alternatives: Sequence[tuple[str, float]] = field(default_factory=tuple)

    @property
    def is_mismatch(self) -> bool:
        return self.status == STATUS_MISMATCH

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_sha256": self.text_sha256,
            "expected_language": self.expected_language,
            "predicted_label": self.predicted_label,
            "predicted_confidence": self.predicted_confidence,
            "status": self.status,
            "model": self.model,
            "model_revision": self.model_revision,
            "alternatives": [list(a) for a in self.alternatives],
            "is_adequacy_validation": False,
            "clears_human_review": False,
            "note": (
                "Language identity only. A MATCH means the text is in the expected "
                "language, NOT that it is a faithful translation. This never clears "
                "review_required."
            ),
        }


def requires_human_review(language: str, lid: LIDResult | None = None) -> bool:
    """Always True for non-English rows. LID cannot change this.

    The ``lid`` argument is accepted and ignored on purpose: making the
    signature look like it could matter, and then not letting it, is clearer at
    the call site than omitting it and inviting someone to add it later.
    """
    validate_language_code(language)
    return language != PIVOT_LANGUAGE


class GlotLID:
    """Lazy wrapper around the GlotLID fastText model."""

    def __init__(self, *, repo: str = GLOTLID_REPO, filename: str = GLOTLID_FILE,
                 revision: str | None = None, local_files_only: bool = False) -> None:
        self.repo = repo
        self.filename = filename
        self.revision = revision
        self.local_files_only = local_files_only
        self._model = None
        self._labels: frozenset[str] = frozenset()
        self._path: str | None = None

    def load(self) -> GlotLID:
        """Load the model. Raises :class:`LIDUnavailableError` if unavailable."""
        try:
            import fasttext
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise LIDUnavailableError(
                f"GlotLID needs fasttext + huggingface_hub ({exc}). "
                "Install with: python -m pip install -e '.[translate]'"
            ) from exc
        try:
            self._path = hf_hub_download(
                repo_id=self.repo,
                filename=self.filename,
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
            self._model = fasttext.load_model(self._path)
        except Exception as exc:
            raise LIDUnavailableError(
                f"could not load GlotLID from {self.repo!r}: {type(exc).__name__}: {exc}"
            ) from exc
        self._labels = frozenset(
            label.removeprefix("__label__") for label in self._model.get_labels()
        )
        return self

    @property
    def available(self) -> bool:
        return self._model is not None

    def known_labels_for(self, language: str) -> tuple[str, ...]:
        """Expected labels for ``language`` that this model actually knows."""
        return tuple(
            label for label in EXPECTED_LID_LABELS.get(language, ()) if label in self._labels
        )

    def unsupported_languages(self) -> list[str]:
        """Target languages this model has no label for. Reported, not assumed."""
        if not self.available:
            return list(LANGUAGE_CODES)
        return [code for code in LANGUAGE_CODES if not self.known_labels_for(code)]

    def identify(self, text: str, *, expected_language: str, k: int = 3) -> LIDResult:
        """Identify ``text`` and compare against ``expected_language``."""
        from multilingual_safety.hashing import text_sha256

        validate_language_code(expected_language)
        digest = text_sha256(text)

        if not self.available:
            return LIDResult(digest, expected_language, None, None, STATUS_UNAVAILABLE)

        # fastText rejects newlines; normalise without altering orthography.
        cleaned = " ".join(unicodedata.normalize("NFC", text).split())
        if not cleaned:
            return LIDResult(digest, expected_language, None, None, STATUS_INCONCLUSIVE)

        labels, scores = self._model.predict(cleaned, k=k)
        ranked = [
            (label.removeprefix("__label__"), float(score))
            for label, score in zip(labels, scores, strict=False)
        ]
        top_label, top_score = ranked[0]
        accepted = self.known_labels_for(expected_language)

        if not accepted:
            # No label exists for this language: cannot conclude anything.
            status = STATUS_INCONCLUSIVE
        elif top_score < MIN_CONFIDENCE:
            status = STATUS_INCONCLUSIVE
        elif top_label in accepted:
            status = STATUS_MATCH
        else:
            status = STATUS_MISMATCH

        return LIDResult(
            text_sha256=digest,
            expected_language=expected_language,
            predicted_label=top_label,
            predicted_confidence=round(top_score, 6),
            status=status,
            model_revision=self.revision,
            alternatives=tuple(ranked[1:]),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.repo,
            "file": self.filename,
            "revision": self.revision,
            "license": GLOTLID_LICENSE,
            "url": GLOTLID_URL,
            "local_path_present": self._path is not None,
            "available": self.available,
            "n_labels": len(self._labels),
            "unsupported_target_languages": self.unsupported_languages(),
            "role": "language-identity diagnostic only",
            "is_adequacy_validation": False,
            "always_human_review": sorted(ALWAYS_HUMAN_REVIEW),
        }


def summarize(results: Mapping[str, LIDResult]) -> dict[str, Any]:
    """Aggregate LID outcomes. Never presents agreement as validation."""
    by_status: dict[str, int] = {
        s: 0 for s in (STATUS_MATCH, STATUS_MISMATCH, STATUS_INCONCLUSIVE, STATUS_UNAVAILABLE)
    }
    for result in results.values():
        by_status[result.status] = by_status.get(result.status, 0) + 1
    return {
        "n_checked": len(results),
        "by_status": dict(sorted(by_status.items())),
        "mismatched_request_ids": sorted(k for k, v in results.items() if v.is_mismatch),
        "interpretation": (
            "MISMATCH is strong evidence the wrong language was returned. MATCH is weak "
            "evidence only: it confirms language identity, not translation fidelity. No "
            "status here clears review_required for any row."
        ),
        "rows_still_requiring_human_review": len(results),
    }
