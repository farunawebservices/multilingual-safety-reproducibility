"""Human review decisions, and the competence gate that governs them.

The central rule of this module: **a review decision and the clearing of
``review_required`` are two different things.**

A reviewer can record ACCEPT on a row without that acceptance constituting
native-speaker validation. Whether a decision may clear ``review_required``
depends on the reviewer's *language competence*, not on their diligence or on
the decision itself:

===========================  =====================
competence                   may clear review?
===========================  =====================
``native_speaker``           yes
``professional_translator``  yes
``fluent_non_native``        no
``machine_assisted``         **no**
``none``                     no
===========================  =====================

``machine_assisted`` means the reviewer worked with machine output as an aid.
That is real, useful review -- it catches wrong-language output, truncation, and
obvious nonsense -- but it is not evidence that a Yoruba sentence carries the
same meaning as its English source. Recording it as native-speaker validation
would put an unearned claim into the provenance of every downstream result
(CLAUDE.md: no native-speaker claim without a native speaker).

So decisions from a non-clearing competence are recorded as a **separate review
layer**: the decision is preserved in full, and ``review_required`` stays true
until someone with clearing competence signs off.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from multilingual_safety.languages import PIVOT_LANGUAGE, validate_language_code


class ReviewerCompetence(str, Enum):
    """Reviewer's competence in the target language."""

    NATIVE_SPEAKER = "native_speaker"
    PROFESSIONAL_TRANSLATOR = "professional_translator"
    FLUENT_NON_NATIVE = "fluent_non_native"
    MACHINE_ASSISTED = "machine_assisted"
    NONE = "none"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class ReviewDecisionType(str, Enum):
    """What the reviewer concluded about the row."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    REVISE = "REVISE"
    #: Reviewer could not decide -- distinct from PENDING (not yet looked at).
    #: Never clears review, and is reported separately so "unsure" is not
    #: quietly absorbed into either acceptance or rejection.
    UNCERTAIN = "UNCERTAIN"
    PENDING = "PENDING"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: Only these competences may clear ``review_required``. Deliberately narrow.
COMPETENCE_CAN_CLEAR_REVIEW: frozenset[ReviewerCompetence] = frozenset(
    {ReviewerCompetence.NATIVE_SPEAKER, ReviewerCompetence.PROFESSIONAL_TRANSLATOR}
)

#: Only an ACCEPT could ever clear review, and then only with the competence
#: above. A REJECT or REVISE keeps the row under review by construction.
DECISION_CAN_CLEAR_REVIEW: frozenset[ReviewDecisionType] = frozenset(
    {ReviewDecisionType.ACCEPT}
)


class ReviewError(ValueError):
    """Raised when a review decision violates a schema or policy invariant."""


def can_clear_human_review(
    competence: ReviewerCompetence | str,
    decision: ReviewDecisionType | str = ReviewDecisionType.ACCEPT,
) -> bool:
    """Whether this (competence, decision) pair may set ``HUMAN_REVIEWED``."""
    competence = ReviewerCompetence(competence)
    decision = ReviewDecisionType(decision)
    return competence in COMPETENCE_CAN_CLEAR_REVIEW and decision in DECISION_CAN_CLEAR_REVIEW


def competence_rationale(competence: ReviewerCompetence | str) -> str:
    """Human-readable reason a competence does or does not clear review."""
    competence = ReviewerCompetence(competence)
    if competence in COMPETENCE_CAN_CLEAR_REVIEW:
        return f"{competence.value} is sufficient to clear review_required"
    return (
        f"{competence.value} is NOT sufficient to clear review_required: it does not "
        "establish that meaning was preserved, so the decision is recorded as a "
        "separate review layer and review_required stays true"
    )


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    """One reviewer decision about one translated row.

    Carries identifiers and hashes only -- never source or translated text -- so
    a collection of these is safe to summarise into a tracked report.
    """

    review_id: str
    request_id: str
    behavior_id: str
    language: str
    decision: str
    reviewer_id: str
    reviewer_language_competence: str
    #: SHA-256 of the text the reviewer actually saw, recomputed from the review
    #: sheet. Matching the recorded hash proves they reviewed *this* text.
    source_text_sha256: str | None = None
    response_sha256: str | None = None
    source_hash_verified: bool = False
    response_hash_verified: bool = False
    notes_present: bool = False
    notes_sha256: str | None = None
    decided_at: str | None = None
    #: Always computed, never supplied by the caller.
    clears_review_required: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        validate_language_code(self.language)
        if self.language == PIVOT_LANGUAGE:
            raise ReviewError(
                f"{self.review_id}: English rows are canonical source text and are not "
                "reviewed as translations"
            )
        try:
            decision = ReviewDecisionType(self.decision)
        except ValueError as exc:
            raise ReviewError(f"{self.review_id}: unknown decision {self.decision!r}") from exc
        try:
            competence = ReviewerCompetence(self.reviewer_language_competence)
        except ValueError as exc:
            raise ReviewError(
                f"{self.review_id}: unknown competence {self.reviewer_language_competence!r}"
            ) from exc
        if not self.reviewer_id.strip():
            raise ReviewError(f"{self.review_id}: a decision requires a named reviewer_id")

        object.__setattr__(
            self, "clears_review_required", can_clear_human_review(competence, decision)
        )

    @property
    def is_accept(self) -> bool:
        return self.decision == ReviewDecisionType.ACCEPT.value

    @property
    def hashes_verified(self) -> bool:
        return self.source_hash_verified and self.response_hash_verified

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["clears_review_required"] = self.clears_review_required
        payload["contains_text"] = False
        return dict(sorted(payload.items()))


def summarize(decisions: Sequence[ReviewDecision]) -> dict[str, Any]:
    """Aggregate decisions. Counts and identifiers only -- no text."""
    by_language: dict[str, dict[str, int]] = {}
    by_decision: dict[str, int] = {d.value: 0 for d in ReviewDecisionType}
    by_competence: dict[str, int] = {}

    for decision in decisions:
        by_decision[decision.decision] = by_decision.get(decision.decision, 0) + 1
        by_competence[decision.reviewer_language_competence] = (
            by_competence.get(decision.reviewer_language_competence, 0) + 1
        )
        bucket = by_language.setdefault(
            decision.language, {d.value: 0 for d in ReviewDecisionType}
        )
        bucket[decision.decision] = bucket.get(decision.decision, 0) + 1

    clearing = [d for d in decisions if d.clears_review_required]
    unverified = [d for d in decisions if not d.hashes_verified]

    return {
        "n_decisions": len(decisions),
        "counts_by_decision": dict(sorted(by_decision.items())),
        "counts_by_language": {k: dict(sorted(v.items())) for k, v in sorted(by_language.items())},
        "counts_by_competence": dict(sorted(by_competence.items())),
        "counts_by_language_and_decision": {
            f"{d.language}:{d.decision}": sum(
                1 for x in decisions if x.language == d.language and x.decision == d.decision
            )
            for d in decisions
        },
        "n_clearing_review_required": len(clearing),
        "n_review_required_retained": len(decisions) - len(clearing),
        "reviewers": sorted({d.reviewer_id for d in decisions}),
        "competences_present": sorted({d.reviewer_language_competence for d in decisions}),
        "competences_that_can_clear_review": sorted(c.value for c in COMPETENCE_CAN_CLEAR_REVIEW),
        "n_hash_verified": sum(1 for d in decisions if d.hashes_verified),
        "n_hash_unverified": len(unverified),
        "unverified_review_ids": sorted(d.review_id for d in unverified),
        "native_speaker_validation_claimed": any(
            d.reviewer_language_competence == ReviewerCompetence.NATIVE_SPEAKER.value
            for d in decisions
        ),
        "contains_prompt_text": False,
    }


def validate_against_rows(
    decisions: Sequence[ReviewDecision],
    translated_request_ids: Mapping[str, Any],
) -> list[str]:
    """Cross-check decisions against the set of currently translated rows."""
    problems: list[str] = []
    seen: dict[str, list[str]] = {}
    for decision in decisions:
        seen.setdefault(decision.request_id, []).append(decision.review_id)
        if decision.request_id not in translated_request_ids:
            problems.append(
                f"{decision.review_id}: {decision.request_id} is not a currently translated row"
            )
    for request_id, review_ids in sorted(seen.items()):
        if len(review_ids) > 1:
            problems.append(f"{request_id}: {len(review_ids)} decisions ({sorted(review_ids)})")
    missing = sorted(set(translated_request_ids) - set(seen))
    if missing:
        problems.append(f"{len(missing)} translated row(s) have no decision: {missing[:5]}")
    return problems
