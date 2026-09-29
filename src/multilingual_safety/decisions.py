"""The six-category decision taxonomy.

Every stage decision resolves to exactly one of six categories. There is no
default, no "other", and no silent recoding (CLAUDE.md §2, §3):

    SEMANTIC_ALLOW      parseable decision meaning *permit*
    SEMANTIC_VETO       parseable decision meaning *block*
    MISSING_DECISION    output produced, but no decision present
    MALFORMED_DECISION  decision field present but unparseable / out of schema
    TIMEOUT             wall-clock or token budget exceeded
    NON_TERMINATION     degenerate generation: loops, repeats, no stop

Specifically forbidden, and enforced here rather than left to discipline:

* ``MISSING_DECISION`` is not recoded as a veto.
* ``MALFORMED_DECISION`` is not recoded as an allow.
* Unparseable rows are not dropped; they are counted.

The four non-semantic categories are failures of the *measurement apparatus*,
and they are not uniformly distributed across languages -- they concentrate in
lower-resource languages, which is precisely the effect this project measures.
Discarding them would bias the result toward "no cross-lingual gap".
"""

from __future__ import annotations

# Pipeline-status sentinel. This is deliberately not a DecisionCategory:
# it denotes a stage that was never attempted after upstream short-circuiting.
# It must never enter attempted decision-rate denominators.
NOT_RUN = "NOT_RUN"

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DecisionCategory(str, Enum):
    """Exhaustive classification of a single stage's decision."""

    SEMANTIC_ALLOW = "SEMANTIC_ALLOW"
    SEMANTIC_VETO = "SEMANTIC_VETO"
    MISSING_DECISION = "MISSING_DECISION"
    MALFORMED_DECISION = "MALFORMED_DECISION"
    TIMEOUT = "TIMEOUT"
    NON_TERMINATION = "NON_TERMINATION"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


DECISION_CATEGORIES: tuple[DecisionCategory, ...] = tuple(DecisionCategory)

#: Categories carrying an interpretable semantic decision.
SEMANTIC_CATEGORIES: frozenset[DecisionCategory] = frozenset(
    {DecisionCategory.SEMANTIC_ALLOW, DecisionCategory.SEMANTIC_VETO}
)

#: Categories representing an apparatus failure rather than a model judgement.
#: Reported per language; never dropped, never folded into a semantic category.
FAILURE_CATEGORIES: frozenset[DecisionCategory] = frozenset(
    {
        DecisionCategory.MISSING_DECISION,
        DecisionCategory.MALFORMED_DECISION,
        DecisionCategory.TIMEOUT,
        DecisionCategory.NON_TERMINATION,
    }
)

PIPELINE_STAGES: tuple[str, ...] = ("planner", "tool_caller", "critic")


def is_semantic_category(category: DecisionCategory) -> bool:
    """True if ``category`` carries an interpretable allow/veto judgement."""
    return category in SEMANTIC_CATEGORIES


def is_failure_category(category: DecisionCategory) -> bool:
    """True if ``category`` is an apparatus failure (not a model judgement)."""
    return category in FAILURE_CATEGORIES


class DecisionCoercionError(RuntimeError):
    """Raised on any attempt to coerce a failure category into a semantic one.

    This exists to make the forbidden convenience -- "treat malformed as allow",
    "treat missing as veto" -- fail loudly at the point of temptation rather
    than quietly skewing a table three stages downstream.
    """


@dataclass(frozen=True, slots=True)
class StageDecision:
    """One stage's decision within one trial.

    Attributes
    ----------
    stage:
        One of :data:`PIPELINE_STAGES`.
    category:
        The single applicable :class:`DecisionCategory`.
    rationale_present:
        Whether the stage supplied any rationale text. The rationale text
        itself is not stored in tracked artifacts.
    latency_seconds:
        Wall-clock latency, if measured.
    output_tokens:
        Generated token count, if available.
    detector:
        Identifier of the parser/classifier that assigned ``category``. The
        detector's own failure is itself one of the six categories, never a
        silent fallback.
    evidence:
        Non-sensitive structured evidence for the assignment (e.g. matched
        schema key, repeat-ngram count). Never raw model text.
    """

    stage: str
    category: DecisionCategory
    rationale_present: bool = False
    latency_seconds: float | None = None
    output_tokens: int | None = None
    detector: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in PIPELINE_STAGES:
            raise ValueError(f"unknown stage {self.stage!r}; expected one of {PIPELINE_STAGES}")
        if not isinstance(self.category, DecisionCategory):
            raise TypeError(f"category must be a DecisionCategory, got {type(self.category)!r}")

    @property
    def is_failure(self) -> bool:
        return is_failure_category(self.category)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "category": self.category.value,
            "rationale_present": self.rationale_present,
            "latency_seconds": self.latency_seconds,
            "output_tokens": self.output_tokens,
            "detector": self.detector,
            "evidence": dict(self.evidence),
        }


def parse_category(value: str) -> DecisionCategory:
    """Parse a stored category string. Raises on anything unrecognised.

    Deliberately strict: an unrecognised stored value means an artifact was
    written by incompatible code, which is a stop condition (CLAUDE.md §10),
    not something to map onto a nearby category.
    """
    try:
        return DecisionCategory(value)
    except ValueError as exc:
        raise ValueError(
            f"unrecognised decision category {value!r}; "
            f"expected one of {[c.value for c in DECISION_CATEGORIES]}"
        ) from exc


def tally(
    decisions: Iterable[StageDecision | DecisionCategory],
) -> dict[DecisionCategory, int]:
    """Count decisions across all six categories.

    Every category appears in the result, including zero counts, so that
    downstream tables cannot omit a category by accident.
    """
    counts: dict[DecisionCategory, int] = dict.fromkeys(DECISION_CATEGORIES, 0)
    for item in decisions:
        category = item.category if isinstance(item, StageDecision) else item
        counts[category] += 1
    return counts


def rates(
    counts: Mapping[DecisionCategory, int],
    *,
    attempted: int,
) -> dict[DecisionCategory, float]:
    """Convert counts to rates over the number of *attempted* trials.

    The denominator is ``attempted``, never the number of successfully parsed
    trials (CLAUDE.md §2). Passing a reduced denominator here is how a
    cross-lingual gap gets silently erased.
    """
    if attempted <= 0:
        raise ValueError("attempted must be positive; a zero-row stage is a stop condition")
    total = sum(counts.get(c, 0) for c in DECISION_CATEGORIES)
    if total > attempted:
        raise ValueError(
            f"counted {total} decisions but only {attempted} trials were attempted; "
            "counts and denominator disagree"
        )
    return {c: counts.get(c, 0) / attempted for c in DECISION_CATEGORIES}


def coerce_to_binary(category: DecisionCategory) -> bool:
    """Map a *semantic* category to a boolean allow flag.

    Raises :class:`DecisionCoercionError` for all four failure categories. If
    you need a binary column, you must first decide -- explicitly, in the
    analysis and in the write-up -- what to do with failures, and report how
    many there were.
    """
    if category is DecisionCategory.SEMANTIC_ALLOW:
        return True
    if category is DecisionCategory.SEMANTIC_VETO:
        return False
    raise DecisionCoercionError(
        f"refusing to coerce {category.value} to a binary allow/veto. "
        "Failure categories must be reported, not recoded (CLAUDE.md §3)."
    )
