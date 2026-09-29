"""Frozen dev/test splits.

The split is assigned at the *behavior* level, never at the record level: all
seven language renderings of a behavior land in the same split. Splitting by
record would put the English rendering in dev and the Hausa rendering in test,
which leaks the item and destroys the paired design.

The split file is content-addressed. Its hash goes into every run's metadata, so
a result can always be checked against the split it claims to have used
(README §5).

Access rule: ``test`` is not read during development, prompt iteration, or
intervention-strength selection (CLAUDE.md §1). :func:`load_split` enforces this
with an explicit opt-in flag, so reading test is a deliberate, visible act.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from multilingual_safety.ids import canonical_json, is_behavior_id

DEV = "dev"
TEST = "test"
FINAL_TEST = "final_test"
SPLIT_NAMES: tuple[str, ...] = (DEV, TEST)

#: Any split whose name is here is a HELD-OUT evaluation set: reading it
#: requires an explicit opt-in, whatever it is called. Adding a differently
#: named holdout must not create a back door around the guard in
#: :func:`load_split`.
HOLDOUT_SPLIT_NAMES: frozenset[str] = frozenset({TEST, FINAL_TEST})


class SplitAccessError(RuntimeError):
    """Raised on an attempt to read the test split without explicit opt-in."""


class SplitError(ValueError):
    """Raised when a split file is structurally invalid."""


def assign_split(behavior: str, *, salt: str, test_fraction: float = 0.5) -> str:
    """Deterministically assign a behavior to dev or test.

    Deterministic and salted rather than randomly shuffled: the assignment can
    be recomputed and audited by anyone with the salt, and it does not depend on
    iteration order, dict ordering, or the RNG state of whoever ran the script.

    Changing ``salt`` re-splits the corpus and invalidates every prior test
    result. That is why it is recorded in the split file.
    """
    if not is_behavior_id(behavior):
        raise SplitError(f"malformed behavior id {behavior!r}")
    if not 0.0 < test_fraction < 1.0:
        raise SplitError(f"test_fraction must be in (0, 1), got {test_fraction}")
    digest = hashlib.sha256(f"{salt}:{behavior}".encode()).hexdigest()
    # First 8 hex digits -> uniform [0, 1)
    position = int(digest[:8], 16) / 0x1_0000_0000
    return TEST if position < test_fraction else DEV


def build_splits(
    behaviors: Iterable[str],
    *,
    salt: str,
    test_fraction: float = 0.5,
) -> dict[str, Any]:
    """Build the split payload. Does not write it."""
    assignments: dict[str, str] = {}
    for behavior in sorted(set(behaviors)):
        assignments[behavior] = assign_split(behavior, salt=salt, test_fraction=test_fraction)
    if not assignments:
        raise SplitError("refusing to build an empty split (stop condition)")

    counts = {name: sum(1 for v in assignments.values() if v == name) for name in SPLIT_NAMES}
    if 0 in counts.values():
        raise SplitError(f"degenerate split: {counts} (stop condition)")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "salt": salt,
        "test_fraction": test_fraction,
        "unit": "behavior_id",
        "counts": counts,
        "assignments": assignments,
    }
    payload["split_hash"] = split_hash(payload)
    return payload


def split_hash(payload: Mapping[str, Any]) -> str:
    """Content hash of a split payload, excluding the hash field itself."""
    material = {k: v for k, v in payload.items() if k != "split_hash"}
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


def write_splits(payload: Mapping[str, Any], path: str | Path, *, overwrite: bool = False) -> Path:
    """Write the split file.

    Refuses to overwrite by default. Re-splitting after any test read
    invalidates every prior test result, so it must be an explicit act with a
    visible trail, never an accidental rerun of a build script.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise SplitError(
            f"{path} already exists. The split is frozen before any model run; "
            "re-splitting invalidates prior test results (CLAUDE.md §1). "
            "Pass overwrite=True only with a deliberate, recorded reason."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_splits(path: str | Path) -> dict[str, Any]:
    """Load and hash-verify a split file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"split file not found: {path}. Freeze splits before running any model."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("split_hash")
    actual = split_hash(payload)
    if recorded != actual:
        raise SplitError(
            f"split hash mismatch in {path}: recorded {recorded!r}, computed {actual!r}. "
            "The split file has been edited. All results referencing the old hash are "
            "invalid (CLAUDE.md §1, §10)."
        )
    return payload


def load_split(
    path: str | Path,
    split: str,
    *,
    allow_test: bool = False,
) -> list[str]:
    """Return the behavior IDs in ``split``.

    Reading ``test`` requires ``allow_test=True``. The flag exists so that
    touching the test set is a conscious decision at the call site rather than a
    default that a debugging session can wander into.
    """
    payload_names = set(load_splits(path)["assignments"].values())
    known = payload_names or set(SPLIT_NAMES)
    if split not in known:
        raise SplitError(f"unknown split {split!r}; this file has {sorted(known)}")
    if split in HOLDOUT_SPLIT_NAMES and not allow_test:
        raise SplitAccessError(
            "refusing to read the test split. Development, prompt iteration, threshold "
            "tuning, and intervention-strength selection all happen on dev "
            "(CLAUDE.md §1). Pass allow_test=True only for a final, frozen "
            "configuration, and record the read."
        )
    payload = load_splits(path)
    return sorted(b for b, s in payload["assignments"].items() if s == split)


# ---------------------------------------------------------------------------
def build_stratified_splits(
    items: Sequence[Mapping[str, Any]],
    *,
    salt: str,
    holdout_per_stratum: int,
    stratify_by: Sequence[str] = ("harm_label", "category"),
    dev_name: str = DEV,
    holdout_name: str = FINAL_TEST,
) -> dict[str, Any]:
    """Assign behaviors to splits, stratified and deterministic.

    Within each stratum -- by default (arm, category) -- behaviors are ranked by
    ``sha256(salt : behavior_id)`` and the lowest ``holdout_per_stratum`` are
    held out. Stratifying keeps every category represented in both splits in the
    same proportion, so a per-category effect cannot be an artifact of one
    category landing entirely on one side.

    Ranking by salted hash rather than shuffling means the assignment can be
    recomputed and audited by anyone with the salt, and does not depend on
    iteration order or RNG state.

    ``items`` must be behavior-level: one entry per behavior, never per language
    row. All language variants of a behavior therefore share its split by
    construction -- the split simply has no language dimension.
    """
    if holdout_per_stratum < 0:
        raise SplitError("holdout_per_stratum must be non-negative")
    if not items:
        raise SplitError("refusing to build an empty split (stop condition)")

    seen: set[str] = set()
    strata: dict[tuple[str, ...], list[str]] = {}
    for item in items:
        behavior = item["behavior_id"]
        if not is_behavior_id(behavior):
            raise SplitError(f"malformed behavior id {behavior!r}")
        if behavior in seen:
            raise SplitError(f"duplicate behavior_id {behavior!r} in the input")
        seen.add(behavior)
        key = tuple(str(item[field]) for field in stratify_by)
        strata.setdefault(key, []).append(behavior)

    assignments: dict[str, str] = {}
    stratum_report: dict[str, dict[str, int]] = {}
    for key in sorted(strata):
        ranked = sorted(
            strata[key],
            key=lambda b: (hashlib.sha256(f"{salt}:{b}".encode()).hexdigest(), b),
        )
        if holdout_per_stratum > len(ranked):
            raise SplitError(
                f"stratum {key} has {len(ranked)} behavior(s); cannot hold out "
                f"{holdout_per_stratum}"
            )
        for index, behavior in enumerate(ranked):
            assignments[behavior] = holdout_name if index < holdout_per_stratum else dev_name
        stratum_report[":".join(key)] = {
            holdout_name: holdout_per_stratum,
            dev_name: len(ranked) - holdout_per_stratum,
            "total": len(ranked),
        }

    counts = {
        name: sum(1 for v in assignments.values() if v == name)
        for name in (dev_name, holdout_name)
    }
    if 0 in counts.values():
        raise SplitError(f"degenerate split: {counts} (stop condition)")

    payload: dict[str, Any] = {
        "schema_version": 2,
        "salt": salt,
        "unit": "behavior_id",
        "stratified_by": list(stratify_by),
        "holdout_per_stratum": holdout_per_stratum,
        "split_names": {"dev": dev_name, "holdout": holdout_name},
        "n_strata": len(strata),
        "counts": counts,
        "strata": dict(sorted(stratum_report.items())),
        "assignments": dict(sorted(assignments.items())),
    }
    payload["split_hash"] = split_hash(payload)
    return payload


def assert_no_leakage(
    payload: Mapping[str, Any],
    members: Mapping[str, Iterable[str]] | None = None,
) -> None:
    """Raise if any behavior appears in more than one split.

    In the canonical ``behavior -> split`` mapping, overlap is structurally
    impossible: a dict key holds one value. So passing only that form checks
    little. The real risk is a *derived* view -- per-split member lists built by
    a loader, a merge, or a hand-edited file -- where a behavior can appear
    twice. Pass those as ``members`` to check them properly; when omitted, the
    per-split lists are derived from ``assignments`` and the structural
    guarantee is asserted explicitly.
    """
    assignments = payload["assignments"]
    if members is None:
        by_split: dict[str, set[str]] = {}
        for behavior, split in assignments.items():
            by_split.setdefault(split, set()).add(behavior)
        total = sum(len(v) for v in by_split.values())
        if total != len(assignments):
            raise SplitError(
                f"assignment/membership mismatch: {total} members for "
                f"{len(assignments)} behaviors"
            )
        resolved = by_split
    else:
        resolved = {name: set(ids) for name, ids in members.items()}
        counts = Counter(b for ids in members.values() for b in ids)
        repeated = sorted(b for b, n in counts.items() if n > 1)
        if repeated:
            raise SplitError(
                f"split leakage: {len(repeated)} behavior(s) listed in more than one "
                f"split, e.g. {repeated[:5]}"
            )

    names = sorted(resolved)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sorted(resolved[left] & resolved[right])
            if overlap:
                raise SplitError(
                    f"split leakage: {len(overlap)} behavior(s) in both {left!r} and "
                    f"{right!r}, e.g. {overlap[:5]}"
                )
