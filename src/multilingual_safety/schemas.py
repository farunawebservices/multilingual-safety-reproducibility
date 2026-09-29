"""Expected schemas for Stage 0 sources, and loud failure when they drift.

Every expectation below was **verified against the upstream source**, not
assumed. Two of them contradict the obvious guess, which is precisely why they
are pinned here:

* **JBB-Behaviors has no separate "harmful" and "benign" configurations.** It
  exposes ONE config, ``behaviors``, with two *splits*, ``harmful`` and
  ``benign`` (plus a second, unrelated config ``judge_comparison``). Code that
  assumes ``load_dataset(..., "harmful")`` fails.
* **HarmBench's val/test partition is expressed as separate FILES, not a
  column.** ``harmbench_behaviors_text_all.csv`` carries no split field; the
  partition lives in ``harmbench_behaviors_text_val.csv`` (80 rows) and
  ``harmbench_behaviors_text_test.csv`` (320 rows).

Drift from any of this raises :class:`SchemaMismatchError`. Silently adapting to
a changed upstream schema would let the corpus change underneath a frozen
manifest without anyone noticing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class SchemaMismatchError(RuntimeError):
    """Raised when acquired data does not match its pinned expected schema."""


@dataclass(frozen=True, slots=True)
class PartSchema:
    """Expected shape of one acquired file / split."""

    part_id: str
    columns: tuple[str, ...]
    n_rows: int
    id_column: str
    text_columns: tuple[str, ...]
    #: Columns permitted to be empty. Absent from this set means "must be
    #: non-empty in every row"; a violation is reported, never imputed.
    nullable_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    arm: str | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class SourceSchema:
    """Expected shape of one upstream source."""

    source_id: str
    name: str
    parts: Mapping[str, PartSchema]
    license_id: str = "unknown"
    #: HF config -> splits, when the source is a Hugging Face dataset.
    hf_configs: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------
# JailbreakBench / JBB-Behaviors
# Verified 2026-08-16 at revision 886acc352a31533ffbcf4ef22c744658688086fc.
# ---------------------------------------------------------------------------
JBB_COLUMNS: tuple[str, ...] = ("Index", "Goal", "Target", "Behavior", "Category", "Source")

JBB_SCHEMA = SourceSchema(
    source_id="jailbreakbench",
    name="JailbreakBench/JBB-Behaviors",
    license_id="MIT",
    # VERIFIED, not assumed: one `behaviors` config with harmful/benign SPLITS.
    hf_configs={
        "behaviors": ("harmful", "benign"),
        "judge_comparison": ("test",),
    },
    parts={
        "harmful": PartSchema(
            part_id="harmful",
            columns=JBB_COLUMNS,
            n_rows=100,
            id_column="Index",
            text_columns=("Goal", "Target", "Behavior"),
            categorical_columns=("Category", "Source"),
            arm="harmful",
            notes="data/harmful-behaviors.csv; split 'harmful' of config 'behaviors'.",
        ),
        "benign": PartSchema(
            part_id="benign",
            columns=JBB_COLUMNS,
            n_rows=100,
            id_column="Index",
            text_columns=("Goal", "Target", "Behavior"),
            categorical_columns=("Category", "Source"),
            arm="benign",
            notes="data/benign-behaviors.csv; split 'benign' of config 'behaviors'.",
        ),
    },
    notes=(
        "The `judge_comparison` config is a separate artifact (judge agreement data) "
        "and is NOT acquired at Stage 0. The `Index` column restarts at 0 in each "
        "split, so it is unique only WITHIN a split -- never use it as a corpus-wide key."
    ),
)

# ---------------------------------------------------------------------------
# HarmBench
# Verified 2026-08-16 at commit c0423b952435fcc8467108d8f25962dbae5b7de2.
# ---------------------------------------------------------------------------
HARMBENCH_COLUMNS: tuple[str, ...] = (
    "Behavior",
    "FunctionalCategory",
    "SemanticCategory",
    "Tags",
    "ContextString",
    "BehaviorID",
)

#: `text_all` is the text-only behavior set, as distinct from the repo's
#: `harmbench_behaviors_multimodal_all.csv`. It is NOT uniformly "plain text
#: prompt" though: 100 `contextual` rows carry a required ContextString, and 100
#: `copyright` rows are hash-checked generation requests. Only the 200
#: `standard` rows are self-contained text behaviors.
HARMBENCH_FUNCTIONAL_CATEGORIES: tuple[str, ...] = ("standard", "contextual", "copyright")

_HARMBENCH_PART_KWARGS: dict[str, Any] = {
    "columns": HARMBENCH_COLUMNS,
    "id_column": "BehaviorID",
    "text_columns": ("Behavior", "ContextString"),
    # ContextString is empty for `standard` and `copyright` rows by design;
    # Tags is empty for `standard` rows. Neither is a data-quality defect.
    "nullable_columns": ("ContextString", "Tags"),
    "categorical_columns": ("FunctionalCategory", "SemanticCategory"),
    "arm": "harmful",
}

HARMBENCH_SCHEMA = SourceSchema(
    source_id="harmbench",
    name="HarmBench text behaviors",
    license_id="MIT",
    parts={
        "text_all": PartSchema(
            part_id="text_all", n_rows=400,
            notes="Full text behavior set = val (80) + test (320).",
            **_HARMBENCH_PART_KWARGS,
        ),
        "text_val": PartSchema(
            part_id="text_val", n_rows=80,
            notes="OFFICIAL validation partition -- a separate FILE, not a column.",
            **_HARMBENCH_PART_KWARGS,
        ),
        "text_test": PartSchema(
            part_id="text_test", n_rows=320,
            notes="OFFICIAL test partition -- a separate FILE, not a column.",
            **_HARMBENCH_PART_KWARGS,
        ),
    },
    notes=(
        "No split column exists anywhere in these files. The official partition is "
        "file-level: val + test are disjoint and together reconstruct text_all."
    ),
)

EXPECTED_SCHEMAS: Mapping[str, SourceSchema] = {
    "jailbreakbench": JBB_SCHEMA,
    "harmbench": HARMBENCH_SCHEMA,
}


def get_schema(source_id: str) -> SourceSchema:
    try:
        return EXPECTED_SCHEMAS[source_id]
    except KeyError as exc:
        raise SchemaMismatchError(
            f"no pinned schema for source {source_id!r}; known: {sorted(EXPECTED_SCHEMAS)}"
        ) from exc


def get_part(source_id: str, part_id: str) -> PartSchema:
    schema = get_schema(source_id)
    try:
        return schema.parts[part_id]
    except KeyError as exc:
        raise SchemaMismatchError(
            f"source {source_id!r} has no part {part_id!r}; known: {sorted(schema.parts)}"
        ) from exc


def check_columns(source_id: str, part_id: str, columns: Sequence[str]) -> list[str]:
    """Return column-level violations. Order matters: upstream reordering is drift."""
    expected = get_part(source_id, part_id).columns
    actual = tuple(columns)
    if actual == expected:
        return []

    violations: list[str] = []
    missing = [c for c in expected if c not in actual]
    unexpected = [c for c in actual if c not in expected]
    if missing:
        violations.append(f"missing column(s): {missing}")
    if unexpected:
        violations.append(f"unexpected column(s): {unexpected}")
    if not missing and not unexpected:
        violations.append(f"column ORDER changed: expected {list(expected)}, got {list(actual)}")
    return violations


def check_row_count(source_id: str, part_id: str, n_rows: int) -> list[str]:
    expected = get_part(source_id, part_id).n_rows
    if n_rows != expected:
        return [f"row count changed: expected {expected}, got {n_rows}"]
    return []


def check_hf_configs(source_id: str, configs: Mapping[str, Sequence[str]]) -> list[str]:
    """Verify the *actual* HF config/split names against the pinned ones.

    This is the check that catches the "assumed a `harmful` config" bug: JBB
    exposes splits, not configs, with those names.
    """
    expected = get_schema(source_id).hf_configs
    violations: list[str] = []
    for name, splits in sorted(expected.items()):
        if name not in configs:
            violations.append(f"expected HF config {name!r} is absent; found {sorted(configs)}")
            continue
        actual_splits = tuple(sorted(configs[name]))
        if actual_splits != tuple(sorted(splits)):
            violations.append(
                f"config {name!r}: expected splits {sorted(splits)}, got {list(actual_splits)}"
            )
    for name in sorted(configs):
        if name not in expected:
            violations.append(f"unexpected new HF config {name!r}")
    return violations


def validate_part(
    source_id: str,
    part_id: str,
    *,
    columns: Sequence[str],
    n_rows: int,
) -> list[str]:
    """Collect all violations for one part. Empty list == matches expectations."""
    return [
        *check_columns(source_id, part_id, columns),
        *check_row_count(source_id, part_id, n_rows),
    ]


def assert_part(
    source_id: str,
    part_id: str,
    *,
    columns: Sequence[str],
    n_rows: int,
) -> None:
    """Fail loudly on any schema drift.

    Upstream data changing is a legitimate event, but it must be noticed and
    re-pinned deliberately -- never absorbed silently into a downstream corpus.
    """
    violations = validate_part(source_id, part_id, columns=columns, n_rows=n_rows)
    if violations:
        raise SchemaMismatchError(
            f"SCHEMA DRIFT in {source_id}/{part_id}:\n  - "
            + "\n  - ".join(violations)
            + "\n\nThe pinned expectation lives in src/multilingual_safety/schemas.py. "
            "Re-pin it deliberately (and re-verify every downstream hash) rather than "
            "loosening this check."
        )
