"""Duplicate detection over acquired source tables.

The distinction this module exists to enforce: **a repeated primary text is not
the same thing as a duplicated item.** HarmBench pairs a single instruction with
several different ``ContextString`` values, so five ``Behavior`` strings recur
while every row remains a distinct behavior. Deduplicating on the primary text
column alone would silently delete seven real items.

So duplicates are reported at three levels, and only the composite level is
treated as a genuine defect:

    ids        -- repeated identifier: always a defect
    text       -- repeated primary text column: informational
    composite  -- repeated across ALL text columns: a genuine duplicate item

Everything is reported by **hash**, never by echoing the text, so results are
safe to write into a committed report (THIRD_PARTY_DATASETS.md).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from multilingual_safety.hashing import text_sha256

#: Separator used to join columns into a composite key. A unit separator cannot
#: occur in normal text, so "a" + "bc" cannot collide with "ab" + "c".
COMPOSITE_SEPARATOR = "\x1f"


@dataclass(frozen=True, slots=True)
class DuplicateReport:
    """Duplicate findings for one table. Contains hashes and counts, never text."""

    n_rows: int
    key_name: str
    n_distinct: int
    n_duplicate_keys: int
    n_rows_affected: int
    duplicate_keys: tuple[str, ...] = ()
    counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def has_duplicates(self) -> bool:
        return self.n_duplicate_keys > 0

    def to_dict(self, *, max_keys: int = 50) -> dict[str, Any]:
        return {
            "key_name": self.key_name,
            "n_rows": self.n_rows,
            "n_distinct": self.n_distinct,
            "n_duplicate_keys": self.n_duplicate_keys,
            "n_rows_affected": self.n_rows_affected,
            "duplicate_keys": list(self.duplicate_keys[:max_keys]),
            "truncated": len(self.duplicate_keys) > max_keys,
        }


def _report(values: Sequence[str], key_name: str) -> DuplicateReport:
    counts = Counter(values)
    duplicates = sorted(k for k, n in counts.items() if n > 1)
    return DuplicateReport(
        n_rows=len(values),
        key_name=key_name,
        n_distinct=len(counts),
        n_duplicate_keys=len(duplicates),
        n_rows_affected=sum(counts[k] for k in duplicates),
        duplicate_keys=tuple(duplicates),
        counts={k: counts[k] for k in duplicates},
    )


def find_duplicate_ids(rows: Sequence[Mapping[str, Any]], id_column: str) -> DuplicateReport:
    """Duplicate identifiers. Always a defect -- IDs are meant to be unique.

    Values are reported verbatim because identifiers are metadata, not content.
    """
    return _report([str(r.get(id_column, "")) for r in rows], id_column)


def find_duplicate_text(rows: Sequence[Mapping[str, Any]], text_column: str) -> DuplicateReport:
    """Repeated values in one text column, reported as hashes.

    Informational only: see the module docstring for why this is not by itself
    evidence of duplicated items.
    """
    return _report([text_sha256(str(r.get(text_column, ""))) for r in rows], f"sha256({text_column})")


def find_duplicate_composite(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> DuplicateReport:
    """Rows identical across *every* listed column. This is a genuine duplicate."""
    if not columns:
        raise ValueError("composite duplicate detection requires at least one column")
    keys = [
        text_sha256(COMPOSITE_SEPARATOR.join(str(r.get(c, "")) for c in columns)) for r in rows
    ]
    return _report(keys, f"sha256({'+'.join(columns)})")


def duplicate_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    id_column: str,
    text_columns: Sequence[str],
) -> dict[str, Any]:
    """All three levels at once, plus the adjudication of what counts as a defect."""
    ids = find_duplicate_ids(rows, id_column)
    primary = find_duplicate_text(rows, text_columns[0]) if text_columns else None
    composite = find_duplicate_composite(rows, text_columns) if text_columns else None

    return {
        "ids": ids.to_dict(),
        "primary_text": primary.to_dict() if primary else None,
        "composite": composite.to_dict() if composite else None,
        "composite_key_columns": list(text_columns),
        # Only IDs and composite duplicates are defects.
        "is_defective": bool(ids.has_duplicates or (composite and composite.has_duplicates)),
        "note": (
            "A repeated primary text column is NOT a duplicate item when another text "
            "column distinguishes the rows (e.g. HarmBench pairs one Behavior with "
            "several ContextStrings). Only composite duplicates are genuine."
        ),
    }
