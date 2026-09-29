"""Deterministic hashing and provenance digests.

Everything here is **content-addressed and order-independent**: the same bytes,
or the same logical table, always produce the same digest, on any machine, in
any Python process. That is what makes "we built the same corpus" a checkable
claim rather than an assertion (THIRD_PARTY_DATASETS.md).

Two properties are load-bearing and are asserted in ``tests/test_hashing.py``:

* **Determinism across processes.** No ``hash()``, no ``set`` iteration order,
  no dict insertion order, no locale-dependent formatting.
* **Timestamp separation.** Digests never include wall-clock time or run IDs.
  A report can therefore be compared across runs to prove the *content* is
  unchanged even though the run ID and retrieval timestamp differ.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

CHUNK_SIZE = 1 << 20  # 1 MiB

#: Fields excluded from `deterministic_digest`: they change every run by design.
NONDETERMINISTIC_FIELDS: frozenset[str] = frozenset(
    {
        "retrieved_at",
        "generated_at",
        "started_at",
        "timestamp",
        "run_id",
        "duration_seconds",
        "elapsed_seconds",
        "environment",
        "hostname",
    }
)


def bytes_sha256(payload: bytes) -> str:
    """SHA-256 of raw bytes, lowercase hex."""
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, streamed so large files do not load into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str, *, normalize_form: str = "NFC") -> str:
    """SHA-256 of Unicode-normalised text.

    Normalising first means an NFD round-trip does not read as corruption,
    while genuine diacritic loss still changes the digest.
    """
    return hashlib.sha256(unicodedata.normalize(normalize_form, text).encode("utf-8")).hexdigest()


def canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, tight separators, UTF-8 preserved."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def json_sha256(payload: Any) -> str:
    """SHA-256 over a canonical JSON encoding. Key order cannot affect it."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _strip_nondeterministic(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return {
            k: _strip_nondeterministic(v)
            for k, v in sorted(payload.items())
            if k not in NONDETERMINISTIC_FIELDS
        }
    if isinstance(payload, (list, tuple)):
        return [_strip_nondeterministic(v) for v in payload]
    return payload


def deterministic_digest(payload: Any) -> str:
    """Content digest of a report, ignoring timestamps and run IDs.

    Re-running an acquisition must reproduce this digest exactly. If it changes,
    the upstream data changed -- which is a finding, not a nuisance.
    """
    return json_sha256(_strip_nondeterministic(payload))


def table_digest(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
) -> str:
    """Order-independent digest of a table's full contents.

    Row order does not affect the result (row digests are sorted), but column
    names, values, and row *multiplicity* all do -- so a reordered file matches
    while an edited or de-duplicated one does not.
    """
    if columns is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = sorted(seen)
    else:
        columns = list(columns)

    row_digests = sorted(
        json_sha256([str(row.get(col, "")) for col in columns]) for row in rows
    )
    return json_sha256({"columns": list(columns), "rows": row_digests})


def column_digest(values: Iterable[Any]) -> str:
    """Order-independent digest of a single column's values."""
    return json_sha256(sorted(text_sha256(str(v)) for v in values))


def directory_digest(directory: str | Path, *, pattern: str = "*") -> dict[str, str]:
    """Map of ``relative path -> sha256`` for every file under ``directory``.

    Sorted by path, so the mapping serialises identically everywhere.
    """
    root = Path(directory)
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): file_sha256(p)
        for p in sorted(root.rglob(pattern))
        if p.is_file()
    }


def verify_file(path: str | Path, expected_sha256: str) -> bool:
    """Constant-shape comparison of a file against an expected digest."""
    return file_sha256(path) == expected_sha256.strip().lower()
