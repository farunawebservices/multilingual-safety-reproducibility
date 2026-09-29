from __future__ import annotations

import hashlib
import json
from pathlib import Path

SOURCES = [
    Path(
        "data/raw/phase1/gemma_4_31b/originals/"
        "gemma_4_31b_baseline_results (1).jsonl"
    ),
    Path(
        "data/raw/phase1/gemma_4_31b/originals/"
        "gemma_4_31b_steered_results (1).jsonl"
    ),
]

for source in SOURCES:
    print(f"\n--- {source.name} ---")
    invalid = []

    with source.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                decoded = raw.decode("utf-8", errors="replace")
                invalid.append({
                    "line_number": line_number,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "error_type": type(exc).__name__,
                    "error_position": getattr(exc, "pos", None),
                    "starts_with_open_brace": decoded.lstrip().startswith("{"),
                    "ends_with_close_brace": decoded.rstrip().endswith("}"),
                    "contains_null_byte": "\x00" in decoded,
                    "newline_count": decoded.count("\n"),
                })

    print(f"Invalid nonempty lines: {len(invalid)}")
    for row in invalid:
        print(
            "line={line_number} bytes={bytes} "
            "error={error_type} pos={error_position} "
            "starts_brace={starts_with_open_brace} "
            "ends_brace={ends_with_close_brace} "
            "null_byte={contains_null_byte} "
            "embedded_newlines={newline_count} sha256={sha256}".format(**row)
        )
