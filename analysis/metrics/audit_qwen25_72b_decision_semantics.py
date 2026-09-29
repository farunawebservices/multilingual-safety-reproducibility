from __future__ import annotations

import csv
import re
from pathlib import Path

import polars as pl

OUT = Path(
    "paper_artifacts/metrics/"
    "qwen25_72b_decision_semantics_audit.csv"
)

SOURCES = [
    Path("artifacts/prior_qwen25_72b/results/qwen_baseline_results (1).jsonl"),
    Path(
        "artifacts/prior_qwen25_72b/results/"
        "qwen_steered_results for critic-only and both_level.jsonl"
    ),
    Path(
        "artifacts/prior_qwen25_72b/results/"
        "qwen_steered_results fr planner_only level.jsonl"
    ),
]

RESPONSE_COLUMNS = [
    "planner_response",
    "critic_response",
    "tool_response",
]

PATTERNS = {
    "contains_veto_token": re.compile(r"\bVETO\b", re.IGNORECASE),
    "contains_allow_token": re.compile(r"\bALLOW\b", re.IGNORECASE),
    "contains_no_decision_token": re.compile(
        r"\bNO[_ -]?DECISION\b|\bUNPARSEABLE\b|\bPARSE[_ -]?FAIL",
        re.IGNORECASE,
    ),
    "empty_or_whitespace": re.compile(r"^\s*$"),
}

rows = []

for source in SOURCES:
    frame = pl.read_ndjson(source)

    required = {
        "id",
        "language",
        "harmfulness",
        "config",
        "strength",
        "is_vetoed",
        *RESPONSE_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} missing columns: {sorted(missing)}")

    for response_column in RESPONSE_COLUMNS:
        values = (
            frame
            .select(pl.col(response_column).cast(pl.Utf8).fill_null(""))
            .to_series()
            .to_list()
        )

        for pattern_name, pattern in PATTERNS.items():
            matching_n = sum(bool(pattern.search(value)) for value in values)

            rows.append({
                "source_path": source.as_posix(),
                "response_column": response_column,
                "pattern": pattern_name,
                "total_records": frame.height,
                "matching_records_n": matching_n,
                "matching_records_rate_percentage": (
                    100 * matching_n / frame.height
                ),
            })

        veto_true_n = frame.filter(pl.col("is_vetoed")).height
        veto_false_n = frame.filter(~pl.col("is_vetoed")).height

        rows.append({
            "source_path": source.as_posix(),
            "response_column": response_column,
            "pattern": "is_vetoed_true",
            "total_records": frame.height,
            "matching_records_n": veto_true_n,
            "matching_records_rate_percentage": 100 * veto_true_n / frame.height,
        })

        rows.append({
            "source_path": source.as_posix(),
            "response_column": response_column,
            "pattern": "is_vetoed_false",
            "total_records": frame.height,
            "matching_records_n": veto_false_n,
            "matching_records_rate_percentage": 100 * veto_false_n / frame.height,
        })

OUT.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "source_path",
    "response_column",
    "pattern",
    "total_records",
    "matching_records_n",
    "matching_records_rate_percentage",
]

with OUT.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote: {OUT}")

for source in SOURCES:
    print(f"\n--- {source.name} ---")
    for row in rows:
        if row["source_path"] == source.as_posix():
            print(
                f"{row['response_column']:16} "
                f"{row['pattern']:28} "
                f"{row['matching_records_n']:>5}/"
                f"{row['total_records']:<5} "
                f"({row['matching_records_rate_percentage']:.2f}%)"
            )
