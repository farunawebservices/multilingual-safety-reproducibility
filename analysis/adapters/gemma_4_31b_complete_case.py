from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import polars as pl

SOURCES = [
    (
        "baseline",
        Path(
            "data/raw/phase1/gemma_4_31b/originals/"
            "gemma_4_31b_baseline_results (1).jsonl"
        ),
    ),
    (
        "steered",
        Path(
            "data/raw/phase1/gemma_4_31b/originals/"
            "gemma_4_31b_steered_results (1).jsonl"
        ),
    ),
]

OUT_PARQUET = Path(
    "data/derived/canonical/gemma_4_31b_phase1_complete_case.parquet"
)

OUT_AUDIT = Path(
    "paper_artifacts/metrics/gemma_4_31b_phase1_complete_case_audit.csv"
)

OUT_MANIFEST = Path(
    "paper_artifacts/metrics/gemma_4_31b_phase1_source_manifest.csv"
)

REQUIRED = {
    "id",
    "language",
    "harmfulness",
    "config",
    "strength",
    "is_vetoed",
    "critic_response",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


frames = []
manifest_rows = []

for source_group, source_path in SOURCES:
    valid_records = []
    invalid_rows = []

    with source_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue

            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                invalid_rows.append({
                    "line_number": line_number,
                    "line_bytes": len(raw_line),
                    "line_sha256": sha256_bytes(raw_line),
                })
                continue

            missing = REQUIRED - set(record)
            if missing:
                raise ValueError(
                    f"{source_path}, line {line_number}: "
                    f"missing required fields {sorted(missing)}"
                )

            null_required = [
                field for field in REQUIRED
                if record[field] is None
            ]
            if null_required:
                raise ValueError(
                    f"{source_path}, line {line_number}: "
                    f"null required fields {null_required}"
                )

            valid_records.append(record)

    if len(invalid_rows) != 1:
        raise ValueError(
            f"Expected exactly one invalid JSONL row in {source_path}; "
            f"found {len(invalid_rows)}."
        )

    frame = pl.DataFrame(valid_records)

    canonical = frame.select([
        pl.lit("gemma_4_31b").alias("model"),
        pl.lit("phase1_complete_case").alias("analysis_population"),
        pl.lit(source_group).alias("source_file_group"),
        pl.col("id").cast(pl.Utf8).alias("record_id"),
        pl.col("language").cast(pl.Utf8).alias("language"),
        pl.col("harmfulness").cast(pl.Utf8).alias("prompt_class"),
        pl.col("config").cast(pl.Utf8).alias("config"),
        pl.col("strength").cast(pl.Float64).alias("alpha"),
        pl.col("is_vetoed").cast(pl.Boolean).alias("is_vetoed"),
        pl.when(pl.col("is_vetoed"))
        .then(pl.lit("VETO_BOOLEAN_TRUE"))
        .otherwise(pl.lit("NON_VETO_BOOLEAN_FALSE"))
        .alias("decision_status"),
        (~pl.col("is_vetoed"))
        .cast(pl.Boolean)
        .alias("operational_failure"),
        pl.col("critic_response")
        .cast(pl.Utf8)
        .str.contains(r"(?i)\bVETO\b")
        .fill_null(False)
        .alias("critic_response_contains_veto_token"),
        pl.col("critic_response")
        .cast(pl.Utf8)
        .str.contains(r"(?i)\bALLOW\b")
        .fill_null(False)
        .alias("critic_response_contains_allow_token"),
    ])

    frames.append(canonical)

    manifest_rows.append({
        "source_file_group": source_group,
        "source_path": source_path.as_posix(),
        "source_sha256": sha256_file(source_path),
        "source_nonempty_line_n": len(valid_records) + len(invalid_rows),
        "valid_json_record_n": len(valid_records),
        "invalid_json_record_n": len(invalid_rows),
        "excluded_invalid_line_number": invalid_rows[0]["line_number"],
        "excluded_invalid_line_bytes": invalid_rows[0]["line_bytes"],
        "excluded_invalid_line_sha256": invalid_rows[0]["line_sha256"],
        "exclusion_reason": (
            "Truncated JSONL terminal record; is_vetoed outcome absent "
            "before truncation; not imputed."
        ),
    })

combined = pl.concat(frames, how="vertical")

if combined.filter(pl.col("record_id").str.strip_chars() == "").height:
    raise ValueError("Blank record ID found.")

audit = (
    combined
    .group_by([
        "source_file_group",
        "config",
        "language",
        "prompt_class",
        "alpha",
    ])
    .agg([
        pl.len().alias("valid_record_n"),
        pl.col("record_id").n_unique().alias("unique_record_id_n"),
        pl.col("is_vetoed").sum().alias("veto_boolean_true_n"),
        (~pl.col("is_vetoed")).sum().alias("non_veto_boolean_false_n"),
        pl.col("operational_failure").sum().alias("operational_failure_n"),
        pl.col("critic_response_contains_veto_token")
        .sum()
        .alias("critic_veto_token_n"),
        pl.col("critic_response_contains_allow_token")
        .sum()
        .alias("critic_allow_token_n"),
    ])
    .with_columns([
        (
            pl.col("valid_record_n")
            == pl.col("unique_record_id_n")
        ).alias("record_ids_unique_within_cell"),
        (
            pl.col("operational_failure_n")
            == pl.col("non_veto_boolean_false_n")
        ).alias("failure_definition_valid"),
    ])
    .sort([
        "source_file_group",
        "prompt_class",
        "language",
        "config",
        "alpha",
    ])
)

for column in [
    "record_ids_unique_within_cell",
    "failure_definition_valid",
]:
    if not audit[column].all():
        raise ValueError(f"Audit validation failed: {column}")

OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

combined.write_parquet(OUT_PARQUET)
audit.write_csv(OUT_AUDIT)

with OUT_MANIFEST.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"Wrote: {OUT_PARQUET}")
print(f"Wrote: {OUT_AUDIT}")
print(f"Wrote: {OUT_MANIFEST}")
print(f"Canonical complete-case rows: {combined.height}")
print(f"Audit cells: {audit.height}")
print(f"Excluded invalid source rows: {sum(r['invalid_json_record_n'] for r in manifest_rows)}")
