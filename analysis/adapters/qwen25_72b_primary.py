from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import polars as pl

SOURCES = [
    (
        "baseline",
        Path("artifacts/prior_qwen25_72b/results/qwen_baseline_results (1).jsonl"),
    ),
    (
        "critic_and_both_steered",
        Path(
            "artifacts/prior_qwen25_72b/results/"
            "qwen_steered_results for critic-only and both_level.jsonl"
        ),
    ),
    (
        "planner_steered",
        Path(
            "artifacts/prior_qwen25_72b/results/"
            "qwen_steered_results fr planner_only level.jsonl"
        ),
    ),
]

OUT_PARQUET = Path(
    "data/derived/canonical/qwen25_72b_primary_role_strength.parquet"
)
OUT_AUDIT = Path(
    "paper_artifacts/metrics/qwen25_72b_primary_canonical_audit.csv"
)
OUT_SOURCES = Path(
    "paper_artifacts/metrics/qwen25_72b_primary_source_manifest.csv"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


frames = []
source_rows = []

for source_label, source_path in SOURCES:
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source: {source_path}")

    frame = pl.read_ndjson(source_path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{source_path} missing required fields: {sorted(missing)}")

    invalid_nulls = frame.filter(
        pl.col("id").is_null()
        | pl.col("language").is_null()
        | pl.col("harmfulness").is_null()
        | pl.col("config").is_null()
        | pl.col("strength").is_null()
        | pl.col("is_vetoed").is_null()
    ).height

    if invalid_nulls:
        raise ValueError(
            f"{source_path} has {invalid_nulls} rows with null required values."
        )

    canonical = frame.select([
        pl.lit("qwen25_72b").alias("model"),
        pl.lit(source_label).alias("source_file_group"),
        pl.col("id").cast(pl.Utf8).alias("record_id"),
        pl.col("language").cast(pl.Utf8).alias("language"),
        pl.col("harmfulness").cast(pl.Utf8).alias("prompt_class"),
        pl.col("config").cast(pl.Utf8).alias("config"),
        pl.col("strength").cast(pl.Float64).alias("alpha"),
        pl.col("is_vetoed").cast(pl.Boolean).alias("is_vetoed"),
        pl.when(pl.col("is_vetoed"))
        .then(pl.lit("VETO_EXPLICIT"))
        .otherwise(pl.lit("NON_VETO_UNRESOLVED"))
        .alias("parser_outcome"),
        (~pl.col("is_vetoed"))
        .cast(pl.Boolean)
        .alias("operational_failure"),
        pl.when(pl.col("critic_response").cast(pl.Utf8).str.contains(r"(?i)\bVETO\b"))
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("critic_response_contains_veto_token"),
        pl.when(pl.col("critic_response").cast(pl.Utf8).str.contains(r"(?i)\bALLOW\b"))
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("critic_response_contains_allow_token"),
    ])

    frames.append(canonical)
    source_rows.append({
        "source_file_group": source_label,
        "source_path": source_path.as_posix(),
        "source_bytes": source_path.stat().st_size,
        "source_sha256": sha256(source_path),
        "source_rows": frame.height,
        "source_null_required_rows": invalid_nulls,
    })

combined = pl.concat(frames, how="vertical")

if combined.filter(pl.col("record_id").str.strip_chars() == "").height:
    raise ValueError("Found blank record_id values.")

if combined.filter(~(
    pl.col("is_vetoed")
    == (pl.col("parser_outcome") == "VETO_EXPLICIT")
)).height:
    raise ValueError("parser_outcome is inconsistent with is_vetoed.")

if combined.filter(
    pl.col("operational_failure") == pl.col("is_vetoed")
).height:
    raise ValueError("operational_failure is inconsistent with is_vetoed.")

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
        pl.len().alias("record_n"),
        pl.col("record_id").n_unique().alias("unique_record_id_n"),
        pl.col("is_vetoed").sum().alias("veto_explicit_n"),
        (~pl.col("is_vetoed")).sum().alias("non_veto_unresolved_n"),
        pl.col("operational_failure").sum().alias("operational_failure_n"),
        pl.col("critic_response_contains_veto_token")
        .sum()
        .alias("critic_response_veto_token_n"),
        pl.col("critic_response_contains_allow_token")
        .sum()
        .alias("critic_response_allow_token_n"),
    ])
    .with_columns([
        (
            100
            * pl.col("veto_explicit_n")
            / pl.col("record_n")
        ).alias("veto_explicit_rate_percentage"),
        (
            100
            * pl.col("non_veto_unresolved_n")
            / pl.col("record_n")
        ).alias("non_veto_unresolved_rate_percentage"),
        (
            100
            * pl.col("operational_failure_n")
            / pl.col("record_n")
        ).alias("operational_failure_rate_percentage"),
        (
            pl.col("record_n") == pl.col("unique_record_id_n")
        ).alias("record_ids_unique_within_cell"),
        (
            pl.col("veto_explicit_n")
            == pl.col("critic_response_veto_token_n")
        ).alias("veto_boolean_matches_critic_veto_token"),
        (
            pl.col("operational_failure_n")
            == pl.col("non_veto_unresolved_n")
        ).alias("failure_matches_non_veto_unresolved"),
    ])
    .sort([
        "source_file_group",
        "config",
        "language",
        "prompt_class",
        "alpha",
    ])
)

if not audit["record_ids_unique_within_cell"].all():
    raise ValueError("Duplicate record IDs found within at least one analysis cell.")

if not audit["veto_boolean_matches_critic_veto_token"].all():
    raise ValueError("is_vetoed does not match critic VETO-token counts.")

if not audit["failure_matches_non_veto_unresolved"].all():
    raise ValueError("Failure count does not match non-veto count.")

OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
OUT_SOURCES.parent.mkdir(parents=True, exist_ok=True)

combined.write_parquet(OUT_PARQUET)
audit.write_csv(OUT_AUDIT)

with OUT_SOURCES.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
    writer.writeheader()
    writer.writerows(source_rows)

print(f"Wrote: {OUT_PARQUET}")
print(f"Wrote: {OUT_AUDIT}")
print(f"Wrote: {OUT_SOURCES}")
print(f"Canonical rows: {combined.height}")
print(f"Audit cells: {audit.height}")

with pl.Config(tbl_cols=-1, tbl_width_chars=220):
    print(audit)
