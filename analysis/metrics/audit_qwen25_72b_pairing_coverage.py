from __future__ import annotations

from pathlib import Path

import polars as pl

INFILE = Path(
    "data/derived/canonical/qwen25_72b_primary_role_strength.parquet"
)

OUT_COVERAGE = Path(
    "paper_artifacts/metrics/qwen25_72b_primary_pairing_coverage.csv"
)

OUT_MATCHES = Path(
    "paper_artifacts/metrics/qwen25_72b_primary_pairing_matches.csv"
)

baseline = pl.read_parquet(INFILE).filter(
    (pl.col("source_file_group") == "baseline")
    & (pl.col("alpha") == 0.0)
)

steered = pl.read_parquet(INFILE).filter(
    pl.col("source_file_group").is_in([
        "critic_and_both_steered",
        "planner_steered",
    ])
    & (pl.col("alpha") > 0.0)
)

if baseline.height == 0:
    raise ValueError("No baseline alpha-0 records found.")

if steered.height == 0:
    raise ValueError("No positive-strength steered records found.")

coverage = (
    pl.concat([
        baseline.with_columns(pl.lit("baseline").alias("condition")),
        steered.with_columns(pl.lit("steered").alias("condition")),
    ])
    .group_by([
        "condition",
        "source_file_group",
        "config",
        "language",
        "prompt_class",
        "alpha",
    ])
    .agg([
        pl.len().alias("record_n"),
        pl.col("record_id").n_unique().alias("unique_record_id_n"),
    ])
    .with_columns(
        (pl.col("record_n") == pl.col("unique_record_id_n"))
        .alias("record_ids_unique_within_cell")
    )
    .sort([
        "prompt_class",
        "language",
        "config",
        "alpha",
        "condition",
    ])
)

if not coverage["record_ids_unique_within_cell"].all():
    raise ValueError("Duplicate record IDs detected within at least one cell.")

baseline_keys = (
    baseline
    .select([
        "config",
        "language",
        "prompt_class",
        "record_id",
    ])
    .rename({"record_id": "baseline_record_id"})
)

match_rows = []

steered_cells = (
    steered
    .select([
        "source_file_group",
        "config",
        "language",
        "prompt_class",
        "alpha",
    ])
    .unique()
    .sort([
        "source_file_group",
        "config",
        "language",
        "prompt_class",
        "alpha",
    ])
    .iter_rows(named=True)
)

for cell in steered_cells:
    current = steered.filter(
        (pl.col("source_file_group") == cell["source_file_group"])
        & (pl.col("config") == cell["config"])
        & (pl.col("language") == cell["language"])
        & (pl.col("prompt_class") == cell["prompt_class"])
        & (pl.col("alpha") == cell["alpha"])
    )

    current_ids = current.select("record_id").rename(
        {"record_id": "steered_record_id"}
    )

    corresponding_baseline = baseline_keys.filter(
        (pl.col("config") == cell["config"])
        & (pl.col("language") == cell["language"])
        & (pl.col("prompt_class") == cell["prompt_class"])
    ).select("baseline_record_id")

    matched_n = (
        current_ids
        .join(
            corresponding_baseline,
            left_on="steered_record_id",
            right_on="baseline_record_id",
            how="inner",
            validate="1:1",
        )
        .height
    )

    steered_n = current.height
    baseline_n = corresponding_baseline.height

    match_rows.append({
        "source_file_group": cell["source_file_group"],
        "config": cell["config"],
        "language": cell["language"],
        "prompt_class": cell["prompt_class"],
        "alpha": cell["alpha"],
        "baseline_record_n": baseline_n,
        "steered_record_n": steered_n,
        "matched_record_n": matched_n,
        "baseline_only_record_n": baseline_n - matched_n,
        "steered_only_record_n": steered_n - matched_n,
        "all_steered_ids_match_baseline": matched_n == steered_n,
        "all_baseline_ids_match_steered": matched_n == baseline_n,
        "exact_complete_pairing": (
            matched_n == baseline_n == steered_n
        ),
    })

matches = (
    pl.DataFrame(match_rows)
    .sort([
        "prompt_class",
        "language",
        "config",
        "alpha",
        "source_file_group",
    ])
)

OUT_COVERAGE.parent.mkdir(parents=True, exist_ok=True)
coverage.write_csv(OUT_COVERAGE)
matches.write_csv(OUT_MATCHES)

print(f"Wrote: {OUT_COVERAGE}")
print(f"Wrote: {OUT_MATCHES}")
print(f"Coverage cells: {coverage.height}")
print(f"Pairing checks: {matches.height}")
print(
    "Exact complete pairs: "
    f"{matches.filter(pl.col('exact_complete_pairing')).height}/"
    f"{matches.height}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=220):
    print(
        matches.select([
            "source_file_group",
            "config",
            "language",
            "prompt_class",
            "alpha",
            "baseline_record_n",
            "steered_record_n",
            "matched_record_n",
            "baseline_only_record_n",
            "steered_only_record_n",
            "exact_complete_pairing",
        ])
    )
