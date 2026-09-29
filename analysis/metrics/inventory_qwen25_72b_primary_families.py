from pathlib import Path

import polars as pl

INFILE = Path(
    "paper_artifacts/metrics/"
    "qwen25_72b_primary_paired_inference_holm.csv"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "qwen25_72b_primary_family_inventory.csv"
)

frame = pl.read_csv(INFILE)

required = [
    "source_file_group",
    "baseline_config",
    "steered_config",
    "alpha",
    "language",
    "prompt_class",
    "paired_record_n",
    "baseline_only_record_n",
    "steered_only_record_n",
    "partial_baseline_coverage",
    "all_steered_ids_match_baseline",
    "correction_family",
    "correction_family_size",
]

missing = sorted(set(required) - set(frame.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if frame.height != 33:
    raise ValueError(f"Expected 33 inference rows; found {frame.height}")

result = (
    frame
    .group_by([
        "source_file_group",
        "baseline_config",
        "steered_config",
        "alpha",
        "correction_family",
        "correction_family_size",
    ])
    .agg([
        pl.len().alias("cell_n"),
        pl.col("language").n_unique().alias("language_n"),
        pl.col("prompt_class").n_unique().alias("prompt_class_n"),
        pl.col("paired_record_n").sum().alias("paired_record_n_total"),
        pl.col("baseline_only_record_n")
        .sum()
        .alias("baseline_only_record_n_total"),
        pl.col("steered_only_record_n")
        .sum()
        .alias("steered_only_record_n_total"),
        pl.col("partial_baseline_coverage")
        .sum()
        .alias("partial_baseline_coverage_cell_n"),
        (
            ~pl.col("all_steered_ids_match_baseline")
        )
        .sum()
        .alias("steered_ids_not_subset_cell_n"),
    ])
    .with_columns([
        (
            pl.col("language_n") * pl.col("prompt_class_n")
        ).alias("expected_full_grid_cell_n"),
        (
            pl.col("cell_n")
            == pl.col("language_n") * pl.col("prompt_class_n")
        ).alias("has_full_language_prompt_grid"),
        (
            (pl.col("partial_baseline_coverage_cell_n") == 0)
            & (pl.col("steered_ids_not_subset_cell_n") == 0)
        ).alias("complete_pairing_coverage"),
    ])
    .sort([
        "source_file_group",
        "baseline_config",
        "steered_config",
        "alpha",
    ])
)

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Inference rows inventoried: {frame.height}")
print(f"Candidate comparison families: {result.height}")
print(
    "Families with complete pairing coverage: "
    f"{result.filter(pl.col('complete_pairing_coverage')).height}"
)
print(
    "Families with a full language/prompt grid: "
    f"{result.filter(pl.col('has_full_language_prompt_grid')).height}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=260):
    print(result)
