from pathlib import Path

import polars as pl

INFILE = Path(
    "data/derived/canonical/gemma_4_31b_phase1_complete_case.parquet"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "gemma_4_31b_phase1_complete_case_pairing_audit.csv"
)

frame = pl.read_parquet(INFILE)

baseline = frame.filter(
    (pl.col("source_file_group") == "baseline")
    & (pl.col("alpha") == 0.0)
)

steered = frame.filter(
    (pl.col("source_file_group") == "steered")
    & (pl.col("alpha") > 0.0)
)

if baseline.height == 0 or steered.height == 0:
    raise ValueError("Expected nonempty baseline and steered data.")

baseline_key = ["language", "prompt_class", "record_id"]

duplicates = (
    baseline
    .group_by(baseline_key)
    .len()
    .filter(pl.col("len") > 1)
    .height
)

if duplicates:
    raise ValueError(f"Baseline duplicate ID keys: {duplicates}")

rows = []

cells = (
    steered
    .select(["config", "language", "prompt_class", "alpha"])
    .unique()
    .sort(["prompt_class", "language", "config", "alpha"])
    .iter_rows(named=True)
)

for cell in cells:
    current = steered.filter(
        (pl.col("config") == cell["config"])
        & (pl.col("language") == cell["language"])
        & (pl.col("prompt_class") == cell["prompt_class"])
        & (pl.col("alpha") == cell["alpha"])
    )

    steered_duplicates = (
        current
        .group_by("record_id")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )

    if steered_duplicates:
        raise ValueError(
            f"Duplicate steered IDs in cell {cell}: {steered_duplicates}"
        )

    reference = baseline.filter(
        (pl.col("language") == cell["language"])
        & (pl.col("prompt_class") == cell["prompt_class"])
    )

    matched_n = (
        current
        .select(pl.col("record_id").alias("steered_record_id"))
        .join(
            reference.select(
                pl.col("record_id").alias("baseline_record_id")
            ),
            left_on="steered_record_id",
            right_on="baseline_record_id",
            how="inner",
            validate="1:1",
        )
        .height
    )

    baseline_n = reference.height
    steered_n = current.height

    rows.append({
        "steered_config": cell["config"],
        "language": cell["language"],
        "prompt_class": cell["prompt_class"],
        "alpha": cell["alpha"],
        "baseline_valid_record_n": baseline_n,
        "steered_valid_record_n": steered_n,
        "matched_valid_record_n": matched_n,
        "baseline_only_valid_record_n": baseline_n - matched_n,
        "steered_only_valid_record_n": steered_n - matched_n,
        "all_steered_ids_match_baseline": matched_n == steered_n,
        "all_baseline_ids_match_steered": matched_n == baseline_n,
        "exact_complete_case_pairing": (
            matched_n == baseline_n == steered_n
        ),
        "source_truncation_caveat": (
            "One invalid terminal JSONL record excluded from each source; "
            "outcomes unavailable and not imputed."
        ),
    })

result = (
    pl.DataFrame(rows)
    .sort(["prompt_class", "language", "steered_config", "alpha"])
)

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Steered comparison cells: {result.height}")
print(
    "Exact complete-case pairs: "
    f"{result.filter(pl.col('exact_complete_case_pairing')).height}/"
    f"{result.height}"
)
print(
    "All steered IDs represented in baseline: "
    f"{result.filter(pl.col('all_steered_ids_match_baseline')).height}/"
    f"{result.height}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=240):
    print(result)
