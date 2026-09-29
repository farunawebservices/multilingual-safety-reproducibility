from __future__ import annotations

from pathlib import Path

import polars as pl

INFILE = Path(
    "data/derived/canonical/qwen25_72b_primary_role_strength.parquet"
)

OUT = Path(
    "paper_artifacts/metrics/qwen25_72b_primary_pairing_matches_v2.csv"
)

frame = pl.read_parquet(INFILE)

baseline = frame.filter(
    (pl.col("source_file_group") == "baseline")
    & (pl.col("alpha") == 0.0)
)

steered = frame.filter(
    pl.col("source_file_group").is_in([
        "critic_and_both_steered",
        "planner_steered",
    ])
    & (pl.col("alpha") > 0.0)
)

key_columns = ["language", "prompt_class"]

for name, data in [("baseline", baseline), ("steered", steered)]:
    duplicate_n = (
        data.group_by(key_columns + ["record_id"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_n:
        raise ValueError(
            f"{name} has {duplicate_n} duplicate IDs within language/prompt-class."
        )

rows = []

cells = (
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

for cell in cells:
    current = steered.filter(
        (pl.col("source_file_group") == cell["source_file_group"])
        & (pl.col("config") == cell["config"])
        & (pl.col("language") == cell["language"])
        & (pl.col("prompt_class") == cell["prompt_class"])
        & (pl.col("alpha") == cell["alpha"])
    )

    reference = baseline.filter(
        (pl.col("language") == cell["language"])
        & (pl.col("prompt_class") == cell["prompt_class"])
    )

    steered_ids = current.select("record_id").rename(
        {"record_id": "steered_record_id"}
    )
    baseline_ids = reference.select("record_id").rename(
        {"record_id": "baseline_record_id"}
    )

    matched_n = (
        steered_ids
        .join(
            baseline_ids,
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
        "source_file_group": cell["source_file_group"],
        "baseline_config": "baseline",
        "steered_config": cell["config"],
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

result = (
    pl.DataFrame(rows)
    .sort([
        "prompt_class",
        "language",
        "steered_config",
        "alpha",
        "source_file_group",
    ])
)

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Pairing checks: {result.height}")
print(
    "Exact complete pairs: "
    f"{result.filter(pl.col('exact_complete_pairing')).height}/"
    f"{result.height}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=240):
    print(result)
