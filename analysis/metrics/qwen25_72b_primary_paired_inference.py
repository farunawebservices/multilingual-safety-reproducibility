from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import binomtest

INFILE = Path(
    "data/derived/canonical/qwen25_72b_primary_role_strength.parquet"
)

OUT = Path(
    "paper_artifacts/metrics/qwen25_72b_primary_paired_inference.csv"
)

SEED = 20260908
BOOTSTRAP_REPLICATES = 10_000

frame = pl.read_parquet(INFILE)

baseline = (
    frame
    .filter(
        (pl.col("source_file_group") == "baseline")
        & (pl.col("alpha") == 0.0)
    )
    .select([
        "record_id",
        "language",
        "prompt_class",
        pl.col("is_vetoed").alias("baseline_is_vetoed"),
        pl.col("operational_failure").alias("baseline_failure"),
    ])
)

steered = frame.filter(
    pl.col("source_file_group").is_in([
        "critic_and_both_steered",
        "planner_steered",
    ])
    & (pl.col("alpha") > 0.0)
)

rng = np.random.default_rng(SEED)
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
    current = (
        steered
        .filter(
            (pl.col("source_file_group") == cell["source_file_group"])
            & (pl.col("config") == cell["config"])
            & (pl.col("language") == cell["language"])
            & (pl.col("prompt_class") == cell["prompt_class"])
            & (pl.col("alpha") == cell["alpha"])
        )
        .select([
            "record_id",
            pl.col("is_vetoed").alias("steered_is_vetoed"),
            pl.col("operational_failure").alias("steered_failure"),
        ])
    )

    if current.select("record_id").n_unique() != current.height:
        raise ValueError(f"Duplicate steered IDs in cell: {cell}")

    reference = baseline.filter(
        (pl.col("language") == cell["language"])
        & (pl.col("prompt_class") == cell["prompt_class"])
    )

    if reference.select("record_id").n_unique() != reference.height:
        raise ValueError(
            "Duplicate baseline IDs in language/prompt-class cell: "
            f"{cell['language']}, {cell['prompt_class']}"
        )

    paired = (
        reference
        .join(
            current,
            on="record_id",
            how="inner",
            validate="1:1",
        )
        .sort("record_id")
    )

    n = paired.height
    if n == 0:
        raise ValueError(f"No matched records for cell: {cell}")

    baseline_values = (
        paired["baseline_failure"]
        .cast(pl.Int8)
        .to_numpy()
    )
    steered_values = (
        paired["steered_failure"]
        .cast(pl.Int8)
        .to_numpy()
    )

    baseline_failure_n = int(baseline_values.sum())
    steered_failure_n = int(steered_values.sum())

    worsened_n = int(
        ((baseline_values == 0) & (steered_values == 1)).sum()
    )
    improved_n = int(
        ((baseline_values == 1) & (steered_values == 0)).sum()
    )
    both_safe_n = int(
        ((baseline_values == 0) & (steered_values == 0)).sum()
    )
    both_failure_n = int(
        ((baseline_values == 1) & (steered_values == 1)).sum()
    )
    discordant_n = worsened_n + improved_n

    if discordant_n == 0:
        p_value = 1.0
    else:
        p_value = float(
            binomtest(
                min(worsened_n, improved_n),
                n=discordant_n,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    paired_deltas = steered_values - baseline_values
    bootstrap_indices = rng.integers(
        0,
        n,
        size=(BOOTSTRAP_REPLICATES, n),
    )
    bootstrap_deltas_pp = (
        paired_deltas[bootstrap_indices].mean(axis=1) * 100
    )

    ci_low, ci_high = np.quantile(
        bootstrap_deltas_pp,
        [0.025, 0.975],
    )

    baseline_total_n = reference.height
    steered_total_n = current.height

    rows.append({
        "source_file_group": cell["source_file_group"],
        "baseline_config": "baseline",
        "steered_config": cell["config"],
        "language": cell["language"],
        "prompt_class": cell["prompt_class"],
        "alpha": cell["alpha"],
        "baseline_total_record_n": baseline_total_n,
        "steered_total_record_n": steered_total_n,
        "paired_record_n": n,
        "baseline_only_record_n": baseline_total_n - n,
        "steered_only_record_n": steered_total_n - n,
        "partial_baseline_coverage": n != baseline_total_n,
        "all_steered_ids_match_baseline": n == steered_total_n,
        "baseline_failure_n": baseline_failure_n,
        "steered_failure_n": steered_failure_n,
        "baseline_failure_rate_percentage": 100 * baseline_failure_n / n,
        "steered_failure_rate_percentage": 100 * steered_failure_n / n,
        "delta_steered_minus_baseline_percentage_points": (
            100 * (steered_failure_n - baseline_failure_n) / n
        ),
        "worsened_safe_to_failure_n": worsened_n,
        "improved_failure_to_safe_n": improved_n,
        "both_safe_n": both_safe_n,
        "both_failure_n": both_failure_n,
        "discordant_n": discordant_n,
        "mcnemar_exact_two_sided_p": p_value,
        "bootstrap_seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "ci95_low_percentage_points": float(ci_low),
        "ci95_high_percentage_points": float(ci_high),
        "ci_excludes_zero": bool(ci_high < 0 or ci_low > 0),
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

if result.height != 33:
    raise ValueError(f"Expected 33 steered cells; found {result.height}.")

if not result["all_steered_ids_match_baseline"].all():
    raise ValueError("At least one steered ID lacks a baseline match.")

if result.filter(
    pl.col("paired_record_n") != pl.col("steered_total_record_n")
).height:
    raise ValueError("Paired n does not equal steered n for at least one cell.")

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Inference rows: {result.height}")
print(
    "Partial baseline coverage rows: "
    f"{result.filter(pl.col('partial_baseline_coverage')).height}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=260):
    print(result)
