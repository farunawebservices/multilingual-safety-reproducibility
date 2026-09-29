from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import binomtest

INFILE = Path(
    "data/derived/canonical/gemma_4_31b_phase1_complete_case.parquet"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "gemma_4_31b_phase1_complete_case_paired_inference.csv"
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
        pl.col("operational_failure").alias("baseline_failure"),
    ])
)

steered = frame.filter(
    (pl.col("source_file_group") == "steered")
    & (pl.col("alpha") > 0.0)
)

rng = np.random.default_rng(SEED)
rows = []

cells = (
    steered
    .select(["config", "language", "prompt_class", "alpha"])
    .unique()
    .sort(["prompt_class", "language", "config", "alpha"])
    .iter_rows(named=True)
)

for cell in cells:
    current = (
        steered
        .filter(
            (pl.col("config") == cell["config"])
            & (pl.col("language") == cell["language"])
            & (pl.col("prompt_class") == cell["prompt_class"])
            & (pl.col("alpha") == cell["alpha"])
        )
        .select([
            "record_id",
            pl.col("operational_failure").alias("steered_failure"),
        ])
    )

    reference = baseline.filter(
        (pl.col("language") == cell["language"])
        & (pl.col("prompt_class") == cell["prompt_class"])
    )

    paired = (
        reference
        .join(current, on="record_id", how="inner", validate="1:1")
        .sort("record_id")
    )

    n = paired.height
    if n == 0:
        raise ValueError(f"No matched records for {cell}")

    baseline_values = paired["baseline_failure"].cast(pl.Int8).to_numpy()
    steered_values = paired["steered_failure"].cast(pl.Int8).to_numpy()

    worsened = int(((baseline_values == 0) & (steered_values == 1)).sum())
    improved = int(((baseline_values == 1) & (steered_values == 0)).sum())
    discordant = worsened + improved

    p_value = (
        1.0 if discordant == 0
        else float(
            binomtest(
                min(worsened, improved),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    )

    deltas = steered_values - baseline_values
    indices = rng.integers(
        0, n, size=(BOOTSTRAP_REPLICATES, n)
    )
    bootstrap_pp = deltas[indices].mean(axis=1) * 100
    ci_low, ci_high = np.quantile(bootstrap_pp, [0.025, 0.975])

    baseline_total = reference.height
    steered_total = current.height

    rows.append({
        "model": "gemma_4_31b",
        "analysis_population": "phase1_complete_case",
        "steered_config": cell["config"],
        "language": cell["language"],
        "prompt_class": cell["prompt_class"],
        "alpha": cell["alpha"],
        "baseline_valid_record_n": baseline_total,
        "steered_valid_record_n": steered_total,
        "paired_record_n": n,
        "baseline_only_valid_record_n": baseline_total - n,
        "steered_only_valid_record_n": steered_total - n,
        "partial_coverage": (
            n != baseline_total or n != steered_total
        ),
        "baseline_failure_n": int(baseline_values.sum()),
        "steered_failure_n": int(steered_values.sum()),
        "baseline_failure_rate_percentage": (
            100 * baseline_values.mean()
        ),
        "steered_failure_rate_percentage": (
            100 * steered_values.mean()
        ),
        "delta_steered_minus_baseline_percentage_points": (
            100 * deltas.mean()
        ),
        "worsened_safe_to_failure_n": worsened,
        "improved_failure_to_safe_n": improved,
        "both_safe_n": int(
            ((baseline_values == 0) & (steered_values == 0)).sum()
        ),
        "both_failure_n": int(
            ((baseline_values == 1) & (steered_values == 1)).sum()
        ),
        "discordant_n": discordant,
        "mcnemar_exact_two_sided_p": p_value,
        "bootstrap_seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "ci95_low_percentage_points": float(ci_low),
        "ci95_high_percentage_points": float(ci_high),
        "ci_excludes_zero": bool(ci_high < 0 or ci_low > 0),
        "source_truncation_caveat": (
            "One truncated terminal JSONL record excluded from each source; "
            "missing outcomes were not imputed."
        ),
    })

result = (
    pl.DataFrame(rows)
    .sort(["prompt_class", "language", "steered_config", "alpha"])
)

if result.height != 3:
    raise ValueError(f"Expected 3 comparison cells; found {result.height}")

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Inference rows: {result.height}")
print(
    "Partial-coverage rows: "
    f"{result.filter(pl.col('partial_coverage')).height}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=240):
    print(result)
