from pathlib import Path

import numpy as np
import polars as pl

BASELINE = "data/derived/canonical/gpt_oss_20b_phase1_baseline.parquet"
STEERED = "data/derived/canonical/gpt_oss_20b_phase1_planner_only_alpha1.parquet"
OUT = Path("paper_artifacts/metrics/gpt_oss_20b_planner_alpha1_paired_bootstrap_ci.csv")

SEED = 20260908
N_BOOT = 10_000
ALPHA = 0.05
rng = np.random.default_rng(SEED)

base = (
    pl.read_parquet(BASELINE)
    .select(["record_id", "language", "prompt_class", "legacy_operational_failure"])
    .rename({"legacy_operational_failure": "baseline_failure"})
)

steer = (
    pl.read_parquet(STEERED)
    .select(["record_id", "language", "prompt_class", "legacy_operational_failure"])
    .rename({"legacy_operational_failure": "steered_failure"})
)

paired = (
    base.join(
        steer,
        on=["record_id", "language", "prompt_class"],
        how="inner",
        validate="1:1",
    )
    .filter(pl.col("prompt_class") == "harmful")
)

if paired.height != 1200:
    raise SystemExit(f"Expected 1200 paired harmful rows; found {paired.height}.")

rows = []

for language in sorted(paired["language"].unique().to_list()):
    block = paired.filter(pl.col("language") == language)

    baseline = (
        block.get_column("baseline_failure")
        .cast(pl.Int8)
        .to_numpy()
    )
    steered = (
        block.get_column("steered_failure")
        .cast(pl.Int8)
        .to_numpy()
    )

    n = len(baseline)

    if n != len(steered):
        raise SystemExit(f"Unpaired rows detected for {language}.")

    per_prompt_delta = steered - baseline
    point_delta = float(per_prompt_delta.mean())

    sample_indices = rng.integers(0, n, size=(N_BOOT, n))
    bootstrap_deltas = per_prompt_delta[sample_indices].mean(axis=1)

    low = float(np.quantile(bootstrap_deltas, ALPHA / 2))
    high = float(np.quantile(bootstrap_deltas, 1 - ALPHA / 2))

    rows.append({
        "model": "openai/gpt-oss-20b",
        "comparison": "baseline_vs_planner_only_alpha1",
        "metric": "legacy_harmful_failure_rate_difference",
        "language": language,
        "paired_harmful_n": n,
        "baseline_failure_n": int(baseline.sum()),
        "steered_failure_n": int(steered.sum()),
        "baseline_failure_rate": float(baseline.mean()),
        "steered_failure_rate": float(steered.mean()),
        "delta_steered_minus_baseline": point_delta,
        "delta_percentage_points": 100 * point_delta,
        "ci_method": "paired_nonparametric_bootstrap_percentile",
        "bootstrap_replicates": N_BOOT,
        "random_seed": SEED,
        "ci95_low": low,
        "ci95_high": high,
        "ci95_low_percentage_points": 100 * low,
        "ci95_high_percentage_points": 100 * high,
        "ci_excludes_zero": high < 0 or low > 0,
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
result = pl.DataFrame(rows).sort("language")
result.write_csv(OUT)

print(
    result.select([
        "language",
        "paired_harmful_n",
        "baseline_failure_n",
        "steered_failure_n",
        "delta_percentage_points",
        "ci95_low_percentage_points",
        "ci95_high_percentage_points",
        "ci_excludes_zero",
    ])
)
print(f"\nWrote: {OUT}")
