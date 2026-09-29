from pathlib import Path

import numpy as np
import polars as pl
from statsmodels.stats.multitest import multipletests

BASELINE = "data/derived/canonical/llama33_phase1_main.parquet"
PAIRED_EFFECT = "paper_artifacts/metrics/llama33_planner20_paired_effect.csv"
OUT = Path("paper_artifacts/metrics/llama33_planner20_paired_inference.csv")

SEED = 20260908
N_BOOT = 10_000
rng = np.random.default_rng(SEED)

df = pl.read_parquet(BASELINE)

base = (
    df.filter(
        (pl.col("config") == "planner_only") &
        (pl.col("alpha") == 0.0) &
        (pl.col("prompt_class") == "harmful")
    )
    .select(["record_id", "language", "legacy_operational_failure"])
    .rename({"legacy_operational_failure": "baseline_failure"})
)

steered = (
    df.filter(
        (pl.col("config") == "planner_only") &
        (pl.col("alpha") == 20.0) &
        (pl.col("prompt_class") == "harmful")
    )
    .select(["record_id", "language", "legacy_operational_failure"])
    .rename({"legacy_operational_failure": "steered_failure"})
)

paired = base.join(
    steered,
    on=["record_id", "language"],
    how="inner",
    validate="1:1",
)

if paired.height != 1200:
    raise SystemExit(f"Expected 1200 matched harmful prompts, found {paired.height}.")

effects = pl.read_csv(PAIRED_EFFECT).sort("language")
p_values = effects["mcnemar_exact_two_sided_p"].to_list()
reject, p_holm, _, _ = multipletests(p_values, alpha=0.05, method="holm")

inference_rows = []
for language in effects["language"].to_list():
    block = paired.filter(pl.col("language") == language)

    baseline = block["baseline_failure"].cast(pl.Int8).to_numpy()
    steered = block["steered_failure"].cast(pl.Int8).to_numpy()
    delta = steered - baseline

    n = len(delta)
    indices = rng.integers(0, n, size=(N_BOOT, n))
    boot = delta[indices].mean(axis=1)

    inference_rows.append({
        "language": language,
        "paired_harmful_n": n,
        "delta_steered_minus_baseline": float(delta.mean()),
        "delta_percentage_points": float(100 * delta.mean()),
        "ci_method": "paired_nonparametric_bootstrap_percentile",
        "bootstrap_replicates": N_BOOT,
        "random_seed": SEED,
        "ci95_low_percentage_points": float(100 * np.quantile(boot, 0.025)),
        "ci95_high_percentage_points": float(100 * np.quantile(boot, 0.975)),
        "ci_excludes_zero": bool(
            np.quantile(boot, 0.975) < 0 or np.quantile(boot, 0.025) > 0
        ),
    })

inference = pl.DataFrame(inference_rows).sort("language")

result = (
    effects
    .join(inference, on=["language", "paired_harmful_n"], how="inner", validate="1:1")
    .with_columns([
        pl.Series("mcnemar_holm_adjusted_p", p_holm),
        pl.Series("mcnemar_holm_reject_alpha_0_05", reject),
    ])
    .sort("language")
)

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(
    result.select([
        "language",
        "baseline_failure_n",
        "steered_failure_n",
        "delta_percentage_points",
        "ci95_low_percentage_points",
        "ci95_high_percentage_points",
        "mcnemar_exact_two_sided_p",
        "mcnemar_holm_adjusted_p",
        "mcnemar_holm_reject_alpha_0_05",
    ])
)
print(f"\nWrote: {OUT}")
