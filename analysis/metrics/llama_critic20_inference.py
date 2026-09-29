from pathlib import Path

import numpy as np
import polars as pl
from statsmodels.stats.multitest import multipletests

SOURCE = "data/derived/canonical/llama33_phase1_main.parquet"
EFFECTS = "paper_artifacts/metrics/llama33_critic20_paired_effect.csv"
OUT = Path("paper_artifacts/metrics/llama33_critic20_paired_inference.csv")

LANGUAGES = ["Arabic", "English", "Hausa"]
SEED = 20260908
N_BOOT = 10_000
rng = np.random.default_rng(SEED)

df = pl.read_parquet(SOURCE)

base = (
    df.filter(
        (pl.col("config") == "critic_only")
        & (pl.col("alpha") == 0.0)
        & (pl.col("prompt_class") == "harmful")
        & pl.col("language").is_in(LANGUAGES)
    )
    .select(["record_id", "language", "legacy_operational_failure"])
    .rename({"legacy_operational_failure": "baseline_failure"})
)

steered = (
    df.filter(
        (pl.col("config") == "critic_only")
        & (pl.col("alpha") == 20.0)
        & (pl.col("prompt_class") == "harmful")
        & pl.col("language").is_in(LANGUAGES)
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

if paired.height != 900:
    raise SystemExit(f"Expected 900 paired harmful prompts; found {paired.height}.")

effects = pl.read_csv(EFFECTS).sort("language")
p_values = effects["mcnemar_exact_two_sided_p"].to_list()
reject, p_holm, _, _ = multipletests(p_values, alpha=0.05, method="holm")

rows = []
for language in LANGUAGES:
    block = paired.filter(pl.col("language") == language)

    baseline = block["baseline_failure"].cast(pl.Int8).to_numpy()
    steered_values = block["steered_failure"].cast(pl.Int8).to_numpy()
    delta = steered_values - baseline

    n = len(delta)
    sample_indices = rng.integers(0, n, size=(N_BOOT, n))
    bootstrap_delta = delta[sample_indices].mean(axis=1)

    low = float(np.quantile(bootstrap_delta, 0.025))
    high = float(np.quantile(bootstrap_delta, 0.975))

    rows.append({
        "language": language,
        "paired_harmful_n": n,
        "delta_steered_minus_baseline_percentage_points": float(100 * delta.mean()),
        "ci_method": "paired_nonparametric_bootstrap_percentile",
        "bootstrap_replicates": N_BOOT,
        "random_seed": SEED,
        "ci95_low_percentage_points": float(100 * low),
        "ci95_high_percentage_points": float(100 * high),
        "ci_excludes_zero": bool(high < 0 or low > 0),
    })

inference = pl.DataFrame(rows).sort("language")

result = (
    effects.join(
        inference,
        on=["language", "paired_harmful_n", "delta_steered_minus_baseline_percentage_points"],
        how="inner",
        validate="1:1",
    )
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
        "delta_steered_minus_baseline_percentage_points",
        "ci95_low_percentage_points",
        "ci95_high_percentage_points",
        "mcnemar_exact_two_sided_p",
        "mcnemar_holm_adjusted_p",
        "mcnemar_holm_reject_alpha_0_05",
    ])
)
print(f"\nWrote: {OUT}")
