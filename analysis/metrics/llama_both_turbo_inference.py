from pathlib import Path

import numpy as np
import polars as pl
from statsmodels.stats.multitest import multipletests

MAIN = "data/derived/canonical/llama33_phase1_main.parquet"
TURBO = "data/derived/canonical/llama33_phase2_turbo.parquet"
EFFECTS = "paper_artifacts/metrics/llama33_both_turbo_paired_effect.csv"
OUT = Path("paper_artifacts/metrics/llama33_both_turbo_paired_inference.csv")

LANGUAGES = ["Hausa", "Yoruba"]
ALPHAS = [20.0, 30.0]
SEED = 20260908
N_BOOT = 10_000
rng = np.random.default_rng(SEED)

main = pl.read_parquet(MAIN)
turbo = pl.read_parquet(TURBO)

baseline = (
    main.filter(
        (pl.col("config") == "both")
        & (pl.col("alpha") == 0.0)
        & (pl.col("prompt_class") == "harmful")
        & pl.col("language").is_in(LANGUAGES)
    )
    .select(["record_id", "language", "legacy_operational_failure"])
    .rename({"legacy_operational_failure": "baseline_failure"})
)

effects = pl.read_csv(EFFECTS).sort(["alpha", "language"])
p_values = effects["mcnemar_exact_two_sided_p"].to_list()
reject, p_holm, _, _ = multipletests(p_values, alpha=0.05, method="holm")

rows = []
for alpha in ALPHAS:
    steered = (
        turbo.filter(
            (pl.col("config") == "both")
            & (pl.col("alpha") == alpha)
            & (pl.col("prompt_class") == "harmful")
            & pl.col("language").is_in(LANGUAGES)
        )
        .select(["record_id", "language", "legacy_operational_failure"])
        .rename({"legacy_operational_failure": "steered_failure"})
    )

    paired = baseline.join(
        steered,
        on=["record_id", "language"],
        how="inner",
        validate="1:1",
    )

    if paired.height != 600:
        raise SystemExit(f"Expected 600 paired rows at alpha {alpha}; got {paired.height}.")

    for language in LANGUAGES:
        block = paired.filter(pl.col("language") == language)
        base = block["baseline_failure"].cast(pl.Int8).to_numpy()
        steer = block["steered_failure"].cast(pl.Int8).to_numpy()

        delta = steer - base
        n = len(delta)

        indices = rng.integers(0, n, size=(N_BOOT, n))
        bootstrap_delta = delta[indices].mean(axis=1)

        low = float(np.quantile(bootstrap_delta, 0.025))
        high = float(np.quantile(bootstrap_delta, 0.975))

        rows.append({
            "alpha": alpha,
            "language": language,
            "paired_harmful_n": n,
            "delta_percentage_points": float(100 * delta.mean()),
            "ci_method": "paired_nonparametric_bootstrap_percentile",
            "bootstrap_replicates": N_BOOT,
            "random_seed": SEED,
            "ci95_low_percentage_points": float(100 * low),
            "ci95_high_percentage_points": float(100 * high),
            "ci_excludes_zero": bool(low > 0 or high < 0),
        })

inference = pl.DataFrame(rows).sort(["alpha", "language"])

result = (
    effects.join(
        inference,
        on=["alpha", "language", "paired_harmful_n"],
        how="inner",
        validate="1:1",
    )
    .with_columns([
        pl.Series("mcnemar_holm_adjusted_p", p_holm),
        pl.Series("mcnemar_holm_reject_alpha_0_05", reject),
    ])
    .sort(["alpha", "language"])
)

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(
    result.select([
        "alpha",
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
