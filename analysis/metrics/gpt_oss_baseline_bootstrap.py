from pathlib import Path
import numpy as np
import polars as pl

SOURCE = "data/derived/canonical/gpt_oss_20b_phase1_baseline.parquet"
OUT = Path("paper_artifacts/metrics/gpt_oss_20b_phase1_baseline_bootstrap_ci.csv")

SEED = 20260908
N_BOOT = 10_000
RNG = np.random.default_rng(SEED)

df = (
    pl.read_parquet(SOURCE)
    .filter(pl.col("prompt_class") == "harmful")
    .select(["language", "legacy_operational_failure"])
)

rows = []
for language in sorted(df["language"].unique().to_list()):
    values = (
        df.filter(pl.col("language") == language)
        .get_column("legacy_operational_failure")
        .cast(pl.Int8)
        .to_numpy()
    )

    n = len(values)
    point = float(values.mean())

    indices = RNG.integers(0, n, size=(N_BOOT, n))
    boot_rates = values[indices].mean(axis=1)

    rows.append({
        "model": "openai/gpt-oss-20b",
        "phase": "phase1",
        "config": "baseline",
        "metric": "legacy_harmful_failure_rate",
        "language": language,
        "numerator": int(values.sum()),
        "denominator": n,
        "rate": point,
        "ci_method": "nonparametric_bootstrap_percentile",
        "bootstrap_replicates": N_BOOT,
        "random_seed": SEED,
        "ci95_low": float(np.quantile(boot_rates, 0.025)),
        "ci95_high": float(np.quantile(boot_rates, 0.975)),
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
pl.DataFrame(rows).sort("language").write_csv(OUT)
print(pl.read_csv(OUT))
print(f"\nWrote: {OUT}")
