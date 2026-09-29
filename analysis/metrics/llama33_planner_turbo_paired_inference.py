from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

MAIN = "data/derived/canonical/llama33_phase1_main.parquet"
TURBO = "data/derived/canonical/llama33_phase2_turbo.parquet"
OUT = Path("paper_artifacts/metrics/llama33_planner_turbo_paired_inference.csv")

LANGUAGES = ["Hausa", "Yoruba"]
ALPHAS = [20.0, 30.0]
CONFIG = "planner_only"
PROMPT_CLASS = "harmful"
BOOTSTRAP_REPLICATES = 10_000
RANDOM_SEED = 20260908

required = {
    "record_id",
    "language",
    "prompt_class",
    "config",
    "alpha",
    "operational_failure",
}

main = pl.read_parquet(MAIN)
turbo = pl.read_parquet(TURBO)

for name, frame in [("main", main), ("turbo", turbo)]:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

baseline_source = (
    main.filter(
        (pl.col("config") == CONFIG)
        & (pl.col("prompt_class") == PROMPT_CLASS)
        & (pl.col("alpha") == 0.0)
        & pl.col("language").is_in(LANGUAGES)
    )
    .select([
        "record_id",
        "language",
        pl.col("operational_failure").cast(pl.Int8).alias("baseline_failure"),
    ])
)

steered_source = (
    turbo.filter(
        (pl.col("config") == CONFIG)
        & (pl.col("prompt_class") == PROMPT_CLASS)
        & pl.col("language").is_in(LANGUAGES)
        & pl.col("alpha").is_in(ALPHAS)
    )
    .select([
        "record_id",
        "language",
        "alpha",
        pl.col("operational_failure").cast(pl.Int8).alias("steered_failure"),
    ])
)

rng = np.random.default_rng(RANDOM_SEED)
rows = []

for language in LANGUAGES:
    baseline = (
        baseline_source
        .filter(pl.col("language") == language)
        .select(["record_id", "baseline_failure"])
    )

    for alpha in ALPHAS:
        steered = (
            steered_source
            .filter(
                (pl.col("language") == language)
                & (pl.col("alpha") == alpha)
            )
            .select(["record_id", "steered_failure"])
        )

        if baseline["record_id"].n_unique() != baseline.height:
            raise ValueError(f"Duplicate baseline record_id values: {language}")

        if steered["record_id"].n_unique() != steered.height:
            raise ValueError(f"Duplicate steered record_id values: {language}, alpha={alpha}")

        paired = baseline.join(
            steered,
            on="record_id",
            how="full",
            coalesce=True,
        )

        unmatched = paired.filter(
            pl.col("baseline_failure").is_null()
            | pl.col("steered_failure").is_null()
        )

        if unmatched.height > 0:
            raise ValueError(
                f"Unmatched record IDs for {language}, alpha={alpha}: "
                f"{unmatched.height}; baseline={baseline.height}; "
                f"steered={steered.height}"
            )

        paired = paired.select([
            "record_id",
            "baseline_failure",
            "steered_failure",
        ])

        base = paired["baseline_failure"].to_numpy()
        steer = paired["steered_failure"].to_numpy()
        paired_n = paired.height

        improved = int(((base == 1) & (steer == 0)).sum())
        worsened = int(((base == 0) & (steer == 1)).sum())
        both_safe = int(((base == 0) & (steer == 0)).sum())
        both_failure = int(((base == 1) & (steer == 1)).sum())
        discordant_n = improved + worsened

        p_value = (
            1.0
            if discordant_n == 0
            else binomtest(
                min(improved, worsened),
                n=discordant_n,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

        point_delta_pp = 100 * (steer.mean() - base.mean())

        indices = rng.integers(
            low=0,
            high=paired_n,
            size=(BOOTSTRAP_REPLICATES, paired_n),
        )

        bootstrap_delta_pp = 100 * (
            steer[indices].mean(axis=1) - base[indices].mean(axis=1)
        )

        ci_low, ci_high = np.quantile(
            bootstrap_delta_pp,
            [0.025, 0.975],
        )

        rows.append({
            "model": "llama33",
            "comparison": "planner_only_turbo_alpha_vs_phase1_alpha0",
            "outcome_definition": "harmful_operational_failure",
            "config": CONFIG,
            "language": language,
            "alpha": alpha,
            "paired_harmful_n": paired_n,
            "baseline_failure_n": int(base.sum()),
            "steered_failure_n": int(steer.sum()),
            "baseline_failure_rate": 100 * base.mean(),
            "steered_failure_rate": 100 * steer.mean(),
            "delta_steered_minus_baseline_percentage_points": point_delta_pp,
            "worsened_safe_to_failure_n": worsened,
            "improved_failure_to_safe_n": improved,
            "both_safe_n": both_safe,
            "both_failure_n": both_failure,
            "discordant_n": discordant_n,
            "mcnemar_exact_two_sided_p": p_value,
            "ci_method": "paired_nonparametric_bootstrap_percentile",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "random_seed": RANDOM_SEED,
            "ci95_low_percentage_points": float(ci_low),
            "ci95_high_percentage_points": float(ci_high),
            "ci_excludes_zero": bool(ci_high < 0 or ci_low > 0),
        })

result = pl.DataFrame(rows).sort(["alpha", "language"])

reject, adjusted_p, _, _ = multipletests(
    result["mcnemar_exact_two_sided_p"].to_numpy(),
    alpha=0.05,
    method="holm",
)

result = result.with_columns([
    pl.Series("mcnemar_holm_adjusted_p", adjusted_p),
    pl.Series("mcnemar_holm_reject_alpha_0_05", reject),
])

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

with pl.Config(tbl_cols=-1, tbl_width_chars=250):
    print(result)

print(f"\nWrote: {OUT}")
