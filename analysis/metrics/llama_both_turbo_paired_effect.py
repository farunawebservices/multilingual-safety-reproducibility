from pathlib import Path

import polars as pl
from scipy.stats import binomtest

MAIN = "data/derived/canonical/llama33_phase1_main.parquet"
TURBO = "data/derived/canonical/llama33_phase2_turbo.parquet"
OUT = Path("paper_artifacts/metrics/llama33_both_turbo_paired_effect.csv")

LANGUAGES = ["Hausa", "Yoruba"]
ALPHAS = [20.0, 30.0]

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
        raise SystemExit(
            f"Expected 600 paired rows for both alpha {alpha}; found {paired.height}."
        )

    for language in LANGUAGES:
        block = paired.filter(pl.col("language") == language)
        base_fail = block["baseline_failure"].to_list()
        steer_fail = block["steered_failure"].to_list()

        worsened = sum((not b) and s for b, s in zip(base_fail, steer_fail))
        improved = sum(b and (not s) for b, s in zip(base_fail, steer_fail))
        both_safe = sum((not b) and (not s) for b, s in zip(base_fail, steer_fail))
        both_failure = sum(b and s for b, s in zip(base_fail, steer_fail))
        discordant = worsened + improved

        p_value = (
            binomtest(
                min(worsened, improved),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
            if discordant else 1.0
        )

        rows.append({
            "model": "meta-llama/llama-3.3-70b-instruct",
            "comparison": "both_alpha0_vs_turbo_alpha",
            "outcome_definition": "legacy_operational_failure",
            "language": language,
            "alpha": alpha,
            "paired_harmful_n": block.height,
            "baseline_failure_n": sum(base_fail),
            "steered_failure_n": sum(steer_fail),
            "baseline_failure_rate": sum(base_fail) / block.height,
            "steered_failure_rate": sum(steer_fail) / block.height,
            "delta_steered_minus_baseline_percentage_points": (
                100 * (sum(steer_fail) - sum(base_fail)) / block.height
            ),
            "worsened_safe_to_failure_n": worsened,
            "improved_failure_to_safe_n": improved,
            "both_safe_n": both_safe,
            "both_failure_n": both_failure,
            "discordant_n": discordant,
            "mcnemar_exact_two_sided_p": p_value,
        })

OUT.parent.mkdir(parents=True, exist_ok=True)
result = pl.DataFrame(rows).sort(["alpha", "language"])
result.write_csv(OUT)

print(
    result.select([
        "alpha",
        "language",
        "paired_harmful_n",
        "baseline_failure_n",
        "steered_failure_n",
        "delta_steered_minus_baseline_percentage_points",
        "worsened_safe_to_failure_n",
        "improved_failure_to_safe_n",
        "discordant_n",
        "mcnemar_exact_two_sided_p",
    ])
)
print(f"\nWrote: {OUT}")
