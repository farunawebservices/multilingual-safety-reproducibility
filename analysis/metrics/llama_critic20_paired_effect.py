from pathlib import Path

import polars as pl
from scipy.stats import binomtest

SOURCE = "data/derived/canonical/llama33_phase1_main.parquet"
OUT = Path("paper_artifacts/metrics/llama33_critic20_paired_effect.csv")

LANGUAGES = ["Arabic", "English", "Hausa"]

df = pl.read_parquet(SOURCE)

base = (
    df.filter(
        (pl.col("config") == "critic_only")
        & (pl.col("alpha") == 0.0)
        & (pl.col("prompt_class") == "harmful")
        & (pl.col("language").is_in(LANGUAGES))
    )
    .select(["record_id", "language", "legacy_operational_failure"])
    .rename({"legacy_operational_failure": "baseline_failure"})
)

steered = (
    df.filter(
        (pl.col("config") == "critic_only")
        & (pl.col("alpha") == 20.0)
        & (pl.col("prompt_class") == "harmful")
        & (pl.col("language").is_in(LANGUAGES))
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

expected_n = 900
if paired.height != expected_n:
    raise SystemExit(
        f"Expected {expected_n} matched harmful prompts across "
        f"{LANGUAGES}, found {paired.height}."
    )

rows = []
for language in LANGUAGES:
    block = paired.filter(pl.col("language") == language)

    baseline = block["baseline_failure"].to_list()
    steered_values = block["steered_failure"].to_list()

    worsened = sum((not b) and s for b, s in zip(baseline, steered_values))
    improved = sum(b and (not s) for b, s in zip(baseline, steered_values))
    both_safe = sum((not b) and (not s) for b, s in zip(baseline, steered_values))
    both_failure = sum(b and s for b, s in zip(baseline, steered_values))
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
        "comparison": "critic_only_alpha0_vs_alpha20",
        "outcome_definition": "legacy_operational_failure",
        "language": language,
        "paired_harmful_n": block.height,
        "baseline_failure_n": sum(baseline),
        "steered_failure_n": sum(steered_values),
        "baseline_failure_rate": sum(baseline) / block.height,
        "steered_failure_rate": sum(steered_values) / block.height,
        "delta_steered_minus_baseline_percentage_points": (
            100 * (sum(steered_values) - sum(baseline)) / block.height
        ),
        "worsened_safe_to_failure_n": worsened,
        "improved_failure_to_safe_n": improved,
        "both_safe_n": both_safe,
        "both_failure_n": both_failure,
        "discordant_n": discordant,
        "mcnemar_exact_two_sided_p": p_value,
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
result = pl.DataFrame(rows)
result.write_csv(OUT)

print(
    result.select([
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
