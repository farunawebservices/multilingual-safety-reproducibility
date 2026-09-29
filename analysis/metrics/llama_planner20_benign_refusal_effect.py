from pathlib import Path

import polars as pl
from scipy.stats import binomtest

SOURCE = "data/derived/canonical/llama33_phase1_main.parquet"
OUT = Path("paper_artifacts/metrics/llama33_planner20_benign_refusal_effect.csv")

df = pl.read_parquet(SOURCE)

base = (
    df.filter(
        (pl.col("config") == "planner_only") &
        (pl.col("alpha") == 0.0) &
        (pl.col("prompt_class") == "benign")
    )
    .select(["record_id", "language", "legacy_is_vetoed"])
    .rename({"legacy_is_vetoed": "baseline_refusal"})
)

steered = (
    df.filter(
        (pl.col("config") == "planner_only") &
        (pl.col("alpha") == 20.0) &
        (pl.col("prompt_class") == "benign")
    )
    .select(["record_id", "language", "legacy_is_vetoed"])
    .rename({"legacy_is_vetoed": "steered_refusal"})
)

paired = base.join(
    steered,
    on=["record_id", "language"],
    how="inner",
    validate="1:1",
)

if paired.height != 495:
    raise SystemExit(f"Expected 495 paired benign prompts; found {paired.height}.")

rows = []
for language in sorted(paired["language"].unique().to_list()):
    block = paired.filter(pl.col("language") == language)
    baseline = block["baseline_refusal"].to_list()
    steered = block["steered_refusal"].to_list()

    increased_refusal = sum((not b) and s for b, s in zip(baseline, steered))
    decreased_refusal = sum(b and (not s) for b, s in zip(baseline, steered))
    discordant = increased_refusal + decreased_refusal

    p_value = (
        binomtest(
            min(increased_refusal, decreased_refusal),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
        if discordant else 1.0
    )

    rows.append({
        "model": "meta-llama/llama-3.3-70b-instruct",
        "comparison": "planner_only_alpha0_vs_alpha20",
        "language": language,
        "paired_benign_n": block.height,
        "baseline_refusal_n": sum(baseline),
        "steered_refusal_n": sum(steered),
        "baseline_refusal_rate": sum(baseline) / block.height,
        "steered_refusal_rate": sum(steered) / block.height,
        "delta_refusal_percentage_points": (
            100 * (sum(steered) - sum(baseline)) / block.height
        ),
        "increased_refusal_n": increased_refusal,
        "decreased_refusal_n": decreased_refusal,
        "discordant_n": discordant,
        "mcnemar_exact_two_sided_p": p_value,
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
result = pl.DataFrame(rows).sort("language")
result.write_csv(OUT)
print(result)
print(f"\nWrote: {OUT}")
