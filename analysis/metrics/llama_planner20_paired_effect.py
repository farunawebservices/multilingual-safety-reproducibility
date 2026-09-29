from pathlib import Path

import polars as pl
from scipy.stats import binomtest

SOURCE = "data/derived/canonical/llama33_phase1_main.parquet"
OUT = Path("paper_artifacts/metrics/llama33_planner20_paired_effect.csv")

df = pl.read_parquet(SOURCE)

base = (
    df.filter(
        (pl.col("config") == "planner_only") &
        (pl.col("alpha") == 0.0)
    )
    .select(["record_id", "language", "prompt_class", "legacy_operational_failure"])
    .rename({"legacy_operational_failure": "baseline_failure"})
)

steered = (
    df.filter(
        (pl.col("config") == "planner_only") &
        (pl.col("alpha") == 20.0)
    )
    .select(["record_id", "language", "prompt_class", "legacy_operational_failure"])
    .rename({"legacy_operational_failure": "steered_failure"})
)

paired = (
    base.join(
        steered,
        on=["record_id", "language", "prompt_class"],
        how="inner",
        validate="1:1",
    )
    .filter(pl.col("prompt_class") == "harmful")
)

if paired.height != 1200:
    raise SystemExit(f"Expected 1200 paired harmful prompts; got {paired.height}.")

rows = []
for language in sorted(paired["language"].unique().to_list()):
    block = paired.filter(pl.col("language") == language)
    base_fail = block["baseline_failure"].to_list()
    steer_fail = block["steered_failure"].to_list()

    worsened = sum((not b) and s for b, s in zip(base_fail, steer_fail))
    improved = sum(b and (not s) for b, s in zip(base_fail, steer_fail))
    both_safe = sum((not b) and (not s) for b, s in zip(base_fail, steer_fail))
    both_failure = sum(b and s for b, s in zip(base_fail, steer_fail))
    discordant = worsened + improved

    p_value = (
        binomtest(min(worsened, improved), n=discordant, p=0.5).pvalue
        if discordant else 1.0
    )

    rows.append({
        "model": "meta-llama/llama-3.3-70b-instruct",
        "comparison": "planner_only_alpha0_vs_alpha20",
        "language": language,
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
result = pl.DataFrame(rows).sort("language")
result.write_csv(OUT)
print(result)
print(f"\nWrote: {OUT}")
