from pathlib import Path
import polars as pl
from scipy.stats import binomtest

BASELINE = "data/derived/canonical/gpt_oss_20b_phase1_baseline.parquet"
STEERED = "data/derived/canonical/gpt_oss_20b_phase1_planner_only_alpha1.parquet"
OUT = Path("paper_artifacts/metrics/gpt_oss_20b_planner_alpha1_paired_effect.csv")

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

joined = base.join(
    steer,
    on=["record_id", "language", "prompt_class"],
    how="inner",
    validate="1:1",
).filter(pl.col("prompt_class") == "harmful")

if joined.height != 1200:
    raise SystemExit(f"Expected 1200 paired harmful rows, found {joined.height}.")

rows = []
for language in sorted(joined["language"].unique().to_list()):
    block = joined.filter(pl.col("language") == language)

    base_fail = block["baseline_failure"].to_list()
    steer_fail = block["steered_failure"].to_list()

    both_safe = sum((not b) and (not s) for b, s in zip(base_fail, steer_fail))
    worsened = sum((not b) and s for b, s in zip(base_fail, steer_fail))
    improved = sum(b and (not s) for b, s in zip(base_fail, steer_fail))
    both_fail = sum(b and s for b, s in zip(base_fail, steer_fail))

    discordant = worsened + improved
    p_value = (
        binomtest(min(worsened, improved), n=discordant, p=0.5, alternative="two-sided").pvalue
        if discordant else 1.0
    )

    rows.append({
        "model": "openai/gpt-oss-20b",
        "comparison": "baseline_vs_planner_only_alpha1",
        "language": language,
        "paired_harmful_n": block.height,
        "baseline_failure_n": sum(base_fail),
        "steered_failure_n": sum(steer_fail),
        "baseline_failure_rate": sum(base_fail) / block.height,
        "steered_failure_rate": sum(steer_fail) / block.height,
        "absolute_change_percentage_points": 100 * (
            sum(steer_fail) - sum(base_fail)
        ) / block.height,
        "both_safe_n": both_safe,
        "worsened_safe_to_failure_n": worsened,
        "improved_failure_to_safe_n": improved,
        "both_failure_n": both_fail,
        "discordant_n": discordant,
        "mcnemar_exact_two_sided_p": p_value,
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
result = pl.DataFrame(rows).sort("language")
result.write_csv(OUT)
print(result)
print(f"\nWrote: {OUT}")
