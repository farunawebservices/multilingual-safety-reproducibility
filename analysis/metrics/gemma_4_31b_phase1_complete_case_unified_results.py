from pathlib import Path

import polars as pl

INFERENCE = Path(
    "paper_artifacts/metrics/"
    "gemma_4_31b_phase1_complete_case_paired_inference_holm.csv"
)

SEMANTICS = Path(
    "paper_artifacts/metrics/"
    "gemma_4_31b_phase1_complete_case_decision_semantics_audit.csv"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "gemma_4_31b_phase1_complete_case_unified_results.csv"
)

inference = pl.read_csv(INFERENCE)
semantics = pl.read_csv(SEMANTICS)

if inference.height != 3:
    raise ValueError(f"Expected 3 inference rows; found {inference.height}")

baseline = (
    semantics
    .filter(
        (pl.col("source_file_group") == "baseline")
        & (pl.col("config") == "baseline")
        & (pl.col("alpha") == 0.0)
    )
    .select([
        "language",
        "prompt_class",
        pl.col("record_n").alias("baseline_composition_record_n"),
        pl.col("is_vetoed_true_n").alias("baseline_veto_explicit_n"),
        pl.col("is_vetoed_false_n").alias("baseline_non_veto_n"),
        pl.col("critic_allow_token_n").alias("baseline_allow_token_n"),
        pl.col("veto_boolean_matches_critic_veto_token")
        .alias("baseline_veto_marker_valid"),
        pl.col("failure_matches_non_veto_boolean")
        .alias("baseline_failure_definition_valid"),
    ])
)

steered = (
    semantics
    .filter(
        (pl.col("source_file_group") == "steered")
        & (pl.col("alpha") > 0.0)
    )
    .select([
        pl.col("config").alias("steered_config"),
        "language",
        "prompt_class",
        "alpha",
        pl.col("record_n").alias("steered_composition_record_n"),
        pl.col("is_vetoed_true_n").alias("steered_veto_explicit_n"),
        pl.col("is_vetoed_false_n").alias("steered_non_veto_n"),
        pl.col("critic_allow_token_n").alias("steered_allow_token_n"),
        pl.col("veto_boolean_matches_critic_veto_token")
        .alias("steered_veto_marker_valid"),
        pl.col("failure_matches_non_veto_boolean")
        .alias("steered_failure_definition_valid"),
    ])
)

result = (
    inference
    .join(
        baseline,
        on=["language", "prompt_class"],
        how="inner",
        validate="m:1",
    )
    .join(
        steered,
        on=["steered_config", "language", "prompt_class", "alpha"],
        how="inner",
        validate="1:1",
    )
    .with_columns([
        pl.lit(
            "is_vetoed=false denotes non-veto operational failure; "
            "non-veto is not assumed to be explicit ALLOW."
        ).alias("outcome_caveat"),
        pl.when(pl.col("partial_coverage"))
        .then(pl.lit(
            "Matched subset only: 101 shared valid IDs; "
            "11 steered-only valid IDs."
        ))
        .otherwise(pl.lit(
            "Complete matching among valid JSON records."
        ))
        .alias("pairing_coverage_note"),
    ])
    .sort(["prompt_class", "language", "steered_config", "alpha"])
)

if result.height != 3:
    raise ValueError(f"Expected 3 unified rows; found {result.height}")

for column in [
    "baseline_veto_marker_valid",
    "baseline_failure_definition_valid",
    "steered_veto_marker_valid",
    "steered_failure_definition_valid",
]:
    if not result[column].all():
        raise ValueError(f"Semantic validation failed: {column}")

if not (
    result["paired_record_n"] <= result["baseline_valid_record_n"]
).all():
    raise ValueError("Paired n exceeds baseline valid n.")

if not (
    result["paired_record_n"] <= result["steered_valid_record_n"]
).all():
    raise ValueError("Paired n exceeds steered valid n.")

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Unified rows: {result.height}")
print(
    "Partial-coverage rows: "
    f"{result.filter(pl.col('partial_coverage')).height}"
)
print(
    "Holm-significant rows: "
    f"{result.filter(pl.col('mcnemar_holm_reject_alpha_0_05')).height}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=260):
    print(result)
