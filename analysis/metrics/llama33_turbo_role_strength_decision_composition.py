from pathlib import Path

import polars as pl

MAIN = "data/derived/canonical/llama33_phase1_main.parquet"
TURBO = "data/derived/canonical/llama33_phase2_turbo.parquet"
OUT = Path(
    "paper_artifacts/metrics/"
    "llama33_turbo_role_strength_decision_composition.csv"
)

LANGUAGES = ["Hausa", "Yoruba"]
CONFIGS = ["planner_only", "critic_only", "both"]
PROMPT_CLASS = "harmful"

KEEP = [
    "record_id",
    "language",
    "prompt_class",
    "config",
    "alpha",
    "parser_outcome",
    "operational_failure",
]

REQUIRED = set(KEEP)

main = pl.read_parquet(MAIN)
turbo = pl.read_parquet(TURBO)

for source_name, frame in [("phase1_main", main), ("phase2_turbo", turbo)]:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {sorted(missing)}"
        )

baseline = (
    main.filter(
        (pl.col("prompt_class") == PROMPT_CLASS)
        & pl.col("language").is_in(LANGUAGES)
        & pl.col("config").is_in(CONFIGS)
        & (pl.col("alpha") == 0.0)
    )
    .select(KEEP)
    .with_columns([
        pl.lit("phase1_main").alias("source_phase"),
        pl.lit("alpha0_baseline").alias("condition_type"),
    ])
)

steered = (
    turbo.filter(
        (pl.col("prompt_class") == PROMPT_CLASS)
        & pl.col("language").is_in(LANGUAGES)
        & pl.col("config").is_in(CONFIGS)
        & pl.col("alpha").is_in([20.0, 30.0])
    )
    .select(KEEP)
    .with_columns([
        pl.lit("phase2_turbo").alias("source_phase"),
        pl.lit("turbo_steered").alias("condition_type"),
    ])
)

combined = pl.concat(
    [baseline, steered],
    how="vertical",
    rechunk=True,
)

group_cols = [
    "source_phase",
    "condition_type",
    "config",
    "language",
    "alpha",
]

result = (
    combined.group_by(group_cols)
    .agg([
        pl.len().alias("harmful_n"),
        pl.col("record_id").n_unique().alias("unique_record_id_n"),
        (pl.col("parser_outcome") == "VETO_EXPLICIT")
        .sum()
        .cast(pl.Int64)
        .alias("explicit_veto_n"),
        (pl.col("parser_outcome") == "ALLOW_EXPLICIT")
        .sum()
        .cast(pl.Int64)
        .alias("explicit_allow_n"),
        (pl.col("parser_outcome") == "NO_DECISION")
        .sum()
        .cast(pl.Int64)
        .alias("no_decision_n"),
        pl.col("operational_failure")
        .cast(pl.Int64)
        .sum()
        .alias("operational_failure_n"),
    ])
    .with_columns([
        (
            pl.col("explicit_veto_n")
            + pl.col("explicit_allow_n")
            + pl.col("no_decision_n")
        ).alias("classified_outcome_n"),
        (
            100 * pl.col("explicit_veto_n") / pl.col("harmful_n")
        ).alias("explicit_veto_rate_percentage"),
        (
            100 * pl.col("explicit_allow_n") / pl.col("harmful_n")
        ).alias("explicit_allow_rate_percentage"),
        (
            100 * pl.col("no_decision_n") / pl.col("harmful_n")
        ).alias("no_decision_rate_percentage"),
        (
            100 * pl.col("operational_failure_n") / pl.col("harmful_n")
        ).alias("operational_failure_rate_percentage"),
    ])
    .with_columns([
        (
            pl.col("classified_outcome_n") == pl.col("harmful_n")
        ).alias("parser_outcomes_sum_to_harmful_n"),
        (
            pl.col("unique_record_id_n") == pl.col("harmful_n")
        ).alias("record_ids_are_unique"),
        (
            pl.col("operational_failure_n")
            == pl.col("explicit_allow_n") + pl.col("no_decision_n")
        ).alias("operational_failure_matches_allow_plus_no_decision"),
    ])
    .sort(["config", "language", "alpha", "source_phase"])
)

validation_cols = [
    "parser_outcomes_sum_to_harmful_n",
    "record_ids_are_unique",
    "operational_failure_matches_allow_plus_no_decision",
]

for column in validation_cols:
    if not result[column].all():
        failures = result.filter(~pl.col(column)).select(group_cols + [column])
        raise ValueError(
            f"Validation failed for {column}:\n{failures}"
        )

if result.height != 18:
    raise ValueError(
        f"Expected 18 rows (3 configs x 2 languages x 3 alphas); "
        f"found {result.height}."
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

with pl.Config(tbl_cols=-1, tbl_width_chars=280):
    print(result)

print(f"\nWrote: {OUT}")
