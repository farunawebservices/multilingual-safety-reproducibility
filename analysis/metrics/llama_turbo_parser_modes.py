from pathlib import Path

import polars as pl

MAIN = "data/derived/canonical/llama33_phase1_main.parquet"
TURBO = "data/derived/canonical/llama33_phase2_turbo.parquet"
OUT = Path("paper_artifacts/metrics/llama33_turbo_parser_modes.csv")

LANGUAGES = ["Hausa", "Yoruba"]
CONFIGS = ["planner_only", "critic_only", "both"]
OUTCOMES = ["VETO_EXPLICIT", "ALLOW_EXPLICIT", "NO_DECISION", "BOTH_LABELS"]

main = pl.read_parquet(MAIN)
turbo = pl.read_parquet(TURBO)

base = (
    main.filter(
        (pl.col("alpha") == 0.0)
        & (pl.col("prompt_class") == "harmful")
        & pl.col("language").is_in(LANGUAGES)
        & pl.col("config").is_in(CONFIGS)
    )
    .group_by(["config", "language"])
    .agg([
        pl.len().alias("baseline_alpha0_n"),
        (pl.col("parser_outcome") == "VETO_EXPLICIT")
        .sum()
        .alias("baseline_veto_explicit_n"),
        (pl.col("parser_outcome") == "ALLOW_EXPLICIT")
        .sum()
        .alias("baseline_allow_explicit_n"),
        (pl.col("parser_outcome") == "NO_DECISION")
        .sum()
        .alias("baseline_no_decision_n"),
        (pl.col("parser_outcome") == "BOTH_LABELS")
        .sum()
        .alias("baseline_both_labels_n"),
    ])
)

turbo_counts = (
    turbo.filter(
        (pl.col("prompt_class") == "harmful")
        & pl.col("language").is_in(LANGUAGES)
        & pl.col("config").is_in(CONFIGS)
        & pl.col("alpha").is_in([20.0, 30.0])
    )
    .group_by(["config", "alpha", "language"])
    .agg([
        pl.len().alias("turbo_n"),
        (pl.col("parser_outcome") == "VETO_EXPLICIT")
        .sum()
        .alias("veto_explicit_n"),
        (pl.col("parser_outcome") == "ALLOW_EXPLICIT")
        .sum()
        .alias("allow_explicit_n"),
        (pl.col("parser_outcome") == "NO_DECISION")
        .sum()
        .alias("no_decision_n"),
        (pl.col("parser_outcome") == "BOTH_LABELS")
        .sum()
        .alias("both_labels_n"),
    ])
    .join(base, on=["config", "language"], how="left", validate="m:1")
    .with_columns([
        (100 * pl.col("veto_explicit_n") / pl.col("turbo_n"))
        .alias("veto_explicit_rate_percentage"),
        (100 * pl.col("allow_explicit_n") / pl.col("turbo_n"))
        .alias("allow_explicit_rate_percentage"),
        (100 * pl.col("no_decision_n") / pl.col("turbo_n"))
        .alias("no_decision_rate_percentage"),
        (100 * pl.col("both_labels_n") / pl.col("turbo_n"))
        .alias("both_labels_rate_percentage"),
        (100 * pl.col("baseline_veto_explicit_n") / pl.col("baseline_alpha0_n"))
        .alias("baseline_veto_explicit_rate_percentage"),
        (100 * pl.col("baseline_allow_explicit_n") / pl.col("baseline_alpha0_n"))
        .alias("baseline_allow_explicit_rate_percentage"),
        (100 * pl.col("baseline_no_decision_n") / pl.col("baseline_alpha0_n"))
        .alias("baseline_no_decision_rate_percentage"),
        (100 * pl.col("baseline_both_labels_n") / pl.col("baseline_alpha0_n"))
        .alias("baseline_both_labels_rate_percentage"),
    ])
    .sort(["config", "alpha", "language"])
)

OUT.parent.mkdir(parents=True, exist_ok=True)
turbo_counts.write_csv(OUT)

print(
    turbo_counts.select([
        "config",
        "alpha",
        "language",
        "turbo_n",
        "veto_explicit_n",
        "allow_explicit_n",
        "no_decision_n",
        "both_labels_n",
        "veto_explicit_rate_percentage",
        "allow_explicit_rate_percentage",
        "no_decision_rate_percentage",
        "baseline_no_decision_rate_percentage",
    ])
)
print(f"\nWrote: {OUT}")
