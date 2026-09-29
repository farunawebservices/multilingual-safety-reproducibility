from pathlib import Path
import polars as pl

SOURCE = "data/derived/canonical/gpt_oss_20b_phase1_baseline.parquet"
OUT = Path("paper_artifacts/metrics/gpt_oss_20b_phase1_baseline_by_language.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pl.read_parquet(SOURCE)

metrics = (
    df.group_by(["model", "phase", "config", "language", "prompt_class"])
      .agg([
          pl.len().alias("attempted_n"),
          (pl.col("parser_outcome") == "VETO_EXPLICIT").sum().alias("explicit_veto_n"),
          (pl.col("parser_outcome") == "ALLOW_EXPLICIT").sum().alias("explicit_allow_n"),
          (pl.col("parser_outcome") == "BOTH_LABELS").sum().alias("both_labels_n"),
          (pl.col("parser_outcome") == "NO_DECISION").sum().alias("no_decision_n"),
          (pl.col("parser_outcome") == "NONSTRING_RESPONSE").sum().alias("nonstring_response_n"),
          pl.col("legacy_is_vetoed").sum().alias("legacy_veto_n"),
          pl.col("legacy_operational_failure").sum().alias("legacy_failure_n"),
          pl.col("operational_failure").sum().alias("strict_failure_n"),
      ])
      .with_columns([
          (pl.col("explicit_veto_n") / pl.col("attempted_n")).alias("explicit_veto_rate"),
          (pl.col("legacy_veto_n") / pl.col("attempted_n")).alias("legacy_veto_rate"),
          (pl.col("legacy_failure_n") / pl.col("attempted_n")).alias("legacy_failure_rate"),
          (pl.col("strict_failure_n") / pl.col("attempted_n")).alias("strict_failure_rate"),
      ])
      .sort(["language", "prompt_class"])
)

metrics.write_csv(OUT)
print(metrics)
print(f"\nWrote: {OUT}")
