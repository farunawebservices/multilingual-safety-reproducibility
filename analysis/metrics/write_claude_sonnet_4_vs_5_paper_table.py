from pathlib import Path

import polars as pl

INFILE = Path(
    "paper_artifacts/metrics/"
    "claude_sonnet_4_vs_5_outcome_comparison.csv"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "claude_sonnet_4_vs_5_paper_table.csv"
)

frame = pl.read_csv(INFILE)

required = [
    "language",
    "harmfulness",
    "shared_record_n",
    "sonnet_4_veto_n",
    "sonnet_5_veto_n",
    "sonnet_4_veto_rate_percentage",
    "sonnet_5_veto_rate_percentage",
    "sonnet_5_minus_sonnet_4_veto_rate_percentage_points",
    "decision_agreement_n",
    "decision_agreement_percentage",
]

missing = sorted(set(required) - set(frame.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if frame.height != 8:
    raise ValueError(f"Expected 8 language-by-harmfulness rows; found {frame.height}")

if not (
    frame["decision_agreement_n"] <= frame["shared_record_n"]
).all():
    raise ValueError("Agreement count exceeds shared record count.")

result = (
    frame
    .select([
        pl.lit("Claude Sonnet 4 vs Claude Sonnet 5")
        .alias("Comparison"),
        pl.col("language").alias("Language"),
        pl.col("harmfulness").alias("Prompt class"),
        pl.col("shared_record_n").alias("Matched n"),
        pl.col("sonnet_4_veto_n").alias("Sonnet 4 veto n"),
        pl.col("sonnet_4_veto_rate_percentage")
        .round(2)
        .alias("Sonnet 4 veto (%)"),
        pl.col("sonnet_5_veto_n").alias("Sonnet 5 veto n"),
        pl.col("sonnet_5_veto_rate_percentage")
        .round(2)
        .alias("Sonnet 5 veto (%)"),
        pl.col("sonnet_5_minus_sonnet_4_veto_rate_percentage_points")
        .round(2)
        .alias("Sonnet 5 minus Sonnet 4 (pp)"),
        pl.col("decision_agreement_n").alias("Decision agreement n"),
        pl.col("decision_agreement_percentage")
        .round(2)
        .alias("Decision agreement (%)"),
        pl.when(
            (pl.col("language") == "Yoruba")
            & (pl.col("harmfulness") == "benign")
        )
        .then(pl.lit("Partial cell: 45 prompts"))
        .otherwise(pl.lit("Complete cell"))
        .alias("Coverage"),
    ])
    .sort(["Prompt class", "Language"])
)

if result.filter(pl.col("Coverage") == "Partial cell: 45 prompts").height != 1:
    raise ValueError("Expected exactly one partial Yoruba-benign cell.")

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Paper-table rows: {result.height}")
print(
    "Partial-coverage rows: "
    f"{result.filter(pl.col('Coverage') == 'Partial cell: 45 prompts').height}"
)
print(
    "Overall matched prompts: "
    f"{result['Matched n'].sum()}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=260):
    print(result)
