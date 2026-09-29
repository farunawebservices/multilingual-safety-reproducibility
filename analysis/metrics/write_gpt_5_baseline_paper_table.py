from pathlib import Path

import polars as pl

INFILE = Path(
    "paper_artifacts/metrics/"
    "gpt_5_clean_frontier_baseline_audit.csv"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "gpt_5_frontier_baseline_paper_table.csv"
)

frame = pl.read_csv(INFILE)

required = [
    "language",
    "harmfulness",
    "record_n",
    "unique_id_n",
    "veto_n",
    "non_veto_n",
    "veto_rate_percentage",
    "non_veto_failure_rate_percentage",
]

missing = sorted(set(required) - set(frame.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if frame.height != 8:
    raise ValueError(f"Expected 8 language-by-harmfulness rows; found {frame.height}")

if not (frame["record_n"] == frame["unique_id_n"]).all():
    raise ValueError("Each cell must have one unique ID per record.")

result = (
    frame
    .select([
        pl.lit("GPT-5 frontier baseline")
        .alias("Condition"),
        pl.col("language").alias("Language"),
        pl.col("harmfulness").alias("Prompt class"),
        pl.col("record_n").alias("Valid n"),
        pl.col("veto_n").alias("Veto n"),
        pl.col("non_veto_n").alias("Non-veto n"),
        pl.col("veto_rate_percentage")
        .round(2)
        .alias("Veto rate (%)"),
        pl.col("non_veto_failure_rate_percentage")
        .round(2)
        .alias("Non-veto operational failure (%)"),
        pl.when(
            (pl.col("harmfulness") == "benign")
            & (pl.col("language") == "Yoruba")
        )
        .then(pl.lit("Partial: 45 prompts"))
        .when(
            (pl.col("harmfulness") == "harmful")
            & (pl.col("record_n") < 300)
        )
        .then(
            pl.concat_str([
                pl.lit("Partial: "),
                pl.col("record_n").cast(pl.String),
                pl.lit(" prompts"),
            ])
        )
        .otherwise(pl.lit("Complete cell"))
        .alias("Coverage"),
    ])
    .sort(["Prompt class", "Language"])
)

partial_n = result.filter(
    pl.col("Coverage") != "Complete cell"
).height

if partial_n != 5:
    raise ValueError(
        f"Expected 5 partial-coverage rows; found {partial_n}."
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Paper-table rows: {result.height}")
print(f"Total valid records: {result['Valid n'].sum()}")
print(f"Partial-coverage rows: {partial_n}")
print(
    "Harmful partial-coverage rows: "
    f"{result.filter(pl.col('Prompt class') == 'harmful').filter(pl.col('Coverage') != 'Complete cell').height}"
)
print("Comparison or inference claim included: False")

with pl.Config(tbl_cols=-1, tbl_width_chars=240):
    print(result)
