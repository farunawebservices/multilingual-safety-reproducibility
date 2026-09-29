from pathlib import Path

import polars as pl

INVENTORY = Path(
    "paper_artifacts/metrics/closed_model_source_inventory.csv"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "claude_sonnet_4_vs_5_outcome_comparison.csv"
)

inventory = pl.read_csv(INVENTORY)

def source_for(model: str) -> Path:
    paths = (
        inventory
        .filter(
            (pl.col("model") == model)
            & (pl.col("file_type") == "jsonl")
            & (pl.col("location") == "raw")
        )
        .select("path")
        .to_series()
        .to_list()
    )

    if len(paths) != 1:
        raise ValueError(
            f"Expected exactly one raw JSONL for {model}; found {len(paths)}."
        )

    return Path(paths[0])

def load(model: str) -> pl.DataFrame:
    frame = pl.read_ndjson(source_for(model))

    required = ["id", "language", "harmfulness", "is_vetoed"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{model} missing fields: {missing}")

    if frame.select(pl.col("id").n_unique() != pl.len()).item():
        raise ValueError(f"{model} has duplicate IDs.")

    return frame.select([
        "id",
        "language",
        "harmfulness",
        pl.col("is_vetoed").alias(f"{model}_is_vetoed"),
    ])

sonnet4 = load("claude_sonnet_4")
sonnet5 = load("claude_sonnet_5")

paired = sonnet4.join(
    sonnet5,
    on=["id", "language", "harmfulness"],
    how="inner",
    validate="1:1",
)

if paired.height != 1695:
    raise ValueError(f"Expected 1695 shared rows; found {paired.height}")

result = (
    paired
    .group_by(["language", "harmfulness"])
    .agg([
        pl.len().alias("shared_record_n"),
        pl.col("claude_sonnet_4_is_vetoed")
        .sum()
        .alias("sonnet_4_veto_n"),
        pl.col("claude_sonnet_5_is_vetoed")
        .sum()
        .alias("sonnet_5_veto_n"),
        (
            pl.col("claude_sonnet_4_is_vetoed")
            == pl.col("claude_sonnet_5_is_vetoed")
        )
        .sum()
        .alias("decision_agreement_n"),
        (
            pl.col("claude_sonnet_4_is_vetoed")
            & (~pl.col("claude_sonnet_5_is_vetoed"))
        )
        .sum()
        .alias("sonnet_4_veto_sonnet_5_non_veto_n"),
        (
            (~pl.col("claude_sonnet_4_is_vetoed"))
            & pl.col("claude_sonnet_5_is_vetoed")
        )
        .sum()
        .alias("sonnet_4_non_veto_sonnet_5_veto_n"),
    ])
    .with_columns([
        (
            100.0 * pl.col("sonnet_4_veto_n")
            / pl.col("shared_record_n")
        ).alias("sonnet_4_veto_rate_percentage"),
        (
            100.0 * pl.col("sonnet_5_veto_n")
            / pl.col("shared_record_n")
        ).alias("sonnet_5_veto_rate_percentage"),
        (
            100.0 * (
                pl.col("sonnet_5_veto_n").cast(pl.Int64)
                - pl.col("sonnet_4_veto_n").cast(pl.Int64)
            )
            / pl.col("shared_record_n").cast(pl.Int64)
        ).alias("sonnet_5_minus_sonnet_4_veto_rate_percentage_points"),
        (
            100.0 * pl.col("decision_agreement_n")
            / pl.col("shared_record_n")
        ).alias("decision_agreement_percentage"),
    ])
    .sort(["harmfulness", "language"])
)

if not (
    result["decision_agreement_n"]
    + result["sonnet_4_veto_sonnet_5_non_veto_n"]
    + result["sonnet_4_non_veto_sonnet_5_veto_n"]
    == result["shared_record_n"]
).all():
    raise ValueError("Agreement/disagreement counts do not sum to shared n.")

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Shared records compared: {paired.height}")
print(f"Language/harmfulness cells: {result.height}")
overall_agreement_n = paired.select(
    (
        pl.col("claude_sonnet_4_is_vetoed")
        == pl.col("claude_sonnet_5_is_vetoed")
    ).sum()
).item()

overall_agreement_percentage = (
    100.0 * overall_agreement_n / paired.height
)

print(f"Overall agreement: {overall_agreement_percentage:.2f}%")

with pl.Config(tbl_cols=-1, tbl_width_chars=260):
    print(result)
