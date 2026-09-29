from pathlib import Path

import polars as pl

INVENTORY = Path(
    "paper_artifacts/metrics/closed_model_source_inventory.csv"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "claude_sonnet_4_vs_5_prompt_coverage_audit.csv"
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

claude4_path = source_for("claude_sonnet_4")
claude5_path = source_for("claude_sonnet_5")

claude4 = pl.read_ndjson(claude4_path).select([
    "id",
    "language",
    "harmfulness",
    "is_vetoed",
])

claude5 = pl.read_ndjson(claude5_path).select([
    "id",
    "language",
    "harmfulness",
    "is_vetoed",
])

for model, frame in [
    ("claude_sonnet_4", claude4),
    ("claude_sonnet_5", claude5),
]:
    required = ["id", "language", "harmfulness", "is_vetoed"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{model} missing required fields: {missing}")

    if frame.select(pl.col("id").is_null().any()).item():
        raise ValueError(f"{model} has null IDs.")

    if frame.select(pl.col("id").n_unique() != pl.len()).item():
        raise ValueError(f"{model} has duplicate IDs.")

claude4_ids = claude4.select("id").unique()
claude5_ids = claude5.select("id").unique()

shared = claude4_ids.join(claude5_ids, on="id", how="inner").height
only4 = claude4_ids.join(claude5_ids, on="id", how="anti").height
only5 = claude5_ids.join(claude4_ids, on="id", how="anti").height

coverage4 = (
    claude4
    .group_by(["language", "harmfulness"])
    .agg(pl.len().alias("claude_sonnet_4_record_n"))
)

coverage5 = (
    claude5
    .group_by(["language", "harmfulness"])
    .agg(pl.len().alias("claude_sonnet_5_record_n"))
)

result = (
    coverage4
    .join(
        coverage5,
        on=["language", "harmfulness"],
        how="full",
        coalesce=True,
    )
    .with_columns([
        pl.col("claude_sonnet_4_record_n").fill_null(0),
        pl.col("claude_sonnet_5_record_n").fill_null(0),
    ])
    .with_columns(
        (
            pl.col("claude_sonnet_5_record_n")
            - pl.col("claude_sonnet_4_record_n")
        ).alias("sonnet_5_minus_sonnet_4_record_n")
    )
    .sort(["harmfulness", "language"])
)

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Claude Sonnet 4 records: {claude4.height}")
print(f"Claude Sonnet 5 records: {claude5.height}")
print(f"Shared IDs: {shared}")
print(f"Claude Sonnet 4-only IDs: {only4}")
print(f"Claude Sonnet 5-only IDs: {only5}")

with pl.Config(tbl_cols=-1, tbl_width_chars=220):
    print(result)
