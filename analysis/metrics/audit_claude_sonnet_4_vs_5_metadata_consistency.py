from pathlib import Path

import polars as pl

INVENTORY = Path(
    "paper_artifacts/metrics/closed_model_source_inventory.csv"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "claude_sonnet_4_vs_5_metadata_consistency_audit.csv"
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
            f"Expected one raw JSONL for {model}; found {len(paths)}."
        )

    return Path(paths[0])

def load(model: str) -> pl.DataFrame:
    frame = pl.read_ndjson(source_for(model))

    required = ["id", "language", "harmfulness", "is_vetoed"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{model} missing fields: {missing}")

    if frame.select(pl.col("id").is_null().any()).item():
        raise ValueError(f"{model} contains null IDs.")

    if frame.select(pl.col("id").n_unique() != pl.len()).item():
        raise ValueError(f"{model} contains duplicate IDs.")

    return frame.select([
        "id",
        pl.col("language").alias(f"{model}_language"),
        pl.col("harmfulness").alias(f"{model}_harmfulness"),
        pl.col("is_vetoed").alias(f"{model}_is_vetoed"),
    ])

sonnet4 = load("claude_sonnet_4")
sonnet5 = load("claude_sonnet_5")

paired = sonnet4.join(
    sonnet5,
    on="id",
    how="full",
    coalesce=True,
    validate="1:1",
)

paired = paired.with_columns([
    pl.col("claude_sonnet_4_language").is_not_null()
    .alias("present_in_sonnet_4"),
    pl.col("claude_sonnet_5_language").is_not_null()
    .alias("present_in_sonnet_5"),
    (
        pl.col("claude_sonnet_4_language")
        == pl.col("claude_sonnet_5_language")
    )
    .fill_null(False)
    .alias("language_matches"),
    (
        pl.col("claude_sonnet_4_harmfulness")
        == pl.col("claude_sonnet_5_harmfulness")
    )
    .fill_null(False)
    .alias("harmfulness_matches"),
])

result = (
    paired
    .group_by([
        "present_in_sonnet_4",
        "present_in_sonnet_5",
        "language_matches",
        "harmfulness_matches",
    ])
    .agg(pl.len().alias("record_n"))
    .sort([
        "present_in_sonnet_4",
        "present_in_sonnet_5",
        "language_matches",
        "harmfulness_matches",
    ])
)

if paired.height != 1695:
    raise ValueError(f"Expected 1695 union IDs; found {paired.height}")

if not paired["present_in_sonnet_4"].all():
    raise ValueError("Found Sonnet 5-only IDs.")

if not paired["present_in_sonnet_5"].all():
    raise ValueError("Found Sonnet 4-only IDs.")

if not paired["language_matches"].all():
    raise ValueError("Language metadata mismatch found.")

if not paired["harmfulness_matches"].all():
    raise ValueError("Harmfulness metadata mismatch found.")

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Union IDs audited: {paired.height}")
print(f"Language matches: {paired['language_matches'].sum()}")
print(f"Harmfulness matches: {paired['harmfulness_matches'].sum()}")
print("All metadata checks passed: True")

with pl.Config(tbl_cols=-1, tbl_width_chars=180):
    print(result)
