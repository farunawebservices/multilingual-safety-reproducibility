from pathlib import Path
import json

import polars as pl

RECONCILIATION = Path(
    "paper_artifacts/metrics/gpt_5_source_reconciliation.csv"
)

OUT = Path(
    "paper_artifacts/metrics/"
    "gpt_5_clean_vs_partial_overlap_audit.csv"
)

reconciliation = pl.read_csv(RECONCILIATION)

jsonl_paths = (
    reconciliation
    .filter(
        (pl.col("location") == "raw")
        & (pl.col("file_type") == "jsonl")
    )
    .select([
        "path",
        "record_n_if_json",
        "parse_error_n_if_json",
    ])
)

if jsonl_paths.height != 2:
    raise ValueError("Expected exactly two raw GPT-5 JSONL files.")

clean_path = Path(
    jsonl_paths
    .filter(pl.col("parse_error_n_if_json") == 0)
    .select("path")
    .item()
)

partial_path = Path(
    jsonl_paths
    .filter(pl.col("parse_error_n_if_json") > 0)
    .select("path")
    .item()
)

clean = pl.read_ndjson(clean_path).select([
    "id",
    "language",
    "harmfulness",
    "is_vetoed",
])

partial_rows = []
parse_error_n = 0

with partial_path.open("r", encoding="utf-8", errors="replace") as handle:
    for line in handle:
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            parse_error_n += 1
            continue

        partial_rows.append({
            "id": record.get("id"),
            "language": record.get("language"),
            "harmfulness": record.get("harmfulness"),
            "is_vetoed": record.get("is_vetoed"),
        })

partial = pl.DataFrame(partial_rows)

for label, frame in [("clean", clean), ("partial", partial)]:
    required = ["id", "language", "harmfulness", "is_vetoed"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing fields: {missing}")

    if frame.select(pl.col("id").is_null().any()).item():
        raise ValueError(f"{label} contains null IDs.")

    if frame.select(pl.col("id").n_unique() != pl.len()).item():
        raise ValueError(f"{label} contains duplicate IDs.")

paired = clean.join(
    partial,
    on="id",
    how="full",
    suffix="_partial",
    coalesce=True,
    validate="1:1",
)

paired = paired.with_columns([
    pl.col("language").is_not_null().alias("present_in_clean"),
    pl.col("language_partial").is_not_null().alias("present_in_partial"),
    (
        pl.col("language")
        == pl.col("language_partial")
    )
    .fill_null(False)
    .alias("language_matches"),
    (
        pl.col("harmfulness")
        == pl.col("harmfulness_partial")
    )
    .fill_null(False)
    .alias("harmfulness_matches"),
    (
        pl.col("is_vetoed")
        == pl.col("is_vetoed_partial")
    )
    .fill_null(False)
    .alias("veto_outcome_matches"),
])

result = (
    paired
    .group_by([
        "present_in_clean",
        "present_in_partial",
        "language_matches",
        "harmfulness_matches",
        "veto_outcome_matches",
    ])
    .agg(pl.len().alias("record_n"))
    .sort([
        "present_in_clean",
        "present_in_partial",
        "language_matches",
        "harmfulness_matches",
        "veto_outcome_matches",
    ])
)

clean_n = clean.height
partial_n = partial.height
shared_n = paired.filter(
    pl.col("present_in_clean")
    & pl.col("present_in_partial")
).height

if parse_error_n != 1:
    raise ValueError(
        f"Expected one partial-file parse error; found {parse_error_n}."
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Clean valid records: {clean_n}")
print(f"Partial valid records: {partial_n}")
print(f"Partial parse errors: {parse_error_n}")
print(f"Shared IDs: {shared_n}")
print(
    "Shared IDs with matching language: "
    f"{paired.filter(pl.col('present_in_clean') & pl.col('present_in_partial') & pl.col('language_matches')).height}"
)
print(
    "Shared IDs with matching harmfulness: "
    f"{paired.filter(pl.col('present_in_clean') & pl.col('present_in_partial') & pl.col('harmfulness_matches')).height}"
)
print(
    "Shared IDs with matching veto outcome: "
    f"{paired.filter(pl.col('present_in_clean') & pl.col('present_in_partial') & pl.col('veto_outcome_matches')).height}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=220):
    print(result)
