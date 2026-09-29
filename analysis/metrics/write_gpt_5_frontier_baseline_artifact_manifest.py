from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json

import polars as pl

METRICS = Path("paper_artifacts/metrics")
TABLES = Path("paper_artifacts/tables")
FIGURES = Path("paper_artifacts/figures")

ARTIFACTS = [
    METRICS / "closed_model_source_inventory.csv",
    METRICS / "closed_model_schema_inventory.csv",
    METRICS / "gpt_5_source_reconciliation.csv",
    METRICS / "gpt_5_jsonl_compatibility_audit.csv",
    METRICS / "gpt_5_clean_frontier_baseline_audit.csv",
    METRICS / "gpt_5_clean_vs_partial_overlap_audit.csv",
    METRICS / "gpt_5_frontier_baseline_paper_table.csv",
    TABLES / "gpt_5_frontier_baseline_descriptive_results.tex",
    FIGURES / "gpt_5_frontier_baseline_veto_rates.pdf",
    FIGURES / "gpt_5_frontier_baseline_veto_rates.png",
]

OUT = METRICS / "gpt_5_frontier_baseline_artifact_manifest.json"

def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for path in ARTIFACTS:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")

paper_table_path = METRICS / "gpt_5_frontier_baseline_paper_table.csv"
paper_table = pl.read_csv(paper_table_path)

if paper_table.height != 8:
    raise ValueError(f"Expected 8 paper-table rows; found {paper_table.height}")

if paper_table["Valid n"].sum() != 1651:
    raise ValueError(
        f"Expected 1651 valid records; found {paper_table['Valid n'].sum()}."
    )

partial_n = paper_table.filter(
    pl.col("Coverage") != "Complete cell"
).height

if partial_n != 5:
    raise ValueError(f"Expected 5 partial-coverage rows; found {partial_n}")

harmful_partial_n = paper_table.filter(
    (pl.col("Prompt class") == "harmful")
    & (pl.col("Coverage") != "Complete cell")
).height

if harmful_partial_n != 4:
    raise ValueError(
        f"Expected 4 harmful partial-coverage rows; found {harmful_partial_n}"
    )

entries = []
for path in ARTIFACTS:
    kind = (
        "metric_csv"
        if path.suffix == ".csv"
        else "latex_table"
        if path.suffix == ".tex"
        else "figure"
    )

    entry = {
        "path": str(path),
        "kind": kind,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }

    if path.suffix == ".csv":
        entry["row_count"] = pl.read_csv(path).height

    entries.append(entry)

manifest = {
    "artifact_set": "gpt_5_frontier_baseline_descriptive_results",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "authoritative_paper_table": str(paper_table_path),
    "authoritative_latex_table": str(
        TABLES / "gpt_5_frontier_baseline_descriptive_results.tex"
    ),
    "authoritative_figure_pdf": str(
        FIGURES / "gpt_5_frontier_baseline_veto_rates.pdf"
    ),
    "analysis_type": (
        "Descriptive single-configuration baseline summary."
    ),
    "model_identifier": "openai/gpt-5",
    "configuration": "frontier_baseline",
    "strength": 0.0,
    "outcome": {
        "veto": "is_vetoed=true",
        "non_veto_operational_failure": "is_vetoed=false",
    },
    "interpretation_constraints": [
        "Do not describe this baseline summary as a steering effect.",
        "Do not make a causal or cross-condition comparison claim.",
        "Do not pool in the separate 570-record duplicate subset.",
        "The duplicate subset has one malformed terminal JSONL line.",
        "Yoruba benign has 45 valid prompts.",
        "Harmful cells have 286 to 291 valid prompts, rather than 300.",
    ],
    "source_reconciliation": {
        "clean_source_valid_records": 1651,
        "duplicate_subset_valid_records": 570,
        "duplicate_subset_parse_errors": 1,
        "duplicate_subset_shared_ids": 570,
        "duplicate_subset_matching_language": 570,
        "duplicate_subset_matching_harmfulness": 570,
        "duplicate_subset_matching_veto_outcome": 570,
    },
    "coverage": {
        "paper_table_rows": 8,
        "valid_record_n": 1651,
        "partial_coverage_rows": 5,
        "harmful_partial_coverage_rows": 4,
    },
    "artifacts": entries,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

print(f"Wrote: {OUT}")
print(f"Manifested artifacts: {len(entries)}")
print(
    "Metric CSV artifacts: "
    f"{sum(item['kind'] == 'metric_csv' for item in entries)}"
)
print(
    "LaTeX table artifacts: "
    f"{sum(item['kind'] == 'latex_table' for item in entries)}"
)
print(
    "Figure artifacts: "
    f"{sum(item['kind'] == 'figure' for item in entries)}"
)
print("Valid records: 1651")
print("Partial-coverage rows: 5")
print("Excluded duplicate-subset valid records: 570")
print("Comparison or inference claim allowed: False")
