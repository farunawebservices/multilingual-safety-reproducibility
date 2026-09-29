from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json

import polars as pl

METRICS = Path("paper_artifacts/metrics")
TABLES = Path("paper_artifacts/tables")
FIGURES = Path("paper_artifacts/figures")

ARTIFACTS = [
    METRICS / "gemma_4_31b_phase1_complete_case_paired_inference.csv",
    METRICS / "gemma_4_31b_phase1_complete_case_paired_inference_holm.csv",
    METRICS / "gemma_4_31b_phase1_complete_case_decision_semantics_audit.csv",
    METRICS / "gemma_4_31b_phase1_complete_case_unified_results.csv",
    METRICS / "gemma_4_31b_phase1_complete_case_paper_table.csv",
    TABLES / "gemma_4_31b_phase1_complete_case_results.tex",
    FIGURES / "gemma_4_31b_phase1_complete_case_effects.pdf",
    FIGURES / "gemma_4_31b_phase1_complete_case_effects.png",
]

OUT = METRICS / "gemma_4_31b_phase1_complete_case_artifact_manifest.json"

def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for path in ARTIFACTS:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact is missing: {path}")

paper_table_path = METRICS / "gemma_4_31b_phase1_complete_case_paper_table.csv"
paper_table = pl.read_csv(paper_table_path)

if paper_table.height != 3:
    raise ValueError(f"Expected 3 paper-table rows; found {paper_table.height}")

if paper_table.filter(pl.col("Coverage") == "Matched subset").height != 1:
    raise ValueError("Expected exactly one matched-subset paper-table row.")

if paper_table.filter(pl.col("Holm significant (0.05)") == "Yes").height != 0:
    raise ValueError("Expected zero Holm-significant paper-table rows.")

entries = []
for path in ARTIFACTS:
    entry = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "kind": (
            "metric_csv"
            if path.suffix == ".csv"
            else "latex_table"
            if path.suffix == ".tex"
            else "figure"
        ),
    }

    if path.suffix == ".csv":
        entry["row_count"] = pl.read_csv(path).height

    entries.append(entry)

manifest = {
    "artifact_set": "gemma_4_31b_phase1_complete_case",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "authoritative_paper_table": str(paper_table_path),
    "authoritative_latex_table": str(
        TABLES / "gemma_4_31b_phase1_complete_case_results.tex"
    ),
    "authoritative_figure_pdf": str(
        FIGURES / "gemma_4_31b_phase1_complete_case_effects.pdf"
    ),
    "analysis_population": "complete-case valid JSON records, with paired inference on shared record IDs",
    "primary_outcome": (
        "Non-veto operational failure: is_vetoed=false. "
        "This must not be interpreted as explicit ALLOW."
    ),
    "inference": {
        "paired_test": "Exact two-sided McNemar test",
        "uncertainty": "10,000-replicate paired bootstrap percentile 95% CI",
        "multiple_testing": "Holm correction across the 3 Gemma Phase 1 cells",
        "familywise_alpha": 0.05,
        "holm_significant_rows": 0,
    },
    "coverage": {
        "paper_table_rows": 3,
        "partial_coverage_rows": 1,
        "partial_coverage_detail": (
            "English benign uses 101 shared valid IDs; "
            "11 steered-only valid IDs are excluded from paired inference."
        ),
    },
    "source_caveat": (
        "One truncated terminal JSONL record is documented; "
        "analyses use complete-case valid JSON records only."
    ),
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
print("Authoritative paper table rows: 3")
print("Partial-coverage rows: 1")
print("Holm-significant rows: 0")
