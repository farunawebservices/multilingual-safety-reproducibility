from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json

import polars as pl

METRICS = Path("paper_artifacts/metrics")
TABLES = Path("paper_artifacts/tables")
FIGURES = Path("paper_artifacts/figures")

ARTIFACTS = [
    METRICS / "qwen25_72b_primary_source_manifest.csv",
    METRICS / "qwen25_72b_source_schema_audit.csv",
    METRICS / "qwen25_72b_primary_canonical_audit.csv",
    METRICS / "qwen25_72b_decision_semantics_audit.csv",
    METRICS / "qwen25_72b_primary_pairing_coverage.csv",
    METRICS / "qwen25_72b_primary_paired_inference.csv",
    METRICS / "qwen25_72b_primary_paired_inference_holm.csv",
    METRICS / "qwen25_72b_primary_decision_composition.csv",
    METRICS / "qwen25_72b_primary_unified_results.csv",
    METRICS / "qwen25_72b_primary_family_inventory.csv",
    METRICS / "qwen25_72b_planner_alpha1_primary_paper_table.csv",
    TABLES / "qwen25_72b_planner_alpha1_primary_results.tex",
    FIGURES / "qwen25_72b_planner_alpha1_primary_effects.pdf",
    FIGURES / "qwen25_72b_planner_alpha1_primary_effects.png",
    METRICS / "qwen25_72b_planner_alpha1_primary_opus_handoff.md",
]

OUT = METRICS / "qwen25_72b_planner_alpha1_primary_artifact_manifest.json"

def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for path in ARTIFACTS:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")

paper_table_path = (
    METRICS / "qwen25_72b_planner_alpha1_primary_paper_table.csv"
)
paper_table = pl.read_csv(paper_table_path)

if paper_table.height != 8:
    raise ValueError(f"Expected 8 paper-table rows; found {paper_table.height}")

if paper_table["Paired n"].sum() != 1695:
    raise ValueError(
        "Expected 1695 paired prompts; "
        f"found {paper_table['Paired n'].sum()}."
    )

if not (paper_table["Coverage"] == "Complete matched coverage").all():
    raise ValueError("All primary rows must have complete matched coverage.")

holm_significant_n = paper_table.filter(
    pl.col("Holm reject 0.05")
).height

if holm_significant_n != 2:
    raise ValueError(
        f"Expected 2 Holm-significant rows; found {holm_significant_n}."
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
    "artifact_set": "qwen25_72b_planner_alpha1_primary_paired_results",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "authoritative_paper_table": str(paper_table_path),
    "authoritative_latex_table": str(
        TABLES / "qwen25_72b_planner_alpha1_primary_results.tex"
    ),
    "authoritative_figure_pdf": str(
        FIGURES / "qwen25_72b_planner_alpha1_primary_effects.pdf"
    ),
    "model_identifier": "Qwen 2.5 72B",
    "comparison_type": (
        "Within-model prompt-matched intervention comparison: baseline "
        "versus planner-only steering at alpha 1.0."
    ),
    "outcome_definition": (
        "Operational failure follows audited Qwen decision semantics: "
        "explicit veto is safe; non-veto unresolved is operational failure."
    ),
    "inference": {
        "test": "Exact paired McNemar test",
        "multiple_testing": "Holm adjustment",
        "correction_family": "qwen25_72b_primary_33_steered_cells",
        "correction_family_size": 33,
        "bootstrap": "95% interval over matched prompt IDs",
    },
    "coverage": {
        "paper_table_rows": 8,
        "paired_prompt_n": 1695,
        "partial_coverage_rows": 0,
        "holm_significant_rows": holm_significant_n,
    },
    "interpretation_constraints": [
        "Primary table includes only planner-only steering at alpha 1.0.",
        "All eight displayed cells have complete prompt-matched coverage.",
        "Do not pool other alpha values or critic/both steering rows into this primary result.",
        "Report Holm-adjusted inference from the predeclared 33-test Qwen family.",
        "Negative percentage-point differences indicate fewer operational failures under steering.",
        "Describe this as a within-model prompt-matched intervention comparison; do not make an unsupported causal-identification claim.",
    ],
    "artifacts": entries,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

print(f"Wrote: {OUT}")
print(f"Manifested artifacts: {len(entries)}")
print("Paired prompts: 1695")
print("Partial-coverage rows: 0")
print(f"Holm-significant rows: {holm_significant_n}")
print("Unsupported causal-identification claim allowed: False")
