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
    METRICS / "claude_sonnet_4_phase1_schema_profile.csv",
    METRICS / "claude_sonnet_4_phase1_baseline_audit.csv",
    METRICS / "claude_sonnet_4_vs_5_prompt_coverage_audit.csv",
    METRICS / "claude_sonnet_4_vs_5_metadata_consistency_audit.csv",
    METRICS / "claude_sonnet_4_vs_5_outcome_comparison.csv",
    METRICS / "claude_sonnet_4_vs_5_paper_table.csv",
    TABLES / "claude_sonnet_4_vs_5_descriptive_comparison.tex",
    FIGURES / "claude_sonnet_4_vs_5_veto_rate_comparison.pdf",
    FIGURES / "claude_sonnet_4_vs_5_veto_rate_comparison.png",
]

OUT = METRICS / "claude_sonnet_4_vs_5_artifact_manifest.json"

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
    METRICS / "claude_sonnet_4_vs_5_paper_table.csv"
)
paper_table = pl.read_csv(paper_table_path)

if paper_table.height != 8:
    raise ValueError(f"Expected 8 paper-table rows; found {paper_table.height}")

partial_n = paper_table.filter(
    pl.col("Coverage") == "Partial cell: 45 prompts"
).height

if partial_n != 1:
    raise ValueError(f"Expected 1 partial-coverage row; found {partial_n}")

if paper_table["Matched n"].sum() != 1695:
    raise ValueError(
        "Expected 1695 matched prompts; "
        f"found {paper_table['Matched n'].sum()}."
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
    "artifact_set": "claude_sonnet_4_vs_5_phase1_descriptive_comparison",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "authoritative_paper_table": str(paper_table_path),
    "authoritative_latex_table": str(
        TABLES / "claude_sonnet_4_vs_5_descriptive_comparison.tex"
    ),
    "authoritative_figure_pdf": str(
        FIGURES / "claude_sonnet_4_vs_5_veto_rate_comparison.pdf"
    ),
    "comparison_type": (
        "Prompt-matched descriptive comparison of separate model versions."
    ),
    "outcome": "Veto status, defined by is_vetoed=true.",
    "interpretation_constraints": [
        "Do not describe this as a within-model steering effect.",
        "Do not make a causal claim from this cross-model comparison.",
        "Do not apply or report the Qwen/Gemma Holm families here.",
        "Yoruba benign has partial coverage: 45 shared prompts.",
        "All 1695 IDs have matching language and harmfulness metadata.",
    ],
    "coverage": {
        "paper_table_rows": 8,
        "matched_prompt_n": 1695,
        "partial_coverage_rows": 1,
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
print("Matched prompts: 1695")
print("Partial-coverage rows: 1")
print("Inference claim allowed: False")
