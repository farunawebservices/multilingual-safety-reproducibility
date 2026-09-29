from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json

import polars as pl

RAW = Path("data/raw/phase1/gpt_oss_20b/originals")
CANONICAL = Path("data/derived/canonical")
REGISTRY = Path("registry")
METRICS = Path("paper_artifacts/metrics")
TABLES = Path("paper_artifacts/tables")
FIGURES = Path("paper_artifacts/figures")

ARTIFACTS = [
    RAW / "gpt_oss_baseline_results.jsonl",
    RAW / "gpt_oss_steered_results (2).jsonl",
    CANONICAL / "gpt_oss_20b_phase1_baseline.parquet",
    CANONICAL / "gpt_oss_20b_phase1_planner_only_alpha1.parquet",
    REGISTRY / "manifests/gpt_oss_20b_phase1_baseline.json",
    REGISTRY / "manifests/gpt_oss_20b_phase1_planner_only_alpha1.json",
    REGISTRY / "gpt_oss_decision_semantics_audit.json",
    REGISTRY / "gpt_oss_parser_outcomes.json",
    METRICS / "gpt_oss_20b_phase1_baseline_by_language.csv",
    METRICS / "gpt_oss_20b_phase1_baseline_bootstrap_ci.csv",
    METRICS / "gpt_oss_20b_planner_alpha1_paired_effect.csv",
    METRICS / "gpt_oss_20b_planner_alpha1_paired_bootstrap_ci.csv",
    METRICS / "gpt_oss_20b_planner_alpha1_paired_effect_holm.csv",
    METRICS / "gpt_oss_20b_planner_alpha1_primary_canonical_audit.csv",
    METRICS / "gpt_oss_20b_planner_alpha1_primary_harmful_paper_table.csv",
    TABLES / "gpt_oss_20b_planner_alpha1_primary_harmful_results.tex",
    FIGURES / "gpt_oss_20b_planner_alpha1_primary_harmful_effects.pdf",
    FIGURES / "gpt_oss_20b_planner_alpha1_primary_harmful_effects.png",
    METRICS / "gpt_oss_20b_planner_alpha1_primary_harmful_opus_handoff.md",
]

OUT = METRICS / "gpt_oss_20b_planner_alpha1_primary_harmful_artifact_manifest.json"

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
    METRICS / "gpt_oss_20b_planner_alpha1_primary_harmful_paper_table.csv"
)
paper_table = pl.read_csv(paper_table_path)

if paper_table.height != 4:
    raise ValueError(f"Expected 4 paper-table rows; found {paper_table.height}.")

if paper_table["Paired n"].sum() != 1200:
    raise ValueError(
        "Expected 1200 paired harmful prompts; "
        f"found {paper_table['Paired n'].sum()}."
    )

if not (paper_table["Prompt class"] == "Harmful").all():
    raise ValueError("Paper table must be harmful prompts only.")

if not (paper_table["Coverage"] == "Complete matched coverage").all():
    raise ValueError("All primary rows must have complete matched coverage.")

holm_significant_n = paper_table.filter(
    pl.col("Holm reject 0.05")
).height

negative_effect_n = paper_table.filter(
    pl.col("Steered minus baseline (pp)") < 0
).height

positive_effect_n = paper_table.filter(
    pl.col("Steered minus baseline (pp)") > 0
).height

if holm_significant_n != 2:
    raise ValueError(
        f"Expected 2 Holm-significant rows; found {holm_significant_n}."
    )

if negative_effect_n != 2:
    raise ValueError(
        f"Expected 2 improvement rows; found {negative_effect_n}."
    )

if positive_effect_n != 1:
    raise ValueError(
        f"Expected 1 worsening row; found {positive_effect_n}."
    )

entries = []
for path in ARTIFACTS:
    if path.suffix == ".csv":
        kind = "metric_csv"
    elif path.suffix == ".parquet":
        kind = "canonical_parquet"
    elif path.suffix == ".jsonl":
        kind = "raw_result_jsonl"
    elif path.suffix == ".tex":
        kind = "latex_table"
    elif path.suffix in {".pdf", ".png"}:
        kind = "figure"
    else:
        kind = "provenance_or_audit"

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
    "artifact_set": "gpt_oss_20b_planner_alpha1_primary_harmful_paired_results",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "model_identifier": "openai/gpt-oss-20b",
    "authoritative_paper_table": str(paper_table_path),
    "authoritative_latex_table": str(
        TABLES / "gpt_oss_20b_planner_alpha1_primary_harmful_results.tex"
    ),
    "authoritative_figure_pdf": str(
        FIGURES / "gpt_oss_20b_planner_alpha1_primary_harmful_effects.pdf"
    ),
    "comparison_type": (
        "Within-model prompt-matched intervention comparison: baseline "
        "versus planner-only steering at alpha 1.0."
    ),
    "primary_scope": (
        "Harmful prompts only: Arabic, English, Hausa, and Yoruba; "
        "300 matched prompts per language."
    ),
    "outcome_definition": (
        "Operational failure follows the registered GPT-OSS decision "
        "semantics and parser-outcome audits."
    ),
    "inference": {
        "test": "Exact paired McNemar test",
        "multiple_testing": "Holm adjustment",
        "correction_family": "four language-specific harmful comparisons",
        "correction_family_size": 4,
        "bootstrap": "Paired bootstrap intervals are retained as a separate metric artifact.",
    },
    "coverage": {
        "paper_table_rows": 4,
        "paired_harmful_prompt_n": 1200,
        "paired_harmful_prompt_n_per_language": 300,
        "partial_coverage_rows": 0,
        "holm_significant_rows": holm_significant_n,
        "improvement_rows_negative_pp": negative_effect_n,
        "worsening_rows_positive_pp": positive_effect_n,
    },
    "interpretation_constraints": [
        "Primary table includes harmful prompts only.",
        "Primary table includes planner-only steering at alpha 1.0 only.",
        "All four displayed language rows have complete prompt-matched coverage.",
        "Negative percentage-point differences indicate fewer operational failures under steering.",
        "The result is mixed by language: improvements and worsening must both be reported.",
        "Report Holm-adjusted inference for language-specific significance claims.",
        "Describe this as a within-model prompt-matched intervention comparison; do not make an unsupported causal-identification claim.",
    ],
    "artifacts": entries,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

print(f"Wrote: {OUT}")
print(f"Manifested artifacts: {len(entries)}")
print(f"Metric CSV artifacts: {sum(item['kind'] == 'metric_csv' for item in entries)}")
print(f"Canonical Parquet artifacts: {sum(item['kind'] == 'canonical_parquet' for item in entries)}")
print(f"Raw result JSONL artifacts: {sum(item['kind'] == 'raw_result_jsonl' for item in entries)}")
print(f"Paired harmful prompts: {paper_table['Paired n'].sum()}")
print(f"Holm-significant rows: {holm_significant_n}")
print(f"Improvement rows (negative pp): {negative_effect_n}")
print(f"Worsening rows (positive pp): {positive_effect_n}")
print("Complete matched harmful coverage: True")
