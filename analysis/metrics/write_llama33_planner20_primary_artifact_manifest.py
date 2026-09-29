from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json

import polars as pl

RAW = Path(
    "artifacts/prior_llama33_phase2/results/"
    "baseline_completed_backup (1).jsonl"
)
CANONICAL = Path("data/derived/canonical/llama33_phase1_main.parquet")
REGISTRY = Path("registry")
METRICS = Path("paper_artifacts/metrics")
TABLES = Path("paper_artifacts/tables")
FIGURES = Path("paper_artifacts/figures")

ARTIFACTS = [
    RAW,
    CANONICAL,
    REGISTRY / "manifests/llama33_phase1_main.json",
    REGISTRY / "llama/main_semantics_audit.tsv",
    REGISTRY / "llama/main_semantics_audit_console.tsv",
    METRICS / "llama33_planner20_paired_effect.csv",
    METRICS / "llama33_planner20_paired_inference.csv",
    METRICS / "llama33_planner20_benign_refusal_effect.csv",
    METRICS / "llama33_planner20_benign_refusal_inference.csv",
    METRICS / "llama33_planner20_primary_raw_separation_audit.csv",
    METRICS / "llama33_planner20_primary_paper_table.csv",
    TABLES / "llama33_planner20_primary_results.tex",
    FIGURES / "llama33_planner20_primary_effects.pdf",
    FIGURES / "llama33_planner20_primary_effects.png",
    METRICS / "llama33_planner20_primary_opus_handoff.md",
]

OUT = METRICS / "llama33_planner20_primary_artifact_manifest.json"

def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for path in ARTIFACTS:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")

paper_table = pl.read_csv(
    METRICS / "llama33_planner20_primary_paper_table.csv"
)

if paper_table.height != 4:
    raise ValueError(f"Expected 4 paper-table rows; found {paper_table.height}.")

if paper_table["Harmful paired n"].sum() != 1200:
    raise ValueError("Expected 1200 paired harmful prompts.")

if paper_table["Benign paired n"].sum() != 495:
    raise ValueError("Expected 495 paired benign prompts.")

harmful_significant = paper_table.filter(
    pl.col("Harmful Holm reject 0.05")
).height

benign_significant = paper_table.filter(
    pl.col("Benign Holm reject 0.05")
).height

if harmful_significant != 3:
    raise ValueError(
        f"Expected 3 harmful Holm-significant rows; found {harmful_significant}."
    )

if benign_significant != 0:
    raise ValueError(
        f"Expected 0 benign Holm-significant rows; found {benign_significant}."
    )

if paper_table["Harmful paired n"].min() != 300:
    raise ValueError("Expected 300 harmful matched prompts per language.")

expected_benign_n = {
    "Arabic": 150,
    "English": 150,
    "Hausa": 150,
    "Yoruba": 45,
}

actual_benign_n = dict(
    zip(
        paper_table["Language"].to_list(),
        paper_table["Benign paired n"].to_list(),
    )
)

if actual_benign_n != expected_benign_n:
    raise ValueError(
        f"Unexpected benign per-language sample sizes: {actual_benign_n}."
    )

entries = []

for path in ARTIFACTS:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        kind = "metric_csv"
    elif suffix == ".parquet":
        kind = "canonical_parquet"
    elif suffix == ".jsonl":
        kind = "raw_result_jsonl"
    elif suffix == ".tsv":
        kind = "registry_audit_tsv"
    elif suffix == ".tex":
        kind = "latex_table"
    elif suffix in {".pdf", ".png"}:
        kind = "figure"
    else:
        kind = "registry_manifest"

    entry = {
        "path": str(path),
        "kind": kind,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }

    if suffix == ".csv":
        entry["row_count"] = pl.read_csv(path).height

    entries.append(entry)

manifest = {
    "artifact_set": "llama33_planner20_primary_paired_results",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "model_identifier": "meta-llama/llama-3.3-70b-instruct",
    "authoritative_paper_table": str(
        METRICS / "llama33_planner20_primary_paper_table.csv"
    ),
    "authoritative_latex_table": str(
        TABLES / "llama33_planner20_primary_results.tex"
    ),
    "authoritative_figure_pdf": str(
        FIGURES / "llama33_planner20_primary_effects.pdf"
    ),
    "comparison_type": (
        "Within-phase, within-model, prompt-matched comparison: "
        "planner-only alpha 0 versus planner-only alpha 20."
    ),
    "primary_scope": (
        "Arabic, English, Hausa, and Yoruba. Harmful prompts use "
        "operational failure; benign prompts use refusal rate."
    ),
    "raw_provenance": {
        "raw_source": str(RAW),
        "raw_source_sha256": sha256_file(RAW),
        "source_structure": (
            "One combined raw JSONL contains distinct planner-only "
            "alpha 0 and alpha 20 records."
        ),
        "raw_to_canonical_audit": str(
            METRICS / "llama33_planner20_primary_raw_separation_audit.csv"
        ),
    },
    "inference": {
        "harmful_test": "Exact paired McNemar test",
        "benign_test": "Exact paired McNemar test",
        "multiple_testing": (
            "Holm adjustment applied separately to four harmful and "
            "four benign language-specific comparisons."
        ),
        "harmful_correction_family_size": 4,
        "benign_correction_family_size": 4,
        "intervals": (
            "Paired nonparametric bootstrap percentile intervals, "
            "10,000 replicates, seed 20260908."
        ),
    },
    "coverage": {
        "paper_table_rows": 4,
        "harmful_paired_prompt_n": 1200,
        "harmful_paired_prompt_n_per_language": 300,
        "benign_paired_prompt_n": 495,
        "benign_paired_prompt_n_by_language": expected_benign_n,
        "harmful_holm_significant_rows": harmful_significant,
        "benign_holm_significant_rows": benign_significant,
        "partial_harmful_coverage_rows": 0,
        "partial_benign_coverage_rows": 1,
    },
    "interpretation_constraints": [
        "Harmful operational-failure and benign-refusal outcomes must be reported separately.",
        "Negative harmful percentage-point differences indicate fewer operational failures under steering.",
        "Positive benign percentage-point differences indicate more false refusals under steering.",
        "Harmful comparisons have complete matched coverage: 300 prompts per language.",
        "Benign coverage differs by language: Yoruba has 45 matched prompts; Arabic, English, and Hausa each have 150.",
        "Report Holm-adjusted inference separately for the harmful and benign correction families.",
        "This primary package excludes Phase 2 Turbo results and cross-phase comparisons.",
        "Describe this as a within-phase, within-model prompt-matched intervention comparison; do not make an unsupported causal-identification claim.",
    ],
    "artifacts": entries,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

print(f"Wrote: {OUT}")
print(f"Manifested artifacts: {len(entries)}")
print(f"Metric CSV artifacts: {sum(item['kind'] == 'metric_csv' for item in entries)}")
print(f"Registry audit TSV artifacts: {sum(item['kind'] == 'registry_audit_tsv' for item in entries)}")
print(f"Canonical Parquet artifacts: {sum(item['kind'] == 'canonical_parquet' for item in entries)}")
print(f"Raw result JSONL artifacts: {sum(item['kind'] == 'raw_result_jsonl' for item in entries)}")
print(f"Paired harmful prompts: {paper_table['Harmful paired n'].sum()}")
print(f"Paired benign prompts: {paper_table['Benign paired n'].sum()}")
print(f"Harmful Holm-significant rows: {harmful_significant}")
print(f"Benign Holm-significant rows: {benign_significant}")
print("Phase 2 Turbo artifacts included: False")
print("Harmful and benign outcomes kept separate: True")
