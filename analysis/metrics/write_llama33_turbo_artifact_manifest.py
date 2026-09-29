from __future__ import annotations

import csv
import hashlib
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import scipy
import statsmodels

OUT = Path("paper_artifacts/metrics/llama33_turbo_artifact_manifest.csv")

ARTIFACTS = [
    {
        "artifact_path": "paper_artifacts/metrics/llama33_planner20_paired_inference.csv",
        "producer_script": "analysis/metrics/llama_planner20_inference.py",
        "source_inputs": "data/derived/canonical/llama33_phase1_main.parquet",
        "artifact_type": "paired_inference",
        "outcome_definition": "harmful operational failure",
        "comparison_scope": "planner_only alpha20 vs alpha0",
        "correction_family": "planner_main_language_family",
        "cross_phase_comparison": False,
        "random_seed": "",
        "bootstrap_replicates": "",
    },
    {
        "artifact_path": "paper_artifacts/metrics/llama33_critic20_paired_inference.csv",
        "producer_script": "analysis/metrics/llama_critic20_inference.py",
        "source_inputs": "data/derived/canonical/llama33_phase1_main.parquet",
        "artifact_type": "paired_inference",
        "outcome_definition": "harmful operational failure",
        "comparison_scope": "critic_only alpha20 vs alpha0",
        "correction_family": "critic_main_language_family",
        "cross_phase_comparison": False,
        "random_seed": "",
        "bootstrap_replicates": "",
    },
    {
        "artifact_path": "paper_artifacts/metrics/llama33_both_turbo_paired_inference.csv",
        "producer_script": "analysis/metrics/llama_both_turbo_inference.py",
        "source_inputs": (
            "data/derived/canonical/llama33_phase1_main.parquet;"
            "data/derived/canonical/llama33_phase2_turbo.parquet"
        ),
        "artifact_type": "paired_inference",
        "outcome_definition": "harmful operational failure",
        "comparison_scope": "both alpha20/alpha30 vs alpha0",
        "correction_family": "both_turbo_alpha_language_family",
        "cross_phase_comparison": True,
        "random_seed": "",
        "bootstrap_replicates": "",
    },
    {
        "artifact_path": (
            "paper_artifacts/metrics/"
            "llama33_planner_turbo_paired_inference.csv"
        ),
        "producer_script": (
            "analysis/metrics/"
            "llama33_planner_turbo_paired_inference.py"
        ),
        "source_inputs": (
            "data/derived/canonical/llama33_phase1_main.parquet;"
            "data/derived/canonical/llama33_phase2_turbo.parquet"
        ),
        "artifact_type": "paired_inference",
        "outcome_definition": "harmful operational failure",
        "comparison_scope": "planner_only alpha20/alpha30 vs alpha0",
        "correction_family": "planner_turbo_alpha_language_family",
        "cross_phase_comparison": True,
        "random_seed": 20260908,
        "bootstrap_replicates": 10000,
    },
    {
        "artifact_path": (
            "paper_artifacts/metrics/"
            "llama33_turbo_alpha30_role_strength_paired_inference.csv"
        ),
        "producer_script": (
            "analysis/metrics/"
            "llama33_turbo_alpha30_role_strength_paired_inference.py"
        ),
        "source_inputs": (
            "data/derived/canonical/llama33_phase1_main.parquet;"
            "data/derived/canonical/llama33_phase2_turbo.parquet"
        ),
        "artifact_type": "paired_inference",
        "outcome_definition": "harmful operational failure",
        "comparison_scope": "all roles alpha30 vs alpha0; Hausa and Yoruba",
        "correction_family": "turbo_alpha30_role_strength_6_tests",
        "cross_phase_comparison": True,
        "random_seed": 20260908,
        "bootstrap_replicates": 10000,
    },
    {
        "artifact_path": (
            "paper_artifacts/metrics/"
            "llama33_turbo_role_strength_decision_composition.csv"
        ),
        "producer_script": (
            "analysis/metrics/"
            "llama33_turbo_role_strength_decision_composition.py"
        ),
        "source_inputs": (
            "data/derived/canonical/llama33_phase1_main.parquet;"
            "data/derived/canonical/llama33_phase2_turbo.parquet"
        ),
        "artifact_type": "descriptive_decision_composition",
        "outcome_definition": (
            "parser outcome counts; operational failure = "
            "explicit ALLOW plus NO_DECISION"
        ),
        "comparison_scope": "all roles alpha0/alpha20/alpha30; Hausa and Yoruba",
        "correction_family": "not_applicable",
        "cross_phase_comparison": True,
        "random_seed": "",
        "bootstrap_replicates": "",
    },
    {
        "artifact_path": (
            "paper_artifacts/metrics/"
            "llama33_turbo_role_strength_summary.csv"
        ),
        "producer_script": (
            "analysis/metrics/"
            "llama_turbo_role_strength_summary.py"
        ),
        "source_inputs": (
            "data/derived/canonical/llama33_phase1_main.parquet;"
            "data/derived/canonical/llama33_phase2_turbo.parquet;"
            "paper_artifacts/metrics/llama33_planner20_paired_inference.csv;"
            "paper_artifacts/metrics/llama33_critic20_paired_inference.csv;"
            "paper_artifacts/metrics/llama33_both_turbo_paired_inference.csv"
        ),
        "artifact_type": "descriptive_summary_with_available_inference",
        "outcome_definition": "harmful operational failure",
        "comparison_scope": "role-strength summary",
        "correction_family": "inherits_source_artifact",
        "cross_phase_comparison": True,
        "random_seed": "",
        "bootstrap_replicates": "",
    },
    {
        "artifact_path": (
            "paper_artifacts/metrics/"
            "llama33_turbo_alpha30_role_strength_unified_results.csv"
        ),
        "producer_script": (
            "analysis/metrics/"
            "llama33_write_alpha30_unified_results.py"
        ),
        "source_inputs": (
            "paper_artifacts/metrics/"
            "llama33_turbo_alpha30_role_strength_paired_inference.csv;"
            "paper_artifacts/metrics/"
            "llama33_turbo_role_strength_decision_composition.csv"
        ),
        "artifact_type": "unified_reporting_results",
        "outcome_definition": "harmful operational failure",
        "comparison_scope": "all roles alpha30 vs alpha0; Hausa and Yoruba",
        "correction_family": "turbo_alpha30_role_strength_6_tests",
        "cross_phase_comparison": True,
        "random_seed": 20260908,
        "bootstrap_replicates": 10000,
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


rows = []
for item in ARTIFACTS:
    artifact = Path(item["artifact_path"])
    script = Path(item["producer_script"])

    if not artifact.exists():
        raise FileNotFoundError(f"Missing artifact: {artifact}")

    if not script.exists():
        raise FileNotFoundError(f"Missing producer script: {script}")

    rows.append({
        **item,
        "artifact_exists": True,
        "artifact_rows": count_csv_rows(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": sha256(artifact),
        "producer_script_sha256": sha256(script),
    })

OUT.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "artifact_path",
    "artifact_type",
    "artifact_exists",
    "artifact_rows",
    "artifact_bytes",
    "artifact_sha256",
    "producer_script",
    "producer_script_sha256",
    "source_inputs",
    "outcome_definition",
    "comparison_scope",
    "correction_family",
    "cross_phase_comparison",
    "random_seed",
    "bootstrap_replicates",
]

with OUT.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote: {OUT}")
print(f"Generated at UTC: {datetime.now(timezone.utc).isoformat()}")
print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
print(f"Polars: {pl.__version__}")
print(f"NumPy: {np.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"statsmodels: {statsmodels.__version__}")

for row in rows:
    print(
        f"{row['artifact_rows']:>3} rows  "
        f"{row['artifact_path']}  "
        f"{row['artifact_sha256']}"
    )
