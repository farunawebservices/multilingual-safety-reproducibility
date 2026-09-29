#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(".")
OUT = Path(
    "paper_artifacts/handoff/"
    "addition1_llama33_hausa_patching_v3_artifact_manifest.json"
)

ARTIFACT_PATHS = {
    "canonical_parquet": Path(
        "data/derived/canonical/"
        "addition1_llama33_hausa_patching_v3.parquet"
    ),
    "audited_metrics_csv": Path(
        "paper_artifacts/metrics/"
        "addition1_llama33_hausa_patching_v3_audit.csv"
    ),
    "latex_table": Path(
        "paper_artifacts/tables/"
        "addition1_llama33_hausa_patching_v3_table.tex"
    ),
    "decision_figure_png": Path(
        "paper_artifacts/figures/"
        "addition1_llama33_hausa_patching_v3_decisions.png"
    ),
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

artifacts = {}
for name, path in ARTIFACT_PATHS.items():
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")

    artifacts[name] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }

manifest = {
    "schema_version": 1,
    "artifact_chain": [
        "canonical_parquet",
        "audited_metrics_csv",
        "latex_table",
        "decision_figure_png",
    ],
    "artifacts": artifacts,
    "paper_safe_interpretation": (
        "In the 11-row Hausa patching v3 artifact, zero of one evaluable "
        "clean-versus-corrupted decision changes recovered at the patched "
        "endpoint. This is a descriptive result with an evaluable denominator of one."
    ),
    "prohibited_overclaim": (
        "Do not generalize the observed 0/1 endpoint recovery result as an "
        "estimated population recovery rate or as evidence of broad patching failure."
    ),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

if not OUT.is_file() or OUT.stat().st_size == 0:
    raise RuntimeError("Manifest generation failed")

print(f"Wrote {OUT}")
for name, metadata in artifacts.items():
    print(f"{name}: {metadata['sha256']}  {metadata['path']}")
