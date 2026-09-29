#!/usr/bin/env python3
"""Write Addition 6 artifact manifest."""

import hashlib
import json
from pathlib import Path

METRICS_CSV = Path(
    "paper_artifacts/metrics/"
    "addition6_prompt_filter_baseline_metrics.csv"
)
TABLE_CSV = Path(
    "paper_artifacts/metrics/"
    "addition6_prompt_filter_baseline_table.csv"
)
TABLE_TEX = Path(
    "paper_artifacts/tables/"
    "addition6_prompt_filter_baseline.tex"
)
FIGURE_PDF = Path(
    "paper_artifacts/figures/"
    "addition6_prompt_filter_baseline.pdf"
)
FIGURE_PNG = Path(
    "paper_artifacts/figures/"
    "addition6_prompt_filter_baseline.png"
)
HANDOFF_MD = Path(
    "paper_artifacts/metrics/"
    "addition6_prompt_filter_baseline_OPUS_HANDOFF.md"
)
MANIFEST_JSON = Path(
    "paper_artifacts/metrics/"
    "addition6_prompt_filter_baseline_artifact_manifest.json"
)
MANIFEST_PATH = Path(
    "addition_6_prompt_filter_baseline/prompts/"
    "addition6_full_manifest_v1.jsonl"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


required_paths = {
    "canonical_source": MANIFEST_PATH,
    "audited_metrics_csv": METRICS_CSV,
    "paper_table_csv": TABLE_CSV,
    "paper_table_latex": TABLE_TEX,
    "figure_pdf": FIGURE_PDF,
    "figure_png": FIGURE_PNG,
}

missing = [str(path) for path in required_paths.values() if not path.exists()]
if missing:
    raise SystemExit("Missing required files:\n" + "\n".join(missing))

manifest = {
    "addition": "Addition 6: Prompt Filter Baseline",
    "description": (
        "Three-condition comparison: no intervention, prompt filter, "
        "SAE steering"
    ),
    "canonical_source": {
        "path": str(MANIFEST_PATH),
        "sha256": sha256(MANIFEST_PATH),
    },
    "artifacts": {
        "audited_metrics_csv": {
            "path": str(METRICS_CSV),
            "sha256": sha256(METRICS_CSV),
        },
        "paper_table_csv": {
            "path": str(TABLE_CSV),
            "sha256": sha256(TABLE_CSV),
        },
        "paper_table_latex": {
            "path": str(TABLE_TEX),
            "sha256": sha256(TABLE_TEX),
        },
        "figure_pdf": {
            "path": str(FIGURE_PDF),
            "sha256": sha256(FIGURE_PDF),
        },
        "figure_png": {
            "path": str(FIGURE_PNG),
            "sha256": sha256(FIGURE_PNG),
        },
    },
    "opus_handoff": {
        "path": str(HANDOFF_MD),
        "note": (
            "Narrative handoff intentionally excluded from hashed artifacts "
            "to avoid a circular dependency."
        ),
    },
}

MANIFEST_JSON.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

print(f"Wrote: {MANIFEST_JSON}")
print(f"Artifacts: {len(manifest['artifacts'])}")
