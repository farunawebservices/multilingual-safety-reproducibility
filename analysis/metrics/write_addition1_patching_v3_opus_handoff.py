#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

MANIFEST = Path(
    "paper_artifacts/handoff/"
    "addition1_llama33_hausa_patching_v3_artifact_manifest.json"
)
OUT = Path(
    "paper_artifacts/handoff/"
    "addition1_llama33_hausa_patching_v3_OPUS_HANDOFF.md"
)

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

required = {
    "schema_version",
    "artifact_chain",
    "artifacts",
    "paper_safe_interpretation",
    "prohibited_overclaim",
}
missing = required - set(manifest)
if missing:
    raise ValueError(f"Manifest missing fields: {sorted(missing)}")

expected_chain = [
    "canonical_parquet",
    "audited_metrics_csv",
    "latex_table",
    "decision_figure_png",
]
if manifest["artifact_chain"] != expected_chain:
    raise ValueError("Unexpected artifact-chain order")

artifacts = manifest["artifacts"]
if set(artifacts) != set(expected_chain):
    raise ValueError("Manifest artifact keys do not match expected chain")

for name in expected_chain:
    metadata = artifacts[name]
    if not {"path", "bytes", "sha256"} <= set(metadata):
        raise ValueError(f"Manifest metadata incomplete for: {name}")

lines = [
    "# Opus Writing Handoff: Addition 1 Hausa Patching v3",
    "",
    "## Scope",
    "",
    "This handoff covers only the 11-row Hausa patching v3 artifact.",
    "",
    "## Paper-safe result",
    "",
    manifest["paper_safe_interpretation"],
    "",
    "## Required limitation",
    "",
    "Only one row was evaluable because recovery is defined only when clean and corrupted endpoint decisions differ.",
    "",
    "## Do not claim",
    "",
    manifest["prohibited_overclaim"],
    "",
    "## Citable artifacts",
    "",
]

for name in expected_chain:
    metadata = artifacts[name]
    lines.extend([
        f"- `{name}`",
        f"  - Path: `{metadata['path']}`",
        f"  - SHA-256: `{metadata['sha256']}`",
        f"  - Bytes: {metadata['bytes']}",
    ])

lines.extend([
    "",
    "## Writing instruction",
    "",
    "Describe the result as an observed endpoint outcome in this artifact, "
    "report the denominator explicitly as 0/1, and retain the limitation "
    "that the result is not a general population estimate.",
    "",
])

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT}")
