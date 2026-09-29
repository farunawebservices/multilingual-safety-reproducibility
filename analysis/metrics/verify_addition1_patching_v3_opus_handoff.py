#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST = Path(
    "paper_artifacts/handoff/"
    "addition1_llama33_hausa_patching_v3_artifact_manifest.json"
)
HANDOFF = Path(
    "paper_artifacts/handoff/"
    "addition1_llama33_hausa_patching_v3_OPUS_HANDOFF.md"
)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if not MANIFEST.is_file():
    raise FileNotFoundError(f"Missing manifest: {MANIFEST}")
if not HANDOFF.is_file():
    raise FileNotFoundError(f"Missing handoff: {HANDOFF}")

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
handoff = HANDOFF.read_text(encoding="utf-8")

required_chain = [
    "canonical_parquet",
    "audited_metrics_csv",
    "latex_table",
    "decision_figure_png",
]
if manifest.get("artifact_chain") != required_chain:
    raise ValueError("Manifest chain is missing or incorrectly ordered")

for name in required_chain:
    metadata = manifest["artifacts"][name]
    path = Path(metadata["path"])

    if not path.is_file():
        raise FileNotFoundError(f"Missing referenced artifact: {path}")
    if path.stat().st_size != metadata["bytes"]:
        raise ValueError(f"Byte-size mismatch for {name}: {path}")
    if sha256(path) != metadata["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {name}: {path}")

    if metadata["path"] not in handoff:
        raise ValueError(f"Handoff does not cite path for {name}")
    if metadata["sha256"] not in handoff:
        raise ValueError(f"Handoff does not cite hash for {name}")

for required_text in [
    "zero of one evaluable",
    "Only one row was evaluable",
    "Do not claim",
    "not a general population estimate",
]:
    if required_text not in handoff:
        raise ValueError(f"Handoff lacks required safeguard: {required_text!r}")

print("PASS: Addition 1 Hausa patching v3 Opus handoff verified.")
print(f"Manifest: {MANIFEST}")
print(f"Handoff: {HANDOFF}")
for name in required_chain:
    print(f"verified: {name}")
