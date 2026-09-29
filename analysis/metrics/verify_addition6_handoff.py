#!/usr/bin/env python3
"""Verify Addition 6 manifest, hashes, and Opus handoff."""

import hashlib
import json
import sys
from pathlib import Path

MANIFEST_JSON = Path(
    "paper_artifacts/metrics/"
    "addition6_prompt_filter_baseline_artifact_manifest.json"
)
HANDOFF_MD = Path(
    "paper_artifacts/metrics/"
    "addition6_prompt_filter_baseline_OPUS_HANDOFF.md"
)
METRICS_CSV = Path(
    "paper_artifacts/metrics/"
    "addition6_prompt_filter_baseline_metrics.csv"
)

REQUIRED_ARTIFACTS = [
    "audited_metrics_csv",
    "paper_table_csv",
    "paper_table_latex",
    "figure_pdf",
    "figure_png",
]

REQUIRED_CONCEPTS = [
    "does not establish",
    "superiority",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


errors = []

if not MANIFEST_JSON.exists():
    errors.append(f"Missing manifest: {MANIFEST_JSON}")

if not HANDOFF_MD.exists():
    errors.append(f"Missing handoff: {HANDOFF_MD}")

if errors:
    print("FAIL: Addition 6 verification FAILED")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)

manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
handoff = HANDOFF_MD.read_text(encoding="utf-8")
artifacts = manifest.get("artifacts", {})

if "opus_handoff_md" in artifacts:
    errors.append(
        "Manifest contains circular opus_handoff_md artifact; "
        "remove it from artifacts"
    )

for name in REQUIRED_ARTIFACTS:
    record = artifacts.get(name)
    if record is None:
        errors.append(f"Manifest missing artifact: {name}")
        continue

    path = Path(record["path"])
    expected_hash = record["sha256"]

    if not path.exists():
        errors.append(f"Missing artifact path for {name}: {path}")
        continue

    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        errors.append(
            f"Hash mismatch for {name}: "
            f"expected {expected_hash}, got {actual_hash}"
        )

    if str(path) not in handoff:
        errors.append(f"Handoff does not cite path for {name}")

    if expected_hash not in handoff:
        errors.append(f"Handoff does not cite hash for {name}")

if not METRICS_CSV.exists():
    errors.append(f"Missing metrics CSV: {METRICS_CSV}")
else:
    import csv

    with METRICS_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 3:
        errors.append(f"Expected 3 metric rows; found {len(rows)}")

    expected_rates = [52.5, 53.5, 57.5]
    try:
        actual_rates = [
            100.0 * float(row["harmful_veto_rate"])
            for row in rows
        ]
        for actual, expected in zip(actual_rates, expected_rates):
            if abs(actual - expected) > 0.1:
                errors.append(
                    "Harmful veto rate mismatch: "
                    f"{actual} vs {expected}"
                )
    except (KeyError, ValueError) as exc:
        errors.append(f"Could not validate harmful veto rates: {exc}")

lower_handoff = handoff.lower()
for concept in REQUIRED_CONCEPTS:
    if concept not in lower_handoff:
        errors.append(f"Handoff lacks required concept: {concept}")

if errors:
    print("FAIL: Addition 6 verification FAILED")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)

print("PASS: Addition 6 handoff verified")
print(f"Manifest: {MANIFEST_JSON}")
print(f"Handoff: {HANDOFF_MD}")
print(f"Verified artifacts: {len(REQUIRED_ARTIFACTS)}")
