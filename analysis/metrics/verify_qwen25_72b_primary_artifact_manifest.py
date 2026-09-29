from hashlib import sha256
from pathlib import Path
import json

import polars as pl

MANIFEST = Path(
    "paper_artifacts/metrics/"
    "qwen25_72b_planner_alpha1_primary_artifact_manifest.json"
)

def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if not MANIFEST.exists():
    raise FileNotFoundError(f"Missing manifest: {MANIFEST}")

manifest = json.loads(MANIFEST.read_text())
artifacts = manifest.get("artifacts", [])

if len(artifacts) != 15:
    raise ValueError(f"Expected 15 manifest artifacts; found {len(artifacts)}.")

handoff_path = (
    "paper_artifacts/metrics/"
    "qwen25_72b_planner_alpha1_primary_opus_handoff.md"
)

if handoff_path not in {entry["path"] for entry in artifacts}:
    raise ValueError("Verified Opus handoff is absent from the manifest.")

checked = 0
csv_checked = 0

for entry in artifacts:
    path = Path(entry["path"])

    if not path.exists():
        raise FileNotFoundError(f"Missing manifested artifact: {path}")

    actual_sha256 = sha256_file(path)
    expected_sha256 = entry["sha256"]

    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )

    if path.suffix == ".csv":
        actual_rows = pl.read_csv(path).height
        expected_rows = entry.get("row_count")

        if expected_rows is None:
            raise ValueError(f"Manifest is missing row_count for {path}.")

        if actual_rows != expected_rows:
            raise ValueError(
                f"Row-count mismatch for {path}: "
                f"expected {expected_rows}, got {actual_rows}."
            )

        csv_checked += 1

    checked += 1

coverage = manifest.get("coverage", {})

if coverage.get("paper_table_rows") != 8:
    raise ValueError("Manifest does not record 8 primary paper-table rows.")

if coverage.get("paired_prompt_n") != 1695:
    raise ValueError("Manifest does not record 1695 paired prompts.")

if coverage.get("partial_coverage_rows") != 0:
    raise ValueError("Manifest does not record zero partial-coverage rows.")

if coverage.get("holm_significant_rows") != 2:
    raise ValueError("Manifest does not record two Holm-significant rows.")

constraints = manifest.get("interpretation_constraints", [])
required_constraint = (
    "Describe this as a within-model prompt-matched intervention comparison; "
    "do not make an unsupported causal-identification claim."
)

if required_constraint not in constraints:
    raise ValueError("Required causal-language constraint is absent.")

print(f"Verified manifest: {MANIFEST}")
print(f"Verified artifacts: {checked}")
print(f"Verified metric CSV row counts: {csv_checked}")
print("Paper-table rows: 8")
print("Paired prompts: 1695")
print("Partial-coverage rows: 0")
print("Holm-significant rows: 2")
print("Causal-identification claim permitted: False")
