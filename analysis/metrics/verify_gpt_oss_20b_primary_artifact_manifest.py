from hashlib import sha256
from pathlib import Path
import json

import polars as pl

MANIFEST = Path(
    "paper_artifacts/metrics/"
    "gpt_oss_20b_planner_alpha1_primary_harmful_artifact_manifest.json"
)

EXPECTED_ARTIFACTS = 19
EXPECTED_CSVS = 7
EXPECTED_TABLE_ROWS = 4
EXPECTED_PAIRED_HARMFUL_PROMPTS = 1200
EXPECTED_HOLM_SIGNIFICANT = 2
EXPECTED_IMPROVEMENTS = 2
EXPECTED_WORSENINGS = 1

def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if not MANIFEST.exists():
    raise FileNotFoundError(f"Missing manifest: {MANIFEST}")

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
artifacts = manifest.get("artifacts", [])

if len(artifacts) != EXPECTED_ARTIFACTS:
    raise ValueError(
        f"Expected {EXPECTED_ARTIFACTS} manifest artifacts; "
        f"found {len(artifacts)}."
    )

handoff_path = (
    "paper_artifacts/metrics/"
    "gpt_oss_20b_planner_alpha1_primary_harmful_opus_handoff.md"
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
        expected_rows = entry.get("row_count")

        if expected_rows is None:
            raise ValueError(f"Manifest is missing row_count for {path}.")

        actual_rows = pl.read_csv(path).height

        if actual_rows != expected_rows:
            raise ValueError(
                f"Row-count mismatch for {path}: "
                f"expected {expected_rows}, got {actual_rows}."
            )

        csv_checked += 1

    checked += 1

if csv_checked != EXPECTED_CSVS:
    raise ValueError(
        f"Expected {EXPECTED_CSVS} verified CSV row counts; "
        f"found {csv_checked}."
    )

coverage = manifest.get("coverage", {})

if coverage.get("paper_table_rows") != EXPECTED_TABLE_ROWS:
    raise ValueError("Manifest does not record four paper-table rows.")

if (
    coverage.get("paired_harmful_prompt_n")
    != EXPECTED_PAIRED_HARMFUL_PROMPTS
):
    raise ValueError("Manifest does not record 1200 paired harmful prompts.")

if coverage.get("paired_harmful_prompt_n_per_language") != 300:
    raise ValueError("Manifest does not record 300 paired prompts per language.")

if coverage.get("partial_coverage_rows") != 0:
    raise ValueError("Manifest does not record zero partial-coverage rows.")

if coverage.get("holm_significant_rows") != EXPECTED_HOLM_SIGNIFICANT:
    raise ValueError("Manifest does not record two Holm-significant rows.")

if coverage.get("improvement_rows_negative_pp") != EXPECTED_IMPROVEMENTS:
    raise ValueError("Manifest does not record two improvement rows.")

if coverage.get("worsening_rows_positive_pp") != EXPECTED_WORSENINGS:
    raise ValueError("Manifest does not record one worsening row.")

constraints = manifest.get("interpretation_constraints", [])

required_constraints = [
    "Primary table includes harmful prompts only.",
    "Primary table includes planner-only steering at alpha 1.0 only.",
    "All four displayed language rows have complete prompt-matched coverage.",
    "The result is mixed by language: improvements and worsening must both be reported.",
    "Describe this as a within-model prompt-matched intervention comparison; do not make an unsupported causal-identification claim.",
]

missing_constraints = [
    item for item in required_constraints if item not in constraints
]

if missing_constraints:
    raise ValueError(
        "Required interpretation constraints are absent: "
        f"{missing_constraints}"
    )

print(f"Verified manifest: {MANIFEST}")
print(f"Verified artifacts: {checked}")
print(f"Verified metric CSV row counts: {csv_checked}")
print(f"Paper-table rows: {EXPECTED_TABLE_ROWS}")
print(f"Paired harmful prompts: {EXPECTED_PAIRED_HARMFUL_PROMPTS}")
print("Paired harmful prompts per language: 300")
print("Partial-coverage rows: 0")
print(f"Holm-significant rows: {EXPECTED_HOLM_SIGNIFICANT}")
print(f"Improvement rows (negative pp): {EXPECTED_IMPROVEMENTS}")
print(f"Worsening rows (positive pp): {EXPECTED_WORSENINGS}")
print("Harmful-only scope: True")
print("Mixed-direction result required in interpretation: True")
print("Causal-identification claim permitted: False")
