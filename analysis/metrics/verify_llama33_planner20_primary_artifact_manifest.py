from hashlib import sha256
from pathlib import Path
import json

import polars as pl

MANIFEST = Path(
    "paper_artifacts/metrics/"
    "llama33_planner20_primary_artifact_manifest.json"
)

EXPECTED_ARTIFACTS = 15
EXPECTED_CSVS = 6
EXPECTED_TABLE_ROWS = 4
EXPECTED_HARMFUL_N = 1200
EXPECTED_BENIGN_N = 495
EXPECTED_HARMFUL_SIGNIFICANT = 3
EXPECTED_BENIGN_SIGNIFICANT = 0

EXPECTED_BENIGN_N_BY_LANGUAGE = {
    "Arabic": 150,
    "English": 150,
    "Hausa": 150,
    "Yoruba": 45,
}

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
    "llama33_planner20_primary_opus_handoff.md"
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

    if actual_sha256 != entry["sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {path}: "
            f"expected {entry['sha256']}, got {actual_sha256}."
        )

    if path.suffix == ".csv":
        expected_rows = entry.get("row_count")

        if expected_rows is None:
            raise ValueError(f"Manifest lacks row_count for {path}.")

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

if coverage.get("harmful_paired_prompt_n") != EXPECTED_HARMFUL_N:
    raise ValueError("Manifest does not record 1200 paired harmful prompts.")

if coverage.get("harmful_paired_prompt_n_per_language") != 300:
    raise ValueError(
        "Manifest does not record 300 paired harmful prompts per language."
    )

if coverage.get("benign_paired_prompt_n") != EXPECTED_BENIGN_N:
    raise ValueError("Manifest does not record 495 paired benign prompts.")

if (
    coverage.get("benign_paired_prompt_n_by_language")
    != EXPECTED_BENIGN_N_BY_LANGUAGE
):
    raise ValueError("Manifest records incorrect benign sample sizes.")

if coverage.get("partial_harmful_coverage_rows") != 0:
    raise ValueError("Manifest does not record zero partial harmful rows.")

if coverage.get("partial_benign_coverage_rows") != 1:
    raise ValueError("Manifest does not record one partial benign row.")

if (
    coverage.get("harmful_holm_significant_rows")
    != EXPECTED_HARMFUL_SIGNIFICANT
):
    raise ValueError("Manifest does not record three harmful significant rows.")

if (
    coverage.get("benign_holm_significant_rows")
    != EXPECTED_BENIGN_SIGNIFICANT
):
    raise ValueError("Manifest does not record zero benign significant rows.")

constraints = manifest.get("interpretation_constraints", [])

required_constraints = [
    "Harmful operational-failure and benign-refusal outcomes must be reported separately.",
    "Harmful comparisons have complete matched coverage: 300 prompts per language.",
    "Benign coverage differs by language: Yoruba has 45 matched prompts; Arabic, English, and Hausa each have 150.",
    "Report Holm-adjusted inference separately for the harmful and benign correction families.",
    "This primary package excludes Phase 2 Turbo results and cross-phase comparisons.",
    "Describe this as a within-phase, within-model prompt-matched intervention comparison; do not make an unsupported causal-identification claim.",
]

missing_constraints = [
    item for item in required_constraints if item not in constraints
]

if missing_constraints:
    raise ValueError(
        "Required interpretation constraints are absent: "
        f"{missing_constraints}"
    )

if any(
    "turbo" in entry["path"].lower()
    for entry in artifacts
):
    raise ValueError("Phase 2 Turbo artifact found in primary manifest.")

print(f"Verified manifest: {MANIFEST}")
print(f"Verified artifacts: {checked}")
print(f"Verified metric CSV row counts: {csv_checked}")
print(f"Paper-table rows: {EXPECTED_TABLE_ROWS}")
print(f"Paired harmful prompts: {EXPECTED_HARMFUL_N}")
print("Paired harmful prompts per language: 300")
print(f"Paired benign prompts: {EXPECTED_BENIGN_N}")
print("Paired benign prompts: Arabic=150, English=150, Hausa=150, Yoruba=45")
print("Partial harmful-coverage rows: 0")
print("Partial benign-coverage rows: 1")
print(f"Harmful Holm-significant rows: {EXPECTED_HARMFUL_SIGNIFICANT}")
print(f"Benign Holm-significant rows: {EXPECTED_BENIGN_SIGNIFICANT}")
print("Phase 2 Turbo artifacts included: False")
print("Harmful and benign outcomes required separately: True")
print("Causal-identification claim permitted: False")
