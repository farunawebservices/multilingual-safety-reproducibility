from hashlib import sha256
from pathlib import Path
import json

ROOT = Path("paper_artifacts/metrics")

MANIFEST = ROOT / "gpt_5_frontier_baseline_artifact_manifest.json"
HANDOFF = ROOT / "gpt_5_frontier_baseline_OPUS_HANDOFF.md"

def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if not MANIFEST.exists():
    raise FileNotFoundError(f"Missing manifest: {MANIFEST}")

if not HANDOFF.exists():
    raise FileNotFoundError(f"Missing handoff: {HANDOFF}")

manifest = json.loads(MANIFEST.read_text())
handoff_text = HANDOFF.read_text()

if manifest["coverage"]["paper_table_rows"] != 8:
    raise ValueError("Manifest does not report 8 paper-table rows.")

if manifest["coverage"]["valid_record_n"] != 1651:
    raise ValueError("Manifest does not report 1651 valid records.")

if manifest["coverage"]["partial_coverage_rows"] != 5:
    raise ValueError("Manifest does not report 5 partial-coverage rows.")

if manifest["source_reconciliation"]["duplicate_subset_valid_records"] != 570:
    raise ValueError("Manifest does not report 570 duplicate-subset records.")

if manifest["source_reconciliation"]["duplicate_subset_parse_errors"] != 1:
    raise ValueError("Manifest does not report one duplicate-subset parse error.")

for key in [
    "authoritative_paper_table",
    "authoritative_latex_table",
    "authoritative_figure_pdf",
]:
    if manifest[key] not in handoff_text:
        raise ValueError(f"Handoff does not name {key}: {manifest[key]}")

required_phrases = [
    "descriptive single-configuration baseline summary",
    "Do **not** describe this baseline summary as a steering effect",
    "Do **not** make a causal or cross-condition comparison claim",
    "is_vetoed=true",
    "is_vetoed=false",
    "Do **not** pool the separate 570-record file",
    "Yoruba benign has 45 valid prompts",
    "Harmful cells contain 286--291 valid prompts",
]

for phrase in required_phrases:
    if phrase not in handoff_text:
        raise ValueError(
            f"Required handoff content missing: {phrase}"
        )

checked = 0
for item in manifest["artifacts"]:
    path = Path(item["path"])

    if not path.exists():
        raise FileNotFoundError(f"Manifested artifact missing: {path}")

    observed = sha256_file(path)
    if observed != item["sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {path}: "
            f"expected {item['sha256']}, got {observed}"
        )

    checked += 1

print("GPT-5 frontier-baseline handoff verification passed.")
print(f"Verified artifact hashes: {checked}")
print("Authoritative paper-table rows: 8")
print("Valid records: 1651")
print("Partial-coverage rows: 5")
print("Excluded duplicate-subset valid records: 570")
print("Comparison or inference claim allowed: False")
