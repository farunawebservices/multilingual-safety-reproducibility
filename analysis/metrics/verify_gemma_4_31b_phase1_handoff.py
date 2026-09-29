from hashlib import sha256
from pathlib import Path
import json

ROOT = Path("paper_artifacts/metrics")

MANIFEST = ROOT / "gemma_4_31b_phase1_complete_case_artifact_manifest.json"
HANDOFF = ROOT / "gemma_4_31b_phase1_complete_case_OPUS_HANDOFF.md"

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

if manifest["coverage"]["paper_table_rows"] != 3:
    raise ValueError("Manifest does not report 3 paper-table rows.")

if manifest["coverage"]["partial_coverage_rows"] != 1:
    raise ValueError("Manifest does not report 1 partial-coverage row.")

if manifest["inference"]["holm_significant_rows"] != 0:
    raise ValueError("Manifest does not report 0 Holm-significant rows.")

if manifest["authoritative_paper_table"] not in handoff_text:
    raise ValueError("Handoff does not name the authoritative paper table.")

required_phrase_groups = {
    "outcome definition": [
        "non-veto operational failure",
    ],
    "explicit-ALLOW caveat": [
        "not be interpreted as explicit ALLOW",
        "Do not label non-veto outcomes as explicit ALLOW",
    ],
    "Holm result": [
        "None of the three comparisons survived Holm correction",
    ],
    "English coverage caveat": [
        "101 shared valid IDs",
    ],
    "source truncation caveat": [
        "One truncated terminal JSONL record",
    ],
}

for check_name, acceptable_phrases in required_phrase_groups.items():
    if not any(phrase in handoff_text for phrase in acceptable_phrases):
        raise ValueError(
            f"Required handoff caveat missing: {check_name}. "
            f"Expected one of: {acceptable_phrases}"
        )

checked = 0
for item in manifest["artifacts"]:
    path = Path(item["path"])

    if not path.exists():
        raise FileNotFoundError(f"Manifested artifact missing: {path}")

    observed_hash = sha256_file(path)

    if observed_hash != item["sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {path}: "
            f"expected {item['sha256']}, got {observed_hash}"
        )

    checked += 1

print("Gemma Phase 1 handoff verification passed.")
print(f"Verified artifact hashes: {checked}")
print("Authoritative paper-table rows: 3")
print("Partial-coverage rows: 1")
print("Holm-significant rows: 0")
