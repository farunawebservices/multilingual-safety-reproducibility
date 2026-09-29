from hashlib import sha256
from pathlib import Path
import json

ROOT = Path("paper_artifacts/metrics")

MANIFEST = ROOT / "claude_sonnet_4_vs_5_artifact_manifest.json"
HANDOFF = ROOT / "claude_sonnet_4_vs_5_OPUS_HANDOFF.md"

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

if manifest["coverage"]["matched_prompt_n"] != 1695:
    raise ValueError("Manifest does not report 1695 matched prompts.")

if manifest["coverage"]["partial_coverage_rows"] != 1:
    raise ValueError("Manifest does not report 1 partial-coverage row.")

for path_key in [
    "authoritative_paper_table",
    "authoritative_latex_table",
    "authoritative_figure_pdf",
]:
    path = manifest[path_key]
    if path not in handoff_text:
        raise ValueError(f"Handoff does not name {path_key}: {path}")

required_phrase_groups = {
    "descriptive-comparison label": [
        "prompt-matched descriptive comparison",
    ],
    "no causal claim": [
        "Do **not** make a causal claim",
        "Do not make a causal claim",
    ],
    "no steering-effect claim": [
        "Do **not** describe this result as a within-model steering effect",
        "Do not describe this result as a within-model steering effect",
    ],
    "outcome definition": [
        "is_vetoed=true",
    ],
    "partial-coverage caveat": [
        "Yoruba benign is a partial-coverage cell with 45 shared prompts",
    ],
    "Holm-family caveat": [
        "Do **not** apply or report the Qwen/Gemma Holm correction families",
        "Do not apply or report the Qwen/Gemma Holm correction families",
    ],
}

for check_name, acceptable_phrases in required_phrase_groups.items():
    if not any(phrase in handoff_text for phrase in acceptable_phrases):
        raise ValueError(
            f"Required handoff content missing: {check_name}. "
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

print("Claude Sonnet 4 vs 5 handoff verification passed.")
print(f"Verified artifact hashes: {checked}")
print("Authoritative paper-table rows: 8")
print("Matched prompts: 1695")
print("Partial-coverage rows: 1")
print("Inference claim allowed: False")
