from pathlib import Path
import json

ROOT = Path("paper_artifacts/metrics")

MANIFEST = ROOT / "gemma_4_31b_phase1_complete_case_artifact_manifest.json"
OUT = ROOT / "gemma_4_31b_phase1_complete_case_OPUS_HANDOFF.md"

if not MANIFEST.exists():
    raise FileNotFoundError(f"Missing manifest: {MANIFEST}")

manifest = json.loads(MANIFEST.read_text())

if manifest["coverage"]["paper_table_rows"] != 3:
    raise ValueError("Expected 3 paper-table rows.")

if manifest["coverage"]["partial_coverage_rows"] != 1:
    raise ValueError("Expected 1 partial-coverage row.")

if manifest["inference"]["holm_significant_rows"] != 0:
    raise ValueError("Expected 0 Holm-significant rows.")

paper_table = manifest["authoritative_paper_table"]
artifacts = {Path(x["path"]).name: x for x in manifest["artifacts"]}

required = [
    "gemma_4_31b_phase1_complete_case_paper_table.csv",
    "gemma_4_31b_phase1_complete_case_unified_results.csv",
    "gemma_4_31b_phase1_complete_case_artifact_manifest.json",
]

for name in required:
    path = ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Required handoff file missing: {path}")

text = f"""# Gemma 4 31B Phase 1: Opus Handoff

## Authoritative displayed-results table
Use:

`{paper_table}`

This file contains the three paper-facing comparison rows.

## Reproducibility source
Use:

`paper_artifacts/metrics/gemma_4_31b_phase1_complete_case_unified_results.csv`

This preserves paired counts, discordant-pair counts, bootstrap intervals, exact McNemar p-values, Holm-adjusted p-values, coverage flags, source caveats, and semantic-validation fields.

## Required interpretation

- Primary outcome: **non-veto operational failure**, defined as `is_vetoed=false`.
- Do not label non-veto outcomes as explicit ALLOW: `ALLOW` tokens do not exhaust all non-veto decisions.
- In all three rows, steering increased observed non-veto failure.
- None of the three comparisons survived Holm correction across the three-cell Gemma family at family-wise alpha = 0.05.
- English benign is a matched-subset result: 101 shared valid IDs; 11 steered-only valid IDs were excluded from paired inference.
- Arabic benign and Arabic harmful have complete matching among valid JSON records.
- One truncated terminal JSONL record is documented; analyses use complete-case valid JSON records only.

## Files and integrity

| File | Rows | SHA-256 |
|---|---:|---|
"""

for item in manifest["artifacts"]:
    text += (
        f"| `{Path(item['path']).name}` | "
        f"{item['row_count']} | `{item['sha256']}` |\n"
    )

text += f"""
## Exact artifact manifest

`{MANIFEST}`

Generated from the artifact manifest timestamp:

`{manifest["created_utc"]}`
"""

OUT.write_text(text)

print(f"Wrote: {OUT}")
print(f"Manifested CSV artifacts: {len(artifacts)}")
print("Displayed results rows: 3")
print("Partial-coverage rows: 1")
print("Holm-significant rows: 0")
