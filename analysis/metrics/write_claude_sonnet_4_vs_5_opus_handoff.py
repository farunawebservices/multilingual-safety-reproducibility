from pathlib import Path
import json

ROOT = Path("paper_artifacts/metrics")

MANIFEST = ROOT / "claude_sonnet_4_vs_5_artifact_manifest.json"
OUT = ROOT / "claude_sonnet_4_vs_5_OPUS_HANDOFF.md"

if not MANIFEST.exists():
    raise FileNotFoundError(f"Missing manifest: {MANIFEST}")

manifest = json.loads(MANIFEST.read_text())

if manifest["coverage"]["paper_table_rows"] != 8:
    raise ValueError("Expected 8 paper-table rows.")

if manifest["coverage"]["matched_prompt_n"] != 1695:
    raise ValueError("Expected 1695 matched prompts.")

if manifest["coverage"]["partial_coverage_rows"] != 1:
    raise ValueError("Expected 1 partial-coverage row.")

if "descriptive" not in manifest["comparison_type"].lower():
    raise ValueError("Manifest does not label the comparison descriptive.")

constraints = manifest["interpretation_constraints"]
if not any("causal" in item.lower() for item in constraints):
    raise ValueError("Manifest is missing the causal-claim constraint.")

if not any("steering effect" in item.lower() for item in constraints):
    raise ValueError("Manifest is missing the steering-effect constraint.")

paper_table = manifest["authoritative_paper_table"]
latex_table = manifest["authoritative_latex_table"]
figure_pdf = manifest["authoritative_figure_pdf"]

for path in [paper_table, latex_table, figure_pdf]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Required handoff file missing: {path}")

text = f"""# Claude Sonnet 4 vs Claude Sonnet 5: Opus Handoff

## Authoritative artifacts

- Displayed results table (CSV): `{paper_table}`
- Paper table (LaTeX): `{latex_table}`
- Paper figure (PDF): `{figure_pdf}`
- Provenance and integrity manifest: `{MANIFEST}`

## Comparison type

This is a **prompt-matched descriptive comparison of separate model versions** across 1,695 shared prompt IDs.

## Required interpretation

- Do **not** describe this result as a within-model steering effect.
- Do **not** make a causal claim from this cross-model comparison.
- Outcome is veto status: `is_vetoed=true`.
- Do **not** apply or report the Qwen/Gemma Holm correction families for this comparison.
- Claude Sonnet 4 and Claude Sonnet 5 share all 1,695 IDs, and language plus harmfulness labels match for every ID.
- Yoruba benign is a partial-coverage cell with 45 shared prompts; all other cells use their complete matched coverage.

## Permitted descriptive statements

- The models agree on veto status for 96.70% of the 1,695 shared prompts.
- All benign cells have zero vetoes for both model versions.
- In harmful Arabic and English, Sonnet 5 has a lower observed veto rate than Sonnet 4 by 3.33 percentage points in each cell.
- In harmful Hausa and Yoruba, Sonnet 5 has higher observed veto rates by 0.33 and 2.33 percentage points, respectively.
- These are descriptive prompt-matched differences, not estimates of an intervention effect.

## Artifact inventory

| File | Kind | Rows (CSV only) | SHA-256 |
|---|---|---:|---|
"""

for item in manifest["artifacts"]:
    row_count = item.get("row_count", "")
    text += (
        f"| `{Path(item['path']).name}` | {item['kind']} | "
        f"{row_count} | `{item['sha256']}` |\n"
    )

text += f"""
## Manifest timestamp

`{manifest["created_utc"]}`
"""

OUT.write_text(text)

print(f"Wrote: {OUT}")
print(f"Manifested artifacts referenced: {len(manifest['artifacts'])}")
print("Displayed results rows: 8")
print("Matched prompts: 1695")
print("Partial-coverage rows: 1")
print("Inference claim allowed: False")
