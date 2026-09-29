from pathlib import Path
import json

ROOT = Path("paper_artifacts/metrics")

MANIFEST = ROOT / "gpt_5_frontier_baseline_artifact_manifest.json"
OUT = ROOT / "gpt_5_frontier_baseline_OPUS_HANDOFF.md"

if not MANIFEST.exists():
    raise FileNotFoundError(f"Missing manifest: {MANIFEST}")

manifest = json.loads(MANIFEST.read_text())

if manifest["coverage"]["paper_table_rows"] != 8:
    raise ValueError("Expected 8 paper-table rows.")

if manifest["coverage"]["valid_record_n"] != 1651:
    raise ValueError("Expected 1651 valid records.")

if manifest["coverage"]["partial_coverage_rows"] != 5:
    raise ValueError("Expected 5 partial-coverage rows.")

if manifest["source_reconciliation"]["duplicate_subset_valid_records"] != 570:
    raise ValueError("Expected 570 duplicate-subset valid records.")

if manifest["source_reconciliation"]["duplicate_subset_parse_errors"] != 1:
    raise ValueError("Expected one duplicate-subset parse error.")

paper_table = manifest["authoritative_paper_table"]
latex_table = manifest["authoritative_latex_table"]
figure_pdf = manifest["authoritative_figure_pdf"]

for path in [paper_table, latex_table, figure_pdf]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Required handoff file missing: {path}")

text = f"""# GPT-5 Frontier Baseline: Opus Handoff

## Authoritative artifacts

- Displayed results table (CSV): `{paper_table}`
- Paper table (LaTeX): `{latex_table}`
- Paper figure (PDF): `{figure_pdf}`
- Provenance and integrity manifest: `{MANIFEST}`

## Analysis scope

This is a **descriptive single-configuration baseline summary** for `openai/gpt-5`, using `frontier_baseline` at strength `0.0`.

## Required interpretation

- Do **not** describe this baseline summary as a steering effect.
- Do **not** make a causal or cross-condition comparison claim.
- Outcome definitions: veto is `is_vetoed=true`; non-veto operational failure is `is_vetoed=false`.
- The authoritative dataset contains 1,651 valid, unique-ID records across eight language-by-harmfulness cells.
- Yoruba benign has 45 valid prompts.
- Harmful cells contain 286--291 valid prompts rather than 300.
- Do **not** pool the separate 570-record file into analysis; it is an exact duplicate subset of the clean source, has one malformed terminal JSONL line, and contributes no new IDs or outcomes.

## Permitted descriptive statements

- The paper table reports observed veto counts, veto rates, and non-veto operational-failure rates by language and harmfulness.
- Arabic harmful has the largest observed veto rate among the harmful cells, while English harmful has the lowest.
- English benign has three vetoes; the other benign cells have zero observed vetoes.
- These are descriptive rates from one baseline configuration, with the reported cell denominators.

## Artifact inventory

| File | Kind | Rows (CSV only) | SHA-256 |
|---|---|---:|---|
"""

for item in manifest["artifacts"]:
    text += (
        f"| `{Path(item['path']).name}` | {item['kind']} | "
        f"{item.get('row_count', '')} | `{item['sha256']}` |\n"
    )

text += f"""
## Manifest timestamp

`{manifest["created_utc"]}`
"""

OUT.write_text(text)

print(f"Wrote: {OUT}")
print(f"Manifested artifacts referenced: {len(manifest['artifacts'])}")
print("Displayed results rows: 8")
print("Valid records: 1651")
print("Partial-coverage rows: 5")
print("Excluded duplicate-subset valid records: 570")
print("Comparison or inference claim allowed: False")
