# GPT-5 Frontier Baseline: Opus Handoff

## Authoritative artifacts

- Displayed results table (CSV): `paper_artifacts/metrics/gpt_5_frontier_baseline_paper_table.csv`
- Paper table (LaTeX): `paper_artifacts/tables/gpt_5_frontier_baseline_descriptive_results.tex`
- Paper figure (PDF): `paper_artifacts/figures/gpt_5_frontier_baseline_veto_rates.pdf`
- Provenance and integrity manifest: `paper_artifacts/metrics/gpt_5_frontier_baseline_artifact_manifest.json`

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
| `closed_model_source_inventory.csv` | metric_csv | 11 | `10237b0962f1cc421bf124e8be0f69c0e4f83a055ebe3d1eec071a5666639aca` |
| `closed_model_schema_inventory.csv` | metric_csv | 11 | `9a0309a555a9e898e6ea43e8970397c637309b9dd54c17e1d7791a37796bddf3` |
| `gpt_5_source_reconciliation.csv` | metric_csv | 7 | `e35d5a49da32ca269fe4bf9465e8ba8f523fff75d08c451b8e962414db0fafe8` |
| `gpt_5_jsonl_compatibility_audit.csv` | metric_csv | 11 | `1aa6d289960c6f107688f91bf9b0f22482f472f4737d14606efcc5097890d7db` |
| `gpt_5_clean_frontier_baseline_audit.csv` | metric_csv | 8 | `6b502e10724dd1dce0418ba4ad34531cfe1e06deeae350f8bbbcf4fea9413d37` |
| `gpt_5_clean_vs_partial_overlap_audit.csv` | metric_csv | 2 | `f1c928b32446cea07d8222170a2d8a9acca9b0f676c859b176651b1f0ce3d48e` |
| `gpt_5_frontier_baseline_paper_table.csv` | metric_csv | 8 | `9ce860c83a269e0a187ab6d967900d495cc3b9c40f28c32646a8941fd41a2a69` |
| `gpt_5_frontier_baseline_descriptive_results.tex` | latex_table |  | `f78df835af2c47d10c046b81840782096dc66d39e6bc6438bc9459c126c8fa19` |
| `gpt_5_frontier_baseline_veto_rates.pdf` | figure |  | `6f1d192fae08364893fdcb3ef398b7f678f40adb6e3c1aee7a28d365477400f5` |
| `gpt_5_frontier_baseline_veto_rates.png` | figure |  | `688567dcb6b3f0046021597512abe032bfbebe62f64d1a2b01217c0c8e7b39ea` |

## Manifest timestamp

`2026-09-08T19:12:00.783071+00:00`
