# Claude Sonnet 4 vs Claude Sonnet 5: Opus Handoff

## Authoritative artifacts

- Displayed results table (CSV): `paper_artifacts/metrics/claude_sonnet_4_vs_5_paper_table.csv`
- Paper table (LaTeX): `paper_artifacts/tables/claude_sonnet_4_vs_5_descriptive_comparison.tex`
- Paper figure (PDF): `paper_artifacts/figures/claude_sonnet_4_vs_5_veto_rate_comparison.pdf`
- Provenance and integrity manifest: `paper_artifacts/metrics/claude_sonnet_4_vs_5_artifact_manifest.json`

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
| `closed_model_source_inventory.csv` | metric_csv | 11 | `10237b0962f1cc421bf124e8be0f69c0e4f83a055ebe3d1eec071a5666639aca` |
| `closed_model_schema_inventory.csv` | metric_csv | 11 | `9a0309a555a9e898e6ea43e8970397c637309b9dd54c17e1d7791a37796bddf3` |
| `claude_sonnet_4_phase1_schema_profile.csv` | metric_csv | 5 | `f1def937861fb3712bfa1ce2c9be6cf514a8aa9e26d7b8812e40dd3f197000f6` |
| `claude_sonnet_4_phase1_baseline_audit.csv` | metric_csv | 8 | `53b6785fac8dc140a7178f12996010e3a3b9630365bc60b4ef8e889074d4af24` |
| `claude_sonnet_4_vs_5_prompt_coverage_audit.csv` | metric_csv | 8 | `21b08521e73f9707b9b64e8c0c10c8f3bc95e3a8a6a115b0b2e22a8334eea269` |
| `claude_sonnet_4_vs_5_metadata_consistency_audit.csv` | metric_csv | 1 | `bd246d340959defe2d3e501c2010653cf4265fb2128ffd1fb4d2e7dd7359c52c` |
| `claude_sonnet_4_vs_5_outcome_comparison.csv` | metric_csv | 8 | `788cb5a48899aafad806be6b711b75f6f4dfaa791872b2e07cde528c8f763d8f` |
| `claude_sonnet_4_vs_5_paper_table.csv` | metric_csv | 8 | `a657226e6660d853039b7c4e1ba2cb20997a000d09e172adf0cfc44d8e7cf422` |
| `claude_sonnet_4_vs_5_descriptive_comparison.tex` | latex_table |  | `1b393eccfc533b9bbdfb52b50f95de3f0f266e9a5d77abd62315199d31e7b3b8` |
| `claude_sonnet_4_vs_5_veto_rate_comparison.pdf` | figure |  | `9bf499e140a0062d2e6a40ffe0069d243b9653dbacaefa49e325b902ea135e9c` |
| `claude_sonnet_4_vs_5_veto_rate_comparison.png` | figure |  | `5eba0df0d0ca1d96c7845a1ef51168b6ac6cd9a7e5c5fe92458a590da44fce93` |

## Manifest timestamp

`2026-09-08T18:57:58.588235+00:00`
