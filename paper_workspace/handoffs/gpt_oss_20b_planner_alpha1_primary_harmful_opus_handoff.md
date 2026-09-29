# GPT-OSS-20B Primary Results: Review Handoff

## Review target

This package evaluates a within-model, prompt-matched intervention comparison for `openai/gpt-oss-20b`.

- Baseline condition: no steering.
- Intervention condition: planner-only steering at alpha 1.0.
- Primary scope: harmful prompts only.
- Languages: Arabic, English, Hausa, and Yoruba.
- Pairing: 300 matched harmful prompts per language, 1,200 total.
- Outcome: operational failure, using the registered GPT-OSS parser-outcome and decision-semantics audits.

## Authoritative artifacts

| Purpose | Path |
|---|---|
| Baseline canonical data | `data/derived/canonical/gpt_oss_20b_phase1_baseline.parquet` |
| Planner-only canonical data | `data/derived/canonical/gpt_oss_20b_phase1_planner_only_alpha1.parquet` |
| Canonical coverage audit | `paper_artifacts/metrics/gpt_oss_20b_planner_alpha1_primary_canonical_audit.csv` |
| Baseline metrics | `paper_artifacts/metrics/gpt_oss_20b_phase1_baseline_by_language.csv` |
| Baseline bootstrap intervals | `paper_artifacts/metrics/gpt_oss_20b_phase1_baseline_bootstrap_ci.csv` |
| Paired effects | `paper_artifacts/metrics/gpt_oss_20b_planner_alpha1_paired_effect.csv` |
| Paired bootstrap intervals | `paper_artifacts/metrics/gpt_oss_20b_planner_alpha1_paired_bootstrap_ci.csv` |
| Holm-adjusted inference | `paper_artifacts/metrics/gpt_oss_20b_planner_alpha1_paired_effect_holm.csv` |
| Paper-facing table | `paper_artifacts/metrics/gpt_oss_20b_planner_alpha1_primary_harmful_paper_table.csv` |
| LaTeX table | `paper_artifacts/tables/gpt_oss_20b_planner_alpha1_primary_harmful_results.tex` |
| Figure, vector format | `paper_artifacts/figures/gpt_oss_20b_planner_alpha1_primary_harmful_effects.pdf` |
| Figure, raster format | `paper_artifacts/figures/gpt_oss_20b_planner_alpha1_primary_harmful_effects.png` |
| Hash manifest | `paper_artifacts/metrics/gpt_oss_20b_planner_alpha1_primary_harmful_artifact_manifest.json` |

## Confirmed results

- Arabic: 14.3% to 10.0%; change -4.3 percentage points; Holm-adjusted p = 0.070164.
- English: 58.0% to 47.0%; change -11.0 percentage points; Holm-adjusted p = 0.004977.
- Hausa: 15.0% to 22.0%; change +7.0 percentage points; Holm-adjusted p = 0.000197.
- Yoruba: 16.0% to 16.0%; change 0.0 percentage points; Holm-adjusted p = 1.0.

Negative changes mean fewer operational failures under steering. English improves significantly after Holm correction; Hausa worsens significantly after Holm correction. Arabic improves numerically but does not meet the Holm-adjusted threshold, and Yoruba is unchanged.

## Review constraints

1. Do not describe the result as uniform improvement across languages.
2. Do not pool these language results into a single headline effect without a separately specified pooled analysis.
3. Do not generalize this primary result to benign prompts.
4. Do not generalize it to critic-only, joint, or other alpha settings.
5. Report Holm-adjusted p-values for significance claims.
6. Describe the result as a prompt-matched within-model intervention comparison, not as an unsupported causal-identification result.
7. Use the canonical audit to verify the 1,200 exact harmful-prompt matches.

## Reproducibility check

Run:

```bash
python analysis/metrics/verify_gpt_oss_20b_primary_artifact_manifest.py
```

Expected verification summary:

- Verified artifacts: 18.
- Verified metric CSV row counts: 7.
- Paper-table rows: 4.
- Paired harmful prompts: 1200.
- Paired harmful prompts per language: 300.
- Partial-coverage rows: 0.
- Holm-significant rows: 2.
- Improvement rows: 2.
- Worsening rows: 1.
- Harmful-only scope: True.
- Mixed-direction result required in interpretation: True.
- Causal-identification claim permitted: False.

## Claim boundary

Supported: in this four-language harmful-prompt comparison, planner-only steering at alpha 1.0 had heterogeneous associations with operational-failure outcomes, including a Holm-significant reduction in English and a Holm-significant increase in Hausa.

Not supported by this package alone: a general safety-improvement claim, a cross-language pooled claim, a benign-utility claim, or causal identification beyond the documented matched intervention comparison.
