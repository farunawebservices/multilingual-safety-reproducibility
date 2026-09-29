# Llama-3.3-70B-Instruct Primary Results: Review Handoff

## Review target

This package evaluates a within-phase, within-model, prompt-matched comparison for `meta-llama/llama-3.3-70b-instruct`.

- Baseline condition: planner-only configuration at alpha 0.
- Intervention condition: planner-only steering at alpha 20.
- Languages: Arabic, English, Hausa, and Yoruba.
- Harmful outcome: operational failure.
- Benign outcome: refusal rate.
- Raw source structure: one combined JSONL contains distinct alpha 0 and alpha 20 planner-only records; the raw-separation audit verifies exact raw-to-canonical subset matching.

## Authoritative artifacts

| Purpose | Path |
|---|---|
| Combined raw source | `artifacts/prior_llama33_phase2/results/baseline_completed_backup (1).jsonl` |
| Phase 1 canonical data | `data/derived/canonical/llama33_phase1_main.parquet` |
| Phase 1 run manifest | `registry/manifests/llama33_phase1_main.json` |
| Decision-semantics audit | `registry/llama/main_semantics_audit.tsv` |
| Decision-semantics console audit | `registry/llama/main_semantics_audit_console.tsv` |
| Harmful paired effects | `paper_artifacts/metrics/llama33_planner20_paired_effect.csv` |
| Harmful paired inference | `paper_artifacts/metrics/llama33_planner20_paired_inference.csv` |
| Benign paired effects | `paper_artifacts/metrics/llama33_planner20_benign_refusal_effect.csv` |
| Benign paired inference | `paper_artifacts/metrics/llama33_planner20_benign_refusal_inference.csv` |
| Raw-separation audit | `paper_artifacts/metrics/llama33_planner20_primary_raw_separation_audit.csv` |
| Paper-facing table | `paper_artifacts/metrics/llama33_planner20_primary_paper_table.csv` |
| LaTeX table | `paper_artifacts/tables/llama33_planner20_primary_results.tex` |
| Figure, vector format | `paper_artifacts/figures/llama33_planner20_primary_effects.pdf` |
| Figure, raster format | `paper_artifacts/figures/llama33_planner20_primary_effects.png` |
| Hash manifest | `paper_artifacts/metrics/llama33_planner20_primary_artifact_manifest.json` |

## Confirmed harmful results

Each language has 300 prompt-matched harmful examples, for 1,200 total.

- Arabic: 35.0% to 20.7%; change -14.3 percentage points; 95% paired-bootstrap CI [-21.0, -7.3]; Holm-adjusted p = 0.000225.
- English: 13.7% to 13.3%; change -0.3 percentage points; 95% paired-bootstrap CI [-4.3, 3.7]; Holm-adjusted p = 1.0.
- Hausa: 68.3% to 58.3%; change -10.0 percentage points; 95% paired-bootstrap CI [-17.0, -3.0]; Holm-adjusted p = 0.016762.
- Yoruba: 74.7% to 58.0%; change -16.7 percentage points; 95% paired-bootstrap CI [-22.3, -11.3]; Holm-adjusted p < 0.001.

Negative harmful changes indicate fewer operational failures under steering. Arabic, Hausa, and Yoruba improve significantly after Holm correction; English does not show a significant harmful-outcome change.

## Confirmed benign results

The benign outcome is refusal rate, so a positive change indicates additional false refusals.

- Arabic: 0.0% to 0.7%; change +0.7 percentage points; 150 paired examples; Holm-adjusted p = 1.0.
- English: 0.0% to 0.0%; change 0.0 percentage points; 150 paired examples; Holm-adjusted p = 1.0.
- Hausa: 0.0% to 1.3%; change +1.3 percentage points; 150 paired examples; Holm-adjusted p = 1.0.
- Yoruba: 0.0% to 0.0%; change 0.0 percentage points; 45 paired examples; Holm-adjusted p = 1.0.

No benign language-specific comparison is Holm-significant. Yoruba’s benign estimate uses 45 matched prompts, whereas the other three languages use 150; do not describe the benign arm as equally powered across languages.

## Review constraints

1. Report harmful operational failure and benign refusal as separate outcomes.
2. Do not combine harmful and benign sample sizes, rates, p-values, or correction families.
3. State that harmful coverage is complete: 300 matched prompts per language.
4. State benign denominators: Arabic 150, English 150, Hausa 150, Yoruba 45.
5. Report Holm-adjusted inference separately for the four harmful and four benign comparisons.
6. Do not claim a uniform harmful improvement: English is not significant.
7. Do not interpret absence of a significant benign effect as proof of no utility cost, especially for Yoruba.
8. Exclude Phase 2 Turbo results from this primary package.
9. Describe this as a within-phase, within-model, prompt-matched intervention comparison, not causal identification.

## Reproducibility check

Run:

```bash
python analysis/metrics/verify_llama33_planner20_primary_artifact_manifest.py
```

Expected verification summary:

- Verified artifacts: 14.
- Verified metric CSV row counts: 6.
- Paper-table rows: 4.
- Paired harmful prompts: 1200.
- Paired harmful prompts per language: 300.
- Paired benign prompts: 495.
- Paired benign prompts: Arabic=150, English=150, Hausa=150, Yoruba=45.
- Partial harmful-coverage rows: 0.
- Partial benign-coverage rows: 1.
- Harmful Holm-significant rows: 3.
- Benign Holm-significant rows: 0.
- Phase 2 Turbo artifacts included: False.
- Harmful and benign outcomes required separately: True.
- Causal-identification claim permitted: False.

## Claim boundary

Supported: in this within-phase, four-language prompt-matched comparison, planner-only steering at alpha 20 was associated with statistically significant reductions in harmful operational-failure rates in Arabic, Hausa, and Yoruba; no language-specific benign-refusal increase was significant after Holm correction.

Not supported by this package alone: uniform safety improvement across all languages, equivalently powered benign results across languages, absence of any benign utility cost, results for other roles or alpha settings, Phase 2 Turbo conclusions, or causal identification beyond the documented matched intervention comparison.
