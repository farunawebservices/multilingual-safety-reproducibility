# Addition 4: Judge Calibration Opus Handoff

## Purpose
This handoff provides the authoritative artifact chain for writing about Addition 4 (judge calibration study) in the ICLR paper.

## Nature of Addition 4
Addition 4 is a **descriptive inter-rater reliability study**, NOT a model intervention experiment. It measures annotation agreement quality, not model behavior changes.

## Authoritative numerical source
All numerical reporting MUST come from:
- `paper_artifacts/metrics/addition4_judge_calibration_metrics.csv`
  - SHA-256: `b1d476512529cb0b3ebc211ed3921495abe55b8222bcce1b34b8ae1f3c48e476`
  - 8 rows: 7 languages + overall
  - 98 total prompts (14 per language)

## Complete artifact chain
1. **Raw annotation data:** `addition_4_calibration/final_consensus_annotations.jsonl` (SHA: `a636d6e5677f5615...`)
2. **Metrics CSV:** `paper_artifacts/metrics/addition4_judge_calibration_metrics.csv`
3. **Paper table CSV:** `paper_artifacts/metrics/addition4_calibration_paper_table.csv` (SHA: `6621096d89e69fcc...`)
4. **Paper table LaTeX:** `paper_artifacts/tables/addition4_calibration_agreement.tex`
5. **Figure PDF:** `paper_artifacts/figures/addition4_calibration_kappa_by_language.pdf`
6. **Figure PNG:** `paper_artifacts/figures/addition4_calibration_kappa_by_language.png`
7. **Methodology doc:** `addition_4_calibration/ADDITION_4_JUDGE_CALIBRATION_RESULTS.md`

## Key results
- **Overall Cohen's κ = 0.98** (95% CI: 0.94–1.00)
- **Observed agreement: 99%** (97/98 prompts)
- **1 disagreement** resolved by blinded adjudication
- **All languages** meet κ ≥ 0.60 target
- **Interpretation:** "Almost perfect" inter-rater agreement

## Allowed qualitative summary
"Two independent annotators achieved almost perfect agreement (Cohen's κ = 0.98, 95% CI: 0.94–1.00) across 98 prompts in 7 languages. A single disagreement was resolved by blinded adjudication. All languages exceeded the κ ≥ 0.60 target, indicating the annotation scheme was interpreted consistently."

## Explicitly forbidden claims
- Do NOT claim this measures model performance
- Do NOT claim this is a causal intervention
- Do NOT generalize beyond these 98 specific prompts
- Do NOT claim annotators were blinded to original labels (they were NOT)

## Limitations (REQUIRED in all discussions)
"Annotators were not blinded to the original labels; this may have inflated agreement. The calibration set is relatively small (98 prompts). The adjudicator saw only disagreement cases."

## Placement recommendation
Methods appendix or reproducibility section, NOT a main results section.

## Citation format
Cite the metrics CSV path and SHA-256 hash when reporting numbers.
