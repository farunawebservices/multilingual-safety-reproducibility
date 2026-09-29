# Addition 4: Judge Calibration Results

## Scope

This directory contains the calibration exercise for the judge in Addition 4 of the multilingual multi-agent safety evaluation. Two independent annotators labeled 98 prompts across six languages, followed by blinded adjudication of any disagreements.

## Data

- `raw_annotations/annotator_1_completed.csv`: annotations from annotator 1 (98 rows)
- `raw_annotations/annotator_2_completed.csv`: annotations from annotator 2 (98 rows)
- `merged_annotations.jsonl`: row-wise merge of both annotators’ labels, plus agreement flag and Cohen’s kappa
- `raw_annotations/adjudicator_completed.csv`: blinded adjudicator’s decision for disagreement cases only
- `final_consensus_annotations.jsonl`: final consensus labels used for downstream analysis

## Agreement Results

- Total prompts: 98
- Initial agreement: 97 / 98 (98.98%)
- Initial disagreement: 1 / 98 (1.02%)
- Cohen’s kappa (two raters, before adjudication): **0.9796** (95% CI: 0.9482–1.0000)

The single disagreement occurred on a Yoruba prompt (`ann_098`), where annotator 1 labeled `NO` and annotator 2 labeled `YES`. The blinded adjudicator assigned `YES`, which is now the final consensus label.

## Interpretation

The extremely high kappa indicates that the two annotators interpreted the annotation instructions and the YES/NO definition in a highly consistent way across all six languages. However, because the annotators could see the original label column during annotation, this result should be interpreted as evidence of **high inter-rater consistency**, not as a fully unbiased estimate of annotation quality.

The blinded adjudicator step ensures that the final consensus labels used for evaluation are not determined solely by the initial annotators’ judgments in disagreement cases.

## Limitations

- Annotators were not blinded to the original labels; this may have inflated agreement.
- The calibration set is relatively small (98 prompts), so confidence intervals around kappa are wide.
- The adjudicator saw only disagreement cases; this design does not allow estimation of adjudicator reliability on agreement cases.

## Files to Cite

When referencing Addition 4 judge calibration in reports or papers, cite:

- `merged_annotations.jsonl` for the full two-rater dataset and kappa
- `final_consensus_annotations.jsonl` for the final consensus labels used in evaluation
- This `ADDITION_4_JUDGE_CALIBRATION_RESULTS.md` file for the methodology and interpretation
