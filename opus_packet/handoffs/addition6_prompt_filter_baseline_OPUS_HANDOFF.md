# Addition 6: Prompt Filter Baseline Opus Handoff

## Purpose

Addition 6 compares three conditions on the specified evaluation prompt set:

1. No intervention.
2. Prompt filter baseline.
3. SAE steering.

All numerical claims must be taken from the audited metrics CSV and associated
paper table.

## Key results

| Condition | Harmful veto rate | Benign FPR | Missing labels |
|---|---:|---:|---:|
| No intervention | 52.5% (210/400) | 0.0% (0/60) | 4 |
| Prompt filter | 53.5% (214/400) | 0.0% (0/60) | 5 |
| SAE steering | 57.5% (230/400) | 5.0% (3/60) | 25 |

SAE steering increased the harmful veto rate by 5.0 percentage points relative
to no intervention and by 4.0 percentage points relative to the prompt-filter
baseline. It also introduced three benign false positives and more missing
labels.

## Interpretation boundary

This three-condition comparison evaluates outcomes for the specified prompt set,
prompt-filter implementation, SAE steering intervention, model configuration,
and evaluation protocol. It does not establish that SAE steering is causally
superior to prompt filtering in other models, languages, datasets, prompt
distributions, or deployment settings. The results therefore do not establish
general superiority of SAE steering over prompt filtering.

## Artifact provenance

- `audited_metrics_csv`: `paper_artifacts/metrics/addition6_prompt_filter_baseline_metrics.csv`
  SHA-256: `c85ca2aa8b4cf4123aebc0fe66bc9dfd3e495d5256b23e402413523033d7e7a9`
- `paper_table_csv`: `paper_artifacts/metrics/addition6_prompt_filter_baseline_table.csv`
  SHA-256: `a52bda22c0a02c8a73621ed4f4436db0f35ec31501f370500677485a9b7554d7`
- `paper_table_latex`: `paper_artifacts/tables/addition6_prompt_filter_baseline.tex`
  SHA-256: `82e2455b2d118d8c10f33b009ab76f04e589610fe1eb4f2b58fa9b4627a969e0`
- `figure_pdf`: `paper_artifacts/figures/addition6_prompt_filter_baseline.pdf`
  SHA-256: `46697c26a46ee169179bc200186e73ee532714dbf375877c19bbe8d56ef8769e`
- `figure_png`: `paper_artifacts/figures/addition6_prompt_filter_baseline.png`
  SHA-256: `4ee7457d2361789f05db1e4ecc5738c9607e58f257fc3d2fb70aca799887b1a2`

## Writing guidance

Present Addition 6 as a baseline comparison with trade-offs, not as a
universal success claim. Retain the increased benign false-positive rate and
missing-label count when discussing the steering condition.
