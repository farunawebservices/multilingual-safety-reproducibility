# Opus Writing Handoff: Addition 1 Hausa Patching v3

## Scope

This handoff covers only the 11-row Hausa patching v3 artifact.

## Paper-safe result

In the 11-row Hausa patching v3 artifact, zero of one evaluable clean-versus-corrupted decision changes recovered at the patched endpoint. This is a descriptive result with an evaluable denominator of one.

## Required limitation

Only one row was evaluable because recovery is defined only when clean and corrupted endpoint decisions differ.

## Do not claim

Do not generalize the observed 0/1 endpoint recovery result as an estimated population recovery rate or as evidence of broad patching failure.

## Citable artifacts

- `canonical_parquet`
  - Path: `data/derived/canonical/addition1_llama33_hausa_patching_v3.parquet`
  - SHA-256: `482b20c89d322327d4f8fb144bd0723fc546561f1338c76e6bc93b235b141874`
  - Bytes: 61436
- `audited_metrics_csv`
  - Path: `paper_artifacts/metrics/addition1_llama33_hausa_patching_v3_audit.csv`
  - SHA-256: `689e479e774a09723a8d738953dfcb1a277c4b38e8083f49713d82a75542ca7f`
  - Bytes: 302
- `latex_table`
  - Path: `paper_artifacts/tables/addition1_llama33_hausa_patching_v3_table.tex`
  - SHA-256: `fb53ef7d21abfe34e03d2fd0a3d979ae2ac205318ffcbd06a510ca5a8c66404b`
  - Bytes: 624
- `decision_figure_png`
  - Path: `paper_artifacts/figures/addition1_llama33_hausa_patching_v3_decisions.png`
  - SHA-256: `5c584ed60a5fa8353d6a0c9510815a8ae93dd2824e01d7e9e35b6c7774ddd60d`
  - Bytes: 69105

## Writing instruction

Describe the result as an observed endpoint outcome in this artifact, report the denominator explicitly as 0/1, and retain the limitation that the result is not a general population estimate.
