# Qwen 2.5 72B Primary Results: Review Handoff

## Review target

This handoff packages the paper-facing Qwen 2.5 72B primary comparison:

- Model: Qwen 2.5 72B.
- Comparison: within-model, prompt-matched intervention comparison.
- Conditions: baseline versus planner-only steering at alpha 1.0.
- Outcome: operational failure, using the audited Qwen decision semantics.
- Safety coding: explicit veto is safe; non-veto unresolved is operational failure.

## Authoritative artifacts

| Purpose | Path |
|---|---|
| Canonical-source audit | `paper_artifacts/metrics/qwen25_72b_primary_canonical_audit.csv` |
| Decision-semantics audit | `paper_artifacts/metrics/qwen25_72b_decision_semantics_audit.csv` |
| Pairing and coverage audit | `paper_artifacts/metrics/qwen25_72b_primary_pairing_coverage.csv` |
| Paired inference, unadjusted | `paper_artifacts/metrics/qwen25_72b_primary_paired_inference.csv` |
| Paired inference, Holm-adjusted | `paper_artifacts/metrics/qwen25_72b_primary_paired_inference_holm.csv` |
| Combined analysis table | `paper_artifacts/metrics/qwen25_72b_primary_unified_results.csv` |
| Test-family inventory | `paper_artifacts/metrics/qwen25_72b_primary_family_inventory.csv` |
| Paper-facing eight-row table | `paper_artifacts/metrics/qwen25_72b_planner_alpha1_primary_paper_table.csv` |
| LaTeX table | `paper_artifacts/tables/qwen25_72b_planner_alpha1_primary_results.tex` |
| Figure, vector format | `paper_artifacts/figures/qwen25_72b_planner_alpha1_primary_effects.pdf` |
| Figure, raster format | `paper_artifacts/figures/qwen25_72b_planner_alpha1_primary_effects.png` |
| Artifact manifest | `paper_artifacts/metrics/qwen25_72b_planner_alpha1_primary_artifact_manifest.json` |

## Confirmed scope

The paper table contains eight language-by-prompt-class rows and 1,695 paired prompts in total.

All eight displayed rows have complete matched coverage. Do not generalize the primary table to excluded alpha values, critic-only steering, or joint steering conditions.

The inference table evaluates a predeclared 33-cell Qwen family. Exact paired McNemar tests are reported with Holm adjustment. Bootstrap 95% intervals are computed over matched prompt IDs.

Negative steered-minus-baseline percentage-point values mean fewer operational failures under planner-only steering.

## Checks for reviewer

Verify the following before accepting a paper claim:

1. Every paper-table row is planner-only steering at alpha 1.0.
2. Every displayed row has complete matched coverage.
3. Report adjusted, rather than unadjusted, inference when discussing statistical significance.
4. Confirm any numerical claim against the authoritative paper-table CSV.
5. Treat the comparison as a prompt-matched within-model intervention comparison, not as an unsupported causal-identification result.
6. Use the manifest verifier to confirm that all artifact hashes and recorded CSV row counts still match.

## Reproducibility command

Run:

```bash
python analysis/metrics/verify_qwen25_72b_primary_artifact_manifest.py
```

Expected verification summary:

- Verified artifacts: 14.
- Verified metric CSV row counts: 11.
- Paper-table rows: 8.
- Paired prompts: 1695.
- Partial-coverage rows: 0.
- Holm-significant rows: 2.
- Causal-identification claim permitted: False.

## Claim boundaries

Supported: the audited, prompt-matched Qwen primary comparison reports condition-specific operational-failure differences for planner-only steering at alpha 1.0.

Not supported by this artifact package alone: a claim that steering has been causally identified to improve safety generally, a claim about conditions omitted from the primary table, or a pooled claim across intervention roles or alpha values.
