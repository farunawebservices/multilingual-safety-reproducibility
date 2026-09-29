# Gemma 4 31B Phase 1: Opus Handoff

## Authoritative displayed-results table
Use:

`paper_artifacts/metrics/gemma_4_31b_phase1_complete_case_paper_table.csv`

This file contains the three paper-facing comparison rows.

## Reproducibility source
Use:

`paper_artifacts/metrics/gemma_4_31b_phase1_complete_case_unified_results.csv`

This preserves paired counts, discordant-pair counts, bootstrap intervals, exact McNemar p-values, Holm-adjusted p-values, coverage flags, source caveats, and semantic-validation fields.

## Required interpretation

- Primary outcome: **non-veto operational failure**, defined as `is_vetoed=false`.
- Do not label non-veto outcomes as explicit ALLOW: `ALLOW` tokens do not exhaust all non-veto decisions.
- In all three rows, steering increased observed non-veto failure.
- None of the three comparisons survived Holm correction across the three-cell Gemma family at family-wise alpha = 0.05.
- English benign is a matched-subset result: 101 shared valid IDs; 11 steered-only valid IDs were excluded from paired inference.
- Arabic benign and Arabic harmful have complete matching among valid JSON records.
- One truncated terminal JSONL record is documented; analyses use complete-case valid JSON records only.

## Files and integrity

| File | Rows | SHA-256 |
|---|---:|---|
| `gemma_4_31b_phase1_complete_case_paired_inference.csv` | 3 | `c1ad878df1c45d42a5d364a9493406051a54cb6f7fb753fabd70315864a9cb55` |
| `gemma_4_31b_phase1_complete_case_paired_inference_holm.csv` | 3 | `aa01feeafbbb68bf61845f55f581ea36f45a226c2b6f0575a51c87e85df12ca2` |
| `gemma_4_31b_phase1_complete_case_decision_semantics_audit.csv` | 6 | `cd1589675c6fa22d0b1535ffc0277298b3748058bf28873f04668ffaa37adf10` |
| `gemma_4_31b_phase1_complete_case_unified_results.csv` | 3 | `00c7aa4eef6aebc8f6e816e15af3c1958b79ddfd6b4006a18e7407087443e438` |
| `gemma_4_31b_phase1_complete_case_paper_table.csv` | 3 | `f6784b1b199948dcd27502a79f9a3b6aaaad42bf8593695887ac759ec1325c0d` |

## Exact artifact manifest

`paper_artifacts/metrics/gemma_4_31b_phase1_complete_case_artifact_manifest.json`

Generated from the artifact manifest timestamp:

`2026-09-08T18:22:41.680900+00:00`
