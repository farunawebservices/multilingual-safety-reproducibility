# Addition 2: Llama 3.3 Turbo Alpha 30 Opus Handoff

## Purpose
This handoff provides the authoritative artifact chain and claim boundaries for writing about Addition 2 (archived Llama 3.3 Phase 2 "Turbo" role/strength reanalysis).

**Critical framing:** Addition 2 is an **archived cross-phase role/strength reanalysis**, NOT a new causal experiment. All comparisons are **associational**, not causal.

## Authoritative numerical source
All numerical reporting MUST come from:
- `paper_artifacts/metrics/llama33_turbo_alpha30_role_strength_unified_results.csv`
  - SHA-256: `cf386d2aa06c27717771720343590798cd7bd21e89dff0c07645c189cc4655f4`
  - Six rows: 3 configurations × 2 languages, 300 matched harmful prompts per cell (1,800 total)

## Complete artifact chain
1. Canonical Parquet: `data/derived/canonical/llama33_phase1_main.parquet`, `data/derived/canonical/llama33_phase2_turbo.parquet`
2. Audited metrics: `paper_artifacts/metrics/llama33_turbo_alpha30_role_strength_unified_results.csv` (SHA: `cf386d2aa06c2771...`)
3. Paper table CSV: `paper_artifacts/metrics/llama33_turbo_alpha30_paper_table.csv` (SHA: `48c551adecf6955c...`)
4. Paper table LaTeX: `paper_artifacts/tables/llama33_turbo_alpha30_cross_phase_comparison.tex`
5. Figure PDF: `paper_artifacts/figures/llama33_turbo_alpha30_cross_phase_comparison.pdf`
6. Figure PNG: `paper_artifacts/figures/llama33_turbo_alpha30_cross_phase_comparison.png`
7. Manifest: `paper_artifacts/metrics/llama33_turbo_artifact_manifest.csv`

## Outcome definition
**Operational failure = explicit ALLOW + NO_DECISION**

## Allowed qualitative summary
"Across archived, record-ID-matched harmful-prompt outputs at Alpha 30, operational failure rates differed by intervention configuration and language. Planner-only steering was associated with lower failure rates than matched Alpha 0 outputs in Hausa and Yoruba, whereas critic-only and dual-role steering showed heterogeneous associations. Because the matched conditions originate from distinct experiment phases, these comparisons are associational rather than causal."

## Explicitly forbidden claims
- Causal effects (do not use "causes", "caused", "causal", "effect of", "impact of")
- Role specificity or vector specificity mechanisms
- Random-vector controls were used (NOT available)
- That Phase 1 and Phase 2 run conditions were equivalent

## Cross-phase caveat (REQUIRED)
"Matched by record_id; phases differ so non-prompt run-condition equivalence is not established by pairing alone. These comparisons are cross-phase and associational, not causal."

## Placement recommendation
Appendix or secondary analysis, NOT a headline main-text causal result.

## No original vector file available
The original `.pt` refusal vector file is unavailable. Do not claim otherwise.
