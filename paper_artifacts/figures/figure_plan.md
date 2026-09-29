# ICLR 2027 figure plan

## Figure 1: Pipeline and role-specific overview

**Message:** Intervention location changes the direction of observed effects.

Panels: Planner--Tool-Caller--Critic pipeline; Planner-only/Critic-only/Both intervention locations; compact Hausa/Yoruba alpha-30 effect plot.

Caption opening: “Role and language change the direction of the Llama alpha-30 contrast.”

## Figure 2: Llama role-strength effects

**Message:** Planner-only improves both Hausa and Yoruba, Critic-only worsens both, and joint steering is language-dependent.

Use `paper_artifacts/metrics/llama33_turbo_alpha30_role_strength_paired_inference.csv`. Plot percentage-point effects and paired bootstrap intervals. Mark Holm decisions within the six-test family. Label the comparison cross-phase.

## Figure 3: Decision composition

**Message:** Aggregate operational-failure changes coincide with shifts in explicit VETO, explicit ALLOW, and no-decision outputs.

Use `paper_artifacts/metrics/llama33_turbo_role_strength_decision_composition.csv`. Plot stacked bars for baseline, Planner-only, Critic-only, and Both at alpha 20 and alpha 30 for Hausa and Yoruba.

## Figure 4: Canonical cross-model primary effects

**Message:** Planner-only effects transfer incompletely across models and languages.

Use the canonical Llama, Qwen, Gemma, and GPT-OSS primary inference CSVs. Keep model-specific outcome definitions in panel labels or caption.

## Figure 5: Secondary audit trade-offs

**Message:** Higher refusal rates can accompany parser and benign-cost trade-offs.

Use Addition 6 only if its final model identity and source manifest are clearly labeled. Otherwise move it to the supplement.

All figures should be generated from canonical CSVs. Avoid manually transcribing percentages and include exact $n$ and correction-family notes in captions.