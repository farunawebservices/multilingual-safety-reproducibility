# Addition 2 Status and Claim Boundary

## Status

Addition 2 is retained as a retrospective role-specificity analysis.

It is not a completed matched random-vector control experiment.

## Evidence available

- Implementation: `addition_2_matched_controls.py`
- Historical input: `artifacts/prior_llama33_phase2/results/turbo_steering_results_phase2.jsonl`
- Real-vector metadata input: `artifacts/prior_llama33_phase2/vectors/sae_refusal_feature.pt`
- Intended derived output: `results/addition_2_matched_controls/analysis.json`
- Historical commit containing the implementation: `f5efb1ebadaa123da9fc01e2a8d9e8d187eee071`

## What can be claimed

The same historical steering direction produced role-dependent outcome patterns when applied to different agent roles. This observation is consistent with role sensitivity and supports treating intervention location as an empirical design variable.

## What cannot be claimed

- A matched random-vector control was run.
- Random-vector controls ruled out generic perturbation.
- Addition 2 independently establishes causality.
- The retrospective result alone demonstrates vector specificity.

## Paper wording

Preferred:
"Retrospective role-specificity analyses found different outcome patterns when the same steering direction was applied to different agent roles; this pattern is consistent with role sensitivity."

Avoid:
"Matched random-vector controls rule out generic perturbation."
"Addition 2 causally establishes role specificity."

## Scope

The verified Phase 1 Llama planner-only alpha-20 paired analysis remains the primary Llama evidence. Addition 2 is supplementary and must not expand the primary causal claim.
