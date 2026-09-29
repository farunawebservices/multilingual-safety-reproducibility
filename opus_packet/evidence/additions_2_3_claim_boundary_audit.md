# Additions 2 and 3: Claim-Boundary Audit

## Addition 2: Retrospective role-specificity analysis

Authoritative implementation:
`addition_2_matched_controls.py`

Observed inputs:
- Historical Turbo steering results:
  `artifacts/prior_llama33_phase2/results/turbo_steering_results_phase2.jsonl`
- SAE refusal feature:
  `artifacts/prior_llama33_phase2/vectors/sae_refusal_feature.pt`

Observed output:
`results/addition_2_matched_controls/analysis.json`

What the script does:
- Loads the real SAE refusal vector.
- Aggregates historical results by configuration, strength, and language.
- Prints Hausa veto rates for planner-only, critic-only, and both-role steering.
- Labels opposite role-dependent patterns as role specificity.

What the script does not do:
- It does not generate a random direction or random vector.
- It does not run a same-session matched random-vector control.
- It does not compare real-vector intervention against a matched random-vector intervention.
- It therefore does not identify or rule out the generic-perturbation null through a random-vector control.

Permitted phrasing:
"Retrospective analyses of the same steering direction across roles showed role-dependent outcome patterns, which is consistent with role sensitivity."

Prohibited phrasing:
"Matched random-vector controls ruled out generic perturbation."
"Addition 2 establishes a causal random-vector control."

## Addition 3: Hindi, Korean, and Swahili replication

Llama artifact:
`results/addition_3/results.jsonl`

Qwen artifact:
`results/addition_3/qwen25_72b_results.jsonl`

Common design:
- Hindi, Korean, and Swahili.
- 50 harmful and 50 benign prompts per language.
- 300 paired rows per model.
- Baseline and steered decision stored for each row.

Llama documented harmful VETO results:
- Hindi: 100.0% baseline and 100.0% steered.
- Korean: 88.0% baseline and 90.0% steered.
- Swahili: 94.0% baseline and 94.0% steered.
- Overall: 94.0% baseline and 94.7% steered.

Qwen documented harmful VETO results:
- Hindi: 86.0% baseline and 86.0% steered.
- Korean: 94.0% baseline and 92.0% steered.
- Swahili: 66.0% baseline and 68.0% steered.
- Overall: 82.0% baseline and 82.0% steered.

Interpretation limits:
- These are small, heterogeneous changes; they do not support a 70–80% recovery claim.
- Llama has high baseline harmful VETO rates in this set, which limits detectable improvement.
- Qwen has documented malformed outputs, especially in Swahili, which must remain visible in interpretation.
- The reports do not provide paired inferential tests, confidence intervals, or multiple-testing correction.

Permitted phrasing:
"On Hindi, Korean, and Swahili, effects were small and heterogeneous, with Llama near a harmful-VETO ceiling and Qwen showing mixed language-specific changes."

Prohibited phrasing:
"Addition 3 demonstrated 70–80% recovery."
"Addition 3 established reliable cross-lingual improvement."
