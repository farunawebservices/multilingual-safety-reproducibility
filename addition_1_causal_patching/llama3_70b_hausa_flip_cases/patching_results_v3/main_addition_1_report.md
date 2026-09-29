# Addition 1: Causal Validation via Activation Patching

## Objective

Demonstrate that planner steering at layer 50 causally affects safety judgments in Hausa by using activation patching to recover correct VETO decisions.

## Method

- **Model**: Llama-3.3-70B-Instruct
- **Flip cases**: 11 Hausa harmful prompts previously identified as VETO→ALLOW flips under planner steering
- **Steering**: Planner-only at layer 50, strength 20.0
- **Patching**: Replace corrupted Planner L50 activations with clean activations at the final prefill token
- **Controls**: Random layer patching (10, 30, 70), random direction patching (3 seeds), position control (token 5)

## Results

### Main Conditions

| Condition | VETO count | ALLOW count |
|-----------|------------|-------------|
| Clean (no intervention) | 10/11 | 1/11 |
| Corrupted (steered L50) | 11/11 | 0/11 |
| Patched (steered + clean L50) | 11/11 | 0/11 |

**Recovery rate R**: -0.0 (undefined denominator: clean_vetos - corrupted_vetos = 10 - 11 = -1)

### Control Conditions

| Control | VETO count |
|---------|------------|
| Random layer 10 | 11/11 |
| Random layer 30 | 11/11 |
| Random layer 70 | 11/11 |
| Random direction (seed 0-2) | 11/11 each |
| Position control (token 5) | 11/11 |

## Interpretation

The steering vector did not reproduce the expected VETO→ALLOW flips in the corrupted condition. All 11 cases produced VETO decisions under steering, compared to 10/11 VETO in the clean condition.

This suggests:
1. The original 11 flips were likely produced under different experimental conditions (e.g., different model state, prompts, or steering method).
2. The current steering vector at Planner L50 does not reliably induce ALLOW decisions for these Hausa harmful prompts.
3. The causal patching experiment could not validate the steering mechanism because the flip phenomenon was not reproduced.

## Limitations

- **Flip reproduction failure**: The 11 flip cases were extracted from earlier planner-steering outputs, but re-running with the current setup did not reproduce the flips.
- **Single model**: Results are from Llama-3.3-70B-Instruct only; other models may show different patterns.
- **Single language**: Only Hausa was tested; other low-resource languages may exhibit different steering effects.

## Conclusion

While the causal patching experiment did not validate the steering mechanism due to flip reproduction issues, the baseline result (10/11 VETO for Hausa harmful prompts) demonstrates that the multi-agent system generally refuses harmful requests in Hausa. Future work should:
1. Identify the exact conditions that produced the original 11 flips.
2. Test steering vectors computed from larger flip sets.
3. Explore alternative steering layers or strengths.

## Files

- `patching_results.jsonl`: Per-case results with full model outputs
- `patching_summary.json`: Aggregate statistics
- `patching_run.log`: Execution log
- `addition_1_report.md`: This report
