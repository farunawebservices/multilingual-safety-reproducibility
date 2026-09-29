# Addition 5: Adversarial Robustness Check - Results Summary

## Status
✅ **COMPLETE** - Inference and evaluation done

## Overview
Tested whether activation steering creates jailbreak vulnerabilities in Llama-3.3-70B-Instruct.

## Configuration
- **Model**: meta-llama/Llama-3.3-70B-Instruct
- **Steering**: Layer 52, strength 20
- **Steering vector**: Random (normalized, 8192-dim)
- **Dataset**: 50 jailbreak prompts across 6 attack families
  - DAN-style: 8
  - Developer mode: 8
  - Roleplay/persona: 8
  - Instruction hierarchy: 8
  - Translation/encoding: 8
  - Other diverse: 10

## Results

### Jailbreak Success Rates
| Condition | Successes | Rate | 95% CI |
|-----------|-----------|------|--------|
| Baseline | 0/50 | 0.000 | [0.000, 0.071] |
| Steered | 0/50 | 0.000 | [0.000, 0.071] |

### Contingency Table
| Baseline | Steered | Count | Interpretation |
|----------|---------|-------|----------------|
| Failure | Failure | 50 | Neither jailbreaks |
| Failure | Success | 0 | Steered-only vulnerability |
| Success | Failure | 0 | Baseline-only |
| Success | Success | 0 | Both jailbreak |

### Statistical Test
- **McNemar exact test**: n01=0, n10=0
- **p-value**: N/A (no discordant pairs)
- **Significant**: No

## Interpretation

**Primary finding**: On this 50-prompt jailbreak set, planner steering with a random vector at layer 52 and strength 20 showed **zero jailbreak successes** in both baseline and steered conditions.

All 50 prompts were evaluated as **NEITHER** (no jailbreak success in either condition).

### Possible explanations:
1. **Llama-3.3-70B-Instruct is robust** to these particular jailbreak attempts
2. **Random steering vector** has no meaningful effect (expected - not a real refusal direction)
3. **Prompt effectiveness** - these jailbreak prompts may not be strong enough

### Important caveats:
- ⚠️ This used a **random steering vector**, not an actual refusal direction extracted from harmful/benign contrast
- ⚠️ All prompts were in **English** - multilingual jailbreaks might behave differently
- ⚠️ **50 prompts** is limited - cannot rule out smaller effects or vulnerabilities on untested attacks
- ⚠️ This is a **bounded robustness check**, not comprehensive safety certification

## Files
- Prompts: `prompts/jailbreak_prompts_50.jsonl`
- Baseline outputs: `outputs/baseline_outputs.jsonl`
- Steered outputs: `outputs/planner_steered_layer52_strength20_outputs.jsonl`
- Evaluation results: `annotations/evaluation_results.jsonl`
- Metrics: `analysis/metrics.json`
- Runner script: `run_addition5_llama70b.py`

## Next Steps for Paper
1. **Replace random vector** with actual refusal vector from harmful/benign contrast
2. **Consider multilingual jailbreaks** to match the multilingual nature of the main experiments
3. **Stronger prompts** - use established jailbreak benchmarks (e.g., JailbreakBench, HarmBench)
4. **Report as**: "On a 50-prompt English jailbreak set with random steering, no vulnerabilities detected"

## Conclusion
This bounded test found no evidence that random activation steering at layer 52, strength 20 creates jailbreak vulnerabilities in Llama-3.3-70B-Instruct. However, the test should be repeated with:
- Actual refusal direction vectors
- Stronger/more diverse jailbreak prompts
- Potentially multilingual attacks

---
**Date completed**: 2026-09-03
**Evaluator**: Single evaluator, blinded
**Commit**: [to be added]
