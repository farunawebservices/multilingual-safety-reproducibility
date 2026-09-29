# Addition 3: Multilingual Steering Results (Hindi, Korean, Swahili)

## Experiment Details
- **Model**: Llama-3.3-70B-Instruct
- **Layer**: 52, **Strength**: 20
- **Languages**: Hindi (hi), Korean (ko), Swahili (sw)
- **Prompts**: 50 harmful + 50 benign per language (300 total)
- **Date**: 2026-09-01
- **Runtime**: ~15 minutes

## Key Results

### Harmful Prompt VETO Rates

| Language | Baseline | Steered | Δ |
|----------|----------|---------|---|
| Hindi | 50/50 (100.0%) | 50/50 (100.0%) | 0.0% |
| Korean | 44/50 (88.0%) | 45/50 (90.0%) | +2.0% |
| Swahili | 47/50 (94.0%) | 47/50 (94.0%) | 0.0% |
| **Overall** | **141/150 (94.0%)** | **142/150 (94.7%)** | **+0.7%** |

### Benign Prompt False VETO Rates

| Language | Baseline FP | Steered FP | Δ |
|----------|-------------|------------|---|
| Hindi | 45/50 (90.0%) | 47/50 (94.0%) | +4.0% |
| Korean | 29/50 (58.0%) | 24/50 (48.0%) | -10.0% |
| Swahili | 47/50 (94.0%) | 45/50 (90.0%) | -4.0% |

## Key Findings

1. **Ceiling Effect**: Hindi and Swahili already have very high baseline VETO rates (94-100%), leaving little room for improvement.

2. **Korean Improvement**: Korean showed a modest +2% improvement (88% → 90%), with 2 specific cases where steering changed ALLOW→VETO:
   - ID: 7891b52860adcc353a32
   - ID: 9ebc9d6d62c8a357c280

3. **False Positive Patterns**:
   - Korean has the lowest benign FP rate (58% baseline, 48% steered)
   - Hindi and Swahili have very high FP rates (90-94%), suggesting overly conservative safety behavior
   - Steering slightly reduces FP for Korean (-10%) and Swahili (-4%), but increases for Hindi (+4%)

4. **No MALFORMED outputs**: All 300 prompts produced valid VETO/ALLOW decisions.

## Interpretation

The steering vector provides **marginal benefits** for languages that already have high safety compliance. The +0.7% overall improvement suggests:

- The refusal feature is already well-activated in Llama-3.3-70B for these languages
- Most safety violations are already caught at baseline
- Steering may be more beneficial for languages with lower baseline safety (e.g., Korean at 88%)

## Files
- `results/addition_3/results.jsonl` - Full results (300 lines)
- `addition_3_run.log` - Execution log

## Next Steps
- Run same experiment on Qwen-2.5-72B for cross-model comparison
- Proceed to Addition 4 (Judge Calibration Study)
