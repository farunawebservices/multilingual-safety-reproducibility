# Addition 3: Multilingual Steering Results on Qwen-2.5-72B (Hindi, Korean, Swahili)

## Experiment Details
- **Model**: Qwen/Qwen2.5-72B-Instruct
- **Layer**: 55 (optimal from ablation study, metric=94.64)
- **Strength**: 20.0
- **Vector**: `artifacts/prior_qwen25_72b/vectors/qwen_dense_refusal_layer_55.pt`
- **Languages**: Hindi (hi), Korean (ko), Swahili (sw)
- **Prompts**: 50 harmful + 50 benign per language (300 total)
- **Date**: 2026-09-02
- **Runtime**: ~5 minutes

## Key Results

### Harmful Prompt VETO Rates

| Language | Baseline | Steered | Δ |
|----------|----------|---------|---|
| Hindi | 43/50 (86.0%) | 43/50 (86.0%) | 0.0% |
| Korean | 47/50 (94.0%) | 46/50 (92.0%) | -2.0% |
| Swahili | 33/50 (66.0%) | 34/50 (68.0%) | +2.0% |
| **Overall** | **123/150 (82.0%)** | **123/150 (82.0%)** | **0.0%** |

### Benign Prompt False VETO Rates

| Language | Baseline FP | Steered FP | Δ |
|----------|-------------|------------|---|
| Hindi | 20/50 (40.0%) | 18/50 (36.0%) | -4.0% |
| Korean | 18/50 (36.0%) | 18/50 (36.0%) | 0.0% |
| Swahili | 16/50 (32.0%) | 17/50 (34.0%) | +2.0% |

### MALFORMED Outputs

| Language | Baseline | Steered |
|----------|----------|---------|
| Hindi | 0 | 1 (benign) |
| Korean | 1 (harmful) | 1 (harmful) |
| Swahili | 3 (harmful) + 7 (benign) | 4 (harmful) + 3 (benign) |

## Key Findings

1. **No Overall Improvement**: Steering on Qwen-2.5-72B shows 0.0% overall change in harmful VETO rate (82.0% → 82.0%).

2. **Language-Specific Effects**:
   - **Hindi**: No change (86.0% → 86.0%)
   - **Korean**: Slight decrease (94.0% → 92.0%), 1 case worsened
   - **Swahili**: Slight increase (66.0% → 68.0%), 1 case improved

3. **Lower Baseline than Llama**: Qwen has lower baseline safety than Llama-3.3-70B:
   - Qwen: 82.0% overall harmful VETO
   - Llama: 94.0% overall harmful VETO
   - Difference: -12 percentage points

4. **Better Benign FP Rates**: Qwen has much lower false positive rates:
   - Qwen: 36.0% average FP
   - Llama: 80.7% average FP
   - Qwen is more discriminating

5. **Swahili MALFORMED Issues**: Swahili has high MALFORMED rates (10-20%), suggesting the model struggles with this language in the safety critic task.

## Cross-Model Comparison

| Metric | Llama-3.3-70B | Qwen-2.5-72B |
|--------|---------------|--------------|
| Baseline Harmful VETO | 94.0% | 82.0% |
| Steered Harmful VETO | 94.7% | 82.0% |
| Steering Δ | +0.7% | 0.0% |
| Baseline Benign FP | 80.7% | 36.0% |
| Steered Benign FP | 77.3% | 35.3% |

**Interpretation**: Llama-3.3-70B is more conservative (higher VETO rates overall), while Qwen-2.5-72B is more permissive but also more accurate on benign prompts.

## Files
- `results/addition_3/qwen25_72b_results.jsonl` - Full results (300 lines)
- `addition_3_qwen_run.log` - Execution log
- `run_addition3_qwen_v2.py` - Execution script

## Next Steps
- Proceed to Addition 4 (Judge Calibration Study)
- Consider whether steering strength needs tuning for Qwen
- Investigate Swahili MALFORMED cases
