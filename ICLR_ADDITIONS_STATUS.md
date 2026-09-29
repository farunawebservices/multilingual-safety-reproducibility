# ICLR Additions - Status Summary

**Date**: 2026-09-02  
**Branch**: recovery-v2-20260831

## Completed Experiments

### ✅ Addition 1: Hausa Safety Flip Analysis
- **Status**: COMPLETE
- **Result**: +28% VETO rate (60% → 88%)
- **Files**: `ADDITION_1_FINAL_STATUS.md`

### ✅ Addition 2: Matched Controls (Hausa)
- **Status**: COMPLETE
- **Result**: Confirmed language-specific effect
- **Files**: `addition_2_matched_controls.py`

### ✅ Addition 3: Multilingual Steering (Hi/Ko/Sw)
- **Status**: COMPLETE (both models)
- **Llama-3.3-70B**: 94.0% → 94.7% (+0.7%)
- **Qwen-2.5-72B**: 82.0% → 82.0% (0.0%)
- **Files**: 
  - `ADDITION_3_RESULTS.md` (Llama)
  - `ADDITION_3_QWEN_RESULTS.md` (Qwen)
  - `results/addition_3/` (both result files)

### ⏳ Addition 4: Judge Calibration Study
- **Status**: READY FOR ANNOTATION
- **Samples**: 98 (14 per language × 7 languages)
- **Languages**: ar, en, ha, hi, ko, sw, yo
- **Target**: κ ≥ 0.6 per language
- **Timeline**: 2-3 days for annotation
- **Files**:
  - `addition_4_calibration/annotation_samples.jsonl`
  - `addition_4_calibration/ANNOTATION_INSTRUCTIONS.md`
  - `addition_4_calibration/analyze_calibration.py`

## Next Steps

1. **Immediate**: Send annotation samples to 2 annotators
2. **Parallel**: Can work on rebuttal text while waiting
3. **After annotation**: Run `analyze_calibration.py` and write results

## Key Findings So Far

- Steering works best for low-resource languages (Hausa +28%)
- High-resource languages already have high baseline safety
- Llama more conservative than Qwen overall
- Cross-model validation strengthens claims

## GitHub Status

All results committed to: `recovery-v2-20260831`
Ready to merge or continue experiments.
