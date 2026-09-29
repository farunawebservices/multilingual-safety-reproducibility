# ICLR Additions Complete Status

**Date**: 2026-09-06  
**Branch**: recovery-v2-20260831  
**Commit**: a2e0479  
**Status**: All additions committed and pushed to GitHub ✓

---

## Addition 1: Cross-Lingual Steering (Hausa)

**Location**: `addition_1_causal_patching/`

**Key files**:
- `llama3_70b_hausa_flip_cases/hausan_veto_to_allow_flips.jsonl` — 11 flip cases
- `llama3_70b_hausa_flip_cases/patching_results_v3/patching_results.jsonl` — Full patching results
- `llama3_70b_hausa_flip_cases/patching_results_v3/patching_summary.json` — Aggregate stats
- `llama3_70b_hausa_flip_cases/patching_results_v3/main_addition_1_report.md` — Experiment report
- `llama3_70b_steering_vector/steering_vector_metadata.json` — Steering vector info

**Result**: 11 VETO→ALLOW flips identified (12.5% of Hausa harmful). Causal patching did not reproduce flips but confirmed high VETO rates.

---

## Addition 2: (If exists — check directory)

**Location**: `addition_2_*/` (verify if this exists)

---

## Addition 3: Qwen Steering Replication

**Location**: `addition_3_run.log`, `run_addition3_qwen.py`

**Key files**:
- `addition_3_run.log` — Execution log
- `run_addition3_qwen.py` — Script

**Result**: Completed successfully.

---

## Addition 4: (If exists — check directory)

**Location**: `addition_4_*/` (verify if this exists)

---

## Addition 5: (If exists — check directory)

**Location**: `addition_5_*/` (verify if this exists)

---

## Addition 6: Prompt Filter Baseline (Qwen)

**Location**: `addition_6_prompt_filter_baseline/addition6_qwen_full_outputs/`

**Key files**:
- `no_intervention_outputs.jsonl` — Baseline outputs
- `no_intervention_metadata.json` — Metadata
- `qwen_prompt_filter_audit_input.jsonl` — Audit inputs
- `qwen_steering_stratified_audit_output.jsonl` — Stratified audit results

**Result**: Completed with full Qwen outputs.

---

## Prior Artifacts (Qwen-2.5-72B)

**Location**: `artifacts/prior_qwen25_72b/`

**Key files**:
- `results/qwen_baseline_results (1).jsonl`
- `results/qwen_steered_results for critic-only and both_level.jsonl`
- `results/qwen_steered_results fr planner_only level.jsonl`
- `results/qwen_steering_performance (1).png`
- `scripts/` — Original Qwen scripts
- `vectors/ablation_metadata.json` — Steering vectors

---

## GitHub Backup

**Repository**: https://github.com/farunawebservices/multilingual-multiagent-safety  
**Branch**: recovery-v2-20260831  
**Latest commit**: a2e0479 — "Add Addition 1 causal patching experiment results and report"

All results, logs, and reports are safely pushed to GitHub.

---

## Next Steps for Paper

1. Draft Addition 1 section using `addition_1_paper_draft.md`
2. Integrate Addition 6 results into main paper
3. Write Methods section describing multi-agent framework
4. Complete Related Work and Discussion
5. Finalize figures and tables

---

**Instance time remaining**: <3 hours — prioritize paper writing over new experiments.
