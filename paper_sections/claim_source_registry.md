# Claim-to-source registry

This registry defines the source of every quantitative claim intended for the revised main paper.

| Claim | Canonical source | Status | Interpretation constraint |
|---|---|---|---|
| Llama Planner-only alpha-20 harmful effects | `paper_artifacts/metrics/llama33_planner20_paired_inference.csv` | Primary | Within-phase, within-model; harmful family of four |
| Llama Planner-only alpha-20 benign effects | `paper_artifacts/metrics/llama33_planner20_benign_refusal_inference.csv` | Primary | Separate benign family; Yoruba n=45 |
| Llama alpha-30 role-strength effects | `paper_artifacts/metrics/llama33_turbo_alpha30_role_strength_paired_inference.csv` | Secondary | Cross-phase, record-ID-matched; six-test Holm family |
| Llama decision composition | `paper_artifacts/metrics/llama33_turbo_role_strength_decision_composition.csv` | Secondary descriptive | Operational failure = ALLOW + no-decision |
| Qwen Planner-only alpha-1 effects | `paper_artifacts/metrics/qwen25_72b_planner_alpha1_primary_paper_table.csv` | Primary | 33-cell Holm family; model-specific semantics |
| Gemma complete-case effects | `paper_artifacts/metrics/gemma_4_31b_phase1_complete_case_paper_table.csv` | Primary | Three-cell family; reduced English benign matching |
| GPT-OSS Planner-only alpha-1 effects | `paper_artifacts/metrics/gpt_oss_20b_planner_alpha1_primary_harmful_paper_table.csv` | Primary | Four-cell harmful family |
| Judge calibration | `paper_artifacts/metrics/addition4_calibration_paper_table.csv` | Exploratory audit | Annotation agreement, not full pipeline validation |
| Adversarial audit | `paper_artifacts/metrics/addition5_adversarial_robustness_metrics.csv` | Exploratory audit | Zero jailbreaks in both conditions; no differential estimate |
| Prompt-filter comparison | `paper_artifacts/metrics/addition6_prompt_filter_baseline_metrics.csv` | Exploratory | Harmful VETO and missing labels; separate Llama addition |
| Causal-patching attempt | `ADDITION_1_FINAL_STATUS.md` and `paper_artifacts/metrics/addition1_llama33_hausa_patching_v3_audit.csv` | Inconclusive | Do not use for causal claims |

All percentages in the paper are formatted directly from canonical values. The manuscript must not apply a second multiplication by 100 to CSV fields already stored in percentage units.