# Paper Outline — Multilingual Safety Steering (claim-register-bound)

**Register coverage note (read before use).**
- Claim register contains exactly 11 claim_ids: `LL-A, LL-B, LL-C, QW-A, QW-B, QW-C, GM-A, G5-A, CS-A, GO-A, GO-B`.
- **No claim_id in the register supports subsections 6.1 (Addition 1 Hausa patching), 6.3 (Addition 3 Hindi/Korean/Swahili replication), 6.4 (Addition 4 judge calibration), 6.5 (Addition 5 adversarial robustness), or 6.6 (Addition 6 prompt-filter baseline).** These are retained as placeholders only and are marked NOT WRITABLE.
- 6.2 (Addition 2 role-specificity / matched controls) has only `LL-C`, whose `forbidden_wording` explicitly bars role-specificity, vector-specificity, and random-vector-control claims. 6.2 is therefore **rescoped** to an associational cross-phase role/strength comparison, or dropped.
- `GO-B`'s `allowed_wording` field is **truncated in the register** ("State only that…"). GO-B may not be written until the register field is restored; treat as NOT WRITABLE-pending.
- Global forbidden artifacts (declared defective / absent, must not be cited as evidence): `metrics/llama33_turbo_alpha30_paper_table.csv`, `tables/llama33_turbo_alpha30_cross_phase_comparison.tex`; all CSVs listed in the register as "NOT in packet."

---

## 1. Introduction

- **Research question:** Does planner-targeted activation steering change multilingual harmful/benign decision outcomes, and does the effect differ across languages and models?
- **Claim_ids covered:** none newly asserted. Forward-references only: `LL-A`, `LL-B`, `QW-A`, `QW-B`, `GM-A`, `GO-A`, `G5-A`, `CS-A`.
- **Artifacts / rows cited:** none directly; all numbers deferred to §4–§6 with their source tables.
- **Estimand(s) and scope:** statement of scope only — 5 open-weight models + 2 closed-model baselines; Arabic, English, Hausa, Yoruba; Phase 1 unless explicitly labelled cross-phase.
- **May claim:** that the paper reports prompt-matched within-model comparisons (Llama, Qwen, Gemma, GPT-OSS) and descriptive single/cross-version baselines (GPT-5, Sonnet 4 vs 5); that per-model correction families are separate.
- **Must not claim:** any pooled cross-model effect; any causal or mechanistic claim; any coverage of Hindi/Korean/Swahili, patching, judge calibration, adversarial robustness, or prompt-filter baselines; any combination of harmful and benign rates or families.

## 2. Related work

- **Research question:** How do prior activation-steering, refusal-direction, and multilingual-safety results frame the present measurements?
- **Claim_ids covered:** none.
- **Artifacts / rows cited:** none.
- **Estimand(s) and scope:** N/A (literature only).
- **May claim:** positioning statements about prior work; that this packet contains no mechanistic-localisation evidence.
- **Must not claim:** that the present results replicate, confirm, or extend any mechanistic refusal-direction finding; that a refusal vector artifact is available (`LL-C`: original `.pt` vector declared unavailable).

---

## 3. Experimental setup

### 3.1 Models and phases

- **Research question:** What model versions and run phases define the analysis units?
- **Claim_ids covered:** scope metadata of `LL-A`, `LL-B`, `LL-C`, `QW-A`, `GM-A`, `G5-A`, `CS-A`, `GO-A`.
- **Artifacts / rows cited:** `handoffs/llama33_planner20_primary_opus_handoff.md`; `handoffs/llama33_turbo_alpha30_OPUS_HANDOFF.md`; `handoffs/qwen25_72b_planner_alpha1_primary_opus_handoff.md`; manifests `llama33_planner20_primary_artifact_manifest.json`, `qwen25_72b_planner_alpha1_primary_artifact_manifest.json`.
- **Estimand(s) and scope:**
  - `meta-llama/llama-3.3-70b-instruct` — Phase 1 "main" (`LL-A`, `LL-B`); Phase 2 Turbo explicitly excluded from `LL-A`/`LL-B`; Phase 1 main α0 vs Phase 2 Turbo α30 for `LL-C`.
  - Qwen 2.5 72B (`qwen25_72b`) — no vendor path/snapshot recorded.
  - `gemma_4_31b` — no vendor identifier/snapshot recorded anywhere in packet; Phase 1 complete-case.
  - `openai/gpt-5` — Phase 1, `frontier_baseline`, strength 0.0.
  - Claude Sonnet 4 and Sonnet 5 jointly — Phase 1 shared prompts; no standalone per-version analysis exists.
  - `openai/gpt-oss-20b` — Phase 1.
- **May claim:** the exact model strings and phase labels above; that Qwen, Gemma, and Claude entries lack vendor snapshot identifiers.
- **Must not claim:** that Llama Phase 1 and Phase 2 run conditions were equivalent; that `LL-A`/`LL-B` describe Phase 2 Turbo; that a standalone Sonnet 4 or Sonnet 5 analysis exists.

### 3.2 Prompt sets and languages

- **Research question:** What are the per-cell denominators, pairing bases, and coverage shortfalls?
- **Claim_ids covered:** denominator/coverage metadata of `LL-A`, `LL-B`, `QW-A`, `QW-B`, `GM-A`, `G5-A`, `CS-A`, `GO-A`.
- **Artifacts / rows cited:** `tables/llama33_planner20_primary_results.tex`; `metrics/qwen25_72b_primary_unified_results.csv` (33 rows); `metrics/gemma_4_31b_phase1_complete_case_unified_results.csv` (3 rows); `metrics/gpt_5_frontier_baseline_paper_table.csv` (8 cells); `metrics/claude_sonnet_4_vs_5_outcome_comparison.csv` (8 rows) + `..._metadata_consistency_audit.csv` (1 row); `metrics/gpt_oss_20b_planner_alpha1_primary_canonical_audit.csv`.
- **Estimand(s) and scope:** Arabic, English, Hausa, Yoruba × {benign, harmful}; Yoruba benign n = 45 throughout; harmful 300/language where paired; exceptions to be tabulated verbatim: Llama benign 150/150/150/45 = 495 with 1 partial benign-coverage row; Gemma English benign 101 paired (baseline 101, steered 112, 11 steered-only excluded); GPT-5 harmful 286/291/289/290 (5 partial-coverage cells incl. Yoruba benign); Qwen partial cells n = 6, 160, 173; Claude 1,695 shared IDs, 1 partial-coverage row (Yoruba benign 45); GPT-OSS 1,200 paired harmful, 0 partial rows.
- **May claim:** exact denominators and partial-coverage flags as printed; that GPT-5's harmful shortfall is absent records, not invalid ones (`parse_failure_n = 0`, `non_termination_n = 0` in all 8 cells).
- **Must not claim:** that GPT-5 harmful denominators are 300; that Gemma's three denominators may be summed (no total recorded); that Llama harmful and benign totals (1,200 and 495) may be combined; rounding-up or silent dropping of Qwen partial denominators.

### 3.3 Interventions and configurations

- **Research question:** Which steering roles and strengths are compared against which baselines?
- **Claim_ids covered:** intervention/phase scope of `LL-A`, `LL-B`, `LL-C`, `QW-A`, `QW-B`, `GM-A`, `G5-A`, `CS-A`, `GO-A`.
- **Artifacts / rows cited:** `metrics/llama33_turbo_alpha30_role_strength_unified_results.csv` (authoritative, 6 rows = 3 configurations × 2 languages); `metrics/qwen25_72b_primary_paired_inference.csv`; `metrics/gemma_4_31b_phase1_complete_case_paired_inference.csv`; `metrics/gpt_oss_20b_planner_alpha1_paired_effect.csv`; `tables/gpt_5_frontier_baseline_descriptive_results.tex`.
- **Estimand(s) and scope:** Llama planner-only α0 vs α20 (Phase 1); Llama cross-phase `planner_only`/`critic_only`/`both` α30 (Phase 2 Turbo) vs Phase 1 α0; Qwen baseline vs `planner_only`/`critic_only`/`both` at α1.0 and α2.0; Gemma baseline vs planner-only α1.0; GPT-OSS baseline vs planner-only α1.0; GPT-5 single configuration (no intervention arm); Claude two versions, no steering on either side.
- **May claim:** the configuration × strength grid exactly as recorded; that GPT-5 and Claude analyses contain no intervention arm.
- **Must not claim:** that random-vector or matched controls were run (`LL-C` forbidden); that the Llama cross-phase contrast isolates role or vector specificity; citation of `llama33_turbo_alpha30_paper_table.csv` or `tables/llama33_turbo_alpha30_cross_phase_comparison.tex`.

### 3.4 Outcome definitions and decision semantics

- **Research question:** How is "operational failure" (and refusal) defined per model, and how comparable are those definitions?
- **Claim_ids covered:** `QW-C` (primary), plus outcome definitions in `LL-A`, `LL-B`, `LL-C`, `QW-A`, `GM-A`, `G5-A`, `CS-A`, `GO-A`.
- **Artifacts / rows cited:** `metrics/qwen25_72b_primary_canonical_audit.csv` (41 rows); `metrics/gemma_4_31b_phase1_complete_case_audit.csv` (note: filename differs from manifest-listed `..._decision_semantics_audit.csv`); `metrics/gpt_5_frontier_baseline_canonical_audit.csv` (note: filename differs from manifest-listed `gpt_5_clean_frontier_baseline_audit.csv`); `metrics/gpt_oss_20b_planner_alpha1_primary_canonical_audit.csv`.
- **Estimand(s) and scope:** per-model outcome definitions — Llama cross-phase: explicit ALLOW + NO_DECISION, NO_DECISION folded into failure numerator (`LL-C`); Qwen: explicit veto = safe, non-veto unresolved = operational failure (`QW-A`); Gemma: non-veto failure = `is_vetoed=false`, complete-case valid-JSON, one truncated terminal JSONL record excluded per source, no imputation (`GM-A`); GPT-5: veto = `is_vetoed=true` (`G5-A`); Claude: veto status `is_vetoed=true` (`CS-A`); Llama benign: refusal rate, positive Δ = more false refusals (`LL-B`).
- **May claim:** the descriptive composition, critic token counts, uniqueness and definition-validity flags from the Qwen 41-row audit; that GPT-OSS outcome composition depends on registry audits not shipped and is therefore not independently verifiable.
- **Must not claim:** any test, CI, or multiple-testing statement attached to `QW-C`; that Qwen non-veto records are explicit ALLOW; that the Gemma metric is an explicit-ALLOW rate; that Gemma benign rows are "safety failures" or comparable to Llama benign refusal rates; that Llama Phase 1 outcome composition is documented (it is not in the packet); that missing-label handling is established for Llama (`LL-A`, `LL-B`, `LL-C`), Qwen, or Claude.

### 3.5 Statistical procedures and correction families

- **Research question:** What tests, interval procedures, and family-wise corrections govern each claim?
- **Claim_ids covered:** statistical status of `LL-A`, `LL-B`, `LL-C`, `QW-A`, `QW-B`, `GM-A`, `G5-A`, `CS-A`, `GO-A`, `QW-C`.
- **Artifacts / rows cited:** `metrics/llama33_turbo_alpha30_role_strength_paired_inference.csv`; `metrics/qwen25_72b_primary_paired_inference_holm.csv` (33 rows); `metrics/gemma_4_31b_phase1_complete_case_paired_inference_holm.csv` (3 rows); `metrics/gpt_oss_20b_planner_alpha1_paired_effect_holm.csv` (4 rows).
- **Estimand(s) and scope — declared families (must be kept disjoint):**
  - Llama harmful, size 4, α_FW = 0.05 (`LL-A`); Llama benign, separate family, size 4 (`LL-B`).
  - Llama cross-phase `turbo_alpha30_role_strength_6_tests`, size 6 (`LL-C`).
  - Qwen pre-declared `qwen25_72b_primary_33_steered_cells`, size 33, α_FW = 0.05 (`QW-A`, `QW-B`).
  - Gemma `gemma_phase1_three_cell_family`, size 3, α_FW = 0.05 (`GM-A`).
  - GPT-OSS four language-specific harmful comparisons, size 4 (`GO-A`).
  - No family for `G5-A`, `CS-A`, `QW-C` (descriptive).
  - Tests: exact paired two-sided McNemar; paired nonparametric bootstrap percentile 95% CI, 10,000 replicates, seed 20260908 (Llama, Qwen, Gemma); GPT-OSS bootstrap intervals retained in a non-shipped artifact.
- **May claim:** that all inferential claims are non-causal; that `LL-C` is associational/cross-phase; that pairing is by matched prompt/record ID; that CI-based and Holm-based conclusions are reported jointly wherever they diverge.
- **Must not claim:** combination of Llama harmful and benign families; Holm significance for Qwen outside the 33-cell family; conflation of "CI excludes zero" with Holm rejection; causal language anywhere; that pairing establishes non-prompt run-condition equivalence.

---

## 4. Primary results

### 4.1 Llama 3.3 70B harmful and benign

- **Research question:** In Phase 1, does planner-only α20 steering change harmful operational failure and benign refusal rates relative to α0, per language?
- **Claim_ids covered:** `LL-A`, `LL-B`.
- **Artifacts / rows cited:** `tables/llama33_planner20_primary_results.tex`, table `llama33_planner20_primary_results` — harmful-arm columns, 4 language rows (`LL-A`); benign-arm columns 6–9, 4 language rows (`LL-B`); `handoffs/llama33_planner20_primary_opus_handoff.md`; `manifests/llama33_planner20_primary_artifact_manifest.json`. Non-shipped CSVs (`..._paired_effect.csv`, `..._paired_inference.csv`, `..._primary_paper_table.csv`, `..._primary_raw_separation_audit.csv`) not cited as evidence.
- **Estimand(s) and scope:** paired within-model difference in harmful operational failure rate, planner-only α20 vs planner-only α0, Phase 1 "main", Arabic/English/Hausa/Yoruba, 300 prompt-matched harmful prompts per language (1,200 total, 0 partial harmful-coverage rows); and paired difference in benign refusal rate, denominators 150/150/150/45 = 495, 1 partial benign-coverage row.
- **May claim:**
  - Prompt-matched within-model comparison of planner-only α20 vs α0 harmful operational failure in Phase 1; 3 of 4 languages Holm-significant: Arabic 35.0→20.7 (Δ −14.3, CI [−21.0, −7.3], Holm p = 0.000225), Hausa 68.3→58.3 (Δ −10.0, CI [−17.0, −3.0], Holm p = 0.016762), Yoruba 74.7→58.0 (Δ −16.7, CI [−22.3, −11.3], Holm p < 0.001); English 13.7→13.3 (Δ −0.3, CI [−4.3, 3.7], Holm p = 1.0).
  - Benign refusal rate (positive Δ = more false refusals) showed no Holm-significant change under planner-only α20: Arabic 0.0→0.7, English 0.0→0.0, Hausa 0.0→1.3, Yoruba 0.0→0.0; 0 Holm-significant rows.
- **Must not claim:** combining harmful and benign sample sizes, rates, p-values, or correction families; attribution to Phase 2 Turbo; that the absence of a significant benign effect proves no utility cost, in particular not "especially for Yoruba"; any statement about which decision tokens compose the failure outcome; any statement that missing-label handling is established.

### 4.2 Qwen2.5 72B primary and 33-cell family

- **Research question:** Does planner-only α1.0 steering (and the wider role × strength grid) change Qwen operational failure per language and prompt class, under the pre-declared 33-cell correction family?
- **Claim_ids covered:** `QW-A`, `QW-B`, `QW-C`.
- **Artifacts / rows cited:** `tables/qwen25_72b_planner_alpha1_primary_results.tex`, table `qwen25_72b_planner_alpha1_primary_results`, 8 language×class cells (`QW-A`); `metrics/qwen25_72b_primary_paired_inference.csv`, `metrics/qwen25_72b_primary_paired_inference_holm.csv`, `metrics/qwen25_72b_primary_unified_results.csv`, 33 rows (`QW-B`); `metrics/qwen25_72b_primary_canonical_audit.csv`, 41 rows (`QW-C`); `handoffs/qwen25_72b_planner_alpha1_primary_opus_handoff.md`; `manifests/qwen25_72b_planner_alpha1_primary_artifact_manifest.json`. Non-shipped `qwen25_72b_planner_alpha1_primary_paper_table.csv` not cited as evidence.
- **Estimand(s) and scope:** paired within-model difference in operational failure (explicit veto = safe; non-veto unresolved = failure), baseline α0 vs planner-only α1.0 (8 cells; benign 150/150/150/45, harmful 300×4, 1,695 paired prompts, 0 partial rows); plus the full baseline vs {planner_only, critic_only, both} × {α1.0, α2.0} grid, 33 rows, Holm α_FW = 0.05.
- **May claim:**
  - Within-model prompt-matched planner-only α1.0 results: Arabic harmful 48 (16.0)→9 (3.0) (Δ −13.0, CI [−17.3, −9.0], Holm p = 2.5965e-8, significant); English harmful 117 (39.0)→48 (16.0) (Δ −23.0, CI [−29.0, −17.0], Holm p = 2.6313e-11, significant); Hausa harmful 165 (55.0)→147 (49.0) (Δ −6.0, CI [−12.3, 0.3], Holm p = 1.0, not significant); Yoruba harmful 153 (51.0)→132 (44.0) (Δ −7.0, CI [−12.7, −1.3], Holm p = 0.6414, not significant); benign cells Arabic 100.0→100.0, English 100.0→98.0, Hausa +0.0, Yoruba (n = 45) +0.0, none significant; 2 Holm-significant rows.
  - Across the pre-declared 33-cell family, 4 cells are Holm-significant: Arabic harmful α1.0 and α2.0, English harmful α1.0 and α2.0.
  - Partial-coverage cells preserved exactly: `critic_only` Arabic benign α1.0 steered n = 6 vs baseline 150 (`partial_baseline_coverage = true`); `both` Yoruba harmful α2.0 paired n = 160 (baseline 300, 140 baseline-only); `planner_only` Yoruba harmful α2.0 paired n = 173 (127 baseline-only).
  - Descriptive audit of decision composition and validity flags across 41 cells (`QW-C`).
- **Must not claim:** that non-veto records are explicit ALLOW; causal interpretation; Holm significance outside the pre-declared 33-cell family; that CI-excluding-zero cells are significant — explicitly list `critic_only` Hausa benign α1.0, `both` Hausa harmful α2.0, `critic_only` Yoruba harmful α1.0/α2.0, `planner_only` Hausa harmful α2.0, `planner_only` Yoruba harmful α1.0 as CI-excludes-zero but Holm-non-significant; dropping or rounding partial-coverage denominators; any parse-failure/missing count (no such column exists in shipped CSVs); any test, CI, or correction attached to `QW-C`.

### 4.3 Gemma 4 31B complete-case

- **Research question:** Under a complete-case valid-JSON policy, does planner-only α1.0 change Gemma non-veto operational failure in the three available cells?
- **Claim_ids covered:** `GM-A`.
- **Artifacts / rows cited:** `metrics/gemma_4_31b_phase1_complete_case_paired_inference.csv`; `metrics/gemma_4_31b_phase1_complete_case_paired_inference_holm.csv`; `metrics/gemma_4_31b_phase1_complete_case_unified_results.csv` (3 rows); `metrics/gemma_4_31b_phase1_complete_case_audit.csv` (filename differs from manifest-listed `..._decision_semantics_audit.csv`); `tables/gemma_4_31b_phase1_complete_case_results.tex`. Declared-authoritative `gemma_4_31b_phase1_complete_case_paper_table.csv` is NOT in packet and not cited as evidence.
- **Estimand(s) and scope:** paired within-model difference in non-veto operational failure (`is_vetoed=false`), Phase 1 baseline α0 vs planner-only α1.0; Arabic benign (150 paired), English benign (101 paired; baseline 101, steered 112, 11 steered-only valid IDs excluded, `partial_coverage = true`), Arabic harmful (300 paired); no Hausa, no Yoruba, no English harmful; Holm family size 3, α_FW = 0.05.
- **May claim:**
  - Arabic benign 64.00→74.00 (Δ +10.00, CI [−0.67, 20.67], raw p = 0.0912, Holm p = 0.0912); English benign 65.35→81.19 (Δ +15.84, CI [2.97, 27.75], raw p = 0.02263, Holm p = 0.06789); Arabic harmful 64.00→72.00 (Δ +8.00, CI [0.67, 15.33], raw p = 0.04004, Holm p = 0.08007).
  - All three point estimates show increased non-veto failure under steering; Holm-significant rows: 0.
  - Two rows have `ci_excludes_zero = true` while Holm rejects nothing — reported jointly.
  - Complete-case policy: one truncated terminal JSONL record excluded from each source; missing outcomes not imputed.
- **Must not claim:** summing the three denominators (no total recorded); describing the metric as an explicit-ALLOW rate; describing Gemma benign rows as "safety failures" or as comparable to Llama benign refusal rates (Gemma has no separate benign-utility refusal outcome); reporting any cell as significant; that a vendor identifier or snapshot exists for `gemma_4_31b`.

### 4.4 GPT-OSS 20B harmful

- **Research question:** Does planner-only α1.0 change GPT-OSS 20B harmful operational failure, and is the direction consistent across languages?
- **Claim_ids covered:** `GO-A`; `GO-B` (NOT WRITABLE-pending, truncated allowed_wording).
- **Artifacts / rows cited:** `metrics/gpt_oss_20b_planner_alpha1_paired_effect_holm.csv` (4 rows, authoritative for inference); `metrics/gpt_oss_20b_planner_alpha1_paired_effect.csv`; `metrics/gpt_oss_20b_planner_alpha1_primary_canonical_audit.csv`; `tables/gpt_oss_20b_planner_alpha1_primary_harmful_results.tex`. Non-shipped `paper_table.csv` and `paired_bootstrap_ci.csv` not cited as evidence. For `GO-B`: `gpt_oss_20b_phase1_baseline_by_language.csv` (8 rows) and `gpt_oss_20b_phase1_baseline_bootstrap_ci.csv` (4 rows) are referenced but absent — no numbers may be reported.
- **Estimand(s) and scope:** paired within-model difference in harmful operational failure, Phase 1 baseline (α0, no steering) vs planner-only α1.0; Arabic/English/Hausa/Yoruba, harmful only; 300 matched harmful prompts per language, 1,200 paired, 0 partial rows; canonical audit: baseline 1,695 rows (1,200 harmful + 495 benign), `planner_only_alpha1` 1,695 rows, paired harmful subset 1,200, duplicate record IDs 0; Holm family size 4.
- **May claim:**
  - Arabic 43/300 (14.33%)→30/300 (10.00%), Δ −4.33 pp, discordant 33 (10 worsened / 23 improved), raw p = 0.03508, Holm p = 0.070164, **not significant**.
  - English 174/300 (58.00%)→141/300 (47.00%), Δ −11.0 pp, discordant 105 (36/69), raw p = 0.0016590, Holm p = 0.004977, **significant**.
  - Hausa 45/300 (15.00%)→66/300 (22.00%), Δ +7.0 pp (**worsening**), discordant 27 (24 worsened / 3 improved), raw p = 4.92334e-5, Holm p = 0.00019693, **significant**.
  - Yoruba 48/300 (16.00%)→48/300 (16.00%), Δ 0.0, discordant 36 (18/18), p = 1.0.
  - The mixed-direction pattern must be stated in the interpretation (English improved, Hausa worsened, Arabic not significant after Holm, Yoruba null).
- **Must not claim:** causal interpretation; Arabic as significant; omission of the Hausa worsening; that outcome composition is independently verifiable (`registry/gpt_oss_decision_semantics_audit.json`, `registry/gpt_oss_parser_outcomes.json` not shipped); any missing-label count; any GPT-OSS Phase 1 baseline descriptive numbers (`GO-B` unsupported in packet).

---

## 5. Closed-model baselines

### 5.1 GPT-5 frontier baseline

- **Research question:** What are the descriptive veto / non-veto rates of a frontier closed model on the same multilingual prompt set, with no intervention arm?
- **Claim_ids covered:** `G5-A`.
- **Artifacts / rows cited:** `metrics/gpt_5_frontier_baseline_paper_table.csv`, table `gpt_5_frontier_baseline_paper_table`, 8 cells; `metrics/gpt_5_frontier_baseline_canonical_audit.csv` (filename differs from manifest-listed `gpt_5_clean_frontier_baseline_audit.csv`); `tables/gpt_5_frontier_baseline_descriptive_results.tex`.
- **Estimand(s) and scope:** descriptive per-cell veto rate (`is_vetoed=true`) and non-veto operational failure rate (`is_vetoed=false`), `openai/gpt-5`, Phase 1, configuration `frontier_baseline`, strength 0.0, single condition; benign 150/150/150/45, harmful 286/291/289/290; 1,651 valid unique-ID records; 5 partial-coverage rows (4 harmful + Yoruba benign); pairing not applicable.
- **May claim:**
  - Arabic benign 0/150 (0.00% veto, 100.00% non-veto); English benign 3/150 (2.00%, 98.00%); Hausa benign 0/150; Yoruba benign 0/45; Arabic harmful 233/286 (81.47%, 18.53%); English harmful 178/291 (61.17%, 38.83%); Hausa harmful 214/289 (74.05%, 25.95%); Yoruba harmful 204/290 (70.34%, 29.66%).
  - Canonical audit reports `parse_failure_n = 0` and `non_termination_n = 0` in all 8 cells; the harmful shortfall is absent records, not invalid ones.
- **Must not claim:** any inferential or comparative statement (no test, no CI, no correction); pooling the separate 570-record file (exact duplicate subset: 570 shared IDs, 570 matching language/harmfulness/veto outcome, 1 parse error); harmful denominators of 300; any comparison to steered open-weight arms.

### 5.2 Claude Sonnet 4 vs 5 cross-version comparison

- **Research question:** Do veto outcomes differ between two closed-model versions on shared Phase 1 prompts, descriptively and prompt-matched?
- **Claim_ids covered:** `CS-A`.
- **Artifacts / rows cited:** `metrics/claude_sonnet_4_vs_5_outcome_comparison.csv` (8 rows); `metrics/claude_sonnet_4_vs_5_metadata_consistency_audit.csv` (1 row); `tables/claude_sonnet_4_vs_5_descriptive_comparison.tex`; handoff/manifest for the overall agreement figure. Declared-authoritative `claude_sonnet_4_vs_5_paper_table.csv` NOT in packet and not cited as evidence.
- **Estimand(s) and scope:** descriptive prompt-matched cross-version difference in veto status (`is_vetoed=true`) with per-cell discordance counts; 1,695 shared Phase 1 prompt IDs; benign 150/150/150/45, harmful 300 each; 1 partial-coverage row (Yoruba benign, 45 shared prompts); no steering condition on either side; metadata audit: 1,695 records present in both, language and harmfulness match for all.
- **May claim:**
  - All four benign cells: 0 vetoes in both versions, 100.00% agreement.
  - Arabic harmful S4 23 (7.67%) vs S5 13
