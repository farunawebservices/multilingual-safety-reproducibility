# Addition 5: Adversarial Robustness Opus Handoff

## Purpose
This handoff provides the authoritative artifact chain for writing about Addition 5 (adversarial robustness check) in the ICLR paper.

## Nature of Addition 5
Addition 5 is a **bounded adversarial robustness check**, NOT a main intervention experiment. It tests whether activation steering creates jailbreak vulnerabilities.

## Authoritative numerical source
All numerical reporting MUST come from:
- `paper_artifacts/metrics/addition5_adversarial_robustness_metrics.csv`
  - SHA-256: `ef54c870cf23326ecbca12e72ceb905247ce7464153e3693341876d6066c25e9`
  - 7 rows: 6 attack families + overall
  - 50 total jailbreak prompts

## Complete artifact chain
1. **Raw evaluation data:** `addition_5_adversarial_robustness/annotations/evaluation_results.jsonl` (SHA: `7059104c9f6d9010...`)
2. **Metrics CSV:** `paper_artifacts/metrics/addition5_adversarial_robustness_metrics.csv`
3. **Paper table:** `paper_artifacts/metrics/addition5_adversarial_robustness_table.csv` (SHA: `c828bfca2362057d...`)
4. **Paper table LaTeX:** `paper_artifacts/tables/addition5_adversarial_robustness.tex`
5. **Figure:** `paper_artifacts/figures/addition5_adversarial_robustness.pdf`
6. **Results summary:** `addition_5_adversarial_robustness/ADDITION_5_RESULTS_SUMMARY.md`

## Key results
- **50 jailbreak prompts** across 6 attack families
- **Baseline jailbreak rate: 0%** (0/50)
- **Steered jailbreak rate: 0%** (0/50)
- **All prompts evaluated as NEITHER** (no jailbreak in either condition)
- **No discordant pairs** → McNemar test N/A

## Allowed qualitative summary
"On a 50-prompt English jailbreak set across 6 attack families, Llama-3.3-70B-Instruct showed zero jailbreak successes in both baseline and steered conditions (random steering, layer 52, strength 20). No evidence of increased vulnerability from steering was detected."

## Explicitly forbidden claims
- Do NOT claim this proves steering is safe (bounded test only)
- Do NOT generalize beyond these 50 specific prompts
- Do NOT claim this used a refusal vector (it was RANDOM)
- Do NOT claim multilingual robustness (all English)
- Do NOT claim comprehensive safety certification

## Limitations (REQUIRED in all discussions)
"This used a random steering vector (not a refusal direction), 50 English-only prompts, and cannot rule out vulnerabilities on untested attacks or with actual refusal vectors."

## Placement recommendation
Appendix or limitations section as a bounded robustness check, NOT a main results section.

## Citation format
Cite the metrics CSV path and SHA-256 hash when reporting numbers.
