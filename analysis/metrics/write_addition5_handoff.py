#!/usr/bin/env python3
"""Write Addition 5 Opus handoff."""
import hashlib
from pathlib import Path

METRICS_CSV = Path("paper_artifacts/metrics/addition5_adversarial_robustness_metrics.csv")
TABLE_CSV = Path("paper_artifacts/metrics/addition5_adversarial_robustness_table.csv")
TABLE_TEX = Path("paper_artifacts/tables/addition5_adversarial_robustness.tex")
FIGURE_PDF = Path("paper_artifacts/figures/addition5_adversarial_robustness.pdf")
FIGURE_PNG = Path("paper_artifacts/figures/addition5_adversarial_robustness.png")
EVAL_RESULTS = Path("addition_5_adversarial_robustness/annotations/evaluation_results.jsonl")
RESULTS_MD = Path("addition_5_adversarial_robustness/ADDITION_5_RESULTS_SUMMARY.md")

OUT_HANDOFF = Path("paper_artifacts/metrics/addition5_adversarial_robustness_OPUS_HANDOFF.md")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

hashes = {
    "metrics": sha256(METRICS_CSV),
    "table": sha256(TABLE_CSV),
    "figure": sha256(FIGURE_PDF),
    "eval": sha256(EVAL_RESULTS),
}

handoff = f"""# Addition 5: Adversarial Robustness Opus Handoff

## Purpose
This handoff provides the authoritative artifact chain for writing about Addition 5 (adversarial robustness check) in the ICLR paper.

## Nature of Addition 5
Addition 5 is a **bounded adversarial robustness check**, NOT a main intervention experiment. It tests whether activation steering creates jailbreak vulnerabilities.

## Authoritative numerical source
All numerical reporting MUST come from:
- `paper_artifacts/metrics/addition5_adversarial_robustness_metrics.csv`
  - SHA-256: `{hashes['metrics']}`
  - 7 rows: 6 attack families + overall
  - 50 total jailbreak prompts

## Complete artifact chain
1. **Raw evaluation data:** `addition_5_adversarial_robustness/annotations/evaluation_results.jsonl` (SHA: `{hashes['eval'][:16]}...`)
2. **Metrics CSV:** `paper_artifacts/metrics/addition5_adversarial_robustness_metrics.csv`
3. **Paper table:** `paper_artifacts/metrics/addition5_adversarial_robustness_table.csv` (SHA: `{hashes['table'][:16]}...`)
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
"""

OUT_HANDOFF.parent.mkdir(parents=True, exist_ok=True)
OUT_HANDOFF.write_text(handoff, encoding="utf-8")

print(f"Wrote: {OUT_HANDOFF}")
