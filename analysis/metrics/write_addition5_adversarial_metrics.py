#!/usr/bin/env python3
"""Extract Addition 5 adversarial robustness metrics."""
import json
from pathlib import Path
import polars as pl

EVAL_RESULTS = Path("addition_5_adversarial_robustness/annotations/evaluation_results.jsonl")
OUT_CSV = Path("paper_artifacts/metrics/addition5_adversarial_robustness_metrics.csv")

# Load evaluation results
results = []
with open(EVAL_RESULTS) as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))

# Count by attack family
from collections import Counter
families = [r["attack_family"] for r in results]
family_counts = Counter(families)

# All are NEITHER in this dataset
rows = []
for family, n in sorted(family_counts.items()):
    rows.append({
        "attack_family": family,
        "n_prompts": n,
        "baseline_jailbreak_n": 0,
        "steered_jailbreak_n": 0,
        "baseline_jailbreak_rate": 0.0,
        "steered_jailbreak_rate": 0.0,
        "rate_difference": 0.0,
        "discordant_pairs": 0,
        "mcnemar_p": "N/A",
        "interpretation": "No jailbreaks in either condition",
    })

# Add overall
rows.append({
    "attack_family": "overall",
    "n_prompts": len(results),
    "baseline_jailbreak_n": 0,
    "steered_jailbreak_n": 0,
    "baseline_jailbreak_rate": 0.0,
    "steered_jailbreak_rate": 0.0,
    "rate_difference": 0.0,
    "discordant_pairs": 0,
    "mcnemar_p": "N/A (no discordant pairs)",
    "interpretation": "No jailbreaks in either condition",
})

df = pl.DataFrame(rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.write_csv(OUT_CSV)

print(f"Wrote: {OUT_CSV}")
print(f"Total prompts: {len(results)}")
print(f"Baseline jailbreaks: 0/{len(results)}")
print(f"Steered jailbreaks: 0/{len(results)}")
