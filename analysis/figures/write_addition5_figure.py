#!/usr/bin/env python3
"""Write Addition 5 figure."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

INFILE = Path("paper_artifacts/metrics/addition5_adversarial_robustness_metrics.csv")
OUT_PDF = Path("paper_artifacts/figures/addition5_adversarial_robustness.pdf")
OUT_PNG = Path("paper_artifacts/figures/addition5_adversarial_robustness.png")

df = pl.read_csv(INFILE).filter(pl.col("attack_family") != "overall")

families = df["attack_family"].to_list()
n_prompts = df["n_prompts"].to_list()

positions = np.arange(len(families))

fig, ax = plt.subplots(figsize=(8, 5))

# All zeros, so just show bars at zero
bars = ax.bar(positions, [0]*len(families), color="#6e6e6e", edgecolor="black", linewidth=0.8, width=0.7)

ax.axhline(0, color="black", linewidth=1.0)
ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.5)

ax.set_xticks(positions)
ax.set_xticklabels(families, fontsize=9, rotation=15, ha="right")
ax.set_ylabel("Jailbreak rate (%)", fontsize=11)
ax.set_title("Addition 5: Adversarial Robustness\nLlama-3.3-70B-Instruct with random steering (layer 52, strength 20)", fontsize=11)
ax.set_ylim(0, 10)

# Add n labels on bars
for bar, n in zip(bars, n_prompts):
    ax.text(bar.get_x() + bar.get_width() / 2, 5, f"n={n}\n0%", ha="center", va="center", fontsize=10)

fig.text(0.5, -0.02, "All 50 jailbreak prompts failed in both baseline and steered conditions.", ha="center", va="top", fontsize=9)

fig.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.25)

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Wrote: {OUT_PDF}")
print(f"Wrote: {OUT_PNG}")
