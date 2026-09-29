#!/usr/bin/env python3
"""Write Addition 4 calibration figure."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

INFILE = Path("paper_artifacts/metrics/addition4_judge_calibration_metrics.csv")
OUT_PDF = Path("paper_artifacts/figures/addition4_calibration_kappa_by_language.pdf")
OUT_PNG = Path("paper_artifacts/figures/addition4_calibration_kappa_by_language.png")

df = pl.read_csv(INFILE).filter(pl.col("language_code") != "overall").sort("language_code")

languages = df["language_name"].to_list()
kappas = df["cohens_kappa"].to_list()
ci_lows = df["ci95_low"].to_list()
ci_highs = df["ci95_high"].to_list()

positions = np.arange(len(languages))

fig, ax = plt.subplots(figsize=(9, 5))

# Color by kappa value
bar_colors = ["#2166ac" if k > 0.9 else "#b2182b" if k < 0.6 else "#6e6e6e" for k in kappas]

bars = ax.bar(positions, kappas, color=bar_colors, edgecolor="black", linewidth=0.8, width=0.7, zorder=2)

# Error bars
ax.errorbar(positions, kappas, yerr=[[k - low for k, low in zip(kappas, ci_lows)],
          [high - k for k, high in zip(kappas, ci_highs)]],
          fmt="none", ecolor="black", capsize=4, linewidth=1.2, zorder=3)

# Target line
ax.axhline(0.6, color="#b2182b", linestyle="--", linewidth=1.5, label="Target κ ≥ 0.60", zorder=1)
ax.axhline(0.7, color="#2166ac", linestyle="--", linewidth=1.5, label="Target κ ≥ 0.70 (overall)", zorder=1)

ax.axhline(0, color="black", linewidth=1.0, zorder=0)
ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, zorder=0)

ax.set_xticks(positions)
ax.set_xticklabels(languages, fontsize=10)
ax.set_ylabel("Cohen's κ", fontsize=11)
ax.set_title("Addition 4: Inter-rater Agreement by Language\n98 prompts, 2 annotators, blinded adjudication", fontsize=12)
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", fontsize=9)

# Value labels
for bar, kappa in zip(bars, kappas):
    ax.text(bar.get_x() + bar.get_width() / 2, kappa + 0.03, f"{kappa:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

fig.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.2)

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Wrote: {OUT_PDF}")
print(f"Wrote: {OUT_PNG}")
