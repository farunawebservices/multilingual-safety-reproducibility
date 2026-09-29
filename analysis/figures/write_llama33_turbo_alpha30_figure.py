#!/usr/bin/env python3
"""Write Addition 2 figure."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

INFILE = Path("paper_artifacts/metrics/llama33_turbo_alpha30_paper_table.csv")
OUT_PDF = Path("paper_artifacts/figures/llama33_turbo_alpha30_cross_phase_comparison.pdf")
OUT_PNG = Path("paper_artifacts/figures/llama33_turbo_alpha30_cross_phase_comparison.png")

frame = pl.read_csv(INFILE).sort(["Configuration", "Language"])

configs = frame["Configuration"].to_list()
languages = frame["Language"].to_list()
effects = frame["Change (pp)"].to_list()
ci_lows = frame["95% CI low (pp)"].to_list()
ci_highs = frame["95% CI high (pp)"].to_list()
holm_reject = frame["Holm reject at 0.05"].to_list()

positions = np.arange(6)
bar_colors = ["#2166ac" if e < 0 else "#b2182b" if e > 0 else "#6e6e6e" for e in effects]

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.bar(positions, effects, color=bar_colors, edgecolor="black", linewidth=0.8, width=0.7, zorder=2)

ax.errorbar(positions, effects, yerr=[[abs(e - low) for e, low in zip(effects, ci_lows)],
          [abs(high - e) for e, high in zip(effects, ci_highs)]],
          fmt="none", ecolor="black", capsize=4, linewidth=1.2, zorder=3)

ax.axhline(0, color="black", linewidth=1.0, zorder=1)
ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, zorder=0)

ax.set_xticks(positions)
ax.set_xticklabels([f"{c}\n{l}" for c, l in zip(configs, languages)], fontsize=9)
ax.set_ylabel("Change in failure rate (pp)", fontsize=11)
ax.set_title("Llama 3.3 Turbo Alpha 30: Cross-phase comparison\nArchived role/strength reanalysis (Phase 2 vs Phase 1)", fontsize=12)

for bar, effect, significant in zip(bars, effects, holm_reject):
    label = f"{effect:+.1f}" + ("*" if significant else "")
    y_offset = -0.8 if effect < 0 else 0.6
    va = "top" if effect < 0 else "bottom"
    ax.text(bar.get_x() + bar.get_width() / 2, effect + y_offset, label, ha="center", va=va, fontsize=10, fontweight="bold" if significant else "normal")

y_min = min(effects) - 3
y_max = max(effects) + 3
ax.set_ylim(y_min, y_max)

fig.text(0.5, -0.02, "Negative values indicate lower failure rates at Alpha 30. Error bars show 95% bootstrap CIs (10,000 replicates). Asterisks mark Holm-adjusted p < 0.05. All comparisons are cross-phase and associational, not causal.", ha="center", va="top", fontsize=9, wrap=True)

fig.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.25)

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Wrote: {OUT_PDF}")
print(f"Wrote: {OUT_PNG}")
