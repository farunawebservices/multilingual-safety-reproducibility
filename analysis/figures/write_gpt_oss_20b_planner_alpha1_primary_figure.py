from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

INFILE = Path(
    "paper_artifacts/metrics/"
    "gpt_oss_20b_planner_alpha1_primary_harmful_paper_table.csv"
)

OUT_PDF = Path(
    "paper_artifacts/figures/"
    "gpt_oss_20b_planner_alpha1_primary_harmful_effects.pdf"
)

OUT_PNG = Path(
    "paper_artifacts/figures/"
    "gpt_oss_20b_planner_alpha1_primary_harmful_effects.png"
)

frame = pl.read_csv(INFILE).sort("Language")

required = [
    "Language",
    "Prompt class",
    "Paired n",
    "Steered minus baseline (pp)",
    "Holm-adjusted p",
    "Holm reject 0.05",
    "Coverage",
]

missing = [column for column in required if column not in frame.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if frame.height != 4:
    raise ValueError(f"Expected 4 rows; found {frame.height}.")

if frame["Paired n"].sum() != 1200:
    raise ValueError("Expected 1200 paired harmful prompts.")

if not (frame["Prompt class"] == "Harmful").all():
    raise ValueError("Figure input must be harmful prompts only.")

if not (frame["Coverage"] == "Complete matched coverage").all():
    raise ValueError("All figure rows must have complete matched coverage.")

effects = frame["Steered minus baseline (pp)"].to_numpy()
languages = frame["Language"].to_list()
significant = frame["Holm reject 0.05"].to_numpy()

colors = [
    "#b2182b" if effect > 0 else "#2166ac" if effect < 0 else "#6e6e6e"
    for effect in effects
]

fig, ax = plt.subplots(figsize=(7.4, 4.6))
positions = np.arange(frame.height)

ax.axhline(0, color="black", linewidth=0.9, zorder=1)
bars = ax.bar(
    positions,
    effects,
    color=colors,
    edgecolor="black",
    linewidth=0.7,
    width=0.66,
    zorder=2,
)

for index, (bar, effect, is_significant) in enumerate(
    zip(bars, effects, significant)
):
    marker = "*" if is_significant else ""
    vertical = 0.45 if effect >= 0 else -0.65
    va = "bottom" if effect >= 0 else "top"

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        effect + vertical,
        f"{effect:+.1f}{marker}",
        ha="center",
        va=va,
        fontsize=10,
        fontweight="bold" if is_significant else "normal",
    )

ax.set_xticks(positions, languages)
ax.set_ylabel("Steered minus baseline\noperational-failure rate (pp)")
ax.set_title(
    "GPT-OSS-20B: planner-only steering on harmful prompts",
    pad=10,
)
ax.set_ylim(-14, 10)
ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, zorder=0)

fig.text(
    0.5,
    0.02,
    "Negative values indicate fewer operational failures under steering. "
    "Asterisks mark Holm-adjusted p < 0.05; n = 300 paired harmful prompts per language.",
    ha="center",
    va="bottom",
    fontsize=8.8,
)

fig.subplots_adjust(left=0.16, right=0.98, top=0.86, bottom=0.26)

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=300)
plt.close(fig)

print(f"Wrote: {OUT_PDF}")
print(f"Wrote: {OUT_PNG}")
print(f"Plotted rows: {frame.height}")
print(f"Paired harmful prompts: {frame['Paired n'].sum()}")
print(f"Holm-significant points: {int(significant.sum())}")
print(
    "Improvement points (negative pp): "
    f"{int((effects < 0).sum())}"
)
print(
    "Worsening points (positive pp): "
    f"{int((effects > 0).sum())}"
)
print("Complete-coverage rows: 4")
