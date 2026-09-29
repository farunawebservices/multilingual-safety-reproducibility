from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

INFILE = Path(
    "paper_artifacts/metrics/llama33_planner20_primary_paper_table.csv"
)

OUT_PDF = Path(
    "paper_artifacts/figures/llama33_planner20_primary_effects.pdf"
)

OUT_PNG = Path(
    "paper_artifacts/figures/llama33_planner20_primary_effects.png"
)

frame = pl.read_csv(INFILE).sort("Language")

required = [
    "Language",
    "Harmful paired n",
    "Harmful delta (pp)",
    "Harmful Holm reject 0.05",
    "Benign paired n",
    "Benign delta refusal (pp)",
    "Benign Holm reject 0.05",
]

missing = [column for column in required if column not in frame.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if frame.height != 4:
    raise ValueError(f"Expected 4 language rows; found {frame.height}.")

if frame["Harmful paired n"].sum() != 1200:
    raise ValueError("Expected 1200 paired harmful prompts.")

if frame["Benign paired n"].sum() != 495:
    raise ValueError("Expected 495 paired benign prompts.")

harmful_effects = frame["Harmful delta (pp)"].to_numpy()
harmful_significant = frame["Harmful Holm reject 0.05"].to_numpy()

benign_effects = frame["Benign delta refusal (pp)"].to_numpy()
benign_significant = frame["Benign Holm reject 0.05"].to_numpy()

languages = frame["Language"].to_list()
positions = np.arange(frame.height)

fig, axes = plt.subplots(
    ncols=2,
    figsize=(10.6, 4.7),
    sharex=True,
)

harmful_ax, benign_ax = axes

harmful_colors = [
    "#2166ac" if effect < 0 else "#b2182b" if effect > 0 else "#6e6e6e"
    for effect in harmful_effects
]

harmful_bars = harmful_ax.bar(
    positions,
    harmful_effects,
    color=harmful_colors,
    edgecolor="black",
    linewidth=0.7,
    width=0.66,
    zorder=2,
)

harmful_ax.axhline(0, color="black", linewidth=0.9, zorder=1)
harmful_ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, zorder=0)
harmful_ax.set_ylim(-23, 6)
harmful_ax.set_title("Harmful prompts: operational failure")
harmful_ax.set_ylabel("Steered minus baseline (pp)")

for bar, effect, significant in zip(
    harmful_bars,
    harmful_effects,
    harmful_significant,
):
    label = f"{effect:+.1f}" + ("*" if significant else "")
    y_offset = -0.75 if effect < 0 else 0.45
    va = "top" if effect < 0 else "bottom"

    harmful_ax.text(
        bar.get_x() + bar.get_width() / 2,
        effect + y_offset,
        label,
        ha="center",
        va=va,
        fontsize=10,
        fontweight="bold" if significant else "normal",
    )

benign_colors = [
    "#b2182b" if effect > 0 else "#2166ac" if effect < 0 else "#6e6e6e"
    for effect in benign_effects
]

benign_bars = benign_ax.bar(
    positions,
    benign_effects,
    color=benign_colors,
    edgecolor="black",
    linewidth=0.7,
    width=0.66,
    zorder=2,
)

benign_ax.axhline(0, color="black", linewidth=0.9, zorder=1)
benign_ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, zorder=0)
benign_ax.set_ylim(-0.4, 2.3)
benign_ax.set_title("Benign prompts: refusal rate")
benign_ax.set_ylabel("Steered minus baseline (pp)")

for bar, effect, significant in zip(
    benign_bars,
    benign_effects,
    benign_significant,
):
    label = f"{effect:+.1f}" + ("*" if significant else "")
    y_offset = 0.10 if effect >= 0 else -0.10
    va = "bottom" if effect >= 0 else "top"

    benign_ax.text(
        bar.get_x() + bar.get_width() / 2,
        effect + y_offset,
        label,
        ha="center",
        va=va,
        fontsize=10,
        fontweight="bold" if significant else "normal",
    )

for axis in axes:
    axis.set_xticks(positions, languages)

fig.suptitle(
    "Llama-3.3-70B-Instruct: planner-only steering at alpha 20",
    y=0.98,
    fontsize=12,
)

fig.text(
    0.5,
    0.02,
    "Left: negative values indicate fewer harmful operational failures. "
    "Right: positive values indicate more benign false refusals. "
    "Asterisks mark Holm-adjusted p < 0.05.",
    ha="center",
    va="bottom",
    fontsize=8.8,
)

fig.subplots_adjust(
    left=0.09,
    right=0.98,
    top=0.84,
    bottom=0.24,
    wspace=0.30,
)

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=300)
plt.close(fig)

print(f"Wrote: {OUT_PDF}")
print(f"Wrote: {OUT_PNG}")
print(f"Plotted language rows: {frame.height}")
print(f"Paired harmful prompts: {frame['Harmful paired n'].sum()}")
print(f"Paired benign prompts: {frame['Benign paired n'].sum()}")
print(f"Harmful Holm-significant points: {int(harmful_significant.sum())}")
print(f"Benign Holm-significant points: {int(benign_significant.sum())}")
print("Harmful and benign outcomes plotted separately: True")
