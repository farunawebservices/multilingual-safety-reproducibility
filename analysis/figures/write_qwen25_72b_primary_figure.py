from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

INFILE = Path(
    "paper_artifacts/metrics/"
    "qwen25_72b_planner_alpha1_primary_paper_table.csv"
)

OUT_PDF = Path(
    "paper_artifacts/figures/"
    "qwen25_72b_planner_alpha1_primary_effects.pdf"
)

OUT_PNG = Path(
    "paper_artifacts/figures/"
    "qwen25_72b_planner_alpha1_primary_effects.png"
)

frame = pl.read_csv(INFILE)

required = [
    "Language",
    "Prompt class",
    "Paired n",
    "Steered minus baseline (pp)",
    "Bootstrap 95% CI low (pp)",
    "Bootstrap 95% CI high (pp)",
    "Holm reject 0.05",
    "Coverage",
]

missing = sorted(set(required) - set(frame.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if frame.height != 8:
    raise ValueError(f"Expected 8 rows; found {frame.height}")

if frame["Paired n"].sum() != 1695:
    raise ValueError("Expected 1695 paired prompts.")

if not (frame["Coverage"] == "Complete matched coverage").all():
    raise ValueError("Expected complete matched coverage.")

plot = (
    frame
    .with_columns(
        (
            pl.col("Language")
            + pl.lit(" — ")
            + pl.col("Prompt class")
        ).alias("label")
    )
    .sort(["Prompt class", "Language"])
)

labels = plot["label"].to_list()
effects = np.array(
    plot["Steered minus baseline (pp)"].to_list(),
    dtype=float,
)
ci_low = np.array(
    plot["Bootstrap 95% CI low (pp)"].to_list(),
    dtype=float,
)
ci_high = np.array(
    plot["Bootstrap 95% CI high (pp)"].to_list(),
    dtype=float,
)
holm = np.array(
    plot["Holm reject 0.05"].to_list(),
    dtype=bool,
)

lower_error = effects - ci_low
upper_error = ci_high - effects

if (lower_error < 0).any() or (upper_error < 0).any():
    raise ValueError("A confidence interval does not contain its point estimate.")

y = np.arange(plot.height)

colors = [
    "#C44E52" if significant else "#4C72B0"
    for significant in holm
]

fig, ax = plt.subplots(figsize=(8.4, 4.9))

ax.errorbar(
    effects,
    y,
    xerr=np.vstack([lower_error, upper_error]),
    fmt="none",
    ecolor="#444444",
    elinewidth=1.25,
    capsize=3,
    zorder=1,
)

ax.scatter(
    effects,
    y,
    s=45,
    c=colors,
    zorder=2,
)

ax.axvline(0, color="black", linewidth=0.9)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlabel("Steered minus baseline operational-failure rate (percentage points)")
ax.set_title("Qwen 2.5 72B: planner-only steering at alpha 1.0")
ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)

ax.text(
    0.01,
    -0.15,
    "Points are paired rate differences; bars are existing 95% bootstrap intervals over matched prompt IDs. "
    "Red points are Holm-significant within the predeclared 33-test Qwen family. "
    "Negative values indicate fewer operational failures under steering.",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=8,
)

fig.tight_layout()

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Wrote: {OUT_PDF}")
print(f"Wrote: {OUT_PNG}")
print(f"Plotted rows: {plot.height}")
print(f"Paired prompts: {plot['Paired n'].sum()}")
print(f"Holm-significant points: {holm.sum()}")
print(
    "Negative effect points: "
    f"{sum(effects < 0)}"
)
print("Complete-coverage rows: 8")
