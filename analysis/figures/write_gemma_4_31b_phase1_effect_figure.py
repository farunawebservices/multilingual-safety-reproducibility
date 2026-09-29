from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

INFILE = Path(
    "paper_artifacts/metrics/"
    "gemma_4_31b_phase1_complete_case_paper_table.csv"
)

OUT_PDF = Path(
    "paper_artifacts/figures/"
    "gemma_4_31b_phase1_complete_case_effects.pdf"
)

OUT_PNG = Path(
    "paper_artifacts/figures/"
    "gemma_4_31b_phase1_complete_case_effects.png"
)

frame = pl.read_csv(INFILE)

required = [
    "Language",
    "Prompt class",
    "Change (percentage points)",
    "Bootstrap 95% CI low (pp)",
    "Bootstrap 95% CI high (pp)",
    "Coverage",
]

missing = sorted(set(required) - set(frame.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if frame.height != 3:
    raise ValueError(f"Expected 3 paper rows; found {frame.height}")

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
    plot["Change (percentage points)"].to_list(),
    dtype=float,
)
lower = np.array(
    plot["Bootstrap 95% CI low (pp)"].to_list(),
    dtype=float,
)
upper = np.array(
    plot["Bootstrap 95% CI high (pp)"].to_list(),
    dtype=float,
)

xerr = np.vstack([effects - lower, upper - effects])
colors = [
    "#C44E52" if coverage == "Matched subset" else "#4C72B0"
    for coverage in plot["Coverage"].to_list()
]

fig, ax = plt.subplots(figsize=(7.2, 3.8))

ax.axvline(
    0,
    color="black",
    linewidth=1,
    linestyle="--",
    zorder=1,
)

y = np.arange(len(labels))

ax.errorbar(
    effects,
    y,
    xerr=xerr,
    fmt="none",
    ecolor="black",
    elinewidth=1.4,
    capsize=4,
    zorder=2,
)

ax.scatter(
    effects,
    y,
    s=60,
    c=colors,
    zorder=3,
)

ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlabel(
    "Steered minus baseline non-veto failure "
    "(percentage points)"
)
ax.set_title(
    "Gemma 4 31B Phase 1: paired change under planner-only steering"
)
ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)

ax.text(
    0.01,
    -0.28,
    "Points show paired estimates; bars show paired-bootstrap 95% CIs. "
    "Red indicates the English benign matched-subset result.",
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
print(
    "Matched-subset plotted rows: "
    f"{plot.filter(pl.col('Coverage') == 'Matched subset').height}"
)
intervals_excluding_zero = plot.filter(
    (pl.col("Bootstrap 95% CI low (pp)") > 0)
    | (pl.col("Bootstrap 95% CI high (pp)") < 0)
).height

print(f"Intervals excluding zero: {intervals_excluding_zero}")
