from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

INFILE = Path(
    "paper_artifacts/metrics/"
    "gpt_5_frontier_baseline_paper_table.csv"
)

OUT_PDF = Path(
    "paper_artifacts/figures/"
    "gpt_5_frontier_baseline_veto_rates.pdf"
)

OUT_PNG = Path(
    "paper_artifacts/figures/"
    "gpt_5_frontier_baseline_veto_rates.png"
)

frame = pl.read_csv(INFILE)

required = [
    "Language",
    "Prompt class",
    "Valid n",
    "Veto rate (%)",
    "Coverage",
]

missing = sorted(set(required) - set(frame.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if frame.height != 8:
    raise ValueError(f"Expected 8 rows; found {frame.height}")

partial_n = frame.filter(
    pl.col("Coverage") != "Complete cell"
).height

if partial_n != 5:
    raise ValueError(f"Expected 5 partial rows; found {partial_n}")

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
rates = np.array(plot["Veto rate (%)"].to_list(), dtype=float)
coverage = plot["Coverage"].to_list()

colors = [
    "#DD8452" if label.endswith("harmful") else "#4C72B0"
    for label in labels
]

fig, ax = plt.subplots(figsize=(8.2, 4.8))
y = np.arange(len(labels))

ax.barh(y, rates, color=colors, height=0.65)

for index, cell_coverage in enumerate(coverage):
    if cell_coverage != "Complete cell":
        ax.axhspan(
            index - 0.5,
            index + 0.5,
            color="#999999",
            alpha=0.10,
            zorder=0,
        )

for index, rate in enumerate(rates):
    ax.text(
        rate + 1.0,
        index,
        f"{rate:.1f}%",
        va="center",
        fontsize=8,
    )

ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlim(0, max(100, float(rates.max()) + 10))
ax.set_xlabel("Veto rate (%)")
ax.set_title("GPT-5 frontier baseline: descriptive veto rates")
ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)

ax.text(
    0.01,
    -0.14,
    "Shaded rows have partial coverage. This figure describes one frontier-baseline "
    "configuration at strength 0.0; it is not a steering-effect estimate.",
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
print(f"Partial-coverage shaded rows: {partial_n}")
print(f"Nonzero veto-rate rows: {sum(rates > 0)}")
print("Comparison or inference claim included: False")
