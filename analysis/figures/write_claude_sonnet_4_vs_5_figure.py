from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

INFILE = Path(
    "paper_artifacts/metrics/"
    "claude_sonnet_4_vs_5_paper_table.csv"
)

OUT_PDF = Path(
    "paper_artifacts/figures/"
    "claude_sonnet_4_vs_5_veto_rate_comparison.pdf"
)

OUT_PNG = Path(
    "paper_artifacts/figures/"
    "claude_sonnet_4_vs_5_veto_rate_comparison.png"
)

frame = pl.read_csv(INFILE)

required = [
    "Language",
    "Prompt class",
    "Matched n",
    "Sonnet 4 veto (%)",
    "Sonnet 5 veto (%)",
    "Coverage",
]

missing = sorted(set(required) - set(frame.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if frame.height != 8:
    raise ValueError(f"Expected 8 rows; found {frame.height}")

if frame.filter(pl.col("Coverage") == "Partial cell: 45 prompts").height != 1:
    raise ValueError("Expected exactly one partial-coverage cell.")

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
sonnet4 = np.array(plot["Sonnet 4 veto (%)"].to_list(), dtype=float)
sonnet5 = np.array(plot["Sonnet 5 veto (%)"].to_list(), dtype=float)
coverage = plot["Coverage"].to_list()

y = np.arange(len(labels))
height = 0.36

fig, ax = plt.subplots(figsize=(8.2, 5.0))

ax.barh(
    y - height / 2,
    sonnet4,
    height=height,
    color="#4C72B0",
    label="Claude Sonnet 4",
)

ax.barh(
    y + height / 2,
    sonnet5,
    height=height,
    color="#DD8452",
    label="Claude Sonnet 5",
)

for index, cell_coverage in enumerate(coverage):
    if cell_coverage == "Partial cell: 45 prompts":
        ax.axhspan(
            index - 0.5,
            index + 0.5,
            color="#999999",
            alpha=0.12,
            zorder=0,
        )

ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlabel("Veto rate (%)")
ax.set_title(
    "Claude Sonnet 4 vs Sonnet 5: prompt-matched descriptive comparison"
)
ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")

ax.text(
    0.01,
    -0.14,
    "Shaded row: Yoruba benign has 45 shared prompts; all other rows have complete matched coverage. "
    "This figure is descriptive and does not estimate a steering effect.",
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
    "Partial-coverage shaded rows: "
    f"{sum(item == 'Partial cell: 45 prompts' for item in coverage)}"
)
print(
    "Nonzero veto-rate rows: "
    f"{sum((sonnet4 > 0) | (sonnet5 > 0))}"
)
