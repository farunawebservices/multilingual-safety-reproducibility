#!/usr/bin/env python3
"""Generate Addition 6 prompt-filter baseline comparison figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

INFILE = Path("paper_artifacts/metrics/addition6_prompt_filter_baseline_metrics.csv")
OUT_PDF = Path("paper_artifacts/figures/addition6_prompt_filter_baseline.pdf")
OUT_PNG = Path("paper_artifacts/figures/addition6_prompt_filter_baseline.png")

df = pl.read_csv(INFILE)

required = {
    "condition",
    "harmful_total",
    "harmful_vetoes",
    "harmful_veto_rate",
    "benign_total",
    "benign_vetoes",
    "benign_veto_rate",
    "missing_labels",
}
missing = sorted(required - set(df.columns))
if missing:
    raise ValueError(f"Missing required metric columns: {missing}")

if df.height != 3:
    raise ValueError(f"Expected exactly three conditions; found {df.height}.")

if set(df["condition"].to_list()) != {
    "no_intervention",
    "prompt_filter",
    "planner_steering_l50_s20_historical_all_steps",
}:
    raise ValueError("Unexpected Addition 6 condition set.")

if set(df["harmful_total"].to_list()) != {400}:
    raise ValueError("Expected 400 harmful prompts in every condition.")

if set(df["benign_total"].to_list()) != {60}:
    raise ValueError("Expected 60 benign prompts in every condition.")

display_names = {
    "no_intervention": "No\nintervention",
    "prompt_filter": "Planner prompt\nfilter",
    "planner_steering_l50_s20_historical_all_steps": "Planner SAE\nsteering",
}

ordered_conditions = [
    "no_intervention",
    "prompt_filter",
    "planner_steering_l50_s20_historical_all_steps",
]
df = (
    df.with_columns(
        pl.col("condition")
        .replace_strict(
            ordered_conditions,
            [0, 1, 2],
            return_dtype=pl.Int64,
        )
        .alias("_order")
    )
    .sort("_order")
)

conditions = [display_names[x] for x in df["condition"].to_list()]
harmful_rates = [100.0 * float(x) for x in df["harmful_veto_rate"].to_list()]
benign_rates = [100.0 * float(x) for x in df["benign_veto_rate"].to_list()]
missing_labels = [int(x) for x in df["missing_labels"].to_list()]

expected_harmful = [52.5, 53.5, 57.5]
expected_benign = [0.0, 0.0, 5.0]
expected_missing = [4, 5, 25]

for a, b in zip(harmful_rates, expected_harmful):
    if abs(a - b) > 0.01:
        raise ValueError(f"Unexpected harmful veto rate: {a} vs expected {b}")
if benign_rates != expected_benign:
    raise ValueError(f"Unexpected benign FPRs: {benign_rates}")
if missing_labels != expected_missing:
    raise ValueError(f"Unexpected missing-label counts: {missing_labels}")

positions = np.arange(len(conditions))
width = 0.35

fig, ax = plt.subplots(figsize=(8.6, 5.25))

harmful_bars = ax.bar(
    positions - width / 2,
    harmful_rates,
    width,
    label="Harmful Critic-VETO rate",
    color="#2166ac",
    edgecolor="black",
    linewidth=0.8,
    zorder=2,
)
benign_bars = ax.bar(
    positions + width / 2,
    benign_rates,
    width,
    label="Benign false-positive VETO rate",
    color="#b2182b",
    edgecolor="black",
    linewidth=0.8,
    zorder=2,
)

ax.set_xticks(positions, conditions)
ax.set_ylabel("Rate (%)")
ax.set_ylim(0, 70)
ax.set_title(
    "Addition 6: Prompt filter baseline and planner SAE steering\n"
    "Matched 460-case Llama evaluation (400 harmful; 60 benign)"
)
ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, zorder=0)
ax.legend(loc="upper left", frameon=True)

for bar, rate in zip(harmful_bars, harmful_rates):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        rate + 1.0,
        f"{rate:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )

for bar, rate in zip(benign_bars, benign_rates):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        rate + 1.0,
        f"{rate:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.text(
    0.5,
    0.02,
    "Team outcome is the Critic's exact terminal label. Missing Critic labels "
    "are retained separately; counts are 4, 5, and 25 respectively. "
    "This descriptive comparison shows trade-offs and does not establish "
    "uniform superiority of steering over prompt filtering.",
    ha="center",
    va="bottom",
    fontsize=8.5,
    wrap=True,
)

fig.subplots_adjust(left=0.11, right=0.98, top=0.84, bottom=0.25)

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Wrote: {OUT_PDF}")
print(f"Wrote: {OUT_PNG}")
print(f"Harmful veto rates (%): {harmful_rates}")
print(f"Benign false-positive rates (%): {benign_rates}")
print(f"Missing Critic labels: {missing_labels}")
