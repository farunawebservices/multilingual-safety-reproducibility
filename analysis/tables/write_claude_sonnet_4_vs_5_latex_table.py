from pathlib import Path

import polars as pl

INFILE = Path(
    "paper_artifacts/metrics/"
    "claude_sonnet_4_vs_5_paper_table.csv"
)

OUT = Path(
    "paper_artifacts/tables/"
    "claude_sonnet_4_vs_5_descriptive_comparison.tex"
)

frame = pl.read_csv(INFILE)

required = [
    "Language",
    "Prompt class",
    "Matched n",
    "Sonnet 4 veto n",
    "Sonnet 4 veto (%)",
    "Sonnet 5 veto n",
    "Sonnet 5 veto (%)",
    "Sonnet 5 minus Sonnet 4 (pp)",
    "Decision agreement (%)",
    "Coverage",
]

missing = sorted(set(required) - set(frame.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if frame.height != 8:
    raise ValueError(f"Expected 8 rows; found {frame.height}")

if frame.filter(pl.col("Coverage") == "Partial cell: 45 prompts").height != 1:
    raise ValueError("Expected exactly one partial-coverage cell.")

def tex(value: str) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )

rows = []
for row in frame.sort(["Prompt class", "Language"]).iter_rows(named=True):
    coverage = (
        "Partial$^{\\dagger}$"
        if row["Coverage"] == "Partial cell: 45 prompts"
        else "Complete"
    )

    rows.append(
        " & ".join([
            tex(row["Language"]),
            tex(row["Prompt class"]),
            str(row["Matched n"]),
            f"{row['Sonnet 4 veto n']} ({row['Sonnet 4 veto (%)']:.2f})",
            f"{row['Sonnet 5 veto n']} ({row['Sonnet 5 veto (%)']:.2f})",
            f"{row['Sonnet 5 minus Sonnet 4 (pp)']:+.2f}",
            f"{row['Decision agreement (%)']:.2f}",
            coverage,
        ]) + r" \\"
    )

table = r"""\begin{table*}[t]
\centering
\caption{Prompt-matched descriptive comparison of Claude Sonnet 4 and Claude Sonnet 5 on shared Phase 1 prompts. Values in the model columns are veto count (veto percentage). The difference is Sonnet 5 minus Sonnet 4 in veto-rate percentage points. This is a cross-model descriptive comparison, not a within-model steering effect or causal estimate.}
\label{tab:claude-sonnet-4-vs-5}
\small
\begin{tabular}{llrrrrrl}
\toprule
Language & Prompt class & Matched $n$ & Sonnet 4 veto $n$ (\%) & Sonnet 5 veto $n$ (\%) & Difference (pp) & Agreement (\%) & Coverage \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\vspace{2pt}
\footnotesize{$^{\dagger}$Yoruba benign contains 45 shared prompts; all other cells have complete matched coverage.}
\end{table*}
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(table)

print(f"Wrote: {OUT}")
print(f"LaTeX rows: {frame.height}")
print(
    "Partial-coverage footnote required: "
    f"{frame.filter(pl.col('Coverage') == 'Partial cell: 45 prompts').height == 1}"
)
print(f"Overall matched prompts: {frame['Matched n'].sum()}")
print("Inference claim included: False")
