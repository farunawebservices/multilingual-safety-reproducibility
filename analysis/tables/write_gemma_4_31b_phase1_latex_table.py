from pathlib import Path

import polars as pl

INFILE = Path(
    "paper_artifacts/metrics/"
    "gemma_4_31b_phase1_complete_case_paper_table.csv"
)

OUT = Path(
    "paper_artifacts/tables/"
    "gemma_4_31b_phase1_complete_case_results.tex"
)

frame = pl.read_csv(INFILE)

required = [
    "Language",
    "Prompt class",
    "Matched n",
    "Baseline non-veto failure (%)",
    "Steered non-veto failure (%)",
    "Change (percentage points)",
    "Bootstrap 95% CI low (pp)",
    "Bootstrap 95% CI high (pp)",
    "Holm-adjusted p",
    "Holm significant (0.05)",
    "Coverage",
]

missing = sorted(set(required) - set(frame.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if frame.height != 3:
    raise ValueError(f"Expected 3 paper rows; found {frame.height}")

if frame.filter(pl.col("Holm significant (0.05)") == "Yes").height != 0:
    raise ValueError("Expected zero Holm-significant rows.")

def escape_latex(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )

rows = []
for row in frame.sort(["Prompt class", "Language"]).iter_rows(named=True):
    interval = (
        f"[{row['Bootstrap 95% CI low (pp)']:.2f}, "
        f"{row['Bootstrap 95% CI high (pp)']:.2f}]"
    )

    coverage = (
        "Matched subset$^{\\dagger}$"
        if row["Coverage"] == "Matched subset"
        else "Complete"
    )

    rows.append(
        " & ".join([
            escape_latex(row["Language"]),
            escape_latex(row["Prompt class"]),
            str(row["Matched n"]),
            f"{row['Baseline non-veto failure (%)']:.2f}",
            f"{row['Steered non-veto failure (%)']:.2f}",
            f"{row['Change (percentage points)']:+.2f}",
            interval,
            f"{row['Holm-adjusted p']:.4f}",
            coverage,
        ]) + r" \\"
    )

table = r"""\begin{table*}[t]
\centering
\caption{Gemma 4 31B Phase 1 paired complete-case results for planner-only steering at $\alpha=1.0$. Outcome is non-veto operational failure ($\texttt{is\_vetoed}=\texttt{false}$); it is not an explicit-ALLOW rate. Changes are steered minus baseline percentage points. Confidence intervals are paired-bootstrap percentile 95\% intervals. Exact McNemar $p$-values were Holm-adjusted across the three Gemma cells; no cell is significant at family-wise $\alpha=0.05$.}
\label{tab:gemma-phase1-complete-case}
\small
\begin{tabular}{llrrrrrrl}
\toprule
Language & Prompt class & Matched $n$ & Baseline (\%) & Steered (\%) & $\Delta$ pp & 95\% CI (pp) & Holm $p$ & Coverage \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\vspace{2pt}
\footnotesize{$^{\dagger}$English benign includes 101 shared valid IDs; 11 steered-only valid IDs were excluded from paired inference. One truncated terminal JSONL record is excluded under the complete-case valid-JSON policy.}
\end{table*}
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(table)

print(f"Wrote: {OUT}")
print(f"LaTeX rows: {frame.height}")
print(
    "Matched-subset footnote required: "
    f"{frame.filter(pl.col('Coverage') == 'Matched subset').height == 1}"
)
print("Holm-significant rows: 0")
