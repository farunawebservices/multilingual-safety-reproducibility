from pathlib import Path

import polars as pl

INFILE = Path(
    "paper_artifacts/metrics/"
    "qwen25_72b_planner_alpha1_primary_paper_table.csv"
)

OUT = Path(
    "paper_artifacts/tables/"
    "qwen25_72b_planner_alpha1_primary_results.tex"
)

frame = pl.read_csv(INFILE)

required = [
    "Language",
    "Prompt class",
    "Paired n",
    "Baseline failure n",
    "Baseline failure (%)",
    "Steered failure n",
    "Steered failure (%)",
    "Steered minus baseline (pp)",
    "Worsened n",
    "Improved n",
    "McNemar exact p",
    "Holm adjusted p",
    "Holm reject 0.05",
    "Bootstrap 95% CI low (pp)",
    "Bootstrap 95% CI high (pp)",
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
    raise ValueError("Expected complete matched coverage in all rows.")

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
    holm = (
        r"\textbf{Yes}"
        if row["Holm reject 0.05"]
        else "No"
    )

    rows.append(
        " & ".join([
            tex(row["Language"]),
            tex(row["Prompt class"]),
            str(row["Paired n"]),
            (
                f"{row['Baseline failure n']} "
                f"({row['Baseline failure (%)']:.1f})"
            ),
            (
                f"{row['Steered failure n']} "
                f"({row['Steered failure (%)']:.1f})"
            ),
            f"{row['Steered minus baseline (pp)']:+.1f}",
            (
                f"[{row['Bootstrap 95% CI low (pp)']:.1f}, "
                f"{row['Bootstrap 95% CI high (pp)']:.1f}]"
            ),
            holm,
        ]) + r" \\"
    )

table = r"""\begin{table*}[t]
\centering
\caption{Qwen 2.5 72B planner-only steering at $\alpha=1.0$ versus the matched baseline. Values in the condition columns are operational-failure count (percentage). Differences are steered minus baseline percentage points; negative values indicate fewer operational failures under steering. Exact paired McNemar tests use Holm adjustment from the predeclared 33-test Qwen primary family. Bootstrap intervals are 95\% intervals over matched prompt IDs.}
\label{tab:qwen25-72b-planner-alpha1}
\small
\begin{tabular}{llrrrrrl}
\toprule
Language & Prompt class & Paired $n$ & Baseline failure $n$ (\%) & Steered failure $n$ (\%) & Difference (pp) & 95\% CI (pp) & Holm $p<.05$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\vspace{2pt}
\footnotesize{Operational failure follows the audited Qwen decision semantics: an explicit veto is treated as safe, whereas a non-veto unresolved outcome is treated as an operational failure. All eight cells have complete prompt-matched coverage (1,695 paired IDs total).}
\end{table*}
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(table)

print(f"Wrote: {OUT}")
print(f"LaTeX rows: {frame.height}")
print(f"Paired prompts: {frame['Paired n'].sum()}")
print(
    "Holm-significant rows: "
    f"{frame.filter(pl.col('Holm reject 0.05')).height}"
)
print("Complete-coverage rows: 8")
print("Inference family stated: 33-test Holm")
