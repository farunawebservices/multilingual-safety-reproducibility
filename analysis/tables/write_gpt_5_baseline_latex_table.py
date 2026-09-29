from pathlib import Path

import polars as pl

INFILE = Path(
    "paper_artifacts/metrics/"
    "gpt_5_frontier_baseline_paper_table.csv"
)

OUT = Path(
    "paper_artifacts/tables/"
    "gpt_5_frontier_baseline_descriptive_results.tex"
)

frame = pl.read_csv(INFILE)

required = [
    "Language",
    "Prompt class",
    "Valid n",
    "Veto n",
    "Non-veto n",
    "Veto rate (%)",
    "Non-veto operational failure (%)",
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

rows = []
for row in frame.sort(["Prompt class", "Language"]).iter_rows(named=True):
    coverage = (
        "Complete"
        if row["Coverage"] == "Complete cell"
        else row["Coverage"].replace("Partial: ", "Partial$^{\\dagger}$: ")
    )

    rows.append(
        " & ".join([
            str(row["Language"]),
            str(row["Prompt class"]),
            str(row["Valid n"]),
            str(row["Veto n"]),
            str(row["Non-veto n"]),
            f"{row['Veto rate (%)']:.2f}",
            f"{row['Non-veto operational failure (%)']:.2f}",
            coverage,
        ]) + r" \\"
    )

table = r"""\begin{table*}[t]
\centering
\caption{GPT-5 frontier-baseline descriptive results on valid Phase 1 records. Veto status is defined by \texttt{is\_vetoed=true}; non-veto operational failure is \texttt{is\_vetoed=false}. This table describes one baseline configuration at strength $0.0$ and does not estimate a steering effect or causal contrast.}
\label{tab:gpt-5-frontier-baseline}
\small
\begin{tabular}{llrrrrrl}
\toprule
Language & Prompt class & Valid $n$ & Veto $n$ & Non-veto $n$ & Veto (\%) & Non-veto failure (\%) & Coverage \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\vspace{2pt}
\footnotesize{$^{\dagger}$Yoruba benign contains 45 valid prompts. Harmful cells contain 286--291 valid prompts, rather than 300. The analysis excludes a separate 570-record duplicate subset with one malformed terminal JSONL line.}
\end{table*}
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(table)

print(f"Wrote: {OUT}")
print(f"LaTeX rows: {frame.height}")
print(f"Partial-coverage footnote rows: {partial_n}")
print(f"Total valid records: {frame['Valid n'].sum()}")
print("Inference claim included: False")
print("Excluded duplicate-subset caveat included: True")
