from pathlib import Path

import polars as pl

INFILE = Path(
    "paper_artifacts/metrics/"
    "gpt_oss_20b_planner_alpha1_primary_harmful_paper_table.csv"
)

OUT = Path(
    "paper_artifacts/tables/"
    "gpt_oss_20b_planner_alpha1_primary_harmful_results.tex"
)

frame = pl.read_csv(INFILE).sort("Language")

required = [
    "Language",
    "Prompt class",
    "Paired n",
    "Baseline failure (%)",
    "Planner-only alpha=1 failure (%)",
    "Steered minus baseline (pp)",
    "Holm-adjusted p",
    "Holm reject 0.05",
    "Coverage",
]

missing = [column for column in required if column not in frame.columns]
if missing:
    raise ValueError(f"Missing required table columns: {missing}")

if frame.height != 4:
    raise ValueError(f"Expected 4 paper-table rows; found {frame.height}.")

if frame["Paired n"].sum() != 1200:
    raise ValueError("Expected 1200 paired harmful prompts.")

if not (frame["Coverage"] == "Complete matched coverage").all():
    raise ValueError("All displayed rows must have complete matched coverage.")

def fmt_p(value: float) -> str:
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"

lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{GPT-OSS-20B harmful-prompt operational-failure rates under baseline and planner-only steering ($\alpha=1$). Negative changes indicate fewer operational failures under steering. Exact paired McNemar tests use Holm adjustment across the four language comparisons.}",
    r"\label{tab:gpt_oss_20b_planner_alpha1_harmful}",
    r"\small",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{lrrrrr}",
    r"\toprule",
    r"Language & $n$ & Baseline (\%) & Steered (\%) & $\Delta$ pp & Holm $p$ \\",
    r"\midrule",
]

for row in frame.to_dicts():
    marker = r"$^{*}$" if row["Holm reject 0.05"] else ""
    lines.append(
        f'{row["Language"]} & '
        f'{row["Paired n"]} & '
        f'{row["Baseline failure (%)"]:.1f} & '
        f'{row["Planner-only alpha=1 failure (%)"]:.1f} & '
        f'{row["Steered minus baseline (pp)"]:+.1f} & '
        f'{fmt_p(row["Holm-adjusted p"])}{marker} \\\\'
    )

lines.extend(
    [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}",
        r"\footnotesize $^{*}$Holm-adjusted $p < 0.05$. Each row contains 300 prompt-matched harmful examples; $\Delta$ is steered minus baseline.",
        r"\end{table}",
    ]
)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Wrote: {OUT}")
print(f"Rendered rows: {frame.height}")
print(f"Paired harmful prompts: {frame['Paired n'].sum()}")
print(
    "Holm-significant rows: "
    f"{frame.filter(pl.col('Holm reject 0.05')).height}"
)
print("Harmful-only scope: True")
