from pathlib import Path

import polars as pl

INFILE = Path(
    "paper_artifacts/metrics/llama33_planner20_primary_paper_table.csv"
)

OUT = Path(
    "paper_artifacts/tables/llama33_planner20_primary_results.tex"
)

frame = pl.read_csv(INFILE).sort("Language")

required = [
    "Language",
    "Harmful paired n",
    "Harmful baseline failure (%)",
    "Harmful steered failure (%)",
    "Harmful delta (pp)",
    "Harmful Holm p",
    "Harmful Holm reject 0.05",
    "Benign paired n",
    "Benign baseline refusal (%)",
    "Benign steered refusal (%)",
    "Benign delta refusal (pp)",
    "Benign Holm p",
    "Benign Holm reject 0.05",
]

missing = [column for column in required if column not in frame.columns]
if missing:
    raise ValueError(f"Missing required table columns: {missing}")

if frame.height != 4:
    raise ValueError(f"Expected 4 language rows; found {frame.height}.")

if frame["Harmful paired n"].sum() != 1200:
    raise ValueError("Expected 1200 paired harmful prompts.")

if frame["Benign paired n"].sum() != 495:
    raise ValueError("Expected 495 paired benign prompts.")

def fmt_p(value: float) -> str:
    if value < 0.001:
        return r"$<0.001$"
    return f"{value:.3f}"

lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Llama-3.3-70B-Instruct planner-only steering ($\alpha=20$). Harmful-prompt outcomes are operational-failure rates; benign-prompt outcomes are refusal rates. Negative harmful changes indicate fewer operational failures, whereas positive benign changes indicate more false refusals. Exact paired McNemar tests use Holm adjustment separately within the four harmful and four benign language comparisons.}",
    r"\label{tab:llama33_planner20_primary}",
    r"\small",
    r"\setlength{\tabcolsep}{3.5pt}",
    r"\begin{tabular}{lrrrrrrrr}",
    r"\toprule",
    r"& \multicolumn{4}{c}{Harmful prompts: operational failure} & \multicolumn{4}{c}{Benign prompts: refusal} \\",
    r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
    r"Language & $n$ & Base. (\%) & Steered (\%) & $\Delta$ pp & $n$ & Base. (\%) & Steered (\%) & $\Delta$ pp \\",
    r"\midrule",
]

for row in frame.to_dicts():
    harmful_marker = (
        r"$^{*}$" if row["Harmful Holm reject 0.05"] else ""
    )
    benign_marker = (
        r"$^{*}$" if row["Benign Holm reject 0.05"] else ""
    )

    lines.append(
        f'{row["Language"]} & '
        f'{row["Harmful paired n"]} & '
        f'{row["Harmful baseline failure (%)"]:.1f} & '
        f'{row["Harmful steered failure (%)"]:.1f} & '
        f'{row["Harmful delta (pp)"]:+.1f}{harmful_marker} & '
        f'{row["Benign paired n"]} & '
        f'{row["Benign baseline refusal (%)"]:.1f} & '
        f'{row["Benign steered refusal (%)"]:.1f} & '
        f'{row["Benign delta refusal (pp)"]:+.1f}{benign_marker} \\\\'
    )

lines.extend(
    [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}",
        r"\footnotesize $^{*}$Holm-adjusted $p < 0.05$. Harmful arm: 300 prompt-matched examples per language (1,200 total). Benign arm: 150 prompt-matched examples in Arabic, English, and Hausa, and 45 in Yoruba (495 total).",
        r"\end{table*}",
    ]
)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Wrote: {OUT}")
print(f"Rendered rows: {frame.height}")
print(f"Paired harmful prompts: {frame['Harmful paired n'].sum()}")
print(f"Paired benign prompts: {frame['Benign paired n'].sum()}")
print(
    "Harmful Holm-significant rows: "
    f"{frame.filter(pl.col('Harmful Holm reject 0.05')).height}"
)
print(
    "Benign Holm-significant rows: "
    f"{frame.filter(pl.col('Benign Holm reject 0.05')).height}"
)
print("Different harmful and benign denominators disclosed: True")
