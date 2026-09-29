#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

AUDIT = Path(
    "paper_artifacts/metrics/"
    "addition1_llama33_hausa_patching_v3_audit.csv"
)
TABLE = Path(
    "paper_artifacts/tables/"
    "addition1_llama33_hausa_patching_v3_table.tex"
)
FIGURE = Path(
    "paper_artifacts/figures/"
    "addition1_llama33_hausa_patching_v3_decisions.png"
)

audit = pd.read_csv(AUDIT)
values = dict(zip(audit["metric"], audit["value"]))

required = {
    "canonical_rows",
    "clean_allow",
    "clean_veto",
    "corrupted_allow",
    "corrupted_veto",
    "patched_allow",
    "patched_veto",
    "evaluable_clean_ne_corrupted",
    "recovered_patched_eq_clean",
    "recovery_rate_among_evaluable",
}
missing = required - set(values)
if missing:
    raise ValueError(f"Audit CSV missing metrics: {sorted(missing)}")

n_rows = int(values["canonical_rows"])
evaluable = int(values["evaluable_clean_ne_corrupted"])
recovered = int(values["recovered_patched_eq_clean"])
rate = float(values["recovery_rate_among_evaluable"])

if n_rows != 11 or evaluable != 1 or recovered != 0 or rate != 0.0:
    raise ValueError("Unexpected audited values; refuse to generate paper artifacts.")

TABLE.parent.mkdir(parents=True, exist_ok=True)
FIGURE.parent.mkdir(parents=True, exist_ok=True)

table_text = f"""% Generated from {AUDIT.as_posix()}
% Do not edit by hand.
\\begin{{table}}[t]
\\centering
\\caption{{Endpoint decisions for the 11-row Hausa patching v3 artifact. Recovery is evaluated only where the clean and corrupted decisions differ.}}
\\label{{tab:addition1-hausa-patching-v3}}
\\begin{{tabular}}{{lrr}}
\\toprule
Endpoint & ALLOW & VETO \\\\
\\midrule
Clean & {int(values["clean_allow"])} & {int(values["clean_veto"])} \\\\
Corrupted & {int(values["corrupted_allow"])} & {int(values["corrupted_veto"])} \\\\
Patched & {int(values["patched_allow"])} & {int(values["patched_veto"])} \\\\
\\midrule
Evaluable rows & \\multicolumn{{2}}{{r}}{{{evaluable}}} \\\\
Recovered rows & \\multicolumn{{2}}{{r}}{{{recovered}}} \\\\
Recovery rate & \\multicolumn{{2}}{{r}}{{{rate:.0%}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
TABLE.write_text(table_text, encoding="utf-8")

endpoints = ["Clean", "Corrupted", "Patched"]
allow = [
    int(values["clean_allow"]),
    int(values["corrupted_allow"]),
    int(values["patched_allow"]),
]
veto = [
    int(values["clean_veto"]),
    int(values["corrupted_veto"]),
    int(values["patched_veto"]),
]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.bar(endpoints, allow, label="ALLOW", color="#4C78A8")
ax.bar(endpoints, veto, bottom=allow, label="VETO", color="#E45756")
ax.set_ylabel("Rows")
ax.set_ylim(0, n_rows)
ax.set_title("Hausa patching v3 endpoint decisions")
ax.legend(frameon=False)
ax.text(
    0.5,
    -0.20,
    f"Recovery among evaluable rows: {recovered}/{evaluable} ({rate:.0%})",
    ha="center",
    va="top",
    transform=ax.transAxes,
)
fig.tight_layout()
fig.savefig(FIGURE, dpi=300, bbox_inches="tight")
plt.close(fig)

if not TABLE.exists() or TABLE.stat().st_size == 0:
    raise ValueError("LaTeX table generation failed")
if not FIGURE.exists() or FIGURE.stat().st_size == 0:
    raise ValueError("Figure generation failed")

print(f"Wrote {TABLE}")
print(f"Wrote {FIGURE}")
print(f"recovery={recovered}/{evaluable} ({rate:.0%})")
