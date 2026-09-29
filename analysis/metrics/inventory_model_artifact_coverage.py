from __future__ import annotations

import csv
from pathlib import Path

OUT = Path("paper_artifacts/metrics/model_artifact_coverage_inventory.csv")

ROOTS = [
    Path("analysis"),
    Path("data/derived"),
    Path("paper_artifacts"),
    Path("docs"),
    Path("registry"),
]

MODELS = {
    "gpt_oss": ["gpt_oss", "gpt-oss"],
    "llama33": ["llama33", "llama-3.3", "llama_3.3"],
    "qwen25_72b": ["qwen", "qwen2.5", "qwen-2.5", "72b"],
    "gemma_4_31b": ["gemma", "31b"],
    "gpt_5": ["gpt_5", "gpt-5", "gpt5"],
    "claude_sonnet_4": ["claude_sonnet_4", "claude-sonnet-4", "sonnet_4"],
    "claude_sonnet_5": ["claude_sonnet_5", "claude-sonnet-5", "sonnet_5"],
}

SUFFIXES = {".py", ".csv", ".parquet", ".json", ".jsonl", ".md", ".tex", ".pdf"}

rows = []

for root in ROOTS:
    if not root.exists():
        continue

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUFFIXES:
            continue

        path_str = path.as_posix()
        lower = path_str.lower()

        matches = [
            model
            for model, terms in MODELS.items()
            if any(term in lower for term in terms)
        ]

        if not matches:
            continue

        rows.append({
            "path": path_str,
            "model_matches": ";".join(matches),
            "suffix": path.suffix.lower(),
            "bytes": path.stat().st_size,
            "location": path.parts[0] if path.parts else "",
            "is_script": path.suffix.lower() == ".py",
            "is_canonical_parquet": (
                path.suffix.lower() == ".parquet"
                and "data/derived" in path_str
            ),
            "is_metric_csv": (
                path.suffix.lower() == ".csv"
                and "paper_artifacts/metrics" in path_str
            ),
            "is_table": (
                path.suffix.lower() == ".tex"
                and "paper_artifacts/tables" in path_str
            ),
            "is_figure": (
                path.suffix.lower() == ".pdf"
                and "paper_artifacts/figures" in path_str
            ),
            "is_methods_doc": (
                path.suffix.lower() == ".md"
                and "docs/" in path_str
            ),
        })

OUT.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "path",
    "model_matches",
    "suffix",
    "bytes",
    "location",
    "is_script",
    "is_canonical_parquet",
    "is_metric_csv",
    "is_table",
    "is_figure",
    "is_methods_doc",
]

with OUT.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote: {OUT}")
print(f"Matched files: {len(rows)}")

for model in MODELS:
    matched = [row for row in rows if model in row["model_matches"].split(";")]
    scripts = sum(row["is_script"] for row in matched)
    parquet = sum(row["is_canonical_parquet"] for row in matched)
    metrics = sum(row["is_metric_csv"] for row in matched)
    tables = sum(row["is_table"] for row in matched)
    figures = sum(row["is_figure"] for row in matched)
    docs = sum(row["is_methods_doc"] for row in matched)

    print(
        f"{model:18} "
        f"files={len(matched):3} "
        f"scripts={scripts:2} "
        f"parquet={parquet:2} "
        f"metrics={metrics:2} "
        f"tables={tables:2} "
        f"figures={figures:2} "
        f"docs={docs:2}"
    )

print("\nQwen-related paths:")
for row in rows:
    if "qwen25_72b" in row["model_matches"].split(";"):
        print(row["path"])
