from pathlib import Path
import hashlib

import polars as pl

ROOT = Path(".")
METRICS = ROOT / "paper_artifacts" / "metrics"
TABLES = ROOT / "paper_artifacts" / "tables"
FIGURES = ROOT / "paper_artifacts" / "figures"
OUT = METRICS / "paper_artifact_status_inventory.csv"

MODEL_PATTERNS = {
    "qwen_25_72b": ["qwen", "qwen25", "qwen_25"],
    "gemma_4_31b": ["gemma"],
    "gpt_oss_20b": ["gpt_oss", "gpt-oss"],
    "gpt_5": ["gpt_5", "gpt-5"],
    "claude_sonnet_4": ["claude_sonnet_4", "claude-sonnet-4"],
    "claude_sonnet_5": ["claude_sonnet_5", "claude-sonnet-5"],
}

SEARCH_DIRS = [METRICS, TABLES, FIGURES]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

files = []
for directory in SEARCH_DIRS:
    if directory.exists():
        files.extend(
            path for path in directory.rglob("*")
            if path.is_file()
        )

rows = []
for model, patterns in MODEL_PATTERNS.items():
    matched = [
        path for path in files
        if any(pattern in path.name.lower() for pattern in patterns)
    ]

    metric_files = [
        path for path in matched
        if path.parent == METRICS and path.suffix == ".csv"
    ]
    table_files = [
        path for path in matched
        if path.parent == TABLES and path.suffix == ".tex"
    ]
    figure_files = [
        path for path in matched
        if path.parent == FIGURES
        and path.suffix.lower() in {".pdf", ".png", ".svg"}
    ]
    manifest_files = [
        path for path in matched
        if path.name.endswith("_artifact_manifest.json")
    ]
    handoff_files = [
        path for path in matched
        if path.name.endswith("_OPUS_HANDOFF.md")
    ]

    rows.append({
        "model": model,
        "metric_csv_n": len(metric_files),
        "latex_table_n": len(table_files),
        "figure_n": len(figure_files),
        "manifest_n": len(manifest_files),
        "opus_handoff_n": len(handoff_files),
        "sha256_inventory": sha256_file(matched[0]) if matched else None,
        "status": (
            "complete"
            if (
                metric_files
                and table_files
                and figure_files
                and manifest_files
                and handoff_files
            )
            else "incomplete_or_not_started"
        ),
    })

result = pl.DataFrame(rows).sort("model")
OUT.parent.mkdir(parents=True, exist_ok=True)
result.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Models inventoried: {result.height}")
print(
    "Complete models: "
    f"{result.filter(pl.col('status') == 'complete').height}"
)
print(
    "Incomplete/not-started models: "
    f"{result.filter(pl.col('status') != 'complete').height}"
)

with pl.Config(tbl_cols=-1, tbl_width_chars=220):
    print(result)
