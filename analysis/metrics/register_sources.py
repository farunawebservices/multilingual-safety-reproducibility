#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOTS = [
    Path("artifacts"),
    Path("results"),
    Path("addition_1_causal_patching"),
    Path("addition_4_calibration"),
    Path("addition_5_adversarial_robustness"),
    Path("addition_6_prompt_filter_baseline"),
    Path("data/raw"),
]

OUT = Path("registry/source_artifacts.csv")
SKIP = {".git", ".venv", "__pycache__"}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def category(path: Path) -> str:
    text = str(path).lower()
    for key in (
        "addition_1", "addition_2", "addition_3",
        "addition_4", "addition_5", "addition_6",
        "llama", "qwen", "gemma", "gpt_oss",
        "gpt_5", "claude_sonnet_4", "claude_sonnet_5",
    ):
        if key in text:
            return key
    return "unclassified"

rows = []
for root in ROOTS:
    if not root.exists():
        continue
    for path in root.rglob("*"):
        if not path.is_file() or any(part in SKIP for part in path.parts):
            continue
        rows.append({
            "path": str(path),
            "artifact_group": category(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "suffix": path.suffix.lower(),
        })

rows.sort(key=lambda row: (row["artifact_group"], row["path"]))
OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        "path", "artifact_group", "bytes", "sha256", "suffix"
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {OUT}")
print(f"Registered {len(rows)} files")
for group in sorted({row['artifact_group'] for row in rows}):
    print(f"{group}: {sum(row['artifact_group'] == group for row in rows)}")
