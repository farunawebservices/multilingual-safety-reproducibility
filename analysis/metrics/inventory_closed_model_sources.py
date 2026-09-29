from hashlib import sha256
from pathlib import Path
import csv

RAW_ROOT = Path("data/raw")
DERIVED_ROOT = Path("data/derived")
REGISTRY_ROOT = Path("registry")

OUT = Path(
    "paper_artifacts/metrics/"
    "closed_model_source_inventory.csv"
)

MODELS = {
    "gpt_5": ["gpt_5", "gpt-5", "gpt5"],
    "claude_sonnet_4": [
        "claude_sonnet_4",
        "claude-sonnet-4",
        "claude sonnet 4",
        "sonnet_4",
        "sonnet-4",
    ],
    "claude_sonnet_5": [
        "claude_sonnet_5",
        "claude-sonnet-5",
        "claude sonnet 5",
        "sonnet_5",
        "sonnet-5",
    ],
}

SEARCH_ROOTS = [RAW_ROOT, DERIVED_ROOT, REGISTRY_ROOT]
SUFFIXES = {".jsonl", ".json", ".csv", ".parquet", ".yaml", ".yml"}

def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

files = []
for root in SEARCH_ROOTS:
    if root.exists():
        files.extend(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUFFIXES
        )

rows = []
for model, patterns in MODELS.items():
    matches = sorted(
        path for path in files
        if any(pattern in str(path).lower() for pattern in patterns)
    )

    if not matches:
        rows.append({
            "model": model,
            "path": None,
            "file_type": None,
            "bytes": None,
            "sha256": None,
            "location": "not_found",
        })
        continue

    for path in matches:
        if path.is_relative_to(RAW_ROOT):
            location = "raw"
        elif path.is_relative_to(DERIVED_ROOT):
            location = "derived"
        elif path.is_relative_to(REGISTRY_ROOT):
            location = "registry"
        else:
            location = "other"

        rows.append({
            "model": model,
            "path": str(path),
            "file_type": path.suffix.lower().lstrip("."),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "location": location,
        })

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "model",
            "path",
            "file_type",
            "bytes",
            "sha256",
            "location",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote: {OUT}")
print(f"Closed models inventoried: {len(MODELS)}")
print(f"Source/derived files found: {sum(row['path'] is not None for row in rows)}")

for model in MODELS:
    model_rows = [row for row in rows if row["model"] == model]
    found = [row for row in model_rows if row["path"] is not None]
    print(f"{model}: {len(found)} file(s)")
