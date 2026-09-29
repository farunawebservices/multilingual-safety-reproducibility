from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

FILES = [
    Path("data/raw/phase1/gpt_5/originals/frontier_baseline_gpt_5_results.jsonl"),
    Path("data/raw/phase1/gpt_oss_20b/originals/gpt_oss_baseline_results.jsonl"),
    Path("data/raw/phase1/gemma_4_31b/originals/gemma_4_31b_baseline_results (1).jsonl"),
    Path("data/raw/phase1/gemma_4_31b/originals/gemma_4_31b_steered_results (1).jsonl"),
]

def text_or_missing(row, key):
    value = row.get(key)
    return str(value) if value is not None else "<missing>"

for path in FILES:
    counts = Counter()
    ids = []
    parsed = invalid = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue

            if not isinstance(row, dict):
                invalid += 1
                continue

            parsed += 1
            counts[
                (
                    text_or_missing(row, "config"),
                    text_or_missing(row, "strength"),
                    text_or_missing(row, "language"),
                    text_or_missing(row, "harmfulness"),
                )
            ] += 1
            if row.get("id") is not None:
                ids.append(str(row["id"]))

    duplicate_ids = len(ids) - len(set(ids))

    print(f"\n=== {path} ===")
    print(f"parsed_rows={parsed}\tinvalid_rows={invalid}\tduplicate_ids={duplicate_ids}")
    print("config\tstrength\tlanguage\tharmfulness\tn")
    for (config, strength, language, harmfulness), n in sorted(counts.items()):
        print(f"{config}\t{strength}\t{language}\t{harmfulness}\t{n}")
