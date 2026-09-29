from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

targets = [
    Path("data/raw/phase1/gpt_5/originals/frontier_baseline_gpt_5_results.jsonl"),
    Path("data/raw/phase1/gpt_oss_20b/originals/gpt_oss_baseline_results.jsonl"),
]

pattern = re.compile(
    r"(id|lang|language|role|agent|model|decision|verdict|label|status|parse|"
    r"terminat|harmful|benign|alpha|strength|layer|steer|config)",
    re.I,
)

def safe_value(value: Any) -> str | None:
    if value is None or isinstance(value, (bool, int, float)):
        return str(value)
    if isinstance(value, str) and len(value) <= 120:
        return value
    return None

for path in targets:
    all_keys = Counter()
    candidates: dict[str, Counter] = defaultdict(Counter)
    nested: dict[str, Counter] = defaultdict(Counter)
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
            all_keys.update(row.keys())

            for key, value in row.items():
                if not pattern.search(key):
                    continue

                if isinstance(value, dict):
                    nested[key].update(value.keys())
                    continue

                rendered = safe_value(value)
                if rendered is not None:
                    candidates[key][rendered] += 1
                else:
                    candidates[key][f"<{type(value).__name__}>"] += 1

    print(f"\n=== {path} ===")
    print(f"PARSED_ROWS: {parsed}; INVALID_ROWS: {invalid}")
    print("TOP_LEVEL_KEYS:", ", ".join(sorted(all_keys)))

    print("\nCANDIDATE FIELDS")
    for key in sorted(candidates):
        print(f"{key}: {dict(candidates[key].most_common(40))}")

    if nested:
        print("\nNESTED FIELD NAMES")
        for key in sorted(nested):
            print(f"{key}: {sorted(nested[key])}")
