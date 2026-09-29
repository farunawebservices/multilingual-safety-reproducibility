from __future__ import annotations

import json
from pathlib import Path

SOURCE = Path("artifacts/prior_llama33_phase2/results/baseline_completed_backup (1).jsonl")

PAIRS = [
    ("planner_only", 0.0, "planner_only", 20.0),
    ("critic_only", 0.0, "critic_only", 20.0),
]

records = {}

with SOURCE.open(encoding="utf-8", errors="replace") as handle:
    for line in handle:
        row = json.loads(line)
        key = (
            row.get("config"),
            float(row.get("strength")),
            row.get("language"),
            row.get("harmfulness"),
        )
        records.setdefault(key, set()).add(str(row.get("id")))

for base_config, base_strength, steer_config, steer_strength in PAIRS:
    print(f"\nPAIR\t{base_config}:{base_strength}\tvs\t{steer_config}:{steer_strength}")
    print("language\tharmfulness\tbaseline_n\tsteered_n\tshared_n\tbaseline_only\tsteered_only")

    languages = sorted({
        key[2] for key in records
        if key[0] in {base_config, steer_config}
    })
    classes = ["benign", "harmful"]

    for language in languages:
        for harmfulness in classes:
            base = records.get((base_config, base_strength, language, harmfulness), set())
            steer = records.get((steer_config, steer_strength, language, harmfulness), set())
            print(
                language,
                harmfulness,
                len(base),
                len(steer),
                len(base & steer),
                len(base - steer),
                len(steer - base),
                sep="\t",
            )
