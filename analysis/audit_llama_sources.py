from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

SOURCES = [
    Path("artifacts/prior_llama33_phase2/results/agents_experiment_results_log.jsonl"),
    Path("artifacts/prior_llama33_phase2/results/baseline_completed_backup (1).jsonl"),
    Path("artifacts/prior_llama33_phase2/results/single_agent_baseline_log (1).jsonl"),
    Path("artifacts/prior_llama33_phase2/results/turbo_steering_results_phase2.jsonl"),
    Path("artifacts/prior_llama33_phase2/results/phase2_sae_steering_sweep_log (1).jsonl"),
]

for path in SOURCES:
    fields = Counter()
    types = Counter()
    cells = Counter()
    valid = invalid = nonobject = 0

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue

            if not isinstance(row, dict):
                nonobject += 1
                continue

            valid += 1
            fields.update(row.keys())
            for key, value in row.items():
                types[(key, type(value).__name__)] += 1

            cells[(
                str(row.get("config", "<missing>")),
                str(row.get("strength", row.get("alpha", "<missing>"))),
                str(row.get("language", "<missing>")),
                str(row.get("harmfulness", "<missing>")),
            )] += 1

    print(f"\nFILE\t{path}")
    print(f"VALID_ROWS\t{valid}")
    print(f"INVALID_JSON_ROWS\t{invalid}")
    print(f"NONOBJECT_ROWS\t{nonobject}")
    print("FIELDS\t" + ",".join(sorted(fields)))
    print("CELLS: config\tstrength_or_alpha\tlanguage\tharmfulness\tcount")
    for (config, strength, language, harmfulness), count in sorted(cells.items()):
        print(f"{config}\t{strength}\t{language}\t{harmfulness}\t{count}")
