from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

MAIN = Path("artifacts/prior_llama33_phase2/results/baseline_completed_backup (1).jsonl")
TURBO = Path("artifacts/prior_llama33_phase2/results/turbo_steering_results_phase2.jsonl")

LANGUAGES = ["Hausa", "Yoruba"]
TARGETS = ["planner_only", "critic_only", "both"]
STRENGTHS = [20.0, 30.0]

VETO_RE = re.compile(r"FINAL_DECISION\s*:\s*VETO\b", re.I)
ALLOW_RE = re.compile(r"FINAL_DECISION\s*:\s*ALLOW\b", re.I)

def parser_outcome(response: object) -> str:
    if not isinstance(response, str):
        return "NONSTRING_RESPONSE"
    veto = bool(VETO_RE.search(response))
    allow = bool(ALLOW_RE.search(response))
    if veto and allow:
        return "BOTH_LABELS"
    if veto:
        return "VETO_EXPLICIT"
    if allow:
        return "ALLOW_EXPLICIT"
    return "NO_DECISION"

def load_rows(path: Path):
    rows = []
    invalid = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if isinstance(row, dict):
                row["_source_line"] = line_number
                rows.append(row)
            else:
                invalid += 1
    return rows, invalid

main_rows, main_invalid = load_rows(MAIN)
turbo_rows, turbo_invalid = load_rows(TURBO)

main_ids = {}
for row in main_rows:
    if (
        row.get("config") in TARGETS
        and float(row.get("strength", -999)) == 0.0
        and row.get("language") in LANGUAGES
        and row.get("harmfulness") == "harmful"
    ):
        key = (row["config"], row["language"])
        main_ids.setdefault(key, set()).add(str(row.get("id")))

turbo_ids = {}
turbo_outcomes = Counter()
turbo_duplicates = Counter()

for row in turbo_rows:
    try:
        strength = float(row.get("strength"))
    except (TypeError, ValueError):
        continue

    if (
        row.get("config") in TARGETS
        and strength in STRENGTHS
        and row.get("language") in LANGUAGES
        and row.get("harmfulness") == "harmful"
    ):
        key = (row["config"], strength, row["language"])
        turbo_ids.setdefault(key, []).append(str(row.get("id")))
        turbo_outcomes[(row["config"], strength, row["language"], parser_outcome(row.get("critic_response")))] += 1

print(f"main_invalid_json_rows={main_invalid}")
print(f"turbo_invalid_json_rows={turbo_invalid}")

print("\nTURBO_PARSER_OUTCOMES")
print("config\tstrength\tlanguage\tparser_outcome\tcount")
for key, count in sorted(turbo_outcomes.items()):
    print(*key, count, sep="\t")

print("\nTURBO_ID_AND_PAIRING_AUDIT")
print("config\tstrength\tlanguage\tmain_alpha0_n\tturbo_n\tturbo_unique_n\tshared_n\tmain_only\tturbo_only\tduplicate_turbo_rows")

for config in TARGETS:
    for strength in STRENGTHS:
        for language in LANGUAGES:
            base = main_ids.get((config, language), set())
            turbo_list = turbo_ids.get((config, strength, language), [])
            turbo_set = set(turbo_list)

            print(
                config,
                strength,
                language,
                len(base),
                len(turbo_list),
                len(turbo_set),
                len(base & turbo_set),
                len(base - turbo_set),
                len(turbo_set - base),
                len(turbo_list) - len(turbo_set),
                sep="\t",
            )
