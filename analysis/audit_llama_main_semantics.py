from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

SOURCE = Path("artifacts/prior_llama33_phase2/results/baseline_completed_backup (1).jsonl")
OUT = Path("registry/llama/main_semantics_audit.tsv")

VETO_RE = re.compile(r"FINAL_DECISION\s*:\s*VETO\b", re.I)
ALLOW_RE = re.compile(r"FINAL_DECISION\s*:\s*ALLOW\b", re.I)

def outcome(response):
    if not isinstance(response, str):
        return "NONSTRING_RESPONSE"
    has_veto = bool(VETO_RE.search(response))
    has_allow = bool(ALLOW_RE.search(response))
    if has_veto and has_allow:
        return "BOTH_LABELS"
    if has_veto:
        return "VETO_EXPLICIT"
    if has_allow:
        return "ALLOW_EXPLICIT"
    return "NO_DECISION"

counts = Counter()
ids = {}
legacy_mismatch = Counter()
invalid = 0

with SOURCE.open(encoding="utf-8", errors="replace") as handle:
    for line_number, line in enumerate(handle, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue

        config = row.get("config")
        strength = row.get("strength")
        language = row.get("language")
        harmfulness = row.get("harmfulness")
        parsed = outcome(row.get("critic_response"))
        legacy_veto = row.get("is_vetoed") is True

        key = (config, strength, language, harmfulness, parsed)
        counts[key] += 1
        ids.setdefault((config, strength, language, harmfulness), []).append(str(row.get("id")))

        if legacy_veto != (parsed in {"VETO_EXPLICIT", "BOTH_LABELS"}):
            legacy_mismatch[(config, strength, language, harmfulness)] += 1

with OUT.open("w", encoding="utf-8") as handle:
    handle.write("config\tstrength\tlanguage\tharmfulness\tparser_outcome\tcount\n")
    for key, count in sorted(counts.items()):
        handle.write("\t".join(map(str, key)) + f"\t{count}\n")

print(f"invalid_json_rows={invalid}")
print("config\tstrength\tlanguage\tharmfulness\tparser_outcome\tcount")
for key, count in sorted(counts.items()):
    print(*key, count, sep="\t")

print("\nLEGACY_PARSER_DISAGREEMENTS")
print("config\tstrength\tlanguage\tharmfulness\tcount")
for key, count in sorted(legacy_mismatch.items()):
    print(*key, count, sep="\t")

print("\nDUPLICATE_IDS_WITHIN_CELL")
print("config\tstrength\tlanguage\tharmfulness\tn\tunique_ids\tduplicate_rows")
for key, values in sorted(ids.items()):
    unique_n = len(set(values))
    print(*key, len(values), unique_n, len(values) - unique_n, sep="\t")
