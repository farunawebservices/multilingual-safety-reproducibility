from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

SOURCE = Path("data/raw/phase1/gpt_oss_20b/originals/gpt_oss_baseline_results.jsonl")
OUT = Path("registry/gpt_oss_decision_semantics_audit.json")

final_pattern = re.compile(
    r"FINAL_DECISION\s*:\s*(VETO|ALLOW)",
    flags=re.IGNORECASE,
)

counts = Counter()
by_language = Counter()
examples = []

with SOURCE.open("r", encoding="utf-8", errors="replace") as handle:
    for line_number, line in enumerate(handle, start=1):
        row = json.loads(line)

        raw_flag = row.get("is_vetoed")
        recorded = "VETO" if raw_flag is True else "ALLOW" if raw_flag is False else "UNKNOWN"

        response = row.get("critic_response")
        if not isinstance(response, str):
            extracted = "MISSING_OR_NONSTRING"
        else:
            matches = final_pattern.findall(response)
            extracted = matches[-1].upper() if matches else "NO_FINAL_DECISION"

        status = (
            "MATCH"
            if extracted in {"VETO", "ALLOW"} and extracted == recorded
            else "MISMATCH"
            if extracted in {"VETO", "ALLOW"}
            else extracted
        )

        counts[(recorded, extracted, status)] += 1
        by_language[(str(row.get("language")), status)] += 1

        if status == "MISMATCH":
            examples.append({
                "source_line": line_number,
                "record_id": row.get("id"),
                "language": row.get("language"),
                "harmfulness": row.get("harmfulness"),
                "recorded_decision": recorded,
                "extracted_decision": extracted,
            })

report = {
    "source_file": str(SOURCE),
    "overall": [
        {
            "recorded_decision": recorded,
            "extracted_decision": extracted,
            "status": status,
            "count": count,
        }
        for (recorded, extracted, status), count in sorted(counts.items())
    ],
    "by_language": [
        {"language": language, "status": status, "count": count}
        for (language, status), count in sorted(by_language.items())
    ],
    "mismatch_metadata_only": examples[:50],
    "mismatch_count": len(examples),
}
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("recorded\textracted\tstatus\tcount")
for (recorded, extracted, status), count in sorted(counts.items()):
    print(f"{recorded}\t{extracted}\t{status}\t{count}")

print("\nlanguage\tstatus\tcount")
for (language, status), count in sorted(by_language.items()):
    print(f"{language}\t{status}\t{count}")

print(f"\nMISMATCH_COUNT\t{len(examples)}")
