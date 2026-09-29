from pathlib import Path
import hashlib
import json

manifest_path = Path(
    "addition_6_prompt_filter_baseline/prompts/"
    "addition6_pilot_manifest_v1.jsonl"
)

rows = []

with manifest_path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue

        row = json.loads(line)
        prompt_text = row["prompt_text"]

        row["prompt_sha256"] = hashlib.sha256(
            prompt_text.encode("utf-8")
        ).hexdigest()

        rows.append(row)

with manifest_path.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(
            json.dumps(row, ensure_ascii=False) + "\n"
        )

print(f"Fixed {len(rows)} rows.")
print(
    "Manifest SHA256:",
    hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
)