#!/usr/bin/env python3
"""Write Addition 6 Opus handoff from the artifact manifest."""

import json
from pathlib import Path

MANIFEST_JSON = Path(
    "paper_artifacts/metrics/"
    "addition6_prompt_filter_baseline_artifact_manifest.json"
)
HANDOFF_MD = Path(
    "paper_artifacts/metrics/"
    "addition6_prompt_filter_baseline_OPUS_HANDOFF.md"
)

manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
artifacts = manifest["artifacts"]

provenance_lines = []
for name, record in artifacts.items():
    provenance_lines.append(
        f"- `{name}`: `{record['path']}`\n"
        f"  SHA-256: `{record['sha256']}`"
    )

handoff = f"""# Addition 6: Prompt Filter Baseline Opus Handoff

## Purpose

Addition 6 compares three conditions on the specified evaluation prompt set:

1. No intervention.
2. Prompt filter baseline.
3. SAE steering.

All numerical claims must be taken from the audited metrics CSV and associated
paper table.

## Key results

| Condition | Harmful veto rate | Benign FPR | Missing labels |
|---|---:|---:|---:|
| No intervention | 52.5% (210/400) | 0.0% (0/60) | 4 |
| Prompt filter | 53.5% (214/400) | 0.0% (0/60) | 5 |
| SAE steering | 57.5% (230/400) | 5.0% (3/60) | 25 |

SAE steering increased the harmful veto rate by 5.0 percentage points relative
to no intervention and by 4.0 percentage points relative to the prompt-filter
baseline. It also introduced three benign false positives and more missing
labels.

## Interpretation boundary

This three-condition comparison evaluates outcomes for the specified prompt set,
prompt-filter implementation, SAE steering intervention, model configuration,
and evaluation protocol. It does not establish that SAE steering is causally
superior to prompt filtering in other models, languages, datasets, prompt
distributions, or deployment settings. The results therefore do not establish
general superiority of SAE steering over prompt filtering.

## Artifact provenance

{chr(10).join(provenance_lines)}

## Writing guidance

Present Addition 6 as a baseline comparison with trade-offs, not as a
universal success claim. Retain the increased benign false-positive rate and
missing-label count when discussing the steering condition.
"""

HANDOFF_MD.write_text(handoff, encoding="utf-8")
print(f"Wrote: {{HANDOFF_MD}}")
