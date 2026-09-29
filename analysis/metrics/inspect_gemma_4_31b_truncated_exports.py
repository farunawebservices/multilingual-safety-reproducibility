from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

OUT = Path("registry/field_reports/gemma_4_31b_truncated_export_diagnostic.md")

SOURCES = [
    Path(
        "data/raw/phase1/gemma_4_31b/originals/"
        "gemma_4_31b_baseline_results (1).jsonl"
    ),
    Path(
        "data/raw/phase1/gemma_4_31b/originals/"
        "gemma_4_31b_steered_results (1).jsonl"
    ),
]

LOGS = [
    Path(
        "data/raw/phase1/gemma_4_31b/originals/"
        "gemma_baseline_run.log"
    ),
    Path(
        "data/raw/phase1/gemma_4_31b/originals/"
        "gemma_steering_run.log"
    ),
]

CONTEXT_RADIUS = 180
LOG_TERMS = re.compile(
    r"error|exception|traceback|keyboardinterrupt|cancel|oom|out of memory|"
    r"timeout|killed|write|json|flush|save|output|complete|finish",
    re.IGNORECASE,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def escaped_context(text: str, position: int) -> str:
    start = max(0, position - CONTEXT_RADIUS)
    end = min(len(text), position + CONTEXT_RADIUS)
    excerpt = text[start:end]
    return (
        excerpt
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def count_unescaped(text: str, target: str) -> int:
    count = 0
    escaped = False

    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == target:
            count += 1

    return count


lines = [
    "# Gemma 4 31B Truncated-Export Diagnostic",
    "",
    "This report is diagnostic only. It does not repair or modify source files.",
    "",
]

for source in SOURCES:
    raw_lines = source.read_bytes().splitlines()
    invalid = []

    for line_number, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            continue
        try:
            json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            text = raw.decode("utf-8", errors="replace")
            position = getattr(exc, "pos", 0)
            invalid.append((line_number, raw, text, exc, position))

    lines.extend([
        f"## {source.name}",
        "",
        f"- Source SHA-256: `{sha256_bytes(source.read_bytes())}`",
        f"- Nonempty physical lines: `{sum(bool(x.strip()) for x in raw_lines)}`",
        f"- Invalid nonempty lines: `{len(invalid)}`",
        "",
    ])

    for line_number, raw, text, exc, position in invalid:
        open_braces = count_unescaped(text, "{")
        close_braces = count_unescaped(text, "}")
        open_brackets = count_unescaped(text, "[")
        close_brackets = count_unescaped(text, "]")
        quote_count = count_unescaped(text, '"')

        lines.extend([
            f"### Invalid line {line_number}",
            "",
            f"- Line SHA-256: `{sha256_bytes(raw)}`",
            f"- Bytes: `{len(raw)}`",
            f"- Decoder error: `{type(exc).__name__}: {exc}`",
            f"- Decoder position: `{position}`",
            f"- Starts with `{{`: `{text.lstrip().startswith('{')}`",
            f"- Ends with `}}`: `{text.rstrip().endswith('}')}`",
            f"- Unescaped brace balance: `{open_braces - close_braces}`",
            f"- Unescaped bracket balance: `{open_brackets - close_brackets}`",
            f"- Unescaped quote count: `{quote_count}`",
            "",
            "Escaped context around decoder position:",
            "",
            "```text",
            escaped_context(text, position),
            "```",
            "",
        ])

lines.extend([
    "# Generation-log signals",
    "",
])

for log_path in LOGS:
    lines.extend([
        f"## {log_path.name}",
        "",
    ])

    if not log_path.is_file():
        lines.extend(["- Log not found.", ""])
        continue

    matches = []
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle, start=1):
            if LOG_TERMS.search(line):
                cleaned = " ".join(line.split())
                matches.append((number, cleaned[:500]))

    lines.append(f"- Log SHA-256: `{sha256_bytes(log_path.read_bytes())}`")
    lines.append(f"- Matching operational lines: `{len(matches)}`")
    lines.append("")

    for number, cleaned in matches[-30:]:
        lines.append(f"- Line {number}: `{cleaned}`")

    lines.append("")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8")

print(f"Wrote: {OUT}")
print("Source files were not modified.")
print("Review only these sections:")
print("  1. Decoder error and escaped context")
print("  2. Brace/bracket/quote balance")
print("  3. Final generation-log signals")
