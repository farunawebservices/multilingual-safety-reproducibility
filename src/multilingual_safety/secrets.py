"""Credential detection for files that are about to be committed.

Two rules shape this module:

1. **Never echo a match.** Findings report the pattern name, file, and line
   number. A scanner that prints the secret it found has copied that secret into
   a log, a terminal scrollback, and possibly a chat transcript.
2. **Prefer a false positive to a false negative.** A spurious hit costs a
   glance; a missed key costs a rotation.

This is a tripwire for the obvious mistake -- a pasted key, a committed `.env` --
not a replacement for dedicated secret scanning or pre-receive hooks.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

#: (name, pattern). Anchored on distinctive prefixes to limit false positives.
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_api_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("anthropic_api_key", re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{20,}")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("aws_secret_access_key", re.compile(r"(?i)aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{40}")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer_token_literal", re.compile(r"(?i)authorization\s*[=:]\s*['\"]?bearer\s+[A-Za-z0-9\-._~+/]{20,}")),
    ("generic_assigned_secret", re.compile(
        r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|password)\b\s*[=:]\s*['\"][^'\"\s]{12,}['\"]"
    )),
)

#: Values that look like credentials but are placeholders. Matching one of these
#: suppresses the finding, so `.env.example` and docs do not trip the scanner.
PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "your-",
    "your_",
    "xxx",
    "<",
    "changeme",
    "placeholder",
    "example",
    "dummy",
    "redacted",
    "fake",
    "test-key",
    "...",
)

TEXT_SUFFIXES: frozenset[str] = frozenset(
    {".py", ".md", ".yaml", ".yml", ".json", ".jsonl", ".toml", ".txt", ".cfg", ".ini", ".sh", ".env"}
)


@dataclass(frozen=True, slots=True)
class SecretFinding:
    """A potential credential. Deliberately carries no matched value."""

    pattern_name: str
    path: str
    line_number: int

    def __str__(self) -> str:
        return f"{self.path}:{self.line_number}: possible {self.pattern_name}"


def _is_placeholder(matched: str) -> bool:
    lowered = matched.lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def scan_text(text: str, *, path: str = "<string>") -> list[SecretFinding]:
    """Scan text for credential-shaped literals. Returns findings without values."""
    findings: list[SecretFinding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for name, pattern in SECRET_PATTERNS:
            match = pattern.search(line)
            if match and not _is_placeholder(match.group(0)):
                findings.append(SecretFinding(pattern_name=name, path=path, line_number=lineno))
    return findings


def scan_file(path: str | Path) -> list[SecretFinding]:
    """Scan one file. Unreadable or binary files yield no findings."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return scan_text(text, path=str(p))


def scan_paths(paths: Iterable[str | Path]) -> list[SecretFinding]:
    """Scan many files, skipping non-text suffixes. Sorted for determinism."""
    findings: list[SecretFinding] = []
    for path in paths:
        p = Path(path)
        if p.is_file() and p.suffix in TEXT_SUFFIXES:
            findings.extend(scan_file(p))
    return sorted(findings, key=lambda f: (f.path, f.line_number, f.pattern_name))


def assert_no_secrets(paths: Iterable[str | Path]) -> None:
    """Raise if any file appears to contain a credential.

    The error names locations only -- it never includes the matched value, which
    would defeat the purpose by copying the secret into the traceback.
    """
    findings = scan_paths(paths)
    if findings:
        raise RuntimeError(
            "possible credential(s) detected:\n  "
            + "\n  ".join(str(f) for f in findings)
            + "\n\nRotate the credential first, then remove it. Values are referenced by "
            "environment variable NAME only (CLAUDE.md §7)."
        )
