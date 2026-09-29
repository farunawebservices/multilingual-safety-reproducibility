#!/usr/bin/env python3
"""
Verify Addition 2 (Llama 3.3 Turbo Alpha 30) Opus handoff.

This script validates that:
1. All required artifacts exist
2. SHA-256 hashes match the handoff
3. The handoff contains required cross-phase limitation language
4. The handoff does NOT contain forbidden causal claims
5. Numerical results match the authoritative unified CSV

Fails if any check fails.
"""
import hashlib
import re
from pathlib import Path

# Required artifacts
UNIFIED_RESULTS = Path(
    "paper_artifacts/metrics/llama33_turbo_alpha30_role_strength_unified_results.csv"
)
PAPER_TABLE_CSV = Path(
    "paper_artifacts/metrics/llama33_turbo_alpha30_paper_table.csv"
)
PAPER_TABLE_TEX = Path(
    "paper_artifacts/tables/llama33_turbo_alpha30_cross_phase_comparison.tex"
)
FIGURE_PDF = Path(
    "paper_artifacts/figures/llama33_turbo_alpha30_cross_phase_comparison.pdf"
)
FIGURE_PNG = Path(
    "paper_artifacts/figures/llama33_turbo_alpha30_cross_phase_comparison.png"
)
MANIFEST_CSV = Path(
    "paper_artifacts/metrics/llama33_turbo_artifact_manifest.csv"
)
HANDOFF = Path(
    "paper_artifacts/metrics/llama33_turbo_alpha30_OPUS_HANDOFF.md"
)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

errors = []

# Check 1: All artifacts exist
print("Checking artifact existence...")
for artifact in [UNIFIED_RESULTS, PAPER_TABLE_CSV, PAPER_TABLE_TEX, FIGURE_PDF, FIGURE_PNG, MANIFEST_CSV, HANDOFF]:
    if not artifact.is_file():
        errors.append(f"Missing artifact: {artifact}")
        print(f"  MISSING: {artifact}")
    else:
        print(f"  OK: {artifact}")

if errors:
    print("\nFAIL: Missing artifacts")
    for e in errors:
        print(f"  - {e}")
    raise SystemExit(1)

# Check 2: SHA-256 hashes in handoff match actual files
print("\nChecking SHA-256 hashes...")
handoff_text = HANDOFF.read_text(encoding="utf-8")
for artifact, name in [
    (UNIFIED_RESULTS, "unified_results"),
    (PAPER_TABLE_CSV, "paper_table_csv"),
    (PAPER_TABLE_TEX, "paper_table_tex"),
    (FIGURE_PDF, "figure_pdf"),
    (FIGURE_PNG, "figure_png"),
    (MANIFEST_CSV, "manifest_csv"),
]:
    actual_hash = sha256(artifact)
    if actual_hash[:16] not in handoff_text:
        errors.append(f"Handoff does not contain hash for {name}: {actual_hash[:16]}...")
        print(f"  MISMATCH: {name}")
    else:
        print(f"  OK: {name} ({actual_hash[:16]}...)")

# Check 3: Required cross-phase limitation language
print("\nChecking required limitation language...")
required_phrases = [
    "cross-phase",
    "associational",
    "not causal",
    "Matched by record_id; phases differ",
    "non-prompt run-condition equivalence is not established",
    "random-vector control",
    "NOT a new causal experiment",
    "archived cross-phase role/strength reanalysis",
]
for phrase in required_phrases:
    if phrase not in handoff_text:
        errors.append(f"Handoff lacks required phrase: {phrase!r}")
        print(f"  MISSING: {phrase!r}")
    else:
        print(f"  OK: {phrase!r}")

# Check 4: Forbidden causal claims
print("\nChecking for forbidden causal claims...")
forbidden_patterns = [
    r"\bcauses\b",
    r"\bcaused\b",
    r"\bcausal effect\b",
    r"\beffect of\b",
    r"\bimpact of\b",
    r"\bleads to\b",
    r"\bdemonstrates role specificity\b",
    r"\bproves\b",
]
# But allow these in negated contexts
allowed_negated = [
    "NOT a new causal experiment",
    "not causal",
    "DO NOT claim",
    "MUST NOT claim",
]

for pattern in forbidden_patterns:
    matches = re.finditer(pattern, handoff_text, re.IGNORECASE)
    for match in matches:
        # Check if this is in an allowed negated context
        start = max(0, match.start() - 50)
        end = min(len(handoff_text), match.end() + 50)
        context = handoff_text[start:end]
        if not any(allowed in context for allowed in allowed_negated):
            errors.append(f"Forbidden causal claim found: {match.group()!r} in context: {context!r}")
            print(f"  FORBIDDEN: {match.group()!r}")

if not any("FORBIDDEN" in str(e) for e in errors):
    print("  OK: No forbidden causal claims found")

# Check 5: Numerical results match unified CSV
print("\nChecking numerical consistency...")
import polars as pl
unified = pl.read_csv(UNIFIED_RESULTS)
table_csv = pl.read_csv(PAPER_TABLE_CSV)

if unified.height != 6:
    errors.append(f"Unified results should have 6 rows; found {unified.height}")
if table_csv.height != 6:
    errors.append(f"Paper table should have 6 rows; found {table_csv.height}")

# Check key numerical fields match
unified_sorted = unified.sort(["config", "language"])
table_sorted = table_csv.sort(["Configuration", "Language"])

for field_map in [
    ("baseline_failure_rate", "Baseline failure rate (%)"),
    ("steered_failure_rate", "Alpha 30 failure rate (%)"),
    ("delta_steered_minus_baseline_percentage_points", "Change (pp)"),
]:
    unified_col, table_col = field_map
    unified_vals = unified_sorted[unified_col].to_list()
    table_vals = table_sorted[table_col].to_list()
    # Allow for rounding differences
    for i, (u, t) in enumerate(zip(unified_vals, table_vals)):
        if abs(u - t) > 0.15:  # Allow 0.1% rounding tolerance
            errors.append(f"Numerical mismatch in {field_map}: row {i}, unified={u}, table={t}")
            print(f"  MISMATCH: {field_map} row {i}")

if not any("MISMATCH" in str(e) or "Numerical" in str(e) for e in errors):
    print("  OK: Numerical results consistent")

# Final report
print("\n" + "="*60)
if errors:
    print("FAIL: Addition 2 handoff verification FAILED")
    print(f"Found {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    raise SystemExit(1)
else:
    print("PASS: Addition 2 handoff verified successfully")
    print(f"  Handoff: {HANDOFF}")
    print(f"  Unified results: {UNIFIED_RESULTS}")
    print(f"  Paper table CSV: {PAPER_TABLE_CSV}")
    print(f"  Paper table LaTeX: {PAPER_TABLE_TEX}")
    print(f"  Figure PDF: {FIGURE_PDF}")
    print(f"  Figure PNG: {FIGURE_PNG}")
    print(f"  Manifest CSV: {MANIFEST_CSV}")
