"""The pilot writer honours redirected output paths (dataset-v2 support).

``scripts/04_translate_pilot.py`` used to write its five outputs to hardcoded v1
paths, two of which are TRACKED history:

    data/manifests/translation_pilot_manifest.json
    results/translation_pilot_inventory.json

Running the dataset-v2 pilot would therefore have overwritten frozen v1
artifacts. The outputs are now parameterised, with the v1 paths as defaults.

These tests assert both halves of that: redirection actually redirects, and the
v1 destinations are left byte-identical. The fixture text below is invented
placeholder prose -- no corpus prompt text appears in this file.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

from multilingual_safety.hashing import text_sha256
from multilingual_safety.paths import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "04_translate_pilot.py"

#: The two tracked v1 artifacts the default paths point at.
V1_TRACKED_OUTPUTS = (
    REPO_ROOT / "data" / "manifests" / "translation_pilot_manifest.json",
    REPO_ROOT / "results" / "translation_pilot_inventory.json",
)

# Invented placeholder, deliberately not drawn from any benchmark.
_FIXTURE_TEXT = "Describe how a public library organises its catalogue."

SELECTION_ROW = {
    "behavior_id": "beh_0123456789ab",
    "source": "jailbreakbench",
    "source_item_id": "0",
    "category": "Disinformation",
    "harm_label": "benign",
    "source_text": _FIXTURE_TEXT,
    "source_text_sha256": text_sha256(_FIXTURE_TEXT),
}


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


@pytest.fixture
def selection_file(tmp_path):
    path = tmp_path / "selection.jsonl"
    path.write_text(json.dumps(SELECTION_ROW, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _run(selection_file, tmp_path, extra=()):
    out = {name: tmp_path / f"{name}.out" for name in
           ("requests", "outputs", "queue", "manifest", "inventory")}
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--selection", str(selection_file),
         *[arg for name, path in out.items() for arg in (f"--{name}", str(path))],
         *extra],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    return result, out


def test_redirected_outputs_are_written_where_asked(selection_file, tmp_path):
    result, out = _run(selection_file, tmp_path)
    assert result.returncode == 0, result.stderr
    for name, path in out.items():
        assert path.exists(), f"--{name} was not honoured (nothing at {path})"
        assert path.stat().st_size > 0, f"--{name} produced an empty file"


def test_tracked_v1_artifacts_are_not_touched(selection_file, tmp_path):
    """The regression this parameterisation exists to prevent."""
    before = {p: _digest(p) for p in V1_TRACKED_OUTPUTS}
    result, _ = _run(selection_file, tmp_path)
    assert result.returncode == 0, result.stderr
    after = {p: _digest(p) for p in V1_TRACKED_OUTPUTS}
    changed = [str(p.relative_to(REPO_ROOT)) for p in V1_TRACKED_OUTPUTS if before[p] != after[p]]
    assert not changed, f"frozen v1 artifacts were modified: {changed}"


def test_no_translation_happens_without_a_provider(selection_file, tmp_path):
    """No provider configured => rows stay pending; nothing is invented."""
    result, out = _run(selection_file, tmp_path)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in
            out["outputs"].read_text(encoding="utf-8").splitlines() if line.strip()]
    non_english = [r for r in rows if r["language"] != "en"]
    assert non_english, "expected non-English rows"
    assert all(r["translated_text"] is None for r in non_english)
    assert all(r["translation_status"] == "PENDING_TRANSLATION" for r in non_english)
    assert all(r["review_required"] for r in non_english)


def test_dry_run_writes_nothing(selection_file, tmp_path):
    result, out = _run(selection_file, tmp_path, extra=("--dry-run",))
    assert result.returncode == 0, result.stderr
    assert not [p for p in out.values() if p.exists()]


def test_defaults_still_point_at_the_v1_paths():
    """Preserving v1 behavior: an unflagged invocation is unchanged."""
    help_text = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    ).stdout
    for expected in (
        "translation_pilot_requests.jsonl",
        "translation_pilot_outputs.jsonl",
        "translation_review_queue.md",
        "translation_pilot_manifest.json",
        "translation_pilot_inventory.json",
    ):
        assert expected in help_text, f"default for {expected} changed"
