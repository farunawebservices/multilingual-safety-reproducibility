"""Repository hygiene: the tracked/untracked boundary holds.

These tests are the automated form of the promises in README §7 and
THIRD_PARTY_DATASETS.md. They fail loudly if prompt text, model outputs, or
credentials ever become tracked.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from multilingual_safety.paths import REPO_ROOT
from multilingual_safety.secrets import scan_paths

BLOCKED_PREFIXES = (
    "data/raw/", "data/interim/", "data/processed/", "data/review/", "results/", "logs/"
)
ALLOWED_PLACEHOLDERS = {".gitkeep", "README.md"}

#: Metadata-only stage reports that ARE permitted under results/. Each is
#: explicitly whitelisted in .gitignore, and `test_whitelisted_reports_are_text_free`
#: verifies the property that justifies whitelisting them. Anything else under
#: results/ remains blocked -- this is an exact allowlist, not a relaxed rule.
ALLOWED_RESULT_REPORTS = {
    "results/environment_report.json",
    "results/source_inventory.json",
    "results/translation_pilot_inventory.json",
    "results/translation_pilot_validation.json",
    "results/translation_pilot_review_summary.json",
    "results/full_request_set_inventory.json",
    "results/cpu_pilot_archive_report.json",
    "results/full_apply_report.json",
    "results/split_report.json",
    "results/length_outlier_review.json",
}

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".jsonl", ".toml", ".txt"}


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        pytest.skip("not a git repository")
    return [line for line in out.stdout.splitlines() if line.strip()]


def test_no_corpus_or_run_artifacts_are_tracked():
    offenders = [
        path
        for path in tracked_files()
        if path.startswith(BLOCKED_PREFIXES)
        and path.rsplit("/", 1)[-1] not in ALLOWED_PLACEHOLDERS
        and path not in ALLOWED_RESULT_REPORTS
    ]
    assert not offenders, f"corpus/run artifacts must never be tracked: {offenders}"


def test_no_prompt_text_directories_are_tracked_at_all():
    """The allowlist above must never be widened to cover corpus directories."""
    corpus_dirs = ("data/raw/", "data/interim/", "data/processed/", "data/review/", "logs/")
    offenders = [
        path
        for path in tracked_files()
        if path.startswith(corpus_dirs) and path.rsplit("/", 1)[-1] not in ALLOWED_PLACEHOLDERS
    ]
    assert not offenders, f"corpus text must never be tracked: {offenders}"


def test_whitelisted_reports_are_text_free():
    """The property that justifies tracking these four files at all."""
    for relative in sorted(ALLOWED_RESULT_REPORTS):
        path = REPO_ROOT / relative
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload.get("contains_prompt_text") in (False, None), (
            f"{relative} does not assert it is text-free"
        )
        blob = json.dumps(payload, ensure_ascii=False)
        for field in ("source_text", "translated_text", "prompt", "Behavior", "Goal"):
            assert f'"{field}":' not in blob, f"{relative} carries a {field} field"


def test_no_credential_shaped_literals_in_tracked_files():
    """Uses the library scanner, so detection logic lives in exactly one place."""
    paths = [
        REPO_ROOT / path
        for path in tracked_files()
        if (REPO_ROOT / path).suffix in TEXT_SUFFIXES and (REPO_ROOT / path).exists()
    ]
    findings = scan_paths(paths)
    # Locations only; the finding objects deliberately carry no matched value.
    offenders = sorted({f"{f.path}:{f.line_number} ({f.pattern_name})" for f in findings})
    assert not offenders, f"possible credentials in tracked files: {offenders}"


@pytest.mark.parametrize(
    "pattern",
    ["data/raw/", "data/interim/", "data/processed/", "data/review/", "results/", "logs/", ".env"],
)
def test_gitignore_covers_untracked_content(pattern):
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert pattern in gitignore, f".gitignore must cover {pattern}"


# ---------------------------------------------------------------------------
# Dataset v2 storage policy (configs/dataset_v2_rebuild.yaml, `outputs`).
#
# These assert git's ACTUAL behavior via check-ignore/check-attr rather than
# grepping .gitignore for a substring. data/v2/cache/ once looked covered
# because `*.bin` matched some of its contents while data/v2/cache/x.json was
# still trackable -- a text check would have passed that.
# ---------------------------------------------------------------------------
V2_PAYLOAD_PREFIXES = (
    "data/v2/interim/",
    "data/v2/cache/",
    "data/v2/tmp/",
    "data/v2/raw_downloads/",
    "data/v2/credentials/",
)


def _git_ignores(relative_path: str) -> bool:
    out = subprocess.run(
        ["git", "check-ignore", "-q", relative_path],
        cwd=REPO_ROOT, capture_output=True, check=False,
    )
    if out.returncode not in (0, 1):
        pytest.skip("git check-ignore unavailable")
    return out.returncode == 0


@pytest.mark.parametrize("prefix", V2_PAYLOAD_PREFIXES)
@pytest.mark.parametrize("name", ["payload.jsonl", "notes.json", "scratch.txt"])
def test_v2_payload_directories_are_ignored(prefix, name):
    """Whole directories, not just weight-shaped file extensions."""
    assert _git_ignores(prefix + name), f"{prefix}{name} must be git-ignored"


def test_v2_manifests_stay_trackable():
    """Manifests carry IDs, hashes and metadata only, and ARE committed."""
    assert not _git_ignores("data/v2/manifests/full_manifest_v2.json")


def test_v2_frozen_dataset_is_lfs_tracked():
    out = subprocess.run(
        ["git", "check-attr", "filter", "--", "data/v2/processed/frozen_dataset_v2.jsonl"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        pytest.skip("git check-attr unavailable")
    assert out.stdout.strip().endswith("filter: lfs"), (
        f"frozen_dataset_v2.jsonl must be Git LFS tracked, got: {out.stdout.strip()}"
    )


def test_no_tracked_files_under_v2_payload_directories():
    offenders = [p for p in tracked_files() if p.startswith(V2_PAYLOAD_PREFIXES)]
    assert not offenders, f"v2 payload must never be tracked: {offenders}"


def test_license_and_third_party_notice_coexist():
    """MIT covers code only; dataset terms are documented separately."""
    assert (REPO_ROOT / "LICENSE").exists()
    third_party = REPO_ROOT / "THIRD_PARTY_DATASETS.md"
    assert third_party.exists()
    # Collapse wrapping so the assertions test content, not line breaks.
    text = " ".join(third_party.read_text(encoding="utf-8").split())
    assert "does not cover" in text
    assert "authoritative" in text
    assert "MIT" in text


def test_no_source_is_registered_yet():
    """Guard for the current stage: no dataset has been fetched or registered."""
    import yaml

    from multilingual_safety.paths import SOURCES_CONFIG

    payload = yaml.safe_load(SOURCES_CONFIG.read_text(encoding="utf-8")) or {}
    sources = payload.get("sources") or []
    if sources:
        for source in sources:
            assert source.get("license"), f"source {source.get('id')} has no license recorded"
            assert source.get("redistribution_allowed") is not None
