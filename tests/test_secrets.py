"""No-secret detection, and a live check on every Stage 0 artifact.

The credential-shaped strings in this file are **synthetic** — assembled at
runtime from fragments so that the literals never appear intact in the source,
and constructed to match the pattern shape only. None is a real credential.
"""

from __future__ import annotations

import pytest

from multilingual_safety.paths import MANIFESTS_DIR, REPO_ROOT, RESULTS_DIR
from multilingual_safety.secrets import (
    SECRET_PATTERNS,
    assert_no_secrets,
    scan_file,
    scan_paths,
    scan_text,
)

# Synthetic, never-real credentials, built so the literal is not present in source.
FAKE_OPENAI = "sk-" + "A" * 32
FAKE_ANTHROPIC = "sk-ant-" + "B" * 40
FAKE_HF = "hf_" + "C" * 34
FAKE_AWS_ID = "AKIA" + "D" * 16
FAKE_GITHUB = "ghp_" + "E" * 36
FAKE_GOOGLE = "AIza" + "F" * 35


@pytest.mark.parametrize(
    ("name", "sample"),
    [
        ("openai_api_key", FAKE_OPENAI),
        ("anthropic_api_key", FAKE_ANTHROPIC),
        ("huggingface_token", FAKE_HF),
        ("aws_access_key_id", FAKE_AWS_ID),
        ("github_token", FAKE_GITHUB),
        ("google_api_key", FAKE_GOOGLE),
    ],
)
def test_credential_shapes_are_detected(name, sample):
    findings = scan_text(f"token = '{sample}'")
    assert findings, f"{name} was not detected"
    assert any(f.pattern_name == name for f in findings)


def test_private_key_block_is_detected():
    # Assembled at runtime: this file must not itself contain a
    # credential-shaped literal, or it trips the repo hygiene scan.
    header = "-----" + "BEGIN RSA PRIVATE KEY" + "-----"
    assert scan_text(f"{header}\nabc\n")


def test_generic_assigned_secret_is_detected():
    key_name = "api" + "_key"
    pw_name = "pass" + "word"
    assert scan_text(f"{key_name} = 'abcdef1234567890xyz'")
    assert scan_text(f'{pw_name}: "hunter2hunter2hunter2"')


def test_findings_never_contain_the_matched_value():
    """A scanner that prints the secret has copied it into your logs."""
    findings = scan_text(f"key={FAKE_OPENAI}", path="demo.py")
    assert findings
    for finding in findings:
        rendered = str(finding)
        assert FAKE_OPENAI not in rendered
        assert "A" * 32 not in rendered
        assert "demo.py" in rendered and "openai_api_key" in rendered
        assert not hasattr(finding, "value")


def test_placeholders_do_not_trip_the_scanner():
    assert not scan_text("ANTHROPIC_API_KEY=")
    assert not scan_text("api_key: 'your-api-key-here'")
    assert not scan_text("api_key = '<REDACTED>'")
    assert not scan_text("OPENAI_API_KEY=changeme-please-set-this")


def test_ordinary_code_is_not_flagged():
    assert not scan_text("api_key_env: OPENAI_API_KEY  # name only, never a value")
    assert not scan_text("sha256 = '8d81accedd38eaaf8b760618622bb888417d1fd0'")
    assert not scan_text("def load(path): return Path(path).read_text()")


def test_line_numbers_are_reported():
    findings = scan_text(f"line one\nline two\nkey={FAKE_HF}\n")
    assert findings[0].line_number == 3


def test_scan_paths_is_sorted_and_skips_binaries(tmp_path):
    (tmp_path / "b.py").write_text(f"k='{FAKE_HF}'", encoding="utf-8")
    (tmp_path / "a.py").write_text(f"k='{FAKE_GITHUB}'", encoding="utf-8")
    (tmp_path / "c.bin").write_bytes(FAKE_OPENAI.encode())
    findings = scan_paths(sorted(tmp_path.iterdir()))
    assert [f.path for f in findings] == sorted(f.path for f in findings)
    assert not any(f.path.endswith(".bin") for f in findings)


def test_assert_no_secrets_raises_without_leaking(tmp_path):
    bad = tmp_path / "leak.py"
    bad.write_text(f"KEY = '{FAKE_ANTHROPIC}'", encoding="utf-8")
    with pytest.raises(RuntimeError) as exc:
        assert_no_secrets([bad])
    message = str(exc.value)
    assert "leak.py" in message
    assert FAKE_ANTHROPIC not in message, "the error must not echo the secret"


def test_missing_or_unreadable_file_yields_no_findings(tmp_path):
    assert scan_file(tmp_path / "absent.py") == []


def test_pattern_registry_is_well_formed():
    names = [name for name, _ in SECRET_PATTERNS]
    assert len(names) == len(set(names))
    assert all(name.islower() for name in names)


# --- live checks on the real Stage 0 artifacts -----------------------------
STAGE0_ARTIFACTS = [
    RESULTS_DIR / "environment_report.json",
    RESULTS_DIR / "source_inventory.json",
    MANIFESTS_DIR / "source_manifest.yaml",
]


def test_stage0_artifacts_contain_no_secrets():
    existing = [p for p in STAGE0_ARTIFACTS if p.exists()]
    if not existing:
        pytest.skip("Stage 0 has not been run in this working tree")
    assert_no_secrets(existing)


def test_stage0_scripts_and_configs_contain_no_secrets():
    paths = [
        *sorted((REPO_ROOT / "scripts").glob("*.py")),
        *sorted((REPO_ROOT / "src" / "multilingual_safety").rglob("*.py")),
        *sorted((REPO_ROOT / "configs").glob("*.yaml")),
    ]
    assert_no_secrets(paths)


def test_stage0_reports_contain_no_prompt_text():
    """Reports are commit candidates, so they must carry metadata only."""
    inventory = RESULTS_DIR / "source_inventory.json"
    if not inventory.exists():
        pytest.skip("Stage 0 inspection has not been run")
    text = inventory.read_text(encoding="utf-8")
    # Column names may appear; actual behavior text must not. Spot-check with
    # phrasing characteristic of the acquired corpora.
    for fragment in ("Write a tutorial", "Give step-by-step instructions", "Explain how to"):
        assert fragment not in text, f"prompt text leaked into the inventory: {fragment!r}"
