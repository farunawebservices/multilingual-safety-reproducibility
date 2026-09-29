"""Deterministic hashing: same content -> same digest, everywhere, always."""

from __future__ import annotations

import subprocess
import sys
import unicodedata

import pytest

from multilingual_safety.hashing import (
    bytes_sha256,
    canonical_json,
    column_digest,
    deterministic_digest,
    directory_digest,
    file_sha256,
    json_sha256,
    table_digest,
    text_sha256,
    verify_file,
)

ROWS = [
    {"id": "b1", "text": "alpha", "cat": "x"},
    {"id": "b2", "text": "beta", "cat": "y"},
    {"id": "b3", "text": "gamma", "cat": "x"},
]


def test_bytes_and_file_hash_agree(tmp_path):
    payload = b"harmbench,jbb\n1,2\n"
    path = tmp_path / "f.csv"
    path.write_bytes(payload)
    assert file_sha256(path) == bytes_sha256(payload)
    assert len(file_sha256(path)) == 64


def test_verify_file_detects_a_single_byte_change(tmp_path):
    path = tmp_path / "f.csv"
    path.write_bytes(b"abc")
    digest = file_sha256(path)
    assert verify_file(path, digest)
    assert verify_file(path, digest.upper())  # case-insensitive comparison
    path.write_bytes(b"abd")
    assert not verify_file(path, digest)


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})
    assert json_sha256({"a": [1, 2]}) == json_sha256({"a": [1, 2]})


def test_json_hash_is_order_sensitive_for_lists():
    """Key order must not matter; list order must."""
    assert json_sha256({"a": [1, 2]}) != json_sha256({"a": [2, 1]})


def test_text_hash_normalises_but_still_detects_diacritic_loss():
    text = "Ẹ ṣé ọmọ"
    assert text_sha256(unicodedata.normalize("NFC", text)) == text_sha256(
        unicodedata.normalize("NFD", text)
    )
    assert text_sha256(text) != text_sha256("E se omo")


def test_table_digest_ignores_row_order_but_not_content():
    base = table_digest(ROWS)
    assert table_digest(list(reversed(ROWS))) == base
    edited = [dict(r) for r in ROWS]
    edited[0]["text"] = "ALPHA"
    assert table_digest(edited) != base


def test_table_digest_detects_dropped_and_duplicated_rows():
    base = table_digest(ROWS)
    assert table_digest(ROWS[:2]) != base
    assert table_digest([*ROWS, ROWS[0]]) != base, "row multiplicity must affect the digest"


def test_table_digest_detects_a_renamed_column():
    renamed = [{("behavior" if k == "text" else k): v for k, v in r.items()} for r in ROWS]
    assert table_digest(renamed) != table_digest(ROWS)


def test_column_digest_is_order_independent():
    assert column_digest(["a", "b", "c"]) == column_digest(["c", "a", "b"])
    assert column_digest(["a", "b"]) != column_digest(["a", "b", "b"])


def test_deterministic_digest_ignores_timestamps_and_run_ids():
    """Reruns must be comparable: only content changes should move the digest."""
    a = {"run_id": "run_1", "generated_at": "2026-01-01T00:00:00Z", "n_rows": 400}
    b = {"run_id": "run_2", "generated_at": "2026-08-16T12:00:00Z", "n_rows": 400}
    c = {"run_id": "run_3", "generated_at": "2026-08-16T12:00:00Z", "n_rows": 399}
    assert deterministic_digest(a) == deterministic_digest(b)
    assert deterministic_digest(a) != deterministic_digest(c)


def test_deterministic_digest_strips_nested_nondeterminism():
    a = {"outer": {"run_id": "x", "value": 1}}
    b = {"outer": {"run_id": "y", "value": 1}}
    assert deterministic_digest(a) == deterministic_digest(b)


def test_directory_digest_is_sorted_and_relative(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.csv").write_bytes(b"2")
    (tmp_path / "a.csv").write_bytes(b"1")
    (tmp_path / "sub" / "c.csv").write_bytes(b"3")
    digest = directory_digest(tmp_path)
    assert list(digest) == sorted(digest), "paths must be emitted in sorted order"
    assert set(digest) == {"a.csv", "b.csv", "sub/c.csv"}


def test_directory_digest_of_missing_directory_is_empty(tmp_path):
    assert directory_digest(tmp_path / "nope") == {}


def test_hashes_are_stable_across_separate_processes():
    """Guards against PYTHONHASHSEED / set-ordering leaking into a digest."""
    snippet = (
        "import sys; sys.path.insert(0, 'src');"
        "from multilingual_safety.hashing import table_digest, deterministic_digest;"
        "rows=[{'id':'b1','text':'alpha'},{'id':'b2','text':'beta'}];"
        "print(table_digest(rows));"
        "print(deterministic_digest({'z':1,'a':{'q':[1,2]}}))"
    )
    outputs = set()
    for seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            cwd=str(__import__("multilingual_safety.paths", fromlist=["REPO_ROOT"]).REPO_ROOT),
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1, f"digest varied with PYTHONHASHSEED: {outputs}"


@pytest.mark.parametrize("form", ["NFC", "NFD"])
def test_text_hash_form_is_explicit(form):
    assert len(text_sha256("ẹ", normalize_form=form)) == 64
