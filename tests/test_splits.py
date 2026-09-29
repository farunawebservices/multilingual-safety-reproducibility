"""Split freezing: behavior-level, content-addressed, test guarded."""

from __future__ import annotations

import json

import pytest

from multilingual_safety.ids import behavior_id
from multilingual_safety.splits import (
    SplitAccessError,
    SplitError,
    assign_split,
    build_splits,
    load_split,
    load_splits,
    write_splits,
)


def behaviors(n: int = 40) -> list[str]:
    return [behavior_id(source_id="demo", source_item_id=str(i), arm="harmful") for i in range(n)]


def test_assignment_is_deterministic_given_a_salt():
    bid = behaviors(1)[0]
    assert assign_split(bid, salt="s1") == assign_split(bid, salt="s1")


def test_changing_the_salt_can_change_the_assignment():
    """Documented consequence: a new salt re-splits and invalidates test results."""
    bs = behaviors(60)
    a = {b: assign_split(b, salt="s1") for b in bs}
    c = {b: assign_split(b, salt="s2") for b in bs}
    assert a != c


def test_split_is_at_behavior_level_so_languages_never_straddle():
    """All seven renderings share a behavior id, hence a split. No leakage."""
    bid = behaviors(1)[0]
    assignments = {lang: assign_split(bid, salt="s1") for lang in ("en", "ha", "yo", "ko")}
    assert len(set(assignments.values())) == 1


def test_build_splits_is_hash_verified(tmp_path):
    payload = build_splits(behaviors(50), salt="s1")
    path = write_splits(payload, tmp_path / "splits.json")
    loaded = load_splits(path)
    assert loaded["split_hash"] == payload["split_hash"]
    assert sum(loaded["counts"].values()) == 50


def test_edited_split_file_is_detected(tmp_path):
    path = write_splits(build_splits(behaviors(50), salt="s1"), tmp_path / "splits.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    victim = next(iter(payload["assignments"]))
    payload["assignments"][victim] = "test" if payload["assignments"][victim] == "dev" else "dev"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SplitError, match="split hash mismatch"):
        load_splits(path)


def test_write_refuses_to_overwrite_a_frozen_split(tmp_path):
    path = tmp_path / "splits.json"
    write_splits(build_splits(behaviors(50), salt="s1"), path)
    with pytest.raises(SplitError, match="already exists"):
        write_splits(build_splits(behaviors(50), salt="s2"), path)


def test_reading_test_split_requires_explicit_opt_in(tmp_path):
    path = write_splits(build_splits(behaviors(50), salt="s1"), tmp_path / "splits.json")
    dev = load_split(path, "dev")
    assert dev
    with pytest.raises(SplitAccessError, match="refusing to read the test split"):
        load_split(path, "test")
    test = load_split(path, "test", allow_test=True)
    assert test
    assert not set(dev) & set(test)


def test_empty_or_degenerate_split_is_rejected():
    with pytest.raises(SplitError, match="empty split"):
        build_splits([], salt="s1")
    with pytest.raises(SplitError, match="test_fraction"):
        build_splits(behaviors(10), salt="s1", test_fraction=0.0)
