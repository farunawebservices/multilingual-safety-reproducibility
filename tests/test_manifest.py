"""Manifest invariants: text-free, complete, duplicate-free, never empty."""

from __future__ import annotations

import json

import pytest

from multilingual_safety.ids import behavior_id, content_hash, record_id
from multilingual_safety.languages import LANGUAGE_CODES
from multilingual_safety.manifest import (
    BehaviorRecord,
    ManifestError,
    assert_no_forbidden_fields,
    read_manifest,
    summarize,
    write_manifest,
)


def make_record(*, item: str = "1", language: str = "en", arm: str = "harmful") -> BehaviorRecord:
    bid = behavior_id(source_id="demo", source_item_id=item, arm=arm)
    return BehaviorRecord(
        record_id=record_id(behavior=bid, language=language),
        behavior_id=bid,
        language=language,
        arm=arm,
        behavior_family="family_a",
        source_id="demo",
        source_item_id=item,
        content_sha256=content_hash(f"placeholder-{item}-{language}"),
        translation_method="original" if language == "en" else "machine_unreviewed",
    )


def test_record_roundtrip(tmp_path):
    records = [make_record(item="1", language=lang) for lang in LANGUAGE_CODES]
    path = tmp_path / "behaviors.jsonl"
    assert write_manifest(records, path) == 7
    loaded = read_manifest(path)
    assert len(loaded) == 7
    assert {r.language for r in loaded} == set(LANGUAGE_CODES)


def test_manifest_rows_contain_no_prompt_text(tmp_path):
    path = tmp_path / "behaviors.jsonl"
    write_manifest([make_record(item="1", language=lang) for lang in LANGUAGE_CODES], path)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert "prompt" not in row
        assert "text" not in row
        assert len(row["content_sha256"]) == 64


@pytest.mark.parametrize("field", ["prompt", "text", "completion", "api_key", "transcript"])
def test_forbidden_fields_are_rejected(field):
    with pytest.raises(ManifestError, match="forbidden field"):
        assert_no_forbidden_fields({"record_id": "rec_en_x", field: "anything"})


def test_empty_manifest_is_a_stop_condition(tmp_path):
    with pytest.raises(ManifestError, match="empty manifest"):
        write_manifest([], tmp_path / "behaviors.jsonl")


def test_duplicate_record_ids_are_a_stop_condition(tmp_path):
    rec = make_record()
    with pytest.raises(ManifestError, match="duplicate record_id"):
        write_manifest([rec, rec], tmp_path / "behaviors.jsonl")


def test_non_pivot_language_must_declare_translation_provenance():
    with pytest.raises(ManifestError, match="translation_method='original'"):
        bid = behavior_id(source_id="demo", source_item_id="9", arm="harmful")
        BehaviorRecord(
            record_id=record_id(behavior=bid, language="yo"),
            behavior_id=bid,
            language="yo",
            arm="harmful",
            behavior_family="family_a",
            source_id="demo",
            source_item_id="9",
            content_sha256=content_hash("x"),
            translation_method="original",
        )


def test_malformed_hash_rejected():
    bid = behavior_id(source_id="demo", source_item_id="9", arm="harmful")
    with pytest.raises(ManifestError, match="content_sha256"):
        BehaviorRecord(
            record_id=record_id(behavior=bid, language="en"),
            behavior_id=bid,
            language="en",
            arm="harmful",
            behavior_family="family_a",
            source_id="demo",
            source_item_id="9",
            content_sha256="deadbeef",
        )


def test_malformed_line_raises_rather_than_being_skipped(tmp_path):
    path = tmp_path / "behaviors.jsonl"
    write_manifest([make_record()], path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    with pytest.raises(ManifestError, match="malformed JSON"):
        read_manifest(path)


def test_summarize_surfaces_incomplete_language_coverage():
    records = [make_record(item="1", language=lang) for lang in ("en", "ha")]
    stats = summarize(records)
    assert stats["n_rows"] == 2
    assert stats["n_behaviors"] == 1
    assert stats["n_behaviors_incomplete"] == 1
    assert "ko" in stats["languages_with_zero_rows"]
    assert "benign" in stats["arms_with_zero_rows"]
