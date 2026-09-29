"""Identifier stability and content hashing."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from multilingual_safety.ids import (
    behavior_id,
    content_hash,
    is_behavior_id,
    is_run_id,
    record_id,
    run_id,
    stable_hash,
)


def test_behavior_id_is_stable_and_text_free():
    a = behavior_id(source_id="src", source_item_id="42", arm="harmful")
    b = behavior_id(source_id="src", source_item_id="42", arm="harmful")
    assert a == b
    assert is_behavior_id(a)
    assert a.startswith("beh_")


def test_arm_is_part_of_the_behavior_key():
    """Harmful and benign items sharing a source id must not collapse together."""
    harmful = behavior_id(source_id="src", source_item_id="42", arm="harmful")
    benign = behavior_id(source_id="src", source_item_id="42", arm="benign")
    assert harmful != benign


def test_behavior_id_rejects_bad_inputs():
    with pytest.raises(ValueError, match="arm must be"):
        behavior_id(source_id="src", source_item_id="1", arm="neutral")
    with pytest.raises(ValueError, match="non-empty"):
        behavior_id(source_id="", source_item_id="1", arm="harmful")


def test_one_behavior_yields_distinct_record_ids_per_language():
    bid = behavior_id(source_id="src", source_item_id="7", arm="benign")
    ids = {lang: record_id(behavior=bid, language=lang) for lang in ("en", "ha", "yo")}
    assert len(set(ids.values())) == 3
    assert ids["yo"].startswith("rec_yo_")


def test_record_variant_produces_a_new_id_not_an_overwrite():
    bid = behavior_id(source_id="src", source_item_id="7", arm="benign")
    base = record_id(behavior=bid, language="ha")
    retranslated = record_id(behavior=bid, language="ha", variant="retranslation_v2")
    assert base != retranslated


def test_record_id_requires_a_behavior_id():
    with pytest.raises(ValueError, match="expected a behavior id"):
        record_id(behavior="12345", language="en")


def test_content_hash_normalises_before_hashing():
    """NFC and NFD renderings of the same Yoruba string must hash identically."""
    import unicodedata

    text = "Ẹ ṣé"
    assert content_hash(unicodedata.normalize("NFC", text)) == content_hash(
        unicodedata.normalize("NFD", text)
    )
    assert len(content_hash(text)) == 64


def test_content_hash_distinguishes_diacritic_loss():
    """ASCII-folded text must NOT hash the same as the correctly accented text."""
    assert content_hash("Ẹ ṣé ọmọ") != content_hash("E se omo")


def test_stable_hash_is_key_order_independent():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_run_id_shape_and_uniqueness_over_time():
    t1 = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 8, 16, 12, 0, 1, tzinfo=timezone.utc)
    r1 = run_id(config_hash="abc", now=t1)
    r2 = run_id(config_hash="abc", now=t2)
    assert is_run_id(r1) and is_run_id(r2)
    assert r1 != r2, "reruns of the same config must not collide; they are separate runs"
    assert r1.startswith("run_20260816T120000Z_")


def test_run_id_requires_a_config_hash():
    with pytest.raises(ValueError, match="config_hash is required"):
        run_id(config_hash="")


def test_is_run_id_rejects_malformed():
    assert not is_run_id("run_nope")
    assert not is_run_id("20260816T120000Z_abcdef12")
