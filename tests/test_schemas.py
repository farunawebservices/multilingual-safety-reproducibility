"""Schema validation: pinned expectations, and loud failure on drift.

The two expectations most worth protecting are the ones that contradict the
obvious guess, and both are asserted here:

* JBB exposes ONE `behaviors` config with `harmful`/`benign` **splits** --
  not two configs by those names.
* HarmBench's official val/test partition is **file-level**; no split column
  exists in any of its CSVs.
"""

from __future__ import annotations

import pytest

from multilingual_safety.schemas import (
    EXPECTED_SCHEMAS,
    HARMBENCH_COLUMNS,
    HARMBENCH_FUNCTIONAL_CATEGORIES,
    JBB_COLUMNS,
    SchemaMismatchError,
    assert_part,
    check_columns,
    check_hf_configs,
    check_row_count,
    get_part,
    get_schema,
    validate_part,
)


def test_both_sources_are_pinned_separately():
    assert sorted(EXPECTED_SCHEMAS) == ["harmbench", "jailbreakbench"]


def test_jbb_uses_splits_not_configs_for_harmful_and_benign():
    """The load-bearing correction: `harmful`/`benign` are SPLITS of one config."""
    schema = get_schema("jailbreakbench")
    assert "behaviors" in schema.hf_configs
    assert set(schema.hf_configs["behaviors"]) == {"harmful", "benign"}
    # There is no config named `harmful` or `benign`.
    assert "harmful" not in schema.hf_configs
    assert "benign" not in schema.hf_configs


def test_jbb_expected_counts_and_columns():
    for part_id, arm in (("harmful", "harmful"), ("benign", "benign")):
        part = get_part("jailbreakbench", part_id)
        assert part.n_rows == 100
        assert part.columns == JBB_COLUMNS
        assert part.arm == arm
        assert part.id_column == "Index"


def test_harmbench_partition_is_file_level_and_sums():
    all_ = get_part("harmbench", "text_all")
    val = get_part("harmbench", "text_val")
    test = get_part("harmbench", "text_test")
    assert val.n_rows + test.n_rows == all_.n_rows == 400
    assert val.n_rows == 80
    assert test.n_rows == 320
    # No split column exists in the pinned column set.
    for part in (all_, val, test):
        assert part.columns == HARMBENCH_COLUMNS
        assert not any("split" in c.lower() for c in part.columns)


def test_harmbench_functional_categories_are_pinned():
    assert HARMBENCH_FUNCTIONAL_CATEGORIES == ("standard", "contextual", "copyright")


def test_harmbench_context_string_is_nullable_by_design():
    """ContextString is empty for non-contextual rows; that is not a defect."""
    part = get_part("harmbench", "text_all")
    assert "ContextString" in part.nullable_columns
    assert "Behavior" not in part.nullable_columns


# --- drift detection -------------------------------------------------------
def test_missing_column_is_detected():
    violations = check_columns("jailbreakbench", "harmful", ["Index", "Goal", "Target"])
    assert any("missing column" in v for v in violations)


def test_unexpected_column_is_detected():
    violations = check_columns("jailbreakbench", "harmful", [*JBB_COLUMNS, "Extra"])
    assert any("unexpected column" in v for v in violations)


def test_column_reordering_is_detected():
    reordered = (JBB_COLUMNS[1], JBB_COLUMNS[0], *JBB_COLUMNS[2:])
    violations = check_columns("jailbreakbench", "harmful", list(reordered))
    assert any("ORDER changed" in v for v in violations)


def test_row_count_drift_is_detected():
    assert check_row_count("harmbench", "text_all", 400) == []
    assert any("row count changed" in v for v in check_row_count("harmbench", "text_all", 399))


def test_matching_schema_produces_no_violations():
    assert validate_part("jailbreakbench", "harmful", columns=list(JBB_COLUMNS), n_rows=100) == []
    assert_part("jailbreakbench", "harmful", columns=list(JBB_COLUMNS), n_rows=100)


def test_assert_part_raises_loudly_on_drift():
    with pytest.raises(SchemaMismatchError) as exc:
        assert_part("harmbench", "text_val", columns=list(HARMBENCH_COLUMNS), n_rows=81)
    message = str(exc.value)
    assert "SCHEMA DRIFT" in message
    assert "text_val" in message
    # The error must point at the fix, not just complain.
    assert "schemas.py" in message


def test_hf_config_drift_is_detected():
    assert check_hf_configs(
        "jailbreakbench", {"behaviors": ["harmful", "benign"], "judge_comparison": ["test"]}
    ) == []
    # The classic wrong assumption: separate harmful/benign configs.
    violations = check_hf_configs("jailbreakbench", {"harmful": ["train"], "benign": ["train"]})
    assert any("is absent" in v for v in violations)
    assert any("unexpected new HF config" in v for v in violations)


def test_hf_split_rename_is_detected():
    violations = check_hf_configs(
        "jailbreakbench", {"behaviors": ["train", "test"], "judge_comparison": ["test"]}
    )
    assert any("expected splits" in v for v in violations)


def test_unknown_source_or_part_raises():
    with pytest.raises(SchemaMismatchError, match="no pinned schema"):
        get_schema("advbench")
    with pytest.raises(SchemaMismatchError, match="has no part"):
        get_part("harmbench", "multimodal_all")
