"""The frozen dataset and its behavior-level splits.

The decisive test is :func:`test_no_behavior_id_appears_in_both_splits`. Every
other guarantee in this project — paired cross-lingual comparison, an honest
holdout, a reproducible split_hash — rests on that one not being violated.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import pytest

from multilingual_safety.languages import LANGUAGE_CODES
from multilingual_safety.paths import PROCESSED_DIR, RESULTS_DIR
from multilingual_safety.splits import (
    DEV,
    FINAL_TEST,
    HOLDOUT_SPLIT_NAMES,
    SplitAccessError,
    SplitError,
    assert_no_leakage,
    build_stratified_splits,
    load_split,
    load_splits,
    split_hash,
)

FROZEN_PATH = PROCESSED_DIR / "frozen_dataset_v1.jsonl"
SPLITS_PATH = PROCESSED_DIR / "splits_v1.json"
REPORT_PATH = RESULTS_DIR / "split_report.json"


@pytest.fixture(scope="module")
def splits() -> dict:
    if not SPLITS_PATH.exists():
        pytest.skip("splits have not been frozen in this working tree")
    return load_splits(SPLITS_PATH)


@pytest.fixture(scope="module")
def frozen() -> list[dict]:
    if not FROZEN_PATH.exists():
        pytest.skip("frozen dataset not present")
    return [
        json.loads(line)
        for line in FROZEN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- the decisive property --------------------------------------------------
def test_no_behavior_id_appears_in_both_splits(splits):
    by_split: dict[str, set[str]] = defaultdict(set)
    for behavior, split in splits["assignments"].items():
        by_split[split].add(behavior)
    assert set(by_split) == {DEV, FINAL_TEST}
    overlap = by_split[DEV] & by_split[FINAL_TEST]
    assert overlap == set(), f"split leakage: {sorted(overlap)[:5]}"
    assert len(by_split[DEV]) + len(by_split[FINAL_TEST]) == len(splits["assignments"])


def test_all_seven_language_rows_share_their_behavior_split(frozen):
    """A behavior's split must have no language dimension."""
    by_behavior: dict[str, set[str]] = defaultdict(set)
    languages: dict[str, set[str]] = defaultdict(set)
    for row in frozen:
        by_behavior[row["behavior_id"]].add(row["split"])
        languages[row["behavior_id"]].add(row["language"])
    straddling = {b: sorted(s) for b, s in by_behavior.items() if len(s) > 1}
    assert not straddling, f"behaviors straddling splits: {list(straddling)[:5]}"
    assert all(langs == set(LANGUAGE_CODES) for langs in languages.values())


# --- counts and stratification ---------------------------------------------
def test_expected_counts(splits, frozen):
    assert len(splits["assignments"]) == 200
    assert splits["counts"] == {DEV: 140, FINAL_TEST: 60}
    assert len(frozen) == 1400


def test_seventy_thirty_per_arm(frozen):
    arm = {r["behavior_id"]: r["harm_label"] for r in frozen}
    split = {r["behavior_id"]: r["split"] for r in frozen}
    counts = Counter((split[b], arm[b]) for b in arm)
    assert counts[(DEV, "harmful")] == 70
    assert counts[(DEV, "benign")] == 70
    assert counts[(FINAL_TEST, "harmful")] == 30
    assert counts[(FINAL_TEST, "benign")] == 30


def test_every_category_is_represented_in_both_splits(frozen):
    category = {r["behavior_id"]: r["category"] for r in frozen}
    split = {r["behavior_id"]: r["split"] for r in frozen}
    counts = Counter((split[b], category[b]) for b in category)
    categories = sorted(set(category.values()))
    assert len(categories) == 10
    for cat in categories:
        assert counts[(DEV, cat)] == 14, cat
        assert counts[(FINAL_TEST, cat)] == 6, cat


def test_stratification_is_recorded(splits):
    assert splits["stratified_by"] == ["harm_label", "category"]
    assert splits["n_strata"] == 20
    assert splits["holdout_per_stratum"] == 3
    assert splits["unit"] == "behavior_id"


# --- determinism and integrity ---------------------------------------------
def test_split_hash_matches_content(splits):
    assert splits["split_hash"] == split_hash(splits)


def test_salt_is_recorded(splits):
    assert splits["salt"], "the salt must be recorded so the split can be recomputed"


def test_split_is_reproducible_from_the_salt(splits, frozen):
    behaviors = [
        {"behavior_id": b, "harm_label": r["harm_label"], "category": r["category"]}
        for b, r in {row["behavior_id"]: row for row in frozen}.items()
    ]
    rebuilt = build_stratified_splits(
        behaviors, salt=splits["salt"], holdout_per_stratum=splits["holdout_per_stratum"],
        dev_name=DEV, holdout_name=FINAL_TEST,
    )
    assert rebuilt["assignments"] == splits["assignments"]
    assert rebuilt["split_hash"] == splits["split_hash"]


def test_a_different_salt_produces_a_different_split(frozen):
    behaviors = [
        {"behavior_id": b, "harm_label": r["harm_label"], "category": r["category"]}
        for b, r in {row["behavior_id"]: row for row in frozen}.items()
    ]
    a = build_stratified_splits(behaviors, salt="salt-a", holdout_per_stratum=3)
    b = build_stratified_splits(behaviors, salt="salt-b", holdout_per_stratum=3)
    assert a["assignments"] != b["assignments"]
    assert a["counts"] == b["counts"], "counts stay exact regardless of salt"


def test_edited_split_file_is_detected(tmp_path, splits):
    path = tmp_path / "splits.json"
    payload = dict(splits)
    victim = next(iter(payload["assignments"]))
    payload["assignments"] = dict(payload["assignments"])
    payload["assignments"][victim] = (
        FINAL_TEST if payload["assignments"][victim] == DEV else DEV
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SplitError, match="split hash mismatch"):
        load_splits(path)


def test_frozen_rows_carry_the_split_hash(frozen, splits):
    assert {r["split_hash"] for r in frozen} == {splits["split_hash"]}
    assert {r["frozen_dataset_version"] for r in frozen} == {"v1"}


# --- holdout access guard ---------------------------------------------------
def test_final_test_is_treated_as_a_holdout():
    assert FINAL_TEST in HOLDOUT_SPLIT_NAMES


def test_reading_the_holdout_requires_explicit_opt_in(splits):
    dev = load_split(SPLITS_PATH, DEV)
    assert len(dev) == 140
    with pytest.raises(SplitAccessError, match="refusing to read"):
        load_split(SPLITS_PATH, FINAL_TEST)
    holdout = load_split(SPLITS_PATH, FINAL_TEST, allow_test=True)
    assert len(holdout) == 60
    assert not set(dev) & set(holdout)


def test_unknown_split_name_rejected():
    with pytest.raises(SplitError, match="unknown split"):
        load_split(SPLITS_PATH, "train")


# --- leakage helper ---------------------------------------------------------
def test_assert_no_leakage_passes_on_a_clean_split(splits):
    assert_no_leakage(splits)


def test_assert_no_leakage_catches_a_planted_overlap():
    """Leakage is only expressible in the derived per-split member lists."""
    payload = {"assignments": {"beh_0123456789ab": DEV, "beh_ffffffffffff": FINAL_TEST}}
    assert_no_leakage(payload)                       # canonical form is clean

    leaky = {DEV: ["beh_0123456789ab"], FINAL_TEST: ["beh_0123456789ab"]}
    with pytest.raises(SplitError, match="split leakage"):
        assert_no_leakage(payload, members=leaky)


def test_assert_no_leakage_checks_the_real_split_members(splits):
    """The frozen split's own derived member lists must not overlap."""
    members = {
        DEV: [b for b, s in splits["assignments"].items() if s == DEV],
        FINAL_TEST: [b for b, s in splits["assignments"].items() if s == FINAL_TEST],
    }
    assert_no_leakage(splits, members=members)
    assert len(members[DEV]) == 140 and len(members[FINAL_TEST]) == 60


def test_stratified_builder_rejects_duplicates_and_overdraw():
    items = [{"behavior_id": "beh_0123456789ab", "harm_label": "harmful", "category": "X"}]
    with pytest.raises(SplitError, match="duplicate behavior_id"):
        build_stratified_splits(items * 2, salt="s", holdout_per_stratum=1)
    with pytest.raises(SplitError, match="cannot hold out"):
        build_stratified_splits(items, salt="s", holdout_per_stratum=5)
    with pytest.raises(SplitError, match="empty split"):
        build_stratified_splits([], salt="s", holdout_per_stratum=1)


# --- tracked report ---------------------------------------------------------
def test_tracked_report_is_text_free_and_consistent(splits, frozen):
    if not REPORT_PATH.exists():
        pytest.skip("split report not generated")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["contains_prompt_text"] is False
    assert report["models_run"] is False
    assert report["split_hash"] == splits["split_hash"]
    assert report["counts"] == {DEV: 140, FINAL_TEST: 60}
    assert report["assignments"] == splits["assignments"]

    blob = json.dumps(report, ensure_ascii=False)
    texts = [r["source_text"] for r in frozen] + [
        r["translated_text"] for r in frozen if r["translated_text"]
    ]
    leaked = [t for t in texts if t and t in blob]
    assert not leaked, f"{len(leaked)} prompt/translation string(s) leaked into the report"


def test_frozen_data_is_git_ignored():
    import subprocess

    from multilingual_safety.paths import REPO_ROOT

    for path in (FROZEN_PATH, SPLITS_PATH):
        if not path.exists():
            continue
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)], cwd=REPO_ROOT, check=False
        )
        assert result.returncode == 0, f"{path} must be git-ignored"
