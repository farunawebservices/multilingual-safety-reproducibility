#!/usr/bin/env python3
"""Stage 3.0 -- freeze the multilingual dataset and assign behavior-level splits.

This is the least reversible step in the project. Once a holdout is read, a
re-split invalidates every result that referenced the old ``split_hash``
(CLAUDE.md §1). So the script verifies integrity *first*, refuses to overwrite an
existing frozen split without an explicit flag, and records the salt so anyone
can recompute the assignment.

Splits are assigned to **behaviors**, never to rows. All seven language variants
of a behavior therefore share its split by construction -- the assignment simply
has no language dimension. Splitting per row would put the English rendering in
dev and the Hausa rendering in the holdout, leaking the item and destroying the
paired cross-lingual design.

Stratification is (arm x category): 20 strata of 10 behaviors each, 3 held out
per stratum. That yields exactly 70 harmful + 70 benign in dev and 30 + 30 in
final_test, with every category represented in both splits in identical
proportion -- so a per-category effect can never be an artifact of one category
landing entirely on one side.

Integrity gates (all abort before anything is written)
------------------------------------------------------
duplicate behavior IDs, split leakage, missing languages, source-hash drift
against the acquisition manifest, and inconsistent behavior-level metadata.

Outputs (v1 -- the defaults, unchanged)
---------------------------------------
    data/processed/frozen_dataset_v1.jsonl   frozen rows + split (git-ignored)
    data/processed/splits_v1.json            assignments + salt (git-ignored)
    results/split_report.json                metadata-only, TEXT-FREE (tracked)

Dataset v2
----------
v2 is detected as soon as any of ``--outputs``, ``--splits-output`` or
``--report`` points inside ``data/v2/``. In v2 the translated artifact is
**already** the frozen dataset (``data/v2/processed/frozen_dataset_v2.jsonl``,
stage ``translated_applied_pre_split``), so this stage does **not** rewrite it:
it is the input, it is opened read-only, and its SHA-256 is checked before and
after. The v2 run therefore writes two files, not three:

    data/v2/processed/splits_v2.json          assignments + salt
    data/v2/manifests/split_report_v2.json    metadata-only, TEXT-FREE

A v2 run refuses the v1 salt, refuses to write anywhere under
``data/processed/`` or ``results/``, and requires every path to sit in the v2
tree -- all three checks run before a single byte is written. The consequence of
not rewriting the input is that the v2 split lives only in ``splits_v2.json``;
the frozen rows carry no ``split`` column and must be joined on ``behavior_id``.

No model is run. No translated prompt text reaches a tracked artifact.

Usage
-----
    python scripts/15_freeze_splits.py --dry-run
    python scripts/15_freeze_splits.py

    python scripts/15_freeze_splits.py --dry-run \\
        --outputs data/v2/processed/frozen_dataset_v2.jsonl \\
        --splits-output data/v2/processed/splits_v2.json \\
        --report data/v2/manifests/split_report_v2.json \\
        --salt mls-v2-splits-20260818
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.hashing import deterministic_digest, file_sha256, text_sha256
from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE
from multilingual_safety.paths import (
    INTERIM_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    V2_DIR,
    V2_MANIFESTS_DIR,
    V2_PROCESSED_DIR,
    display_path,
)
from multilingual_safety.run_context import stage_report_header
from multilingual_safety.splits import (
    DEV,
    FINAL_TEST,
    SplitError,
    assert_no_leakage,
    build_stratified_splits,
)
from multilingual_safety.translation import TranslationRow

FULL_DIR = INTERIM_DIR / "full"
OUTPUTS_PATH = FULL_DIR / "full_translation_outputs.jsonl"
SELECTION_PATH = FULL_DIR / "full_selection.jsonl"

FROZEN_PATH = PROCESSED_DIR / "frozen_dataset_v1.jsonl"
SPLITS_PATH = PROCESSED_DIR / "splits_v1.json"
REPORT_PATH = RESULTS_DIR / "split_report.json"

#: Dataset v2. The input is the pre-split translated artifact and is READ-ONLY
#: here; only these two paths are written.
V2_INPUT_PATH = V2_PROCESSED_DIR / "frozen_dataset_v2.jsonl"
V2_SPLITS_PATH = V2_PROCESSED_DIR / "splits_v2.json"
V2_REPORT_PATH = V2_MANIFESTS_DIR / "split_report_v2.json"

#: Recorded in the split file. Changing it re-splits and invalidates prior
#: holdout results. Never edit without a recorded reason.
SPLIT_SALT = "mls-splits-v1"

#: configs/dataset_v2_rebuild.yaml, `split.salt`. v2 MUST NOT reuse the v1 salt:
#: the behavior IDs are identical, so the v1 salt would reproduce v1's
#: assignment exactly and yield v1's split_hash, letting v2 results be compared
#: against a corpus that can no longer be re-verified.
V2_SPLIT_SALT = "mls-v2-splits-20260818"

HOLDOUT_PER_STRATUM = 3          # 3 of 10 per (arm, category) => 30 per arm

EXPECTED_BEHAVIORS = 200
EXPECTED_ROWS = EXPECTED_BEHAVIORS * len(LANGUAGE_CODES)
EXPECTED_DEV_PER_ARM = 70
EXPECTED_HOLDOUT_PER_ARM = 30
CONSISTENCY_FIELDS = ("source", "source_item_id", "category", "harm_label")


def _under(path: Path, root: Path) -> bool:
    """Lexical containment on RESOLVED paths, so a relative CLI arg still works."""
    return path.expanduser().resolve().is_relative_to(root.resolve())


def detect_dataset_version(outputs: Path, splits_output: Path, report: Path) -> str:
    """``"v2"`` as soon as ANY of the three paths points inside ``data/v2/``.

    Deliberately triggered by *any* of them rather than all three: a run that
    reads the v2 corpus but still carries a default v1 destination is exactly the
    mistake worth catching, and it is caught by returning "v2" here and then
    failing the path gate below -- not by quietly writing a v1 artifact.
    """
    return "v2" if any(_under(p, V2_DIR) for p in (outputs, splits_output, report)) else "v1"


def v2_path_problems(outputs: Path, splits_output: Path, report: Path, salt: str) -> list[str]:
    """Everything wrong with a v2 invocation, checked before anything is written."""
    problems: list[str] = []

    if salt == SPLIT_SALT:
        problems.append(
            f"salt {salt!r} is v1's. Reusing it would reproduce v1's assignment and "
            f"v1's split_hash exactly, because the behavior IDs are identical. Pass "
            f"--salt {V2_SPLIT_SALT} (configs/dataset_v2_rebuild.yaml, split.salt)."
        )

    for label, path in (("--splits-output", splits_output), ("--report", report)):
        if _under(path, PROCESSED_DIR):
            problems.append(
                f"{label} {display_path(path)} is under data/processed/, which holds v1's "
                "frozen split. A v2 run may not write there."
            )
        if _under(path, RESULTS_DIR):
            problems.append(
                f"{label} {display_path(path)} is under results/, which holds v1's tracked "
                "split report. A v2 run may not write there."
            )

    if not _under(outputs, V2_DIR):
        problems.append(
            f"--outputs {display_path(outputs)} is outside data/v2/ but this run was "
            "detected as v2. Mixing corpora across dataset versions is never intended."
        )
    if not _under(splits_output, V2_PROCESSED_DIR):
        problems.append(
            f"--splits-output must be under {display_path(V2_PROCESSED_DIR)}/ for a v2 run; "
            f"got {display_path(splits_output)}. Expected "
            f"--splits-output {display_path(V2_SPLITS_PATH)}"
        )
    if not _under(report, V2_MANIFESTS_DIR):
        problems.append(
            f"--report must be under {display_path(V2_MANIFESTS_DIR)}/ for a v2 run; "
            f"got {display_path(report)}. Expected "
            f"--report {display_path(V2_REPORT_PATH)}"
        )

    for label, path in (("--splits-output", splits_output), ("--report", report)):
        if path.expanduser().resolve() == outputs.expanduser().resolve():
            problems.append(
                f"{label} {display_path(path)} is the input artifact. The v2 translated "
                "corpus is read-only at this stage and is never rewritten."
            )
    return problems


def load_rows(path: Path) -> list[TranslationRow]:
    rows: list[TranslationRow] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(TranslationRow.from_dict(json.loads(line)))
        except Exception as exc:
            raise SystemExit(f"{path.name}:{lineno}: cannot parse row: {exc}") from exc
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--outputs", type=Path, default=OUTPUTS_PATH)
    parser.add_argument("--salt", default=SPLIT_SALT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--refreeze", action="store_true",
        help="Overwrite an existing frozen split. Invalidates every result that "
             "referenced the old split_hash.",
    )
    # Destinations. Defaults are the v1 paths, so an unflagged invocation is
    # unchanged. They are overridable because dataset v2 writes under
    # data/v2/processed/ and data/v2/manifests/ and must never touch v1's.
    parser.add_argument(
        "--splits-output", type=Path, default=SPLITS_PATH,
        help="Split assignments + salt (default: %(default)s).",
    )
    parser.add_argument(
        "--report", type=Path, default=REPORT_PATH,
        help="Text-free metadata report (default: %(default)s).",
    )
    args = parser.parse_args()

    if not args.outputs.exists():
        print(f"missing {args.outputs}", file=sys.stderr)
        return 1

    # --- dataset version and path safety, before anything is written -------
    dataset_version = detect_dataset_version(args.outputs, args.splits_output, args.report)
    is_v2 = dataset_version == "v2"
    if is_v2:
        problems = v2_path_problems(
            args.outputs, args.splits_output, args.report, args.salt
        )
        if problems:
            print(f"STOP: this run is dataset {dataset_version} and "
                  f"{len(problems)} path/salt rule(s) refuse it:", file=sys.stderr)
            for problem in problems:
                print(f"  FAIL: {problem}", file=sys.stderr)
            print("\nNothing written.", file=sys.stderr)
            return 1

    if args.splits_output.exists() and not args.refreeze and not args.dry_run:
        print(f"STOP: {display_path(args.splits_output)} already exists. The split is "
              "frozen; re-splitting invalidates prior holdout results (CLAUDE.md §1). "
              "Pass --refreeze deliberately.", file=sys.stderr)
        return 1

    header = stage_report_header(
        "stage3.0_freeze_splits",
        {"salt": args.salt, "holdout_per_stratum": HOLDOUT_PER_STRATUM,
         "splits": [DEV, FINAL_TEST]},
    )
    print(f"run_id:     {header['run_id']}")
    print(f"git commit: {header['git_commit']}  (dirty={header['git_dirty']})")
    print(f"salt:       {args.salt}")

    print(f"\nPATHS  (dataset {dataset_version})")
    print(f"  input   : {display_path(args.outputs)}"
          f"{'  (READ-ONLY, never rewritten)' if is_v2 else ''}")
    if is_v2:
        print("  frozen  : (not written; the input IS the frozen dataset)")
    else:
        print(f"  frozen  : {display_path(FROZEN_PATH)}")
    print(f"  splits  : {display_path(args.splits_output)}")
    print(f"  report  : {display_path(args.report)}")
    if is_v2 and args.salt != V2_SPLIT_SALT:
        print(f"  WARNING: salt {args.salt!r} is not the pre-registered v2 salt "
              f"{V2_SPLIT_SALT!r} (configs/dataset_v2_rebuild.yaml, split.salt). "
              "The split_hash will not be the one the protocol describes.")

    # The v2 input is hashed before the run so "read-only" is a checked claim.
    input_sha_before = file_sha256(args.outputs)

    rows = load_rows(args.outputs)
    groups: dict[str, list[TranslationRow]] = defaultdict(list)
    for row in rows:
        groups[row.behavior_id].append(row)

    failures: list[str] = []

    # --- 1. duplicates and language completeness --------------------------
    missing_languages = {
        bid: sorted(set(LANGUAGE_CODES) - {r.language for r in group})
        for bid, group in sorted(groups.items())
        if {r.language for r in group} != set(LANGUAGE_CODES)
    }
    duplicate_language_rows = {
        bid: sorted(lang for lang, n in Counter(r.language for r in group).items() if n > 1)
        for bid, group in sorted(groups.items())
        if any(n > 1 for n in Counter(r.language for r in group).values())
    }

    # --- 2. behavior-level metadata consistency ---------------------------
    inconsistent: dict[str, list[str]] = {}
    for bid, group in sorted(groups.items()):
        bad = [
            field for field in CONSISTENCY_FIELDS
            if len({str(getattr(r, field)) for r in group}) > 1
        ]
        if bad:
            inconsistent[bid] = bad

    # --- 3. source-hash drift against the acquisition manifest ------------
    drift: list[str] = []
    if SELECTION_PATH.exists():
        recorded = {
            item["behavior_id"]: item["source_text_sha256"]
            for item in (
                json.loads(line)
                for line in SELECTION_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        for bid, group in sorted(groups.items()):
            english = next((r for r in group if r.language == PIVOT_LANGUAGE), None)
            if english is None:
                continue
            actual = text_sha256(english.source_text)
            if bid in recorded and recorded[bid] != actual:
                drift.append(bid)
        orphans = sorted(set(groups) - set(recorded))
        if orphans:
            failures.append(f"{len(orphans)} behavior(s) absent from the acquisition manifest")

    print("\nINTEGRITY")
    print(f"  behaviors                  : {len(groups)} (expected {EXPECTED_BEHAVIORS})")
    print(f"  rows                       : {len(rows)} (expected {EXPECTED_ROWS})")
    print(f"  behaviors missing a language: {len(missing_languages)}")
    print(f"  duplicate language rows    : {len(duplicate_language_rows)}")
    print(f"  inconsistent metadata      : {len(inconsistent)}")
    print(f"  source-hash drift          : {len(drift)}")

    if len(groups) != EXPECTED_BEHAVIORS:
        failures.append(f"expected {EXPECTED_BEHAVIORS} behaviors, got {len(groups)}")
    if len(rows) != EXPECTED_ROWS:
        failures.append(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")
    if missing_languages:
        failures.append(f"{len(missing_languages)} behavior(s) missing a language")
    if duplicate_language_rows:
        failures.append(f"{len(duplicate_language_rows)} behavior(s) with duplicate language rows")
    if inconsistent:
        failures.append(f"{len(inconsistent)} behavior(s) with inconsistent metadata")
    if drift:
        failures.append(f"{len(drift)} behavior(s) with source-hash drift: {drift[:5]}")

    for failure in failures:
        print(f"  FAIL: {failure}", file=sys.stderr)
    if failures:
        print("\nNothing written.", file=sys.stderr)
        return 1

    # --- 4. build the stratified split ------------------------------------
    behaviors = [
        {
            "behavior_id": bid,
            "harm_label": group[0].harm_label,
            "category": group[0].category,
        }
        for bid, group in sorted(groups.items())
    ]
    try:
        payload = build_stratified_splits(
            behaviors, salt=args.salt, holdout_per_stratum=HOLDOUT_PER_STRATUM,
            dev_name=DEV, holdout_name=FINAL_TEST,
        )
        assert_no_leakage(payload)
    except SplitError as exc:
        print(f"  FAIL: {exc}", file=sys.stderr)
        return 1

    assignments = payload["assignments"]
    arm_of = {b["behavior_id"]: b["harm_label"] for b in behaviors}
    category_of = {b["behavior_id"]: b["category"] for b in behaviors}
    per_split_arm = Counter((assignments[b], arm_of[b]) for b in assignments)
    per_split_cat = Counter((assignments[b], category_of[b]) for b in assignments)

    print("\nSPLIT")
    print(f"  strata            : {payload['n_strata']} (arm x category)")
    print(f"  holdout/stratum   : {HOLDOUT_PER_STRATUM}")
    print(f"  counts            : {payload['counts']}")
    print(f"  split_hash        : {payload['split_hash']}")
    print("  by split x arm    :")
    for split in (DEV, FINAL_TEST):
        print(f"    {split:<11} harmful={per_split_arm[(split, 'harmful')]:>3}  "
              f"benign={per_split_arm[(split, 'benign')]:>3}")
    print("  by split x category:")
    for category in sorted({c for c in category_of.values()}):
        print(f"    {category:<30} dev={per_split_cat[(DEV, category)]:>3}  "
              f"final_test={per_split_cat[(FINAL_TEST, category)]:>3}")

    for split, expected in ((DEV, EXPECTED_DEV_PER_ARM), (FINAL_TEST, EXPECTED_HOLDOUT_PER_ARM)):
        for arm in ("harmful", "benign"):
            if per_split_arm[(split, arm)] != expected:
                failures.append(
                    f"{split}/{arm}: expected {expected}, got {per_split_arm[(split, arm)]}"
                )
    dev_ids = {b for b, s in assignments.items() if s == DEV}
    holdout_ids = {b for b, s in assignments.items() if s == FINAL_TEST}
    if dev_ids & holdout_ids:
        failures.append(f"split leakage: {len(dev_ids & holdout_ids)} behavior(s) in both")
    if len(assignments) != EXPECTED_BEHAVIORS:
        failures.append(f"expected {EXPECTED_BEHAVIORS} assignments, got {len(assignments)}")

    for failure in failures:
        print(f"  FAIL: {failure}", file=sys.stderr)
    if failures:
        print("\nNothing written.", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\nDry run: nothing written. Split NOT frozen.")
        return 0

    # --- 5. write ----------------------------------------------------------
    # v1 materialises the split into a new frozen dataset. v2's input already IS
    # the frozen dataset, so it is left exactly as found and only the split
    # assignments are written.
    if is_v2:
        frozen_rows = len(rows)
    else:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        frozen_rows = 0
        with FROZEN_PATH.open("w", encoding="utf-8") as fh:
            for row in sorted(rows, key=lambda r: (r.behavior_id, r.language)):
                record = row.to_dict()
                record["split"] = assignments[row.behavior_id]
                record["split_hash"] = payload["split_hash"]
                record["frozen_dataset_version"] = "v1"
                fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                frozen_rows += 1
    args.splits_output.parent.mkdir(parents=True, exist_ok=True)
    args.splits_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Tracked report: IDs, counts and hashes only. No prompt or translated text.
    report = {
        **header,
        "frozen_dataset_version": dataset_version,
        "salt": args.salt,
        "split_hash": payload["split_hash"],
        "unit": "behavior_id",
        "stratified_by": payload["stratified_by"],
        "holdout_per_stratum": HOLDOUT_PER_STRATUM,
        "split_names": {"dev": DEV, "holdout": FINAL_TEST},
        "n_behaviors": len(assignments),
        "n_rows": frozen_rows,
        "n_languages": len(LANGUAGE_CODES),
        "counts": payload["counts"],
        "counts_by_split_and_arm": {
            f"{split}:{arm}": per_split_arm[(split, arm)]
            for split in (DEV, FINAL_TEST) for arm in ("harmful", "benign")
        },
        "counts_by_split_and_category": {
            f"{split}:{category}": per_split_cat[(split, category)]
            for split in (DEV, FINAL_TEST) for category in sorted(set(category_of.values()))
        },
        "strata": payload["strata"],
        "n_strata": payload["n_strata"],
        # Behavior IDs are metadata, not content: safe to track, and they are
        # what makes the split independently auditable.
        "assignments": assignments,
        "leakage": {"dev_and_holdout_overlap": 0, "checked": True},
        "integrity": {
            "duplicate_behavior_ids": 0,
            "behaviors_missing_a_language": len(missing_languages),
            "duplicate_language_rows": len(duplicate_language_rows),
            "inconsistent_metadata": len(inconsistent),
            "source_hash_drift": len(drift),
        },
        "models_run": False,
        "contains_prompt_text": False,
        "source_hashes": (
            {
                "outputs_jsonl": input_sha_before,
                "splits_json": file_sha256(args.splits_output),
            }
            if is_v2 else
            {
                "outputs_jsonl": file_sha256(args.outputs),
                "frozen_dataset_jsonl": file_sha256(FROZEN_PATH),
                "splits_json": file_sha256(args.splits_output),
            }
        ),
        "output_paths": (
            {
                "splits": display_path(args.splits_output),
                "report": display_path(args.report),
            }
            if is_v2 else
            {
                "frozen_dataset": display_path(FROZEN_PATH),
                "splits": display_path(args.splits_output),
            }
        ),
        "notes": [
            "Splits are assigned to behaviors, so all seven language variants share "
            "a split by construction.",
            "final_test is a HOLDOUT: reading it requires allow_test=True "
            "(multilingual_safety.splits.load_split).",
            "Changing the salt re-splits the corpus and invalidates every result that "
            "referenced the old split_hash.",
            "No prompt or translated text appears in this report.",
        ],
    }
    if is_v2:
        # "The input was not touched" is checked, not promised.
        input_sha_after = file_sha256(args.outputs)
        report["input_artifact"] = {
            "path": display_path(args.outputs),
            "sha256_before": input_sha_before,
            "sha256_after": input_sha_after,
            "unchanged": input_sha_after == input_sha_before,
            "n_rows": len(rows),
            "opened": "read-only",
        }
        report["frozen_dataset_written"] = False
        report["frozen_dataset_note"] = (
            "The v2 translated corpus is already the frozen artifact "
            "(translated_applied_pre_split) and is not rewritten by this stage. The "
            "split therefore lives only in the splits file: the rows carry no `split` "
            "column and must be joined on behavior_id."
        )
        report["salt_matches_preregistered_v2_salt"] = args.salt == V2_SPLIT_SALT
        report["preregistered_v2_salt"] = V2_SPLIT_SALT
        report["v1_artifacts_written"] = False
        if not report["input_artifact"]["unchanged"]:
            print("  FAIL: the v2 input artifact changed during the run", file=sys.stderr)
            return 1

    report["deterministic_digest"] = deterministic_digest(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if is_v2:
        print(f"\ninput unchanged: {display_path(args.outputs)}  "
              f"sha256={input_sha_before[:16]}  (not rewritten)")
    else:
        print(f"\nwrote {display_path(FROZEN_PATH)}  ({frozen_rows} rows, git-ignored)")
    print(f"wrote {display_path(args.splits_output)}")
    print(f"wrote {display_path(args.report)}  (metadata-only)")
    print(f"\nSPLIT FROZEN. split_hash={payload['split_hash'][:24]}")
    print("Record this hash: every run must reference it, and re-splitting invalidates "
          "prior holdout results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
