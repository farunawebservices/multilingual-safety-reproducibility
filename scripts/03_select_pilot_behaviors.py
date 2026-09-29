#!/usr/bin/env python3
"""Stage 1.0 -- deterministically select the translation pilot behaviors.

Selects **10 harmful + 10 benign** behaviors from JailbreakBench only. HarmBench
is deliberately not used for the pilot: the two sources overlap in provenance
(Stage 0 found 27 JBB harmful behaviors sourced from TDC/HarmBench), and mixing
them is a decision for later.

Stratification is exact rather than approximate. JBB provides 10 categories with
10 behaviors each per arm, so the pilot takes **one behavior per category per
arm** -- every category is represented exactly once in each arm, and the harmful
and benign arms are matched category-for-category.

Selection is deterministic and auditable: within each (arm, category) stratum,
candidates are ranked by ``sha256(salt : behavior_id)`` and the lowest is taken.
This does not depend on file order, dict ordering, or RNG state, and anyone with
the salt can recompute it.

Behavior IDs are the repository's provenance-derived ``beh_<hex>`` -- stable
across all seven languages -- and each row also records the original JBB
``Index`` as ``source_item_id`` so upstream identity is preserved. (JBB's own
``Index`` restarts at 0 per split and collides across arms, so it cannot serve
as a corpus-wide key; Stage 0 verified this.)

Writes ``data/interim/translation_pilot_selection.jsonl`` (git-ignored: carries
source text).

Usage
-----
    python scripts/03_select_pilot_behaviors.py --dry-run
    python scripts/03_select_pilot_behaviors.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.hashing import text_sha256
from multilingual_safety.ids import behavior_id
from multilingual_safety.paths import INTERIM_DIR, RAW_DIR, display_path
from multilingual_safety.run_context import stage_report_header

JBB_DIR = RAW_DIR / "jailbreakbench"
JBB_FILES = {"harmful": "harmful-behaviors.csv", "benign": "benign-behaviors.csv"}
SELECTION_PATH = INTERIM_DIR / "translation_pilot_selection.jsonl"

#: Recorded in the output. Changing it re-selects the pilot.
SELECTION_SALT = "mls-pilot-v1"
PER_CATEGORY_PER_ARM = 1
SOURCE_ID = "jailbreakbench"

#: The behavior request text. JBB's `Goal` is the instruction given to the
#: model; `Target` is the desired affirmative prefix used by attack methods and
#: is NOT part of the request, so it is not translated.
TEXT_COLUMN = "Goal"


def rank_key(salt: str, bid: str) -> str:
    return hashlib.sha256(f"{salt}:{bid}".encode()).hexdigest()


def load_arm(arm: str) -> list[dict[str, str]]:
    path = JBB_DIR / JBB_FILES[arm]
    if not path.exists():
        raise SystemExit(
            f"missing {path}. Run scripts/01_download_sources.py first (Stage 0)."
        )
    with path.open(encoding="utf-8", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def select() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    strata_report: dict[str, Any] = {}

    for arm in sorted(JBB_FILES):
        rows = load_arm(arm)
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for row in rows:
            bid = behavior_id(
                source_id=SOURCE_ID, source_item_id=str(row["Index"]), arm=arm
            )
            by_category[row["Category"]].append(
                {
                    "behavior_id": bid,
                    "source": SOURCE_ID,
                    "source_item_id": str(row["Index"]),
                    "category": row["Category"],
                    "harm_label": arm,
                    "source_text": row[TEXT_COLUMN],
                    "source_text_sha256": text_sha256(row[TEXT_COLUMN]),
                    "upstream_source_label": row.get("Source", ""),
                    "_rank": rank_key(SELECTION_SALT, bid),
                }
            )

        strata_report[arm] = {
            "n_candidates": len(rows),
            "n_categories": len(by_category),
            "candidates_per_category": dict(sorted((c, len(v)) for c, v in by_category.items())),
        }

        # Deterministic: sort strata by name, candidates by hash rank.
        for category in sorted(by_category):
            ranked = sorted(by_category[category], key=lambda r: (r["_rank"], r["behavior_id"]))
            for chosen in ranked[:PER_CATEGORY_PER_ARM]:
                selected.append({k: v for k, v in chosen.items() if k != "_rank"})

    # Stable global ordering: arm, then category, then behavior_id.
    selected.sort(key=lambda r: (r["harm_label"], r["category"], r["behavior_id"]))
    return selected, strata_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=SELECTION_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    header = stage_report_header(
        "stage1.0_select_pilot",
        {"salt": SELECTION_SALT, "per_category_per_arm": PER_CATEGORY_PER_ARM,
         "source": SOURCE_ID, "text_column": TEXT_COLUMN},
    )
    print(f"run_id:     {header['run_id']}")
    print(f"git commit: {header['git_commit']}  (dirty={header['git_dirty']})")
    print(f"salt:       {SELECTION_SALT}")
    print(f"source:     {SOURCE_ID} only (HarmBench deliberately excluded from the pilot)")

    selected, strata = select()

    harmful = [r for r in selected if r["harm_label"] == "harmful"]
    benign = [r for r in selected if r["harm_label"] == "benign"]
    ids = [r["behavior_id"] for r in selected]

    print("\nSELECTION")
    for arm, stats in sorted(strata.items()):
        print(f"  {arm:8} candidates={stats['n_candidates']} categories={stats['n_categories']}")
    print(f"  selected harmful: {len(harmful)}")
    print(f"  selected benign:  {len(benign)}")
    print(f"  distinct behavior_ids: {len(set(ids))}")

    print("\n  per-category (harmful / benign):")
    h_cat = Counter(r["category"] for r in harmful)
    b_cat = Counter(r["category"] for r in benign)
    for category in sorted(set(h_cat) | set(b_cat)):
        print(f"    {category:32} {h_cat[category]} / {b_cat[category]}")

    # --- stop conditions --------------------------------------------------
    failures: list[str] = []
    if len(harmful) != 10:
        failures.append(f"expected 10 harmful, got {len(harmful)}")
    if len(benign) != 10:
        failures.append(f"expected 10 benign, got {len(benign)}")
    if len(set(ids)) != len(ids):
        failures.append("duplicate behavior_id in the selection")
    if set(h_cat) != set(b_cat):
        failures.append("harmful and benign arms cover different categories")
    if any(n != PER_CATEGORY_PER_ARM for n in list(h_cat.values()) + list(b_cat.values())):
        failures.append("stratification is uneven across categories")

    if failures:
        for f in failures:
            print(f"  FAIL: {f}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(selected)} rows -> {display_path(args.out)} (git-ignored)")
    print("Selection is deterministic: rerunning with the same salt reproduces it exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
