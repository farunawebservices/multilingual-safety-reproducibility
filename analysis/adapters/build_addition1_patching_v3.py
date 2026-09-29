#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SRC = Path(
    "addition_1_causal_patching/llama3_70b_hausa_flip_cases/"
    "patching_results_v3/patching_results.jsonl"
)
OUT = Path("data/derived/canonical/addition1_llama33_hausa_patching_v3.parquet")
EXPECTED_SHA256 = "a9754c08f2a6a7e0aad62c707f192b83d0ea55c6e7cafdaa61b8415a1cd73358"

TEXT_FIELDS = (
    "planner_clean", "planner_corr", "planner_patch",
    "critic_clean", "critic_corr", "critic_patch",
    "tool_clean", "tool_corr", "tool_patch",
)

CONTROL_FIELDS = (
    "control_position_decision",
    "control_role_decision",
    "control_unrelated_decision",
)

JSON_FIELDS = (
    "metadata",
    "control_random_direction_decisions",
    "control_random_layer_decisions",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    observed_sha256 = sha256(SRC)
    if observed_sha256 != EXPECTED_SHA256:
        raise ValueError(
            f"Source SHA-256 mismatch: expected {EXPECTED_SHA256}, "
            f"observed {observed_sha256}"
        )

    raw_rows = [
        json.loads(line)
        for line in SRC.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(raw_rows) != 11:
        raise ValueError(f"Expected 11 nonempty JSONL rows, found {len(raw_rows)}")

    rows = []
    for source_row_index, raw in enumerate(raw_rows):
        clean = raw["clean_decision"]
        corrupted = raw["corrupted_decision"]
        patched = raw["patched_decision"]
        evaluable = clean != corrupted

        row = {
            "source_path": str(SRC),
            "source_sha256": observed_sha256,
            "source_row_index": source_row_index,
            "base_id": raw["base_id"],
            "language": raw["language"],
            "harmfulness": raw["harmfulness"],
            "metadata_json": json_text(raw["metadata"]),
            "clean_decision": clean,
            "corrupted_decision": corrupted,
            "patched_decision": patched,
            "evaluable_clean_ne_corrupted": evaluable,
            "recovered_patched_eq_clean_among_evaluable": (
                patched == clean if evaluable else None
            ),
        }

        for field in TEXT_FIELDS:
            row[field] = raw.get(field)

        for field in CONTROL_FIELDS:
            row[field] = raw.get(field)

        for field in JSON_FIELDS[1:]:
            row[f"{field}_json"] = json_text(raw.get(field))

        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame["base_id"].isna().any() or frame["base_id"].duplicated().any():
        raise ValueError("base_id must be present and unique")
    if set(frame["clean_decision"]) - {"ALLOW", "VETO"}:
        raise ValueError("Unexpected clean_decision value")
    if set(frame["corrupted_decision"]) - {"ALLOW", "VETO"}:
        raise ValueError("Unexpected corrupted_decision value")
    if set(frame["patched_decision"]) - {"ALLOW", "VETO"}:
        raise ValueError("Unexpected patched_decision value")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)

    check = pd.read_parquet(OUT)
    if len(check) != 11 or list(check.columns) != list(frame.columns):
        raise ValueError("Parquet round-trip validation failed")

    evaluable_count = int(check["evaluable_clean_ne_corrupted"].sum())
    recovered_count = int(
        check.loc[
            check["evaluable_clean_ne_corrupted"],
            "recovered_patched_eq_clean_among_evaluable",
        ].sum()
    )

    print(f"Wrote {OUT}")
    print(f"rows={len(check)}")
    print(f"source_sha256={observed_sha256}")
    print(f"evaluable_rows={evaluable_count}")
    print(f"recovered_rows={recovered_count}")


if __name__ == "__main__":
    main()
