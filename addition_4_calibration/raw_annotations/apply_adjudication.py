#!/usr/bin/env python3
"""
Apply the blinded adjudicator's decision to Addition 4 disagreement cases.

This script does not change the raw annotation CSVs or the two-rater kappa
results. It produces a final consensus file for downstream auditing/reporting.
"""

import csv
import json
from pathlib import Path

BASE = Path("addition_4_calibration")

MERGED_PATH = BASE / "merged_annotations.jsonl"
ADJUDICATOR_PATH = BASE / "raw_annotations" / "adjudicator_completed.csv"
FINAL_PATH = BASE / "final_consensus_annotations.jsonl"

VALID_LABELS = {"YES", "NO"}
ADJUDICATOR_LABEL_COLUMN = "adjudicator_label_YES_or_NO"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    if not MERGED_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MERGED_PATH}. Run merge_and_analyze.py first."
        )

    if not ADJUDICATOR_PATH.exists():
        raise FileNotFoundError(f"Missing {ADJUDICATOR_PATH}.")

    merged = read_jsonl(MERGED_PATH)
    disagreements = [row for row in merged if not row["agreement"]]
    disagreement_ids = {row["annotation_id"] for row in disagreements}

    with open(
        ADJUDICATOR_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        if "annotation_id" not in fieldnames:
            raise ValueError("Adjudicator CSV has no annotation_id column.")

        if ADJUDICATOR_LABEL_COLUMN not in fieldnames:
            raise ValueError(
                "Adjudicator CSV has no "
                f"{ADJUDICATOR_LABEL_COLUMN!r} column. "
                f"Found: {fieldnames}"
            )

        adjudications = {}

        for row_number, row in enumerate(reader, start=2):
            annotation_id = (row.get("annotation_id") or "").strip()
            label = (row.get(ADJUDICATOR_LABEL_COLUMN) or "").strip().upper()

            if not annotation_id:
                raise ValueError(
                    f"Blank annotation ID at adjudicator CSV row {row_number}."
                )

            if label not in VALID_LABELS:
                raise ValueError(
                    f"Invalid adjudicator label at row {row_number}: "
                    f"{label!r}. Expected YES or NO."
                )

            if annotation_id in adjudications:
                raise ValueError(
                    f"Duplicate adjudicator decision for {annotation_id}."
                )

            adjudications[annotation_id] = label

    adjudication_ids = set(adjudications)

    missing = disagreement_ids - adjudication_ids
    unexpected = adjudication_ids - disagreement_ids

    if missing:
        raise ValueError(
            f"Missing adjudications for disagreement IDs: {sorted(missing)}"
        )

    if unexpected:
        raise ValueError(
            f"Adjudicator file includes IDs that were not disagreements: "
            f"{sorted(unexpected)}"
        )

    final_rows = []

    for row in merged:
        final_row = dict(row)

        if row["agreement"]:
            final_row["final_consensus_label"] = row["annotator_1"]
            final_row["consensus_source"] = "annotator_agreement"
        else:
            final_row["adjudicator_label"] = adjudications[row["annotation_id"]]
            final_row["final_consensus_label"] = adjudications[row["annotation_id"]]
            final_row["consensus_source"] = "blinded_adjudication"

        final_rows.append(final_row)

    with open(FINAL_PATH, "w", encoding="utf-8") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Addition 4 adjudication successfully applied.")
    print(f"Total rows: {len(final_rows)}")
    print(f"Initial disagreements: {len(disagreements)}")
    print(f"Adjudicated cases: {len(adjudications)}")
    print(f"Final consensus file: {FINAL_PATH}")

    for annotation_id, label in sorted(adjudications.items()):
        print(f"  {annotation_id}: final consensus = {label}")


if __name__ == "__main__":
    main()
