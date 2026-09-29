#!/usr/bin/env python3
"""
Addition 4: Judge Calibration Study

This script:
1. Reads the original 98 sampled prompts.
2. Reads two independently completed annotation CSV files.
3. Validates IDs and YES/NO labels.
4. Merges annotations by annotation_id.
5. Computes observed agreement and Cohen's kappa per language and overall.
6. Creates a blinded adjudication CSV containing only disagreements.

Raw annotation CSV files are never modified.
"""

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path("addition_4_calibration")

ORIGINAL_PATH = BASE / "annotation_samples.jsonl"
ANNOTATOR_1_PATH = BASE / "raw_annotations" / "annotator_1_completed.csv"
ANNOTATOR_2_PATH = BASE / "raw_annotations" / "annotator_2_completed.csv"

ANNOTATOR_1_COLUMN = "Annotator_1_YES_or_NO"
ANNOTATOR_2_COLUMN = "Annotator_2_YES_or_NO"

VALID_LABELS = {"YES", "NO"}

LANGUAGE_NAMES = {
    "ar": "Arabic",
    "en": "English",
    "ha": "Hausa",
    "hi": "Hindi",
    "ko": "Korean",
    "sw": "Swahili",
    "yo": "Yoruba",
}


def read_jsonl(path):
    """Return non-empty JSONL records from path."""
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_annotation_csv(path, label_column):
    """
    Read annotation IDs and normalize labels from the exported Google Sheets CSV.

    In this export, YES values are in the named label column and NO values
    are in the accidental blank-header column. Exactly one of those columns
    must contain a valid YES/NO label on every row.
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if "annotation_id" not in fieldnames:
        raise ValueError(
            f"Missing annotation_id column in {path}. Available columns: {fieldnames}"
        )

    if label_column not in fieldnames:
        raise ValueError(
            f"Missing label column {label_column!r} in {path}. "
            f"Available columns: {fieldnames}"
        )

    if "" not in fieldnames:
        raise ValueError(
            f"Expected blank-header companion column in {path}, but did not find one. "
            f"Available columns: {fieldnames}"
        )

    labels = {}
    duplicates = []

    for row_number, row in enumerate(rows, start=2):
        annotation_id = (row.get("annotation_id") or "").strip()

        named_value = (row.get(label_column) or "").strip().upper()
        blank_column_value = (row.get("") or "").strip().upper()

        entered = [
            value
            for value in [named_value, blank_column_value]
            if value
        ]

        if not annotation_id:
            raise ValueError(
                f"Blank annotation_id in {path} at CSV row {row_number}."
            )

        if len(entered) != 1:
            raise ValueError(
                f"{path}, row {row_number}, ID {annotation_id}: expected exactly "
                f"one label across {label_column!r} and the blank-header column; "
                f"found {entered!r}."
            )

        label = entered[0]

        if label not in VALID_LABELS:
            raise ValueError(
                f"{path}, row {row_number}, ID {annotation_id}: label must be "
                f"YES or NO, found {label!r}."
            )

        if annotation_id in labels:
            duplicates.append(annotation_id)

        labels[annotation_id] = label

    if duplicates:
        duplicate_preview = sorted(set(duplicates))[:10]
        raise ValueError(
            f"Duplicate annotation IDs in {path}: {duplicate_preview}"
        )

    return labels, len(rows)


def compute_kappa(labels_1, labels_2):
    """Compute observed agreement, expected agreement, and Cohen's kappa."""
    if len(labels_1) != len(labels_2):
        raise ValueError("Rater lists have different lengths.")

    if not labels_1:
        raise ValueError("Cannot calculate kappa on zero rows.")

    n = len(labels_1)
    pair_counts = Counter(zip(labels_1, labels_2))

    observed_agreement = sum(
        label_1 == label_2
        for label_1, label_2 in zip(labels_1, labels_2)
    ) / n

    a1_yes = labels_1.count("YES")
    a1_no = labels_1.count("NO")
    a2_yes = labels_2.count("YES")
    a2_no = labels_2.count("NO")

    expected_agreement = (
        (a1_yes / n) * (a2_yes / n)
        + (a1_no / n) * (a2_no / n)
    )

    if math.isclose(expected_agreement, 1.0):
        kappa = 1.0 if math.isclose(observed_agreement, 1.0) else None
    else:
        kappa = (
            observed_agreement - expected_agreement
        ) / (
            1.0 - expected_agreement
        )

    return {
        "n": n,
        "observed_agreement": observed_agreement,
        "expected_agreement": expected_agreement,
        "kappa": kappa,
        "annotator_1_yes": a1_yes,
        "annotator_1_no": a1_no,
        "annotator_2_yes": a2_yes,
        "annotator_2_no": a2_no,
        "contingency": {
            "a1_yes_a2_yes": pair_counts[("YES", "YES")],
            "a1_yes_a2_no": pair_counts[("YES", "NO")],
            "a1_no_a2_yes": pair_counts[("NO", "YES")],
            "a1_no_a2_no": pair_counts[("NO", "NO")],
        },
    }


def bootstrap_kappa_ci(labels_1, labels_2, seed, n_bootstrap=10000):
    """Compute a percentile bootstrap 95% confidence interval for kappa."""
    n = len(labels_1)

    if n < 2:
        return None, None

    rng = random.Random(seed)
    bootstrap_values = []

    for _ in range(n_bootstrap):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_1 = [labels_1[index] for index in indices]
        sample_2 = [labels_2[index] for index in indices]

        kappa = compute_kappa(sample_1, sample_2)["kappa"]

        if kappa is not None and math.isfinite(kappa):
            bootstrap_values.append(kappa)

    if not bootstrap_values:
        return None, None

    bootstrap_values.sort()

    lower_index = int(0.025 * (len(bootstrap_values) - 1))
    upper_index = int(0.975 * (len(bootstrap_values) - 1))

    return (
        bootstrap_values[lower_index],
        bootstrap_values[upper_index],
    )


def interpret_kappa(kappa):
    """Return a conventional descriptive interpretation of Cohen's kappa."""
    if kappa is None:
        return "Undefined"
    if kappa < 0.0:
        return "Less than chance"
    if kappa < 0.20:
        return "Slight"
    if kappa < 0.40:
        return "Fair"
    if kappa < 0.60:
        return "Moderate"
    if kappa < 0.80:
        return "Substantial"
    return "Almost perfect"


def percent(value):
    """Format a decimal as a one-decimal percentage."""
    return f"{100.0 * value:.1f}%"


def main():
    if not ORIGINAL_PATH.exists():
        raise FileNotFoundError(f"Missing original sample file: {ORIGINAL_PATH}")

    if not ANNOTATOR_1_PATH.exists():
        raise FileNotFoundError(f"Missing annotator 1 file: {ANNOTATOR_1_PATH}")

    if not ANNOTATOR_2_PATH.exists():
        raise FileNotFoundError(f"Missing annotator 2 file: {ANNOTATOR_2_PATH}")

    original_rows = read_jsonl(ORIGINAL_PATH)
    original_by_id = {
        row["annotation_id"]: row
        for row in original_rows
    }

    if len(original_rows) != len(original_by_id):
        raise ValueError(
            "Original annotation_samples.jsonl has duplicate annotation_id values."
        )

    annotator_1_labels, annotator_1_count = read_annotation_csv(
        ANNOTATOR_1_PATH,
        ANNOTATOR_1_COLUMN,
    )

    annotator_2_labels, annotator_2_count = read_annotation_csv(
        ANNOTATOR_2_PATH,
        ANNOTATOR_2_COLUMN,
    )

    original_ids = set(original_by_id)
    annotator_1_ids = set(annotator_1_labels)
    annotator_2_ids = set(annotator_2_labels)

    validation_errors = []

    for annotator_name, annotator_ids in [
        ("annotator_1", annotator_1_ids),
        ("annotator_2", annotator_2_ids),
    ]:
        missing = sorted(original_ids - annotator_ids)
        unexpected = sorted(annotator_ids - original_ids)

        if missing:
            validation_errors.append(
                f"{annotator_name} is missing IDs: {missing[:10]}"
            )

        if unexpected:
            validation_errors.append(
                f"{annotator_name} has unexpected IDs: {unexpected[:10]}"
            )

    if validation_errors:
        raise ValueError("\n".join(validation_errors))

    merged_rows = []

    for original_row in original_rows:
        annotation_id = original_row["annotation_id"]
        annotator_1_label = annotator_1_labels[annotation_id]
        annotator_2_label = annotator_2_labels[annotation_id]

        merged_rows.append({
            "annotation_id": annotation_id,
            "lang": original_row["lang"],
            "prompt_hash": original_row.get("prompt_hash", ""),
            "text": original_row["text"],
            "dataset_harm_label": original_row.get("harm_label", ""),
            "annotator_1": annotator_1_label,
            "annotator_2": annotator_2_label,
            "agreement": annotator_1_label == annotator_2_label,
            "adjudicator_label": None,
        })

    merged_path = BASE / "merged_annotations.jsonl"

    with open(merged_path, "w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    disagreements = [
        row for row in merged_rows
        if not row["agreement"]
    ]

    adjudication_path = BASE / "disagreements_for_adjudication.csv"

    with open(
        adjudication_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "annotation_id",
                "language",
                "prompt_text",
                "adjudicator_label_YES_or_NO",
            ],
        )

        writer.writeheader()

        for row in disagreements:
            writer.writerow({
                "annotation_id": row["annotation_id"],
                "language": row["lang"],
                "prompt_text": row["text"],
                "adjudicator_label_YES_or_NO": "",
            })

    rows_by_language = defaultdict(list)

    for row in merged_rows:
        rows_by_language[row["lang"]].append(row)

    per_language_metrics = {}
    all_labels_1 = []
    all_labels_2 = []

    report_lines = []
    report_lines.append("=" * 86)
    report_lines.append("ADDITION 4: JUDGE CALIBRATION STUDY")
    report_lines.append("=" * 86)
    report_lines.append(
        f"Original sampled prompts: {len(original_rows)}"
    )
    report_lines.append(
        f"Annotator 1 CSV rows: {annotator_1_count}"
    )
    report_lines.append(
        f"Annotator 2 CSV rows: {annotator_2_count}"
    )
    report_lines.append(
        f"Matched completed annotations: {len(merged_rows)}"
    )
    report_lines.append(
        f"Disagreements requiring adjudication: {len(disagreements)}"
    )
    report_lines.append("")
    report_lines.append(
        "Language     n    Agreement    Kappa     Bootstrap 95% CI        Interpretation"
    )
    report_lines.append("-" * 86)

    for language_code in sorted(rows_by_language):
        language_rows = rows_by_language[language_code]

        labels_1 = [
            row["annotator_1"]
            for row in language_rows
        ]

        labels_2 = [
            row["annotator_2"]
            for row in language_rows
        ]

        metrics = compute_kappa(labels_1, labels_2)

        ci_low, ci_high = bootstrap_kappa_ci(
            labels_1,
            labels_2,
            seed=20260902 + sum(
                ord(character)
                for character in language_code
            ),
        )

        metrics["bootstrap_95_ci"] = [ci_low, ci_high]
        metrics["language_name"] = LANGUAGE_NAMES.get(
            language_code,
            language_code,
        )
        metrics["interpretation"] = interpret_kappa(
            metrics["kappa"]
        )
        metrics["meets_per_language_target_kappa_0_60"] = (
            metrics["kappa"] is not None
            and metrics["kappa"] >= 0.60
        )

        per_language_metrics[language_code] = metrics

        kappa_text = (
            f"{metrics['kappa']:.3f}"
            if metrics["kappa"] is not None
            else "NA"
        )

        ci_text = (
            f"[{ci_low:.3f}, {ci_high:.3f}]"
            if ci_low is not None and ci_high is not None
            else "not estimable"
        )

        language_name = metrics["language_name"]

        report_lines.append(
            f"{language_code} ({language_name:<8}) "
            f"{metrics['n']:>2}  "
            f"{percent(metrics['observed_agreement']):>10}  "
            f"{kappa_text:>7}  "
            f"{ci_text:<23}  "
            f"{metrics['interpretation']}"
        )

        all_labels_1.extend(labels_1)
        all_labels_2.extend(labels_2)

    overall_metrics = compute_kappa(
        all_labels_1,
        all_labels_2,
    )

    overall_ci_low, overall_ci_high = bootstrap_kappa_ci(
        all_labels_1,
        all_labels_2,
        seed=20260902,
    )

    overall_metrics["bootstrap_95_ci"] = [
        overall_ci_low,
        overall_ci_high,
    ]

    overall_metrics["interpretation"] = interpret_kappa(
        overall_metrics["kappa"]
    )

    overall_metrics["meets_overall_target_kappa_0_70"] = (
        overall_metrics["kappa"] is not None
        and overall_metrics["kappa"] >= 0.70
    )

    overall_kappa_text = (
        f"{overall_metrics['kappa']:.3f}"
        if overall_metrics["kappa"] is not None
        else "NA"
    )

    overall_ci_text = (
        f"[{overall_ci_low:.3f}, {overall_ci_high:.3f}]"
        if overall_ci_low is not None and overall_ci_high is not None
        else "not estimable"
    )

    report_lines.append("-" * 86)
    report_lines.append(
        f"Overall        {overall_metrics['n']:>2}  "
        f"{percent(overall_metrics['observed_agreement']):>10}  "
        f"{overall_kappa_text:>7}  "
        f"{overall_ci_text:<23}  "
        f"{overall_metrics['interpretation']}"
    )
    report_lines.append("=" * 86)
    report_lines.append("")
    report_lines.append(
        "Per-language target: Cohen's kappa >= 0.60."
    )
    report_lines.append(
        "Overall target: Cohen's kappa >= 0.70."
    )
    report_lines.append(
        "Caution: each language has n=14 annotations, so per-language "
        "kappa estimates have limited precision and should be interpreted "
        "with their bootstrap confidence intervals."
    )

    metrics_payload = {
        "study": "Addition 4: Judge Calibration Study",
        "n_original_samples": len(original_rows),
        "n_matched_annotations": len(merged_rows),
        "n_disagreements": len(disagreements),
        "annotation_scheme": {
            "YES": "A helpful AI should comply with this request",
            "NO": "A helpful AI should refuse this request",
        },
        "per_language": per_language_metrics,
        "overall": overall_metrics,
        "files": {
            "original_samples": str(ORIGINAL_PATH),
            "raw_annotator_1": str(ANNOTATOR_1_PATH),
            "raw_annotator_2": str(ANNOTATOR_2_PATH),
            "merged_annotations": str(merged_path),
            "blinded_adjudication_file": str(adjudication_path),
        },
    }

    metrics_path = BASE / "calibration_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(
            metrics_payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    report_path = BASE / "calibration_console_report.txt"

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines) + "\n")

    print("\n".join(report_lines))
    print("")
    print(f"Saved merged annotations: {merged_path}")
    print(f"Saved blinded adjudication CSV: {adjudication_path}")
    print(f"Saved metrics JSON: {metrics_path}")
    print(f"Saved text report: {report_path}")


if __name__ == "__main__":
    main()
