#!/usr/bin/env python3
"""
Build a deterministic, pre-inference Addition 6 legacy-pipeline smoke manifest.

Selection:
    4 languages × 2 arms × 2 prompt-level cases = 16 cases.

The manifest is built from role_level_eval_dataset.jsonl. Every selected base
case must have exactly planner, tool-caller, and critic source rows with one
shared prompt text, language, and harmfulness label.

This script never loads a model and never reads model outputs.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "role_level_eval_dataset.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent / "prompts"

LANGUAGES = ("Arabic", "English", "Hausa", "Yoruba")
ARMS = ("harmful", "benign")
CASES_PER_LANGUAGE_ARM = 2

ROLE_SUFFIXES = ("_planner", "_tool-caller", "_critic")
REQUIRED_ROLES = {"planner", "tool-caller", "critic"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def base_id_from_role_id(role_id: str) -> str:
    for suffix in ROLE_SUFFIXES:
        if role_id.endswith(suffix):
            return role_id[: -len(suffix)]
    raise ValueError(f"Unrecognized role suffix in ID: {role_id!r}")


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Legacy dataset not found: {path}")

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc

            required = {
                "id",
                "role",
                "language",
                "language_code",
                "harmfulness",
                "prompt_text",
                "prompt_type",
                "gold_label",
            }
            missing = sorted(required - set(row))

            if missing:
                raise ValueError(
                    f"Dataset row {line_number} is missing required fields: {missing}"
                )

            rows.append(row)

    if not rows:
        raise ValueError("Legacy dataset contains no rows.")

    return rows


def build_cases(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[base_id_from_role_id(str(row["id"]))].append(row)

    cases: dict[str, dict[str, Any]] = {}

    for base_id, group in grouped.items():
        roles = {str(row["role"]) for row in group}

        if roles != REQUIRED_ROLES or len(group) != 3:
            raise ValueError(
                f"{base_id}: expected exactly {sorted(REQUIRED_ROLES)}, "
                f"got roles={sorted(roles)} with n_rows={len(group)}"
            )

        languages = {str(row["language"]) for row in group}
        language_codes = {str(row["language_code"]) for row in group}
        arms = {str(row["harmfulness"]) for row in group}
        prompts = {str(row["prompt_text"]) for row in group}
        prompt_types = {str(row["prompt_type"]) for row in group}

        if len(languages) != 1:
            raise ValueError(f"{base_id}: inconsistent language: {languages}")
        if len(language_codes) != 1:
            raise ValueError(f"{base_id}: inconsistent language_code: {language_codes}")
        if len(arms) != 1:
            raise ValueError(f"{base_id}: inconsistent harmfulness: {arms}")
        if len(prompts) != 1:
            raise ValueError(f"{base_id}: inconsistent prompt_text across roles")
        if len(prompt_types) != 1:
            raise ValueError(f"{base_id}: inconsistent prompt_type: {prompt_types}")

        prompt_text = next(iter(prompts))

        source_rows = {
            str(row["role"]): {
                "id": str(row["id"]),
                "gold_label": str(row["gold_label"]),
                "expected_behavior": str(row.get("expected_behavior", "")),
            }
            for row in group
        }

        cases[base_id] = {
            "base_id": base_id,
            "language": next(iter(languages)),
            "language_code": next(iter(language_codes)),
            "harmfulness": next(iter(arms)),
            "prompt_type": next(iter(prompt_types)),
            "prompt_text": prompt_text,
            "prompt_sha256": sha256_text(prompt_text),
            "source_role_rows": source_rows,
        }

    return cases


def select_cases(cases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    for language in LANGUAGES:
        for arm in ARMS:
            candidates = sorted(
    (
        case
        for case in cases.values()
        if case["language"] == language and case["harmfulness"] == arm
    ),
    key=lambda case: case["base_id"],
)

            if len(candidates) < CASES_PER_LANGUAGE_ARM:
                raise ValueError(
                    f"Insufficient cases for {language}/{arm}: "
                    f"need {CASES_PER_LANGUAGE_ARM}, found {len(candidates)}"
                )

            selected.extend(candidates[:CASES_PER_LANGUAGE_ARM])

    expected_total = len(LANGUAGES) * len(ARMS) * CASES_PER_LANGUAGE_ARM

    if len(selected) != expected_total:
        raise AssertionError(
            f"Expected {expected_total} selected cases, got {len(selected)}"
        )

    base_ids = [case["base_id"] for case in selected]

    if len(base_ids) != len(set(base_ids)):
        raise AssertionError("Selection contains duplicate base IDs.")

    counts = Counter((case["language"], case["harmfulness"]) for case in selected)

    for language in LANGUAGES:
        for arm in ARMS:
            if counts[(language, arm)] != CASES_PER_LANGUAGE_ARM:
                raise AssertionError(
                    f"Bad stratum count for {language}/{arm}: "
                    f"{counts[(language, arm)]}"
                )

    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing manifest: {path}. "
            "Use a new manifest version rather than changing a pre-inference selection."
        )

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    rows = read_rows(DATASET_PATH)
    cases = build_cases(rows)
    selected = select_cases(cases)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = OUTPUT_DIR / "legacy_smoke_manifest_v1.jsonl"
    metadata_path = OUTPUT_DIR / "legacy_smoke_manifest_v1.metadata.json"

    write_jsonl(manifest_path, selected)

    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_path": str(DATASET_PATH.relative_to(REPO_ROOT)),
        "dataset_sha256": sha256_text(DATASET_PATH.read_text(encoding="utf-8")),
        "selection_rule": (
            "Lexicographically first base IDs within each stratum: "
            "language in [Arabic, English, Hausa, Yoruba], "
            "harmfulness in [harmful, benign], "
            "two prompt-level base cases per stratum."
        ),
        "languages": list(LANGUAGES),
        "arms": list(ARMS),
        "cases_per_language_arm": CASES_PER_LANGUAGE_ARM,
        "n_prompt_level_cases": len(selected),
        "n_agent_generations_per_condition": len(selected) * 3,
        "source_role_rows_required": sorted(REQUIRED_ROLES),
        "manifest_file": str(manifest_path.relative_to(REPO_ROOT)),
        "selection_counts": {
            f"{language}__{arm}": sum(
                case["language"] == language and case["harmfulness"] == arm
                for case in selected
            )
            for language in LANGUAGES
            for arm in ARMS
        },
        "status": "pre_inference_manifest",
        "note": (
            "Engineering smoke-test manifest only. It is not a final statistical "
            "sample and must not be changed after model outputs are inspected."
        ),
    }

    if metadata_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing metadata: {metadata_path}"
        )

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Manifest created successfully.")
    print(f"Prompt-level cases: {len(selected)}")
    print(f"Generations per condition: {len(selected) * 3}")
    print(f"Manifest: {manifest_path}")
    print(f"Metadata: {metadata_path}")

    for case in selected:
        print(
            f"{case['base_id']} | {case['language']} | "
            f"{case['harmfulness']} | {case['prompt_type']}"
        )


if __name__ == "__main__":
    main()