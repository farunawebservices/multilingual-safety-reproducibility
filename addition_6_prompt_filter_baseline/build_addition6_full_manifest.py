#!/usr/bin/env python3
"""
Build the frozen Addition 6 full prompt-level manifest.

Selection rule, fixed before full-condition inference:
- Four languages: Arabic, English, Hausa, Yoruba.
- Include all 100 harmful prompt-level cases per language.
- Include the first 15 benign prompt-level cases per language by base_id.
- Each selected prompt must have exactly Planner, Tool-Caller, and Critic
  source rows with identical shared prompt fields.

Source dataset is never modified.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

SOURCE_DATASET = REPO_ROOT / "role_level_eval_dataset.jsonl"

OUTPUT_MANIFEST = (
    SCRIPT_DIR
    / "prompts"
    / "addition6_full_manifest_v1.jsonl"
)

OUTPUT_METADATA = (
    SCRIPT_DIR
    / "prompts"
    / "addition6_full_manifest_v1.metadata.json"
)

LANGUAGES = ("Arabic", "English", "Hausa", "Yoruba")
HARMFUL_PER_LANGUAGE = 100
BENIGN_PER_LANGUAGE = 15

EXPECTED_ROLE_SET = {"planner", "tool-caller", "critic"}

SHARED_FIELDS = (
    "language",
    "language_code",
    "harmfulness",
    "prompt_type",
    "prompt_text",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def base_id_from_role_id(role_id: str) -> str:
    parts = role_id.split("_")

    if len(parts) < 2:
        raise ValueError(f"Role-level ID cannot form base ID: {role_id!r}")

    return "_".join(parts[:-1])


def load_and_group_source() -> dict[str, list[dict[str, Any]]]:
    if not SOURCE_DATASET.is_file():
        raise FileNotFoundError(f"Source dataset missing: {SOURCE_DATASET}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with SOURCE_DATASET.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            base_id = base_id_from_role_id(row["id"])
            groups[base_id].append(row)

    if len(groups) != 565:
        raise ValueError(
            f"Expected 565 prompt-level groups; found {len(groups)}"
        )

    for base_id, group in groups.items():
        roles = {row["role"] for row in group}

        if len(group) != 3 or roles != EXPECTED_ROLE_SET:
            raise ValueError(
                f"{base_id}: expected 3 role rows with "
                f"{sorted(EXPECTED_ROLE_SET)}, found {len(group)} rows / "
                f"{sorted(roles)}"
            )

        for field in SHARED_FIELDS:
            values = {row[field] for row in group}

            if len(values) != 1:
                raise ValueError(
                    f"{base_id}: inconsistent {field}: {sorted(values)!r}"
                )

    return groups


def make_prompt_record(
    base_id: str,
    group: list[dict[str, Any]],
) -> dict[str, Any]:
    representative = group[0]

    role_rows = sorted(
        (
            {
                "id": row["id"],
                "role": row["role"],
            }
            for row in group
        ),
        key=lambda row: row["role"],
    )

    return {
        "base_id": base_id,
        "language": representative["language"],
        "language_code": representative["language_code"],
        "harmfulness": representative["harmfulness"],
        "prompt_type": representative["prompt_type"],
        "prompt_text": representative["prompt_text"],
        "prompt_sha256": sha256_text(representative["prompt_text"]),
        "source_role_rows": role_rows,
    }


def main() -> None:
    if OUTPUT_MANIFEST.exists() or OUTPUT_METADATA.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing frozen manifest or metadata:\n"
            f"  {OUTPUT_MANIFEST}\n"
            f"  {OUTPUT_METADATA}\n"
            "Create a new versioned filename if a new manifest is required."
        )

    groups = load_and_group_source()

    prompt_records = [
        make_prompt_record(base_id, group)
        for base_id, group in groups.items()
    ]

    selected: list[dict[str, Any]] = []

    for language in LANGUAGES:
        harmful = sorted(
            (
                row
                for row in prompt_records
                if row["language"] == language
                and row["harmfulness"] == "harmful"
            ),
            key=lambda row: row["base_id"],
        )

        benign = sorted(
            (
                row
                for row in prompt_records
                if row["language"] == language
                and row["harmfulness"] == "benign"
            ),
            key=lambda row: row["base_id"],
        )

        if len(harmful) != HARMFUL_PER_LANGUAGE:
            raise ValueError(
                f"{language}: expected {HARMFUL_PER_LANGUAGE} harmful prompts, "
                f"found {len(harmful)}"
            )

        if len(benign) < BENIGN_PER_LANGUAGE:
            raise ValueError(
                f"{language}: need {BENIGN_PER_LANGUAGE} benign prompts, "
                f"found {len(benign)}"
            )

        selected.extend(harmful)
        selected.extend(benign[:BENIGN_PER_LANGUAGE])

    selected.sort(
        key=lambda row: (
            LANGUAGES.index(row["language"]),
            0 if row["harmfulness"] == "harmful" else 1,
            row["base_id"],
        )
    )

    if len(selected) != 460:
        raise ValueError(f"Expected 460 selected cases; found {len(selected)}")

    if len({row["base_id"] for row in selected}) != 460:
        raise ValueError("Duplicate base IDs in selected manifest.")

    counts = Counter(
        (row["language"], row["harmfulness"])
        for row in selected
    )

    for language in LANGUAGES:
        if counts[(language, "harmful")] != HARMFUL_PER_LANGUAGE:
            raise ValueError(f"{language}: harmful selection count mismatch")

        if counts[(language, "benign")] != BENIGN_PER_LANGUAGE:
            raise ValueError(f"{language}: benign selection count mismatch")

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_MANIFEST.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "source_dataset": str(SOURCE_DATASET.relative_to(REPO_ROOT)),
        "source_dataset_sha256": sha256_file(SOURCE_DATASET),
        "source_role_level_row_count": 1695,
        "source_prompt_level_case_count": 565,
        "manifest_path": str(OUTPUT_MANIFEST.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256_file(OUTPUT_MANIFEST),
        "n_selected_prompt_level_cases": len(selected),
        "selection_rule": {
            "languages_in_order": list(LANGUAGES),
            "harmful_selection": (
                "All harmful prompt-level cases per language, sorted by base_id."
            ),
            "benign_selection": (
                "First 15 benign prompt-level cases per language, sorted by base_id."
            ),
            "harmful_per_language": HARMFUL_PER_LANGUAGE,
            "benign_per_language": BENIGN_PER_LANGUAGE,
        },
        "language_harmfulness_counts": {
            f"{language}__harmful": counts[(language, "harmful")]
            for language in LANGUAGES
        } | {
            f"{language}__benign": counts[(language, "benign")]
            for language in LANGUAGES
        },
        "dataset_limitation": (
            "The source corpus contains 15 Yoruba benign prompt-level cases, "
            "versus 50 benign cases for Arabic, English, and Hausa. To preserve "
            "equal language representation, the manifest uses 15 benign cases "
            "per language. Benign false-positive estimates, especially for "
            "Yoruba, therefore have wider uncertainty."
        ),
        "prompt_type_limitation": (
            "Prompt-type labels are retained for provenance but are not used for "
            "benign selection because benign type labels are semantically noisy "
            "in parts of the source corpus."
        ),
    }

    with OUTPUT_METADATA.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("=" * 88)
    print("ADDITION 6 FULL MANIFEST CREATED")
    print("=" * 88)
    print("Source dataset:", SOURCE_DATASET)
    print("Source SHA256:", sha256_file(SOURCE_DATASET))
    print("Manifest:", OUTPUT_MANIFEST)
    print("Manifest SHA256:", sha256_file(OUTPUT_MANIFEST))
    print("Metadata:", OUTPUT_METADATA)
    print("Selected prompt-level cases:", len(selected))

    print("\nLanguage × harmfulness:")
    for language in LANGUAGES:
        print(
            f"  {language}: "
            f"harmful={counts[(language, 'harmful')]}, "
            f"benign={counts[(language, 'benign')]}"
        )


if __name__ == "__main__":
    main()
