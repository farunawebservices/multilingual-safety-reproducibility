#!/usr/bin/env python3
"""Dataset-v2 experiment runner -- preflight only, dev split by default.

This is the smallest runner that can honestly exist right now: the Planner and
Critic backends are not implemented and the Tool-Caller surface is mocked, so
there is no experiment to run. What the script *can* do is everything that must
happen before one: parse the resolved config, verify the three hashes every v2
result has to reference, and resolve the dev behavior set.

Two modes, and only two:

``--dry-run``
    Parse, verify, select, report. No model inference, no dataset artifact, no
    run directory. This is the mode that works today.

no ``--dry-run``
    Refuses with a configuration error naming the missing backends. It does
    **not** mint a run ID, create ``results/<run_id>/``, or emit an empty result
    table. A runner that "completes" against unimplemented backends produces a
    zero-row artifact that looks like a finding; that is the failure this
    refusal exists to prevent (CLAUDE.md §10).

Holdout protection (CLAUDE.md §1)
---------------------------------
``--split`` defaults to ``dev``. Any holdout split -- ``final_test`` by name in
v2, or anything else listed in ``splits.HOLDOUT_SPLIT_NAMES`` -- is refused
unless ``--allow-test`` is passed, and the refusal fires *before* the split file
is opened, so a mistyped invocation cannot even learn which behaviors are held
out. Development, prompt iteration, threshold tuning, and intervention-strength
selection all happen on dev.

The frozen corpus is one file holding all 1,400 rows, so selecting dev means
streaming it and keeping only the rows whose ``behavior_id`` is in the dev set.
Non-selected lines are counted and dropped without their text being retained,
printed, or written anywhere. No prompt text reaches stdout or any artifact in
either mode (repository conventions, THIRD_PARTY_DATASETS.md).

Usage
-----
    python scripts/run_experiment_v2.py --config configs/experiment_v2.example.yaml --dry-run
    python scripts/run_experiment_v2.py --config configs/experiment_v2.example.yaml --dry-run --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from multilingual_safety.agents.tool_caller import ToolCaller
from multilingual_safety.hashing import file_sha256
from multilingual_safety.ids import is_behavior_id, stable_hash
from multilingual_safety.paths import REPO_ROOT, display_path
from multilingual_safety.splits import (
    DEV,
    HOLDOUT_SPLIT_NAMES,
    SplitAccessError,
    SplitError,
    load_split,
    load_splits,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment_v2.example.yaml"

#: Dataset version this runner accepts. v1 has its own frozen artifacts, its own
#: split hash, and its own pre-registration; v1 and v2 results are never merged,
#: so pointing this runner at a v1 config is an error rather than a convenience.
DATASET_VERSION = "v2"

#: ``data`` keys a v2 config must carry. Every one of them ends up in a run's
#: metadata, so a missing key is a configuration error, not a defaultable field.
REQUIRED_DATA_KEYS: tuple[str, ...] = (
    "dataset_version",
    "dataset",
    "dataset_sha256",
    "splits",
    "splits_sha256",
    "split_hash",
)

#: Languages every behavior is rendered in. Used only as an expected-shape check
#: on the selected rows; the config's own list is what a real run would use.
EXPECTED_LANGUAGES: tuple[str, ...] = ("en", "ar", "ha", "yo", "sw", "hi", "ko")

#: Stages with no backend. This gate is deliberately declarative rather than
#: probing ``Planner().run(...)``: if a backend ever lands, the probe itself
#: would issue a model call. Whoever implements one updates this tuple in the
#: same change.
UNIMPLEMENTED_BACKENDS: tuple[str, ...] = ("planner", "critic")

#: Row fields the preflight retains. Deliberately excludes ``source_text`` and
#: ``translated_text``: the preflight needs shape, not corpus text.
RETAINED_FIELDS: tuple[str, ...] = (
    "behavior_id",
    "language",
    "harm_label",
    "category",
    "prompt_hash",
)


class ConfigurationError(RuntimeError):
    """The run cannot proceed as configured. Raised before anything is written."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Parse and structurally validate a v2 experiment config.

    Validates shape only. The hashes it declares are checked against the files
    on disk by :func:`verify_hashes`, which is a separate step because a config
    that parses is not a config that describes the corpus you actually have.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"config not found: {display_path(path)}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"{display_path(path)} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"{display_path(path)} must contain a YAML mapping, got {type(raw).__name__}"
        )

    data = raw.get("data")
    if not isinstance(data, dict):
        raise ConfigurationError(f"{display_path(path)} has no `data:` section")

    missing = [key for key in REQUIRED_DATA_KEYS if data.get(key) in (None, "")]
    if missing:
        raise ConfigurationError(
            f"{display_path(path)} is missing required data keys: {', '.join(missing)}. "
            "Every one of these is recorded in run metadata; none has a default."
        )

    version = str(data["dataset_version"])
    if version != DATASET_VERSION:
        raise ConfigurationError(
            f"{display_path(path)} declares dataset_version {version!r}; this runner "
            f"only accepts {DATASET_VERSION!r}. v1 and v2 results are never merged."
        )

    return raw


def config_split(config: dict[str, Any]) -> str:
    """The split the config itself pre-registers. Defaults to dev."""
    return str(config["data"].get("split") or DEV)


# ---------------------------------------------------------------------------
# Holdout gate
# ---------------------------------------------------------------------------
def resolve_split(requested: str, *, allow_test: bool) -> str:
    """Return ``requested``, or refuse if it is a holdout without an opt-in.

    Runs before the split file is opened. A refused invocation therefore learns
    nothing at all about the holdout -- not its size, not its members.
    """
    if requested in HOLDOUT_SPLIT_NAMES and not allow_test:
        raise SplitAccessError(
            f"refusing to read the {requested!r} split. Development, prompt iteration, "
            "threshold tuning, and intervention-strength selection all happen on dev "
            "(CLAUDE.md §1). Pass --allow-test only for a final, frozen configuration, "
            "and record the read in the config's holdout_policy.final_test_reads."
        )
    return requested


# ---------------------------------------------------------------------------
# Hash verification
# ---------------------------------------------------------------------------
def verify_hashes(config: dict[str, Any], *, root: Path = REPO_ROOT) -> dict[str, str]:
    """Verify the dataset SHA-256, the splits SHA-256, and the split_hash.

    All three, every run. The first two say the bytes are the ones the config
    describes; the third says the *assignment* is the one every v2 result claims
    to reference. A result whose split hash drifted is not comparable with any
    other, so a mismatch aborts rather than warns (CLAUDE.md §1, §10).
    """
    data = config["data"]
    dataset_path = root / data["dataset"]
    splits_path = root / data["splits"]

    for label, path in (("dataset", dataset_path), ("splits", splits_path)):
        if not path.exists():
            raise ConfigurationError(
                f"{label} file not found: {display_path(path)} "
                f"(config `data.{label}`). The v2 corpus is git-ignored; materialize "
                "it before running."
            )

    dataset_sha = file_sha256(dataset_path)
    expected_dataset_sha = str(data["dataset_sha256"]).strip().lower()
    if dataset_sha != expected_dataset_sha:
        raise ConfigurationError(
            f"dataset hash mismatch for {display_path(dataset_path)}: config records "
            f"{expected_dataset_sha}, file hashes to {dataset_sha}. The frozen corpus "
            "has changed; results computed against either version are not comparable."
        )

    splits_sha = file_sha256(splits_path)
    expected_splits_sha = str(data["splits_sha256"]).strip().lower()
    if splits_sha != expected_splits_sha:
        raise ConfigurationError(
            f"splits hash mismatch for {display_path(splits_path)}: config records "
            f"{expected_splits_sha}, file hashes to {splits_sha}. Never re-split "
            "quietly (CLAUDE.md §1)."
        )

    # load_splits recomputes the payload's own content hash and raises on drift,
    # so this catches an edited split file whose recorded hash was edited to match.
    payload = load_splits(splits_path)
    recorded = str(payload["split_hash"]).strip().lower()
    expected_split_hash = str(data["split_hash"]).strip().lower()
    if recorded != expected_split_hash:
        raise ConfigurationError(
            f"split_hash mismatch: config requires {expected_split_hash}, "
            f"{display_path(splits_path)} carries {recorded}. Every v2 run must "
            "reference the frozen assignment."
        )

    return {
        "dataset_sha256": dataset_sha,
        "splits_sha256": splits_sha,
        "split_hash": recorded,
    }


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def select_behaviors(
    splits_path: str | Path,
    split: str,
    *,
    allow_test: bool = False,
    limit: int | None = None,
) -> tuple[list[str], int]:
    """Return ``(selected behavior IDs, total in split)``.

    ``load_split`` enforces the holdout opt-in a second time, at the library
    boundary. The CLI gate in :func:`resolve_split` is the one that keeps the
    file unopened; this one is the guarantee that no call path skips the check.

    ``limit`` truncates the sorted ID list, so the same limit always yields the
    same behaviors. A truncated selection is a reduced denominator and is
    reported as such (CLAUDE.md §2) -- it is a smoke-test aid, never a sample.
    """
    behaviors = load_split(splits_path, split, allow_test=allow_test)
    total = len(behaviors)
    if total == 0:
        raise ConfigurationError(
            f"split {split!r} is empty (stop condition, CLAUDE.md §10)"
        )

    malformed = [b for b in behaviors if not is_behavior_id(b)]
    if malformed:
        raise ConfigurationError(
            f"{len(malformed)} malformed behavior id(s) in split {split!r}, "
            f"e.g. {malformed[:3]}"
        )

    if limit is None:
        return behaviors, total
    if limit < 1:
        raise ConfigurationError(f"--limit must be >= 1, got {limit}")
    return behaviors[:limit], total


@dataclass(slots=True)
class Selection:
    """Metadata-only view of the rows belonging to the selected behaviors."""

    n_rows: int
    n_lines_scanned: int
    by_language: dict[str, int] = field(default_factory=dict)
    by_arm: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)
    behaviors_found: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)


def load_selected_rows(
    dataset_path: str | Path,
    behavior_ids: list[str],
) -> Selection:
    """Stream the frozen corpus, retaining only the selected behaviors' metadata.

    The corpus is a single file containing every split, so the holdout rows are
    physically in the stream. They are matched against the wanted set, skipped,
    and counted; nothing from a non-selected row is retained, printed, or
    written. Selected rows keep :data:`RETAINED_FIELDS` only -- no prompt text
    is loaded into the report.
    """
    wanted = set(behavior_ids)
    if not wanted:
        raise ConfigurationError("refusing to select zero behaviors (stop condition)")

    selection = Selection(n_rows=0, n_lines_scanned=0)
    languages: Counter[str] = Counter()
    arms: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    seen: set[str] = set()

    with Path(dataset_path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            selection.n_lines_scanned += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                # Not skipped silently: a corpus line that will not parse is a
                # corrupted frozen dataset, which invalidates its hash claim.
                raise ConfigurationError(
                    f"{display_path(dataset_path)}:{lineno} is not valid JSON: {exc}"
                ) from exc

            behavior = row.get("behavior_id")
            if behavior not in wanted:
                continue

            seen.add(behavior)
            selection.n_rows += 1
            languages[str(row.get("language"))] += 1
            arms[str(row.get("harm_label"))] += 1
            categories[str(row.get("category"))] += 1
            selection.rows.append({key: row.get(key) for key in RETAINED_FIELDS})

    selection.behaviors_found = len(seen)
    selection.by_language = dict(sorted(languages.items()))
    selection.by_arm = dict(sorted(arms.items()))
    selection.by_category = dict(sorted(categories.items()))

    unexpected = sorted(set(selection.by_language) - set(EXPECTED_LANGUAGES))
    if unexpected:
        raise ConfigurationError(
            f"unexpected language(s) in the selected rows: {unexpected}; "
            f"expected {list(EXPECTED_LANGUAGES)}"
        )
    missing = sorted(wanted - seen)
    if missing:
        raise ConfigurationError(
            f"{len(missing)} selected behavior(s) have no rows in "
            f"{display_path(dataset_path)}, e.g. {missing[:3]}. The split file and "
            "the corpus disagree (CLAUDE.md §10)."
        )
    expected_rows = len(wanted) * len(EXPECTED_LANGUAGES)
    if selection.n_rows != expected_rows:
        raise ConfigurationError(
            f"row count mismatch: {len(wanted)} behavior(s) x "
            f"{len(EXPECTED_LANGUAGES)} languages should be {expected_rows} rows, "
            f"loaded {selection.n_rows} (CLAUDE.md §10)"
        )
    if selection.n_rows == 0:
        raise ConfigurationError("selection produced zero rows (stop condition)")
    if len(selection.by_language) == 1:
        raise ConfigurationError(
            f"every selected row is language {next(iter(selection.by_language))!r} "
            "(stop condition, CLAUDE.md §10)"
        )
    return selection


# ---------------------------------------------------------------------------
# Backend gate
# ---------------------------------------------------------------------------
def assert_tool_caller_is_mocked() -> None:
    """The tool surface is mock-only, under every flag (repository conventions)."""
    if getattr(ToolCaller, "allow_real_execution", False):
        raise ConfigurationError(
            "ToolCaller.allow_real_execution is True. The tool surface is mocked and "
            "no real capability is wired in; refusing to proceed."
        )


def assert_backends_available() -> None:
    """Refuse a real run while the Planner/Critic backends are missing."""
    if UNIMPLEMENTED_BACKENDS:
        stages = " and ".join(name.capitalize() for name in UNIMPLEMENTED_BACKENDS)
        raise ConfigurationError(
            f"{stages} backends are not implemented, so this experiment cannot be run. "
            "No model inference is available at this stage of the project "
            "(src/multilingual_safety/agents/: every stage's run() raises "
            "NotImplementedError). Re-run with --dry-run to verify the config, the "
            "dataset and split hashes, and the dev selection. Nothing was written: a "
            "zero-row result table would be indistinguishable from a finding "
            "(CLAUDE.md §10)."
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in counts.items()) or "(none)"


def render_report(
    *,
    config_path: Path,
    config: dict[str, Any],
    split: str,
    hashes: dict[str, str],
    selected: list[str],
    total_in_split: int,
    selection: Selection,
    limit: int | None,
) -> str:
    data = config["data"]
    lines = [
        "dataset v2 experiment preflight (dry run)",
        "",
        f"config            : {display_path(config_path)}",
        f"config_hash       : {stable_hash(config)}",
        f"experiment        : {config.get('experiment', {}).get('name', '(unnamed)')}",
        f"dataset           : {display_path(REPO_ROOT / data['dataset'])}",
        f"splits            : {display_path(REPO_ROOT / data['splits'])}",
        "",
        "hash verification",
        f"  dataset_sha256  : {hashes['dataset_sha256']}  OK",
        f"  splits_sha256   : {hashes['splits_sha256']}  OK",
        f"  split_hash      : {hashes['split_hash']}  OK",
        "",
        "selection",
        f"  split           : {split}",
        f"  behaviors       : {len(selected)} of {total_in_split} in split",
    ]
    if limit is not None:
        lines.append(
            f"  limit           : {limit} applied -- REDUCED DENOMINATOR, smoke test only"
        )
    lines += [
        f"  rows loaded     : {selection.n_rows} (from {selection.n_lines_scanned} corpus lines)",
        f"  by language     : {_fmt_counts(selection.by_language)}",
        f"  by arm          : {_fmt_counts(selection.by_arm)}",
        f"  categories      : {len(selection.by_category)}",
        "",
        "execution",
        "  model inference : none (dry run)",
        "  tool surface    : mocked (ToolCaller.allow_real_execution=False)",
        f"  backends missing: {', '.join(UNIMPLEMENTED_BACKENDS) or 'none'}",
        "  artifacts       : none written",
        "  a real run would write only under results/<run_id>/ and logs/<run_id>/",
        "",
        f"Dry run: nothing written. Holdout splits ({', '.join(sorted(HOLDOUT_SPLIT_NAMES))}) "
        "were not read.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dataset-v2 experiment runner. Dev split by default; --dry-run verifies "
            "config, dataset/split hashes, and the dev selection without inference."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Resolved v2 experiment config YAML (default: configs/experiment_v2.example.yaml)",
    )
    parser.add_argument(
        "--split",
        default=DEV,
        help="Split to run. Default dev. A holdout split requires --allow-test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify config, hashes, and selection. No inference, no artifacts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N behaviors of the split (smoke test; reduces the denominator).",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicit opt-in required to touch a holdout split (CLAUDE.md §1).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        # Order matters. The holdout gate fires before any file is opened, so a
        # refused invocation cannot learn anything about the holdout.
        split = resolve_split(args.split, allow_test=args.allow_test)
        assert_tool_caller_is_mocked()

        config = load_experiment_config(args.config)
        if split != config_split(config):
            print(
                f"note: --split {split!r} differs from the config's pre-registered "
                f"split {config_split(config)!r}",
                file=sys.stderr,
            )

        hashes = verify_hashes(config)
        selected, total_in_split = select_behaviors(
            REPO_ROOT / config["data"]["splits"],
            split,
            allow_test=args.allow_test,
            limit=args.limit,
        )
        selection = load_selected_rows(REPO_ROOT / config["data"]["dataset"], selected)

        if not args.dry_run:
            assert_backends_available()
            # Unreachable while UNIMPLEMENTED_BACKENDS is non-empty. Whoever
            # implements the backends builds the run loop here -- writing only
            # under results/<run_id>/ and logs/<run_id>/, via RunContext.
            raise ConfigurationError(
                "no run loop is implemented; refusing to report an empty experiment"
            )

        print(
            render_report(
                config_path=args.config,
                config=config,
                split=split,
                hashes=hashes,
                selected=selected,
                total_in_split=total_in_split,
                selection=selection,
                limit=args.limit,
            )
        )
        return 0

    except SplitAccessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ConfigurationError, SplitError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
