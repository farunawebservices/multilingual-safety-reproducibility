"""Run artifact writers, split by whether the content can ever be published.

Two stores, and the boundary between them is the whole point.

``logs/<run_id>/`` -- git-ignored, **carries text**::

    raw/<prompt_id>.<trial>.<seed>.txt    verbatim model output, byte-exact
    prompts/<prompt_id>.json              rendered messages (corpus text)
    hidden/                               captured residual streams
    vectors/                              steering and control vectors

``results/<run_id>/`` -- git-ignored by policy, but **text-free by construction**,
so a collaborator can be sent it without sending them the corpus::

    execution_records.jsonl   one row per attempted stage, no parsed payload
    trials.jsonl              per-trial stage-decision vector
    tool_calls.jsonl          mock tool results, self-labelled
    decision_rates.json       all six categories x language x stage x arm
    tripwire_report.json      every threshold, its value, pass/fail
    run_metadata.json         written by RunContext

The raw text is written *before* it is parsed and is never rewritten by
parsing. If a classification is later disputed, the evidence still exists
(CLAUDE.md §2).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from multilingual_safety.agents.records import ExecutionRecord
from multilingual_safety.decisions import (
    DECISION_CATEGORIES,
    DecisionCategory,
)
from multilingual_safety.paths import run_logs_dir, run_results_dir

#: Substrings that must never appear as a key in a results artifact.
TEXT_BEARING_KEYS: tuple[str, ...] = (
    "source_text",
    "translated_text",
    "raw_text",
    "text",
    "messages",
    "rationale",
    "user_safe_response",
)


def prompt_reference(run_id: str, prompt_id: str) -> str:
    """Repo-relative path of a rendered prompt in the git-ignored text store.

    Recorded on the execution record so a metadata-only artifact can point at
    the exact prompt without carrying it.
    """
    return f"logs/{run_id}/prompts/{prompt_id}.json"


class ArtifactError(RuntimeError):
    """A write would put corpus text into a text-free artifact."""


class RunWriter:
    """Owns both stores for one run ID."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.results_dir = run_results_dir(run_id)
        self.logs_dir = run_logs_dir(run_id)

    def prepare(self) -> None:
        for directory in (
            self.results_dir,
            self.logs_dir,
            self.logs_dir / "raw",
            self.logs_dir / "prompts",
            self.logs_dir / "hidden",
            self.logs_dir / "vectors",
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # --- text store (git-ignored) -----------------------------------------
    def write_raw(self, prompt_id: str, trial_index: int, seed: int | None, text: str) -> Path:
        """Verbatim model output. Written before parsing, never rewritten."""
        path = self.logs_dir / "raw" / f"{prompt_id}.{trial_index}.{seed}.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def write_prompt(self, prompt_id: str, messages: Sequence[Mapping[str, str]]) -> Path:
        path = self.logs_dir / "prompts" / f"{prompt_id}.json"
        if not path.exists():
            path.write_text(
                json.dumps([dict(m) for m in messages], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        return path

    # --- text-free store ---------------------------------------------------
    def write_jsonl(self, name: str, rows: Iterable[Mapping[str, Any]]) -> Path:
        path = self.results_dir / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                assert_text_free(row, name)
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        return path

    def write_json(self, name: str, payload: Mapping[str, Any]) -> Path:
        assert_text_free(payload, name)
        path = self.results_dir / name
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path


def assert_text_free(payload: Any, where: str) -> None:
    """Refuse a results artifact carrying corpus text.

    Checks keys, not values: a value could legitimately be any string, but a
    ``translated_text`` key can only mean the corpus leaked into a publishable
    artifact.
    """
    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if str(key) in TEXT_BEARING_KEYS:
                    raise ArtifactError(
                        f"{where}: key {path}/{key!r} is text-bearing. Results artifacts "
                        "carry hashes and metadata only (README §7)."
                    )
                walk(value, f"{path}/{key}")
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(payload, "")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def decision_rates(
    records: Sequence[ExecutionRecord],
    *,
    languages: Sequence[str],
    stages: Sequence[str],
    arms: Sequence[str] = ("harmful", "benign"),
) -> dict[str, Any]:
    """All six categories per language x stage x arm, over ATTEMPTED trials.

    ``NOT_RUN`` rows are counted separately and excluded from the stage's
    denominator: a stage that was never attempted did not fail, and folding it
    into a rate would understate the stages that did run. The count is reported
    alongside so the reduction is visible in the same table (CLAUDE.md §2).
    """
    table: dict[str, Any] = {
        "denominator": "attempted",
        "note": (
            "Rates are over trials ATTEMPTED at that stage. not_run counts stages the "
            "pipeline never reached because an upstream stage blocked or failed; they "
            "are excluded from the denominator and reported here. Those rows carry "
            "stage_status=NOT_RUN and decision_category=NOT_RUN, a sentinel and not a "
            "seventh category: they are neither MISSING_DECISION nor "
            "MALFORMED_DECISION, and they contribute to no rate."
        ),
        "malformed_note": (
            "MALFORMED_DECISION is reported with a breakdown, because it collects two "
            "different things. `model_output` is the model emitting something "
            "unparseable -- a finding about the model. `apparatus` is the inference "
            "runtime failing, and `pre_inference` is the subset where the backend "
            "refused before the model was ever invoked, which says nothing about the "
            "model at all. Only `model_output` is evidence of malformed model output."
        ),
        "by_language": {},
    }
    for language in languages:
        per_language: dict[str, Any] = {}
        for stage in stages:
            per_stage: dict[str, Any] = {}
            for arm in arms:
                subset = [
                    r
                    for r in records
                    if r.language == language and r.role == stage and r.harm_label == arm
                ]
                attempted = [r for r in subset if r.attempted]
                not_run = len(subset) - len(attempted)
                counts = {c.value: 0 for c in DECISION_CATEGORIES}
                for record in attempted:
                    counts[record.decision_category] += 1
                total = len(attempted)
                malformed = [
                    r
                    for r in attempted
                    if r.category is DecisionCategory.MALFORMED_DECISION
                ]
                apparatus = [r for r in malformed if r.apparatus_error]
                pre_inference = [r for r in apparatus if not r.model_invoked]
                per_stage[arm] = {
                    "attempted": total,
                    "not_run": not_run,
                    "counts": counts,
                    "rates": (
                        {k: v / total for k, v in counts.items()} if total else None
                    ),
                    # See `malformed_note`: these three are different findings and
                    # must never be summed into "the model produced bad output".
                    "malformed_breakdown": {
                        "model_output": len(malformed) - len(apparatus),
                        "apparatus": len(apparatus),
                        "pre_inference": len(pre_inference),
                    },
                    "apparatus_errors": sum(1 for r in attempted if r.apparatus_error),
                    "model_invoked": sum(1 for r in attempted if r.model_invoked),
                    "unreviewed_rows": sum(1 for r in attempted if not r.human_reviewed),
                }
            per_language[stage] = per_stage
        table["by_language"][language] = per_language
    return table


def tripwire_report(
    records: Sequence[ExecutionRecord], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the configured tripwires. A breach halts; it never filters."""
    attempted = [r for r in records if r.attempted]
    total = len(attempted)
    checks: list[dict[str, Any]] = []

    def rate(category: DecisionCategory) -> float:
        if not total:
            return 0.0
        return sum(1 for r in attempted if r.category is category) / total

    for key, category in (
        ("max_timeout_rate", DecisionCategory.TIMEOUT),
        ("max_non_termination_rate", DecisionCategory.NON_TERMINATION),
        ("max_malformed_rate", DecisionCategory.MALFORMED_DECISION),
    ):
        if key not in thresholds:
            continue
        observed = rate(category)
        checks.append(
            {
                "check": key,
                "threshold": thresholds[key],
                "observed": observed,
                "passed": observed <= float(thresholds[key]),
            }
        )

    distinct = len({r.raw_sha256 for r in attempted})
    ratio = distinct / total if total else 0.0
    if "min_distinct_output_ratio" in thresholds:
        checks.append(
            {
                "check": "min_distinct_output_ratio",
                "threshold": thresholds["min_distinct_output_ratio"],
                "observed": ratio,
                "passed": ratio >= float(thresholds["min_distinct_output_ratio"]),
            }
        )
    if thresholds.get("halt_on_zero_rows"):
        checks.append(
            {"check": "halt_on_zero_rows", "threshold": 0, "observed": total, "passed": total > 0}
        )

    failed = [c for c in checks if not c["passed"]]
    # Thresholds and what they measure are unchanged: an apparatus failure still
    # counts toward max_malformed_rate, and it should -- a backend bug that
    # breaks every generation must halt the run. What is added is the reason, so
    # a breach is diagnosable without re-reading the records.
    apparatus = [r for r in attempted if r.apparatus_error]
    pre_inference = [r for r in apparatus if not r.model_invoked]
    return {
        "attempted": total,
        "not_run": len(records) - total,
        "checks": checks,
        "failed": [c["check"] for c in failed],
        "passed": not failed,
        "apparatus_errors": len(apparatus),
        "pre_inference_failures": len(pre_inference),
        "apparatus_note": (
            "pre_inference_failures are trials where the backend refused before the "
            "model was invoked. If a breach is composed of these, it is a bug in the "
            "harness, not a finding about the model."
        ),
        "policy": (
            "A breach halts the run and is reported. Degenerate output is a bug until "
            "proven otherwise; it is never fixed by filtering it out (CLAUDE.md §10)."
        ),
    }
