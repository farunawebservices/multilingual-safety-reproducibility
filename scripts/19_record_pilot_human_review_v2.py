#!/usr/bin/env python3
"""Stage 1.4 (dataset v2) -- record the pilot human review as metadata only.

Two human reviewers independently inspected all 20 pilot behaviors across all
seven languages (140 rows, 120 non-English translations) and reported the pilot
clean and suitable to proceed to full translation, at an estimated 97% semantic
fidelity. No machine or LLM stand-in was used.

**Row-level decisions were not recorded.** The reviewers reported one aggregate
outcome, not 120 per-row verdicts, so there is no per-row label, no per-row
reviewer attribution, no disagreement set, and no basis for a kappa. This script
records the review *and* records that gap, in the same file, so that nothing
downstream can read the outcome as row-level evidence.

Nothing here invents a row label. The private record enumerates all 140 rows
with ``decision_recorded: false`` precisely so a later reader cannot assume one.

What it does NOT do
-------------------
* It does not touch ``translated_text``. Every machine translation is preserved
  byte-for-byte; the outputs file is opened read-only and re-hashed afterwards.
* It does not set ``translation_status: HUMAN_REVIEWED``, does not clear
  ``review_required``, and does not write a ``reviewer_id``. Under
  ``multilingual_safety.review`` only a ``native_speaker`` or
  ``professional_translator`` decision *with a named reviewer* may clear review,
  and this review recorded neither reviewer identity nor competence.
* It does not write ``pilot_review_decisions_v2.jsonl``. Stages 1.3/1.5 read that
  path as a per-row decision layer; an aggregate record placed there would be
  parsed as row-level verdicts that do not exist.
* It does not claim the v2 ``semantic_validation`` freeze gate is met. That gate
  needs row-level gating labels, a kappa >= 0.60, and per-arm preserved counts,
  none of which an aggregate estimate can supply.
* It writes no prompt text and no translated text to either output.

Outputs
-------
    data/v2/manifests/pilot_human_review_v2.json                     tracked, metadata only
    data/v2/interim/translation_validation/
        pilot_human_review_record_v2.json                            private, git-ignored

Usage
-----
    python scripts/19_record_pilot_human_review_v2.py            # dry run
    python scripts/19_record_pilot_human_review_v2.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.hashing import deterministic_digest, file_sha256, json_sha256
from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE
from multilingual_safety.paths import DATA_DIR, REPO_ROOT
from multilingual_safety.providers import STATUS_OK, ResponseLog, make_request_id
from multilingual_safety.review import ReviewerCompetence, competence_rationale
from multilingual_safety.run_context import stage_report_header

V2_DIR = DATA_DIR / "v2"
V2_INTERIM = V2_DIR / "interim"
V2_MANIFESTS = V2_DIR / "manifests"
PRIVATE_REVIEW_DIR = V2_INTERIM / "translation_validation"

OUTPUTS_PATH = V2_INTERIM / "pilot_outputs_v2.jsonl"
RESPONSES_PATH = V2_INTERIM / "pilot_responses_v2.jsonl"
MANIFEST_PATH = V2_MANIFESTS / "pilot_manifest_v2.json"
VALIDATION_PATH = V2_MANIFESTS / "pilot_validation_v2.json"
WIDE_REVIEW_CSV = PRIVATE_REVIEW_DIR / "pilot_wide_review_v2.csv"
PROTOCOL_PATH = REPO_ROOT / "configs" / "dataset_v2_rebuild.yaml"

TRACKED_OUT = V2_MANIFESTS / "pilot_human_review_v2.json"
PRIVATE_OUT = PRIVATE_REVIEW_DIR / "pilot_human_review_record_v2.json"

#: The decisions path stages 1.3/1.5 expect. Deliberately NOT written here.
DECISIONS_JSONL = PRIVATE_REVIEW_DIR / "pilot_review_decisions_v2.jsonl"

#: What the reviewers reported. Supplied by the operator who ran the review;
#: this repository cannot verify any of it, which is why it is nested under a
#: key that says "as reported" and is never merged into the verified block.
REVIEW_AS_REPORTED: dict[str, Any] = {
    "n_reviewers": 2,
    "reviewers_independent": True,
    "languages_reviewed": list(LANGUAGE_CODES),
    "n_behaviors_reviewed": 20,
    "n_rows_reviewed": 140,
    "n_non_english_translations_reviewed": 120,
    "estimated_semantic_fidelity": 0.97,
    "outcome": "clean_suitable_to_proceed_to_full_translation",
    "machine_or_llm_fallback": False,
    "adjudicator_required_only_if_row_level_disagreements_exist": True,
}

#: Verbatim status fields requested for this record.
HUMAN_REVIEWED = True
ROW_LEVEL_DECISIONS_RECORDED = False
ESTIMATED_FIDELITY = 0.97
FORMAL_KAPPA = None
FORMAL_ADJUDICATION = "not_applicable_or_not_recorded"

#: This review's per-row state. Deliberately NOT a ``ReviewDecisionType``:
#: ``PENDING`` would say the rows were never looked at (they were), and
#: ``ACCEPT`` would assert a per-row verdict that was never recorded.
ROW_STATE_LABEL = "REVIEWED_LABEL_NOT_RECORDED"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--write", action="store_true", help="Persist both records.")
    args = parser.parse_args()

    required = (OUTPUTS_PATH, RESPONSES_PATH, MANIFEST_PATH, VALIDATION_PATH)
    for path in required:
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1

    # Hash every input BEFORE and AFTER, so "we preserved the translations" is a
    # checked claim rather than a promise in a docstring.
    hashes_before = {path.name: file_sha256(path) for path in required}

    rows = load_jsonl(OUTPUTS_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    log = ResponseLog(RESPONSES_PATH)
    effective = log.effective()

    non_english = [r for r in rows if r["language"] != PIVOT_LANGUAGE]
    by_language = Counter(r["language"] for r in rows)
    behavior_ids = sorted({r["behavior_id"] for r in rows})

    # --- the reviewed scope must be the corpus that actually exists ---------
    problems: list[str] = []
    if len(rows) != REVIEW_AS_REPORTED["n_rows_reviewed"]:
        problems.append(
            f"{len(rows)} rows on disk, review reports "
            f"{REVIEW_AS_REPORTED['n_rows_reviewed']}"
        )
    if len(behavior_ids) != REVIEW_AS_REPORTED["n_behaviors_reviewed"]:
        problems.append(
            f"{len(behavior_ids)} behaviors on disk, review reports "
            f"{REVIEW_AS_REPORTED['n_behaviors_reviewed']}"
        )
    if len(non_english) != REVIEW_AS_REPORTED["n_non_english_translations_reviewed"]:
        problems.append(
            f"{len(non_english)} non-English rows on disk, review reports "
            f"{REVIEW_AS_REPORTED['n_non_english_translations_reviewed']}"
        )
    if sorted(by_language) != sorted(LANGUAGE_CODES):
        problems.append(f"languages on disk {sorted(by_language)} != {sorted(LANGUAGE_CODES)}")
    if manifest.get("counts_by_language") != dict(sorted(by_language.items())):
        problems.append("pilot_manifest_v2 counts_by_language disagrees with the outputs file")
    if validation.get("n_rows") != len(rows):
        problems.append("pilot_validation_v2 n_rows disagrees with the outputs file")

    # An empty translation is not a reviewable row; a "clean" outcome over one
    # would be describing text that is not there.
    empty = [r for r in non_english if not (r.get("translated_text") or "").strip()]
    if empty:
        problems.append(f"{len(empty)} non-English row(s) have empty translated_text")

    missing_response = [
        r for r in non_english
        if effective.get(make_request_id(r["behavior_id"], r["language"]), {}).get("status")
        != STATUS_OK
    ]
    if missing_response:
        problems.append(f"{len(missing_response)} non-English row(s) have no effective OK response")

    # A non-English row's prompt_hash IS the hash of its translated_text, so it
    # must equal the response log's response_sha256. Any drift between the
    # translation on disk and the translation as generated shows up here -- this
    # is what evidences that the machine translations are preserved untouched.
    hash_matched = sum(
        1 for r in non_english
        if r["prompt_hash"]
        == effective.get(make_request_id(r["behavior_id"], r["language"]), {}).get(
            "response_sha256"
        )
    )
    if hash_matched != len(non_english):
        problems.append(
            f"{len(non_english) - hash_matched} non-English row(s) have a translation "
            "that does not match the response log"
        )

    # Refuse to record a human review on top of an existing review claim.
    preexisting = [r for r in rows if r.get("reviewer_id") or r["translation_status"] == "HUMAN_REVIEWED"]
    if preexisting:
        problems.append(f"{len(preexisting)} row(s) already carry a reviewer claim")

    if problems:
        for problem in problems:
            print(f"  FAIL: {problem}", file=sys.stderr)
        print("\nrefusing to record a review whose scope does not match the corpus",
              file=sys.stderr)
        return 1

    # --- open heuristic flags carried into the review ----------------------
    flagged = [r for r in validation["rows"] if r.get("issues")]
    flags_by_language: dict[str, dict[str, int]] = {}
    for row in flagged:
        bucket = flags_by_language.setdefault(row["language"], {})
        for issue in row["issues"]:
            bucket[issue] = bucket.get(issue, 0) + 1
    flagged_request_ids = sorted(
        make_request_id(r["behavior_id"], r["language"]) for r in flagged
    )

    # --- the v2 semantic_validation gate, criterion by criterion ------------
    # configs/dataset_v2_rebuild.yaml, semantic_validation. Evaluated honestly:
    # an aggregate estimate cannot satisfy a row-level criterion.
    gate = {
        "config": "configs/dataset_v2_rebuild.yaml:semantic_validation",
        "gate_satisfied": False,
        "gate_scope_note": (
            "This gate governs the FULL v2 corpus before freeze (200 behaviors, "
            "1,400 rows). This review covers the 20-behavior pilot only, so the "
            "gate is not addressed even where a criterion is met."
        ),
        "criteria": {
            "annotators_per_language_2": {
                "status": "met_as_reported",
                "detail": "two independent human reviewers, all seven languages",
            },
            "machine_or_llm_fallback_false": {
                "status": "met_as_reported",
                "detail": "no machine or LLM stand-in was used",
            },
            "adjudicator_required_for_disagreement": {
                "status": "cannot_evaluate",
                "detail": (
                    "disagreement is defined row-by-row; with no row-level "
                    "decisions there is no disagreement set to adjudicate"
                ),
            },
            "min_kappa_0.60": {
                "status": "cannot_evaluate",
                "detail": "kappa requires paired per-row labels; none were recorded",
            },
            "min_preserved_fraction_0.80": {
                "status": "cannot_evaluate",
                "detail": (
                    "0.97 is a reviewer estimate, not a measured preserved fraction "
                    "over labelled rows, and is not comparable to this threshold"
                ),
            },
            "min_preserved_rows_per_arm_40": {
                "status": "cannot_evaluate",
                "detail": "per-arm preserved counts require row-level labels",
            },
            "gating_labels_recorded": {
                "status": "not_met",
                "detail": (
                    "meaning_preservation, safety_label_preservation and "
                    "qualifier_preservation were not recorded per row"
                ),
            },
        },
    }

    limits = {
        "row_level_decisions_recorded": ROW_LEVEL_DECISIONS_RECORDED,
        "per_row_reviewer_attribution": False,
        "reviewer_ids_recorded": False,
        "reviewer_language_competence_recorded": False,
        "reviewed_text_hashes_recorded_per_row": False,
        "disagreements_recorded": False,
        "review_decision_schema_representable": False,
        "review_decision_schema_note": (
            "multilingual_safety.review.ReviewDecision cannot represent this "
            "review: it requires a named reviewer_id, a competence, and one of "
            "five decision values per row. None exist here, so this is a "
            "standalone record and not an entry in the review-decision layer. "
            f"The per-row state is recorded as {ROW_STATE_LABEL!r}, deliberately "
            "outside that vocabulary: PENDING would say the rows were never "
            "inspected, and ACCEPT would assert a verdict nobody recorded."
        ),
        "clears_review_required": False,
        "clears_review_required_rationale": competence_rationale(ReviewerCompetence.NONE),
        "native_speaker_validation_claimed": False,
        "row_level_native_speaker_claim_permitted": False,
        "estimated_fidelity_note": (
            "0.97 is a reviewer estimate over the whole pilot, not a tally: "
            "0.97 x 120 non-English rows is 116.4, which is not a row count. It "
            "cannot be attributed to specific rows, cannot be recomputed, and "
            "does not identify which rows the residual 3% refers to."
        ),
        "heuristic_flags_resolved_by_this_review": False,
        "heuristic_flags_note": (
            f"{len(flagged)} non-English row(s) carried open heuristic flags at "
            "review time. With no row-level labels, the aggregate 'clean' outcome "
            "cannot be attributed to these rows, so every flag stays open."
        ),
    }

    header = stage_report_header(
        "stage1.4_record_pilot_human_review_v2",
        {
            "dataset_version": "v2",
            "review_scope": "pilot",
            "row_level_decisions_recorded": ROW_LEVEL_DECISIONS_RECORDED,
        },
    )

    # --- provenance of the reviewed corpus, recorded as found --------------
    recorded_seed = manifest.get("generation_seed")
    corpus_state = {
        "translation_model": manifest["backend"]["probe"]["model_id"],
        "translation_model_revision": manifest["backend"]["probe"]["model_revision"],
        "effective_provider": manifest["backend"]["effective_provider"],
        "translation_config_path_recorded": manifest.get("config_path"),
        "generation_seed_recorded": recorded_seed,
        "generation_seed_declared_by_protocol": 20260818,
        "generation_seed_matches_protocol": recorded_seed == 20260818,
        "manifest_digest": manifest.get("manifest_digest"),
        "counts_by_status": manifest.get("counts_by_status"),
        "counts_by_harm_label": manifest.get("counts_by_harm_label"),
        "open_heuristic_flags": len(flagged),
        "open_heuristic_flags_by_language": dict(sorted(flags_by_language.items())),
        "open_heuristic_flag_request_ids": flagged_request_ids,
        "response_log": log.retry_summary(),
    }

    observations: list[str] = []
    if not corpus_state["generation_seed_matches_protocol"]:
        observations.append(
            f"The reviewed corpus records generation seed {recorded_seed} and "
            f"config_path {manifest.get('config_path')!r}, while "
            "configs/dataset_v2_rebuild.yaml declares seed 20260818 (and "
            "configs/translation_v2.yaml carries it). Generation is beam search "
            "with do_sample=false, so the seed does not change the output text, "
            "but the recorded provenance of these rows names v1's config and "
            "seed. Recorded here as found; not changed by this stage."
        )
    observations.append(
        "The review outcome is aggregate. It is sufficient to authorise the next "
        "pipeline stage and insufficient to support any per-row, per-language, or "
        "per-arm fidelity claim."
    )
    observations.append(
        "11 of the 14 open heuristic flags are MISSING_EXPECTED_DIACRITICS in ha, "
        "the lowest-resource language in the pilot. Heuristic flags are not "
        "fidelity judgements, but they remain open and concentrated."
    )

    common = {
        **header,
        "artifact_type": "pilot_human_review_record",
        "schema_version": "pilot-human-review.v2",
        "dataset_version": "v2",
        "review_scope": "pilot",
        # --- the requested status fields, verbatim -------------------------
        "human_reviewed": HUMAN_REVIEWED,
        "row_level_decisions_recorded": ROW_LEVEL_DECISIONS_RECORDED,
        "estimated_fidelity": ESTIMATED_FIDELITY,
        "formal_kappa": FORMAL_KAPPA,
        "formal_adjudication": FORMAL_ADJUDICATION,
        # --- what was reported, and what this repo could check -------------
        "review_as_reported": dict(sorted(REVIEW_AS_REPORTED.items())),
        "review_as_reported_is_operator_supplied": True,
        "verified_against_artifacts": {
            "n_rows": len(rows),
            "n_behaviors": len(behavior_ids),
            "n_non_english_rows": len(non_english),
            "counts_by_language": dict(sorted(by_language.items())),
            "n_rows_with_empty_translation": 0,
            "n_non_english_rows_without_ok_response": 0,
            "n_non_english_rows_matching_response_log": hash_matched,
            "translations_byte_identical_to_response_log": True,
            "scope_matches_reported_review": True,
            "note": (
                "Row counts, behavior count, language coverage and translation "
                "presence were re-derived from the artifacts. The reviewers' "
                "identities, independence, and fidelity estimate are reported, "
                "not verifiable from this repository."
            ),
        },
        "limits": dict(sorted(limits.items())),
        "semantic_validation_gate": gate,
        "corpus_state_at_review": corpus_state,
        "row_state_unchanged": {
            "translated_text_modified": False,
            "translation_status_counts": dict(
                sorted(Counter(r["translation_status"] for r in rows).items())
            ),
            "review_required_true": sum(1 for r in rows if r.get("review_required")),
            "review_required_cleared": 0,
            "reviewer_ids_written": 0,
            "human_reviewed_rows_written": 0,
        },
        "authorizes": ["proceed_to_full_translation"],
        "does_not_authorize": [
            "clearing_review_required",
            "setting_translation_status_HUMAN_REVIEWED",
            "row_level_native_speaker_validation_claim",
            "per_language_or_per_arm_fidelity_claim",
            "freeze_of_dataset_v2",
        ],
        "decisions_jsonl_written": False,
        "decisions_jsonl_note": (
            f"{DECISIONS_JSONL.relative_to(REPO_ROOT)} is deliberately not written. "
            "Stages 1.3 and 1.5 read it as a per-row decision layer; an aggregate "
            "record placed there would be read as row-level verdicts that do not exist."
        ),
        "observations": observations,
        "split_touched": None,
        "split_note": "Pilot rows are pre-split (split_placeholder=UNASSIGNED). No test read.",
        "contains_prompt_text": False,
        "source_hashes": {
            **hashes_before,
            "pilot_wide_review_v2.csv": (
                file_sha256(WIDE_REVIEW_CSV) if WIDE_REVIEW_CSV.exists() else None
            ),
            "dataset_v2_rebuild.yaml": (
                file_sha256(PROTOCOL_PATH) if PROTOCOL_PATH.exists() else None
            ),
        },
    }

    # --- private record: full row scope, zero decision labels --------------
    private_rows = []
    for row in sorted(rows, key=lambda r: (r["language"], r["behavior_id"])):
        request_id = make_request_id(row["behavior_id"], row["language"])
        response = effective.get(request_id, {})
        is_pivot = row["language"] == PIVOT_LANGUAGE
        private_rows.append({
            "behavior_id": row["behavior_id"],
            "language": row["language"],
            "request_id": None if is_pivot else request_id,
            "prompt_hash": row["prompt_hash"],
            "response_sha256": None if is_pivot else response.get("response_sha256"),
            "in_review_scope": True,
            "reviewed_as_translation": not is_pivot,
            "row_state": "CANONICAL_SOURCE" if is_pivot else ROW_STATE_LABEL,
            "decision_recorded": False,
            "decision": None,
            "reviewer_id": None,
            "reviewer_language_competence": None,
            "decided_at": None,
            "heuristic_issues": next(
                (v["issues"] for v in validation["rows"]
                 if v["behavior_id"] == row["behavior_id"] and v["language"] == row["language"]),
                [],
            ),
            "translation_status_unchanged": row["translation_status"],
            "review_required_unchanged": row["review_required"],
        })

    private = {
        **common,
        "record_visibility": "private",
        "git_ignored": True,
        "rows": private_rows,
        "rows_note": (
            "All 140 rows are enumerated with decision_recorded=false. The "
            "enumeration exists so that no later reader can infer a per-row "
            "label from the aggregate outcome. No decision was invented."
        ),
        "n_rows_enumerated": len(private_rows),
        "n_decisions_recorded": 0,
    }
    private["deterministic_digest"] = deterministic_digest(private)

    tracked = {
        **common,
        "record_visibility": "tracked",
        "git_ignored": False,
        "n_decisions_recorded": 0,
        "private_record": {
            "path": str(PRIVATE_OUT.relative_to(REPO_ROOT)),
            "git_ignored": True,
            "sha256": json_sha256(private),
            "deterministic_digest": private["deterministic_digest"],
            "n_rows_enumerated": len(private_rows),
            "contains_prompt_text": False,
        },
    }
    tracked["deterministic_digest"] = deterministic_digest(tracked)

    # --- inputs must be byte-identical to how we found them ----------------
    hashes_after = {path.name: file_sha256(path) for path in required}
    if hashes_after != hashes_before:
        changed = [k for k in hashes_before if hashes_before[k] != hashes_after[k]]
        print(f"input files changed during the run ({changed}); refusing to write",
              file=sys.stderr)
        return 1

    # --- report ------------------------------------------------------------
    print("PILOT HUMAN REVIEW -- METADATA-ONLY RECORD (dataset v2)")
    print(f"run_id:     {header['run_id']}")
    print(f"git commit: {header['git_commit']}  (dirty={header['git_dirty']})")
    print("\nRECORDED STATUS")
    for key in ("human_reviewed", "row_level_decisions_recorded", "estimated_fidelity",
                "formal_kappa", "formal_adjudication"):
        print(f"  {key:<42}: {tracked[key]}")
    print("\nSCOPE (re-derived from artifacts)")
    verified = tracked["verified_against_artifacts"]
    print(f"  rows                                      : {verified['n_rows']}")
    print(f"  behaviors                                 : {verified['n_behaviors']}")
    print(f"  non-English translations                  : {verified['n_non_english_rows']}")
    print(f"  by language                               : {verified['counts_by_language']}")
    print(f"  scope matches reported review             : {verified['scope_matches_reported_review']}")
    print("\nLIMITS")
    for key in ("row_level_decisions_recorded", "reviewer_ids_recorded",
                "reviewer_language_competence_recorded", "review_decision_schema_representable",
                "clears_review_required", "native_speaker_validation_claimed",
                "heuristic_flags_resolved_by_this_review"):
        print(f"  {key:<42}: {limits[key]}")
    print(f"\nSEMANTIC VALIDATION GATE satisfied          : {gate['gate_satisfied']}")
    for name, criterion in gate["criteria"].items():
        print(f"  {name:<42}: {criterion['status']}")
    print("\nROW STATE (unchanged)")
    for key, value in tracked["row_state_unchanged"].items():
        print(f"  {key:<42}: {value}")
    print(f"\n  open heuristic flags                      : {corpus_state['open_heuristic_flags']}"
          f"  {corpus_state['open_heuristic_flags_by_language']}")
    print(f"  input sha256 unchanged                    : {hashes_after == hashes_before}")
    print("\nOBSERVATIONS")
    for note in observations:
        print(f"  - {note}")

    if args.write:
        PRIVATE_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        V2_MANIFESTS.mkdir(parents=True, exist_ok=True)
        PRIVATE_OUT.write_text(
            json.dumps(private, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        TRACKED_OUT.write_text(
            json.dumps(tracked, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {PRIVATE_OUT.relative_to(REPO_ROOT)}  (git-ignored, metadata-only)")
        print(f"wrote {TRACKED_OUT.relative_to(REPO_ROOT)}  (tracked, metadata-only)")
        print("\nNo translated_text was modified. No row was marked HUMAN_REVIEWED. "
              "review_required is unchanged.")
    else:
        print("\n(nothing written; pass --write to persist)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
