#!/usr/bin/env python3
"""Stage 1.1 -- build translation requests and the pilot output rows.

Backend policy, enforced rather than documented
-----------------------------------------------
The script first resolves which translation provider is actually configured
(``configs/translation.yaml`` plus the environment). **If no safe provider is
configured it produces requests and a manual-review queue, and writes rows with
``translated_text: null`` and status ``PENDING_TRANSLATION``.** It never
fabricates translations, and no placeholder text is ever written into a text
field.

An LLM backend requires an explicit recorded opt-in and must not be a model
under evaluation -- using a system under test to produce or judge its own
evaluation inputs would confound every result derived from them.

English is not translated. English rows carry the canonical source text verbatim
with status ``CANONICAL_SOURCE``.

Every non-English row is ``review_required=true`` on creation, and no machine
path in this repository can clear it (see
``multilingual_safety.translation``).

Outputs
-------
    data/interim/translation_pilot_requests.jsonl   requests (git-ignored)
    data/interim/translation_pilot_outputs.jsonl    rows (git-ignored)
    data/interim/translation_review_queue.md        human-readable queue
    data/manifests/translation_pilot_manifest.json  TEXT-FREE, tracked
    results/translation_pilot_inventory.json        counts, text-free

Usage
-----
    python scripts/04_translate_pilot.py --dry-run
    python scripts/04_translate_pilot.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from multilingual_safety.hashing import deterministic_digest
from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE, load_languages
from multilingual_safety.paths import (
    CONFIGS_DIR,
    INTERIM_DIR,
    LANGUAGES_CONFIG,
    MANIFESTS_DIR,
    REPO_ROOT,
    RESULTS_DIR,
)
from multilingual_safety.providers import (
    STATUS_OK,
    ProviderError,
    ResponseLog,
    TranslationRequest,
    correlate,
    extract_seed,
    get_provider,
    make_request_id,
    probe_provider,
)
from multilingual_safety.run_context import stage_report_header
from multilingual_safety.translation import (
    SPLIT_UNASSIGNED,
    TranslationMethod,
    TranslationRow,
    TranslationStatus,
    compute_prompt_hash,
    make_english_row,
    make_pending_row,
    validate_row_group,
)

TRANSLATION_CONFIG = CONFIGS_DIR / "translation.yaml"
SELECTION_PATH = INTERIM_DIR / "translation_pilot_selection.jsonl"
REQUESTS_PATH = INTERIM_DIR / "translation_pilot_requests.jsonl"
OUTPUTS_PATH = INTERIM_DIR / "translation_pilot_outputs.jsonl"
RESPONSES_PATH = INTERIM_DIR / "translation_pilot_responses.jsonl"
QUEUE_PATH = INTERIM_DIR / "translation_review_queue.md"
MANIFEST_PATH = MANIFESTS_DIR / "translation_pilot_manifest.json"
INVENTORY_PATH = RESULTS_DIR / "translation_pilot_inventory.json"

NON_PIVOT_LANGUAGES = tuple(c for c in LANGUAGE_CODES if c != PIVOT_LANGUAGE)


def select_pilot_subset(rows: list[TranslationRow], n_per_language: int) -> list[TranslationRow]:
    """Pick exactly ``n_per_language`` behaviors for each non-English language.

    Deterministic and arm-balanced. Behaviors are ordered by ``behavior_id``
    within each arm, then drawn alternately starting with ``harmful`` -- so
    n=3 yields 2 harmful + 1 benign rather than the 3 benign a naive sort would
    give (``benign`` sorts before ``harmful``). The same behaviors are used for
    every language, so per-language results stay comparable.

    English rows are never included: they are canonical source text.
    """
    if n_per_language < 0:
        raise ValueError("n_per_language must be non-negative")

    by_arm: dict[str, list[str]] = {"harmful": [], "benign": []}
    for row in rows:
        if row.language != PIVOT_LANGUAGE:
            continue
        if row.behavior_id not in by_arm[row.harm_label]:
            by_arm[row.harm_label].append(row.behavior_id)
    for arm in by_arm:
        by_arm[arm].sort()

    chosen: list[str] = []
    for index in range(max(len(by_arm["harmful"]), len(by_arm["benign"]))):
        for arm in ("harmful", "benign"):          # harmful first, deliberately
            if index < len(by_arm[arm]) and len(chosen) < n_per_language:
                chosen.append(by_arm[arm][index])
        if len(chosen) >= n_per_language:
            break

    selected = [
        row for row in rows if row.behavior_id in set(chosen) and row.language != PIVOT_LANGUAGE
    ]
    selected.sort(key=lambda r: (r.behavior_id, r.language))
    return selected


# ---------------------------------------------------------------------------
def resolve_backend(config: dict[str, Any], override: str | None = None) -> dict[str, Any]:
    """Resolve the effective translation provider. Credential values are never read.

    ``configured_provider`` always reports what ``configs/translation.yaml`` says
    -- it is NOT rewritten by a CLI override. ``effective_provider`` reports what
    would actually be used. Keeping them separate means a run whose provider came
    from the command line is visibly distinguishable from one whose provider was
    committed to the config, which matters for reproducing a run later.

    ``backend_usable`` comes from a cheap probe (no weights loaded) and is a
    prediction, not a guarantee. The authority is
    :meth:`providers.TranslationProvider.preflight`, which still fails closed at
    translation time. There is no fallback at any point.
    """
    configured = config.get("provider")
    effective = override or configured
    allowed = list(config.get("allowed_providers") or [])
    providers = config.get("providers") or {}
    llm_cfg = config.get("llm_backend") or {}

    resolution: dict[str, Any] = {
        "configured_provider": configured,
        "effective_provider": effective,
        "provider_source": (
            "cli_override" if override else ("config" if configured else "none")
        ),
        "backend_usable": False,
        "allowed_providers": sorted(allowed),
        "llm_backend_enabled": bool(llm_cfg.get("enabled")),
        "usable": False,          # retained alias of backend_usable
        "reason": None,
        "credential_env_names_checked": [],
        "credential_present": None,
    }

    # A CLI override takes a different path: the local provider registry decides,
    # and a cheap probe reports availability without loading any weights.
    if override:
        probe = probe_provider(override, config)
        resolution["probe"] = probe
        resolution["backend_usable"] = bool(probe.get("usable"))
        resolution["usable"] = resolution["backend_usable"]
        resolution["reason"] = probe.get("reason")
        resolution["unsupported_languages"] = sorted(
            set(NON_PIVOT_LANGUAGES)
            - set((providers.get(override) or {}).get("supported_languages") or [])
        )
        seen_env: dict[str, bool] = {}
        for _name, spec in sorted(providers.items()):
            for key in ("api_key_env", "credentials_env"):
                env_name = (spec or {}).get(key)
                if env_name:
                    seen_env[env_name] = bool(os.environ.get(env_name))
        resolution["provider_credentials_present_by_name"] = dict(sorted(seen_env.items()))
        return resolution

    # No override: the config alone decides. `provider` here is the configured
    # value, so `configured_provider` and `effective_provider` are identical.
    provider = configured

    # What credentials exist, by NAME only -- so the report shows what was
    # available and deliberately not used.
    seen_env: dict[str, bool] = {}
    for _name, spec in sorted(providers.items()):
        for key in ("api_key_env", "credentials_env"):
            env_name = (spec or {}).get(key)
            if env_name:
                seen_env[env_name] = bool(os.environ.get(env_name))
    resolution["provider_credentials_present_by_name"] = dict(sorted(seen_env.items()))

    if provider is None:
        resolution["reason"] = (
            "configs/translation.yaml declares provider: null -- no translation backend "
            "is configured. Emitting requests and a manual-review queue instead."
        )
        return resolution

    if provider not in allowed:
        if provider == "llm" and not llm_cfg.get("enabled"):
            resolution["reason"] = (
                "an LLM backend requires an explicit recorded opt-in "
                "(llm_backend.enabled) and must not be a model under evaluation"
            )
        else:
            resolution["reason"] = f"provider {provider!r} is not in allowed_providers"
        return resolution

    spec = providers.get(provider) or {}
    env_name = spec.get("api_key_env") or spec.get("credentials_env")
    resolution["credential_env_names_checked"] = [env_name] if env_name else []
    if env_name and not os.environ.get(env_name):
        resolution["credential_present"] = False
        resolution["reason"] = f"provider {provider!r} is configured but {env_name} is not set"
        return resolution

    resolution["credential_present"] = True if env_name else None
    unsupported = sorted(set(NON_PIVOT_LANGUAGES) - set(spec.get("supported_languages") or []))
    resolution["unsupported_languages"] = unsupported
    if unsupported:
        resolution["reason"] = (
            f"provider {provider!r} does not support {unsupported}; those rows would "
            "stay pending even if the rest were translated"
        )
    resolution["usable"] = True
    resolution["backend_usable"] = True
    return resolution


def build_request(row: dict[str, Any], language: str, constraints: dict[str, Any]) -> dict[str, Any]:
    """One translation request. Contains source text; stays git-ignored."""
    return {
        "request_id": f"{row['behavior_id']}:{language}",
        "behavior_id": row["behavior_id"],
        "source": row["source"],
        "source_item_id": row["source_item_id"],
        "category": row["category"],
        "harm_label": row["harm_label"],
        "source_language": PIVOT_LANGUAGE,
        "target_language": language,
        "source_text": row["source_text"],
        "source_text_sha256": row["source_text_sha256"],
        "constraints": dict(sorted(constraints.items())),
        "instructions": (
            f"Translate the source text from English into {language}. Render the meaning "
            "and register faithfully. Do NOT add operational detail, specifics, steps, or "
            "elaboration that the source does not contain; do not soften, sanitise, "
            "refuse, or comment. Preserve all orthography, including tone marks and "
            "hooked letters. Return the translation only, NFC-normalised."
        ),
        "expected_output_field": "translated_text",
        "review_required_on_return": True,
    }


def _provider_seed(translation_report: Mapping[str, Any]) -> int | None:
    """Seed the provider reported at preflight, or None if it did not run."""
    metadata = (translation_report or {}).get("provider_metadata") or {}
    return extract_seed(metadata.get("generation_config") or {})


def effective_seed(
    config: Mapping[str, Any], translation_report: Mapping[str, Any]
) -> int | None:
    """The seed these rows were actually produced under.

    Prefers what the provider reported at preflight, because that is the value
    ``torch.manual_seed`` received. Falls back to the configured seed for a run
    that translated nothing, so the manifest still records which seed the run
    was configured with. ``None`` when no seed is configured anywhere -- recorded
    as null rather than invented.

    ``extract_seed`` does the lookup so nesting rules live in one place
    (``generation_config``/``model_kwargs``/``generation`` are all searched).
    """
    from_provider = _provider_seed(translation_report)
    if from_provider is not None:
        return from_provider
    return extract_seed(config.get("nllb_local_settings") or {})


def run_translation(
    *,
    subset: list[TranslationRow],
    config: dict[str, Any],
    provider_name: str | None,
    run_id: str,
    responses_path: Path,
    force: bool,
) -> dict[str, Any]:
    """Translate the pilot subset with an explicitly named local provider.

    The response log is opened and consistency-checked BEFORE any generation, so
    a provider/model mismatch or an unforced overwrite fails without having
    produced anything.
    """
    provider = get_provider(provider_name, config)   # raises; never falls back
    print(f"\nTRANSLATION  provider={provider.name}  force={force}")
    print("  preflight ...")
    metadata = provider.preflight()                  # raises if unavailable
    for key in sorted(metadata):
        print(f"    {key}: {metadata[key]}")

    log = ResponseLog(responses_path)
    log.check_provider_consistency(provider.name, metadata["model_id"])

    requests = [
        TranslationRequest.create(
            behavior_id=row.behavior_id,
            target_language=row.language,
            source_text=row.source_text,
        )
        for row in subset
    ]
    requests.sort(key=lambda r: r.request_id)

    todo = log.pending(requests, force=force)
    skipped = len(requests) - len(todo)
    if skipped:
        print(f"  {skipped} request(s) already have a successful response; skipping "
              "(pass --force to redo)")

    responses = []
    for index, request in enumerate(todo, start=1):
        response = provider.translate(request, run_id=run_id)
        responses.append(response)
        flag = "ok " if response.status == STATUS_OK else "ERR"
        print(f"    [{index:>2}/{len(todo)}] {flag} {request.request_id} "
              f"{response.latency_seconds:.2f}s sha={(response.response_sha256 or '')[:12]}")

    log.append(responses, force=force)               # raises without --force
    correlation = correlate(requests, responses)

    return {
        "attempted": True,
        "provider": provider.name,
        "provider_metadata": metadata,
        "n_requests": len(requests),
        "n_attempted": len(todo),
        "n_skipped_already_done": skipped,
        "n_responses": len(responses),
        "n_ok": sum(1 for r in responses if r.status == STATUS_OK),
        "n_error": sum(1 for r in responses if r.status != STATUS_OK),
        "correlation": correlation,
        "response_log": str(responses_path),
        # nllb_local runs entirely on local weights.
        "network_translation_api_called": False,
    }


def apply_responses(rows: list[TranslationRow], responses_path: Path) -> list[TranslationRow]:
    """Fold successful responses into the pilot rows.

    Rows become ``MACHINE_TRANSLATED_UNCHECKED`` with
    ``translation_method=machine_unreviewed``. **``review_required`` stays True
    and no row is ever set to HUMAN_REVIEWED here** -- only a named human
    reviewer may do that (``multilingual_safety.translation``).
    """
    if not responses_path.exists():
        return rows
    # effective(), not existing(): the latest SUCCESSFUL attempt wins, so a
    # later transient ERROR cannot retract a translation already produced.
    log = ResponseLog(responses_path).effective()
    updated: list[TranslationRow] = []

    for row in rows:
        record = log.get(make_request_id(row.behavior_id, row.language))
        if (
            row.language == PIVOT_LANGUAGE
            or record is None
            or record.get("status") != STATUS_OK
            or not record.get("response_text")
        ):
            updated.append(row)
            continue
        text = record["response_text"]
        updated.append(
            TranslationRow(
                behavior_id=row.behavior_id,
                source=row.source,
                category=row.category,
                harm_label=row.harm_label,
                source_text=row.source_text,
                language=row.language,
                translated_text=text,
                translation_method=TranslationMethod.MACHINE_UNREVIEWED.value,
                translation_model=record.get("model_id"),
                translation_timestamp=record.get("timestamp"),
                translation_status=TranslationStatus.MACHINE_TRANSLATED_UNCHECKED.value,
                review_required=True,          # never cleared by a machine
                prompt_hash=compute_prompt_hash(text),
                source_item_id=row.source_item_id,
                reviewer_id=None,              # never set here
                issues=row.issues,
            )
        )
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", type=Path, default=SELECTION_PATH)
    parser.add_argument("--config", type=Path, default=TRANSLATION_CONFIG)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate translation REQUESTS only. No rows, no provider, no translation.",
    )
    parser.add_argument(
        "--translate", action="store_true",
        help="Actually translate the pilot subset. Requires --provider.",
    )
    parser.add_argument(
        "--provider", default=None,
        help="Explicit provider id (e.g. nllb_local). No default and no fallback.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Behaviors per target language (default: configs/translation.yaml pilot).",
    )
    parser.add_argument(
        "--require-full-selection", action="store_true",
        help="Fail if the effective limit would translate fewer behaviors than the "
             "selection file contains. Off by default: the v1 pilot subsamples "
             "deliberately (behaviors_per_language: 3).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Permit overwriting existing successful responses. Off by default.",
    )
    parser.add_argument("--responses", type=Path, default=RESPONSES_PATH)
    # Output destinations. Defaults are the v1 pilot paths, so an unflagged
    # invocation behaves exactly as before. They are overridable because the
    # dataset-v2 pilot must write under data/v2/ WITHOUT overwriting the tracked
    # v1 artifacts (data/manifests/translation_pilot_manifest.json and
    # results/translation_pilot_inventory.json), which are frozen history.
    parser.add_argument("--requests", type=Path, default=REQUESTS_PATH)
    parser.add_argument("--outputs", type=Path, default=OUTPUTS_PATH)
    parser.add_argument("--queue", type=Path, default=QUEUE_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--inventory", type=Path, default=INVENTORY_PATH)
    args = parser.parse_args()

    if not args.selection.exists():
        print(f"missing {args.selection}; run scripts/03_select_pilot_behaviors.py first",
              file=sys.stderr)
        return 1

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    constraints = config.get("request_constraints") or {}
    backend = resolve_backend(config, override=args.provider)

    header = stage_report_header(
        "stage1.1_translate_pilot",
        {
            "configured_provider": backend["configured_provider"],
            "effective_provider": backend["effective_provider"],
            "languages": list(LANGUAGE_CODES),
        },
    )
    print(f"run_id:     {header['run_id']}")
    print(f"git commit: {header['git_commit']}  (dirty={header['git_dirty']})")

    print("\nBACKEND RESOLUTION")
    print(f"  configured_provider : {backend['configured_provider']}")
    print(f"  effective_provider  : {backend['effective_provider']}")
    print(f"  backend_usable      : {backend['backend_usable']}")
    print(f"  provider_source     : {backend['provider_source']}")
    print(f"  reason              : {backend['reason']}")
    print(f"  provider credentials present (by name): "
          f"{backend['provider_credentials_present_by_name'] or '{}'}")
    if not backend["backend_usable"]:
        if backend["effective_provider"]:
            print(f"\n  => Provider {backend['effective_provider']!r} is NOT usable. Failing closed:")
            print("     no fallback provider will be selected and nothing will be translated.")
        else:
            print("\n  => No translation backend selected. Producing REQUESTS and a MANUAL")
            print("     REVIEW QUEUE. Rows stay PENDING_TRANSLATION; nothing is invented.")

    load_languages(LANGUAGES_CONFIG)  # validates the seven-language registry
    selection = [
        json.loads(line)
        for line in args.selection.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selection.sort(key=lambda r: (r["harm_label"], r["category"], r["behavior_id"]))

    requests: list[dict[str, Any]] = []
    rows: list[TranslationRow] = []

    for item in selection:
        common = {
            "behavior_id": item["behavior_id"],
            "source": item["source"],
            "category": item["category"],
            "harm_label": item["harm_label"],
            "source_text": item["source_text"],
            "source_item_id": item["source_item_id"],
        }
        rows.append(make_english_row(**common))
        for language in NON_PIVOT_LANGUAGES:
            requests.append(build_request(item, language, constraints))
            rows.append(make_pending_row(language=language, **common))

    # Deterministic ordering everywhere.
    rows.sort(key=lambda r: (r.harm_label, r.category, r.behavior_id, r.language))
    requests.sort(key=lambda r: (r["harm_label"], r["category"], r["behavior_id"], r["target_language"]))

    # --- group invariants -------------------------------------------------
    by_behavior: dict[str, list[TranslationRow]] = {}
    for row in rows:
        by_behavior.setdefault(row.behavior_id, []).append(row)
    group_violations = {
        bid: v for bid, group in sorted(by_behavior.items()) if (v := validate_row_group(group))
    }

    # --- pilot subset (always reported; translated only with --translate) ---
    pilot_cfg = config.get("pilot") or {}
    limit = args.limit if args.limit is not None else int(pilot_cfg.get("behaviors_per_language", 3))
    limit_source = "cli" if args.limit is not None else (
        "config" if pilot_cfg.get("behaviors_per_language") is not None else "builtin_default"
    )
    subset = select_pilot_subset(rows, limit)

    # Coverage is REPORTED unconditionally, whatever the limit. A subsample is a
    # legitimate choice (v1 takes 3 of 20 on purpose) but it must never be a
    # silent one: the v2 pilot lost 102 of 120 translations to an inherited
    # limit of 3 that nothing in the output made obvious.
    coverage = {
        "behaviors_per_language": limit,
        "limit_source": limit_source,
        "n_behaviors_in_selection": len(by_behavior),
        "n_behaviors_in_subset": len({r.behavior_id for r in subset}),
        "n_behaviors_untranslated": len(by_behavior) - len({r.behavior_id for r in subset}),
        "n_subset_rows": len(subset),
        "n_non_pivot_languages": len(NON_PIVOT_LANGUAGES),
        "full_selection_covered": len({r.behavior_id for r in subset}) == len(by_behavior),
        "require_full_selection": bool(args.require_full_selection),
    }

    print("\nPILOT SUBSET")
    print(f"  behaviors/language: {limit} (from {limit_source})   "
          f"non-English languages: {len(NON_PIVOT_LANGUAGES)}   rows: {len(subset)}")
    print(f"  selection file holds {coverage['n_behaviors_in_selection']} behavior(s); "
          f"this run covers {coverage['n_behaviors_in_subset']}")
    if not coverage["full_selection_covered"]:
        print(f"  SUBSAMPLE: {coverage['n_behaviors_untranslated']} behavior(s) of the "
              f"selection will NOT be translated by this run "
              f"({coverage['n_behaviors_untranslated'] * len(NON_PIVOT_LANGUAGES)} rows left "
              f"pending). Raise --limit or pilot.behaviors_per_language to cover them.")
    for row in subset:
        print(f"    {row.behavior_id}  {row.language}  {row.harm_label:8} {row.category}")
    if any(r.language == PIVOT_LANGUAGE for r in subset):
        print("  FAIL: English row present in the translation subset", file=sys.stderr)
        return 1
    if args.require_full_selection and not coverage["full_selection_covered"]:
        print(f"\nSTOP -- --require-full-selection: effective limit {limit} covers "
              f"{coverage['n_behaviors_in_subset']} of {coverage['n_behaviors_in_selection']} "
              f"behaviors in {args.selection}. Nothing was translated and nothing was "
              f"written. Set --limit {coverage['n_behaviors_in_selection']} (or "
              f"pilot.behaviors_per_language) to translate the whole selection.",
              file=sys.stderr)
        return 1

    # --- optional translation ---------------------------------------------
    translation_report: dict[str, Any] = {
        "attempted": False,
        "provider": None,
        "n_requests": 0,
        "n_responses": 0,
        "network_translation_api_called": False,
    }
    if args.dry_run:
        print("\nDry run: requests only. No rows written, no provider, no translation.")
    elif args.translate:
        try:
            translation_report = run_translation(
                subset=subset,
                config=config,
                provider_name=args.provider or config.get("provider"),
                run_id=header["run_id"],
                responses_path=args.responses,
                force=args.force,
            )
        except ProviderError as exc:
            print(f"\nSTOP -- provider failure:\n  {exc}", file=sys.stderr)
            return 3
    else:
        print("\nNo --translate flag: no new translation will be requested.")

    # Rows are ALWAYS derived from the effective response log, whether or not
    # this invocation translated anything. Applying only under --translate meant
    # a later plain run rebuilt every row as PENDING and silently discarded
    # translations that were already produced and recorded (CLAUDE.md §2).
    if not args.dry_run:
        rows = apply_responses(rows, args.responses)
        rows.sort(key=lambda r: (r.harm_label, r.category, r.behavior_id, r.language))

    # --- counts -----------------------------------------------------------
    inventory = {
        **header,
        "backend": backend,
        "n_behaviors": len(by_behavior),
        "n_rows": len(rows),
        "n_requests": len(requests),
        "languages": list(LANGUAGE_CODES),
        "counts_by_language": dict(sorted(Counter(r.language for r in rows).items())),
        "counts_by_harm_label": dict(sorted(Counter(r.harm_label for r in rows).items())),
        "counts_by_category": dict(sorted(Counter(r.category for r in rows).items())),
        "counts_by_status": dict(sorted(Counter(r.translation_status for r in rows).items())),
        "counts_by_method": dict(sorted(Counter(r.translation_method for r in rows).items())),
        "counts_by_language_and_label": {
            f"{lang}:{label}": sum(
                1 for r in rows if r.language == lang and r.harm_label == label
            )
            for lang in LANGUAGE_CODES
            for label in ("harmful", "benign")
        },
        "n_review_required": sum(1 for r in rows if r.review_required),
        "n_translated": sum(1 for r in rows if r.translated_text is not None),
        "n_pending": sum(1 for r in rows if r.translated_text is None),
        "group_violations": group_violations,
        "split_placeholder": SPLIT_UNASSIGNED,
        # Derived from the append-only response log, not from this invocation:
        # a plain run must still report the translations that already exist.
        "response_log": ResponseLog(args.responses).retry_summary(),
        "native_speaker_reviewed_rows": 0,
        "pilot_subset": {
            **coverage,
            "n_rows": len(subset),
            "behavior_ids": sorted({r.behavior_id for r in subset}),
            "languages": sorted({r.language for r in subset}),
        },
        "translation": translation_report,
        "contains_prompt_text": False,
        "notes": [
            "English rows are canonical source text, not translations.",
            "Every non-English row is review_required=true; no machine can clear it.",
            "split_placeholder is a sentinel. Splits are assigned later at behavior "
            "level so no behavior can appear in both development and final evaluation.",
        ],
    }
    inventory["deterministic_digest"] = deterministic_digest(
        {k: v for k, v in inventory.items() if k not in {"backend"}}
    )

    print("\nCOUNTS")
    print(f"  behaviors: {inventory['n_behaviors']}   rows: {inventory['n_rows']}   "
          f"requests: {inventory['n_requests']}")
    print(f"  by language: {inventory['counts_by_language']}")
    print(f"  by label:    {inventory['counts_by_harm_label']}")
    print(f"  by status:   {inventory['counts_by_status']}")
    print(f"  review_required: {inventory['n_review_required']} / {inventory['n_rows']}")

    if group_violations:
        for bid, violations in sorted(group_violations.items()):
            print(f"  FAIL {bid}: {violations}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    # --- write ------------------------------------------------------------
    args.requests.parent.mkdir(parents=True, exist_ok=True)
    args.outputs.parent.mkdir(parents=True, exist_ok=True)
    with args.requests.open("w", encoding="utf-8") as fh:
        for request in requests:
            fh.write(json.dumps(request, sort_keys=True, ensure_ascii=False) + "\n")
    with args.outputs.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")

    # Text-free manifest (tracked).
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "stage": header["stage"],
        "run_id": header["run_id"],
        "git_commit": header["git_commit"],
        "git_dirty": header["git_dirty"],
        "generated_at": header["generated_at"],
        "source": "jailbreakbench",
        "harmbench_used": False,
        "backend": backend,
        # Which seed these rows were actually produced under. Previously the
        # seed reached only the inventory (nested in the provider metadata), so
        # a manifest could not be checked against the seed its dataset version
        # declares -- the v2 pilot recorded v1's 20260816 unnoticed.
        "generation_seed": effective_seed(config, translation_report),
        "generation_seed_source": (
            "provider_metadata" if _provider_seed(translation_report) is not None else "config"
        ),
        "config_path": display_path(args.config),
        # How much of the selection this run actually covered. In the manifest,
        # not just the inventory, so a dataset version's coverage can be audited
        # from the artifact that travels with it.
        "pilot_subset": {
            **coverage,
            "behavior_ids": sorted({r.behavior_id for r in subset}),
        },
        "languages": list(LANGUAGE_CODES),
        "pivot": PIVOT_LANGUAGE,
        "n_behaviors": len(by_behavior),
        "n_rows": len(rows),
        "counts_by_status": inventory["counts_by_status"],
        "response_log": inventory["response_log"],
        "counts_by_language": inventory["counts_by_language"],
        "counts_by_harm_label": inventory["counts_by_harm_label"],
        "split_placeholder": SPLIT_UNASSIGNED,
        "contains_prompt_text": False,
        # Rows carry hashes and lengths only -- no source or translated text.
        "rows": [row.to_manifest_dict() for row in rows],
    }
    manifest["manifest_digest"] = deterministic_digest(manifest)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    args.inventory.write_text(
        json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    write_review_queue(rows, requests, backend, queue_path=args.queue)

    for path in (args.requests, args.outputs, args.queue, args.manifest, args.inventory):
        # Report what git will actually do with the path rather than assuming
        # the v1 layout: under the v2 policy the manifest destination differs.
        print(f"  wrote {display_path(path)}  ({git_disposition(path)})")
    return 0


def display_path(path: Path) -> str:
    """Repo-relative when the path is inside the repo, absolute otherwise."""
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return str(resolved.relative_to(REPO_ROOT))
    return str(resolved)


def git_disposition(path: Path) -> str:
    """What git would do with ``path`` -- asked, not assumed."""
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        return "outside the repository"
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(resolved)],
        cwd=REPO_ROOT, capture_output=True, check=False,
    )
    if result.returncode == 0:
        return "git-ignored"
    if result.returncode == 1:
        return "tracked"
    return "git status unknown"


def write_review_queue(
    rows: list[TranslationRow],
    requests: list[dict[str, Any]],
    backend: dict[str, Any],
    *,
    queue_path: Path = QUEUE_PATH,
) -> None:
    """Human-readable manual translation/review queue.

    Lives in git-ignored ``data/interim/`` because it necessarily contains the
    source text a translator has to read.
    """
    pending = [r for r in rows if r.translated_text is None]
    by_language: dict[str, list[TranslationRow]] = {}
    for row in pending:
        by_language.setdefault(row.language, []).append(row)

    lines = [
        "# Translation pilot -- manual review queue",
        "",
        f"Backend resolution: **{backend['reason']}**",
        "",
        f"- Rows awaiting translation: **{len(pending)}**",
        f"- Languages: {', '.join(sorted(by_language))}",
        f"- Behaviors: {len({r.behavior_id for r in pending})}",
        "",
        "## Rules for whoever completes this",
        "",
        "1. Translate meaning and register faithfully.",
        "2. **Do not add operational detail** the source does not contain. A translation",
        "   must never be more actionable than its source.",
        "3. Do not soften, sanitise, refuse, or comment. A benign item stays benign; a",
        "   harmful item stays exactly as harmful as it was.",
        "4. Preserve orthography: Yoruba tone marks and sub-dots, Hausa hooked letters,",
        "   Arabic diacritics where present. Output NFC.",
        "5. Fill `translated_text` in the corresponding row of",
        "   `data/interim/translation_pilot_outputs.jsonl`, set `translation_method`,",
        "   `translation_model` (or translator identity), and `translation_timestamp`.",
        "6. `review_required` stays **true** until a **named native speaker** signs off,",
        "   at which point set `translation_status: HUMAN_REVIEWED` and `reviewer_id`.",
        "   Do not claim native-speaker validation otherwise.",
        "",
        "Source text is in `data/interim/translation_pilot_requests.jsonl` "
        "(git-ignored, one JSON object per request).",
        "",
        "## Queue by language",
        "",
        "| Language | Rows pending | Harmful | Benign |",
        "|---|---|---|---|",
    ]
    for language in sorted(by_language):
        group = by_language[language]
        lines.append(
            f"| {language} | {len(group)} | "
            f"{sum(1 for r in group if r.harm_label == 'harmful')} | "
            f"{sum(1 for r in group if r.harm_label == 'benign')} |"
        )

    lines += ["", "## Request IDs", "", "Ordered deterministically (label, category, behavior, language).", ""]
    for request in requests:
        lines.append(
            f"- `{request['request_id']}` — {request['harm_label']} / {request['category']} "
            f"— sha256 `{request['source_text_sha256'][:12]}`"
        )
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
