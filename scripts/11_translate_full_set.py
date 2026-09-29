#!/usr/bin/env python3
"""Stage 2.1 -- translate the full 1,200-request set with the local NLLB backend.

Separate from ``scripts/04_translate_pilot.py`` on purpose. The reviewed pilot
files are opened read-only, and only to seed already-completed work: no pilot
path is writable from this script, under any flag.

Paths are parameterised (dataset v2)
------------------------------------
Every default is the v1 path, so an unflagged invocation behaves exactly as it
did before. They are overridable because the dataset-v2 run must write under
``data/v2/interim/`` and must NOT write into ``data/interim/full/``, which holds
the recovered request set and is read-only input here.

Every write target -- v1 or v2 -- goes through
``multilingual_safety.paths.assert_output_path_is_safe`` before anything is
opened. It refuses an output that equals an input, anything under
``data/processed/`` or ``data/v2/processed/``, any reviewed pilot artifact, and
any dataset-v2 corpus path outside ``data/v2/interim/``.

    v1 (unchanged):
        python scripts/11_translate_full_set.py --plan

    dataset v2:
        python scripts/11_translate_full_set.py --plan \\
            --config configs/translation_v2.yaml \\
            --requests data/interim/full/full_translation_requests.jsonl \\
            --responses data/v2/interim/full/full_translation_responses_v2.jsonl \\
            --pilot-responses data/v2/interim/full/pilot_responses_v2.jsonl

A run whose responses land under ``data/v2/interim/`` also writes a text-free
manifest to ``data/v2/manifests/full_translation_manifest_v2.json`` recording
the effective config path, model revision, seed, device, dtype, request hash and
output path. v1 runs write no manifest, exactly as before. ``--no-manifest``
suppresses it; ``--manifest`` redirects it.

Translated rows are NOT written to ``data/v2/processed/`` by this stage, or by
any flag it accepts. This stage produces a response log and nothing else.

The pilot's 120 rows
--------------------
The full set's 1,200 requests include the 120 the pilot already translated.
Two mutually exclusive ways to handle them, neither of them a default:

``--seed-pilot``
    Copy the existing pilot responses in verbatim so they are not redone.
    Opt-in, never silent; ``assert_single_device`` still refuses if their device
    differs from this run's.
``--regenerate-pilot-cuda``
    Regenerate exactly those 120 on CUDA into the full-set path, under a new
    ``run_id``, giving the corpus uniform provenance. The original pilot files
    are never modified; archive them first with ``scripts/12_archive_cpu_pilot.py``.

Either mode fails loudly if ``--pilot-responses`` does not exist. Silently
reusing zero rows would look identical to reusing all of them in the totals.

``assert_single_device`` refuses to append rows on a device the log does not
already use, so a mixed corpus cannot arise by accident. ``--force`` is not
accepted by this script at all.

Device policy
-------------
The device comes from the effective ``--config`` and must be explicit. With
``device: cuda`` the provider raises if CUDA is unavailable rather than
silently dropping to CPU, which would change latency and numerics without
changing any recorded field.

Usage
-----
    python scripts/11_translate_full_set.py --plan                  # inspect only
    python scripts/11_translate_full_set.py --translate --provider nllb_local
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from multilingual_safety.hashing import deterministic_digest, file_sha256
from multilingual_safety.languages import LANGUAGE_CODES, PIVOT_LANGUAGE
from multilingual_safety.paths import (
    CONFIGS_DIR,
    FULL_INTERIM_DIR,
    FULL_REQUESTS_PATH,
    INTERIM_DIR,
    V2_INTERIM_DIR,
    V2_MANIFESTS_DIR,
    PathSafetyError,
    assert_output_path_is_safe,
    display_path,
)
from multilingual_safety.providers import (
    STATUS_OK,
    ProviderError,
    ResponseLog,
    TranslationRequest,
    correlate,
    extract_seed,
    get_provider,
)
from multilingual_safety.run_context import stage_report_header

TRANSLATION_CONFIG = CONFIGS_DIR / "translation.yaml"

#: The dataset-v2 rebuild protocol. Read for one thing only: the request-set
#: hash its `materialization_precondition` pins.
PROTOCOL_CONFIG = CONFIGS_DIR / "dataset_v2_rebuild.yaml"

# --- v1 DEFAULTS. Unchanged; every one of them is overridable. --------------
FULL_DIR = FULL_INTERIM_DIR
REQUESTS_PATH = FULL_REQUESTS_PATH
RESPONSES_PATH = FULL_DIR / "full_translation_responses.jsonl"

#: Read-only. Used solely to seed already-completed translations.
PILOT_RESPONSES = INTERIM_DIR / "translation_pilot_responses.jsonl"

#: Where a dataset-v2 run records what it was configured with. Not used by v1.
V2_MANIFEST_PATH = V2_MANIFESTS_DIR / "full_translation_manifest_v2.json"

NON_PIVOT_LANGUAGES = tuple(c for c in LANGUAGE_CODES if c != PIVOT_LANGUAGE)

#: The reviewed pilot covers 20 behaviors x 6 non-English languages.
EXPECTED_PILOT_ROWS = 120


def is_v2_run(responses_path: Path) -> bool:
    """True when this run's corpus output lands in the dataset-v2 area."""
    return Path(responses_path).expanduser().resolve().is_relative_to(
        V2_INTERIM_DIR.resolve()
    )


def resolve_manifest_path(
    manifest: Path | None, responses_path: Path, *, suppressed: bool
) -> Path | None:
    """Where the run manifest goes, or ``None`` for "do not write one".

    Explicit ``--manifest`` wins. Otherwise a v2 run gets the v2 manifest and a
    v1 run gets nothing, so v1's on-disk footprint is exactly what it was.
    """
    if suppressed:
        return None
    if manifest is not None:
        return manifest
    return V2_MANIFEST_PATH if is_v2_run(responses_path) else None


def protocol_request_sha256() -> str | None:
    """The request-set hash pinned by the v2 protocol, or None if unavailable."""
    if not PROTOCOL_CONFIG.exists():
        return None
    doc = yaml.safe_load(PROTOCOL_CONFIG.read_text(encoding="utf-8")) or {}
    precondition = doc.get("materialization_precondition") or {}
    return (precondition.get("required_sha256") or {}).get("full_translation_requests_jsonl")


def pilot_request_ids(
    requests: list[TranslationRequest], pilot_responses: Path
) -> set[str]:
    """Request IDs that the pilot already translated successfully.

    Read from the pilot log, which this script never writes. Request IDs are a
    pure function of (behavior_id, language), so they carry over unchanged and
    the regenerated rows correlate to the same requests.
    """
    if not pilot_responses.exists():
        return set()
    wanted = {r.request_id for r in requests}
    pilot = ResponseLog(pilot_responses).effective()
    return {
        rid for rid, row in pilot.items()
        if rid in wanted and row.get("status") == STATUS_OK
    }


def assert_single_device(log: ResponseLog, device: str) -> None:
    """Refuse to add rows on a device the log does not already use.

    Prevents silently mixing CPU and CUDA translations in one corpus: beam
    search is not bit-identical across devices, so a mixed log would carry two
    incomparable generation regimes under one name.
    """
    devices = {
        row.get("device")
        for row in log.effective().values()
        if row.get("status") == STATUS_OK and row.get("device")
    }
    foreign = sorted(d for d in devices if d != device)
    if foreign:
        raise SystemExit(
            f"refusing to write {device!r} rows into a log that already contains "
            f"{foreign} rows. Mixing devices in one corpus makes generation "
            "non-comparable. Use a separate log path, or regenerate the existing rows."
        )


def assert_write_path_is_safe(
    path: Path, *, inputs: tuple[Path, ...] = (), kind: str = "corpus", what: str = "output"
) -> Path:
    """``assert_output_path_is_safe``, but exiting instead of raising.

    The rules live in ``multilingual_safety.paths`` so that stage 2.1 and stage
    2.3 cannot drift apart on what "a safe output path" means.
    """
    try:
        return assert_output_path_is_safe(path, kind=kind, inputs=inputs, what=what)
    except PathSafetyError as exc:
        raise SystemExit(str(exc)) from exc


def load_requests(path: Path) -> list[TranslationRequest]:
    requests: list[TranslationRequest] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            requests.append(
                TranslationRequest.create(
                    behavior_id=row["behavior_id"],
                    target_language=row["target_language"],
                    source_text=row["source_text"],
                )
            )
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            raise SystemExit(f"{path.name}:{lineno}: cannot parse request: {exc}") from exc
    requests.sort(key=lambda r: r.request_id)
    return requests


def seed_from_pilot(
    log: ResponseLog,
    requests: list[TranslationRequest],
    pilot_responses: Path,
    *,
    inputs: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Copy the pilot's successful responses into the full log, verbatim.

    Only records whose request_id is in the full set, and only those the full log
    does not already hold. Records keep their original run_id and timestamps, so
    provenance stays intact and auditable.
    """
    report = {"seeded": 0, "already_present": 0, "available": 0, "source": None}
    if not pilot_responses.exists():
        return report

    wanted = {r.request_id for r in requests}
    existing = set(log.effective())
    pilot = ResponseLog(pilot_responses).effective()
    candidates = [
        row for rid, row in sorted(pilot.items())
        if rid in wanted and row.get("status") == STATUS_OK
    ]
    report["available"] = len(candidates)
    report["source"] = display_path(pilot_responses)
    to_seed = [row for row in candidates if row["request_id"] not in existing]
    report["already_present"] = len(candidates) - len(to_seed)

    if to_seed:
        assert_write_path_is_safe(log.path, inputs=inputs, what="response log")
        log.path.parent.mkdir(parents=True, exist_ok=True)
        with log.path.open("a", encoding="utf-8") as fh:
            for row in to_seed:
                payload = dict(row)
                payload["seeded_from"] = report["source"]
                fh.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
        report["seeded"] = len(to_seed)
    return report


def assert_text_free(payload: dict[str, Any], requests: list[TranslationRequest]) -> None:
    """Refuse to write a manifest that carries corpus text.

    Checked rather than asserted in a docstring: the manifest is the one artifact
    of this stage that may become tracked, and every field it holds is supposed
    to be a path, a count, a hash, or a config scalar.
    """
    blob = json.dumps(payload, ensure_ascii=False)
    leaked = sorted({r.source_text for r in requests if r.source_text and r.source_text in blob})
    if leaked:
        raise SystemExit(
            f"refusing to write a manifest containing {len(leaked)} source text(s). "
            "Manifests carry IDs, hashes and metadata only."
        )


def build_manifest(
    *,
    header: dict[str, Any],
    config_path: Path,
    config: dict[str, Any],
    requests_path: Path,
    requests: list[TranslationRequest],
    responses_path: Path,
    pilot_responses: Path,
    provider_name: str | None,
    probe: dict[str, Any] | None,
    preflight: dict[str, Any] | None,
    mode: str,
    seed_report: dict[str, Any],
    pilot_reusable: int,
    translation: dict[str, Any] | None,
) -> dict[str, Any]:
    """The text-free record of what this run was configured with, and did."""
    settings = config.get("nllb_local_settings") or {}
    configured_seed = extract_seed(settings)
    reported_seed = extract_seed((preflight or {}).get("generation_config") or {})
    requests_sha = file_sha256(requests_path)
    pinned = protocol_request_sha256()

    manifest: dict[str, Any] = {
        **header,
        "artifact_type": "full_translation_run_manifest",
        "schema_version": "full-translation-run.v2",
        "dataset_version": "v2" if is_v2_run(responses_path) else "v1",
        "mode": mode,
        "translation_attempted": mode == "translate",
        # --- effective configuration ------------------------------------
        "config_path": display_path(config_path),
        "config_sha256": file_sha256(config_path),
        "provider_requested": provider_name,
        "model_id": settings.get("model_id"),
        "model_revision_configured": settings.get("revision"),
        "model_revision_reported": (preflight or probe or {}).get("model_revision"),
        "device_configured": settings.get("device"),
        "device_reported": (preflight or {}).get("device"),
        "dtype_configured": settings.get("dtype"),
        "dtype_reported": (preflight or {}).get("dtype"),
        "seed_configured": configured_seed,
        "seed_reported_by_provider": reported_seed,
        # The v2 pilot recorded v1's seed unnoticed because only one of these
        # two numbers was ever written down. Both are written down here.
        "seed_matches_config": (
            None if reported_seed is None else reported_seed == configured_seed
        ),
        "generation_config": (preflight or {}).get("generation_config"),
        # --- inputs -------------------------------------------------------
        "requests_path": display_path(requests_path),
        "requests_sha256": requests_sha,
        "requests_sha256_pinned_by_protocol": pinned,
        "requests_match_protocol_precondition": (
            None if pinned is None else requests_sha == pinned
        ),
        "n_requests": len(requests),
        "requests_by_language": dict(
            sorted(Counter(r.target_language for r in requests).items())
        ),
        "pilot_responses_path": display_path(pilot_responses),
        "pilot_responses_exists": pilot_responses.exists(),
        "pilot_responses_sha256": (
            file_sha256(pilot_responses) if pilot_responses.exists() else None
        ),
        "pilot_rows_reusable": pilot_reusable,
        "pilot_rows_seeded_this_run": seed_report["seeded"],
        # --- output -------------------------------------------------------
        "responses_path": display_path(responses_path),
        "responses_path_is_under_v2_interim": is_v2_run(responses_path),
        "responses_sha256": (
            file_sha256(responses_path) if responses_path.exists() else None
        ),
        "translated_rows_written_to_processed": False,
        "translated_rows_written_to_processed_note": (
            "This stage writes a response log only. Applying responses to rows is "
            "stage 2.3; freezing them under data/v2/processed/ is a later stage still."
        ),
        "translation": translation,
        "contains_prompt_text": False,
    }
    manifest["deterministic_digest"] = deterministic_digest(manifest)
    return manifest


def write_manifest(
    path: Path, manifest: dict[str, Any], requests: list[TranslationRequest]
) -> None:
    assert_text_free(manifest, requests)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {display_path(path)}  (metadata only)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--provider", default=None, help="Explicit provider id. No default.")
    parser.add_argument("--plan", action="store_true", help="Inspect and stop. Writes no response.")
    parser.add_argument("--translate", action="store_true", help="Actually translate.")
    parser.add_argument("--limit", type=int, default=None, help="Cap rows this run.")
    parser.add_argument(
        "--seed-pilot", action="store_true",
        help="Copy the existing pilot responses in so they are not redone. OFF by "
             "default; `assert_single_device` still refuses a device mismatch.",
    )
    parser.add_argument(
        "--regenerate-pilot-cuda", action="store_true",
        help="Regenerate exactly the 120 pilot translations on CUDA into the full-set "
             "provenance path. The original pilot files are never modified.",
    )
    # Paths. Defaults are the v1 paths, so an unflagged invocation is unchanged.
    # They are overridable because dataset v2 must write under data/v2/interim/
    # while reading the same recovered request set.
    parser.add_argument(
        "--config", type=Path, default=TRANSLATION_CONFIG,
        help="Translation backend config (default: %(default)s).",
    )
    parser.add_argument(
        "--requests", type=Path, default=REQUESTS_PATH,
        help="Request set. READ-ONLY (default: %(default)s).",
    )
    parser.add_argument(
        "--responses", type=Path, default=RESPONSES_PATH,
        help="Response log; the only corpus write target (default: %(default)s).",
    )
    parser.add_argument(
        "--pilot-responses", type=Path, default=PILOT_RESPONSES,
        help="Pilot response log to reuse completed work from. READ-ONLY "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Where to write the text-free run manifest. Defaults to "
             "data/v2/manifests/full_translation_manifest_v2.json for a run whose "
             "responses are under data/v2/interim/, and to nothing for a v1 run.",
    )
    parser.add_argument(
        "--no-manifest", action="store_true", help="Write no run manifest.",
    )
    args = parser.parse_args()
    # NOTE: --force is deliberately not offered. Nothing here overwrites.

    if not args.config.exists():
        print(f"missing config {args.config}", file=sys.stderr)
        return 1
    if not args.requests.exists():
        print(f"missing {args.requests}; run scripts/10_build_full_request_set.py first",
              file=sys.stderr)
        return 1

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    settings = config.get("nllb_local_settings") or {}
    v2 = is_v2_run(args.responses)
    header = stage_report_header(
        "stage2.1_translate_full_set",
        {
            "provider": args.provider,
            "device": settings.get("device"),
            "config_path": display_path(args.config),
            "dataset_version": "v2" if v2 else "v1",
        },
    )
    print(f"run_id:     {header['run_id']}")
    print(f"git commit: {header['git_commit']}  (dirty={header['git_dirty']})")

    # --- path safety, before anything is opened for writing ---------------
    read_only = (args.requests, args.pilot_responses)
    manifest_path = resolve_manifest_path(
        args.manifest, args.responses, suppressed=args.no_manifest
    )
    print(f"\nPATHS  (dataset {'v2' if v2 else 'v1'})")
    print(f"  config    : {display_path(args.config)}")
    print(f"  requests  : {display_path(args.requests)}  (READ-ONLY)")
    print(f"  responses : {display_path(args.responses)}   <- only corpus write target")
    print(f"  pilot log : {display_path(args.pilot_responses)}  (READ-ONLY, "
          f"exists={args.pilot_responses.exists()})")
    print(f"  manifest  : {display_path(manifest_path) if manifest_path else '(none)'}")
    assert_write_path_is_safe(args.responses, inputs=read_only, what="response log")
    if manifest_path is not None:
        assert_write_path_is_safe(
            manifest_path, kind="metadata", inputs=read_only, what="manifest"
        )
    print("  collision with pilot or processed artifacts: refused by "
          "assert_output_path_is_safe")

    requests = load_requests(args.requests)
    print(f"\nrequests loaded: {len(requests)}")
    print(f"  by language  : {dict(sorted(Counter(r.target_language for r in requests).items()))}")
    if any(r.target_language == PIVOT_LANGUAGE for r in requests):
        print("  FAIL: English appears as a translation target", file=sys.stderr)
        return 1

    # --- the protocol's materialization precondition ----------------------
    # configs/dataset_v2_rebuild.yaml requires this to be re-checked immediately
    # before translation begins, not once at authoring time.
    requests_sha = file_sha256(args.requests)
    pinned_sha = protocol_request_sha256()
    if pinned_sha is not None:
        matches = requests_sha == pinned_sha
        print(f"\nREQUEST SET  sha256={requests_sha[:16]}  "
              f"matches protocol precondition: {matches}")
        if not matches and args.translate:
            print("  FAIL: the request set does not match the hash pinned in "
                  "configs/dataset_v2_rebuild.yaml. Translation cannot begin until "
                  "protocol review resolves the mismatch.", file=sys.stderr)
            return 1
        if not matches:
            print("  WARNING: hash mismatch. --translate will refuse until this is "
                  "resolved.")

    # --- device / provider ------------------------------------------------
    print("\nDEVICE AND MODEL")
    print(f"  configured device : {settings.get('device')}")
    print(f"  configured dtype  : {settings.get('dtype')}")
    print(f"  pinned revision   : {settings.get('revision')}")
    print(f"  configured seed   : {extract_seed(settings)}")
    if settings.get("device") == "auto":
        print("  FAIL: device is 'auto'; set it explicitly so a missing GPU fails "
              "instead of silently using CPU", file=sys.stderr)
        return 1

    provider = None
    probe: dict[str, Any] | None = None
    if args.provider:
        try:
            provider = get_provider(args.provider, config)
            probe = provider.probe()
            print(f"  probe usable      : {probe.get('usable')}  ({probe.get('reason')})")
        except ProviderError as exc:
            print(f"  FAIL: {exc}", file=sys.stderr)
            return 3

    log = ResponseLog(args.responses)

    # --- seed completed pilot work ---------------------------------------
    if args.seed_pilot and args.regenerate_pilot_cuda:
        print("\nFAIL: --seed-pilot and --regenerate-pilot-cuda are mutually "
              "exclusive; one mixes CPU rows in, the other replaces them.", file=sys.stderr)
        return 1

    if (args.seed_pilot or args.regenerate_pilot_cuda) and not args.pilot_responses.exists():
        print(f"\nFAIL: {display_path(args.pilot_responses)} does not exist, so there "
              "are no pilot responses to reuse or regenerate. Reusing zero rows "
              "silently is indistinguishable from reusing all of them in the totals.",
              file=sys.stderr)
        return 1

    # Always computed and always reported, independent of --seed-pilot, so the
    # plan states what reuse is available rather than only what was requested.
    pilot_reusable = len(pilot_request_ids(requests, args.pilot_responses))

    seed = {"seeded": 0, "already_present": 0, "available": 0, "source": None}
    if args.seed_pilot and args.translate:
        seed = seed_from_pilot(log, requests, args.pilot_responses, inputs=read_only)
    elif args.seed_pilot:
        seed["available"] = pilot_reusable
        seed["source"] = display_path(args.pilot_responses)

    if args.regenerate_pilot_cuda:
        pilot_ids = pilot_request_ids(requests, args.pilot_responses)
        requests = [r for r in requests if r.request_id in pilot_ids]
        print(f"\nMODE: --regenerate-pilot-cuda  (restricted to {len(requests)} pilot rows)")
        if len(requests) != EXPECTED_PILOT_ROWS:
            print(f"  FAIL: expected exactly {EXPECTED_PILOT_ROWS} pilot rows, "
                  f"got {len(requests)}", file=sys.stderr)
            return 1

    pending = log.pending(requests)
    if args.limit is not None:
        pending = pending[: args.limit]

    print("\nWORK PLAN")
    print(f"  total requests            : {len(requests)}")
    print(f"  reusable pilot responses  : {seed['available']}  (from {seed['source']})")
    print(f"  reusable with --seed-pilot: {pilot_reusable}  "
          f"(from {display_path(args.pilot_responses)})")
    print(f"  seeded into full log      : {seed['seeded']}")
    print(f"  already complete in log   : {len(requests) - len(pending) if args.translate else '-'}")
    print(f"  pending this run          : {len(pending) if args.translate else len(requests) - seed['available']}")
    print(f"  pending by language       : "
          f"{dict(sorted(Counter(r.target_language for r in pending).items())) if args.translate else '-'}")

    if args.plan or not args.translate:
        print("\nPLAN ONLY: nothing was translated and no response was written.")
        if manifest_path is not None:
            write_manifest(
                manifest_path,
                build_manifest(
                    header=header, config_path=args.config, config=config,
                    requests_path=args.requests, requests=requests,
                    responses_path=args.responses, pilot_responses=args.pilot_responses,
                    provider_name=args.provider, probe=probe, preflight=None,
                    mode="plan", seed_report=seed, pilot_reusable=pilot_reusable,
                    translation=None,
                ),
                requests,
            )
        return 0

    # --- translate --------------------------------------------------------
    if provider is None:
        print("\nFAIL: --translate requires --provider", file=sys.stderr)
        return 1
    metadata = provider.preflight()          # raises if CUDA unavailable
    print("\nPREFLIGHT")
    for key in sorted(metadata):
        print(f"  {key}: {metadata[key]}")
    log.check_provider_consistency(provider.name, metadata["model_id"])
    assert_single_device(log, metadata["device"])
    assert_write_path_is_safe(args.responses, inputs=read_only, what="response log")

    responses = []
    for index, request in enumerate(pending, start=1):
        response = provider.translate(request, run_id=header["run_id"])
        responses.append(response)
        if index % 50 == 0 or index == len(pending):
            ok = sum(1 for r in responses if r.status == STATUS_OK)
            print(f"  [{index:>5}/{len(pending)}] ok={ok} err={len(responses) - ok}")
    log.append(responses)

    correlation = correlate(pending, responses)
    print(f"\ncorrelation: {correlation}")
    print(f"wrote {display_path(args.responses)}  sha256={file_sha256(args.responses)[:16]}")

    if manifest_path is not None:
        write_manifest(
            manifest_path,
            build_manifest(
                header=header, config_path=args.config, config=config,
                requests_path=args.requests, requests=requests,
                responses_path=args.responses, pilot_responses=args.pilot_responses,
                provider_name=args.provider, probe=probe, preflight=metadata,
                mode="translate", seed_report=seed, pilot_reusable=pilot_reusable,
                translation={
                    "n_attempted": len(pending),
                    "n_responses": len(responses),
                    "n_ok": sum(1 for r in responses if r.status == STATUS_OK),
                    "n_not_ok": sum(1 for r in responses if r.status != STATUS_OK),
                    "correlation": correlation,
                    "response_log": log.retry_summary(),
                },
            ),
            requests,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
