#!/usr/bin/env python3
"""Stage 0.0 -- verify the environment can reproducibly acquire the sources.

Runs no model, downloads no dataset. It answers one question: *would acquisition
work, and would it be reproducible?* Writing this down before fetching means a
later "the download was empty" has a recorded cause.

Checks:
  * Python version and platform
  * Required and optional package versions (pinned into the report)
  * Network reachability of each source host (HEAD only, no payload)
  * Credential-bearing environment variables, **by NAME only** (CLAUDE.md §7)
  * Repository git commit and dirty flag
  * Presence and writability of the data/results directories

Writes ``results/environment_report.json``. Exit code is non-zero if a REQUIRED
check fails; optional findings are reported without failing.

Usage
-----
    python scripts/00_check_environment.py
    python scripts/00_check_environment.py --no-network
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import platform
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multilingual_safety.hashing import deterministic_digest
from multilingual_safety.paths import (
    INTERIM_DIR,
    MANIFESTS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    REPO_ROOT,
    RESULTS_DIR,
)
from multilingual_safety.run_context import SECRET_NAME_MARKERS, looks_secret, stage_report_header

REQUIRED_PACKAGES = ("datasets", "huggingface_hub", "pandas", "requests", "pyyaml", "jsonschema")
OPTIONAL_PACKAGES = ("jailbreakbench", "pyarrow", "pytest", "ruff")

#: Hosts contacted during acquisition. Probed with a HEAD request; no dataset
#: payload is downloaded by this script.
NETWORK_TARGETS: tuple[tuple[str, str], ...] = (
    ("huggingface_api", "https://huggingface.co/api/datasets/JailbreakBench/JBB-Behaviors"),
    ("github_raw", "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/README.md"),
    ("github_api", "https://api.github.com/repos/centerforaisafety/HarmBench"),
)

REQUIRED_DIRS = (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MANIFESTS_DIR, RESULTS_DIR)


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in sorted(names):
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return versions


def probe(url: str, timeout: float = 20.0) -> dict[str, Any]:
    """HEAD-probe a URL. Never downloads a payload, never sends credentials."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "mls-stage0/0.0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {"reachable": True, "status": response.status, "error": None}
    except urllib.error.HTTPError as exc:
        # A 4xx still proves the host resolved and answered.
        return {"reachable": True, "status": exc.code, "error": f"HTTPError {exc.code}"}
    except (TimeoutError, urllib.error.URLError, OSError) as exc:
        return {"reachable": False, "status": None, "error": type(exc).__name__}


def directory_state(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "exists": exists,
        "writable": os.access(path, os.W_OK) if exists else False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "environment_report.json")
    parser.add_argument("--no-network", action="store_true", help="Skip reachability probes.")
    args = parser.parse_args()

    report: dict[str, Any] = stage_report_header(
        "stage0.0_check_environment", {"no_network": args.no_network}
    )

    report["python"] = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "in_virtualenv": sys.prefix != sys.base_prefix,
    }
    report["platform"] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }
    report["packages"] = {
        "required": package_versions(REQUIRED_PACKAGES),
        "optional": package_versions(OPTIONAL_PACKAGES),
    }
    report["tools"] = {name: bool(shutil.which(name)) for name in sorted(("git", "curl"))}

    report["network"] = (
        {"skipped": True}
        if args.no_network
        else {name: probe(url) | {"url": url} for name, url in sorted(NETWORK_TARGETS)}
    )

    # Credentials: NAMES ONLY. No value is read, printed, or hashed.
    secret_names = sorted(name for name in os.environ if looks_secret(name))
    report["credentials"] = {
        "secret_env_names_present": secret_names,
        "n_secret_env_names": len(secret_names),
        "hf_token_env_set": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "markers_used": list(SECRET_NAME_MARKERS),
        "note": "Names only. No credential value is read, logged, or hashed (CLAUDE.md §7).",
    }

    report["directories"] = [directory_state(d) for d in REQUIRED_DIRS]

    # --- adjudicate -------------------------------------------------------
    failures: list[str] = []
    warnings: list[str] = []

    for name, version in sorted(report["packages"]["required"].items()):
        if version is None:
            failures.append(f"required package not installed: {name}")
    for name, present in sorted(report["tools"].items()):
        if not present:
            warnings.append(f"tool not on PATH: {name}")
    if not args.no_network:
        for name, result in sorted(report["network"].items()):
            if not result["reachable"]:
                failures.append(f"unreachable acquisition host: {name} ({result['error']})")
    for state in report["directories"]:
        if not state["exists"]:
            failures.append(f"missing directory: {state['path']}")
        elif not state["writable"]:
            failures.append(f"directory not writable: {state['path']}")
    if report["git_dirty"]:
        warnings.append("git tree is dirty; the commit alone will not reproduce this run")

    report["failures"] = sorted(failures)
    report["warnings"] = sorted(warnings)
    report["ok"] = not failures
    # Excludes timestamps/run_id, so reruns are comparable.
    report["deterministic_digest"] = deterministic_digest(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"run_id:     {report['run_id']}")
    print(f"git commit: {report['git_commit']}  (dirty={report['git_dirty']})")
    print(f"python:     {report['python']['version']}  venv={report['python']['in_virtualenv']}")
    print("\nrequired packages:")
    for name, version in sorted(report["packages"]["required"].items()):
        print(f"  {'OK ' if version else 'MISSING'} {name:16} {version or ''}")
    if not args.no_network:
        print("\nnetwork:")
        for name, result in sorted(report["network"].items()):
            print(f"  {'OK ' if result['reachable'] else 'FAIL'} {name:18} status={result['status']}")
    print(f"\nsecret env vars detected (names only): {report['credentials']['n_secret_env_names']}")
    for warning in report["warnings"]:
        print(f"  WARN: {warning}")
    for failure in report["failures"]:
        print(f"  FAIL: {failure}", file=sys.stderr)
    print(f"\nwrote {args.out}")
    print(f"deterministic_digest: {report['deterministic_digest'][:16]}...")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
