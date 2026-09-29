#!/usr/bin/env python3
"""Stage 0.1 -- reproducible acquisition of JBB-Behaviors and HarmBench.

The two sources are kept **strictly separate** on disk and in the manifest:

    data/raw/jailbreakbench/
    data/raw/harmbench/

They are NOT merged into a single benchmark at this stage. That separation is
not merely tidiness -- 27 of JBB's 100 harmful behaviors are themselves sourced
from TDC/HarmBench, so a naive concatenation would silently double-count
overlapping behaviors and inflate any per-language rate computed over the union.

Acquisition method per source
-----------------------------
* **JBB-Behaviors** -- the Hugging Face ``datasets`` library is used to
  *discover and verify* the real config/split names (never assumed), and
  ``huggingface_hub`` pins byte-exact CSVs at a fixed revision so the SHA-256
  hashes are stable and re-checkable.
* **HarmBench** -- fetched from the official GitHub repository at a **pinned
  commit** (never ``main``, which moves). URL, commit, SHA-256, and retrieval
  timestamp are all recorded.

Every acquired file is schema-checked on arrival; drift raises loudly rather
than being absorbed (``src/multilingual_safety/schemas.py``).

Raw data is written to git-ignored ``data/raw/`` only. Both sources are
MIT-licensed, so redistribution would be *legally* permitted -- but repository
policy still keeps harmful prompt text out of git (THIRD_PARTY_DATASETS.md).
The acquisition script plus hashes are what get committed.

Usage
-----
    python scripts/01_download_sources.py --dry-run     # show the plan
    python scripts/01_download_sources.py               # acquire
    python scripts/01_download_sources.py --verify-only # re-hash local copies
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from multilingual_safety.hashing import file_sha256, table_digest
from multilingual_safety.paths import MANIFESTS_DIR, RAW_DIR, REPO_ROOT
from multilingual_safety.run_context import stage_report_header
from multilingual_safety.schemas import (
    SchemaMismatchError,
    assert_part,
    check_hf_configs,
)

# --- pinned provenance -----------------------------------------------------
JBB_REPO = "JailbreakBench/JBB-Behaviors"
# Pinned revision, verified 2026-08-16. Never track a moving ref.
JBB_REVISION = "886acc352a31533ffbcf4ef22c744658688086fc"
JBB_FILES = {"harmful": "data/harmful-behaviors.csv", "benign": "data/benign-behaviors.csv"}

HARMBENCH_REPO = "centerforaisafety/HarmBench"
# Pinned commit for data/behavior_datasets/, verified 2026-08-16.
HARMBENCH_COMMIT = "c0423b952435fcc8467108d8f25962dbae5b7de2"
HARMBENCH_RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
HARMBENCH_FILES = {
    "text_all": "data/behavior_datasets/harmbench_behaviors_text_all.csv",
    "text_val": "data/behavior_datasets/harmbench_behaviors_text_val.csv",
    "text_test": "data/behavior_datasets/harmbench_behaviors_text_test.csv",
}

JBB_DIR = RAW_DIR / "jailbreakbench"
HARMBENCH_DIR = RAW_DIR / "harmbench"
SOURCE_MANIFEST = MANIFESTS_DIR / "source_manifest.yaml"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return columns, rows


def describe_file(path: Path, source_id: str, part_id: str) -> dict[str, Any]:
    """Hash + schema-check one acquired file. Raises on drift."""
    columns, rows = read_csv_rows(path)
    assert_part(source_id, part_id, columns=columns, n_rows=len(rows))
    return {
        "part_id": part_id,
        "local_path": str(path.relative_to(REPO_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "n_rows": len(rows),
        "columns": columns,
        # Order-independent: a reordered file matches, an edited one does not.
        "table_digest": table_digest(rows, columns=columns),
        "schema_ok": True,
    }


# ---------------------------------------------------------------------------
def acquire_jbb(dry_run: bool) -> dict[str, Any]:
    """Acquire JBB-Behaviors, verifying the REAL config/split names."""
    from datasets import get_dataset_config_names, get_dataset_split_names
    from huggingface_hub import hf_hub_download

    print("\n--- JailbreakBench / JBB-Behaviors ---")
    print(f"repo:     {JBB_REPO}")
    print(f"revision: {JBB_REVISION}")

    # VERIFY, never assume. The obvious guess -- a `harmful` config and a
    # `benign` config -- is wrong: there is one `behaviors` config with two
    # SPLITS by those names.
    configs = get_dataset_config_names(JBB_REPO, revision=JBB_REVISION)
    observed: dict[str, list[str]] = {}
    for config in sorted(configs):
        observed[config] = sorted(
            get_dataset_split_names(JBB_REPO, config, revision=JBB_REVISION)
        )
    print(f"observed configs -> splits: {json.dumps(observed, sort_keys=True)}")

    violations = check_hf_configs("jailbreakbench", observed)
    if violations:
        raise SchemaMismatchError(
            "JBB config/split structure drifted from the pinned expectation:\n  - "
            + "\n  - ".join(violations)
        )
    print("config/split structure matches the pinned expectation")

    record: dict[str, Any] = {
        "source_id": "jailbreakbench",
        "name": JBB_REPO,
        "acquisition_method": "huggingface_hub.hf_hub_download (byte-exact) + "
        "datasets.get_dataset_config_names/get_dataset_split_names (structure verification)",
        "url": f"https://huggingface.co/datasets/{JBB_REPO}",
        "revision": JBB_REVISION,
        "revision_type": "hf_dataset_commit_sha",
        "observed_configs": observed,
        "config_used": "behaviors",
        "configs_not_acquired": ["judge_comparison"],
        "license_id": "MIT",
        "license_url": f"https://huggingface.co/datasets/{JBB_REPO}/blob/{JBB_REVISION}/LICENSE",
        "local_dir": str(JBB_DIR.relative_to(REPO_ROOT)),
        "retrieved_at": None,
        "files": [],
    }

    if dry_run:
        for _part, remote in sorted(JBB_FILES.items()):
            print(f"  [dry-run] would download {remote} -> {JBB_DIR / Path(remote).name}")
        return record

    JBB_DIR.mkdir(parents=True, exist_ok=True)
    record["retrieved_at"] = now_iso()
    for part, remote in sorted(JBB_FILES.items()):
        cached = hf_hub_download(
            repo_id=JBB_REPO, filename=remote, repo_type="dataset", revision=JBB_REVISION
        )
        local = JBB_DIR / Path(remote).name
        local.write_bytes(Path(cached).read_bytes())
        info = describe_file(local, "jailbreakbench", part)
        info["remote_path"] = remote
        record["files"].append(info)
        print(f"  {part:8} {info['n_rows']:>4} rows  sha256={info['sha256'][:16]}...  -> {local.name}")

    record["files"].sort(key=lambda f: f["part_id"])
    return record


def acquire_harmbench(dry_run: bool) -> dict[str, Any]:
    """Acquire HarmBench text behaviors from the official repo at a pinned commit."""
    print("\n--- HarmBench ---")
    print(f"repo:   {HARMBENCH_REPO}")
    print(f"commit: {HARMBENCH_COMMIT}")

    record: dict[str, Any] = {
        "source_id": "harmbench",
        "name": "HarmBench text behaviors",
        "acquisition_method": "HTTPS GET from raw.githubusercontent.com at a pinned commit",
        "url": f"https://github.com/{HARMBENCH_REPO}",
        "revision": HARMBENCH_COMMIT,
        "revision_type": "git_commit_sha",
        "license_id": "MIT",
        "license_url": f"https://github.com/{HARMBENCH_REPO}/blob/{HARMBENCH_COMMIT}/LICENSE",
        "local_dir": str(HARMBENCH_DIR.relative_to(REPO_ROOT)),
        "retrieved_at": None,
        "files": [],
        "notes": "text_all is the text-only set (vs harmbench_behaviors_multimodal_all.csv). "
        "The official val/test partition is file-level, not a column.",
    }

    if dry_run:
        for _part, remote in sorted(HARMBENCH_FILES.items()):
            print(f"  [dry-run] would download {remote}")
        return record

    HARMBENCH_DIR.mkdir(parents=True, exist_ok=True)
    record["retrieved_at"] = now_iso()
    for part, remote in sorted(HARMBENCH_FILES.items()):
        url = HARMBENCH_RAW.format(repo=HARMBENCH_REPO, commit=HARMBENCH_COMMIT, path=remote)
        local = HARMBENCH_DIR / Path(remote).name
        request = urllib.request.Request(url, headers={"User-Agent": "mls-stage0/0.0.1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            local.write_bytes(response.read())
        info = describe_file(local, "harmbench", part)
        info["remote_path"] = remote
        info["url"] = url
        record["files"].append(info)
        print(f"  {part:10} {info['n_rows']:>4} rows  sha256={info['sha256'][:16]}...  -> {local.name}")

    record["files"].sort(key=lambda f: f["part_id"])
    return record


def verify_only() -> int:
    """Re-hash local copies against the recorded manifest. Repairs nothing."""
    if not SOURCE_MANIFEST.exists():
        print(f"no manifest at {SOURCE_MANIFEST}; run acquisition first", file=sys.stderr)
        return 1
    manifest = yaml.safe_load(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    failures = 0
    for source in sorted(manifest.get("sources", []), key=lambda s: s["source_id"]):
        for info in sorted(source.get("files", []), key=lambda f: f["part_id"]):
            path = REPO_ROOT / info["local_path"]
            if not path.exists():
                print(f"  MISSING  {info['local_path']}")
                failures += 1
                continue
            actual = file_sha256(path)
            ok = actual == info["sha256"]
            failures += not ok
            print(f"  {'OK      ' if ok else 'MISMATCH'} {info['local_path']}")
    print(f"\n{failures} verification failure(s)")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Show the plan; download nothing.")
    parser.add_argument("--verify-only", action="store_true", help="Re-hash local copies only.")
    parser.add_argument("--out", type=Path, default=SOURCE_MANIFEST)
    args = parser.parse_args()

    if args.verify_only:
        return verify_only()

    header = stage_report_header(
        "stage0.1_download_sources",
        {"jbb_revision": JBB_REVISION, "harmbench_commit": HARMBENCH_COMMIT},
    )
    print(f"run_id:     {header['run_id']}")
    print(f"git commit: {header['git_commit']}  (dirty={header['git_dirty']})")
    print(f"mode:       {'DRY-RUN' if args.dry_run else 'ACQUIRE'}")

    try:
        sources = [acquire_jbb(args.dry_run), acquire_harmbench(args.dry_run)]
    except SchemaMismatchError as exc:
        print(f"\nSTOP -- schema drift detected:\n{exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print("\nDry run complete. Nothing was downloaded and no manifest was written.")
        return 0

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": header["stage"],
        "run_id": header["run_id"],
        "git_commit": header["git_commit"],
        "git_dirty": header["git_dirty"],
        "generated_at": header["generated_at"],
        "policy": {
            "sources_kept_separate": True,
            "merged_into_single_benchmark": False,
            "raw_data_committed": False,
            "raw_data_committed_reason": (
                "Both sources are MIT-licensed, so redistribution is legally permitted. "
                "Repository policy still keeps harmful prompt text out of git "
                "(THIRD_PARTY_DATASETS.md); the acquisition script and hashes are "
                "what make the corpus reproducible."
            ),
            "translated": False,
            "model_run": False,
        },
        "sources": sorted(sources, key=lambda s: s["source_id"]),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        yaml.safe_dump(manifest, sort_keys=True, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )

    total = sum(f["n_rows"] for s in sources for f in s["files"])
    print(f"\nacquired {total} rows across {sum(len(s['files']) for s in sources)} files")
    print(f"wrote manifest -> {args.out}")
    print("\nSources are stored separately and are NOT merged (see policy in the manifest).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
