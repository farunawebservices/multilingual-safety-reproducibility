#!/usr/bin/env python3
"""Verify the evaluation models are cached, pinned, and complete. Loads nothing.

Run this before the first inference run. It resolves each model in
``configs/models_v2.yaml`` against the local Hugging Face cache, asserts every
declared weight shard is present, and checks ``config.json`` against the
pre-registered architecture. No weights are loaded, nothing is downloaded, and
no credential is read.

    python scripts/20_verify_model_cache.py
    python scripts/20_verify_model_cache.py --json

Exit 0 if every model is usable, 2 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from multilingual_safety.agents.backends.base import BackendError
from multilingual_safety.agents.backends.cache import resolve_model_snapshot
from multilingual_safety.paths import REPO_ROOT, display_path

MODELS_CONFIG = REPO_ROOT / "configs" / "models_v2.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=MODELS_CONFIG)
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    args = parser.parse_args(argv)

    registry = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report: list[dict[str, object]] = []
    ok = True

    for model in registry.get("models", []):
        entry: dict[str, object] = {
            "id": model["id"],
            "model_id": model["model_id"],
            "revision": model["revision"],
        }
        try:
            resolved = resolve_model_snapshot(
                model["model_id"], model["revision"], expected=model.get("expected", {})
            )
        except BackendError as exc:
            ok = False
            entry.update({"usable": False, "error": f"{type(exc).__name__}: {exc}"})
        else:
            entry.update({"usable": True, **resolved.to_dict()})
        report.append(entry)

    if args.json:
        print(json.dumps({"config": display_path(args.config), "models": report}, indent=2))
        return 0 if ok else 2

    print(f"model cache verification -- {display_path(args.config)}")
    print("no weights loaded, nothing downloaded, no credential read\n")
    for entry in report:
        mark = "OK  " if entry.get("usable") else "FAIL"
        print(f"[{mark}] {entry['id']}")
        print(f"        model     : {entry['model_id']}")
        print(f"        revision  : {entry['revision']}")
        if entry.get("usable"):
            print(f"        snapshot  : {entry['snapshot_dir']}")
            print(
                f"        arch      : {entry['architecture']} "
                f"({entry['num_hidden_layers']} layers, hidden {entry['hidden_size']})"
            )
            print(f"        shards    : {entry['n_shards']} present")
        else:
            print(f"        error     : {entry['error']}")
        print()
    print("all models usable" if ok else "one or more models are NOT usable")
    return 0 if ok else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
