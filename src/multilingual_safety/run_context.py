"""Run identity and provenance.

Nothing is written to a bare filename. Every model output and evaluation log
lives under ``results/<run_id>/`` and ``logs/<run_id>/``, and every run carries a
``run_metadata.json`` recording enough to reconstruct what produced it
(README §6).

A result that cannot be traced back to a run ID is not a result.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multilingual_safety.ids import run_id as mint_run_id
from multilingual_safety.ids import stable_hash, utc_timestamp
from multilingual_safety.paths import REPO_ROOT, run_logs_dir, run_results_dir

#: Substrings that mark an env var as secret-bearing. Such variables are
#: recorded by NAME ONLY -- never by value (CLAUDE.md §7).
SECRET_NAME_MARKERS: tuple[str, ...] = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "AUTH",
    "COOKIE",
)


def looks_secret(name: str) -> bool:
    """True if an environment variable name suggests it holds a credential."""
    upper = name.upper()
    return any(marker in upper for marker in SECRET_NAME_MARKERS)


def git_state(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Commit, branch, and dirty flag. Degrades gracefully outside a repo.

    A dirty tree is recorded rather than blocked -- but a dirty tree means the
    run is not reproducible from the commit alone, and that fact belongs in the
    metadata.
    """

    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = _run("status", "--porcelain")
    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "n_dirty_paths": len(status.splitlines()) if status else 0,
    }


def environment_state() -> dict[str, Any]:
    """Software environment. Contains no credential values."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


@dataclass(slots=True)
class RunContext:
    """Identity and provenance for one execution of one resolved config."""

    run_id: str
    started_at: str
    config_hash: str
    config: dict[str, Any]
    split: str
    split_hash: str | None = None
    manifest_hash: str | None = None
    code_hash: str = ""
    git: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    secret_env_names: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def create(
        cls,
        *,
        config: dict[str, Any],
        split: str,
        split_hash: str | None = None,
        manifest_hash: str | None = None,
        code_hash: str = "",
        notes: str = "",
        secret_env_names: list[str] | None = None,
    ) -> RunContext:
        config_hash = stable_hash(config)
        return cls(
            run_id=mint_run_id(config_hash=config_hash, code_hash=code_hash),
            started_at=datetime.now(timezone.utc).isoformat(),
            config_hash=config_hash,
            config=config,
            split=split,
            split_hash=split_hash,
            manifest_hash=manifest_hash,
            code_hash=code_hash,
            git=git_state(),
            environment=environment_state(),
            # Names only. Values are never read into metadata.
            secret_env_names=sorted(secret_env_names or []),
            notes=notes,
        )

    @property
    def results_dir(self) -> Path:
        return run_results_dir(self.run_id)

    @property
    def logs_dir(self) -> Path:
        return run_logs_dir(self.run_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "config_hash": self.config_hash,
            "config": self.config,
            "split": self.split,
            "split_hash": self.split_hash,
            "manifest_hash": self.manifest_hash,
            "code_hash": self.code_hash,
            "git": self.git,
            "environment": self.environment,
            "secret_env_names": self.secret_env_names,
            "notes": self.notes,
        }

    def materialize(self) -> Path:
        """Create ``results/<run_id>/`` and ``logs/<run_id>/``; write metadata.

        Refuses to reuse an existing run directory. Reruns mint a new run ID;
        overwriting one would destroy the record of what a prior number meant.
        """
        for directory in (self.results_dir, self.logs_dir):
            if directory.exists() and any(directory.iterdir()):
                raise FileExistsError(
                    f"{directory} already exists and is non-empty. Runs are never "
                    "overwritten; mint a new run ID (README §6)."
                )
            directory.mkdir(parents=True, exist_ok=True)
        meta_path = self.results_dir / "run_metadata.json"
        meta_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return meta_path


def stage_report_header(stage: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Provenance header embedded at the top of every stage report.

    Carries the repository git commit and a run ID, so any JSON report can be
    traced back to the exact code that produced it (README §6). ``git_dirty``
    is recorded honestly: a dirty tree means the commit alone does not
    reproduce the run.
    """
    config = config or {}
    config_hash = stable_hash(config)
    git = git_state()
    return {
        "stage": stage,
        "run_id": mint_run_id(config_hash=config_hash, code_hash=git.get("commit") or ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("dirty"),
        "config_hash": config_hash,
        "config": config,
        "tool_version": __import__("multilingual_safety").__version__,
    }


def _cli() -> int:
    """``mls-new-run`` -- mint a run ID from a config file without running anything."""
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Mint a run ID and create its directories.")
    parser.add_argument("--config", required=True, type=Path, help="Resolved config YAML")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run ID and metadata without creating directories.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ctx = RunContext.create(config=config, split=args.split, notes=args.notes)

    if args.dry_run:
        print(json.dumps(ctx.to_dict(), indent=2, sort_keys=True))
        print(f"\n[dry-run] would create: {ctx.results_dir}  {ctx.logs_dir}")
        return 0

    meta = ctx.materialize()
    print(f"run_id: {ctx.run_id}")
    print(f"metadata: {meta}")
    print(f"results: {ctx.results_dir}")
    print(f"logs:    {ctx.logs_dir}")
    print(f"started: {utc_timestamp()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
