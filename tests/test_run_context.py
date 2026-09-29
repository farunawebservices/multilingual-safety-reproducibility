"""Run identity: traceable, non-overwriting, secret-free."""

from __future__ import annotations

import json

import pytest

from multilingual_safety.ids import is_run_id
from multilingual_safety.run_context import RunContext, looks_secret


def test_run_context_is_traceable():
    ctx = RunContext.create(config={"experiment": {"name": "demo"}}, split="dev")
    assert is_run_id(ctx.run_id)
    assert ctx.config_hash
    assert ctx.results_dir.name == ctx.run_id
    assert ctx.logs_dir.name == ctx.run_id


def test_identical_config_yields_identical_config_hash():
    a = RunContext.create(config={"a": 1, "b": 2}, split="dev")
    b = RunContext.create(config={"b": 2, "a": 1}, split="dev")
    assert a.config_hash == b.config_hash


def test_metadata_records_secret_names_only():
    ctx = RunContext.create(
        config={"api_key_env": "OPENAI_API_KEY"},
        split="dev",
        secret_env_names=["OPENAI_API_KEY"],
    )
    payload = json.dumps(ctx.to_dict())
    assert "OPENAI_API_KEY" in payload
    # The config references a variable NAME; no value is present anywhere.
    assert "sk-" not in payload


@pytest.mark.parametrize(
    "name", ["OPENAI_API_KEY", "HF_TOKEN", "MY_SECRET", "DB_PASSWORD", "SESSION_COOKIE"]
)
def test_secret_name_detection(name):
    assert looks_secret(name)


def test_non_secret_names_not_flagged():
    assert not looks_secret("RESULTS_DIR")
    assert not looks_secret("SPLIT")


def test_run_directories_are_never_reused(tmp_path, monkeypatch):
    import multilingual_safety.paths as paths
    import multilingual_safety.run_context as rc

    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(rc, "run_results_dir", lambda rid: tmp_path / "results" / rid)
    monkeypatch.setattr(rc, "run_logs_dir", lambda rid: tmp_path / "logs" / rid)

    ctx = RunContext.create(config={"x": 1}, split="dev")
    meta = ctx.materialize()
    assert meta.exists()
    assert json.loads(meta.read_text(encoding="utf-8"))["run_id"] == ctx.run_id

    with pytest.raises(FileExistsError, match="never overwritten"):
        ctx.materialize()
