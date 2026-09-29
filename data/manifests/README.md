# data/manifests/ — TRACKED

This is the only directory under `data/` whose contents are committed.

## What lives here

| File | Status | Contents |
|---|---|---|
| `behavior.schema.json` | tracked | JSON Schema for one manifest row |
| `behaviors.jsonl` | tracked once built | One row per behavior per language — **metadata only** |
| `splits.json` | tracked once frozen | Behavior-level dev/test assignment + `split_hash` |

**Current state: no manifest and no split file exist. No data has been fetched.**

## What a manifest row may contain

IDs, SHA-256 content hashes, language codes, arm (`harmful`/`benign`), behavior
family, source and license metadata, translation provenance, and a text-free
orthography fingerprint.

## What it may never contain

Prompt text — in any language, original or translated. Model completions, plans,
tool calls, rationales, transcripts. Any credential.

The schema encodes this as a `not/anyOf` clause, `manifest.assert_no_forbidden_fields`
enforces it at write and read time, and `tests/test_repo_hygiene.py` fails the
suite if it is ever violated. Three layers, because the failure is unrecoverable
once pushed.

## Why hashes instead of text

The content hash lets a collaborator prove they built the identical corpus
without either party sending the text. `scripts/verify_manifest.py --data ...`
re-hashes a local build against the manifest and reports every mismatch by
record ID.

Hashes are computed over NFC-normalised text, so a round trip through a tool
that decomposes combining marks does not look like corpus corruption — while
ASCII-folding Yoruba or Hausa (a real bug) still changes the hash.

## Splits

`splits.json` assigns **behaviors**, not records, so all seven language
renderings of a behavior share a split. Record-level splitting would put the
English rendering in dev and the Hausa rendering in test, leaking the item.

The file is content-addressed. Its `split_hash` is recorded in every run's
metadata; if the hash changes, every prior test result is invalidated
(CLAUDE.md §1).
