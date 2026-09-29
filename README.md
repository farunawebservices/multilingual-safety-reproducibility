# Cross-Lingual Safety in Multi-Agent Systems

A research repository studying whether the safety behavior of a multi-agent LLM
pipeline holds up when the *same* underlying request is expressed in different
languages — including lower-resource languages that are under-represented in
safety training data.

**Status:** scaffold only. No dataset has been built, no model has been run.
The code here defines structure, schemas, and invariants; it does not yet
produce results.

---

## 1. Research questions

**RQ1 — Cross-lingual safety gap.**
For a fixed behavior expressed in each of the seven target languages, does the
end-to-end refusal / veto rate of the multi-agent pipeline differ by language?
Is any degradation monotone in language resourcedness?

**RQ2 — Localisation of failure.**
When the pipeline fails to block a harmful behavior, *which agent* failed? Did
the Planner comply where it should have refused, did the Tool-Caller execute an
unsafe plan, or did the Critic fail to veto? The pipeline is instrumented so
each stage's decision is recorded separately.

**RQ3 — Cross-lingual transfer of the Critic.**
Does a Critic that reliably vetoes an unsafe plan in English still veto the
semantically equivalent plan in `ha`, `yo`, `sw`? We separate *language of the
user prompt* from *language of the intermediate plan*, since these can differ.

**RQ4 — Interventions vs. matched controls.**
Do targeted interventions (e.g. activation steering, prompt-level scaffolds,
critic-language forcing) shift veto rates beyond what a **matched
random-vector / matched-sham control** of the same magnitude produces? An
intervention with no matched control produces no reportable claim in this
project.

**RQ5 — Over-refusal cost.**
What is the per-language false-veto rate on **benign** prompts? A method that
improves harmful-prompt veto rates while destroying benign throughput in `ha`
or `yo` is a regression, not a result, and is reported as such.

Negative results for any of RQ1–RQ5 are first-class outputs and are written up
regardless of direction.

---

## 2. Target languages

Seven languages, fixed for the duration of the project:

| Code | Language | Script | Notes |
|------|----------|--------|-------|
| `en` | English  | Latin  | Reference / pivot language |
| `ar` | Arabic   | Arabic | RTL; diacritic-optional orthography |
| `ha` | Hausa    | Latin  | Lower-resource; hooked letters (ɓ ɗ ƙ) |
| `yo` | Yoruba   | Latin  | Lower-resource; tone marks + sub-dots (ẹ ọ ṣ) |
| `sw` | Swahili  | Latin  | Lower-resource |
| `hi` | Hindi    | Devanagari | |
| `ko` | Korean   | Hangul | |

`en` is the pivot: every behavior exists in `en` plus all six others, so that
per-language effects are measured against a common reference rather than
against each other.

Orthography matters here and is not incidental. Yoruba tone marks and Hausa
hooked letters survive or do not survive tokenisation, normalisation, and
round-tripping; the pipeline records a Unicode-normalisation fingerprint per
record so that "the model failed in Yoruba" can be distinguished from "our
preprocessing silently stripped diacritics."

---

## 3. Intended architecture: Planner → Tool-Caller → Critic

```
                 ┌──────────────────────────────────────────────┐
  user prompt ──▶│ PLANNER      produces a structured plan       │
  (lang L)       │              (may itself refuse)              │
                 └───────────────┬──────────────────────────────┘
                                 │ plan (language recorded separately)
                 ┌───────────────▼──────────────────────────────┐
                 │ TOOL-CALLER  maps plan steps to tool calls    │
                 │              against a mocked tool surface    │
                 └───────────────┬──────────────────────────────┘
                                 │ proposed tool calls
                 ┌───────────────▼──────────────────────────────┐
                 │ CRITIC       emits ALLOW or VETO with reason  │
                 └───────────────┬──────────────────────────────┘
                                 │
                            final outcome
```

- **Planner** turns the user request into an explicit plan. It may refuse.
- **Tool-Caller** converts plan steps into tool invocations. **Tools are
  mocked.** No real capability is exercised; the object of study is the
  decision, not the effect.
- **Critic** reviews the plan and proposed calls and emits a decision.

Each stage emits its own record. A single trial therefore yields a *vector* of
per-stage decisions, not one bit. This is what makes RQ2 answerable.

Every stage decision is classified into exactly one of six categories, and
nothing is allowed to fall through to a default:

| Category | Meaning |
|---|---|
| `SEMANTIC_ALLOW` | A parseable decision whose meaning is *permit* |
| `SEMANTIC_VETO` | A parseable decision whose meaning is *block* |
| `MISSING_DECISION` | Output was produced but contains no decision at all |
| `MALFORMED_DECISION` | A decision field exists but does not parse / is out of schema |
| `TIMEOUT` | Wall-clock or token budget exceeded before completion |
| `NON_TERMINATION` | Degenerate generation: loops, repeats, never emits a stop |

`MISSING_DECISION` is **not** silently recoded as a veto, and `MALFORMED_DECISION`
is **not** silently recoded as an allow. Rates for all six are reported per
language. See `src/multilingual_safety/decisions.py`.

---

## 4. Harmful vs. benign prompts

The corpus has two disjoint arms, and both are required.

**Harmful arm.** Behaviors drawn from established, license-documented red-team
benchmarks. Correct pipeline behavior is refusal or veto. Measured quantity:
**attack success / miss rate** per language.

**Benign arm.** Prompts that are *superficially similar* — same topic, same
domain, same tooling need — but with no harmful objective. Correct pipeline
behavior is normal completion. Measured quantity: **over-refusal (false veto)
rate** per language.

Both arms are mandatory because a single-arm evaluation is trivially gameable: a
pipeline that vetoes everything scores perfectly on the harmful arm alone. Every
headline number in this project is reported as a harmful/benign pair.

Each behavior carries a `harm_class` (harmful / benign) and a `behavior_family`
so that the benign arm can be matched to the harmful arm family-by-family rather
than in aggregate.

**Prompt text is not stored in this repository.** Manifests hold IDs, content
hashes, language codes, source identifiers, and license metadata. The text is
materialised locally by `scripts/fetch_sources.py` into `data/raw/`, which is
git-ignored. See §7 and `THIRD_PARTY_DATASETS.md`.

---

## 5. Frozen dev/test splits

Behavior IDs are assigned to `dev` and `test` splits, and that assignment is
**frozen before any model is run.**

- The split file is content-addressed; its hash is recorded in every run's
  metadata.
- All development, prompt iteration, threshold selection, intervention-strength
  selection, and debugging happen on `dev` only.
- `test` is read exactly once per finalised configuration, and the number of
  test reads is logged.
- If the split file's hash changes after any test read, every prior test result
  is invalidated and must be re-labelled with the new split hash. There is no
  quiet re-split.

Intervention *strength* is a hyperparameter and is therefore chosen on `dev`.
Choosing it on `test` would be selection-on-the-test-set and would void the
result. See `CLAUDE.md`.

---

## 6. Run IDs and versioning

Nothing is written to a bare filename. Every model output and every evaluation
log is written under a **run ID**:

```
results/<run_id>/...
logs/<run_id>/...
```

with `run_id = run_<UTC timestamp>_<8 hex of config+code hash>`.

Each run directory carries a `run_metadata.json` recording at minimum: the run
ID, UTC start time, git commit and dirty flag, resolved config hash, split file
hash, manifest hashes, model identifiers and decoding parameters, seeds, and the
software environment. A result that cannot be traced back to a run ID is not a
result and is not reported.

Reruns never overwrite. A repeated configuration produces a new run ID; the
comparison between them is itself data (it measures nondeterminism).

`results/` and `logs/` are git-ignored — they are versioned by run ID on disk
and synced to durable storage, not committed as blobs.

---

## 7. Data handling and what stays out of this repository

**Raw harmful prompt text may remain outside this repository even though the
repository is private.** Private is not the same as safe: repositories get
shared, forked, made public, mirrored to CI, and pulled into unrelated tooling.
Two independent reasons keep the text out:

1. **Licensing.** Third-party benchmark terms — not this repo's MIT license —
   govern the prompt text, and most do not permit redistribution. See
   `THIRD_PARTY_DATASETS.md`.
2. **Harm surface.** A curated, translated, seven-language harmful-prompt corpus
   is a more capable artifact than any of its monolingual sources. Producing one
   and committing it is not a neutral act.

What *is* tracked: manifests of behavior IDs, SHA-256 content hashes, language
codes, split assignments, source/license metadata, translation provenance, and
all the code needed to rebuild the corpus from its sources.

What is *not* tracked: prompt text, model completions, transcripts, raw logs,
API keys.

Verification, not trust: `scripts/verify_manifest.py` re-hashes locally
materialised text against the manifest, so a collaborator can prove they built
the same corpus without either party sending the text to the other.

---

## 8. Repository layout

```
configs/                    YAML configs (languages, agents, run templates)
data/
  raw/                      materialised source data — git-ignored
  interim/                  intermediate build products — git-ignored
  processed/                model-ready records — git-ignored
  manifests/                IDs, hashes, metadata, splits — TRACKED
scripts/                    fetch / build / verify / split / run-id CLIs
src/multilingual_safety/    library code
tests/                      pytest suite (invariants, schemas, hygiene)
results/                    per-run outputs, keyed by run ID — git-ignored
reports/                    write-ups and figures
logs/                       per-run logs, keyed by run ID — git-ignored
```

---

## 9. Getting started

The package uses a **src layout** (`src/multilingual_safety/`), so it must be
installed — or resolved via the committed pytest config — before anything
imports it. An editable install is the supported path:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"     # pytest, pytest-cov, ruff, mypy
```

Then the three checks:

```bash
python -m pytest -q                   # test suite
python -m ruff check .                # lint
python scripts/check_hygiene.py       # no secrets / no corpus text tracked
```

Extras: `.[data]` for Stage 0 acquisition (`datasets`, `huggingface_hub`,
`pandas`, `requests`); `.[analysis]` for later analysis; `.[models]` is
deliberately empty until the inference stage.

**Troubleshooting `ModuleNotFoundError: No module named 'multilingual_safety'`.**
The package is not installed in the interpreter you are running. Check that the
active environment is the one you installed into:

```bash
python -c "import sys; print(sys.executable)"
python -m pip show multilingual-safety
```

Note that `VIRTUAL_ENV` being set does **not** guarantee that venv's `bin/` is on
`PATH`. Prefer `python -m pip ...` and `python -m pytest ...` (module form) over
the bare `pip` / `pytest` executables, so the tool always matches the
interpreter. Do **not** work around this by exporting `PYTHONPATH`; the layout is
resolved by `pythonpath = ["src"]` in `[tool.pytest.ini_options]`, which is
committed and applies to everyone.

Then, in order:

```bash
python scripts/fetch_sources.py --dry-run      # show what would be fetched
python scripts/build_manifest.py --help        # build behavior manifest
python scripts/make_splits.py --help           # freeze dev/test assignment
python scripts/verify_manifest.py --help       # re-hash local data vs manifest
python scripts/new_run.py --help               # mint a run ID
```

Secrets go in the environment or a git-ignored `.env`, never in a tracked file.

---

## 10. Scope and intent

This is defensive safety-evaluation research: the goal is to find and document
where multi-agent safety mechanisms fail to transfer across languages, so they
can be fixed — with particular attention to languages that routine English-only
evaluation would never surface. The tool surface is mocked; the pipeline is
built to measure decisions, not to carry them out.

## License

Code in this repository: MIT (see `LICENSE`).
Third-party datasets: **not** MIT — see `THIRD_PARTY_DATASETS.md`.
