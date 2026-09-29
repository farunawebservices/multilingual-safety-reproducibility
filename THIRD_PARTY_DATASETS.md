# Third-party datasets — licensing and terms

## The short version

**The MIT license in `LICENSE` covers the code in this repository. It does not
cover, and cannot be extended to, any third-party dataset.**

Each external dataset remains governed by **its own license, terms of use, and
access conditions**, which are **authoritative** and take precedence over
anything stated here or elsewhere in this repository. Nothing in this repo
grants any right to any third-party data. If this document and an upstream
license disagree, the upstream license wins.

This project therefore distributes **no third-party prompt text**. It
distributes manifests — identifiers, content hashes, language codes, split
assignments, and license metadata — plus the code to rebuild a corpus locally
from sources the user obtains themselves under those sources' own terms.

---

## What this repository tracks vs. what it does not

**Tracked (safe to commit):**

- Behavior IDs and stable content hashes (SHA-256)
- Language codes, script and normalisation fingerprints
- Split assignments (`dev` / `test`) and their file hashes
- Source identifiers, versions, revisions, upstream URLs
- License names/URLs and access conditions per source
- Translation provenance (method, model or translator, review status)
- Build and verification code

**Not tracked (never committed):**

- Prompt text — harmful or benign, original or translated
- Model completions, plans, tool calls, critic rationales, transcripts
- Raw downloads, intermediate builds, processed model-ready records
- Any credential or gated-access token

`data/raw/`, `data/interim/`, `data/processed/`, `results/`, and `logs/` are
git-ignored for this reason. See `.gitignore` and README §7.

---

## Why prompt text stays out even though the repository is private

Private is not a safety property. Repositories are shared with collaborators,
forked, mirrored into CI, cloned onto laptops, backed up to third parties, and
occasionally flipped to public by accident. Two independent reasons apply, and
either alone is sufficient:

1. **Licensing.** Most red-team benchmarks prohibit redistribution, restrict use
   to research, or gate access behind an agreement. Committing their text —
   even privately, even in translation — is redistribution.
2. **Harm surface.** A curated, aligned, seven-language harmful-prompt corpus is
   a more capable artifact than any monolingual source it was built from.
   Creating one and checking it into a repository is not a neutral act, and the
   repository's visibility setting is not the control that makes it safe.

Derivative status is worth stating plainly: **translations, paraphrases,
filtered subsets, and re-annotations of a licensed dataset are derivative works
and inherit the upstream terms.** Translating a restricted English benchmark
into Hausa does not produce an unrestricted Hausa dataset.

---

## Source registry

Every dataset used must be registered in `configs/data_sources.yaml` **before**
any data is fetched, with all fields below filled in. A source with unresolved
license status is not used.

| Field | Meaning |
|---|---|
| `id` | Stable local identifier used in behavior IDs |
| `name` | Upstream dataset name |
| `version` / `revision` | Pinned version, tag, or commit — never "latest" |
| `url` | Canonical upstream location |
| `license` | SPDX identifier or exact license name |
| `license_url` | Link to the authoritative terms |
| `redistribution_allowed` | `true` / `false` / `unknown` — default `unknown` |
| `requires_agreement` | Whether access is gated by a signed agreement |
| `commercial_use` | As stated upstream |
| `attribution_required` | As stated upstream |
| `contains_harmful_text` | Whether the source contains harmful prompts |
| `citation` | BibTeX key for the report |
| `access_notes` | How to obtain it; never the credential itself |

### Registered sources (as of Stage 0)

Two sources are registered and acquired. They are stored **separately** and are
**not merged** — see "Why these two are not combined" below.

#### 1. JailbreakBench — `JBB-Behaviors`

| Field | Value |
|---|---|
| `id` | `jailbreakbench` |
| Upstream | https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors |
| Code | https://github.com/JailbreakBench/jailbreakbench |
| Revision (pinned) | `886acc352a31533ffbcf4ef22c744658688086fc` |
| License | **MIT** ([LICENSE](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/blob/main/LICENSE)) |
| Redistribution permitted by license | Yes |
| Committed to this repository | **No** — repository policy (see below) |
| Access gate | None; no credential required |
| Contents used | 100 harmful + 100 benign behaviors |
| Paper | [arXiv:2404.01318](https://arxiv.org/abs/2404.01318) |

Structure, **verified rather than assumed**: the dataset exposes **one config,
`behaviors`, with two _splits_ (`harmful`, `benign`)** — *not* two configs by
those names. A second config, `judge_comparison`, exists and is **not** acquired.

#### 2. HarmBench — text behaviors

| Field | Value |
|---|---|
| `id` | `harmbench` |
| Upstream | https://github.com/centerforaisafety/HarmBench |
| Commit (pinned) | `c0423b952435fcc8467108d8f25962dbae5b7de2` |
| License | **MIT** ([LICENSE](https://github.com/centerforaisafety/HarmBench/blob/main/LICENSE)) |
| Redistribution permitted by license | Yes |
| Committed to this repository | **No** — repository policy (see below) |
| Access gate | None; no credential required |
| Contents used | `harmbench_behaviors_text_{all,val,test}.csv` (400 / 80 / 320) |
| Paper | [arXiv:2402.04249](https://arxiv.org/abs/2402.04249) |

Structure, **verified rather than assumed**: the official validation/test
partition is expressed as **separate files**, not as a column — no split field
exists in any HarmBench CSV. `val` (80) and `test` (320) are disjoint and
together reconstruct `text_all` (400).

### License permits ≠ policy permits

Both sources are MIT-licensed, so redistributing their text would be **legally**
permitted. This repository still does not commit it. The licensing objection
falls away; the harm-surface objection in the next section does not, and it is
sufficient on its own. `configs/data_sources.yaml` records these as two separate
fields — `redistribution_allowed: true` (a fact about the license) and
`commit_to_repo: false` (a decision by this project) — so the two are never
conflated.

### Why these two are not combined

**27 of JBB's 100 harmful behaviors carry `Source: TDC/HarmBench`**, and a
further 18 carry `Source: AdvBench`. Concatenating the two sources would
double-count those behaviors and inflate any per-language rate computed over the
union. Stage 0 therefore keeps them in separate directories, with separate
manifests and separate schemas. Any future merge must deduplicate on behavior
semantics first — and the resulting overlap must be reported, not assumed away.

---

## Third-party MODELS (not datasets)

The same principle applies to models used to *produce* corpus content: their
licences are authoritative, are not MIT, and **restrictions propagate to the
outputs**.

### NLLB-200 distilled 600M — the translation backend

| Field | Value |
|---|---|
| Model | `facebook/nllb-200-distilled-600M` |
| Revision (pinned) | `f8d333a098d19b4fd9a8b18f94170487ad3f821d` |
| Licence | **CC-BY-NC-4.0 — NON-COMMERCIAL** |
| Source | https://huggingface.co/facebook/nllb-200-distilled-600M |
| Verified | 2026-08-16 |
| Executed | Locally. No text is sent to any third-party translation API. |

Upstream states, and this project accepts, that the model:

- is **"a research model and is not released for production deployment"**;
- is intended for **single-sentence** translation, not documents;
- was trained on **general-domain** text and is not intended for domain-specific
  material;
- **"cannot be used as certified translations"**;
- may mistranslate, with quality varying sharply by language.

**Consequences for this project, all of them binding:**

1. **Non-commercial use only.** The CC-BY-NC licence covers the model *and*
   propagates to text it generates. Any translated corpus produced with it is a
   derivative and **inherits the non-commercial restriction** — regardless of
   this repository's MIT licence, which covers code only.
2. **Every machine translation requires human review.** Output is recorded as
   `MACHINE_TRANSLATED_UNCHECKED` with `review_required: true`. No automated
   check in this repository can clear that flag; only a **named native-speaker
   reviewer** can, by setting `HUMAN_REVIEWED` with a `reviewer_id`.
3. **Research-only provenance must be stated** in any publication using these
   translations, alongside the languages affected.
4. **Quality is not uniform.** Expect the weakest output for Hausa and Yoruba —
   exactly the languages this project is most interested in. Treat a low score
   there as a possible translation artifact, not automatically a model-safety
   finding (CLAUDE.md §6 on confounds).

If the non-commercial restriction is ever unacceptable for a downstream use, the
translations must be regenerated with a differently-licensed engine; they cannot
be relicensed after the fact.

### GlotLID — language identification

| Field | Value |
|---|---|
| Model | `cis-lmu/glotlid` (fastText, 2102 labels) |
| Licence | **Apache-2.0** (permissive; no NC restriction) |
| Source | https://huggingface.co/cis-lmu/glotlid |
| Verified | 2026-08-16 — all seven target languages have labels |
| Role | **Language-identity diagnostic only** |

GlotLID answers "what language is this?" It does **not** assess translation
adequacy, and it never clears `review_required`. A mismatch is strong evidence
the wrong language was returned; a match is weak evidence only. It is
independent of the translation engine by design — a translator grading its own
output is not evidence.

### Models deliberately NOT used

No commercial translation API (DeepL, Google, Azure) is called: that would
transmit harmful prompt text to a third party. No LLM is used to translate or to
judge translations, and **Llama and Qwen are excluded as judges** by explicit
instruction. No model under evaluation is used to produce or assess its own
evaluation inputs.

---

### Attribution

Both datasets must be cited in any resulting write-up:

```bibtex
@misc{chao2024jailbreakbench,
  title  = {JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models},
  author = {Chao, Patrick and Debenedetti, Edoardo and Robey, Alexander and Andriushchenko, Maksym
            and Croce, Francesco and Sehwag, Vikash and Debenedetti, Edoardo and Flammarion, Nicolas
            and Chiang, Mung and Mittal, Prateek and Hein, Matthias},
  year   = {2024},
  eprint = {2404.01318},
  archivePrefix = {arXiv}
}

@inproceedings{mazeika2024harmbench,
  title     = {HarmBench: A Standardized Evaluation Framework for Automated Red Teaming
               and Robust Refusal},
  author    = {Mazeika, Mantas and Phan, Long and Yin, Xuwang and Zou, Andy and Wang, Zifan
               and Mu, Norman and Sakhaee, Elham and Li, Nathaniel and Basart, Steven
               and Li, Bo and Forsyth, David and Hendrycks, Dan},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024}
}
```

Author lists above are transcribed for convenience and should be checked against
the canonical entries before publication.

---

## Rules

1. Register the source with full license metadata **before** fetching.
2. Pin a version or revision. Never fetch "latest" into a manifest.
3. Never commit prompt text, in any language, original or translated.
4. Never relabel third-party data as MIT, and never let this repo's LICENSE
   appear to cover it.
5. Treat translations and derived subsets as inheriting upstream terms.
6. Honour attribution requirements in every report and publication.
7. Honour deletion, revocation, and takedown requests from upstream maintainers,
   including for locally materialised copies.
8. If `redistribution_allowed` is `unknown`, treat it as `false`.
9. Do not use a gated dataset without completing its access process. Do not
   route around a gate.
10. Record every source's full citation in the report bibliography.

---

## Attribution

Upstream datasets are cited in full in `reports/`. This file is a licensing
notice, not a substitute for citation, and not legal advice.
