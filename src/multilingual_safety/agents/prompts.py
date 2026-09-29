"""Build role prompts from v2 dataset rows.

The one mapping rule that matters
---------------------------------
**``translated_text`` is the prompt. ``source_text`` is never sent to a model.**

In the frozen v2 corpus ``source_text`` holds the English canonical and is
*identical across all seven language rows of a behavior* (verified: 200/200).
Sending it in a Hausa trial would put the English rendering in front of the
model and silently destroy RQ1. ``prompt_hash`` is ``sha256(NFC(translated_text))``
for all 1,400 rows, so the builder recomputes it and refuses on mismatch.

Fields that are never prompted: ``harm_label`` (the ground truth), ``category``
(an analysis stratifier), ``split_placeholder`` (``UNASSIGNED`` on every row --
the split comes only from ``splits_v2.json``), and all translation provenance.
Provenance is carried onto the *result* row instead, so the confound stays
visible without leaking into the measurement (CLAUDE.md §6).

Template provenance
-------------------
Instruction scaffolding is a measurement instrument. Machine-translating it into
six languages and shipping it unreviewed would mean a Yoruba refusal could be
caused by our own broken Yoruba instructions rather than by the model -- exactly
the ambiguity ``languages.py`` exists to prevent for the corpus. So templates are
loaded from disk with a provenance manifest recording, per template, who wrote it
and whether a competent speaker reviewed it. A template that is absent raises
:class:`MissingPromptTemplateError` naming the exact path. Nothing is generated
on the fly and no language silently falls back to English.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multilingual_safety.agents.schemas import ROLES, SCHEMA_VERSIONS, get_schema
from multilingual_safety.hashing import canonical_json, file_sha256, text_sha256
from multilingual_safety.languages import (
    DIACRITIC_SENTINELS,
    LANGUAGE_CODES,
    validate_language_code,
)
from multilingual_safety.paths import CONFIGS_DIR

PROMPTS_DIR = CONFIGS_DIR / "prompts"
MANIFEST_NAME = "prompt_templates.manifest.json"

_SECTION = re.compile(r"^---\s*(system|user)\s*---\s*$", re.MULTILINE)
_PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")

#: Fields a role prompt may never contain. Asserted after rendering.
FORBIDDEN_IN_PROMPT: tuple[str, ...] = ("harm_label", "split_placeholder")


class PromptError(RuntimeError):
    """Base class for prompt-construction failures."""


class MissingPromptTemplateError(PromptError):
    """A required template file does not exist. Never silently substituted."""


class PromptProvenanceError(PromptError):
    """A template is absent from the manifest, or its hash drifted."""


class PromptIntegrityError(PromptError):
    """A rendered prompt failed an integrity check (hash, orthography, leakage)."""


@dataclass(frozen=True, slots=True)
class TemplateProvenance:
    """Who wrote a template and whether anyone competent checked it."""

    template_id: str
    role: str
    language: str
    sha256: str
    authorship: str            # original_en | human_native | machine_unreviewed
    reviewer_id: str | None
    note: str = ""

    @property
    def is_reviewed(self) -> bool:
        """Never inferred from authorship: no reviewer means unreviewed."""
        return bool(self.reviewer_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "role": self.role,
            "language": self.language,
            "sha256": self.sha256,
            "authorship": self.authorship,
            "reviewer_id": self.reviewer_id,
            "reviewed": self.is_reviewed,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class BuiltPrompt:
    """A rendered prompt plus everything needed to record it text-free."""

    messages: tuple[Mapping[str, str], ...]
    prompt_id: str
    role: str
    template_id: str
    template_sha256: str
    stage_language: str
    prompt_language: str
    behavior_id: str
    prompt_hash: str
    orthography_fingerprint: str
    provenance: TemplateProvenance
    tool_results_sha256: str = ""

    def to_metadata_dict(self) -> dict[str, Any]:
        """Text-free projection. Carries no message content."""
        return {
            "prompt_id": self.prompt_id,
            "role": self.role,
            "template_id": self.template_id,
            "template_sha256": self.template_sha256,
            "stage_language": self.stage_language,
            "prompt_language": self.prompt_language,
            "behavior_id": self.behavior_id,
            "prompt_hash": self.prompt_hash,
            "orthography_fingerprint": self.orthography_fingerprint,
            "tool_results_sha256": self.tool_results_sha256,
            "template_provenance": self.provenance.to_dict(),
        }


def orthography_fingerprint(text: str, language: str) -> str:
    """Digest of the contrastive characters present, so loss is detectable.

    Records *which* sentinel characters survived, not how many, so that
    "the model failed in Yoruba" stays distinguishable from "preprocessing
    stripped the tone marks before it ever saw them" (README §2).
    """
    sentinels = DIACRITIC_SENTINELS.get(language, ())
    present = sorted({ch for ch in sentinels if ch in text})
    return text_sha256(
        canonical_json({"language": language, "present": present, "nfc": True})
    )[:16]


class PromptLibrary:
    """Loads role templates and renders them against v2 dataset rows."""

    def __init__(
        self,
        directory: Path = PROMPTS_DIR,
        *,
        provenance: Mapping[str, TemplateProvenance] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self._provenance: dict[str, TemplateProvenance] = dict(provenance or {})
        self._cache: dict[str, tuple[str, str]] = {}

    # --- loading ----------------------------------------------------------
    @classmethod
    def load(cls, directory: Path = PROMPTS_DIR) -> PromptLibrary:
        directory = Path(directory)
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.exists():
            raise PromptProvenanceError(
                f"prompt provenance manifest not found: {manifest_path}. Every template "
                "must declare who wrote it and whether a competent speaker reviewed it; "
                "an unregistered template is not usable as a measurement instrument."
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance = {
            key: TemplateProvenance(
                template_id=key,
                role=entry["role"],
                language=entry["language"],
                sha256=entry["sha256"],
                authorship=entry["authorship"],
                reviewer_id=entry.get("reviewer_id"),
                note=entry.get("note", ""),
            )
            for key, entry in payload["templates"].items()
        }
        return cls(directory, provenance=provenance)

    def template_path(self, role: str, language: str) -> Path:
        return self.directory / f"{role}.{language}.md"

    def available(self) -> dict[str, list[str]]:
        """Languages with a template on disk, per role."""
        return {
            role: sorted(
                language
                for language in LANGUAGE_CODES
                if self.template_path(role, language).exists()
            )
            for role in ROLES
        }

    def assert_available(self, roles: Sequence[str], languages: Sequence[str]) -> None:
        """Refuse up front if any (role, language) template is missing.

        Called before the model loads, so a missing template costs nothing.
        """
        missing = [
            str(self.template_path(role, language))
            for role in roles
            for language in languages
            if not self.template_path(role, language).exists()
        ]
        if missing:
            raise MissingPromptTemplateError(
                f"{len(missing)} role prompt template(s) do not exist:\n  "
                + "\n  ".join(sorted(missing))
                + "\n\nThese are not generated automatically and no language falls back "
                "to English: instruction scaffolding is part of the instrument, and an "
                "unreviewed machine translation of it would confound every result in "
                "that language (CLAUDE.md §6). Author each file, then register its "
                f"sha256 and reviewer in {self.directory / MANIFEST_NAME}."
            )

    def _read(self, role: str, language: str) -> tuple[str, str]:
        key = f"{role}.{language}"
        if key in self._cache:
            return self._cache[key]

        path = self.template_path(role, language)
        if not path.exists():
            self.assert_available([role], [language])       # raises with the full message

        provenance = self._provenance.get(key)
        if provenance is None:
            raise PromptProvenanceError(
                f"template {key} is not registered in {self.directory / MANIFEST_NAME}. "
                "An unregistered template has no recorded authorship or review status."
            )
        actual = file_sha256(path)
        if actual != provenance.sha256:
            raise PromptProvenanceError(
                f"template {key} hash drifted: manifest records {provenance.sha256}, "
                f"file hashes to {actual}. Re-register it with its review status rather "
                "than editing an instrument in place."
            )

        text = path.read_text(encoding="utf-8")
        parts = _SECTION.split(text)
        sections: dict[str, str] = {}
        for index in range(1, len(parts) - 1, 2):
            sections[parts[index]] = parts[index + 1].strip()
        if "system" not in sections or "user" not in sections:
            raise PromptError(
                f"template {key} must contain '--- system ---' and '--- user ---' sections"
            )
        self._cache[key] = (sections["system"], sections["user"])
        return self._cache[key]

    def provenance_for(self, role: str, language: str) -> TemplateProvenance:
        self._read(role, language)
        return self._provenance[f"{role}.{language}"]

    # --- rendering --------------------------------------------------------
    def build(
        self,
        role: str,
        row: Mapping[str, Any],
        *,
        stage_language: str,
        model_id: str,
        model_revision: str,
        tool_registry: Any = None,
        plan: Mapping[str, Any] | None = None,
        tool_calls: Sequence[Mapping[str, Any]] | None = None,
        tool_results: Sequence[Any] | None = None,
    ) -> BuiltPrompt:
        """Render one role prompt from one v2 row."""
        if role not in ROLES:
            raise PromptError(f"unknown role {role!r}; expected one of {list(ROLES)}")
        prompt_language = str(row["language"])
        validate_language_code(prompt_language)
        validate_language_code(stage_language)

        # The prompt is the target-language rendering, never the English canonical.
        user_request = unicodedata.normalize("NFC", str(row["translated_text"]))
        recomputed = text_sha256(user_request)
        if recomputed != row["prompt_hash"]:
            raise PromptIntegrityError(
                f"{row['behavior_id']}/{prompt_language}: prompt_hash mismatch. The row "
                f"records {row['prompt_hash']}, translated_text hashes to {recomputed}. "
                "The corpus and the prompt disagree (CLAUDE.md §10)."
            )

        system_template, user_template = self._read(role, stage_language)
        substitutions = {
            "BEHAVIOR_ID": str(row["behavior_id"]),
            "LANGUAGE": prompt_language,
            "STAGE_LANGUAGE": stage_language,
            "SCHEMA_VERSION": SCHEMA_VERSIONS[role],
            "SCHEMA_JSON": json.dumps(get_schema(role), indent=2, ensure_ascii=False),
            "USER_REQUEST": user_request,
            "TOOL_REGISTRY": _render_registry(tool_registry),
            "PLAN_JSON": json.dumps(plan or {}, indent=2, ensure_ascii=False),
            "TOOL_CALLS_JSON": json.dumps(list(tool_calls or []), indent=2, ensure_ascii=False),
            "TOOL_RESULTS_JSON": _render_tool_results(tool_results),
        }
        # Canonical form of the mock results, hashed into the prompt id below.
        # Hashed from the payload rather than the rendered string so the identity
        # tracks the DATA the stage saw, not incidental formatting.
        tool_results_digest = text_sha256(canonical_json(_canonical_results(tool_results)))
        system = _substitute(system_template, substitutions, role, stage_language)
        user = _substitute(user_template, substitutions, role, stage_language)

        _assert_no_leakage(system, user, row)

        provenance = self._provenance[f"{role}.{stage_language}"]
        messages = (
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        )
        prompt_id = text_sha256(
            canonical_json(
                {
                    "role": role,
                    "template_id": provenance.template_id,
                    "template_sha256": provenance.sha256,
                    "stage_language": stage_language,
                    "prompt_language": prompt_language,
                    "behavior_id": row["behavior_id"],
                    "prompt_hash": row["prompt_hash"],
                    "model_id": model_id,
                    "model_revision": model_revision,
                    "schema_version": SCHEMA_VERSIONS[role],
                    # Downstream context changes the prompt, so it changes the id.
                    "plan_sha256": text_sha256(substitutions["PLAN_JSON"]),
                    "tool_calls_sha256": text_sha256(substitutions["TOOL_CALLS_JSON"]),
                    # Without this, two Critic prompts differing only in their mock
                    # tool results hashed identically, so prompt_id could not attest
                    # what the Critic actually saw.
                    "tool_results_sha256": tool_results_digest,
                }
            )
        )
        return BuiltPrompt(
            messages=messages,
            prompt_id=prompt_id,
            role=role,
            template_id=provenance.template_id,
            template_sha256=provenance.sha256,
            stage_language=stage_language,
            prompt_language=prompt_language,
            behavior_id=str(row["behavior_id"]),
            prompt_hash=str(row["prompt_hash"]),
            orthography_fingerprint=orthography_fingerprint(user, prompt_language),
            provenance=provenance,
            tool_results_sha256=tool_results_digest,
        )


def _substitute(template: str, values: Mapping[str, str], role: str, language: str) -> str:
    rendered = _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), template)
    unresolved = sorted(set(_PLACEHOLDER.findall(rendered)))
    if unresolved:
        raise PromptError(
            f"template {role}.{language} has unresolved placeholder(s) {unresolved}"
        )
    return rendered


def _assert_no_leakage(system: str, user: str, row: Mapping[str, Any]) -> None:
    """The ground-truth label and the English canonical must not be in the prompt."""
    blob = f"{system}\n{user}"
    for field in FORBIDDEN_IN_PROMPT:
        if field in blob:
            raise PromptIntegrityError(
                f"rendered prompt names {field!r}; the ground-truth label is never "
                "shown to a role"
            )
    source = unicodedata.normalize("NFC", str(row.get("source_text") or ""))
    translated = unicodedata.normalize("NFC", str(row["translated_text"]))
    if row["language"] != "en" and source and source in blob and source != translated:
        raise PromptIntegrityError(
            f"{row['behavior_id']}/{row['language']}: the English canonical source_text "
            "reached a non-English prompt. Only translated_text is ever sent."
        )


def _render_registry(registry: Any) -> str:
    if registry is None:
        return "(no tools available)"
    described = registry.describe()
    lines = [
        "The following tools exist. They are MOCKED: deciding to call one records a "
        "decision and produces a deterministic placeholder result. Nothing is executed.",
        "",
    ]
    for tool in described["tools"]:
        required = ", ".join(tool["required_arguments"]) or "(none)"
        optional = ", ".join(tool["optional_arguments"]) or "(none)"
        lines.append(
            f"- {tool['name']} [{tool['capability_class']}]: {tool['description']} "
            f"required={required} optional={optional}"
        )
    return "\n".join(lines)


def _canonical_results(results: Sequence[Any] | None) -> list[dict[str, Any]]:
    """Normalise mock results to plain, sortable dicts for hashing."""
    if not results:
        return []
    return [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in results]


def _render_tool_results(results: Sequence[Any] | None) -> str:
    if not results:
        return "(no tool results)"
    return json.dumps(_canonical_results(results), indent=2, ensure_ascii=False)
