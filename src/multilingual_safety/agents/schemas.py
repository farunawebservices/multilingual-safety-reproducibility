"""JSON Schemas for the three agent outputs.

Every schema sets ``additionalProperties: false`` and lists every field in
``required``. Both directions matter:

* an **extra** field means the model invented structure we do not model, and
  silently dropping it would hide that;
* a **missing** field means the output is incomplete, and filling in a default
  would manufacture a decision the model never made.

The schemas describe *shape only*. Turning a validated payload into a stage
decision is :mod:`multilingual_safety.agents.records`, which maps every outcome
onto the six-category taxonomy in :mod:`multilingual_safety.decisions` -- where
malformed and missing are first-class categories that cannot be coerced into
ALLOW or VETO.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from multilingual_safety.languages import LANGUAGE_CODES

PLANNER = "planner"
TOOL_CALLER = "tool_caller"
CRITIC = "critic"
ROLES: tuple[str, ...] = (PLANNER, TOOL_CALLER, CRITIC)

SCHEMA_VERSIONS: dict[str, str] = {
    PLANNER: "planner.v1",
    TOOL_CALLER: "tool_caller.v1",
    CRITIC: "critic.v1",
}

#: The field each role uses to express its stage decision. Checked for presence
#: *before* full validation so "no decision at all" is distinguishable from
#: "decision present but malformed".
DECISION_FIELD: dict[str, str] = {
    PLANNER: "final_decision",
    TOOL_CALLER: "decision",
    CRITIC: "final_decision",
}

TERMINATION_VALUES: tuple[str, ...] = ("complete", "malformed", "timeout")
RISK_LEVELS: tuple[str, ...] = ("low", "medium", "high", "critical")

_LANGUAGE = {"type": "string", "enum": list(LANGUAGE_CODES)}
_BEHAVIOR_ID = {"type": "string", "pattern": "^beh_[0-9a-f]{12}$"}
_TERMINATION = {"type": "string", "enum": list(TERMINATION_VALUES)}

PLANNER_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "planner.v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version", "behavior_id", "language", "decision", "risk_categories",
        "risk_level", "plan_steps", "user_safe_response", "final_decision", "termination",
    ],
    "properties": {
        "schema_version": {"const": "planner.v1"},
        "behavior_id": _BEHAVIOR_ID,
        "language": _LANGUAGE,
        "decision": {"type": "string", "enum": ["REFUSE", "PLAN"]},
        "risk_categories": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "risk_level": {"type": "string", "enum": list(RISK_LEVELS)},
        "plan_steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["step_id", "description", "tool_name", "arguments"],
                "properties": {
                    "step_id": {"type": "string", "minLength": 1},
                    "description": {"type": "string", "minLength": 1},
                    # null means "no tool for this step", which is different from
                    # an empty string and must stay distinguishable.
                    "tool_name": {"type": ["string", "null"], "minLength": 1},
                    "arguments": {"type": "object"},
                },
            },
        },
        "user_safe_response": {"type": "string"},
        "final_decision": {"type": "string", "enum": ["VETO", "ALLOW"]},
        "termination": _TERMINATION,
    },
}

TOOL_CALLER_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "tool_caller.v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version", "behavior_id", "language", "decision", "tool_name",
        "arguments", "authorization_basis", "safety_check", "termination",
    ],
    "properties": {
        "schema_version": {"const": "tool_caller.v1"},
        "behavior_id": _BEHAVIOR_ID,
        "language": _LANGUAGE,
        "decision": {"type": "string", "enum": ["REFUSE", "CALL", "NO_CALL"]},
        "tool_name": {"type": ["string", "null"], "minLength": 1},
        "arguments": {"type": "object"},
        "authorization_basis": {"type": "string"},
        "safety_check": {
            "type": "object",
            "additionalProperties": False,
            "required": ["harmful_intent_detected", "policy_conflict"],
            "properties": {
                "harmful_intent_detected": {"type": "boolean"},
                "policy_conflict": {"type": "boolean"},
            },
        },
        "termination": _TERMINATION,
    },
}

CRITIC_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "critic.v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version", "behavior_id", "language", "final_decision",
        "harmful_intent_detected", "planner_error", "tool_caller_error",
        "failure_type", "rationale", "termination",
    ],
    "properties": {
        "schema_version": {"const": "critic.v1"},
        "behavior_id": _BEHAVIOR_ID,
        "language": _LANGUAGE,
        "final_decision": {"type": "string", "enum": ["VETO", "ALLOW"]},
        "harmful_intent_detected": {"type": "boolean"},
        "planner_error": {"type": "boolean"},
        "tool_caller_error": {"type": "boolean"},
        "failure_type": {
            "type": ["string", "null"],
            "enum": [
                None, "semantic_misjudgment", "missing_decision",
                "malformed_output", "nontermination",
            ],
        },
        "rationale": {"type": "string"},
        "termination": _TERMINATION,
    },
}

SCHEMAS: Mapping[str, dict[str, Any]] = {
    PLANNER: PLANNER_SCHEMA,
    TOOL_CALLER: TOOL_CALLER_SCHEMA,
    CRITIC: CRITIC_SCHEMA,
}


class UnknownRoleError(ValueError):
    """Raised for a role with no registered schema."""


def get_schema(role: str) -> dict[str, Any]:
    try:
        return SCHEMAS[role]
    except KeyError as exc:
        raise UnknownRoleError(f"unknown role {role!r}; known: {list(ROLES)}") from exc


def validate_payload(role: str, payload: Any) -> list[str]:
    """Return schema violations for ``payload``. Empty list means valid.

    Errors are returned rather than raised so a caller can record *every*
    problem with an output instead of only the first.
    """
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(get_schema(role))
    return [
        f"{'/'.join(str(p) for p in error.path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    ]


def is_valid(role: str, payload: Any) -> bool:
    return not validate_payload(role, payload)
