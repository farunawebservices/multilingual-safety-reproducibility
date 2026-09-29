"""Agent output schemas, parsing, and the mock tool executor.

The load-bearing assertions are negative: malformed, missing, timed-out and
non-terminating output must never become ALLOW or VETO, and the raw output must
survive parsing unchanged.
"""

from __future__ import annotations

import json

import pytest

from multilingual_safety.agents.records import (
    DECISION_MAP,
    ExecutionRecord,
    RawOutput,
    looks_non_terminating,
    parse_agent_output,
)
from multilingual_safety.agents.schemas import (
    CRITIC,
    DECISION_FIELD,
    PLANNER,
    ROLES,
    SCHEMA_VERSIONS,
    TOOL_CALLER,
    UnknownRoleError,
    get_schema,
    is_valid,
    validate_payload,
)
from multilingual_safety.agents.tools import (
    ALLOW_REAL_EXECUTION,
    DEFAULT_TOOLS,
    MockToolRegistry,
    ToolError,
    UnknownToolError,
)
from multilingual_safety.decisions import DecisionCategory, DecisionCoercionError, coerce_to_binary
from multilingual_safety.languages import LANGUAGE_CODES

BEHAVIOR = "beh_0123456789ab"
LANG = "ha"
CONTEXT = {
    "run_id": "run_20260817T000000Z_abcdef12",
    "behavior_id": BEHAVIOR,
    "language": LANG,
    "model_id": "test-model",
    "prompt_hash": "a" * 64,
    "seed": 20260816,
}


def planner_payload(**overrides):
    payload = {
        "schema_version": "planner.v1",
        "behavior_id": BEHAVIOR,
        "language": LANG,
        "decision": "PLAN",
        "risk_categories": ["privacy"],
        "risk_level": "low",
        "plan_steps": [
            {"step_id": "s1", "description": "look it up",
             "tool_name": "web_search", "arguments": {"query": "network safety"}}
        ],
        "user_safe_response": "Here is a summary.",
        "final_decision": "ALLOW",
        "termination": "complete",
    }
    payload.update(overrides)
    return payload


def tool_caller_payload(**overrides):
    payload = {
        "schema_version": "tool_caller.v1",
        "behavior_id": BEHAVIOR,
        "language": LANG,
        "decision": "CALL",
        "tool_name": "web_search",
        "arguments": {"query": "network safety"},
        "authorization_basis": "benign informational request",
        "safety_check": {"harmful_intent_detected": False, "policy_conflict": False},
        "termination": "complete",
    }
    payload.update(overrides)
    return payload


def critic_payload(**overrides):
    payload = {
        "schema_version": "critic.v1",
        "behavior_id": BEHAVIOR,
        "language": LANG,
        "final_decision": "ALLOW",
        "harmful_intent_detected": False,
        "planner_error": False,
        "tool_caller_error": False,
        "failure_type": None,
        "rationale": "no policy conflict",
        "termination": "complete",
    }
    payload.update(overrides)
    return payload


PAYLOADS = {PLANNER: planner_payload, TOOL_CALLER: tool_caller_payload, CRITIC: critic_payload}


def parse(role, payload_or_text, **overrides):
    text = payload_or_text if isinstance(payload_or_text, str) else json.dumps(payload_or_text)
    context = {**CONTEXT, **overrides}
    return parse_agent_output(RawOutput(text=text, role=role), **context)


# --- schemas ---------------------------------------------------------------
def test_three_roles_are_registered():
    assert ROLES == (PLANNER, TOOL_CALLER, CRITIC)
    assert set(SCHEMA_VERSIONS) == set(ROLES)


@pytest.mark.parametrize("role", ROLES)
def test_valid_payload_passes(role):
    assert validate_payload(role, PAYLOADS[role]()) == []
    assert is_valid(role, PAYLOADS[role]())


@pytest.mark.parametrize("role", ROLES)
def test_extra_field_is_rejected(role):
    errors = validate_payload(role, PAYLOADS[role](unexpected_field="x"))
    assert errors and any("unexpected_field" in e or "Additional" in e for e in errors)


@pytest.mark.parametrize("role", ROLES)
def test_every_field_is_required(role):
    payload = PAYLOADS[role]()
    for field in list(payload):
        incomplete = {k: v for k, v in payload.items() if k != field}
        assert validate_payload(role, incomplete), f"{role}: removing {field} should fail"


@pytest.mark.parametrize("role", ROLES)
def test_schema_version_is_pinned(role):
    assert validate_payload(role, PAYLOADS[role](schema_version="wrong.v9"))


@pytest.mark.parametrize("role", ROLES)
def test_unknown_language_is_rejected_by_schema(role):
    assert validate_payload(role, PAYLOADS[role](language="fr"))


@pytest.mark.parametrize("role", ROLES)
def test_malformed_behavior_id_is_rejected(role):
    assert validate_payload(role, PAYLOADS[role](behavior_id="12345"))


def test_planner_rejects_bad_enum_values():
    assert validate_payload(PLANNER, planner_payload(decision="MAYBE"))
    assert validate_payload(PLANNER, planner_payload(risk_level="spicy"))
    assert validate_payload(PLANNER, planner_payload(final_decision="PROBABLY"))


def test_planner_plan_step_requires_all_keys():
    assert validate_payload(PLANNER, planner_payload(plan_steps=[{"step_id": "s1"}]))


def test_planner_tool_name_may_be_null():
    payload = planner_payload(plan_steps=[
        {"step_id": "s1", "description": "think", "tool_name": None, "arguments": {}}
    ])
    assert validate_payload(PLANNER, payload) == []


def test_tool_caller_safety_check_is_strict():
    assert validate_payload(TOOL_CALLER, tool_caller_payload(safety_check={}))
    assert validate_payload(
        TOOL_CALLER,
        tool_caller_payload(safety_check={"harmful_intent_detected": "yes", "policy_conflict": False}),
    )


def test_critic_failure_type_vocabulary():
    for value in ("semantic_misjudgment", "missing_decision", "malformed_output", "nontermination"):
        assert validate_payload(CRITIC, critic_payload(failure_type=value)) == []
    assert validate_payload(CRITIC, critic_payload(failure_type="other"))


def test_unknown_role_raises():
    with pytest.raises(UnknownRoleError):
        get_schema("referee")


# --- valid records ---------------------------------------------------------
@pytest.mark.parametrize("role", ROLES)
def test_valid_record_is_usable(role):
    record = parse(role, PAYLOADS[role]())
    assert record.category is DecisionCategory.SEMANTIC_ALLOW
    assert record.is_usable
    assert record.validation_errors == ()
    assert record.termination == "complete"


def test_refusal_maps_to_veto():
    assert parse(PLANNER, planner_payload(decision="REFUSE", final_decision="VETO")).category is (
        DecisionCategory.SEMANTIC_VETO
    )
    assert parse(TOOL_CALLER, tool_caller_payload(decision="REFUSE", tool_name=None,
                                                  arguments={})).category is (
        DecisionCategory.SEMANTIC_VETO
    )
    assert parse(CRITIC, critic_payload(final_decision="VETO")).category is (
        DecisionCategory.SEMANTIC_VETO
    )


def test_no_call_is_allow_but_the_verbatim_value_survives():
    """NO_CALL did not block, but must stay distinguishable from CALL."""
    record = parse(TOOL_CALLER, tool_caller_payload(decision="NO_CALL", tool_name=None, arguments={}))
    assert record.category is DecisionCategory.SEMANTIC_ALLOW
    assert record.stage_decision_value == "NO_CALL"
    assert DECISION_MAP[TOOL_CALLER]["NO_CALL"] is DecisionCategory.SEMANTIC_ALLOW


# --- malformed -------------------------------------------------------------
@pytest.mark.parametrize("text", ["not json at all", "{broken", "", "[1,2,3]", "null"])
def test_malformed_output_is_malformed_not_a_decision(text):
    record = parse(CRITIC, text)
    assert record.category is DecisionCategory.MALFORMED_DECISION
    assert record.parsed is None or not isinstance(record.parsed, dict) or record.validation_errors
    with pytest.raises(DecisionCoercionError):
        coerce_to_binary(record.category)


@pytest.mark.parametrize("role", ROLES)
def test_unreadable_decision_value_is_malformed(role):
    field = DECISION_FIELD[role]
    record = parse(role, PAYLOADS[role](**{field: "PROBABLY_FINE"}))
    assert record.category is DecisionCategory.MALFORMED_DECISION
    assert record.stage_decision_value == "PROBABLY_FINE"


def test_model_reported_malformed_termination():
    record = parse(CRITIC, critic_payload(termination="malformed"))
    assert record.category is DecisionCategory.MALFORMED_DECISION
    assert record.termination == "malformed"


def test_schema_violation_is_malformed_not_allow():
    record = parse(CRITIC, critic_payload(rationale=123))
    assert record.category is DecisionCategory.MALFORMED_DECISION
    assert record.validation_errors


# --- missing ---------------------------------------------------------------
@pytest.mark.parametrize("role", ROLES)
def test_missing_decision_field_is_its_own_category(role):
    payload = PAYLOADS[role]()
    del payload[DECISION_FIELD[role]]
    record = parse(role, payload)
    assert record.category is DecisionCategory.MISSING_DECISION
    assert record.category is not DecisionCategory.MALFORMED_DECISION
    with pytest.raises(DecisionCoercionError):
        coerce_to_binary(record.category)


def test_missing_is_distinguishable_from_malformed():
    """Different causes, different per-language rates; never merged."""
    missing = parse(CRITIC, {k: v for k, v in critic_payload().items() if k != "final_decision"})
    malformed = parse(CRITIC, critic_payload(final_decision="???"))
    assert missing.category is DecisionCategory.MISSING_DECISION
    assert malformed.category is DecisionCategory.MALFORMED_DECISION


def test_empty_object_is_missing_decision():
    assert parse(CRITIC, {}).category is DecisionCategory.MISSING_DECISION


# --- timeout and non-termination -------------------------------------------
def test_runtime_timeout_beats_content():
    record = parse_agent_output(
        RawOutput(text=json.dumps(critic_payload()), role=CRITIC, timed_out=True), **CONTEXT
    )
    assert record.category is DecisionCategory.TIMEOUT


def test_model_reported_timeout():
    assert parse(CRITIC, critic_payload(termination="timeout")).category is DecisionCategory.TIMEOUT


def test_non_termination_detected_by_repetition():
    degenerate, evidence = looks_non_terminating("alpha beta gamma delta " * 500)
    assert degenerate
    assert evidence["max_ngram_repeats"] > evidence["max_repeat_ngram"]
    assert parse(CRITIC, "alpha beta gamma delta " * 500).category is (
        DecisionCategory.NON_TERMINATION
    )


def test_normal_output_is_not_flagged_as_degenerate():
    degenerate, _ = looks_non_terminating(json.dumps(critic_payload()))
    assert not degenerate


# --- wrong language / behavior ---------------------------------------------
def test_wrong_language_record_is_flagged_and_unusable():
    record = parse(CRITIC, critic_payload(language="yo"), language="ha")
    assert record.language_mismatch is True
    assert record.is_usable is False
    # The decision was still expressed; the category is not silently rewritten.
    assert record.category is DecisionCategory.SEMANTIC_ALLOW
    assert record.evidence["declared_language"] == "yo"
    assert record.evidence["expected_language"] == "ha"


def test_wrong_behavior_id_is_flagged_and_unusable():
    record = parse(CRITIC, critic_payload(behavior_id="beh_ffffffffffff"))
    assert record.behavior_mismatch is True
    assert record.is_usable is False


@pytest.mark.parametrize("language", [c for c in LANGUAGE_CODES if c != "en"])
def test_every_target_language_parses(language):
    record = parse(CRITIC, critic_payload(language=language), language=language)
    assert record.is_usable
    assert record.language == language


def test_caller_supplied_bad_language_raises():
    with pytest.raises(ValueError, match="unknown language code"):
        parse(CRITIC, critic_payload(), language="fr")


# --- duplicates ------------------------------------------------------------
def test_duplicate_json_keys_keep_the_last_value_and_stay_detectable():
    """Duplicate keys are legal JSON; the parser must not crash on them."""
    text = '{"a": 1, "a": 2}'
    record = parse(CRITIC, text)
    assert record.category is DecisionCategory.MISSING_DECISION


def test_duplicate_records_are_detectable_by_identity():
    a = parse(CRITIC, critic_payload())
    b = parse(CRITIC, critic_payload())
    key = lambda r: (r.run_id, r.role, r.behavior_id, r.language)  # noqa: E731
    assert key(a) == key(b), "two records for the same slot collide on their key"
    assert a.raw_sha256 == b.raw_sha256


def test_duplicate_plan_step_ids_are_visible_in_the_payload():
    payload = planner_payload(plan_steps=[
        {"step_id": "s1", "description": "a", "tool_name": None, "arguments": {}},
        {"step_id": "s1", "description": "b", "tool_name": None, "arguments": {}},
    ])
    record = parse(PLANNER, payload)
    ids = [s["step_id"] for s in record.parsed["plan_steps"]]
    assert len(ids) != len(set(ids))


# --- raw is preserved separately -------------------------------------------
def test_raw_is_never_mutated_by_parsing():
    text = json.dumps(critic_payload())
    raw = RawOutput(text=text, role=CRITIC)
    before = raw.text
    parse_agent_output(raw, **CONTEXT)
    assert raw.text == before


def test_record_references_raw_by_hash_not_by_value():
    from multilingual_safety.hashing import text_sha256

    text = json.dumps(critic_payload())
    record = parse(CRITIC, text)
    assert record.raw_sha256 == text_sha256(text)
    assert record.raw_n_chars == len(text)
    assert not hasattr(record, "raw_text")


def test_raw_survives_even_when_parsing_fails():
    raw = RawOutput(text="{broken", role=CRITIC)
    record = parse_agent_output(raw, **CONTEXT)
    assert record.category is DecisionCategory.MALFORMED_DECISION
    assert raw.text == "{broken"
    assert record.raw_sha256 == raw.sha256


def test_metadata_projection_drops_the_parsed_payload():
    record = parse(CRITIC, critic_payload(rationale="a private rationale string"))
    payload = record.to_metadata_dict()
    assert "parsed" not in payload
    assert payload["parsed_present"] is True
    assert len(payload["parsed_sha256"]) == 64
    assert "a private rationale string" not in json.dumps(payload)


# --- outer execution record -------------------------------------------------
@pytest.mark.parametrize(
    "field", ["behavior_id", "language", "model_id", "role", "run_id", "seed", "prompt_hash"]
)
def test_execution_record_carries_every_required_outer_field(field):
    payload = parse(CRITIC, critic_payload()).to_dict()
    assert field in payload, f"execution record is missing {field}"


def test_execution_record_rejects_unknown_role():
    with pytest.raises(UnknownRoleError):
        ExecutionRecord(
            run_id="r", role="referee", behavior_id=BEHAVIOR, language=LANG, model_id="m",
            seed=1, prompt_hash="a" * 64, raw_sha256="b" * 64, raw_n_chars=1, parsed=None,
            decision_category="SEMANTIC_ALLOW", stage_decision_value=None,
            schema_version=None, termination=None,
        )


def test_record_converts_to_a_stage_decision():
    decision = parse(CRITIC, critic_payload()).to_stage_decision()
    assert decision.stage == CRITIC
    assert decision.category is DecisionCategory.SEMANTIC_ALLOW


# --- mock tool executor -----------------------------------------------------
def test_no_real_execution_is_possible():
    assert ALLOW_REAL_EXECUTION is False
    registry = MockToolRegistry()
    result = registry.execute("web_search", {"query": "x"})
    assert result.mock is True
    assert result.executed_for_real is False
    assert result.output.startswith("[mock:web_search]")


def test_executor_is_deterministic_across_calls_and_arg_order():
    registry = MockToolRegistry()
    a = registry.execute("http_request", {"method": "GET", "url": "https://example.org"})
    b = registry.execute("http_request", {"url": "https://example.org", "method": "GET"})
    assert a.result_id == b.result_id == registry.execute(
        "http_request", {"method": "GET", "url": "https://example.org"}
    ).result_id


def test_different_arguments_give_different_results():
    registry = MockToolRegistry()
    assert registry.execute("web_search", {"query": "a"}).result_id != registry.execute(
        "web_search", {"query": "b"}
    ).result_id


def test_unknown_tool_raises_rather_than_no_op():
    with pytest.raises(UnknownToolError, match="unknown tool"):
        MockToolRegistry().execute("launch_missiles", {})


def test_bad_arguments_are_recorded_not_raised():
    result = MockToolRegistry().execute("send_email", {"to": "x"})
    assert result.ok is False
    assert any("missing required" in p for p in result.problems)
    assert "not executed" in result.output


def test_unexpected_arguments_are_reported():
    result = MockToolRegistry().execute("read_file", {"path": "/tmp/x", "sudo": True})
    assert any("unexpected argument" in p for p in result.problems)


def test_registry_describes_itself_as_mock():
    payload = MockToolRegistry().describe()
    assert payload["allow_real_execution"] is False
    assert payload["n_tools"] == len(DEFAULT_TOOLS)
    note = payload["note"].lower()
    assert "mock" in note and "never" in note.replace("no tool is ever", "never")


def test_capability_classes_are_labelled():
    registry = MockToolRegistry()
    assert registry.get("send_email").capability_class == "outward_facing"
    assert registry.get("run_code").capability_class == "code_execution"
    assert registry.get("web_search").capability_class == "read_only"


def test_duplicate_tool_names_rejected():
    from multilingual_safety.agents.tools import MockTool

    with pytest.raises(ToolError, match="duplicate tool name"):
        MockToolRegistry((MockTool("a", "x"), MockTool("a", "y")))


def test_execute_plan_skips_toolless_steps_and_reports_unknown_tools():
    registry = MockToolRegistry()
    results, problems = registry.execute_plan([
        {"step_id": "s1", "tool_name": None, "arguments": {}},
        {"step_id": "s2", "tool_name": "web_search", "arguments": {"query": "x"}},
        {"step_id": "s3", "tool_name": "nonexistent", "arguments": {}},
    ])
    assert len(results) == 1 and results[0].tool_name == "web_search"
    assert len(problems) == 1 and "s3" in problems[0]


def test_planner_plan_can_be_executed_against_the_mock_registry():
    record = parse(PLANNER, planner_payload())
    results, problems = MockToolRegistry().execute_plan(record.parsed["plan_steps"])
    assert not problems
    assert all(r.mock for r in results)
