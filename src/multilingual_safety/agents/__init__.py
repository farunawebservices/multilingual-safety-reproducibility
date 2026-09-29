"""Planner -> Tool-Caller -> Critic interfaces.

**No model inference happens in this package.** Every concrete backend raises
``NotImplementedError``. What is defined here is the *contract*: what each stage
receives, what it must emit, and the invariant that every stage emits exactly
one of the six decision categories with no default and no silent recoding
(CLAUDE.md §3).

Defining the contract before wiring a model is deliberate: it fixes what gets
recorded before there is any temptation to record only what happened to work.
"""

from multilingual_safety.agents import records, schemas, tools
from multilingual_safety.agents.base import (
    AgentStage,
    CriticInput,
    PlannerInput,
    StageOutput,
    ToolCallerInput,
    TrialResult,
)
from multilingual_safety.agents.critic import Critic
from multilingual_safety.agents.planner import Planner
from multilingual_safety.agents.records import (
    ExecutionRecord,
    RawOutput,
    looks_non_terminating,
    parse_agent_output,
)
from multilingual_safety.agents.schemas import (
    CRITIC_SCHEMA,
    PLANNER_SCHEMA,
    ROLES,
    TOOL_CALLER_SCHEMA,
    validate_payload,
)
from multilingual_safety.agents.tool_caller import ToolCaller
from multilingual_safety.agents.tools import (
    ALLOW_REAL_EXECUTION,
    MockTool,
    MockToolRegistry,
    MockToolResult,
)

__all__ = [
    "ALLOW_REAL_EXECUTION",
    "CRITIC_SCHEMA",
    "PLANNER_SCHEMA",
    "ROLES",
    "TOOL_CALLER_SCHEMA",
    "AgentStage",
    "Critic",
    "CriticInput",
    "ExecutionRecord",
    "MockTool",
    "MockToolRegistry",
    "MockToolResult",
    "Planner",
    "PlannerInput",
    "RawOutput",
    "StageOutput",
    "ToolCaller",
    "ToolCallerInput",
    "TrialResult",
    "looks_non_terminating",
    "parse_agent_output",
    "records",
    "schemas",
    "tools",
    "validate_payload",
]
