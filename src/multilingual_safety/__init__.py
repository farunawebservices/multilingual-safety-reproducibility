"""Cross-lingual safety evaluation for multi-agent (Planner/Tool-Caller/Critic) systems.

Scaffold + Stage 0/1: this package defines identifiers, schemas, hashing, and
invariants. It performs no model inference. See README.md and CLAUDE.md.

Every submodule is imported here so that ``import multilingual_safety`` exposes
the full surface, and so a packaging regression (a subpackage silently dropped
from the wheel) fails loudly at import rather than at first use.
"""

from multilingual_safety import (
    agents,
    decisions,
    duplicates,
    hashing,
    ids,
    langcheck,
    languages,
    lid,
    manifest,
    paths,
    providers,
    review,
    run_context,
    schemas,
    secrets,
    splits,
    translation,
)
from multilingual_safety.decisions import (
    DECISION_CATEGORIES,
    DecisionCategory,
    StageDecision,
    is_failure_category,
)
from multilingual_safety.duplicates import DuplicateReport, duplicate_summary
from multilingual_safety.hashing import (
    canonical_json,
    deterministic_digest,
    file_sha256,
    json_sha256,
    table_digest,
    text_sha256,
)
from multilingual_safety.ids import (
    behavior_id,
    content_hash,
    record_id,
    run_id,
)
from multilingual_safety.langcheck import LanguageCheck, check_translation
from multilingual_safety.languages import (
    LANGUAGE_CODES,
    PIVOT_LANGUAGE,
    Language,
    load_languages,
)
from multilingual_safety.lid import GlotLID, LIDResult, requires_human_review
from multilingual_safety.manifest import BehaviorRecord, ManifestError
from multilingual_safety.providers import (
    ResponseLog,
    TranslationRequest,
    TranslationResponse,
    get_provider,
    make_request_id,
)
from multilingual_safety.review import ReviewDecision, ReviewerCompetence, can_clear_human_review
from multilingual_safety.run_context import RunContext, stage_report_header
from multilingual_safety.schemas import EXPECTED_SCHEMAS, SchemaMismatchError
from multilingual_safety.secrets import SecretFinding, assert_no_secrets, scan_text
from multilingual_safety.splits import SplitAccessError, load_split
from multilingual_safety.translation import (
    SPLIT_UNASSIGNED,
    TranslationRow,
    TranslationStatus,
)

__version__ = "0.0.1"

#: Every submodule that ships with the distribution. `tests/test_packaging.py`
#: asserts this matches what is actually importable, so a module added to the
#: source tree but dropped from the wheel is caught.
SUBMODULES: tuple[str, ...] = (
    "agents",
    "agents.base",
    "agents.critic",
    "agents.planner",
    "agents.records",
    "agents.schemas",
    "agents.tool_caller",
    "agents.tools",
    "decisions",
    "duplicates",
    "freeze",
    "hashing",
    "ids",
    "langcheck",
    "languages",
    "lid",
    "manifest",
    "paths",
    "providers",
    "review",
    "run_context",
    "schemas",
    "secrets",
    "splits",
    "translation",
)

__all__ = [
    "DECISION_CATEGORIES",
    "EXPECTED_SCHEMAS",
    "LANGUAGE_CODES",
    "PIVOT_LANGUAGE",
    "SPLIT_UNASSIGNED",
    "SUBMODULES",
    "BehaviorRecord",
    "DecisionCategory",
    "DuplicateReport",
    "GlotLID",
    "LIDResult",
    "Language",
    "LanguageCheck",
    "ManifestError",
    "ResponseLog",
    "ReviewDecision",
    "ReviewerCompetence",
    "RunContext",
    "SchemaMismatchError",
    "SecretFinding",
    "SplitAccessError",
    "StageDecision",
    "TranslationRequest",
    "TranslationResponse",
    "TranslationRow",
    "TranslationStatus",
    "__version__",
    "agents",
    "assert_no_secrets",
    "behavior_id",
    "can_clear_human_review",
    "canonical_json",
    "check_translation",
    "content_hash",
    "decisions",
    "deterministic_digest",
    "duplicate_summary",
    "duplicates",
    "file_sha256",
    "get_provider",
    "hashing",
    "ids",
    "is_failure_category",
    "json_sha256",
    "langcheck",
    "languages",
    "lid",
    "load_languages",
    "load_split",
    "make_request_id",
    "manifest",
    "paths",
    "providers",
    "record_id",
    "requires_human_review",
    "review",
    "run_context",
    "run_id",
    "scan_text",
    "schemas",
    "secrets",
    "splits",
    "stage_report_header",
    "table_digest",
    "text_sha256",
    "translation",
]
