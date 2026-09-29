# Paper Artifact Pipeline

All paper-facing quantities must follow this chain:

raw experiment artifact
-> format-specific adapter
-> canonical Parquet
-> audited metrics CSV
-> generated LaTeX table and figure
-> results manifest
-> verified writing handoff

## Rules

1. Raw inputs are immutable and registered by path and SHA-256.
2. Each source schema uses a format-specific adapter.
3. Canonical Parquet files contain normalized fields plus source provenance.
4. Metrics distinguish VETO, ALLOW, malformed output, parse failure, and non-termination.
5. Tables and figures are generated only from audited metrics.
6. Every manuscript claim must map to one or more rows in the claim-evidence matrix.
7. Addition 1 is reported as an inconclusive causal-patching result.
8. Addition 2 is reported as a retrospective role-specificity analysis, not a matched random-vector control.
9. Each addition retains its original experimental scope and limitations.
