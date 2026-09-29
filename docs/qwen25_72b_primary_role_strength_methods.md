# Qwen 2.5-72B Primary Role-Strength Analysis

## Scope

This package analyzes three Qwen 2.5-72B source-result files: a baseline run,
a critic-only/both-role steering run, and a planner-only steering run. Addition 3
is excluded because it has a separate six-field source schema and has not been
validated as the same experimental population or outcome protocol.

## Canonical data

The canonical dataset is:

`data/derived/canonical/qwen25_72b_primary_role_strength.parquet`

It contains privacy-safe analysis fields only: model, source-file group, record ID,
language, prompt class, configuration, steering strength (`alpha`), veto status,
operational failure, and aggregate-verifiable critic marker indicators. It excludes
prompts and planner, critic, and tool response text.

## Outcome definition

`is_vetoed = true` is treated as an explicit veto. This is validated against the
presence of a `VETO` marker in the critic response for every analysis cell.

`is_vetoed = false` is treated as non-veto operational failure. Non-veto records
are labeled `NON_VETO_UNRESOLVED`; they are not presumed to be explicit ALLOW,
because explicit ALLOW markers do not cover all non-veto records.

## Pairing and coverage

Each positive-alpha steered cell is paired with baseline records by record ID,
within language and prompt class. All 33 steered cells have their steered IDs
represented in baseline. Thirty cells have complete matching baseline and steered
ID sets. Three cells use a matched subset and are explicitly flagged with
`partial_baseline_coverage = true`.

## Inference

For each of 33 positive-alpha steered cells, the analysis reports:

- Matched baseline and steered non-veto operational-failure counts and rates.
- The paired failure-rate change in percentage points.
- Discordant-pair counts.
- Two-sided exact McNemar p-values.
- 95 percent paired bootstrap confidence intervals using 10,000 replicates and
  seed 20260908.
- Holm-adjusted p-values across the family of all 33 primary Qwen comparisons,
  controlling family-wise error at 0.05.

## Reporting constraints

Report harmful and benign prompt classes separately. Treat incomplete-coverage
cells as matched-subset comparisons. Do not describe non-veto records as explicit
ALLOW, parse failures, or non-terminations without additional source validation.
