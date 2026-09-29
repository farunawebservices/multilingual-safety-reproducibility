# results/ — GIT-IGNORED

Per-run outputs, keyed by run ID:

```
results/<run_id>/
  run_metadata.json     run ID, config hash, git state, split hash, seeds, env
  trials.jsonl          per-trial, per-stage decision records
  summary.json          counts and rates across all six decision categories
```

**Empty: no model has been run.**

Rules:

- Nothing is written to a bare filename. A result without a run ID is not a
  result (README §6).
- Runs are never overwritten. A repeated configuration mints a new run ID; the
  difference between the two runs measures nondeterminism and is itself data.
- Summaries report **all six** decision categories over the number of
  **attempted** trials — not over the trials that happened to parse
  (CLAUDE.md §2, §3).
- Model completions and transcripts are not committed. Sync this directory to
  durable storage; do not add it to git.

Mint a run ID with:

```bash
python scripts/new_run.py --config configs/experiment.example.yaml --dry-run
```
