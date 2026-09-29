from pathlib import Path

import polars as pl

BASELINE = Path(
    "data/derived/canonical/gpt_oss_20b_phase1_baseline.parquet"
)
STEERED = Path(
    "data/derived/canonical/gpt_oss_20b_phase1_planner_only_alpha1.parquet"
)
OUT = Path(
    "paper_artifacts/metrics/"
    "gpt_oss_20b_planner_alpha1_primary_canonical_audit.csv"
)

EXPECTED_MODEL = "openai/gpt-oss-20b"
LANGUAGES = ["Arabic", "English", "Hausa", "Yoruba"]

for path in [BASELINE, STEERED]:
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical input: {path}")

baseline = pl.read_parquet(BASELINE)
steered = pl.read_parquet(STEERED)

required = {
    "record_id",
    "model",
    "language",
    "prompt_class",
    "config",
    "intervention_target",
    "alpha",
    "operational_failure",
}

for name, frame in [("baseline", baseline), ("steered", steered)]:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

def summarize(name: str, frame: pl.DataFrame) -> dict:
    models = frame.get_column("model").unique().sort().to_list()
    configs = frame.get_column("config").unique().sort().to_list()
    targets = frame.get_column("intervention_target").unique().sort().to_list()
    alphas = frame.get_column("alpha").unique().sort().to_list()

    if models != [EXPECTED_MODEL]:
        raise ValueError(f"{name} models: {models}")

    return {
        "condition": name,
        "rows": frame.height,
        "unique_record_ids": frame.get_column("record_id").n_unique(),
        "duplicate_record_ids": frame.height - frame.get_column("record_id").n_unique(),
        "models": "|".join(models),
        "configs": "|".join(str(value) for value in configs),
        "intervention_targets": "|".join(str(value) for value in targets),
        "alphas": "|".join(str(value) for value in alphas),
        "harmful_rows": frame.filter(
            pl.col("prompt_class") == "harmful"
        ).height,
        "benign_rows": frame.filter(
            pl.col("prompt_class") == "benign"
        ).height,
    }

baseline_harmful = baseline.filter(
    pl.col("prompt_class") == "harmful"
).filter(
    pl.col("language").is_in(LANGUAGES)
)

steered_harmful = steered.filter(
    pl.col("prompt_class") == "harmful"
).filter(
    pl.col("language").is_in(LANGUAGES)
)

baseline_ids = set(baseline_harmful.get_column("record_id").to_list())
steered_ids = set(steered_harmful.get_column("record_id").to_list())

if len(baseline_ids) != 1200:
    raise ValueError(
        f"Expected 1200 baseline harmful IDs; found {len(baseline_ids)}."
    )

if len(steered_ids) != 1200:
    raise ValueError(
        f"Expected 1200 steered harmful IDs; found {len(steered_ids)}."
    )

if baseline_ids != steered_ids:
    raise ValueError(
        "Baseline and steered harmful record-ID sets do not match exactly."
    )

audit = pl.DataFrame(
    [
        summarize("baseline", baseline),
        summarize("planner_only_alpha1", steered),
        {
            "condition": "paired_harmful_primary_subset",
            "rows": len(baseline_ids),
            "unique_record_ids": len(baseline_ids),
            "duplicate_record_ids": 0,
            "models": EXPECTED_MODEL,
            "configs": "baseline_vs_planner_only_alpha1",
            "intervention_targets": "planner_only",
            "alphas": "1.0",
            "harmful_rows": len(baseline_ids),
            "benign_rows": 0,
        },
    ]
)

OUT.parent.mkdir(parents=True, exist_ok=True)
audit.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Baseline canonical rows: {baseline.height}")
print(f"Steered canonical rows: {steered.height}")
print(f"Matched harmful primary IDs: {len(baseline_ids)}")
print(f"Languages: {', '.join(LANGUAGES)}")
print("Complete matched harmful coverage: True")
