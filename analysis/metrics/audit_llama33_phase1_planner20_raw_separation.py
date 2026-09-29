from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import polars as pl

RAW = Path(
    "artifacts/prior_llama33_phase2/results/"
    "baseline_completed_backup (1).jsonl"
)

CANONICAL = Path("data/derived/canonical/llama33_phase1_main.parquet")

MODEL = "meta-llama/llama-3.3-70b-instruct"
LANGUAGES = ["Arabic", "English", "Hausa", "Yoruba"]

EXPECTED_PER_CONDITION = 1695
EXPECTED_PRIMARY_HARMFUL = 1200
EXPECTED_PRIMARY_BENIGN = 495

OUT = Path(
    "paper_artifacts/metrics/"
    "llama33_planner20_primary_raw_separation_audit.csv"
)

def sha256_file(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()

if not RAW.exists():
    raise FileNotFoundError(f"Missing raw source: {RAW}")

if not CANONICAL.exists():
    raise FileNotFoundError(f"Missing canonical input: {CANONICAL}")

canonical = (
    pl.read_parquet(CANONICAL)
    .filter(
        (pl.col("model") == MODEL)
        & (pl.col("config") == "planner_only")
        & (pl.col("alpha").is_in([0.0, 20.0]))
    )
)

required_canonical = {
    "record_id",
    "source_file",
    "source_sha256",
    "language",
    "prompt_class",
    "config",
    "alpha",
}

missing_canonical = sorted(required_canonical - set(canonical.columns))

if missing_canonical:
    raise ValueError(
        f"Canonical input misses columns: {missing_canonical}"
    )

if canonical.height != 2 * EXPECTED_PER_CONDITION:
    raise ValueError(
        "Unexpected Phase 1 planner-only canonical rows: "
        f"{canonical.height}."
    )

canonical_sources = canonical.select(
    pl.col("source_file").unique(),
    pl.col("source_sha256").unique(),
)

if canonical_sources["source_file"].to_list() != [str(RAW)]:
    raise ValueError(
        "Planner-only canonical subset does not reference the expected "
        "combined raw source."
    )

raw_sha256 = sha256_file(RAW)

if canonical_sources["source_sha256"].to_list() != [raw_sha256]:
    raise ValueError(
        "Raw source SHA-256 does not match canonical provenance."
    )

raw_rows = []

with RAW.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        line = line.strip()

        if not line:
            continue

        row = json.loads(line)

        raw_rows.append(
            {
                "raw_line": line_number,
                "record_id": (
                    row.get("record_id")
                    or row.get("id")
                    or row.get("prompt_id")
                ),
                "config": row.get("config"),
                "alpha": (
                    row.get("alpha")
                    if row.get("alpha") is not None
                    else row.get("strength")
                ),
                "language": row.get("language"),
                "prompt_class": (
                    row.get("prompt_class")
                    or row.get("harmfulness")
                    or row.get("category")
                ),
            }
        )

raw = pl.DataFrame(raw_rows)

if raw.height == 0:
    raise ValueError("Raw source contained no JSON records.")

if raw["record_id"].null_count() > 0:
    raise ValueError("At least one raw record has no recoverable record ID.")

raw_planner = raw.filter(
    (pl.col("config") == "planner_only")
    & (pl.col("alpha").is_in([0.0, 20.0]))
)

if raw_planner.height != 2 * EXPECTED_PER_CONDITION:
    raise ValueError(
        "Unexpected raw planner-only alpha 0/20 records: "
        f"{raw_planner.height}."
    )

rows = []

for alpha in [0.0, 20.0]:
    raw_subset = raw_planner.filter(pl.col("alpha") == alpha)

    canonical_subset = canonical.filter(pl.col("alpha") == alpha)

    raw_ids = set(raw_subset["record_id"].to_list())
    canonical_ids = set(canonical_subset["record_id"].to_list())

    if raw_subset.height != EXPECTED_PER_CONDITION:
        raise ValueError(
            f"Raw alpha={alpha} count is {raw_subset.height}; "
            f"expected {EXPECTED_PER_CONDITION}."
        )

    if canonical_subset.height != EXPECTED_PER_CONDITION:
        raise ValueError(
            f"Canonical alpha={alpha} count is {canonical_subset.height}; "
            f"expected {EXPECTED_PER_CONDITION}."
        )

    if len(raw_ids) != EXPECTED_PER_CONDITION:
        raise ValueError(
            f"Raw alpha={alpha} IDs are not unique: {len(raw_ids)}."
        )

    if len(canonical_ids) != EXPECTED_PER_CONDITION:
        raise ValueError(
            f"Canonical alpha={alpha} IDs are not unique: "
            f"{len(canonical_ids)}."
        )

    if raw_ids != canonical_ids:
        raise ValueError(
            f"Raw and canonical record-ID sets differ for alpha={alpha}."
        )

    language_counts = Counter(raw_subset["language"].to_list())
    prompt_counts = Counter(raw_subset["prompt_class"].to_list())

    rows.append(
        {
            "alpha": alpha,
            "raw_rows": raw_subset.height,
            "raw_unique_record_ids": len(raw_ids),
            "canonical_rows": canonical_subset.height,
            "canonical_unique_record_ids": len(canonical_ids),
            "exact_raw_canonical_id_match": raw_ids == canonical_ids,
            "language_counts": json.dumps(language_counts, sort_keys=True),
            "prompt_class_counts": json.dumps(prompt_counts, sort_keys=True),
            "raw_source_sha256": raw_sha256,
        }
    )

baseline = canonical.filter(pl.col("alpha") == 0.0)
steered = canonical.filter(pl.col("alpha") == 20.0)

for prompt_class, expected_n in [
    ("harmful", EXPECTED_PRIMARY_HARMFUL),
    ("benign", EXPECTED_PRIMARY_BENIGN),
]:
    baseline_ids = set(
        baseline
        .filter(
            (pl.col("prompt_class") == prompt_class)
            & (pl.col("language").is_in(LANGUAGES))
        )
        ["record_id"]
        .to_list()
    )

    steered_ids = set(
        steered
        .filter(
            (pl.col("prompt_class") == prompt_class)
            & (pl.col("language").is_in(LANGUAGES))
        )
        ["record_id"]
        .to_list()
    )

    if len(baseline_ids) != expected_n:
        raise ValueError(
            f"Baseline {prompt_class} IDs: {len(baseline_ids)}; "
            f"expected {expected_n}."
        )

    if len(steered_ids) != expected_n:
        raise ValueError(
            f"Steered {prompt_class} IDs: {len(steered_ids)}; "
            f"expected {expected_n}."
        )

    if baseline_ids != steered_ids:
        raise ValueError(
            f"Baseline and steered {prompt_class} IDs do not match."
        )

audit = pl.DataFrame(rows).sort("alpha")
OUT.parent.mkdir(parents=True, exist_ok=True)
audit.write_csv(OUT)

print(f"Wrote: {OUT}")
print(f"Raw source: {RAW}")
print(f"Raw source SHA-256: {raw_sha256}")
print(f"Raw planner-only alpha 0/20 rows: {raw_planner.height}")
print(f"Raw alpha=0 rows: {raw_planner.filter(pl.col('alpha') == 0.0).height}")
print(f"Raw alpha=20 rows: {raw_planner.filter(pl.col('alpha') == 20.0).height}")
print(f"Canonical planner-only alpha 0/20 rows: {canonical.height}")
print(f"Exact raw-to-canonical subsets: {audit['exact_raw_canonical_id_match'].all()}")
print(f"Matched harmful prompt IDs: {EXPECTED_PRIMARY_HARMFUL}")
print(f"Matched benign prompt IDs: {EXPECTED_PRIMARY_BENIGN}")
print("Combined source file separates baseline and steering: True")
