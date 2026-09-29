#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

SRC = Path("data/derived/canonical/addition1_llama33_hausa_patching_v3.parquet")
OUT = Path("paper_artifacts/metrics/addition1_llama33_hausa_patching_v3_audit.csv")

df = pd.read_parquet(SRC)

required = {
    "source_path",
    "source_sha256",
    "base_id",
    "clean_decision",
    "corrupted_decision",
    "patched_decision",
    "evaluable_clean_ne_corrupted",
    "recovered_patched_eq_clean_among_evaluable",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Canonical Parquet is missing columns: {sorted(missing)}")
if len(df) != 11:
    raise ValueError(f"Expected 11 rows, found {len(df)}")
if df["base_id"].duplicated().any():
    raise ValueError("base_id must be unique")

evaluable = df["evaluable_clean_ne_corrupted"].astype(bool)
recovered = df.loc[evaluable, "recovered_patched_eq_clean_among_evaluable"].astype(bool)

metrics = [
    ("canonical_rows", len(df)),
    ("unique_base_ids", df["base_id"].nunique()),
    ("source_paths", df["source_path"].nunique()),
    ("source_sha256_values", df["source_sha256"].nunique()),
    ("clean_allow", int((df["clean_decision"] == "ALLOW").sum())),
    ("clean_veto", int((df["clean_decision"] == "VETO").sum())),
    ("corrupted_allow", int((df["corrupted_decision"] == "ALLOW").sum())),
    ("corrupted_veto", int((df["corrupted_decision"] == "VETO").sum())),
    ("patched_allow", int((df["patched_decision"] == "ALLOW").sum())),
    ("patched_veto", int((df["patched_decision"] == "VETO").sum())),
    ("evaluable_clean_ne_corrupted", int(evaluable.sum())),
    ("recovered_patched_eq_clean", int(recovered.sum())),
    (
        "recovery_rate_among_evaluable",
        (float(recovered.mean()) if len(recovered) else None),
    ),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(metrics, columns=["metric", "value"]).to_csv(OUT, index=False)

print(f"Wrote {OUT}")
print(pd.read_csv(OUT).to_string(index=False))
