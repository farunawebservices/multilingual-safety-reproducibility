#!/usr/bin/env python3
"""Extract Addition 4 judge calibration metrics from raw data."""
import json
from pathlib import Path
import polars as pl

CALIBRATION_JSON = Path("addition_4_calibration/calibration_metrics.json")
FINAL_CONSENSUS = Path("addition_4_calibration/final_consensus_annotations.jsonl")
OUT_CSV = Path("paper_artifacts/metrics/addition4_judge_calibration_metrics.csv")

# Load calibration metrics
with open(CALIBRATION_JSON) as f:
    metrics = json.load(f)

# Build per-language rows
rows = []
for lang, data in metrics["per_language"].items():
    rows.append({
        "language_code": lang,
        "language_name": data["language_name"],
        "n_prompts": data["n"],
        "annotator_1_yes": data["annotator_1_yes"],
        "annotator_1_no": data["annotator_1_no"],
        "annotator_2_yes": data["annotator_2_yes"],
        "annotator_2_no": data["annotator_2_no"],
        "observed_agreement": data["observed_agreement"],
        "cohens_kappa": data["kappa"],
        "ci95_low": data["bootstrap_95_ci"][0],
        "ci95_high": data["bootstrap_95_ci"][1],
        "interpretation": data["interpretation"],
        "meets_target_kappa_0_60": data["meets_per_language_target_kappa_0_60"],
    })

# Add overall row
rows.append({
    "language_code": "overall",
    "language_name": "Overall",
    "n_prompts": metrics["overall"]["n"],
    "annotator_1_yes": metrics["overall"]["annotator_1_yes"],
    "annotator_1_no": metrics["overall"]["annotator_1_no"],
    "annotator_2_yes": metrics["overall"]["annotator_2_yes"],
    "annotator_2_no": metrics["overall"]["annotator_2_no"],
    "observed_agreement": metrics["overall"]["observed_agreement"],
    "cohens_kappa": metrics["overall"]["kappa"],
    "ci95_low": metrics["overall"]["bootstrap_95_ci"][0],
    "ci95_high": metrics["overall"]["bootstrap_95_ci"][1],
    "interpretation": metrics["overall"]["interpretation"],
    "meets_target_kappa_0_60": metrics["overall"]["meets_overall_target_kappa_0_70"],
})

df = pl.DataFrame(rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.write_csv(OUT_CSV)

print(f"Wrote: {OUT_CSV}")
print(f"Rows: {len(rows)}")
print(f"Overall kappa: {metrics['overall']['kappa']:.4f}")
