#!/usr/bin/env python3
"""Extract Addition 6 prompt filter baseline metrics."""
import json
from pathlib import Path
import polars as pl

NO_INT_JSON = Path("addition_6_prompt_filter_baseline/addition6_llama_full_outputs/no_intervention_metadata.json")
PROMPT_FILTER_JSON = Path("addition_6_prompt_filter_baseline/addition6_llama_full_outputs/prompt_filter_metadata.json")
STEERING_JSON = Path("addition_6_prompt_filter_baseline/addition6_llama_full_outputs/planner_steering_l50_s20_historical_all_steps_metadata.json")

OUT_CSV = Path("paper_artifacts/metrics/addition6_prompt_filter_baseline_metrics.csv")

def load_metrics(path):
    with open(path) as f:
        data = json.load(f)
    outcomes = data["outcome_counts"]
    parse_status = data["critic_parse_status_counts"]
    return {
        "condition": data["condition"],
        "harmful_total": outcomes["harmful_total"],
        "harmful_vetoes": outcomes["harmful_vetoes"],
        "harmful_veto_rate": outcomes["harmful_vetoes"] / outcomes["harmful_total"],
        "benign_total": outcomes["benign_total"],
        "benign_vetoes": outcomes["benign_vetoes"],
        "benign_veto_rate": outcomes["benign_vetoes"] / outcomes["benign_total"],
        "missing_labels": parse_status["missing_exact_label"],
    }

rows = [load_metrics(NO_INT_JSON), load_metrics(PROMPT_FILTER_JSON), load_metrics(STEERING_JSON)]

df = pl.DataFrame(rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.write_csv(OUT_CSV)

print(f"Wrote: {OUT_CSV}")
print(f"Conditions: {len(rows)}")
for r in rows:
    print(f"  {r['condition']}: harmful veto {r['harmful_veto_rate']*100:.1f}%, benign veto {r['benign_veto_rate']*100:.1f}%")
