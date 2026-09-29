#!/usr/bin/env python3
"""Write Addition 4 calibration Opus handoff."""
import hashlib
from pathlib import Path

METRICS_CSV = Path("paper_artifacts/metrics/addition4_judge_calibration_metrics.csv")
PAPER_TABLE_CSV = Path("paper_artifacts/metrics/addition4_calibration_paper_table.csv")
PAPER_TABLE_TEX = Path("paper_artifacts/tables/addition4_calibration_agreement.tex")
FIGURE_PDF = Path("paper_artifacts/figures/addition4_calibration_kappa_by_language.pdf")
FIGURE_PNG = Path("paper_artifacts/figures/addition4_calibration_kappa_by_language.png")
RAW_DATA = Path("addition_4_calibration/final_consensus_annotations.jsonl")
CALIBRATION_MD = Path("addition_4_calibration/ADDITION_4_JUDGE_CALIBRATION_RESULTS.md")

OUT_HANDOFF = Path("paper_artifacts/metrics/addition4_calibration_OPUS_HANDOFF.md")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

hashes = {
    "metrics": sha256(METRICS_CSV),
    "table_csv": sha256(PAPER_TABLE_CSV),
    "table_tex": sha256(PAPER_TABLE_TEX),
    "figure_pdf": sha256(FIGURE_PDF),
    "figure_png": sha256(FIGURE_PNG),
    "raw_data": sha256(RAW_DATA),
}

handoff = f"""# Addition 4: Judge Calibration Opus Handoff

## Purpose
This handoff provides the authoritative artifact chain for writing about Addition 4 (judge calibration study) in the ICLR paper.

## Nature of Addition 4
Addition 4 is a **descriptive inter-rater reliability study**, NOT a model intervention experiment. It measures annotation agreement quality, not model behavior changes.

## Authoritative numerical source
All numerical reporting MUST come from:
- `paper_artifacts/metrics/addition4_judge_calibration_metrics.csv`
  - SHA-256: `{hashes['metrics']}`
  - 8 rows: 7 languages + overall
  - 98 total prompts (14 per language)

## Complete artifact chain
1. **Raw annotation data:** `addition_4_calibration/final_consensus_annotations.jsonl` (SHA: `{hashes['raw_data'][:16]}...`)
2. **Metrics CSV:** `paper_artifacts/metrics/addition4_judge_calibration_metrics.csv`
3. **Paper table CSV:** `paper_artifacts/metrics/addition4_calibration_paper_table.csv` (SHA: `{hashes['table_csv'][:16]}...`)
4. **Paper table LaTeX:** `paper_artifacts/tables/addition4_calibration_agreement.tex`
5. **Figure PDF:** `paper_artifacts/figures/addition4_calibration_kappa_by_language.pdf`
6. **Figure PNG:** `paper_artifacts/figures/addition4_calibration_kappa_by_language.png`
7. **Methodology doc:** `addition_4_calibration/ADDITION_4_JUDGE_CALIBRATION_RESULTS.md`

## Key results
- **Overall Cohen's κ = 0.98** (95% CI: 0.94–1.00)
- **Observed agreement: 99%** (97/98 prompts)
- **1 disagreement** resolved by blinded adjudication
- **All languages** meet κ ≥ 0.60 target
- **Interpretation:** "Almost perfect" inter-rater agreement

## Allowed qualitative summary
"Two independent annotators achieved almost perfect agreement (Cohen's κ = 0.98, 95% CI: 0.94–1.00) across 98 prompts in 7 languages. A single disagreement was resolved by blinded adjudication. All languages exceeded the κ ≥ 0.60 target, indicating the annotation scheme was interpreted consistently."

## Explicitly forbidden claims
- Do NOT claim this measures model performance
- Do NOT claim this is a causal intervention
- Do NOT generalize beyond these 98 specific prompts
- Do NOT claim annotators were blinded to original labels (they were NOT)

## Limitations (REQUIRED in all discussions)
"Annotators were not blinded to the original labels; this may have inflated agreement. The calibration set is relatively small (98 prompts). The adjudicator saw only disagreement cases."

## Placement recommendation
Methods appendix or reproducibility section, NOT a main results section.

## Citation format
Cite the metrics CSV path and SHA-256 hash when reporting numbers.
"""

OUT_HANDOFF.parent.mkdir(parents=True, exist_ok=True)
OUT_HANDOFF.write_text(handoff, encoding="utf-8")

print(f"Wrote: {OUT_HANDOFF}")
