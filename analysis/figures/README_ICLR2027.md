# ICLR 2027 figure generation

Run from the repository root:

```bash
python analysis/figures/write_iclr2027_main_figures.py
```

The script reads only canonical metric files and writes PDF/PNG figures to `paper_artifacts/figures/`.

## Main-paper figures

- `iclr2027_figure1_pipeline_overview`: pipeline diagram plus compact role-strength effect inset.
- `iclr2027_figure2_role_strength_forest`: six Llama alpha-30 cross-phase contrasts.
- `iclr2027_figure3_decision_composition`: VETO/ALLOW/no-decision stacked bars.
- `iclr2027_figure4_cross_model_effects`: canonical Planner-only primary effects.

All percentage-point fields are used directly. The script does not multiply values already stored in percentage units by 100. Source schema and row-count assertions should be added before final submission.

The existing per-model figures remain useful for the supplement, but the four compact figures above are the intended main-paper set under the nine-page limit.