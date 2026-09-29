# ICLR 2027 assembly notes

## Canonical branch

`iclr2027-role-dependent-revision`

## Scientific scope

The revised manuscript treats within-phase Planner-only comparisons as primary, the Llama alpha-30 role-strength package as a secondary cross-phase contrast, and the remaining additions as exploratory or audit analyses.

## Assembly requirements

1. Use the official ICLR 2027 style file when available.
2. Compile `paper_main.tex` from the repository root.
3. Include the authoritative generated tables and figures only.
4. Verify all `\\ref`, `\\cref`, citation, and bibliography keys.
5. Check that percentages are not multiplied by 100 twice.
6. Run the numerical claim registry against the canonical metric CSVs.
7. Build the supplement independently.
8. Check the main-paper page limit and anonymization.

## Required figures

- Pipeline and role-specific overview.
- Llama role-strength effect plot.
- Decision-composition plot.
- Canonical cross-model primary effect plot.

The repository currently contains authoritative primary-effect PDFs and the role-strength comparison figure. New overview and composition plots should be generated from canonical CSVs rather than manually edited.

## Claim boundaries

Do not claim causal localization, mechanistic explanation, matched random-vector specificity, universal multilingual transfer, or adversarial improvement from the zero-jailbreak audit.
