# ICLR 2027 nine-page plan

The main paper is limited to nine pages excluding references and appendix. Figures and tables count toward the nine-page limit.

## Target allocation

1. Page 1: title, abstract, overview figure, opening introduction.
2. Page 2: introduction contributions, related work, background/problem setting.
3. Page 3: methods and experimental setup.
4. Page 4: primary results and compact cross-model table/figure.
5. Page 5: Llama role-strength results.
6. Page 6: decision-composition results and main interpretation.
7. Page 7: cross-model discussion and exploratory-audit summary.
8. Page 8: discussion, limitations, and conclusion.
9. Page 9: overflow buffer for tables, captions, and formatting variation.

## Layout policy

- Use four main figures: overview, role-strength forest, decision composition, and cross-model effects.
- Keep full per-model plots in the supplement.
- Use at most two compact main-paper tables: role-strength results and cross-model primary effects.
- Do not include closed-model figures in the main paper.
- Do not include causal-patching, calibration, or adversarial figures in the main paper.
- If compilation exceeds nine pages, move the cross-model table or decision-composition figure to the supplement before shrinking typography.
- Preserve readable captions and axis labels; do not solve page overflow by using illegible figures.

## Verification commands

```bash
pdflatex -interaction=nonstopmode paper_main.tex
bibtex paper_main
pdflatex -interaction=nonstopmode paper_main.tex
pdflatex -interaction=nonstopmode paper_main.tex
pdfinfo paper_main.pdf | grep Pages
```

The final submission must report nine or fewer main-paper pages before the references.