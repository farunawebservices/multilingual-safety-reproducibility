# Prompt-clustered reanalysis

Reproduces every interval and p-value reported in the paper's main text and
appendix tables from the canonical record tables in `data/derived/canonical/`.

Each harmful cell holds 100 unique prompts executed in three pipeline runs
(300 records). Repeat runs are not independent, so the unique prompt is the
resampling and randomization unit:

- 95% percentile intervals from 10,000 cluster-bootstrap resamples of prompts
  (seed 20260908);
- two-sided prompt-level paired sign-flip randomization tests (10,000 draws);
- Holm correction within the paper's predeclared families.

Record-level McNemar tests and intervals are reported alongside for comparison.

```bash
python analysis/reanalysis/clustered_reanalysis.py \
    analysis/reanalysis . analysis/reanalysis/clustered_reanalysis.csv
python analysis/reanalysis/draw_figure1.py <output_dir>   # Figure 1
python analysis/reanalysis/draw_figure2.py <output_dir>   # Figure 2
```

`minipq.py` is a dependency-free Parquet reader (requires the system
`libzstd`), included so the reanalysis runs without `pyarrow`.

Prompt text and model outputs are not distributed with this package.
