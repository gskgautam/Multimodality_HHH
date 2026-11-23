# Figure 4c — training-seed repeatability (matched initialization)

Exploratory full-selector results on previously used Ch3Ef common-valid records with automated-judge labels. The five outer folds and grouped validation assignments are fixed. The three runs use matched module-wise initialization and are the same full-selector runs used as the reference in the corrected Figure 4a ablations.

Per-seed intervals are group-bootstrap sampling uncertainty. The across-seed range is descriptive variation over these three training runs and is not a confidence interval over new datasets.

| Training seed | Metric | n | Score (%) | Group-bootstrap 95% CI (%) |
|---:|---|---:|---:|---:|
| 20261111 | WR | 498 | 38.15 | [31.61, 44.95] |
| 20261111 | SS | 445 | 19.10 | [15.44, 22.83] |
| 20261111 | TI | 298 | 87.03 | [81.43, 92.12] |
| 20261112 | WR | 498 | 37.75 | [32.79, 43.48] |
| 20261112 | SS | 445 | 22.47 | [18.61, 26.46] |
| 20261112 | TI | 298 | 86.40 | [80.94, 91.52] |
| 20261113 | WR | 498 | 41.97 | [36.16, 48.53] |
| 20261113 | SS | 445 | 20.67 | [16.93, 24.55] |
| 20261113 | TI | 298 | 90.50 | [85.52, 94.95] |

## Across-seed range

| Metric | Minimum (%) | Maximum (%) | Range (pp) |
|---|---:|---:|---:|
| WR | 37.75 | 41.97 | 4.22 |
| SS | 19.10 | 22.47 | 3.37 |
| TI | 86.40 | 90.50 | 4.10 |
