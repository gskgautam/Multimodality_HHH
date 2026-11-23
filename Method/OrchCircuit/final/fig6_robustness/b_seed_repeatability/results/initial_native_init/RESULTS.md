# Figure 4c — training-seed repeatability

Exploratory full-selector results on the same previously used Ch3Ef common-valid records. The five outer folds and grouped inner-validation assignments are fixed; only training RNG seed changes. Per-seed intervals describe group-bootstrap sampling uncertainty. Seed range is descriptive, not a confidence interval over independent datasets.

| Training seed | Metric | n | Score (%) | Group-bootstrap 95% CI (%) |
|---:|---|---:|---:|---:|
| 20261111 | WR | 498 | 38.55 | [33.40, 44.06] |
| 20261111 | SS | 445 | 21.57 | [17.71, 25.52] |
| 20261111 | TI | 298 | 89.23 | [84.63, 93.64] |
| 20261112 | WR | 498 | 40.56 | [35.10, 46.21] |
| 20261112 | SS | 445 | 20.45 | [16.70, 24.33] |
| 20261112 | TI | 298 | 86.09 | [80.52, 91.29] |
| 20261113 | WR | 498 | 38.35 | [32.56, 44.66] |
| 20261113 | SS | 445 | 22.02 | [18.06, 26.01] |
| 20261113 | TI | 298 | 89.55 | [84.56, 94.01] |

## Across-seed range

| Metric | Minimum (%) | Maximum (%) | Range (pp) |
|---|---:|---:|---:|
| WR | 38.35 | 40.56 | 2.21 |
| SS | 20.45 | 22.02 | 1.57 |
| TI | 86.09 | 89.55 | 3.46 |
