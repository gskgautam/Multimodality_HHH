# Figure 4a — component ablation results

Exploratory grouped out-of-fold ablations on previously used Ch3Ef records. Outcomes are automated-judge-relative, not human-verified correctness. Each variant changes one implemented component and shares the full model's grouped folds and training seed.

Delta is ablation minus full in percentage points; higher WR/TI and lower SS are favorable. Intervals are paired group-cluster bootstrap intervals, Bonferroni-adjusted across 9 contrasts.

| Variant | Metric | n | Full (%) | Ablation (%) | Delta (pp) | Adjusted 95% CI (pp) |
|---|---|---:|---:|---:|---:|---:|
| no_set_attention | WR | 498 | 38.55 | 41.77 | +3.21 | [-1.78, +8.74] |
| no_set_attention | SS | 445 | 21.57 | 20.67 | -0.90 | [-5.04, +3.14] |
| no_set_attention | TI | 298 | 89.23 | 81.18 | -8.05 | [-17.12, +0.07] |
| no_candidate_identity | WR | 498 | 38.55 | 33.53 | -5.02 | [-12.31, +0.82] |
| no_candidate_identity | SS | 445 | 21.57 | 20.22 | -1.35 | [-5.94, +3.16] |
| no_candidate_identity | TI | 298 | 89.23 | 81.18 | -8.05 | [-17.44, +1.27] |
| mean_token_pool | WR | 498 | 38.55 | 37.75 | -0.80 | [-6.96, +6.35] |
| mean_token_pool | SS | 445 | 21.57 | 24.04 | +2.47 | [-2.25, +7.33] |
| mean_token_pool | TI | 298 | 89.23 | 88.60 | -0.63 | [-9.43, +7.86] |
