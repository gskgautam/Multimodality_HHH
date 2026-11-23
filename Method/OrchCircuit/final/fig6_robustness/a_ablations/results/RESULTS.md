# Figure 4a — matched-initialization component ablations

Exploratory grouped out-of-fold ablations on previously used Ch3Ef records. Labels are automated-judge outputs, not human-verified correctness. All four model variants were rerun under three training seeds with identical outer/inner folds and deterministic module-wise matched initialization for shared components. Initialization fingerprints were checked across every seed and fold.

Delta is ablation minus full in percentage points; higher WR/TI and lower SS are favorable. The intervals are paired connected-group bootstrap intervals for the three-seed mean delta, Bonferroni-adjusted across the nine planned comparisons. The interval conditions on these three training seeds and does not estimate uncertainty on a new dataset.

| Variant | Metric | n | Full mean (%) | Ablation mean (%) | Mean delta (pp) | Per-seed deltas (pp) | Adjusted 95% CI (pp) |
|---|---|---:|---:|---:|---|---:|
| no_set_attention | WR | 498 | 39.29 | 39.09 | -0.20 | 20261111:+0.00; 20261112:+0.20; 20261113:-0.80 | [-2.18, +1.50] |
| no_candidate_identity | WR | 498 | 39.29 | 33.40 | -5.89 | 20261111:-6.43; 20261112:-6.63; 20261113:-4.62 | [-11.82, -1.46] |
| mean_token_pool | WR | 498 | 39.29 | 37.62 | -1.67 | 20261111:-1.61; 20261112:-1.41; 20261113:-2.01 | [-4.34, +0.63] |
| no_set_attention | SS | 445 | 20.75 | 21.65 | +0.90 | 20261111:+1.35; 20261112:+1.12; 20261113:+0.22 | [-0.83, +2.67] |
| no_candidate_identity | SS | 445 | 20.75 | 20.22 | -0.52 | 20261111:+0.67; 20261112:-2.25; 20261113:+0.00 | [-3.34, +2.28] |
| mean_token_pool | SS | 445 | 20.75 | 25.39 | +4.64 | 20261111:+5.17; 20261112:+6.29; 20261113:+2.47 | [+1.71, +7.65] |
| no_set_attention | TI | 298 | 87.98 | 85.61 | -2.36 | 20261111:-6.45; 20261112:+0.31; 20261113:-0.96 | [-5.84, +0.92] |
| no_candidate_identity | TI | 298 | 87.98 | 83.62 | -4.36 | 20261111:-4.94; 20261112:-3.40; 20261113:-4.74 | [-10.43, +1.65] |
| mean_token_pool | TI | 298 | 87.98 | 81.82 | -6.16 | 20261111:-6.45; 20261112:-7.92; 20261113:-4.10 | [-11.94, -0.63] |

The prior single-seed results used native whole-model initialization, which changed initial values of shared downstream modules when a component was removed. Those initial results are retained in initial_native_init_results.md for audit but are superseded by this matched-initialization analysis.
