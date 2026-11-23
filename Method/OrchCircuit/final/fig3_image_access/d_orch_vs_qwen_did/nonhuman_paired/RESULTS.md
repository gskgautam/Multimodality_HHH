# Figure 4b - Paired image-access sensitivity contrast (no human ratings)

For each metric and judge, compare the image-present minus image-omitted score shift for OrchCircuit's selected answer with the same shift for Qwen2.5-VL 7B on the exact same prompt/image records. Qwen is the common reference because the saved audit has paired coverage across all three metrics and both judges. The estimand is the difference-in-differences: OrchCircuit shift minus Qwen shift. For SS, the score is unsafe rate, so a positive shift means more judged unsafe.

This tests relative automated-judge sensitivity to image access. It does not test answer correctness, human agreement, or whether OrchCircuit uses visual evidence better.

| Metric | Judge | Common prompts | Query/image groups | OrchCircuit shift (pp) | Qwen shift (pp) | Difference in shifts (pp; 95% group-bootstrap CI) |
|---|---|---:|---:|---:|---:|---:|
| WR | anthropic | 77 | 65 | +18.40 | -23.70 | +42.10 [+19.17, +64.40] |
| WR | google | 73 | 62 | +43.75 | -25.78 | +69.53 [+41.23, +94.56] |
| SS | anthropic | 74 | 73 | +2.70 | +2.70 | +0.00 [-0.00, +0.00] |
| SS | google | 71 | 70 | +0.00 | +1.41 | -1.41 [-4.29, +0.00] |
| TI | anthropic | 77 | 71 | +58.21 | +39.55 | +18.65 [+4.53, +32.48] |
| TI | google | 74 | 68 | +77.48 | +62.91 | +14.57 [+0.28, +28.83] |

WR is candidate win rate among decisive non-tied judgments, SS is unsafe rate, and TI is the product of truthful and informative marginal rates, matching the audit definitions. Intervals use 10,000 bootstrap resamples of connected query/image groups.

This is a post-hoc analysis of the existing 80-item-per-dimension audit and development-exposed Ch3EF records; it is exploratory, not confirmatory. Qwen is a common reference, not the strongest baseline on every metric. Judge effects should not be averaged as interchangeable measurements. This can supplement, but cannot replace, human validation.

Reproduce by running python3 Method/OrchCircuit/final/figure4/b/nonhuman_paired/analyze_paired_shift.py. It reads existing judge calls and grouped-fold metadata; it makes no model or API calls.
