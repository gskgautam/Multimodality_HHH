# Figure 4e — Same-prompt HHH trade-off

Exploratory re-score of the existing 600-prompt development cohort. OrchCircuit's saved grouped-OOF selected answer for each prompt was scored on all three available joint automated outcomes. This makes the metric vector use the same prompt cohort, unlike Table 1's criterion-specific denominators. The cohort was used during prior method development; this analysis is not confirmatory.

| System | WR wins / 600 (%) | SS unsafe / 600 (%) | TI truthful (%) | TI informative (%) | TI product (%) | Pareto point |
|---|---:|---:|---:|---:|---:|:---:|
| OrchCircuit | 32.33 | 36.83 | 90.33 | 94.17 | 85.06 | Yes |
| claude-haiku-5-5 | 29.00 | 44.00 | 86.00 | 90.83 | 78.12 | No |
| claude-opus-5-5 | 33.50 | 42.67 | 88.83 | 93.17 | 82.76 | Yes |
| claude-sonnet-5-5 | 33.67 | 43.50 | 89.83 | 93.83 | 84.29 | Yes |
| gemini-3.6-flash | 28.83 | 44.33 | 88.67 | 93.00 | 82.46 | No |
| gemini-3.8-flash | 30.17 | 41.83 | 88.67 | 92.17 | 81.72 | No |
| gemini-4-argon | 32.67 | 41.83 | 90.50 | 93.33 | 84.47 | Yes |
| gemma-4-12b | 24.33 | 41.33 | 80.17 | 86.50 | 69.34 | No |
| gemma3-12b | 23.83 | 41.83 | 77.33 | 84.83 | 65.60 | No |
| gemma3-27b | 24.33 | 44.67 | 80.00 | 86.50 | 69.20 | No |
| gemma4-e2b | 14.67 | 47.67 | 63.50 | 76.67 | 48.68 | No |
| gpt-6-luna | 28.17 | 44.33 | 84.17 | 90.00 | 75.75 | No |
| gpt-6.1-sol | 27.83 | 48.17 | 86.67 | 91.17 | 79.01 | No |
| gpt-6.1-terra | 24.33 | 52.17 | 84.83 | 89.83 | 76.21 | No |
| lfm3-vl-2b | 18.33 | 49.00 | 69.50 | 81.17 | 56.41 | No |
| minicpm-v-4.5-8b | 17.67 | 45.83 | 70.50 | 79.50 | 56.05 | No |
| minicpm-v-4.7-35b-a3b | 28.00 | 39.50 | 85.33 | 91.83 | 78.36 | No |
| pixtral-12b | 18.33 | 46.17 | 74.83 | 82.67 | 61.86 | No |
| qwen2.5-vl-7b | 0.00 | 41.83 | 76.17 | 85.17 | 64.87 | No |
| qwen3-vl-8b | 25.00 | 40.17 | 82.83 | 89.83 | 74.41 | No |
| qwen3.5-2b | 22.00 | 45.50 | 75.83 | 83.17 | 63.07 | No |
| qwen3.5-35b-a3b | 26.00 | 43.67 | 83.33 | 89.67 | 74.72 | No |
| qwen3.6-27b | 29.33 | 39.00 | 85.50 | 90.67 | 77.52 | No |
| qwen3.6-35b-a3b | 27.17 | 39.50 | 85.50 | 90.50 | 77.38 | No |
| qwen3.8-27b | 26.67 | 41.33 | 85.50 | 90.50 | 77.38 | No |

## Paired differences from prespecified fixed baselines

| Metric | Comparator | OrchCircuit favorable delta pp (95% group-bootstrap CI) |
|---|---|---:|
| WR | gemini-4-argon | -0.33 [-1.97, +1.23] |
| SS | gemma3-12b | +5.00 [-0.63, +10.23] |
| TI | gemini-4-argon | +0.60 [-2.84, +4.12] |

The Pareto front is defined without a composite: maximize WR wins and TI product, minimize SS unsafe flags. Confidence intervals are paired over the saved connected query/image groups. WR remains GPT-4o preference against the Qwen anchor; SS is Beaver-Dam text-only; TI is the product of GPT-4o truthful and informative marginals. These are evaluator outputs, not correctness labels.
