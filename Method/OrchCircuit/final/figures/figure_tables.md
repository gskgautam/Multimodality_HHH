# Figure tables

Numeric tables read by make_figures.py (Figs. 3a, 3b, 5a–c, 6a and 6d), copied from the analysis outputs listed in README.md.

### Fig. 3a — WR (old Table 2a)
| Model | Judge | Pairs | Same choice % | WR decisive image % | WR decisive omitted % | Δ WR decisive pp | Ties image | Ties omitted | Wins image / omitted | WR all image % | WR all omitted % | Δ WR all pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5-vl-7b | anthropic | 80 | 65.00 | 20.45 | 42.31 | -21.85 | 36 | 54 | 9 / 11 | 11.25 | 13.75 | -2.50 |
| qwen2.5-vl-7b | google | 76 | 60.53 | 20.93 | 41.38 | -20.45 | 33 | 47 | 9 / 12 | 11.84 | 15.79 | -3.95 |
| lfm3-vl-2b | anthropic | 80 | 63.75 | 69.57 | 60.00 | +9.57 | 34 | 60 | 32 / 12 | 40.00 | 15.00 | +25.00 |
| lfm3-vl-2b | openai | 80 | 72.50 | 62.50 | 80.00 | -17.50 | 40 | 55 | 25 / 20 | 31.25 | 25.00 | +6.25 |
| gemma-4-12b | anthropic | 79 | 56.96 | 46.94 | 50.00 | -3.06 | 30 | 53 | 23 / 13 | 29.11 | 16.46 | +12.66 |
| gemma-4-12b | openai | 80 | 63.75 | 56.52 | 42.31 | +14.21 | 34 | 54 | 26 / 11 | 32.50 | 13.75 | +18.75 |
| qwen3.8-27b | anthropic | 79 | 53.16 | 56.60 | 52.38 | +4.22 | 26 | 58 | 30 / 11 | 37.97 | 13.92 | +24.05 |
| qwen3.8-27b | google | 78 | 62.82 | 47.73 | 21.21 | +26.52 | 34 | 45 | 21 / 7 | 26.92 | 8.97 | +17.95 |
| minicpm-v-4.7-35b-a3b | anthropic | 80 | 58.75 | 57.69 | 48.15 | +9.54 | 28 | 53 | 30 / 13 | 37.50 | 16.25 | +21.25 |
| minicpm-v-4.7-35b-a3b | google | 76 | 71.05 | 53.49 | 36.11 | +17.38 | 33 | 40 | 23 / 13 | 30.26 | 17.11 | +13.16 |
| gpt-6.1-terra | anthropic | 80 | 70.00 | 69.23 | 52.94 | +16.29 | 41 | 63 | 27 / 9 | 33.75 | 11.25 | +22.50 |
| gpt-6.1-terra | google | 77 | 70.13 | 72.73 | 51.85 | +20.88 | 44 | 50 | 24 / 14 | 31.17 | 18.18 | +12.99 |
| gemini-4-argon | anthropic | 80 | 60.00 | 80.36 | 58.62 | +21.74 | 24 | 51 | 45 / 17 | 56.25 | 21.25 | +35.00 |
| gemini-4-argon | openai | 80 | 67.50 | 84.00 | 89.29 | -5.29 | 30 | 52 | 42 / 25 | 52.50 | 31.25 | +21.25 |
| claude-sonnet-5-5 | google | 78 | 51.28 | 86.00 | 36.84 | +49.16 | 28 | 40 | 43 / 14 | 55.13 | 17.95 | +37.18 |
| claude-sonnet-5-5 | openai | 80 | 72.50 | 75.00 | 85.29 | -10.29 | 28 | 46 | 39 / 29 | 48.75 | 36.25 | +12.50 |
| OrchCircuit | anthropic | 77 | 54.55 | 89.83 | 71.43 | +18.40 | 18 | 49 | 53 / 20 | 68.83 | 25.97 | +42.86 |
| OrchCircuit | google | 77 | 58.44 | 88.37 | 41.18 | +47.20 | 34 | 43 | 38 / 14 | 49.35 | 18.18 | +31.17 |

### Fig. 3b — SS (old Table 2b)
| Model | Judge | Pairs | Same label % | Unsafe image % | Unsafe omitted % | Δ unsafe pp | Image unsafe → omitted safe | Image safe → omitted unsafe |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-vl-7b | anthropic | 80 | 97.50 | 8.75 | 6.25 | 2.50 | 2 | 0 |
| qwen2.5-vl-7b | google | 76 | 98.68 | 5.26 | 3.95 | 1.32 | 1 | 0 |
| lfm3-vl-2b | anthropic | 76 | 93.42 | 48.68 | 52.63 | -3.95 | 1 | 4 |
| lfm3-vl-2b | openai | 80 | 92.50 | 66.25 | 58.75 | 7.50 | 6 | 0 |
| gemma-4-12b | anthropic | 79 | 98.73 | 5.06 | 3.80 | 1.27 | 1 | 0 |
| gemma-4-12b | openai | 80 | 95.00 | 8.75 | 13.75 | -5.00 | 0 | 4 |
| qwen3.8-27b | anthropic | 79 | 98.73 | 3.80 | 2.53 | 1.27 | 1 | 0 |
| qwen3.8-27b | google | 76 | 98.68 | 1.32 | 0.00 | 1.32 | 1 | 0 |
| minicpm-v-4.7-35b-a3b | anthropic | 78 | 100.00 | 3.85 | 3.85 | 0.00 | 0 | 0 |
| minicpm-v-4.7-35b-a3b | google | 78 | 97.44 | 2.56 | 0.00 | 2.56 | 2 | 0 |
| gpt-6.1-terra | anthropic | 80 | 97.50 | 22.50 | 25.00 | -2.50 | 0 | 2 |
| gpt-6.1-terra | google | 79 | 89.87 | 13.92 | 6.33 | 7.59 | 7 | 1 |
| gemini-4-argon | anthropic | 80 | 100.00 | 2.50 | 2.50 | 0.00 | 0 | 0 |
| gemini-4-argon | openai | 80 | 92.50 | 7.50 | 10.00 | -2.50 | 2 | 4 |
| claude-sonnet-5-5 | google | 77 | 100.00 | 0.00 | 0.00 | 0.00 | 0 | 0 |
| claude-sonnet-5-5 | openai | 80 | 98.75 | 7.50 | 6.25 | 1.25 | 1 | 0 |
| OrchCircuit | anthropic | 74 | 97.30 | 2.70 | 0.00 | 2.70 | 2 | 0 |
| OrchCircuit | google | 75 | 100.00 | 0.00 | 0.00 | 0.00 | 0 | 0 |

### Fig. 5a–c — Direct human-rated policy/control comparisons (old Figure 3e)
| Net WR, pp (95% CI) | Superseded round | Reported round |
|---|---|---|
| OrchCircuit-WR vs Gemini 4 Argon | **+9.5 [+2.0, +17.0]** | +4.0 [−2.5, +10.5] |
| OrchCircuit-SS vs Gemini 4 Argon | **+11.0 [+1.5, +20.0]** | +5.0 [−1.0, +11.0] |
| OrchCircuit-TI vs Gemma 3 12B | **−16.0 [−25.5, −6.0]** | 0.0 [−7.0, +6.5] |
| OrchCircuit-TI vs Gemini 4 Argon | **−7.0 [−13.0, −1.0]** | +4.5 [−1.5, +10.5] |
| Human "truthful" rate, any system | 96.8–100% (ceiling) | 45.1–94.3% |
| Criterion | Superseded round κ | Reported round κ | Decision |
|---|---:|---:|---|
| Truthfulness | 0.001 (chance; two of three raters almost never used "Partly supported") | 0.639 | **Reported round only.** The superseded ratings are not usable. |
| Informativeness | 0.220 (≈ 97% "Adequate" ceiling) | 0.690 | **Reported round only.** |
| WR | 0.822 | 0.546 | **Both rounds reported.** Quality gives no reason to discard either. |
| Safety | identical across raters | identical across raters | Uninformative in both (floor) |
| Check | Reported round | Superseded round |
|---|---|---|
| Raters' "Supported" rate when the answer matches the key | 78.0% | 100% |
| Raters' "Supported" rate when it does not | 41.4% | 100% |
| Difference, pp (95% CI) | **+36.6 [+24.1, +49.2]** | 0.0 |
| Cohen's κ, truthfulness majority vs key | **0.22 [0.14, 0.31]** | 0.00 |
| WR pairs with exactly one correct answer: majority prefers the correct one | 47.4% [28.9, 65.0] (n = 38 decisive) | 63.9% [52.9, 74.8] (n = 166 decisive) |
| Answers matching the key (all systems) | 80.5–92.5% per system | same answers |
| OrchCircuit policy | Control | Policy wins / control wins / ties / no majority | WR net pp (95% CI) | SS Δ pp | TI Δ joint-positive pp (95% CI) |
|---|---|---:|---:|---:|---:|
| WR | Gemini 4 Argon | 27 / 19 / 153 / 1 | +4.0 [−2.5,+10.5] | 0.0 | −9.4 [−18.4,−0.9] |
| WR | Gemma 3 12B | 15 / 19 / 166 / 0 | −2.0 [−8.0,+3.5] | 0.0 | +22.4 [+10.5,+34.4] |
| WR | Claude Opus 5.5 | 23 / 16 / 159 / 2 | +3.5 [−2.5,+9.5] | 0.0 | −3.8 [−14.6,+7.1] |
| SS | Gemini 4 Argon | 24 / 14 / 162 / 0 | +5.0 [−1.0,+11.0] | 0.0 | −23.1 [−34.1,−11.7] |
| SS | Gemma 3 12B | 17 / 17 / 165 / 1 | 0.0 [−5.5,+5.5] | 0.0 | +10.6 [+0.8,+20.3] |
| SS | Claude Opus 5.5 | 21 / 17 / 162 / 0 | +2.0 [−4.0,+8.0] | 0.0 | −22.8 [−33.9,−11.5] |
| TI | Gemini 4 Argon | 24 / 15 / 159 / 2 | +4.5 [−1.5,+10.5] | 0.0 | +2.3 [−3.7,+8.3] |
| TI | Gemma 3 12B | 24 / 24 / 152 / 0 | 0.0 [−7.0,+6.5] | 0.0 | +32.7 [+20.3,+45.2] |
| TI | Claude Opus 5.5 | 23 / 23 / 154 / 0 | 0.0 [−6.5,+7.0] | 0.0 | −3.0 [−14.6,+9.0] |

### Fig. 6a — Matched-initialisation ablations (old Figure 4a)
| Ablation | Metric | n | Full mean | Ablation mean | Δ pp (adjusted 95% CI) |
|---|---|---:|---:|---:|---:|
| no set attention | WR | 498 | 39.29% | 39.09% | −0.20 [−2.18,+1.50] |
| no candidate identity | WR | 498 | 39.29% | 33.40% | −5.89 [−11.82,−1.46] |
| mean token pooling | WR | 498 | 39.29% | 37.62% | −1.67 [−4.34,+0.63] |
| no set attention | SS unsafe | 445 | 20.75% | 21.65% | +0.90 [−0.83,+2.67] |
| no candidate identity | SS unsafe | 445 | 20.75% | 20.22% | −0.52 [−3.34,+2.28] |
| mean token pooling | SS unsafe | 445 | 20.75% | 25.39% | +4.64 [+1.71,+7.65] |
| no set attention | TI product | 298 | 87.98% | 85.61% | −2.36 [−5.84,+0.92] |
| no candidate identity | TI product | 298 | 87.98% | 83.62% | −4.36 [−10.43,+1.65] |
| mean token pooling | TI product | 298 | 87.98% | 81.82% | −6.16 [−11.94,−0.63] |
| Ablation | Metric | n | Full mean | Ablation mean | Δ pp (adjusted 95% CI) |
|---|---|---:|---:|---:|---:|
| no set attention | TI joint | 298 | 93.51% | 91.95% | −1.57 [−3.44,+0.23] |
| no candidate identity | TI joint | 298 | 93.51% | 90.49% | −3.02 [−6.53,+0.38] |
| mean token pooling | TI joint | 298 | 93.51% | 90.16% | −3.36 [−6.80,−0.31] |

### Fig. 6d — Same-prompt three-objective results, 600-prompt development cohort (old Figure 4e)
| System | WR wins / 600 | SS unsafe / 600 | TI truthful | TI informative | TI product | **TI joint (primary)** | Pareto (joint) |
|---|---:|---:|---:|---:|---:|---:|:---:|
| OrchCircuit | 32.33% | 36.83% | 90.33% | 94.17% | 85.06% | 90.17% | Yes |
| Claude Haiku 5.5 | 29.00% | 44.00% | 86.00% | 90.83% | 78.12% | 85.33% | No |
| Claude Opus 5.5 | 33.50% | 42.67% | 88.83% | 93.17% | 82.76% | 88.67% | Yes |
| Claude Sonnet 5.5 | 33.67% | 43.50% | 89.83% | 93.83% | 84.29% | 89.50% | Yes |
| Gemini 3.6 Flash | 28.83% | 44.33% | 88.67% | 93.00% | 82.46% | 88.67% | No |
| Gemini 3.8 Flash | 30.17% | 41.83% | 88.67% | 92.17% | 81.72% | 88.33% | No |
| Gemini 4 Argon | 32.67% | 41.83% | 90.50% | 93.33% | 84.47% | 89.67% | Yes |
| Gemma 4 12B | 24.33% | 41.33% | 80.17% | 86.50% | 69.34% | 79.33% | No |
| Gemma 3 12B | 23.83% | 41.83% | 77.33% | 84.83% | 65.60% | 76.67% | No |
| Gemma 3 27B | 24.33% | 44.67% | 80.00% | 86.50% | 69.20% | 79.67% | No |
| Gemma 4 E2B | 14.67% | 47.67% | 63.50% | 76.67% | 48.68% | 61.50% | No |
| GPT-6 Luna | 28.17% | 44.33% | 84.17% | 90.00% | 75.75% | 83.17% | No |
| GPT-6.1 Sol | 27.83% | 48.17% | 86.67% | 91.17% | 79.01% | 86.17% | No |
| GPT-6.1 Terra | 24.33% | 52.17% | 84.83% | 89.83% | 76.21% | 83.33% | No |
| LFM3-VL 2B | 18.33% | 49.00% | 69.50% | 81.17% | 56.41% | 68.50% | No |
| MiniCPM-V 4.5 8B | 17.67% | 45.83% | 70.50% | 79.50% | 56.05% | 69.33% | No |
| MiniCPM-V 4.7 35B-A3B | 28.00% | 39.50% | 85.33% | 91.83% | 78.36% | 84.83% | No |
| Pixtral 12B | 18.33% | 46.17% | 74.83% | 82.67% | 61.86% | 73.17% | No |
| Qwen2.5-VL 7B (WR anchor) | 0.00%† | 41.83% | 76.17% | 85.17% | 64.87% | 75.33% | No |
| Qwen3-VL 8B | 25.00% | 40.17% | 82.83% | 89.83% | 74.41% | 82.83% | No |
| Qwen3.5 2B | 22.00% | 45.50% | 75.83% | 83.17% | 63.07% | 74.83% | No |
| Qwen3.5 35B-A3B | 26.00% | 43.67% | 83.33% | 89.67% | 74.72% | 83.00% | No |
| Qwen3.6 27B | 29.33% | 39.00% | 85.50% | 90.67% | 77.52% | 85.17% | No |
| Qwen3.6 35B-A3B | 27.17% | 39.50% | 85.50% | 90.50% | 77.38% | 85.00% | No |
| Qwen3.8 27B | 26.67% | 41.33% | 85.50% | 90.50% | 77.38% | 85.17% | No |
| Metric | Fixed comparator (prespecified from Fig. 2a) | OrchCircuit favourable Δ pp (95% paired group-bootstrap CI) |
|---|---|---:|
| WR | Gemini 4 Argon | −0.33 [−1.97,+1.23] |
| SS unsafe | Gemma 3 12B | +5.00 [−0.63,+10.23] |
| TI product | Gemini 4 Argon | +0.60 [−2.84,+4.12] |
| **TI joint (primary)** | Gemini 4 Argon | +0.50 [−1.75,+2.77] |
