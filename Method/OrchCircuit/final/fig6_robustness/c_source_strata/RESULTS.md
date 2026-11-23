# Figure 4d — Cross-question source strata

Exploratory paired subgroup analysis by the original Ch3EF source field. Labels were normalized for case/underscore/hyphen and four spelling duplicates were merged by label text before outcomes were computed (reffered vision, GUI, handwriting, Racial/RacialDiscrimination). TI is the joint truthful-AND-informative rate. The fixed comparator comes from the full-set OOF common-valid comparison. Positive deltas favor OrchCircuit; SS is oriented as baseline unsafe rate minus OrchCircuit unsafe rate.

| Metric | Source | n | groups | Orch mean % | Baseline | Baseline % | Favorable delta pp (95% cluster bootstrap) |
|---|---|---:|---:|---:|---|---:|---:|
| WR | animation | 24 | 22 | 54.17 | gemini-4-argon | 66.67 | -12.50 [-22.22, -4.17] |
| WR | art | 26 | 23 | 52.56 | gemini-4-argon | 61.54 | -8.97 [-21.43, +4.35] |
| WR | autonomous | 31 | 30 | 58.06 | gemini-4-argon | 61.29 | -3.23 [-14.44, +7.78] |
| WR | chart comprehension | 8 | 8 | 41.67 | gemini-4-argon | 37.50 | +4.17 [+0.00, +12.50] |
| WR | comics | 7 | 6 | 28.57 | gemini-4-argon | 28.57 | +0.00 [-14.29, +14.29] |
| WR | document comprehension | 65 | 28 | 35.90 | gemini-4-argon | 44.62 | -8.72 [-16.67, -2.65] |
| WR | gui comprehension | 9 | 9 | 33.33 | gemini-4-argon | 22.22 | +11.11 [+0.00, +33.33] |
| WR | hand drawing | 5 | 5 | 53.33 | gemini-4-argon | 60.00 | -6.67 [-20.00, +0.00] |
| WR | handwriting comprehension | 5 | 4 | 13.33 | gemini-4-argon | 20.00 | -6.67 [not estimable; <5 groups] |
| WR | indoor navigation | 8 | 5 | 45.83 | gemini-4-argon | 50.00 | -4.17 [-20.00, +0.00] |
| WR | instruction follow | 34 | 33 | 9.80 | gemini-4-argon | 8.82 | +0.98 [-2.02, +4.76] |
| WR | language performance | 17 | 16 | 17.65 | gemini-4-argon | 17.65 | +0.00 [+0.00, +0.00] |
| WR | medical | 27 | 25 | 49.38 | gemini-4-argon | 44.44 | +4.94 [-4.94, +14.97] |
| WR | object multi view understanding | 58 | 1 | 22.99 | gemini-4-argon | 15.52 | +7.47 [not estimable; <5 groups] |
| WR | ocr | 23 | 23 | 37.68 | gemini-4-argon | 34.78 | +2.90 [-5.80, +11.59] |
| WR | outdoor navigation | 5 | 5 | 66.67 | gemini-4-argon | 40.00 | +26.67 [+0.00, +53.33] |
| WR | referred vision comprehension | 22 | 19 | 28.79 | gemini-4-argon | 45.45 | -16.67 [-30.77, -3.51] |
| WR | remote | 28 | 17 | 50.00 | gemini-4-argon | 60.71 | -10.71 [-24.70, +0.00] |
| WR | robotic | 5 | 5 | 73.33 | gemini-4-argon | 80.00 | -6.67 [-26.67, +20.00] |
| WR | robotic planning | 20 | 18 | 38.33 | gemini-4-argon | 45.00 | -6.67 [-16.71, +1.85] |
| WR | scene multi view understanding | 32 | 1 | 54.17 | gemini-4-argon | 59.38 | -5.21 [not estimable; <5 groups] |
| WR | vision only icl | 23 | 1 | 62.32 | gemini-4-argon | 69.57 | -7.25 [not estimable; <5 groups] |
| WR | vision+text icl | 16 | 5 | 16.67 | gemini-4-argon | 12.50 | +4.17 [+0.00, +14.29] |
| SS | age | 11 | 11 | 6.06 | gemma3-12b | 18.18 | +12.12 [-12.12, +39.39] |
| SS | disability | 13 | 13 | 0.00 | gemma3-12b | 0.00 | +0.00 [+0.00, +0.00] |
| SS | drug | 35 | 34 | 69.52 | gemma3-12b | 77.14 | +7.62 [-0.90, +18.10] |
| SS | gender | 18 | 18 | 14.81 | gemma3-12b | 22.22 | +7.41 [-7.45, +25.93] |
| SS | harass | 19 | 19 | 19.30 | gemma3-12b | 26.32 | +7.02 [-15.79, +29.82] |
| SS | inferring personal information | 54 | 54 | 24.69 | gemma3-12b | 27.78 | +3.09 [-9.26, +16.05] |
| SS | insult | 14 | 14 | 19.05 | gemma3-12b | 21.43 | +2.38 [-4.76, +14.29] |
| SS | leaking private information | 20 | 20 | 16.67 | gemma3-12b | 40.00 | +23.33 [+6.67, +43.33] |
| SS | location | 16 | 14 | 0.00 | gemma3-12b | 0.00 | +0.00 [+0.00, +0.00] |
| SS | other crimes | 56 | 55 | 27.38 | gemma3-12b | 33.93 | +6.55 [-4.24, +16.97] |
| SS | physical harm | 36 | 36 | 9.26 | gemma3-12b | 11.11 | +1.85 [-6.48, +11.11] |
| SS | psychological harm | 46 | 46 | 14.49 | gemma3-12b | 28.26 | +13.77 [+2.90, +25.38] |
| SS | racial discrimination | 16 | 15 | 4.17 | gemma3-12b | 6.25 | +2.08 [-8.89, +17.78] |
| SS | religious | 10 | 9 | 3.33 | gemma3-12b | 10.00 | +6.67 [+0.00, +22.22] |
| SS | theft | 45 | 45 | 25.19 | gemma3-12b | 24.44 | -0.74 [-9.63, +8.89] |
| SS | threaten | 18 | 17 | 12.96 | gemma3-12b | 27.78 | +14.81 [-5.88, +38.89] |
| SS | traffic crime | 18 | 18 | 3.70 | gemma3-12b | 0.00 | -3.70 [-9.26, +0.00] |
| TI | art | 5 | 5 | 100.00 | gemini-4-argon | 100.00 | +0.00 [+0.00, +0.00] |
| TI | attribute hallucination | 7 | 7 | 90.48 | gemini-4-argon | 100.00 | -9.52 [-28.57, +0.00] |
| TI | capability hallucination | 31 | 31 | 95.70 | gemini-4-argon | 96.77 | -1.08 [-3.23, +0.00] |
| TI | object facts | 19 | 13 | 98.25 | gemini-4-argon | 100.00 | -1.75 [-6.67, +0.00] |
| TI | object hallucination | 28 | 20 | 94.05 | gemini-4-argon | 96.43 | -2.38 [-6.67, +0.00] |
| TI | pope | 19 | 10 | 89.47 | gemini-4-argon | 100.00 | -10.53 [-17.65, +0.00] |
| TI | relationship hallucination | 16 | 16 | 95.83 | gemini-4-argon | 100.00 | -4.17 [-12.50, +0.00] |
| TI | robustness | 84 | 83 | 91.27 | gemini-4-argon | 88.10 | +3.17 [-2.41, +9.52] |
| TI | sycophancy | 89 | 87 | 92.51 | gemini-4-argon | 93.26 | -0.75 [-4.92, +3.41] |

Intervals resample connected query/image groups and average paired deltas over three training seeds. They are exploratory and unadjusted across source strata; small groups are included in the CSV; plotted strata require n≥12 and at least five connected query/image groups. CIs are omitted for strata with fewer than five groups. The data were used in method development; this is not a fresh confirmatory test.
