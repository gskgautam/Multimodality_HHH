# Human-rated Figure 3e results

Frozen, external-domain HumaniBench task3 sample: 200 prompts (40 per Age, Ethnicity, Gender, Occupation, Sport). Three independent raters scored three OrchCircuit policy selections and three pre-specified controls per prompt. This is a six-system human-rated comparison, not a human evaluation of all 24 candidate generators. No answer key or source reasoning was used in this analysis.

WR uses majority-normalized direct pairwise human judgments. Net preference is (selector wins − control wins)/200 prompts; ties, uncertain decisions, and 3-way no-majority cases contribute zero and remain in the denominator. SS is the paired change in majority unsafe flag rate, selector minus control (negative favors the selector). TI is the paired change in joint strict-positive rate (truthfulness Supported AND informativeness Adequate), selector minus control (positive favors the selector). Pairwise 95% CIs use a 10,000-replicate prompt bootstrap, seed 20261113. Intervals are pointwise and not adjusted across the nine policy-control matchups; treat them as descriptive rather than standalone confirmatory tests.

| OrchCircuit policy | Fixed control | WR policy wins / 200 | Control wins | Ties | Uncertain | No majority | Net WR advantage, pp (95% CI) | SS Δ unsafe, pp (95% CI) | TI Δ joint-positive, pp (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OrchCircuit-WR | Gemini 4 Argon | 41/200 (20.5%) | 22 | 136 | 0 | 1 | +9.5 (+2.0–+17.0) | +0.0 (0.0–0.0) | +0.0 (0.0–0.0) |
| OrchCircuit-WR | Gemma 3 12B | 57/200 (28.5%) | 47 | 94 | 0 | 2 | +5.0 (-5.0–+15.0) | -0.5 (-1.5–0.0) | +0.0 (0.0–0.0) |
| OrchCircuit-WR | Claude Opus 5.5 | 28/200 (14.0%) | 21 | 149 | 0 | 2 | +3.5 (-3.5–+10.5) | +0.0 (0.0–0.0) | +0.6 (0.0–1.7) |
| OrchCircuit-SS | Gemini 4 Argon | 54/200 (27.0%) | 32 | 108 | 0 | 6 | +11.0 (+1.5–+20.0) | +0.0 (0.0–0.0) | +0.0 (0.0–0.0) |
| OrchCircuit-SS | Gemma 3 12B | 36/200 (18.0%) | 39 | 120 | 0 | 5 | -1.5 (-10.0–+7.0) | -0.5 (-1.5–0.0) | +0.0 (0.0–0.0) |
| OrchCircuit-SS | Claude Opus 5.5 | 42/200 (21.0%) | 47 | 103 | 0 | 8 | -2.5 (-12.0–+6.5) | +0.0 (0.0–0.0) | +0.5 (0.0–1.6) |
| OrchCircuit-TI | Gemini 4 Argon | 12/200 (6.0%) | 26 | 162 | 0 | 0 | -7.0 (-13.0–-1.0) | +0.0 (0.0–0.0) | -3.2 (-5.9–-1.0) |
| OrchCircuit-TI | Gemma 3 12B | 37/200 (18.5%) | 69 | 85 | 0 | 9 | -16.0 (-25.5–-6.0) | -0.5 (-1.5–0.0) | -3.2 (-5.9–-1.1) |
| OrchCircuit-TI | Claude Opus 5.5 | 33/200 (16.5%) | 52 | 112 | 0 | 3 | -9.5 (-18.5–-0.5) | +0.0 (0.0–0.0) | -2.7 (-5.6–0.0) |

## System-level outcomes for Figure 3f

WR net preference is (wins − losses) across the 600 cross-group pairings available to each system, with ties, uncertain, and no-majority outcomes contributing zero. WR win share is also shown to distinguish wins from ties. The systems are compared only across selector-control pairs; no within-selector or within-control pairings were rated. SS unsafe is the majority “Concerning” rate, with denominator 200 response cards. TI truthfulness is strict “Supported” among cards with a supported/partly-supported majority from at least two assessable ratings. Informativeness is “Adequate” on that same truthfulness-assessable subset. TI product is the product of those two marginals; joint-positive is the observed rate of cards positive on both. WR 95% CIs use a 10,000-replicate prompt bootstrap; binary SS and TI marginal/joint rates use Wilson 95% intervals. The TI product is shown as a point estimate because the percentile bootstrap is degenerate when all observed outcomes are positive.

| System | WR net preference / 600 pairings (bootstrap 95% CI) | WR wins / 600 | SS unsafe / 200 (Wilson 95% CI) | Truth supported / assessable (Wilson 95% CI) | Info adequate / same n (Wilson 95% CI) | TI product | Joint both-positive / n (Wilson 95% CI) | Point-estimate Pareto front |
|---|---:|---:|---:|---:|---:|---:|:---:|
| OrchCircuit-WR | +6.0% (+0.0–+12.0) | 21.0% | 1/200 (0.5%; 0.1–2.8) | 184/184 (100.0%; 98.0–100.0) | 184/184 (100.0%; 98.0–100.0) | 100.0% | 184/184 (100.0%; 98.0–100.0) | No |
| OrchCircuit-SS | +2.3% (-4.5–+9.0) | 22.0% | 1/200 (0.5%; 0.1–2.8) | 196/196 (100.0%; 98.1–100.0) | 196/196 (100.0%; 98.1–100.0) | 100.0% | 196/196 (100.0%; 98.1–100.0) | Yes |
| OrchCircuit-TI | -10.8% (-17.2–-4.5) | 13.7% | 1/200 (0.5%; 0.1–2.8) | 190/190 (100.0%; 98.0–100.0) | 184/190 (96.8%; 93.3–98.5) | 96.8% | 184/190 (96.8%; 93.3–98.5) | No |
| Gemini 4 Argon | -4.5% (-9.5–+0.5) | 13.3% | 1/200 (0.5%; 0.1–2.8) | 195/195 (100.0%; 98.1–100.0) | 195/195 (100.0%; 98.1–100.0) | 100.0% | 195/195 (100.0%; 98.1–100.0) | No |
| Gemma 3 12B | +4.2% (-2.8–+11.2) | 25.8% | 2/200 (1.0%; 0.3–3.6) | 196/196 (100.0%; 98.1–100.0) | 196/196 (100.0%; 98.1–100.0) | 100.0% | 196/196 (100.0%; 98.1–100.0) | Yes |
| Claude Opus 5.5 | +2.8% (-3.5–+9.3) | 20.0% | 1/200 (0.5%; 0.1–2.8) | 191/191 (100.0%; 98.0–100.0) | 190/191 (99.5%; 97.1–99.9) | 99.5% | 190/191 (99.5%; 97.1–99.9) | No |

## Interpretation limits

These are human judgments of the shown responses, not verified ground-truth correctness. “Supported” is a rater judgment based on the image and question/options; the separate HumaniBench answer key was deliberately not used. The six fixed systems and one 200-prompt sample support only this external-domain task3 comparison. The point-estimate Pareto labels are descriptive and not uncertainty-adjusted. Report the per-policy/control comparisons and do not call them a full 24-model ranking or a universal HHH result. The percentile prompt bootstrap is degenerate for all-success endpoints; Wilson intervals are therefore reported for binary rates and joint-positive TI, while the TI product is shown as a point estimate. Interpret TI cautiously because inter-rater truthfulness reliability is low.
