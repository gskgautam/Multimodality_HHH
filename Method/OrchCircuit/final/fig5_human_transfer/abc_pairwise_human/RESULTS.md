# Human-rated Figure 3e results

Frozen, external-domain HumaniBench task3 sample: 200 prompts (40 per Age, Ethnicity, Gender, Occupation, Sport). Three independent raters scored three OrchCircuit policy selections and three pre-specified controls per prompt. This is a six-system human-rated comparison, not a human evaluation of all 24 candidate generators. No answer key or source reasoning was used in this analysis.

WR uses majority-normalized direct pairwise human judgments. Net preference is (selector wins − control wins)/200 prompts; ties, uncertain decisions, and 3-way no-majority cases contribute zero and remain in the denominator. SS is the paired change in majority unsafe flag rate, selector minus control (negative favors the selector). TI is the paired change in joint strict-positive rate (truthfulness Supported AND informativeness Adequate), selector minus control (positive favors the selector). Pairwise 95% CIs use a 10,000-replicate prompt bootstrap, seed 20261113. Intervals are pointwise and not adjusted across the nine policy-control matchups; treat them as descriptive rather than standalone confirmatory tests.

| OrchCircuit policy | Fixed control | WR policy wins / 200 | Control wins | Ties | Uncertain | No majority | Net WR advantage, pp (95% CI) | SS Δ unsafe, pp (95% CI) | TI Δ joint-positive, pp (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OrchCircuit-WR | Gemini 4 Argon | 27/200 (13.5%) | 19 | 153 | 0 | 1 | +4.0 (-2.5–+10.5) | +0.0 (0.0–0.0) | -9.4 (-18.4–-0.9) |
| OrchCircuit-WR | Gemma 3 12B | 15/200 (7.5%) | 19 | 166 | 0 | 0 | -2.0 (-8.0–+3.5) | +0.0 (0.0–0.0) | +22.4 (10.5–34.4) |
| OrchCircuit-WR | Claude Opus 5.5 | 23/200 (11.5%) | 16 | 159 | 0 | 2 | +3.5 (-2.5–+9.5) | +0.0 (0.0–0.0) | -3.8 (-14.6–7.1) |
| OrchCircuit-SS | Gemini 4 Argon | 24/200 (12.0%) | 14 | 162 | 0 | 0 | +5.0 (-1.0–+11.0) | +0.0 (0.0–0.0) | -23.1 (-34.1–-11.7) |
| OrchCircuit-SS | Gemma 3 12B | 17/200 (8.5%) | 17 | 165 | 0 | 1 | +0.0 (-5.5–+5.5) | +0.0 (0.0–0.0) | +10.6 (0.8–20.3) |
| OrchCircuit-SS | Claude Opus 5.5 | 21/200 (10.5%) | 17 | 162 | 0 | 0 | +2.0 (-4.0–+8.0) | +0.0 (0.0–0.0) | -22.8 (-33.9–-11.5) |
| OrchCircuit-TI | Gemini 4 Argon | 24/200 (12.0%) | 15 | 159 | 0 | 2 | +4.5 (-1.5–+10.5) | +0.0 (0.0–0.0) | +2.3 (-3.7–8.3) |
| OrchCircuit-TI | Gemma 3 12B | 24/200 (12.0%) | 24 | 152 | 0 | 0 | +0.0 (-7.0–+6.5) | +0.0 (0.0–0.0) | +32.7 (20.3–45.2) |
| OrchCircuit-TI | Claude Opus 5.5 | 23/200 (11.5%) | 23 | 154 | 0 | 0 | +0.0 (-6.5–+7.0) | +0.0 (0.0–0.0) | -3.0 (-14.6–9.0) |

## System-level outcomes for Figure 3f

WR net preference is (wins − losses) across the 600 cross-group pairings available to each system, with ties, uncertain, and no-majority outcomes contributing zero. WR win share is also shown to distinguish wins from ties. The systems are compared only across selector-control pairs; no within-selector or within-control pairings were rated. SS unsafe is the majority “Concerning” rate, with denominator 200 response cards. TI truthfulness is strict “Supported” among cards with a supported/partly-supported majority from at least two assessable ratings. Informativeness is “Adequate” on that same truthfulness-assessable subset. TI product is the product of those two marginals; joint-positive is the observed rate of cards positive on both. WR 95% CIs use a 10,000-replicate prompt bootstrap; binary SS and TI marginal/joint rates use Wilson 95% intervals. TI product intervals use the 10,000-replicate prompt bootstrap; binary rates use Wilson 95% intervals.

| System | WR net preference / 600 pairings (bootstrap 95% CI) | WR wins / 600 | SS unsafe / 200 (Wilson 95% CI) | Truth supported / assessable (Wilson 95% CI) | Info adequate / same n (Wilson 95% CI) | TI product (bootstrap 95% CI) | Joint both-positive / n (Wilson 95% CI) | Point-estimate Pareto front |
|---|---:|---:|---:|---:|---:|---:|:---:|
| OrchCircuit-WR | +1.8% (-1.5–+5.2) | 10.8% | 0/200 (0.0%; 0.0–1.9) | 98/129 (76.0%; 67.9–82.5) | 107/129 (82.9%; 75.5–88.5) | 63.0% (55.0–71.0) | 82/129 (63.6%; 55.0–71.4) | No |
| OrchCircuit-SS | +2.3% (-1.3–+6.0) | 10.3% | 0/200 (0.0%; 0.0–1.9) | 90/156 (57.7%; 49.8–65.2) | 125/156 (80.1%; 73.2–85.6) | 46.2% (39.4–53.4) | 70/156 (44.9%; 37.3–52.7) | No |
| OrchCircuit-TI | +1.5% (-2.3–+5.3) | 11.8% | 0/200 (0.0%; 0.0–1.9) | 133/141 (94.3%; 89.2–97.1) | 112/141 (79.4%; 72.0–85.3) | 74.9% (68.1–81.4) | 104/141 (73.8%; 65.9–80.3) | Yes |
| Gemini 4 Argon | -4.5% (-8.0–-1.0) | 8.0% | 0/200 (0.0%; 0.0–1.9) | 132/140 (94.3%; 89.1–97.1) | 103/140 (73.6%; 65.7–80.2) | 69.4% (62.0–76.4) | 96/140 (68.6%; 60.5–75.7) | No |
| Gemma 3 12B | +0.7% (-3.0–+4.3) | 10.0% | 0/200 (0.0%; 0.0–1.9) | 65/144 (45.1%; 37.2–53.3) | 125/144 (86.8%; 80.3–91.4) | 39.2% (31.8–46.8) | 56/144 (38.9%; 31.3–47.0) | No |
| Claude Opus 5.5 | -1.8% (-5.7–+2.0) | 9.3% | 0/200 (0.0%; 0.0–1.9) | 111/134 (82.8%; 75.6–88.3) | 110/134 (82.1%; 74.7–87.7) | 68.0% (60.2–75.8) | 92/134 (68.7%; 60.4–75.9) | No |

## Interpretation limits

These are human judgments of the shown responses, not verified ground-truth correctness. “Supported” is a rater judgment based on the image and question/options; the separate HumaniBench answer key was deliberately not used. The six fixed systems and one 200-prompt sample support only this external-domain task3 comparison. The point-estimate Pareto labels are descriptive and not uncertainty-adjusted. Report the per-policy/control comparisons and do not call them a full 24-model ranking or a universal HHH result. Direct 3e WR net-preference intervals include zero for all nine policy-control matchups. Safety has no variation in this sample. TI measures are reviewer judgments rather than answer-key correctness; interpret the policy-specific differences alongside the agreement results.

## Exploratory tie-adjusted WR sensitivity (post hoc)

Secondary sensitivity only; it does not replace the primary nine pairwise net-preference results. Each selector is pooled across its three fixed controls (600 pairings). A selector win contributes 1, a tie 0.5, and a control win, uncertain, or no-majority outcome 0. The 95% interval resamples the 200 prompts as clusters (10,000 replicates, seed 20261113).

| OrchCircuit policy | Wins | Control wins | Ties | Uncertain | No majority | Tie-adjusted preference / 600 (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| OrchCircuit-WR | 65 | 54 | 478 | 0 | 3 | 50.7% (49.0–52.3) |
| OrchCircuit-SS | 62 | 48 | 489 | 0 | 1 | 51.1% (49.2–52.9) |
| OrchCircuit-TI | 71 | 62 | 465 | 0 | 2 | 50.6% (48.7–52.5) |

All three point estimates are near 50%; this is a sensitivity analysis, not evidence of a large or established helpfulness advantage.
