# Figure 4f - Judge-label disagreement tipping point

## Purpose and scope

This analysis asks how small an item-label change could erase the observed OrchCircuit-versus-fixed-baseline point-estimate margins. It uses saved item counts and does not adjust, relabel, or claim to correct any scores. Full-set outcomes and the 600-prompt joint-label outcomes are separate cohorts.

### Saved-result tipping points

| Cohort | Metric | OrchCircuit outcome counts | Fixed comparator | Favorable margin (pp) | Minimum directional changes to tie/erase | Share of outcomes |
|---|---|---|---|---:|---:|---:|
| Full-set strict common-valid OOF | WR | 192 wins | Gemini 4 Argon: 209 wins | -3.41 | 17 | 3.41% |
| Full-set strict common-valid OOF | SS | 96 unsafe | Gemma 3 12B: 118 unsafe | +4.94 | 22 | 4.94% |
| Full-set strict common-valid OOF | TI | 281 truthful, 282 informative | Gemini 4 Argon: 280 truthful, 280 informative | +0.95 | 3 | 0.50% |
| 600-prompt joint-label development cohort | WR | 194 wins | Gemini 4 Argon: 196 wins | -0.33 | 2 | 0.33% |
| 600-prompt joint-label development cohort | SS | 221 unsafe | Gemma 3 12B: 251 unsafe | +5.00 | 30 | 5.00% |
| 600-prompt joint-label development cohort | TI | 542 truthful, 565 informative | Gemini 4 Argon: 543 truthful, 560 informative | +0.60 | 4 | 0.33% |

WR already trails Gemini 4 Argon in both cohorts: it would need 17 net additional wins (full set) or 2 (600 prompts) to tie. The favorable SS point estimate is erased by 22 adverse labels in the full set and 30 in the 600-prompt cohort. For TI, changing three OrchCircuit informative labels (full set) or four (600-prompt cohort) from positive to negative is sufficient to reach or cross the comparator's truthfulness-by-informativeness product. These are arithmetic tipping points, not probabilities that the observed results are wrong.

### Human-judge disagreement observed in Figure 3a-d

| Dimension | Human-validation n | Judge-human disagreement | 95% interval |
|---|---:|---:|---:|
| WR exact 3-way preference | 147 | 67.3% | [59.4, 74.4]% |
| SS binary safety | 141 | 43.3% | [35.4, 51.5]% |
| TI truthfulness | 156 | 57.1% | [49.2, 64.6]% |
| TI informativeness | 187 | 48.7% | [41.6, 55.8]% |

These disagreement rates are context, not correction rates: they were measured on a separate human-validation sample and cannot be transferred to every item in either cohort. Human-judge disagreement may reflect rubric ambiguity or pluralistic judgments, not simply judge error.

## What this establishes and what remains unmeasured

The SS and TI point-estimate margins are sensitive to a small number of outcome-label changes. The data do not establish that those changes actually occur on the evaluated cases. This calculation also does not show whether noisy training labels would change OrchCircuit's selected candidate; aggregate results cannot reveal that. Estimating selection instability requires perturbing training labels under a prespecified transition model, refitting the selector, and evaluating on independently human-rated, untouched cases.

## Confirmatory status

This retrospective analysis uses Ch3EF data already involved in method development. It is not a confirmatory test, and it cannot make 4d, 4e, or Table 1 confirmatory. For a confirmatory Figure 4, freeze the method and analysis first, then evaluate on a genuinely untouched set with independent human ratings.

## Reproducibility

- Primary counts: `Method/OrchCircuit/final/runs/fullset_group5fold/results.md` and its OOF outputs.
- Joint-cohort counts: `Method/OrchCircuit/final/figure4/e/results.json`.
- Human-judge disagreement: `Method/OrchCircuit/final/figure3/common/human_validation_20261113/figure3a_d_human_validation_results.json`.
- Machine-readable outputs: `tipping_point_counts.csv`, `figure3_human_judge_disagreement_context.csv`, `results.json`.
- No API calls, model generations, or label edits were made.
