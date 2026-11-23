# Figure 4: analyses tied to the observed findings

## Study pattern to explain

Across 24 baselines, OrchCircuit has favorable point estimates for SS and TI, while WR trails three closed models. The all-baseline adjusted confidence-interval gate did not establish a reliable win over every baseline. The image-access audit shows large, judge-dependent WR and TI shifts when images are withheld. Figure 3a-d shows limited judge-human agreement, and Figure 3e-f does not establish a general human-rated HHH advantage. Figure 4a/c show component- and seed-dependent variation.

The added panels test measurement validity, cross-question heterogeneity, same-prompt objective trade-offs, and human-rated outcome differences. They are not attempts to manufacture a superiority claim.

## 4b — Human triangulation of the image-access diagnostic

Human ratings should use the image-present prompt and the exact same response, then compare human ratings with automated outputs from image-present and image-omitted conditions separately for WR, SS and TI.

Existing Figure 3a-d forms overlap only 5 WR, 10 SS and 18 TI exact prompt/model pairs with the diagnostic sample. That is too little for validation. The OrchCircuit diagnostic extension has 77 Helpful, 75 Harmless and 77 Honest eligible records. Prepare a new blinded human rating packet for eligible records; retain the identity mapping privately. No model or judge calls are needed.

This asks whether image-present judge ratings align more closely with human ratings than image-omitted ratings. It does not treat human labels as objective truth. These Ch3EF prompts were used in method development, so the panel is exploratory, not confirmatory.

## 4d — Cross-question source strata

Computed from saved three-seed grouped-OOF selections and per-item evaluator outcomes. Comparators are fixed from the full-set OOF table: WR/TI Gemini 4 Argon, SS Gemma 3 12B. Source labels are normalized only for case, spaces, underscores and hyphens. No outcome-based merging/filtering was used.

All 53 source-by-metric estimates are saved. The plot includes strata with n≥12 and at least five connected query/image groups; the CSV/report retain smaller strata. Intervals resample connected groups and average paired differences over three seeds. Intervals are unadjusted across strata; subgroup findings are exploratory.

Files: d/RESULTS.md, d/source_metric_strata.csv, d/figure4d_source_strata.png/PDF, d_analyze_cross_question.py.

## 4e — Same-prompt three-objective trade-off

Computed on the existing 600-prompt joint-label development cohort. Each saved final OrchCircuit grouped-OOF selection is rescored on joint WR, SS and TI labels; all fixed models use those same prompts and labels.

The cohort was used in earlier OrchCircuit development. This is a post-hoc automated-label diagnostic, not a fresh test or correctness evaluation. WR is preference against Qwen2.5-VL 7B; SS is Beaver-Dam text-only; TI is GPT-4o truthful × informative. No composite score is used.

Files: e/RESULTS.md, e/results.json, e/joint_hhh_same_prompt.csv, e/figure4e_joint_hhh_tradeoff.png/PDF, e_analyze_joint_hhh.py.

## 4f — Human-rated matched comparison

Rate the exact OrchCircuit-selected response against the preselected fixed comparator on the same prompt: Gemini 4 Argon for WR/TI and Gemma 3 12B for SS. Keep the metrics separate. The same human ratings may also support 4b, but these are two analyses of one sample, not independent replications.

Existing Figure 3a-d data exactly match only 8 Helpful, 12 Harmless and 9 Honest V4-selected cases. Insufficient. A new blinded matched human set is required. If reusing prior raters, avoid their previously seen prompts or document repeated exposure. No human-superiority claim until new ratings are returned and analyzed.
