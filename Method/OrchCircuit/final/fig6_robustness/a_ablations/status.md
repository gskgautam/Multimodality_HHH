# Figure 4a — component ablation

**Status: rerun complete; exploratory.** All three planned component removals and the full selector were retrained with matched initialization across three training seeds. Outer and inner query/image-grouped splits were held fixed. Initialization fingerprints confirmed identical starting weights for shared modules in every seed/fold.

- Current results: results/RESULTS.md
- Plot: results/figure4a_ablation_effects.png and .pdf
- Machine-readable results: results/matched_ablation_effects.csv and .json
- All twelve run folders contain fold manifests, per-fold training/calibration records, and out-of-fold selections.

The matched results show reliable reductions on some metric/component pairs: removing candidate-identity embeddings lowers WR, while replacing learned token pooling with mean pooling increases SS flags and lowers TI. The other planned contrasts are inconclusive after adjustment. Report the complete table; do not select only the reductions. Labels are automated-judge outputs, and the data were used in prior development, so these remain exploratory findings rather than claims of human-verified quality or fresh generalization.

The earlier one-seed native-initialization table is retained as results/initial_native_init_results.md for audit and is superseded.
