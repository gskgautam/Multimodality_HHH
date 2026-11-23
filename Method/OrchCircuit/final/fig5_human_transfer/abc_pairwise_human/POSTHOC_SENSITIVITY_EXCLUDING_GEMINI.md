# Post hoc sensitivity: omit Gemini 4 Argon

This is an exploratory sensitivity analysis. Gemini 4 Argon was a pre-specified WR control selected from development-set Table 1 results. Its external-sample generation file has 200 unique case IDs and 200 nonempty responses; the frozen protocol records no data or protocol failure that would justify excluding it. Therefore this analysis does not replace the complete comparison.

For each OrchCircuit policy, results below pool only its 400 pairings against the remaining two fixed controls (Gemma 3 12B and Claude Opus 5.5). Net WR is wins minus losses; ties and no-majority outcomes contribute zero. SS and TI deltas are paired selector-minus-control differences averaged over the two retained controls. CIs are pointwise 10,000-replicate prompt-bootstrap intervals, without multiplicity adjustment.

| Policy | WR wins / losses / ties / split (n=400) | Net WR advantage, pp (95% CI) | Δ unsafe, pp (95% CI) | Δ joint TI-positive, pp (95% CI) |
|---|---:|---:|---:|---:|
| OrchCircuit-WR | 38 / 35 / 325 / 2 | +0.8 (-3.2–+4.8) | +0.0 (+0.0–+0.0) | +8.1 (-1.6–+17.7) |
| OrchCircuit-SS | 38 / 34 / 327 / 1 | +1.0 (-3.2–+5.2) | +0.0 (+0.0–+0.0) | -7.5 (-16.1–+1.1) |
| OrchCircuit-TI | 47 / 47 / 306 / 0 | +0.0 (-4.8–+4.8) | +0.0 (+0.0–+0.0) | +16.8 (+6.7–+27.0) |

## Conclusion

This post hoc omission does not replace the pre-specified full comparison. Results vary across selector-control pairings and criteria, and safety is identical across systems in this sample. Gemini had complete usable outputs and remains in the primary analysis; do not infer overall superiority from this sensitivity.
