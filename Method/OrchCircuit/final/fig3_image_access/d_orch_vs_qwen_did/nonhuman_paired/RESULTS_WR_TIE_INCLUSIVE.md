# Figure 4b WR sensitivity - tie-inclusive definitions

Same records, pairing, groups, 10,000 group-bootstrap replicates and seed as analyze_paired_shift.py; only the WR scoring changes so every paired item stays in both condition denominators.

| WR definition | Judge | Common prompts | Groups | Orch shift pp | Qwen shift pp | Difference pp (95% group-bootstrap CI) |
|---|---|---:|---:|---:|---:|---:|
| wins/all | anthropic | 77 | 65 | +42.86 | -3.90 | +46.75 [+29.85, +61.86] |
| wins/all | google | 73 | 62 | +28.77 | -6.85 | +35.62 [+17.19, +55.00] |
| (wins-losses)/all | anthropic | 77 | 65 | +45.45 | -29.87 | +75.32 [+46.66, +101.22] |
| (wins-losses)/all | google | 73 | 62 | +46.58 | -31.51 | +78.08 [+47.22, +107.32] |
