# Human-rated Figure 3f results

Six systems were rated on the same fixed sample of 200 HumaniBench task3 prompts by three independent reviewers. The six systems are the three criterion-specific OrchCircuit selections plus three development-set controls; this is not a ranking of all 24 candidate generators. No answer key or source reasoning was used.

WR net preference is (wins − losses)/600 across the three selector-control matchups per system; ties, uncertain ratings, and no-majority ratings contribute zero. SS is the majority “Concerning” response rate (lower is better). TI product is strict truthfulness Supported rate multiplied by Adequate informativeness rate on the same truthfulness-assessable subset; joint both-positive is also shown. WR confidence intervals are prompt-bootstrap; binary marginal and joint rates use Wilson intervals. TI-product intervals use the 10,000-replicate prompt bootstrap.

| System | WR net preference / 600 (95% bootstrap CI) | WR win share | SS unsafe / 200 (95% Wilson CI) | Truth supported / assessable (95% Wilson CI) | Info adequate / same n (95% Wilson CI) | TI product | Joint both-positive / n (95% Wilson CI) | Point-estimate Pareto front |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| OrchCircuit-WR | +1.8% (-1.5–+5.2) | 10.8% | 0/200 (0.0%; 0.0–1.9) | 98/129 (76.0%; 67.9–82.5) | 107/129 (82.9%; 75.5–88.5) | 63.0% (55.0–71.0) | 82/129 (63.6%; 55.0–71.4) | No |
| OrchCircuit-SS | +2.3% (-1.3–+6.0) | 10.3% | 0/200 (0.0%; 0.0–1.9) | 90/156 (57.7%; 49.8–65.2) | 125/156 (80.1%; 73.2–85.6) | 46.2% (39.4–53.4) | 70/156 (44.9%; 37.3–52.7) | No |
| OrchCircuit-TI | +1.5% (-2.3–+5.3) | 11.8% | 0/200 (0.0%; 0.0–1.9) | 133/141 (94.3%; 89.2–97.1) | 112/141 (79.4%; 72.0–85.3) | 74.9% (68.1–81.4) | 104/141 (73.8%; 65.9–80.3) | Yes |
| Gemini 4 Argon | -4.5% (-8.0–-1.0) | 8.0% | 0/200 (0.0%; 0.0–1.9) | 132/140 (94.3%; 89.1–97.1) | 103/140 (73.6%; 65.7–80.2) | 69.4% (62.0–76.4) | 96/140 (68.6%; 60.5–75.7) | No |
| Gemma 3 12B | +0.7% (-3.0–+4.3) | 10.0% | 0/200 (0.0%; 0.0–1.9) | 65/144 (45.1%; 37.2–53.3) | 125/144 (86.8%; 80.3–91.4) | 39.2% (31.8–46.8) | 56/144 (38.9%; 31.3–47.0) | No |
| Claude Opus 5.5 | -1.8% (-5.7–+2.0) | 9.3% | 0/200 (0.0%; 0.0–1.9) | 111/134 (82.8%; 75.6–88.3) | 110/134 (82.1%; 74.7–87.7) | 68.0% (60.2–75.8) | 92/134 (68.7%; 60.4–75.9) | No |

## Interpretation

The point-estimate Pareto front is descriptive, not uncertainty-adjusted. OrchCircuit-TI has the highest system-level WR win-share and TI-product point estimates in this six-system set, but the direct 3e WR intervals include zero and TI-product intervals overlap with controls. All systems have zero unsafe flags, so this sample cannot distinguish safety performance. These findings suggest a promising TI-policy point estimate, not confirmed overall HHH superiority or verified correctness.
