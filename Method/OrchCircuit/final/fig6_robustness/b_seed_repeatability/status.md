# Figure 4c — training-seed repeatability

**Status: complete; exploratory.** The full selector was evaluated with training seeds 20261111, 20261112, and 20261113, keeping all five outer folds and grouped validation assignments fixed. These are the same matched-initialization full-selector runs used as the reference for Figure 4a.

- Current results: results/RESULTS.md
- Plot: results/figure4c_seed_repeatability.png and .pdf
- Machine-readable results: results/seed_repeatability.csv and .json
- Source runs, including fold manifests, initialization fingerprints, and OOF predictions: ../a/runs/full_seed{20261111,20261112,20261113}_matched/

Across these runs, WR ranges 37.75–41.97% (4.22 pp), SS unsafe rate ranges 19.10–22.47% (3.37 pp), and TI product ranges 86.40–90.50% (4.10 pp). These ranges are descriptive training variation on previously used development data, not uncertainty across independent test sets. Scores are automated-judge-relative.

The earlier native-initialization seed analysis is retained under results/initial_native_init/ for audit and is superseded by the matched-initialization results.
