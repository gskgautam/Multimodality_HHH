# Figure 4b — external-source transfer: protocol draft

**Status: feasibility draft only; no model outputs or judge labels have been generated.** Do not describe this panel as an experiment until the source, target metric, candidate-answer procedure, holdout, and scoring labels are frozen.

## Source audit already completed

### VistaQA (candidate for TI / hallucination transfer)
- Audited split: train, 1,157 records; exact normalized-question overlap with Ch3Ef: 0.
- The image audit found two exact decoded-pixel matches with Ch3Ef (VistaQA image IDs 1025 and 169) and one image fetch failure (ID 1079). Exclude these three records before freezing any sample; the remaining upper-bound candidate count is 1,154, subject to any further checks.
- VistaQA contains a benchmark answer and a hallucination annotation for its own answer. Those annotations are not labels for newly generated answers from the 24 baseline VLMs.
- The dataset card declares CC BY 4.0. This does not by itself verify upstream rights for every embedded image; source-level image terms remain a release/redistribution check.
- This source supports a metric-specific truthfulness/hallucination transfer test, not full WR/SS/TI evaluation.

### MM-SafetyBench++
- Audited split: test, 2,844 records; exact normalized-question and image-content overlap with Ch3Ef: 0 in the audit.
- It provides safe/unsafe prompt-image contexts, but no WR or TI label. Its safe/unsafe variants must remain grouped in any split. The audit found no explicit pair ID, so the pairing/group reconstruction must be verified before using it.
- The dataset card declares CC BY 4.0; upstream image rights still need review.
- This source can support an SS-specific external test, not full HHH evaluation.

## Proposed analysis structure

If both sources pass the required checks, Figure 4b can be a **metric-specific external-transfer exhibit** with separate TI/VistaQA and SS/MM-SafetyBench++ panels. Keep the outcomes separate; do not average them into a joint HHH score. If the paper needs a single full-HHH transfer claim, these sources alone do not supply it.

Before any generation or judging:
1. Confirm acceptable use of embedded images for research evaluation and any intended public release; keep source images out of Git.
2. Freeze record exclusions, query/image/pair-grouped split, sample size, random seed, and analysis code.
3. Decide how the same fixed 24-candidate pool will be produced on each external record. The current selector expects all 24 candidate slots; a smaller pool changes the experiment.
4. Decide the target labels for generated answers. VistaQA's supplied hallucination labels describe its existing reference answer, not each new candidate. If new candidate responses are scored by a judge or human raters, define that rubric and label-quality plan separately.
5. Set a hard API/compute budget and approve it before paid calls. No paid calls have been made for Figure 4b.

All work would remain exploratory because OrchCircuit and the baseline outputs were developed using Ch3Ef. A fresh source is useful for transfer testing, but it does not make the result confirmatory if the protocol or outcomes are chosen after viewing results.
