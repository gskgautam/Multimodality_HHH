# Figure 4b — frozen external contextual-safety transfer protocol

**Status:** external test frame and pair mapping frozen; candidate inference and response evaluation have not started.
**Study type:** metric-specific external transfer of the already-trained OrchCircuit selector.
**Source:** MM-SafetyBench++ test split, dataset revision 128ea182089b677a8d48d97c9f86425713351ac2.

## What this experiment can test

Test whether the frozen OrchCircuit selector, trained only on Ch3EF, can choose a safer candidate response on an external paired contextual-safety benchmark. This is an SS-focused transfer test. MM-SafetyBench++ does not provide WR or TI labels, so this cannot support full HHH transfer or overall HHH claims.

The source contains 2,844 records organized as 711 source items. Each item has four records: unsafe-intent and safe-intent counterparts, each in Gen and GenOCR modes. Pair identity was reconstructed and checked against all 2,844 corresponding prompt rows in the benchmark authors' safe/unsafe source files. The Hugging Face export has no native pair_id. Details, upstream-source SHA-256 values, and the generated manifest are in source_audit/mm_safetybench_pairing_audit.json, source_audit/mm_safetybench_manifest_summary.json, and source_audit/mm_safetybench_external_test_manifest.jsonl.

## Locked test frame and leakage controls

- Use all 711 paired source items (2,844 rows); no sample is selected after seeing outcomes.
- Freeze OrchCircuit weights, calibration, thresholds, candidate order, and generation settings before external responses are generated.
- Keep all four records belonging to a pair_id together for any resampling or reporting. Use pair-cluster bootstrap intervals.
- Do not train, tune, calibrate, or select a method using this external test set.
- Do not save signed image URLs in Git or the manifest. Retrieve images from the pinned source revision at run time.
- Record exact model IDs/revisions, generation parameters, prompt templates, feature extractor revision, judge versions, code hashes, failures, and cost.

## Candidate pool and scale

The existing OrchCircuit implementation expects the same 24 candidate slots used in the Ch3EF study. A faithful transfer run therefore requires a response for each of the 24 baseline models on each of the 2,844 rows: 68,256 candidate responses before retries. No external candidate responses were found in the repository or its output folders during preflight. Substituting a smaller pool, padding missing slots, or retraining on external labels changes the experiment and must be reported as a different protocol.

The currently saved model output files cover Ch3EF only. External response generation is the next operational dependency; paid providers and large local model downloads/runs have not been started.

## Outcomes

The benchmark's safe/unsafe field describes the request context, not whether a generated response is safe or helpful. It must never be treated as the response-level outcome.

For every generated response, obtain independently defined response-level labels for:
1. Unsafe compliance on unsafe-intent records (lower is better).
2. Unnecessary refusal on safe-intent counterparts (lower is better; needed to detect refusal-only behavior).

Report these outcomes separately, along with their paired difference by source item, category, and mode. Do not collapse them into one score. The already-used Beaver-Dam SS evaluator can be a continuity measure only after its suitability on this source is checked against a blinded human-rated subset. Limit claims to measured outcomes and evaluator validation.

## Comparisons

With the identical 24 generated candidate responses on each row, compare:
- frozen OrchCircuit selection;
- each of the 24 individual candidate models;
- a label-free random candidate selector, using a fixed published seed.

If adding a learned/threshold selector based on external labels, call it an exploratory adaptation and keep it out of the transfer claim. Show all 24 baseline comparisons, denominators, per-category/mode outcomes, and pair-cluster uncertainty.

## Gates before response generation

1. Freeze and hash the model-version list and provider mappings for the existing 24 slots.
2. Inventory local checkpoints and API availability without exposing secrets; calculate cost from the exact planned record count and observed token/image usage assumptions.
3. Freeze the human validation sample, rubric, blinding, independent rating workflow, and adjudication plan before seeing model outputs.
4. Obtain approval for any paid API spend and model/data transfer to external providers.
5. Run a small technical smoke test only after approval; then freeze its prompt, decode parameters, and response schema before launching the locked full frame.

This is exploratory external transfer because OrchCircuit has already been developed on Ch3EF. It is not a new confirmatory test and does not validate WR or TI.


## Server availability and preliminary cost evidence

- Ollama currently lists only Qwen2.5-VL 7B. The Hugging Face cache contains some vision checkpoints, but the full 15-model open-weight set is not present/verified as runnable.
- Existing Ch3EF provider-usage logs total about US$39.50 across the nine closed-source models for the 1,416-record baseline run. A simple 2,844/1,416 scale-up is about US$79.30 for closed-model generation alone. This is a rough historical extrapolation, not a quote; external prompts/images, longer safety responses, retries, pricing changes, and response-label validation can raise it.
- This extrapolation excludes local compute for the 15 open-weight models and human rating. No 4b API requests or model downloads have been made.
- Because the fixed candidate set is incomplete locally and a faithful 4b needs 68,256 outputs, do not launch partial generations and present them as the locked comparison. First freeze the provider/version list, price the actual settings, and cap spend.
