# Figure 4b — external transfer

**Input frame ready; candidate model evaluation has not started.**

## Completed

- Pinned the MM-SafetyBench++ test revision and loaded all 2,844 rows from the test split.
- Decoded all 2,844 images successfully. Every image is 1024 x 1024; no image decode failures. The dataset is cached outside the repository (about 3.1 GB), and no image files were added to Git.
- Verified the exported prompt/mode rows against the benchmark authors' safe and unsafe source files: 2,844/2,844 matched.
- Reconstructed and validated 711 matched source-item groups. Each group contains safe/unsafe contexts in both Gen and GenOCR modes.
- Generated an image-free manifest with no signed URLs:
  - source_audit/mm_safetybench_external_test_manifest.jsonl
  - source_audit/mm_safetybench_manifest_summary.json
  - source_audit/mm_safetybench_pairing_audit.json
  - source_audit/mm_safetybench_data_integrity.json
- Added the reproducible manifest builder and cached-data validator in source_audit/.
- Added the frozen run protocol in RUN_PROTOCOL.md.

## Not started

No candidate model responses, external feature extraction, OrchCircuit scores, response-level safety/refusal ratings, paid judge calls, or 4b performance plots have been produced.

A faithful run needs 24 responses for each of 2,844 prompts: 68,256 candidate responses before retries. The existing selector requires those same 24 slots; shrinking the pool or filling missing slots with placeholders would invalidate direct transfer. The dataset's safe/unsafe tag labels request context, not generated answer quality. Response-level unsafe compliance and unnecessary refusal must be rated separately; the Beaver-Dam evaluator requires a blinded human validation check on this new source.

## Next

Freeze the 24 model versions/settings and validation plan, then calculate a run-specific cost ceiling. Historical usage suggests roughly US$79 for the nine closed-source models alone, before retries or response-label validation; see RUN_PROTOCOL.md. Ollama currently lists only Qwen2.5-VL 7B, so the open-weight pool also needs a runtime/checkpoint inventory before any launch.
