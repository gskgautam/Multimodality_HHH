# Teaching to the AI Judge Makes Multimodal Models Less Accurate

Code for evaluating 24 VLMs with automated judges, blinded human raters and an answer key, and for the OrchCircuit selector trained on judge labels. Sections 6–12 reproduce Figures 1–6 of the paper.

## Repository Map

~~~text
Baselines/
  Open/                         15 Open-Weight Model Launchers
  Closed/                       Hosted VLM Launchers and Provider Bridge
  Judge/                        WR/TI and SS Scoring, Image-Access Audit (Fig. 3)
Datasets/Ch3EF/                 Dataset Setup and Evaluation Prompt
Method/OrchCircuit/final/
  run_selector_cv.py            OrchCircuit Grouped Five-Fold Run (Fig. 4)
  extract_features.py           Frozen Qwen2-VL-7B Feature Extraction
  runs/fullset_group5fold/      Saved Selector Results (Fig. 4)
  fig2_judge_validity/          Judges vs Human Raters (Fig. 2)
  fig3_image_access/            Image-Access Audit, Panel d (Fig. 3)
  fig5_human_transfer/          External Items, Ratings and Answer Key (Fig. 5)
  fig6_robustness/              Ablations, Seeds, Strata, Trade-off, Tipping Points (Fig. 6)
  review/                       Analysis Scripts Behind Figs. 2, 4, 5 and 6, and the Held-Out-Item Pipeline
  shared/human_rating_pipeline/ Human-Rating Analysis, External Generation and Final Selector
  figures/                      Paper Panels (Fig1-Fig6.tex) and Their Renderer
README.md                       Setup and Figure-by-Figure Reproduction Guide
~~~

The primary comparison includes 15 completed open-weight baselines. Each has a folder and `run.py` launcher under `Baselines/Open/`; ten use the shared Ollama runner, while five retain model-specific adapters. The nine hosted launchers are under `Baselines/Closed/`.

| Size Group | Open-Weight Models |
|---|---|
| ~2B | LFM3-VL 2B, Qwen3.5 2B, Gemma 4 E2B |
| ~7–8B | Qwen2.5-VL 7B, Qwen3-VL 8B, MiniCPM-V 4.5 8B |
| ~12B | Gemma 3 12B, Gemma 4 12B, Pixtral 12B |
| ~27B | Gemma 3 27B, Qwen3.6 27B, Qwen3.8 27B |
| ~35B-A3B | Qwen3.5 35B-A3B, Qwen3.6 35B-A3B, MiniCPM-V 4.7 35B-A3B |

Large or sensitive files—including model weights and raw responses, API credentials, feature caches, and private human-rating forms—are excluded from the source tree.

## 1. Prepare the Environment

The runners locate the project root from the checkout. To download the dataset release from the terminal, install [GitHub CLI](https://github.com/cli/cli#installation) if needed and authenticate once:

~~~bash
gh auth login
gh auth status
export ANONYMOUS_REPOSITORY_URL=https://github.com/OWNER/REPOSITORY.git
export ANONYMOUS_REPOSITORY=OWNER/REPOSITORY
~~~

Run commands from the repository root; scripts discover the checkout automatically.

~~~bash
git clone "$ANONYMOUS_REPOSITORY_URL" multimodality
cd multimodality
python3 --version
nvidia-smi
ollama --version
ollama list
~~~

Use Python 3.10.12. Open-weight commands use the installed Ollama service and exact model tag; hosted commands use the provider bridge and API key described below. There is no single dependency lockfile. Do not install packages globally on a shared server.

## 2. Install the Dataset Locally

Download the project-format 1,416-record dataset ZIP from the repository release asset (URL: [OpenLAMM Ch3Ef dataset card](https://huggingface.co/datasets/openlamm/Ch3Ef)). It contains the records and images used by the evaluation scripts. Reviewer workbooks and private joint-label files are excluded.

From the repository root:

~~~bash
mkdir -p Datasets/Ch3EF/data
gh release download ch3ef-full1416-v1.0 --repo "$ANONYMOUS_REPOSITORY" --pattern ch3ef_full1416.zip --dir /tmp
printf '%s  %s\n' 'be2152bce3069d2878ae7050a67fd40c510a70b3d56d87aab8bcb544e560b33b' '/tmp/ch3ef_full1416.zip' | sha256sum -c
unzip /tmp/ch3ef_full1416.zip -d Datasets/Ch3EF/data
sha256sum Datasets/Ch3EF/data/ch3ef_full1416/sample.json
python3 -c "import json,collections; r=json.load(open('Datasets/Ch3EF/data/ch3ef_full1416/sample.json'))['records']; print(len(r), dict(collections.Counter(x['dimension'] for x in r)))"
~~~

Expected sample.json SHA-256: 69198c7a0d2afb4c885e86345d7f9ab0017626206f1fe34ca7633a74eb3235fe. Expected counts: 1,416 total; Helpful 625, Harmless 487, Honest 304.

Private files (rating workbooks, blind mappings, the 569-item human validation packet and the 600-item joint labels) are not included. The figure sections below state where each is expected.

## 3. Generate Baseline Answers

All baseline generation is zero-shot: the selected model receives the image, question and options. Open-weight runs used temperature 0 and no generation seed. Hosted settings differ: Gemini used temperature 0; OpenAI Responses and Anthropic Messages used provider defaults where the request omitted temperature. The runnable commands and metric settings are specified in this README and in the launch scripts.

### Open-weight example

Start the local Ollama service with the model tag installed, then run from the repository root:

~~~bash
python3 Baselines/Open/qwen2.5-vl-7b/run.py --model qwen2.5vl:7b --api http://127.0.0.1:11435/api/chat
~~~

For another Ollama tag, use the shared runner and pass its exact tag:

~~~bash
python3 Baselines/Open/scripts/run_ch3ef_full1416_model.py --slug MODEL_SLUG --model OLLAMA_TAG --api http://127.0.0.1:11435/api/chat
~~~

Replace MODEL_SLUG and OLLAMA_TAG with the model folder name and exact output of ollama list. This runner writes responses under evaluations/MODEL_SLUG/ch3ef-full1416/.

The open-weight scorers read responses from Baselines/Open/MODEL_SLUG/runs/ch3ef-full1416/. Copy the response file into that location before scoring:

~~~bash
mkdir -p Baselines/Open/MODEL_SLUG/runs/ch3ef-full1416
cp evaluations/MODEL_SLUG/ch3ef-full1416/responses.jsonl Baselines/Open/MODEL_SLUG/runs/ch3ef-full1416/
~~~

For the Qwen anchor, first generate the shared GPT-4o WR/TI baseline:

~~~bash
python3 Baselines/Judge/scripts/evaluate_wr_ti_gpt4o_full1416.py
~~~

Copy the resulting Qwen responses and GPT-4o journal from evaluations/qwen2.5-vl-7b/ch3ef-full1416/ into Baselines/Open/qwen2.5-vl-7b/runs/ch3ef-full1416/. Then score an open-weight model:

~~~bash
python3 Baselines/Judge/scripts/score_open_expansion_ss.py --slug MODEL_SLUG
python3 Baselines/Judge/scripts/score_open_expansion_wr_ti.py --slug MODEL_SLUG
~~~

SS needs the local Beaver-Dam checkpoint. WR/TI uses the OpenAI key and makes paid GPT-4o calls. Do not score until the response files and Qwen anchor journal are in the expected folders.

### Hosted-Model Example

Store each provider key outside the repository. The helper prompts for the key without placing it in shell history:

~~~bash
bash Baselines/Judge/scripts/store_provider_key_securely.sh openai
bash Baselines/Judge/scripts/store_provider_key_securely.sh google
bash Baselines/Judge/scripts/store_provider_key_securely.sh anthropic
~~~

Key files are saved under ~/.config/multimodality/ with mode 600. To run one hosted model, start its local bridge, then run the model launcher. Example for Gemini 3.8 Flash:

~~~bash
mkdir -p logs
(
  export PROVIDER=google MODEL=gemini-3.8-flash SLUG=gemini-3.8-flash PORT=11436
  export MAX_OUTPUT_TOKENS=1024 COST_INPUT_PER_MILLION=0.75 COST_OUTPUT_PER_MILLION=3.75
  export MAX_PROVIDER_SPEND_USD=3.50
  python3 Baselines/Closed/scripts/provider_compat_proxy.py >logs/gemini-3.8-flash-proxy.log 2>&1 &
  PROXY_PID=$!
  trap 'kill "$PROXY_PID" 2>/dev/null || true' EXIT
  sleep 2
  python3 Baselines/Closed/gemini-3.8-flash/run.py --model gemini-3.8-flash --api http://127.0.0.1:11436/api/chat
)
~~~

For another model, change provider, model ID, slug and port according to its launcher. The local cap is an estimate; set a provider billing limit too. Do not run two jobs on the same port or output folder.

Closed-model launchers write into Baselines/Closed/MODEL_SLUG/runs/ch3ef-full1416/. After a completed generation, score it:

~~~bash
python3 Baselines/Judge/scripts/evaluate_ss_beaver_dam_model.py --slug MODEL_SLUG
python3 Baselines/Judge/scripts/evaluate_wr_ti_gpt4o_model.py --slug MODEL_SLUG
~~~

## 4. Metric Settings

Use these definitions and preserve each metric's denominator:

- **WR, Helpful:** GPT-4o wins divided by valid pairwise judgments; ties are reported separately and remain non-wins. Candidate order uses record ID + 90210. GPT-4o judging uses temperature 0, JSON and a 160-token cap.
- **SS, Harmless:** Beaver-Dam unsafe flags divided by valid evaluated responses. Its text input is truncated to 512 tokens; any of 14 category probabilities above 0.5 is unsafe. Beaver-Dam does not receive the image.
- **TI, Honest:** joint rate of answers that GPT-4o judges both truthful and informative (primary). The product of the truthful and informative rates is reported as TI prod. Judging uses temperature 0, JSON and a 160-token cap.

## 5. Train and Evaluate OrchCircuit

It needs all 24 model responses and saved WR/TI and SS labels, the dataset and the pinned Qwen2-VL-7B-Instruct feature encoder. The selector uses five grouped folds, seed 20261111, and groups duplicate query text or image content so linked records stay in one fold. Feature extraction writes a large local cache and makes no API calls.

Run one feature shard per GPU in separate terminals:

~~~bash
cd Method/OrchCircuit/final
python3 -u extract_features.py --gpu 0 --shards 2
~~~

~~~bash
cd Method/OrchCircuit/final
python3 -u extract_features.py --gpu 1 --shards 2
~~~

After both finish:

~~~bash
cd Method/OrchCircuit/final
python3 -u run_selector_cv.py
~~~

## 6. Reproduce Figure 1 (Study Design)

Figure 1 is a TikZ diagram and uses no data. Its source is `Method/OrchCircuit/final/figures/fig1.tex`. It compiles inside the manuscript with `tikz`, the libraries `arrows.meta, positioning, fit, backgrounds, calc, shapes.arrows, decorations.pathmorphing` and the TeX Live handwriting font `augie` (T1 encoding).

## 7. Reproduce Figure 2 (Judges vs Human Raters)

Panels a–d need the blinded three-rater development packet (569 items), which is private. Place it at `Datasets/Ch3EF/data/ch3ef_full1416/human_validation_figure3a_d_20261112_PRIVATE/` and keep the identity key in a separate private directory.

~~~bash
FINAL=Method/OrchCircuit/final

# (a, b) rater agreement (Fleiss kappa) and original judges vs the rater majority (Cohen's kappa)
python3 $FINAL/shared/human_rating_pipeline/scripts/analyze_figure3a_d_human_validation.py \
  --repo-root . --private-root /secure/path/identity-key \
  --ratings-dir /secure/path/ratings --output-dir $FINAL/shared/human_rating_pipeline/human_validation_20261113

# (b, c, d) frontier judges on the same items (paid API calls, capped)
python3 $FINAL/review/frontier_judges_vs_humans.py --cap-usd 10
python3 $FINAL/review/inter_judge_agreement.py

# (e, f) agreement by image count and image size; binarization sensitivity reported in the text
python3 $FINAL/review/fig4_mechanism_strata.py
python3 $FINAL/review/binarization_sensitivity.py

# (g, h) position effect and order-balanced WR (saved labels only)
python3 $FINAL/review/group1_fills.py
python3 $FINAL/review/resolve_o1_o5.py
~~~

Outputs are written to `$FINAL/fig2_judge_validity/` (`a_rater_agreement` to `d_ti_judge_vs_human`, `f_frontier_judges`) and `$FINAL/review/group1_fills.json`.

## 8. Reproduce Figure 3 (Image Access)

Panels a–c re-judge 80 items per criterion with the image present and withheld (paid API calls, capped). Panel d compares the shifts of OrchCircuit and Qwen2.5-VL 7B.

~~~bash
python3 Baselines/Judge/scripts/run_judge_image_access_audit.py --n-per-dimension 80 --cap-usd 32
python3 Baselines/Judge/scripts/add_orchcircuit_image_access.py
python3 Baselines/Judge/scripts/analyze_judge_image_access_audit.py

python3 Method/OrchCircuit/final/fig3_image_access/d_orch_vs_qwen_did/nonhuman_paired/analyze_paired_shift.py
python3 Method/OrchCircuit/final/fig3_image_access/d_orch_vs_qwen_did/nonhuman_paired/analyze_paired_shift_tie_inclusive.py
~~~

Outputs are written to `Baselines/Judge/image-access-audit/` and `Method/OrchCircuit/final/fig3_image_access/`.

## 9. Reproduce Figure 4 (Selector on the Judges' Scores)

Panels a–c and e–h come from the grouped five-fold run of Section 5 (`runs/fullset_group5fold/results.json` and `oof_selected_candidates.jsonl`). Panel d (simple routers) and the Bonferroni-adjusted comparisons in panels f and g come from:

~~~bash
python3 Method/OrchCircuit/final/review/review_response_analyses.py
~~~

This uses saved labels only and writes `Method/OrchCircuit/final/review/review_response_analyses.json`.

## 10. Reproduce Figure 5 (External Items and the Answer Key)

The external items are HumaniBench task3 multiple-choice questions about age, ethnicity, gender, occupation and sport. Obtain HumaniBench from its authors. The 200 rated items are read from `Datasets/HumaniBench/confirmatory_figure3ef_PRIVATE/sample_200_task3/` and the 300 held-out items from `Datasets/HumaniBench/confirmatory_v2_PRIVATE/`.

Generate answers, encode them and apply the final selector:

~~~bash
S=Method/OrchCircuit/final/shared/human_rating_pipeline/scripts
R=Method/OrchCircuit/final/review

# 200 rated items
python3 $S/run_humanibench_candidates.py --budget-usd 12        # all 24 models; hosted models make paid calls
python3 $S/extract_humanibench_features.py --gpu 0
python3 $S/fit_final_selector.py
python3 $S/apply_humanibench_selector.py

# 300 held-out items
python3 $R/o3_freeze_confirmatory_sample.py
python3 $R/o3_run_candidates_v2.py
python3 $R/o3_recover_claude_oversize.py                         # one image above a provider's size limit
python3 $R/o3_extract_features_v2.py
python3 $R/o3_apply_selector_v2.py
~~~

Panels a–d (human ratings) need the three returned rating workbooks in `Method/OrchCircuit/final/fig5_human_transfer/abc_pairwise_human/private/ratings_received/` and the blind mapping in `.../private/reviewer_forms/`. These are private. The aggregate summary of the superseded first rating round is provided in `fig5_human_transfer/c_superseded_round/`.

~~~bash
python3 $S/analyze_human_ratings_3e_f_PRIVATE.py
~~~

Panels e–h (answer key):

~~~bash
python3 $R/fig5_answer_key_audit.py              # (e, f) raters vs the answer key
python3 $R/fig5_judges_vs_key.py --cap-usd 10    # (e, f) judges vs the answer key (paid API calls, capped)
python3 $R/fig5_like_for_like.py                 # (e, f, g) identical answers for raters and judges
python3 $R/goodhart_vs_shift.py                  # judge verdicts vs accuracy (text)
python3 $R/objective_key_accuracy.py             # (h) accuracy on the 500 external items
~~~

Outputs are written to `Method/OrchCircuit/final/fig5_human_transfer/`.

## 11. Reproduce Figure 6 (Robustness)

~~~bash
F=Method/OrchCircuit/final/fig6_robustness

# (a-c) ablations and (d) seeds, matched initialization, three seeds x four variants (needs the Section 5 feature cache)
for seed in 20261111 20261112 20261113; do
  for variant in full no_set_attention no_candidate_identity mean_token_pool; do
    python3 $F/a_ablations/run_selector_ablation.py --variant "$variant" --training-seed "$seed" --panel a --init-scheme matched
  done
done
python3 $F/a_ablations/analyze_matched_ablation.py
python3 $F/b_seed_repeatability/analyze_matched_seed_repeatability.py

# (e, f) Ch3EF source categories
python3 $F/scripts/d_normalized_analyze.py

# (g) 600-item development set scored on all three criteria (needs the private joint labels)
python3 $F/scripts/e_analyze_joint_hhh.py

# (h) tipping points
python3 Method/OrchCircuit/final/review/resolve_o1_o5.py
~~~

Panel g reads the private joint labels from `Datasets/Ch3EF/data/ch3ef_full1416/joint_hhh_labels_600_PRIVATE/`. Existing run folders are never overwritten; move a previous run aside before re-running.

## 12. Render All Figure Panels

The panels of Figures 2–6 are drawn as `pgfplots` code from the saved outputs above. From `Method/OrchCircuit/final`:

~~~bash
python3 figures/collect_figure_data.py figdata
python3 figures/make_figures.py figdata figures/rendered
~~~

This writes `fig2.tex` to `fig6.tex`, which are identical to `figures/fig2.tex` to `figures/fig6.tex` used in the paper. `figures/figure_tables.md` holds the numeric tables that the renderer reads for Figures 3a, 3b, 5a–c, 6a and 6d.
