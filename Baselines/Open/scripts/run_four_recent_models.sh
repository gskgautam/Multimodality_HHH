#!/usr/bin/env bash
set -eu
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
OLLAMA="$PWD/tools/ollama-v0.40.13/runtime/bin/ollama"
export OLLAMA_HOST=127.0.0.1:11435
SUITE_LOG="$PWD/tools/ollama-v0.40.13/four-model-suite.log"
: > "$SUITE_LOG"
for spec in \
  'lfm3-vl-2b|lfm3-vl:2b' \
  'gemma-4-12b|gemma4:12b-it-q4_K_M' \
  'qwen3.8-27b|qwen3.8:27b-q4_K_M' \
  'minicpm-v-4.7-35b-a3b|openbmb/minicpm-v4.7:35b-a3b'
do
  slug="${spec%%|*}"
  model="${spec#*|}"
  outdir="$PWD/evaluations/$slug/ch3ef-full1416"
  mkdir -p "$outdir"
  echo "$(date -u +%FT%TZ) GENERATION START slug=$slug model=$model" | tee -a "$SUITE_LOG"
  attempt=1
  while [ "$attempt" -le 3 ]; do
    if python3 scripts/run_ch3ef_full1416_model.py --slug "$slug" --model "$model" > "$outdir/generation.log" 2>&1; then
      break
    fi
    echo "$(date -u +%FT%TZ) RETRY $attempt slug=$slug" | tee -a "$SUITE_LOG"
    attempt=$((attempt + 1))
  done
  if [ "$attempt" -gt 3 ]; then
    echo "$(date -u +%FT%TZ) GENERATION FAILED slug=$slug after retries" | tee -a "$SUITE_LOG"
    exit 1
  fi
  echo "$(date -u +%FT%TZ) GENERATION FINISHED slug=$slug" | tee -a "$SUITE_LOG"
  "$OLLAMA" stop "$model" >/dev/null 2>&1 || true
done
for slug in lfm3-vl-2b gemma-4-12b qwen3.8-27b minicpm-v-4.7-35b-a3b
do
  outdir="$PWD/evaluations/$slug/ch3ef-full1416"
  echo "$(date -u +%FT%TZ) SS START slug=$slug" | tee -a "$SUITE_LOG"
  if ! python3 scripts/evaluate_ss_beaver_dam_model.py --slug "$slug" > "$outdir/ss_run.log" 2>&1; then
    echo "$(date -u +%FT%TZ) SS FAILED slug=$slug" | tee -a "$SUITE_LOG"
    exit 1
  fi
  echo "$(date -u +%FT%TZ) SS FINISHED slug=$slug" | tee -a "$SUITE_LOG"
done
echo "$(date -u +%FT%TZ) ALL LOCAL RUNS FINISHED" | tee -a "$SUITE_LOG"
