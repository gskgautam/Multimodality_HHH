#!/usr/bin/env bash
set -u
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"
RUNNER="$ROOT/Baselines/Open/scripts/run_ollama_ch3ef_full1416.py"
LOG="$ROOT/Baselines/Open/replacement_queue_v2.log"
cd "$ROOT" || exit 1
printf '%s revised replacement queue resumed with Qwen3-VL context cap\n' "$(date -Is)" >> "$LOG"
while IFS='|' read -r slug tag; do
  [ -n "$slug" ] || continue
  free=$(df -PB1 "$ROOT" | awk 'NR==2 {print $4}')
  if [ "${free:-0}" -lt 50000000000 ]; then printf '%s STOP disk_guard free_bytes=%s before=%s\n' "$(date -Is)" "$free" "$slug" >> "$LOG"; break; fi
  outdir="$ROOT/Baselines/Open/$slug/runs/ch3ef-full1416"; mkdir -p "$outdir"; runlog="$outdir/generation.log"
  printf '%s resumed %s (%s) free=%s\n' "$(date -Is)" "$slug" "$tag" "$free" >> "$LOG"
  if ! OLLAMA_HOST=http://127.0.0.1:11435 ollama pull "$tag" >> "$runlog" 2>&1; then printf '%s SKIP pull_failed %s\n' "$(date -Is)" "$slug" >> "$LOG"; continue; fi
  attempt=1
  while [ "$attempt" -le 3 ]; do
    if python3 "$RUNNER" --slug "$slug" --model "$tag" --api http://127.0.0.1:11435/api/chat >> "$runlog" 2>&1; then printf '%s COMPLETE %s\n' "$(date -Is)" "$slug" >> "$LOG"; break; fi
    printf '%s retry %s attempt=%s\n' "$(date -Is)" "$slug" "$attempt" >> "$LOG"; attempt=$((attempt+1))
  done
  if [ "$attempt" -gt 3 ]; then printf '%s INCOMPLETE %s\n' "$(date -Is)" "$slug" >> "$LOG"; fi
done <<'MODELS'
qwen3-vl-8b|qwen3-vl:8b-instruct-q4_K_M
minicpm-v-4.5-8b|minicpm-v4.5:8b
MODELS
printf '%s revised replacement queue finished\n' "$(date -Is)" >> "$LOG"
