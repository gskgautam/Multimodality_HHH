#!/usr/bin/env bash
set -u
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"
RUNNER="$ROOT/Baselines/Open/scripts/run_ollama_ch3ef_full1416.py"
QUEUE_LOG="$ROOT/Baselines/Open/ollama_expansion_queue.log"
cd "$ROOT" || exit 1
printf '%s queue started; waiting for Qwen3.5-2B full run\n' "$(date -Is)" >> "$QUEUE_LOG"
while pgrep -f '^python3 Baselines/Open/scripts/run_ollama_ch3ef_full1416.py --slug qwen3.5-2b' >/dev/null; do sleep 20; done
printf '%s Qwen3.5-2B process ended; starting Q4_K_M models\n' "$(date -Is)" >> "$QUEUE_LOG"
while IFS='|' read -r slug tag; do
  [ -n "$slug" ] || continue
  avail=$(df -PB1 "$ROOT" | awk 'NR==2 {print $4}')
  if [ "${avail:-0}" -lt 60000000000 ]; then
    printf '%s STOP disk_guard free_bytes=%s before=%s\n' "$(date -Is)" "$avail" "$slug" >> "$QUEUE_LOG"
    break
  fi
  outdir="$ROOT/Baselines/Open/$slug/runs/ch3ef-full1416"
  mkdir -p "$outdir"
  log="$outdir/generation.log"
  printf '%s pull start %s (%s) free_bytes=%s\n' "$(date -Is)" "$slug" "$tag" "$avail" >> "$QUEUE_LOG"
  if ! OLLAMA_HOST=http://127.0.0.1:11435 ollama pull "$tag" >> "$log" 2>&1; then
    printf '%s SKIP pull_failed %s (%s); see %s\n' "$(date -Is)" "$slug" "$tag" "$log" >> "$QUEUE_LOG"
    continue
  fi
  printf '%s full start %s\n' "$(date -Is)" "$slug" >> "$QUEUE_LOG"
  attempt=1
  while [ "$attempt" -le 3 ]; do
    if python3 "$RUNNER" --slug "$slug" --model "$tag" --api http://127.0.0.1:11435/api/chat >> "$log" 2>&1; then
      printf '%s COMPLETE %s\n' "$(date -Is)" "$slug" >> "$QUEUE_LOG"
      break
    fi
    printf '%s retry %s attempt=%s (incomplete responses)\n' "$(date -Is)" "$slug" "$attempt" >> "$QUEUE_LOG"
    attempt=$((attempt + 1))
  done
  if [ "$attempt" -gt 3 ]; then printf '%s INCOMPLETE %s after 3 passes\n' "$(date -Is)" "$slug" >> "$QUEUE_LOG"; fi
done <<'MODELS'
gemma3-12b|gemma3:12b
qwen3.6-27b|qwen3.6:27b-q4_K_M
gemma3-27b|gemma3:27b
qwen3.5-35b-a3b|qwen3.5:35b-a3b-q4_K_M
qwen3.6-35b-a3b|qwen3.6:35b-a3b-q4_K_M
MODELS
printf '%s queue finished\n' "$(date -Is)" >> "$QUEUE_LOG"
