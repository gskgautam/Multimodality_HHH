#!/usr/bin/env bash
set -u
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"
RUNNER="$ROOT/Baselines/Open/scripts/run_hf_ch3ef_full1416.py"
QUEUE_LOG="$ROOT/Baselines/Open/hf_expansion_queue.log"
cd "$ROOT" || exit 1
printf '%s queue started; waiting for Ollama expansion queue\n' "$(date -Is)" >> "$QUEUE_LOG"
while pgrep -f 'run_ollama_expansion_queue.sh' >/dev/null; do sleep 30; done
printf '%s Ollama queue ended; starting Transformers-only additions\n' "$(date -Is)" >> "$QUEUE_LOG"

run_one() {
  slug="$1"; model="$2"
  avail=$(df -PB1 "$ROOT" | awk 'NR==2 {print $4}')
  if [ "${avail:-0}" -lt 60000000000 ]; then
    printf '%s STOP disk_guard free_bytes=%s before=%s\n' "$(date -Is)" "$avail" "$slug" >> "$QUEUE_LOG"
    return 99
  fi
  outdir="$ROOT/Baselines/Open/$slug/runs/ch3ef-full1416"
  mkdir -p "$outdir"
  log="$outdir/generation.log"
  printf '%s smoke start %s (%s) free_bytes=%s\n' "$(date -Is)" "$slug" "$model" "$avail" >> "$QUEUE_LOG"
  HF_HOME=${HOME}/.cache/huggingface python3 "$RUNNER" --slug "$slug" --model "$model" --trust-remote-code --limit 1 >> "$log" 2>&1
  smoke_rc=$?
  if [ "$smoke_rc" -ne 0 ] || ! python3 - "$outdir/responses.jsonl" <<'PY' >> "$QUEUE_LOG" 2>&1
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
valid=[r for r in rows if r.get('record_id') == 0 and r.get('dimension') == 'Helpful' and not r.get('error') and not r.get('parse_error') and isinstance((r.get('response') or {}).get('choice_id'),str) and r['response']['choice_id'] in ('A','B','C','D','E','F','G','H')]
if not valid: raise SystemExit('smoke produced no valid choice')
print('smoke valid:',valid[-1]['response']['choice_id'])
PY
  then
    printf '%s SKIP smoke_failed %s rc=%s; see %s\n' "$(date -Is)" "$slug" "$smoke_rc" "$log" >> "$QUEUE_LOG"
    return 0
  fi
  printf '%s full start %s\n' "$(date -Is)" "$slug" >> "$QUEUE_LOG"
  attempt=1
  while [ "$attempt" -le 3 ]; do
    if HF_HOME=${HOME}/.cache/huggingface python3 "$RUNNER" --slug "$slug" --model "$model" --trust-remote-code >> "$log" 2>&1; then
      printf '%s COMPLETE %s\n' "$(date -Is)" "$slug" >> "$QUEUE_LOG"
      return 0
    fi
    printf '%s retry %s attempt=%s (incomplete responses)\n' "$(date -Is)" "$slug" "$attempt" >> "$QUEUE_LOG"
    attempt=$((attempt + 1))
  done
  printf '%s INCOMPLETE %s after 3 passes\n' "$(date -Is)" "$slug" >> "$QUEUE_LOG"
  return 0
}

while IFS='|' read -r slug model; do
  [ -n "$slug" ] || continue
  run_one "$slug" "$model"
  rc=$?
  [ "$rc" -eq 99 ] && break
done <<'MODELS'
internvl3.5-2b|OpenGVLab/InternVL3_5-2B-Instruct
llava-onevision-2-8b|lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
molmo2-8b|allenai/Molmo2-8B
pixtral-12b|mistralai/Pixtral-12B-2409
MODELS
printf '%s queue finished\n' "$(date -Is)" >> "$QUEUE_LOG"
