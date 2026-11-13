#!/usr/bin/env bash
set -u
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"
QUEUE_LOG="$ROOT/Baselines/Open/replacement_queue_v2.log"
RUN_LOG="$ROOT/Baselines/Open/pixtral_vllm_queue.log"
VLLM=${HOME}/.local/share/multimodality-pixtral-vllm-venv/bin/vllm
PYTHON=${HOME}/.local/share/multimodality-pixtral-vllm-venv/bin/python
RUNNER="$ROOT/Baselines/Open/scripts/run_vllm_pixtral_ch3ef_full1416.py"
cd "$ROOT" || exit 1
printf '%s waiting for Ollama replacements to finish\n' "$(date -Is)" >> "$RUN_LOG"
while ! grep -q 'revised replacement queue finished' "$QUEUE_LOG"; do sleep 30; done
free=$(df -PB1 "$ROOT" | awk 'NR==2 {print $4}')
if [ "${free:-0}" -lt 40000000000 ]; then
  printf '%s STOP disk_guard free_bytes=%s before Pixtral model download\n' "$(date -Is)" "$free" >> "$RUN_LOG"
  exit 1
fi
printf '%s starting Pixtral vLLM free=%s\n' "$(date -Is)" "$free" >> "$RUN_LOG"
CUDA_HOME=${HOME}/.local/share/multimodality-pixtral-vllm-venv/lib/python3.10/site-packages/nvidia/cu13
export CUDA_HOME
export PATH="$(dirname "$VLLM"):$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0
nohup "$VLLM" serve mistralai/Pixtral-12B-2409 --host 127.0.0.1 --port 8001 \
  --tokenizer-mode mistral --tensor-parallel-size 2 --max-model-len 16384 \
  --gpu-memory-utilization 0.95 --limit-mm-per-prompt '{"image":4}' \
  --served-model-name pixtral-12b-2409 > "$ROOT/Baselines/Open/pixtral_vllm_server.log" 2>&1 < /dev/null &
server_pid=$!
cleanup() { if kill -0 "$server_pid" 2>/dev/null; then kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT INT TERM
printf '%s vLLM server pid=%s\n' "$(date -Is)" "$server_pid" >> "$RUN_LOG"
ready=0
for i in $(seq 1 360); do
  if "$PYTHON" -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8001/v1/models",timeout=3)' >/dev/null 2>&1; then ready=1; break; fi
  sleep 10
done
if [ "$ready" -ne 1 ]; then
  printf '%s ERROR Pixtral vLLM did not become ready\n' "$(date -Is)" >> "$RUN_LOG"
  exit 1
fi
printf '%s Pixtral vLLM ready; running fixed dataset\n' "$(date -Is)" >> "$RUN_LOG"
"$PYTHON" "$RUNNER" --slug pixtral-12b --model pixtral-12b-2409 --api http://127.0.0.1:8001/v1/chat/completions >> "$RUN_LOG" 2>&1
rc=$?
printf '%s Pixtral runner exit=%s\n' "$(date -Is)" "$rc" >> "$RUN_LOG"
exit "$rc"
