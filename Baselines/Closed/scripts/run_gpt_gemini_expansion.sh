#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"
mkdir -p logs

# Wait until the Claude expansion finishes using the shared bridge and judge GPU.
CLAUDE_RUN_PID=1307372
while kill -0 "$CLAUDE_RUN_PID" 2>/dev/null; do sleep 20; done

run_generation() {
  slug="$1"; provider="$2"; model="$3"; input_rate="$4"; output_rate="$5"; cap="$6"
  export PROVIDER="$provider" MODEL="$model" SLUG="$slug" PORT=11436
  export MAX_OUTPUT_TOKENS=1024 COST_INPUT_PER_MILLION="$input_rate"
  export COST_OUTPUT_PER_MILLION="$output_rate" MAX_PROVIDER_SPEND_USD="$cap"
  if [ "$provider" = openai ]; then export REASONING_EFFORT=low; else unset REASONING_EFFORT || true; fi
  python3 Baselines/Closed/scripts/provider_compat_proxy.py >"logs/$slug-proxy.log" 2>&1 &
  proxy_pid=$!
  cleanup() { kill "$proxy_pid" 2>/dev/null || true; wait "$proxy_pid" 2>/dev/null || true; }
  trap cleanup EXIT
  sleep 2
  for attempt in 1 2 3; do
    set +e
    python3 "Baselines/Closed/$slug/run.py" --model "$model" --api "http://127.0.0.1:11436/api/chat" >"logs/$slug-generation-pass-$attempt.log" 2>&1
    run_code=$?
    set -e
    valid="$(python3 -c 'import json,sys,os; p=sys.argv[1]; d=json.load(open(p)) if os.path.exists(p) else {}; print(d.get("valid_responses",0))' "Baselines/Closed/$slug/runs/ch3ef-full1416/run_metadata.json")"
    if [ "$valid" = 1416 ]; then echo "$slug complete: 1416/1416 valid, pass $attempt"; cleanup; trap - EXIT; return 0; fi
    echo "$slug pass $attempt: $valid/1416 valid (runner exit=$run_code)"
    [ "$attempt" -eq 3 ] || sleep 3
  done
  cleanup; trap - EXIT
  echo "$slug incomplete after three passes" >&2
  return 1
}

score_model() {
  slug="$1"
  python3 Baselines/Judge/scripts/evaluate_ss_beaver_dam_model.py --slug "$slug" >"logs/$slug-ss.log" 2>&1
  python3 Baselines/Judge/scripts/evaluate_wr_ti_gpt4o_model.py --slug "$slug" >"logs/$slug-wr-ti.log" 2>&1
}

if run_generation gemini-3.8-flash google gemini-3.8-flash 0.75 3.75 3.5; then score_model gemini-3.8-flash; else echo "gemini-3.8-flash incomplete; continuing with next model"; fi
if run_generation gemini-3.6-flash google gemini-3.6-flash 0.75 3.75 3.5; then score_model gemini-3.6-flash; else echo "gemini-3.6-flash incomplete; continuing with next model"; fi
if run_generation gpt-6.1-sol openai gpt-6.1-sol 2 10 5; then score_model gpt-6.1-sol; else echo "gpt-6.1-sol incomplete; continuing with next model"; fi
if run_generation gpt-6-luna openai gpt-6-luna 0.1 0.5 0.5; then score_model gpt-6-luna; else echo "gpt-6-luna incomplete; continuing with next model"; fi
