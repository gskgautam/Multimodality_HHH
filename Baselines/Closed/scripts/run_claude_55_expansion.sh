#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"
mkdir -p logs
run_generation() {
  slug="$1"; model="$2"; input_rate="$3"; output_rate="$4"; cap="$5"
  export PROVIDER=anthropic MODEL="$model" SLUG="$slug" PORT=11436
  export MAX_OUTPUT_TOKENS=256 COST_INPUT_PER_MILLION="$input_rate"
  export COST_OUTPUT_PER_MILLION="$output_rate" MAX_PROVIDER_SPEND_USD="$cap"
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
    valid="$(python3 -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)) if __import__("os").path.exists(p) else {}; print(d.get("valid_responses",0))' "Baselines/Closed/$slug/runs/ch3ef-full1416/run_metadata.json")"
    if [ "$valid" = 1416 ]; then echo "$slug generation complete: 1416/1416 valid; pass=$attempt"; cleanup; trap - EXIT; return 0; fi
    echo "$slug generation pass $attempt: $valid/1416 valid (runner exit=$run_code)"
    if [ "$attempt" -lt 3 ]; then sleep 3; fi
  done
  cleanup; trap - EXIT
  echo "$slug generation incomplete after three passes" >&2
  return 1
}
score_model() {
  slug="$1"
  python3 Baselines/Judge/scripts/evaluate_ss_beaver_dam_model.py --slug "$slug" >"logs/$slug-ss.log" 2>&1
  python3 Baselines/Judge/scripts/evaluate_wr_ti_gpt4o_model.py --slug "$slug" >"logs/$slug-wr-ti.log" 2>&1
  python3 - "$slug" <<'PY'
import json, pathlib, sys
root=(pathlib.Path.cwd()); slug=sys.argv[1]
d=root/'Baselines'/'Closed'/slug/'runs'/'ch3ef-full1416'
meta=json.loads((d/'run_metadata.json').read_text())
usage=[json.loads(x) for x in (d/'provider_usage.jsonl').read_text().splitlines() if x.strip()]
meta.update({'provider':'anthropic','provider_model_id':slug,
 'provider_usage_file':str((d/'provider_usage.jsonl').relative_to(root)),
 'estimated_api_cost_usd':round(sum(r.get('estimated_cost_usd',0) for r in usage),6),
 'pricing_usd_per_million_tokens':{'input':0.1 if slug.endswith('haiku-5-5') else 4.0,
 'output':0.5 if slug.endswith('haiku-5-5') else 20.0}})
(d/'run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
print(slug, 'generation_USD=', meta['estimated_api_cost_usd'])
print('SS:', json.loads((d/'ss_beaver_dam.json').read_text())['ss_percent'])
print('WR/TI:', json.loads((d/'wr_ti_gpt4o_summary.json').read_text()))
PY
}
run_generation claude-haiku-5-5 claude-haiku-5-5 0.1 0.5 1.0
score_model claude-haiku-5-5
run_generation claude-opus-5-5 claude-opus-5-5 4 20 15.0
score_model claude-opus-5-5
