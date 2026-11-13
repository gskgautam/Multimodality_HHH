#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"
mkdir -p logs
# Do not share the API bridge or scoring GPU with the four-model queue.
QUEUE_PID=1307712
while kill -0 "$QUEUE_PID" 2>/dev/null; do sleep 20; done
slug=claude-opus-5-5
model=claude-opus-5-5
export PROVIDER=anthropic MODEL="$model" SLUG="$slug" PORT=11436
export MAX_OUTPUT_TOKENS=256 COST_INPUT_PER_MILLION=4 COST_OUTPUT_PER_MILLION=20 MAX_PROVIDER_SPEND_USD=15
python3 Baselines/Closed/scripts/provider_compat_proxy.py >logs/$slug-fallback-proxy.log 2>&1 &
proxy_pid=$!
cleanup() { kill "$proxy_pid" 2>/dev/null || true; wait "$proxy_pid" 2>/dev/null || true; }
trap cleanup EXIT
sleep 2
complete=0
for attempt in 1 2 3; do
  set +e
  python3 "Baselines/Closed/$slug/run.py" --model "$model" --api "http://127.0.0.1:11436/api/chat" >"logs/$slug-fallback-pass-$attempt.log" 2>&1
  run_code=$?
  set -e
  valid="$(python3 -c 'import json,sys,os; p=sys.argv[1]; d=json.load(open(p)) if os.path.exists(p) else {}; print(d.get("valid_responses",0))' "Baselines/Closed/$slug/runs/ch3ef-full1416/run_metadata.json")"
  echo "$slug pass $attempt: $valid/1416 valid (runner exit=$run_code)"
  if [ "$valid" = 1416 ]; then complete=1; break; fi
done
cleanup
trap - EXIT
if [ "$complete" -ne 1 ]; then echo "$slug remains incomplete; metrics skipped" >&2; exit 1; fi
python3 Baselines/Judge/scripts/evaluate_ss_beaver_dam_model.py --slug "$slug" >logs/$slug-ss.log 2>&1
python3 Baselines/Judge/scripts/evaluate_wr_ti_gpt4o_model.py --slug "$slug" >logs/$slug-wr-ti.log 2>&1
python3 - <<'PY'
import json,pathlib
root=(pathlib.Path.cwd())
d=root/'Baselines/Closed/claude-opus-5-5/runs/ch3ef-full1416'
m=json.loads((d/'run_metadata.json').read_text())
u=[json.loads(x) for x in (d/'provider_usage.jsonl').read_text().splitlines() if x.strip()]
m.update({'provider':'anthropic','provider_model_id':'claude-opus-5-5',
 'estimated_api_cost_usd':round(sum(x.get('estimated_cost_usd',0) for x in u),6),
 'pricing_usd_per_million_tokens':{'input':4,'output':20}})
(d/'run_metadata.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
print('Opus generation cost estimate USD',m['estimated_api_cost_usd'])
PY
