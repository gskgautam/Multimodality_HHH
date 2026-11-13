#!/usr/bin/env bash
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
mkdir -p logs
for pass in 1 2 3 4; do
 export PROVIDER=anthropic MODEL=claude-sonnet-5-5 SLUG=claude-sonnet-5-5 PORT=11439 MAX_OUTPUT_TOKENS=512
 python3 scripts/provider_compat_proxy.py > logs/claude-sonnet-5-5-proxy.log 2>&1 &
 proxy_pid=$!
 sleep 2
 python3 scripts/run_ch3ef_full1416_model.py --slug claude-sonnet-5-5 --model claude-sonnet-5-5 --api http://127.0.0.1:11439/api/chat > logs/claude-sonnet-5-5-pass-$pass.log 2>&1
 status=$?
 kill "$proxy_pid" 2>/dev/null || true
 count=$(python3 - <<'PY'
import json,pathlib
p=pathlib.Path('evaluations/claude-sonnet-5-5/ch3ef-full1416/responses.jsonl')
rs=[json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
ok={(r.get('dimension'),r.get('record_id')) for r in rs if not r.get('error') and not r.get('parse_error') and (r.get('response') or {}).get('choice_id') in r.get('choice_labels',[])}
print(len(ok))
PY
)
 echo "CLAUDE_PASS=$pass VALID=$count EXIT=$status"
 [ "$count" -ge 1416 ] && break
done
python3 - <<'PY'
import json,pathlib
R=(pathlib.Path.cwd());d=R/'evaluations/claude-sonnet-5-5/ch3ef-full1416';m=d/'run_metadata.json';x=json.loads(m.read_text())
u=d/'provider_usage.jsonl';rs=[json.loads(z) for z in u.read_text().splitlines() if z.strip()] if u.exists() else []
x.update({'runtime':'Hosted vision API through server-side compatibility bridge','provider':'anthropic','provider_model_id':'claude-sonnet-5-5','temperature':0,'max_output_tokens':512,'provider_usage_file':str(u.relative_to(R)),'estimated_api_cost_usd':round(sum((z.get('input_tokens',0)*2+z.get('output_tokens',0)*10)/1e6 for z in rs),6),'pricing_usd_per_million_tokens':{'input':2,'output':10},'prompt_file':'Datasets/Ch3EF/ch3ef-evaluation-prompt.md'})
m.write_text(json.dumps(x,indent=2)+'\n');print('CLAUDE_VALID',x.get('valid_responses'),'API_COST',x.get('estimated_api_cost_usd'))
PY
if grep -q '"valid_responses": 1416' evaluations/claude-sonnet-5-5/ch3ef-full1416/run_metadata.json; then
 python3 scripts/evaluate_ss_beaver_dam_model.py --slug claude-sonnet-5-5 > logs/claude-sonnet-5-5-ss.log 2>&1
 for pass in 1 2 3 4; do
  MODEL_SLUG=claude-sonnet-5-5 python3 scripts/evaluate_wr_ti_gpt4o_for_model.py > logs/claude-sonnet-5-5-wr-ti.log 2>&1 || true
  python3 - <<'PY'
import json,pathlib
p=pathlib.Path('evaluations/claude-sonnet-5-5/ch3ef-full1416/wr_ti_gpt4o_summary.json')
try:
 x=json.loads(p.read_text());print('JUDGE_N',x['wr']['n'],x['ti']['n'])
except Exception:print('JUDGE_N 0 0')
PY
 done
fi
