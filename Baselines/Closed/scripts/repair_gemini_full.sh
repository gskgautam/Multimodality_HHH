#!/usr/bin/env bash
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
mkdir -p logs
for pass in 1 2 3 4 5 6; do
 export PROVIDER=google MODEL=gemini-4-argon SLUG=gemini-4-argon PORT=11438 MAX_OUTPUT_TOKENS=1024
 python3 scripts/provider_compat_proxy.py > logs/gemini-4-argon-repair-proxy.log 2>&1 &
 proxy_pid=$!
 sleep 2
 python3 scripts/run_ch3ef_full1416_model.py --slug gemini-4-argon --model gemini-4-argon --api http://127.0.0.1:11438/api/chat > logs/gemini-4-argon-repair-$pass.log 2>&1
 status=$?
 kill "$proxy_pid" 2>/dev/null || true
 count=$(python3 - <<'PY'
import json,pathlib
p=pathlib.Path('evaluations/gemini-4-argon/ch3ef-full1416/responses.jsonl')
rs=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
ok={(r.get('dimension'),r.get('record_id')) for r in rs if not r.get('error') and not r.get('parse_error') and (r.get('response') or {}).get('choice_id') in r.get('choice_labels',[])}
print(len(ok))
PY
)
 echo "GEMINI_REPAIR=$pass VALID=$count EXIT=$status"
 [ "$count" -ge 1416 ] && break
done
python3 - <<'PY'
import json,pathlib
R=(pathlib.Path.cwd());d=R/'evaluations/gemini-4-argon/ch3ef-full1416';m=d/'run_metadata.json';x=json.loads(m.read_text());u=d/'provider_usage.jsonl';rs=[json.loads(z) for z in u.read_text().splitlines() if z.strip()]
x.update({'runtime':'Hosted vision API through server-side compatibility bridge','provider':'google','provider_model_id':'gemini-4-argon','temperature':0,'max_output_tokens':1024,'provider_usage_file':str(u.relative_to(R)),'estimated_api_cost_usd':round(sum((z.get('input_tokens',0)*2+z.get('output_tokens',0)*10)/1e6 for z in rs),6),'pricing_usd_per_million_tokens':{'input':2,'output':10},'prompt_file':'Datasets/Ch3EF/ch3ef-evaluation-prompt.md'})
m.write_text(json.dumps(x,indent=2)+'\n');print('GEMINI_VALID',x.get('valid_responses'),'API_COST',x.get('estimated_api_cost_usd'))
PY
if grep -q '"valid_responses": 1416' evaluations/gemini-4-argon/ch3ef-full1416/run_metadata.json; then
 python3 scripts/evaluate_ss_beaver_dam_model.py --slug gemini-4-argon > logs/gemini-4-argon-ss.log 2>&1
 for pass in 1 2 3 4; do MODEL_SLUG=gemini-4-argon python3 scripts/evaluate_wr_ti_gpt4o_for_model.py > logs/gemini-4-argon-wr-ti.log 2>&1 || true; done
fi
