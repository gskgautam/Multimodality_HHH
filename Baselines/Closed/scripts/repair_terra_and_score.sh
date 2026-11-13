#!/usr/bin/env bash
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
mkdir -p logs
export PROVIDER=openai MODEL=gpt-6.1-terra SLUG=gpt-6.1-terra PORT=11439 MAX_OUTPUT_TOKENS=1024
python3 scripts/provider_compat_proxy.py > logs/gpt-6.1-terra-repair-proxy.log 2>&1 &
proxy_pid=$!
sleep 2
python3 scripts/run_ch3ef_full1416_model.py --slug gpt-6.1-terra --model gpt-6.1-terra --api http://127.0.0.1:11439/api/chat > logs/gpt-6.1-terra-repair-final.log 2>&1
status=$?
kill "$proxy_pid" 2>/dev/null || true
if [ "$status" -ne 0 ]; then echo "repair incomplete $status"; exit "$status"; fi
python3 - <<'PY'
import json,pathlib
R=(pathlib.Path.cwd()); d=R/'evaluations/gpt-6.1-terra/ch3ef-full1416'; m=d/'run_metadata.json'; x=json.loads(m.read_text()); u=d/'provider_usage.jsonl'; rows=[json.loads(t) for t in u.read_text().splitlines() if t.strip()]
x.update({'runtime':'Hosted vision API through server-side compatibility bridge','provider':'openai','provider_model_id':'gpt-6.1-terra','temperature':'provider default (sampling temperature not supported)','max_output_tokens':1024,'provider_usage_file':str(u.relative_to(R)),'estimated_api_cost_usd':round(sum(q.get('estimated_cost_usd',0) for q in rows),6),'pricing_usd_per_million_tokens':{'input':1,'output':5},'prompt_file':'Datasets/Ch3EF/ch3ef-evaluation-prompt.md'})
m.write_text(json.dumps(x,indent=2)+'\n')
PY
python3 scripts/evaluate_ss_beaver_dam_model.py --slug gpt-6.1-terra > logs/gpt-6.1-terra-ss.log 2>&1
for pass in 1 2 3; do
 MODEL_SLUG=gpt-6.1-terra python3 scripts/evaluate_wr_ti_gpt4o_for_model.py > logs/gpt-6.1-terra-wr-ti.log 2>&1 || true
done
