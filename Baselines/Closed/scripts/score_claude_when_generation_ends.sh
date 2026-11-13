#!/usr/bin/env bash
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
while pgrep -f '^bash scripts/run_claude_corrected.sh$' >/dev/null; do sleep 20; done
python3 - <<'PY'
import json,pathlib
R=(pathlib.Path.cwd());d=R/'evaluations/claude-sonnet-5-5/ch3ef-full1416';p=d/'run_metadata.json'
if p.exists():
 x=json.loads(p.read_text());u=d/'provider_usage.jsonl';rs=[json.loads(z) for z in u.read_text().splitlines() if z.strip()] if u.exists() else []
 x.update({'runtime':'Hosted vision API through server-side compatibility bridge','provider':'anthropic','provider_model_id':'claude-sonnet-5-5','temperature':0,'max_output_tokens':512,'provider_usage_file':str(u.relative_to(R)),'estimated_api_cost_usd':round(sum((z.get('input_tokens',0)*2+z.get('output_tokens',0)*10)/1e6 for z in rs),6),'pricing_usd_per_million_tokens':{'input':2,'output':10},'prompt_file':'Datasets/Ch3EF/ch3ef-evaluation-prompt.md'})
 p.write_text(json.dumps(x,indent=2)+'\n')
PY
if [ ! -f evaluations/claude-sonnet-5-5/ch3ef-full1416/ss_beaver_dam.json ]; then
 python3 scripts/evaluate_ss_beaver_dam_partial.py --slug claude-sonnet-5-5 > logs/claude-sonnet-5-5-ss-partial.log 2>&1
fi
for pass in 1 2 3; do
 if [ ! -f evaluations/claude-sonnet-5-5/ch3ef-full1416/wr_ti_gpt4o_summary.json ]; then
  MODEL_SLUG=claude-sonnet-5-5 python3 scripts/evaluate_wr_ti_gpt4o_for_model.py > logs/claude-sonnet-5-5-wr-ti-partial.log 2>&1 || true
 fi
done
