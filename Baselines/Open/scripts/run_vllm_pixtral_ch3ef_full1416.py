#!/usr/bin/env python3
"""Run Pixtral through vLLM's OpenAI-compatible API on the fixed Ch3Ef sample."""
import argparse, base64, datetime as dt, json, mimetypes, string, sys, time, urllib.error, urllib.request, importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'Datasets/Ch3EF/data/ch3ef_full1416'
SAMPLE=DATA/'sample.json'
PROMPT=ROOT/'Datasets/Ch3EF/ch3ef-evaluation-prompt.md'
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_ollama_ch3ef_full1416 import read_jsonl, valid, normalize_choice_id

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--slug',default='pixtral-12b')
    ap.add_argument('--model',default='pixtral-12b-2409')
    ap.add_argument('--api',default='http://127.0.0.1:8001/v1/chat/completions')
    ap.add_argument('--smoke-only',action='store_true')
    args=ap.parse_args()
    outdir=ROOT/'Baselines/Open'/args.slug/'runs/ch3ef-full1416'
    out=outdir/('smoke_responses.jsonl' if args.smoke_only else 'responses.jsonl')
    records=json.loads(SAMPLE.read_text())['records']
    if len(records)!=1416: raise SystemExit(f'Expected 1416 records, got {len(records)}')
    by_key={(r['dimension'],r['id']):r for r in records}
    done=set()
    if not args.smoke_only:
        for row in read_jsonl(out):
            rec=by_key.get((row.get('dimension'),row.get('record_id')))
            if rec and valid(row,rec): done.add((rec['dimension'],rec['id']))
    prompt_doc=PROMPT.read_text()
    template=prompt_doc.split('## Prompt\n',1)[1].split('\n## Run settings',1)[0].strip()
    request=urllib.request.Request(args.api.rsplit('/chat/completions',1)[0]+'/models')
    with urllib.request.urlopen(request,timeout=20) as res: model_info=json.loads(res.read().decode())
    outdir.mkdir(parents=True,exist_ok=True)
    start=dt.datetime.now(dt.timezone.utc); attempted=0
    with out.open('w' if args.smoke_only else 'a',encoding='utf-8') as stream:
        for index,record in enumerate(records,1):
            key=(record['dimension'],record['id'])
            if key in done: continue
            labels=list(string.ascii_uppercase[:len(record['options'])])
            choices='\n'.join(f'{label}. {option}' for label,option in zip(labels,record['options']))
            prompt=template.replace('{query}',record['query']).replace('{options}',choices)
            content=[]
            for ref in record['image']:
                path=DATA/ref; mime=mimetypes.guess_type(path.name)[0] or 'image/jpeg'
                encoded=base64.b64encode(path.read_bytes()).decode('ascii')
                content.append({'type':'image_url','image_url':{'url':f'data:{mime};base64,{encoded}'}})
            content.append({'type':'text','text':prompt})
            payload={'model':args.model,'messages':[{'role':'user','content':content}],
                     'temperature':0,'max_tokens':512,'response_format':{'type':'json_object'}}
            req=urllib.request.Request(args.api,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
            began=time.monotonic(); attempted+=1
            try:
                with urllib.request.urlopen(req,timeout=900) as res: body=json.loads(res.read().decode())
                raw=body['choices'][0]['message'].get('content','')
                try: parsed=json.loads(raw); parse_error=None
                except (json.JSONDecodeError,TypeError) as exc: parsed=None; parse_error=str(exc)
                normalization=normalize_choice_id(parsed,record) if isinstance(parsed,dict) else None
                row={'record_id':record['id'],'sample_order':record['sample_order'],'dimension':record['dimension'],
                     'model':args.model,'choice_labels':labels,'image_files':[Path(ref).name for ref in record['image']],
                     'response':parsed,'choice_id_normalization':normalization,'raw_response':raw,'parse_error':parse_error,
                     'finish_reason':body['choices'][0].get('finish_reason'),'prompt_tokens':body.get('usage',{}).get('prompt_tokens'),
                     'completion_tokens':body.get('usage',{}).get('completion_tokens'),'request_seconds':round(time.monotonic()-began,3),'error':None}
            except (urllib.error.URLError,urllib.error.HTTPError,TimeoutError,json.JSONDecodeError,KeyError) as exc:
                row={'record_id':record['id'],'sample_order':record['sample_order'],'dimension':record['dimension'],
                     'model':args.model,'choice_labels':labels,'image_files':[Path(ref).name for ref in record['image']],
                     'response':None,'raw_response':None,'parse_error':None,'request_seconds':round(time.monotonic()-began,3),
                     'error':f'{type(exc).__name__}: {exc}'}
            stream.write(json.dumps(row,ensure_ascii=False)+'\n'); stream.flush()
            if args.smoke_only: break
            if valid(row,record): done.add(key)
            if index%10==0 or index==len(records): print(f'{index}/1416 valid={len(done)} current={record["dimension"]}/{record["id"]} seconds={row["request_seconds"]}',flush=True)
    if args.smoke_only:
        print(json.dumps({'smoke_output':str(out.relative_to(ROOT)),'valid':bool(read_jsonl(out) and valid(read_jsonl(out)[0],records[0])),'model_info':model_info},indent=2),flush=True)
        return
    rows=read_jsonl(out)
    valid_keys={(r['dimension'],r['id']) for row in rows if (r:=by_key.get((row.get('dimension'),row.get('record_id')))) and valid(row,r)}
    metadata={'study':'Ch3Ef complete automated HHH evaluation','model':args.model,'model_slug':args.slug,
      'runtime':'user-level vLLM OpenAI-compatible API on SSH server','vllm_version':importlib.metadata.version('vllm'),
      'torch_version':importlib.metadata.version('torch'),'vllm_model_info':model_info,
      'endpoint':args.api,'temperature':0,'max_tokens':512,'response_format':'json_object',
      'vllm_settings':{'tokenizer_mode':'mistral','tensor_parallel_size':2,'max_model_len':16384,
        'gpu_memory_utilization':0.95,'limit_mm_per_prompt':{'image':4},
        'VLLM_USE_FLASHINFER_SAMPLER':0},
      'records':1416,'valid_responses':len(valid_keys),'requests_attempted_this_run':attempted,
      'started_utc':start.isoformat(),'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
      'sample_file':'Datasets/Ch3EF/data/ch3ef_full1416/sample.json','responses_file':str(out.relative_to(ROOT)),
      'choice_id_normalization_policy':'Same helper and policy as Ollama runs; raw_response preserved.'}
    (outdir/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)
    if len(valid_keys)!=1416: raise SystemExit(f'Only {len(valid_keys)}/1416 valid responses')
if __name__=='__main__': main()
