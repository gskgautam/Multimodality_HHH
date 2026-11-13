#!/usr/bin/env python3
"""Extract frozen shared Qwen2-VL features for the 24-baseline common-record set."""
import argparse,json,time,ast
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor,Qwen2VLForConditionalGeneration

ROOT=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()))
DATA=ROOT/'Datasets/Ch3EF/data/ch3ef_full1416'
OUT=ROOT/'Method/OrchCircuit/final'
CACHE=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac'
MODELS=[
'lfm3-vl-2b','qwen3.5-2b','gemma4-e2b','qwen2.5-vl-7b','qwen3-vl-8b','minicpm-v-4.5-8b',
'gemma3-12b','gemma-4-12b','pixtral-12b','gemma3-27b','qwen3.6-27b','qwen3.8-27b',
'qwen3.5-35b-a3b','qwen3.6-35b-a3b','minicpm-v-4.7-35b-a3b',
'gpt-6.1-terra','gpt-6.1-sol','gpt-6-luna','gemini-4-argon','gemini-3.8-flash','gemini-3.6-flash',
'claude-opus-5-5','claude-sonnet-5-5','claude-haiku-5-5']
OPEN=set(MODELS[:15])
TOKENS,DIM=64,3584

def readjl(p):
    if not p.exists(): return []
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--shards',type=int,default=2)
    args=ap.parse_args()
    if args.gpu>=args.shards: raise SystemExit('gpu shard index must be less than --shards')
    root=OUT/'data_selected_answer';root.mkdir(parents=True,exist_ok=True)
    oracle=json.loads((ROOT/'Method/OrchCircuit/final/inputs/oracle_ceiling_24models.json').read_text())
    orders=sorted(set().union(*(set(oracle['metrics'][m]['common_sample_orders']) for m in ('WR','SS','TI'))))
    records=json.loads((DATA/'sample.json').read_text())['records']
    by_order={int(r['sample_order']):r for r in records}
    if not all(o in by_order for o in orders): raise SystemExit('manifest includes an unknown sample order')
    models=[m for i,m in enumerate(MODELS) if i%args.shards==args.gpu]
    mindex={m:i for i,m in enumerate(models)}
    responses={}
    for m in models:
        family='Open' if m in OPEN else 'Closed'
        p=ROOT/'Baselines'/family/m/'runs/ch3ef-full1416/responses.jsonl'
        responses[m]={int(x['sample_order']):x for x in readjl(p) if x.get('sample_order') is not None}
        missing=[o for o in orders if o not in responses[m] or not responses[m][o].get('response') or responses[m][o].get('error')]
        if missing: raise SystemExit(f'{m} missing/invalid candidate responses for {len(missing)} common records; first={missing[:5]}')
    images=[]
    for o in orders:
        r=by_order[o]
        if not r.get('image'): raise SystemExit(f'missing image list at sample_order={o}')
        p=(DATA/r['image'][0]).resolve()
        if not p.exists(): raise SystemExit(f'missing image file {p}')
        images.append(p)
    manifest={'sample_orders':orders,'models':models,'shard':args.gpu,'shards':args.shards,'tokens':TOKENS,'dim':DIM,
      'backbone':'Qwen/Qwen2-VL-7B-Instruct','revision':'eed13092ef92e448dd6875b2a00151bd3f7db0ac',
      'image_max_pixels':1280*28*28,'input':'frozen image + query + selected answer text + model rationale; matches the established OrchCircuit feature format'}
    mp=root/f'features_shard{args.gpu}.json'
    fp=root/f'features_shard{args.gpu}.f16'
    dp=root/f'features_shard{args.gpu}.done.json'
    if not mp.exists(): mp.write_text(json.dumps(manifest,indent=2)+'\n')
    elif json.loads(mp.read_text())!=manifest: raise SystemExit(f'manifest mismatch in {mp}')
    shape=(len(orders),len(models),TOKENS,DIM)
    if not fp.exists():
        with fp.open('wb') as f: f.truncate(int(np.prod(shape))*2)
    xmap=np.memmap(fp,dtype=np.float16,mode='r+',shape=shape)
    done=json.loads(dp.read_text()) if dp.exists() else {'completed_models':[]}
    completed=set(done['completed_models'])
    torch.manual_seed(20261109+args.gpu)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.cuda.reset_peak_memory_stats()
    proc=AutoProcessor.from_pretrained(CACHE,local_files_only=True)
    proc.image_processor.max_pixels=1280*28*28
    model=Qwen2VLForConditionalGeneration.from_pretrained(CACHE,torch_dtype=torch.bfloat16,local_files_only=True,device_map=f'cuda:{args.gpu}').eval()
    start=time.time()
    resolution_fallbacks=[]
    for m in models:
        if m in completed: print(json.dumps({'skip_completed':m}),flush=True); continue
        print(json.dumps({'start_model':m,'gpu':args.gpu,'records':len(orders)}),flush=True)
        for j,o in enumerate(orders):
            r=by_order[o];row=responses[m][o];response=row['response']
            if isinstance(response,str):
                try: response=json.loads(response)
                except Exception: response=ast.literal_eval(response)
            if not isinstance(response,dict): raise RuntimeError(f'non-structured response for {m}, sample_order={o}')
            choice=str(response.get('choice_id','')).strip().upper()
            labels=row.get('choice_labels') or [chr(65+k) for k in range(len(r['options']))]
            if choice not in labels or labels.index(choice)>=len(r['options']): raise RuntimeError(f'unmatched choice {choice} for {m}, sample_order={o}')
            selected=r['options'][labels.index(choice)]
            rationale=response.get('rationale','')
            image=Image.open(images[j]).convert('RGB')
            text=f"Question: {r['query']}\nSelected answer: {selected}\nModel rationale: {rationale}"
            msg=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':text}]}]
            prompt=proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=False)
            batch=proc(text=[prompt],images=[image],return_tensors='pt').to(f'cuda:{args.gpu}')
            try:
                with torch.inference_mode(): out=model(**batch,output_hidden_states=True,return_dict=True)
            except torch.cuda.OutOfMemoryError:
                del batch
                torch.cuda.empty_cache()
                original_max=proc.image_processor.max_pixels
                proc.image_processor.max_pixels=512*28*28
                batch=proc(text=[prompt],images=[image],return_tensors='pt').to(f'cuda:{args.gpu}')
                with torch.inference_mode(): out=model(**batch,output_hidden_states=True,return_dict=True)
                proc.image_processor.max_pixels=original_max
                resolution_fallbacks.append({'model':m,'sample_order':o,'max_pixels':512*28*28,'reason':'CUDA OOM at standard image cap'})
            h=out.hidden_states[-1].float().transpose(1,2)
            pooled=F.adaptive_avg_pool1d(h,TOKENS).transpose(1,2)[0].cpu().numpy().astype(np.float16)
            xmap[j,mindex[m]]=pooled
            del batch,out,h,pooled,image
            if (j+1)%100==0:
                xmap.flush();print(json.dumps({'model':m,'features':j+1,'total':len(orders),'elapsed_seconds':round(time.time()-start,1)}),flush=True)
        xmap.flush();completed.add(m)
        dp.write_text(json.dumps({'completed_models':[x for x in models if x in completed],'elapsed_seconds':time.time()-start},indent=2)+'\n')
        print(json.dumps({'model_done':m,'elapsed_seconds':round(time.time()-start,1)}),flush=True)
    final={**manifest,'completed_models':[m for m in models if m in completed],'elapsed_seconds':time.time()-start,
      'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(args.gpu) if torch.cuda.is_available() else None,'resolution_fallbacks':resolution_fallbacks,'errors':[]}
    dp.write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps(final,indent=2),flush=True)
if __name__=='__main__': main()
