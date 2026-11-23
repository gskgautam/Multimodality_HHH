#!/usr/bin/env python3
"""Extract frozen Qwen2-VL features for the private 200-case candidate pool."""
import argparse,ast,hashlib,json,time
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor,Qwen2VLForConditionalGeneration
ROOT=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()))
SAMPLE_DIR=ROOT/"Datasets/HumaniBench/confirmatory_figure3ef_PRIVATE/sample_200_task3"
INPUT=SAMPLE_DIR/"inference_prompts.json"
OUT=SAMPLE_DIR/"features_PRIVATE"
RESP=SAMPLE_DIR/"candidate_outputs_PRIVATE"
CACHE=Path.home()/".cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"
MODELS=["lfm3-vl-2b","qwen3.5-2b","gemma4-e2b","qwen2.5-vl-7b","qwen3-vl-8b","minicpm-v-4.5-8b","gemma3-12b","gemma-4-12b","pixtral-12b","gemma3-27b","qwen3.6-27b","qwen3.8-27b","qwen3.5-35b-a3b","qwen3.6-35b-a3b","minicpm-v-4.7-35b-a3b","gpt-6.1-terra","gpt-6.1-sol","gpt-6-luna","gemini-4-argon","gemini-3.8-flash","gemini-3.6-flash","claude-opus-5-5","claude-sonnet-5-5","claude-haiku-5-5"]
TOKENS,DIM=64,3584
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readjl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--gpu",type=int,required=True);ap.add_argument("--shards",type=int,default=2);a=ap.parse_args()
    if not 0<=a.gpu<a.shards:raise SystemExit("gpu index must be in shard range")
    records=json.loads(INPUT.read_text())["records"];models=[m for i,m in enumerate(MODELS) if i%a.shards==a.gpu]
    OUT.mkdir(mode=0o700,parents=True,exist_ok=True)
    manifest={"records":[r["external_case_id"] for r in records],"models":models,"gpu":a.gpu,"shards":a.shards,"tokens":TOKENS,"hidden_dim":DIM,"backbone":"Qwen/Qwen2-VL-7B-Instruct","revision":"eed13092ef92e448dd6875b2a00151bd3f7db0ac","input_manifest_sha256":sha(INPUT),"feature_format":"64-token adaptive average pool of final hidden state; image + question + candidate response; matching OrchCircuit training feature form"}
    mp=OUT/f"feature_shard{a.gpu}.json";fp=OUT/f"features_shard{a.gpu}.f16";dp=OUT/f"feature_shard{a.gpu}.done.json"
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise SystemExit("feature manifest mismatch")
    mp.write_text(json.dumps(manifest,indent=2)+"\n")
    shape=(len(records),len(models),TOKENS,DIM)
    if not fp.exists():
        with fp.open("wb") as f:f.truncate(int(np.prod(shape))*2)
    xmap=np.memmap(fp,dtype=np.float16,mode="r+",shape=shape)
    done=set(json.loads(dp.read_text()).get("completed_models",[])) if dp.exists() else set()
    responses={}
    for m in models:
        rows=readjl(RESP/(m+".jsonl"));responses[m]={r.get("external_case_id"):r for r in rows if r.get("candidate_text") and not r.get("error")}
        missing=[r["external_case_id"] for r in records if r["external_case_id"] not in responses[m]]
        if missing:raise SystemExit(f"{m} missing usable responses for {len(missing)} cases; first={missing[:5]}")
    torch.cuda.set_device(a.gpu);torch.cuda.reset_peak_memory_stats()
    proc=AutoProcessor.from_pretrained(CACHE,local_files_only=True);proc.image_processor.max_pixels=1280*28*28
    model=Qwen2VLForConditionalGeneration.from_pretrained(CACHE,torch_dtype=torch.bfloat16,local_files_only=True,device_map=f"cuda:{a.gpu}").eval()
    start=time.time();fallbacks=[]
    for mi,m in enumerate(models):
        if m in done:print(json.dumps({"skip":m}),flush=True);continue
        print(json.dumps({"start":m,"cases":len(records),"gpu":a.gpu}),flush=True)
        for j,r in enumerate(records):
            row=responses[m][r["external_case_id"]]
            if row.get("valid_choice"):
                answer=row["choice_text"];rationale=row.get("rationale") or ""
            else:
                answer=row["candidate_text"];rationale=""
            text=f"Question: {r['question']}\nSelected answer: {answer}\nModel rationale: {rationale}"
            image=Image.open(SAMPLE_DIR/r["image_file"]).convert("RGB")
            msg=[{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":text}]}]
            prompt=proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=False)
            batch=proc(text=[prompt],images=[image],return_tensors="pt").to(f"cuda:{a.gpu}")
            try:
                with torch.inference_mode():out=model.model(**batch,output_hidden_states=False,return_dict=True,use_cache=False)
            except torch.cuda.OutOfMemoryError:
                del batch;torch.cuda.empty_cache();old=proc.image_processor.max_pixels;proc.image_processor.max_pixels=512*28*28
                batch=proc(text=[prompt],images=[image],return_tensors="pt").to(f"cuda:{a.gpu}")
                with torch.inference_mode():out=model.model(**batch,output_hidden_states=False,return_dict=True,use_cache=False)
                proc.image_processor.max_pixels=old;fallbacks.append({"model":m,"case":r["external_case_id"],"max_pixels":512*28*28})
            h=out.last_hidden_state.float().transpose(1,2)
            pooled=F.adaptive_avg_pool1d(h,TOKENS).transpose(1,2)[0].cpu().numpy().astype(np.float16)
            xmap[j,mi]=pooled
            del batch,out,h,pooled,image
            if (j+1)%50==0:
                xmap.flush();print(json.dumps({"model":m,"completed":j+1,"total":len(records),"elapsed":round(time.time()-start,1)}),flush=True)
        xmap.flush();done.add(m)
        dp.write_text(json.dumps({"completed_models":[x for x in models if x in done],"elapsed_seconds":round(time.time()-start,1),"peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(a.gpu),"fallbacks":fallbacks},indent=2)+"\n")
        print(json.dumps({"done":m,"elapsed":round(time.time()-start,1)}),flush=True)
    print(json.dumps({"completed_models":[x for x in models if x in done],"elapsed_seconds":round(time.time()-start,1),"peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(a.gpu),"fallbacks":fallbacks},indent=2),flush=True)
if __name__=="__main__":main()

