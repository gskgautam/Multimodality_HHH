#!/usr/bin/env python3
"""O3 confirmatory v2: apply the frozen OrchCircuitAll24 checkpoint to the 300-case feature cache.
Generated from apply_humanibench_selector.py by review/make_o3_pipeline.py (sample path only)."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch

ROOT=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()))
SAMPLE=ROOT/"Datasets/HumaniBench/confirmatory_v2_PRIVATE"
FEATURES=SAMPLE/"features_PRIVATE"
RESP=SAMPLE/"candidate_outputs_PRIVATE"
CHECKPOINT=ROOT/"Datasets/Ch3EF/data/ch3ef_full1416/orchcircuit_final_fit_PRIVATE/orchcircuit_selector.pt"
OUT=SAMPLE/"selector_outputs_PRIVATE"
sys.path.insert(0,str(ROOT/"Method/OrchCircuit/final"))
import run_selector_cv as cv

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readjl(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def main():
    records=json.loads((SAMPLE/"inference_prompts.json").read_text())["records"]
    n=len(records);models=cv.MODELS
    c=torch.load(CHECKPOINT,map_location="cpu",weights_only=False)
    if c["candidate_models"]!=models:raise RuntimeError("checkpoint candidate order differs from method inventory")
    if sha(CHECKPOINT)!="feddea3a6186356c9a0d729b377f5804dc8b4482780e1d87f2a32db8b3233dee":raise RuntimeError("checkpoint hash changed")
    mp=[json.loads((FEATURES/f"feature_shard{i}.json").read_text()) for i in (0,1)]
    for i in (0,1):
        done=json.loads((FEATURES/f"feature_shard{i}.done.json").read_text())
        if len(done["completed_models"])!=len(mp[i]["models"]):raise RuntimeError("incomplete feature shard "+str(i))
        if mp[i]["input_manifest_sha256"]!=sha(SAMPLE/"inference_prompts.json"):raise RuntimeError("feature input hash mismatch")
    shapes=[(n,12,64,3584)]*2
    fs=[np.memmap(FEATURES/f"features_shard{i}.f16",dtype=np.float16,mode="r",shape=shapes[i]) for i in (0,1)]
    loc=[{m:j for j,m in enumerate(mp[i]["models"])} for i in (0,1)]
    d=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=cv.OrchCircuitAll24().to(d);model.load_state_dict(c["state_dict"]);model.eval()
    cal=c["config"]["calibration"]
    t=float(cal["wr_temperature"]);sa,sb=cal["ss"];ta,tb=cal["truth"];ia,ib=cal["info"]
    response_maps={}
    for slug in models:
        rows=readjl(RESP/(slug+".jsonl"));m={}
        for r in rows:
            if r.get("candidate_text") and not r.get("error"):m[r["external_case_id"]]=r
        if len(m)!=n:raise RuntimeError(f"{slug} has {len(m)}/{n} unique usable responses")
        response_maps[slug]=m
    OUT.mkdir(mode=0o700,parents=True,exist_ok=True)
    output=OUT/"selections.jsonl"
    if output.exists():raise SystemExit("refusing to overwrite existing selector output")
    bymodel={m:i for i,m in enumerate(models)}
    with torch.inference_mode(),output.open("w") as f:
        for ri,rec in enumerate(records):
            x=np.empty((24,64,3584),dtype=np.float16)
            for mi,m in enumerate(models):
                si=0 if m in loc[0] else 1
                x[mi]=fs[si][ri,loc[si][m]]
            pred=model(torch.from_numpy(x[None]).to(d,dtype=torch.float32))
            wr=torch.softmax(pred["wr"][0]/t,dim=-1)[:,0].cpu().numpy()
            rawss=pred["ss"][0].cpu().numpy();ps=1/(1+np.exp(-np.clip(sa*rawss+sb,-30,30)))
            tr=pred["truth"][0].cpu().numpy();pi=pred["info"][0].cpu().numpy()
            pt=1/(1+np.exp(-np.clip(ta*tr+tb,-30,30)))
            pinf=1/(1+np.exp(-np.clip(ia*pi+ib,-30,30)));tprod=pt*pinf
            ix={"wr":int(np.argmax(wr)),"ss":int(np.argmin(ps)),"ti":int(np.argmax(tprod))}
            row={"external_case_id":rec["external_case_id"],"selection":{"wr":{},"ss":{},"ti":{}}}
            for key,j in ix.items():
                slug=models[j];ans=response_maps[slug][rec["external_case_id"]]
                row["selection"][key]={"model_slug":slug,"candidate_text":ans["candidate_text"],"valid_choice":ans.get("valid_choice"),"predicted_wr_win":float(wr[j]),"predicted_unsafe":float(ps[j]),"predicted_truthful":float(pt[j]),"predicted_informative":float(pinf[j]),"predicted_ti_product":float(tprod[j])}
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    (OUT/"selection_manifest.json").write_text(json.dumps({"sample_manifest_sha256":sha(SAMPLE/"inference_prompts.json"),"checkpoint_sha256":sha(CHECKPOINT),"feature_manifests_sha256":[sha(FEATURES/f"feature_shard{i}.json") for i in (0,1)],"candidate_models":models,"n_records":n,"policies":{"wr":"maximum calibrated win probability","ss":"minimum calibrated unsafe probability","ti":"maximum calibrated truthfulness probability times informativeness probability"}},indent=2)+"\n")
    print(json.dumps({"status":"complete","n":n,"checkpoint_sha256":sha(CHECKPOINT),"output":str(output)},indent=2))
if __name__=="__main__":main()

