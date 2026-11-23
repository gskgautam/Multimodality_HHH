#!/usr/bin/env python3
"""Grouped 5-fold joint HHH selector experiment on the private 600-record development set."""
import json,random,sys,time,hashlib
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
ROOT=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd())); EXP=ROOT/"Method/OrchCircuit/final"; DATA=ROOT/"Datasets/Ch3EF/data/ch3ef_full1416"
PRIVATE=DATA/"joint_hhh_labels_600_PRIVATE"; GPT=PRIVATE/"gpt4o_joint_labels_v2.jsonl"; SS=PRIVATE/"beaver_dam_ss_600x24_summary.json"
OUT=PRIVATE/"orchcircuit_joint_oof_results.json"; PRED=PRIVATE/"orchcircuit_joint_oof_predictions.jsonl"
DEVICE="cuda:0" if torch.cuda.is_available() else "cpu"; SEED=20261112; BATCH=8; EPOCHS=20; PATIENCE=4
sys.path.insert(0,str(EXP)); import run_selector_cv as cv

def readjl(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def json_default(x):
 if isinstance(x,np.generic): return x.item()
 raise TypeError(f"not JSON serializable: {type(x).__name__}")
def metric(y):
 n=len(y['wr']); wr=sum(a==0 for a in y["wr"]); unsafe=int(np.sum(y["ss"])); tr=int(np.sum(y["truth"])); inf=int(np.sum(y["info"]))
 return {"n":n,"wr_wins":wr,"wr_percent":100*wr/n,"ss_unsafe":unsafe,"ss_percent":100*unsafe/n,"ti_truthful":tr,"ti_informative":inf,"ti_product_percent":100*(tr/n)*(inf/n)}

def main():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)
 manifest=json.loads((PRIVATE/"sample_manifest.json").read_text()); orders=[int(o) for o in manifest["orders"]]; n=len(orders)
 if n!=600: raise RuntimeError(f"expected 600 manifest rows, got {n}")
 sample=json.loads((DATA/"sample.json").read_text())["records"]; by={int(r["sample_order"]):r for r in sample}
 sample_by_id={str(r["id"]):r for r in sample}; dim={o:by[o]["dimension"] for o in orders}
 mi={m:i for i,m in enumerate(cv.MODELS)}; oi={o:i for i,o in enumerate(orders)}
 # Reconstitute judge arrays in the fixed model order.
 wr=np.full((n,24),-1,dtype=np.int64); ss=np.full((n,24),-1,dtype=np.int64); truth=np.full((n,24),-1,dtype=np.int64); info=np.full((n,24),-1,dtype=np.int64)
 gpt={}
 for row in readjl(GPT):
  if row.get("error"): continue
  key=(int(row["sample_order"]),row["task"]);gpt[key]=row
 if any((o,t) not in gpt for o in orders for t in ("wr","ti")): raise RuntimeError("GPT-4o labels incomplete for manifest")
 for o in orders:
  for task in ("wr","ti"):
   row=gpt[(o,task)]
   for code,model in row["candidate_map"].items():
    lab=row["labels"][code]; j=mi[model]
    if task=="wr": wr[oi[o],j]={"win":0,"tie":1,"loss":2}[lab["preference"]]
    else: truth[oi[o],j]=int(lab["truthful"]);info[oi[o],j]=int(lab["informative"])
 if np.any(wr<0) or np.any(truth<0) or np.any(info<0): raise RuntimeError("missing WR/TI target")
 ss_json=json.loads(SS.read_text()); ss_by={(int(r["sample_order"]),r["model"]):r for r in ss_json["records"]}
 if len(ss_by)!=n*24: raise RuntimeError(f"expected {n*24} SS labels, got {len(ss_by)}")
 for o in orders:
  for m in cv.MODELS:
   if (o,m) not in ss_by: raise RuntimeError(f"missing SS for {o} {m}")
   ss[oi[o],mi[m]]=int(ss_by[(o,m)]["unsafe"])
 # Model features were extracted for the fixed full common-order set. Select this 600 without recomputing.
 _,feature_orders,_,_,_,_,_,_=cv.labels_and_orders(); eoi={o:i for i,o in enumerate(feature_orders)}
 featdir=EXP/"data_selected_answer"; manifests=[json.loads((featdir/f"features_shard{i}.json").read_text()) for i in (0,1)]
 dones=[json.loads((featdir/f"features_shard{i}.done.json").read_text()) for i in (0,1)]
 for s in (0,1):
  if set(dones[s]["completed_models"])!=set(manifests[s]["models"]): raise RuntimeError(f"incomplete feature shard {s}")
 f0=np.memmap(featdir/"features_shard0.f16",dtype=np.float16,mode="r",shape=(len(feature_orders),12,64,3584))
 f1=np.memmap(featdir/"features_shard1.f16",dtype=np.float16,mode="r",shape=(len(feature_orders),12,64,3584))
 m0={m:i for i,m in enumerate(manifests[0]["models"])};m1={m:i for i,m in enumerate(manifests[1]["models"])}
 def getx(batch):
  ix=np.asarray([eoi[o] for o in batch],dtype=np.int64); x=np.empty((len(ix),24,64,3584),dtype=np.float16)
  for j,m in enumerate(cv.MODELS): x[:,j]=f0[ix,m0[m]] if m in m0 else f1[ix,m1[m]]
  return torch.from_numpy(x).to(DEVICE,dtype=torch.float32)
 group_of,hashes=cv.make_groups(by,orders); foldmap,folds,_=cv.grouped_folds(orders,group_of,dim,SEED)
 if len(foldmap)!=n or len(set(group_of[o] for o in orders))<5: raise RuntimeError("invalid grouped folds")
 if len({group_of[o] for o in orders}) != len(set(group_of.values())): raise RuntimeError("bad groups")
 wr_oof=np.zeros((n,24,3),np.float32);ss_oof=np.zeros((n,24),np.float32);tr_oof=np.zeros((n,24),np.float32);info_oof=np.zeros((n,24),np.float32);selected={}
 started=time.time()
 for fold_id,test_orders in enumerate(folds):
  train_all=sorted(o for o in orders if foldmap[o]!=fold_id)
  tr_orders,val_orders=cv.validation_split(train_all,group_of,dim,SEED+100+fold_id)
  if {group_of[o] for o in test_orders}&{group_of[o] for o in tr_orders+val_orders}: raise RuntimeError("outer-fold group leakage")
  random.seed(SEED+fold_id);np.random.seed(SEED+fold_id);torch.manual_seed(SEED+fold_id)
  model=cv.OrchCircuitAll24().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
  best=float("inf");best_state=None;stale=0;best_ep=0
  def run_batches(pool,train):
   seq=list(pool)
   if train: random.shuffle(seq)
   losses=[]
   for start in range(0,len(seq),BATCH):
    batch=seq[start:start+BATCH]; ix=torch.as_tensor([oi[o] for o in batch],device=DEVICE)
    x=getx(batch); pred=model(x)
    ywr=torch.as_tensor(wr[ix.cpu().numpy()],device=DEVICE,dtype=torch.long)
    yss=torch.as_tensor(ss[ix.cpu().numpy()],device=DEVICE,dtype=torch.float32)
    ytr=torch.as_tensor(truth[ix.cpu().numpy()],device=DEVICE,dtype=torch.float32)
    yin=torch.as_tensor(info[ix.cpu().numpy()],device=DEVICE,dtype=torch.float32)
    loss=(F.cross_entropy(pred["wr"].reshape(-1,3),ywr.reshape(-1))+
          F.binary_cross_entropy_with_logits(pred["ss"],yss)+
          F.binary_cross_entropy_with_logits(pred["truth"],ytr)+
          F.binary_cross_entropy_with_logits(pred["info"],yin))/4
    if train:
     opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
    losses.append(float(loss.detach().cpu()))
   return float(np.mean(losses))
  log=[]
  for ep in range(1,EPOCHS+1):
   model.train();tl=run_batches(tr_orders,True);model.eval()
   with torch.inference_mode(): vl=run_batches(val_orders,False)
   log.append({"epoch":ep,"train_loss":tl,"validation_loss":vl})
   print(json.dumps({"fold":fold_id+1,"epoch":ep,"train_loss":tl,"validation_loss":vl}),flush=True)
   if vl<best-1e-5:
    best=vl;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0;best_ep=ep
   else:
    stale+=1
    if stale>=PATIENCE:break
  if best_state is None: raise RuntimeError("no checkpoint in fold")
  model.load_state_dict(best_state);model.eval()
  vwr=[];vss=[];vtr=[];vin=[]
  with torch.inference_mode():
   for start in range(0,len(val_orders),BATCH):
    batch=val_orders[start:start+BATCH]; z=model(getx(batch))
    vwr.extend(z["wr"].detach().cpu().numpy());vss.extend(z["ss"].detach().cpu().numpy());vtr.extend(z["truth"].detach().cpu().numpy());vin.extend(z["info"].detach().cpu().numpy())
  vt=np.asarray([wr[oi[o]] for o in val_orders]); vs=np.asarray([ss[oi[o]] for o in val_orders]); vtruth=np.asarray([truth[oi[o]] for o in val_orders]); vinfo=np.asarray([info[oi[o]] for o in val_orders])
  temp=cv.fit_temperature(np.asarray(vwr).reshape(-1,3),vt.reshape(-1))
  ab_s=cv.fit_binary(np.asarray(vss).reshape(-1),vs.reshape(-1));ab_t=cv.fit_binary(np.asarray(vtr).reshape(-1),vtruth.reshape(-1));ab_i=cv.fit_binary(np.asarray(vin).reshape(-1),vinfo.reshape(-1))
  model.eval()
  with torch.inference_mode():
   for start in range(0,len(test_orders),BATCH):
    batch=test_orders[start:start+BATCH]; z=model(getx(batch))
    a=z["wr"].detach().cpu().numpy()/temp
    b=z["ss"].detach().cpu().numpy()*ab_s[0]+ab_s[1]
    c=z["truth"].detach().cpu().numpy()*ab_t[0]+ab_t[1]
    d=z["info"].detach().cpu().numpy()*ab_i[0]+ab_i[1]
    for k,o in enumerate(batch):
     idx=oi[o];wr_oof[idx]=a[k];ss_oof[idx]=b[k];tr_oof[idx]=c[k];info_oof[idx]=d[k]
  # Constraint floors use training-only Qwen judge outcomes; no outer test labels influence arbitration.
  q=mi["qwen2.5-vl-7b"]; train_ix=np.asarray([oi[o] for o in tr_orders])
  unsafe_floor=float(ss[train_ix,q].mean());ti_floor=float(truth[train_ix,q].mean()*info[train_ix,q].mean())
  probs=np.exp(wr_oof[[oi[o] for o in test_orders]]-wr_oof[[oi[o] for o in test_orders]].max(axis=2,keepdims=True));probs/=probs.sum(axis=2,keepdims=True)
  ps=1/(1+np.exp(-np.clip(ss_oof[[oi[o] for o in test_orders]],-30,30)))
  pt=1/(1+np.exp(-np.clip(tr_oof[[oi[o] for o in test_orders]],-30,30)))
  pi=1/(1+np.exp(-np.clip(info_oof[[oi[o] for o in test_orders]],-30,30)))
  for k,o in enumerate(test_orders):
   safe=ps[k]<=unsafe_floor; truthful=pt[k]*pi[k]>=ti_floor; feasible=safe&truthful
   if np.any(feasible): cand=np.where(feasible)[0]; pick=int(cand[np.argmax(probs[k,cand,0])])
   else:
    violation=np.maximum(ps[k]-unsafe_floor,0)+np.maximum(ti_floor-pt[k]*pi[k],0)
    cand=np.where(violation==violation.min())[0];pick=int(cand[np.argmax(probs[k,cand,0])])
   selected[o]=pick
  print(json.dumps({"fold_complete":fold_id+1,"test_n":len(test_orders),"best_epoch":best_ep,"safety_floor":unsafe_floor,"ti_floor":ti_floor}),flush=True)
 # Score selector and all candidate baselines on identical human-free judge labels.
 sel={"wr":[],"ss":[],"truth":[],"info":[]}; baseline={m:{"wr":[],"ss":[],"truth":[],"info":[]} for m in cv.MODELS}
 prediction_rows=[]
 for o in orders:
  i=oi[o];pick=selected[o]; sel["wr"].append(wr[i,pick]);sel["ss"].append(ss[i,pick]);sel["truth"].append(truth[i,pick]);sel["info"].append(info[i,pick])
  for j,m in enumerate(cv.MODELS):
   for key,arr in (("wr",wr),("ss",ss),("truth",truth),("info",info)):baseline[m][key].append(arr[i,j])
  prediction_rows.append({"sample_order":o,"record_id":by[o]["id"],"chosen_model":cv.MODELS[pick],"labels":{"wr":["win","tie","loss"][wr[i,pick]],"unsafe":bool(ss[i,pick]),"truthful":bool(truth[i,pick]),"informative":bool(info[i,pick])},"fold":int(foldmap[o])})
 results={"design":"exploratory grouped 5-fold OOF; all 24 models evaluated on same 600 prompt-level joint judge labels; automated-label agreement, not HHH ground truth","n":n,"seed":SEED,"fold_sizes":[len(x) for x in folds],"unique_query_image_groups":len(set(group_of.values())),"group_leakage":False,"joint_label_provenance":{"wr_ti":"GPT-4o; image shown; WR pairwise vs Qwen2.5-VL 7B anchor; TI truthfulness/informativeness","ss":"PKU Beaver-Dam 7B text-only; unsafe threshold 0.5"},"arbitration":"maximize predicted WR win probability subject to calibrated predicted unsafe rate <= training-fold Qwen anchor unsafe rate and predicted TI product >= training-fold Qwen anchor TI product; if no feasible candidate, minimize total constraint violation then maximize WR probability","baseline_metrics":{m:metric({k:np.asarray(v) for k,v in arr.items()}) for m,arr in baseline.items()},"orchcircuit_oof":metric({k:np.asarray(v) for k,v in sel.items()}),"selected_model_counts":{m:sum(r["chosen_model"]==m for r in prediction_rows) for m in cv.MODELS},"elapsed_seconds":round(time.time()-started,1),"outputs":{"prediction_jsonl":str(PRED)}}
 # Group-bootstrap CIs for OOF metric differences versus each baseline.
 group_keys=sorted(set(group_of.values())); group_rows={g:[o for o in orders if group_of[o]==g] for g in group_keys}; rng=np.random.default_rng(SEED); reps=5000; comps={}
 base_arrays={m:{k:np.asarray(v) for k,v in d.items()} for m,d in baseline.items()}; sel_arrays={k:np.asarray(v) for k,v in sel.items()}
 for m in cv.MODELS:
  diffs={"wr":[],"ss":[],"ti":[]}
  for _ in range(reps):
   sampled=rng.choice(group_keys,size=len(group_keys),replace=True); draw=[o for g in sampled for o in group_rows[g]]; ix=np.asarray([oi[o] for o in draw])
   for metric_name,up in (("wr",True),("ss",False),("ti",True)):
    if metric_name=="wr":
     a=sel_arrays["wr"][ix];b=base_arrays[m]["wr"][ix];delta=100*(np.mean(a==0)-np.mean(b==0))
    elif metric_name=="ss":
     a=sel_arrays["ss"][ix];b=base_arrays[m]["ss"][ix];delta=100*(np.mean(b)-np.mean(a))
    else:delta=100*((np.mean(sel_arrays["truth"][ix])*np.mean(sel_arrays["info"][ix]))-(np.mean(base_arrays[m]["truth"][ix])*np.mean(base_arrays[m]["info"][ix])))
    diffs[metric_name].append(float(delta))
  point={"wr":100*(np.mean(sel_arrays["wr"]==0)-np.mean(base_arrays[m]["wr"]==0)),
         "ss":100*(np.mean(base_arrays[m]["ss"])-np.mean(sel_arrays["ss"])),
         "ti":100*((np.mean(sel_arrays["truth"])*np.mean(sel_arrays["info"]))-(np.mean(base_arrays[m]["truth"])*np.mean(base_arrays[m]["info"])))}
  comps[m]={k:{"delta_pp":float(point[k]),"ci95_pp":[float(np.quantile(v,.025)),float(np.quantile(v,.975))]} for k,v in diffs.items()}
 results["paired_group_bootstrap_vs_baselines"]=comps
 with OUT.open("w") as f:json.dump(results,f,indent=2,default=json_default)
 with PRED.open("w") as f:
  for r in prediction_rows:f.write(json.dumps(r)+"\n")
 OUT.chmod(0o600);PRED.chmod(0o600)
 print(json.dumps({"complete":True,"n":n,"orchcircuit":results["orchcircuit_oof"],"elapsed_seconds":results["elapsed_seconds"],"results_path":str(OUT)},indent=2,default=json_default),flush=True)
if __name__=="__main__":main()
