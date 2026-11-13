#!/usr/bin/env python3
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()));E=R/"Method/OrchCircuit/final";F=E/"figure4";O=F/"e";O.mkdir(parents=True,exist_ok=True)
MODELS=["lfm3-vl-2b","qwen3.5-2b","gemma4-e2b","qwen2.5-vl-7b","qwen3-vl-8b","minicpm-v-4.5-8b","gemma3-12b","gemma-4-12b","pixtral-12b","gemma3-27b","qwen3.6-27b","qwen3.8-27b","qwen3.5-35b-a3b","qwen3.6-35b-a3b","minicpm-v-4.7-35b-a3b","gpt-6.1-terra","gpt-6.1-sol","gpt-6-luna","gemini-4-argon","gemini-3.8-flash","gemini-3.6-flash","claude-opus-5-5","claude-sonnet-5-5","claude-haiku-5-5"]
def jl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
P=R/"Datasets/Ch3EF/data/ch3ef_full1416/joint_hhh_labels_600_PRIVATE"
man=json.loads((P/"sample_manifest.json").read_text());orders=list(map(int,man["orders"]))
assert len(orders)==600 and len(set(orders))==600
mi={m:i for i,m in enumerate(MODELS)};N=len(orders);ix={o:i for i,o in enumerate(orders)}
wr=np.full((N,24),-1,np.int8);ss=wr.copy();tr=wr.copy();inf=wr.copy()
labs=jl(P/"gpt4o_joint_labels_v2.jsonl")
lr={(int(x["sample_order"]),x["task"]):x for x in labs if not x.get("error") and isinstance(x.get("labels"),dict)}
for o in orders:
 for task in ("wr","ti"):
  x=lr[(o,task)]
  assert set(x["candidate_map"].values())==set(MODELS)
  for c,m in x["candidate_map"].items():
   v=x["labels"][c];j=mi[m]
   if task=="wr":wr[ix[o],j]=int(v["preference"]=="win")
   else:tr[ix[o],j]=int(bool(v["truthful"]));inf[ix[o],j]=int(bool(v["informative"]))
for x in jl(P/"beaver_dam_ss_600x24.jsonl"):
 o=int(x["sample_order"])
 if o in ix:ss[ix[o],mi[x["model"]]]=int(bool(x["unsafe"]))
assert all(np.all(x>=0) for x in (wr,ss,tr,inf))
pred={int(x["sample_order"]):x for x in jl(E/"runs/fullset_group5fold/oof_selected_candidates.jsonl")}
assert set(orders)<=set(pred)
chosen=np.array([mi[pred[o]["selected_candidate"]] for o in orders])
sys={m:{"wr":100*wr[:,j].mean(),"ss":100*ss[:,j].mean(),"truth":100*tr[:,j].mean(),"info":100*inf[:,j].mean(),"ti":100*tr[:,j].mean()*inf[:,j].mean()} for j,m in enumerate(MODELS)}
r=np.arange(N)
sys["OrchCircuit"]={"wr":100*wr[r,chosen].mean(),"ss":100*ss[r,chosen].mean(),"truth":100*tr[r,chosen].mean(),"info":100*inf[r,chosen].mean(),"ti":100*tr[r,chosen].mean()*inf[r,chosen].mean()}
front=[]
for a,va in sys.items():
 dominated=False
 for b,vb in sys.items():
  if a==b:continue
  if vb["wr"]>=va["wr"] and vb["ss"]<=va["ss"] and vb["ti"]>=va["ti"] and (vb["wr"]>va["wr"] or vb["ss"]<va["ss"] or vb["ti"]>va["ti"]):dominated=True;break
 if not dominated:front.append(a)
base={"WR":"gemini-4-argon","SS":"gemma3-12b","TI":"gemini-4-argon"}
folds=json.loads((E/"runs/fullset_group5fold/folds.json").read_text())["group_ids"]
groups=np.array([str(folds[str(o)]) for o in orders]);gkeys=np.unique(groups);grows={g:np.where(groups==g)[0] for g in gkeys}
rng=np.random.default_rng(20261113);boot={m:[] for m in base};B=10000
for _ in range(B):
 gs=rng.choice(gkeys,len(gkeys),replace=True);ids=np.concatenate([grows[g] for g in gs])
 for m,b in base.items():
  j=mi[b]
  if m=="WR": d=100*(wr[ids,chosen[ids]].mean()-wr[ids,j].mean())
  elif m=="SS": d=100*(ss[ids,j].mean()-ss[ids,chosen[ids]].mean())
  else:d=100*(tr[ids,chosen[ids]].mean()*inf[ids,chosen[ids]].mean()-tr[ids,j].mean()*inf[ids,j].mean())
  boot[m].append(float(d))
rows=[{"system":m,"n":N,"wr_win_pct":v["wr"],"ss_unsafe_pct":v["ss"],"ti_truthful_pct":v["truth"],"ti_informative_pct":v["info"],"ti_product_pct":v["ti"],"pareto_nondominated":m in front} for m,v in sys.items()]
rows.sort(key=lambda x:(x["system"]!="OrchCircuit",x["system"]))
with open(O/"joint_hhh_same_prompt.csv","w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
deltas={}
for m,b in base.items():
 j=mi[b];v=sys["OrchCircuit"];q=sys[b]
 point=(v["wr"]-q["wr"]) if m=="WR" else (q["ss"]-v["ss"] if m=="SS" else v["ti"]-q["ti"])
 deltas[m]={"fixed_comparator":b,"favorable_delta_pp":point,"group_bootstrap_95":[float(np.quantile(boot[m],.025)),float(np.quantile(boot[m],.975))]}
res={"status":"exploratory same-prompt re-score; this 600-prompt set was used in prior OrchCircuit development, so not confirmatory","n":N,"unique_prompt_image_groups":len(gkeys),"selection":"V4 fullset grouped-OOF selected candidate for each record, evaluated on all three joint automated labels","label_sources":{"WR_TI":"GPT-4o joint-development labels; WR is win vs Qwen anchor","SS":"Beaver-Dam 7B text-only safety labels"},"orchcircuit":sys["OrchCircuit"],"fixed_comparators":deltas,"pareto_nondominated_systems":front,"no_composite_score":True}
(O/"results.json").write_text(json.dumps(res,indent=2)+"\n")
md=["# Figure 4e — Same-prompt HHH trade-off","","Exploratory re-score of the existing 600-prompt development cohort. OrchCircuit's saved grouped-OOF selected answer for each prompt was scored on all three available joint automated outcomes. This makes the metric vector use the same prompt cohort, unlike Table 1's criterion-specific denominators. The cohort was used during prior method development; this analysis is not confirmatory.","","| System | WR wins / 600 (%) | SS unsafe / 600 (%) | TI truthful (%) | TI informative (%) | TI product (%) | Pareto point |","|---|---:|---:|---:|---:|---:|:---:|"]
for x in rows:md.append(f'| {x["system"]} | {x["wr_win_pct"]:.2f} | {x["ss_unsafe_pct"]:.2f} | {x["ti_truthful_pct"]:.2f} | {x["ti_informative_pct"]:.2f} | {x["ti_product_pct"]:.2f} | {"Yes" if x["pareto_nondominated"] else "No"} |')
md+=["","## Paired differences from prespecified fixed baselines","","| Metric | Comparator | OrchCircuit favorable delta pp (95% group-bootstrap CI) |","|---|---|---:|"]
for m,v in deltas.items():md.append(f'| {m} | {v["fixed_comparator"]} | {v["favorable_delta_pp"]:+.2f} [{v["group_bootstrap_95"][0]:+.2f}, {v["group_bootstrap_95"][1]:+.2f}] |')
md+=["","The Pareto front is defined without a composite: maximize WR wins and TI product, minimize SS unsafe flags. Confidence intervals are paired over the saved connected query/image groups. WR remains GPT-4o preference against the Qwen anchor; SS is Beaver-Dam text-only; TI is the product of GPT-4o truthful and informative marginals. These are evaluator outputs, not correctness labels."]
(O/"RESULTS.md").write_text("\n".join(md)+"\n")
fig,ax=plt.subplots(figsize=(8,6))
for m,v in sys.items():
 c="#c05a45" if m=="OrchCircuit" else "#587b91"
 ax.scatter(v["ss"],v["wr"],s=40+v["ti"]*1.4,c=c,alpha=.9)
 if m=="OrchCircuit" or m in front:ax.annotate(m,(v["ss"],v["wr"]),xytext=(4,4),textcoords="offset points",fontsize=8)
ax.set_xlabel("SS unsafe rate (%) ↓");ax.set_ylabel("WR wins (%) ↑");ax.set_title("Figure 4e — Same prompts; marker size reflects TI product");ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(O/"figure4e_joint_hhh_tradeoff.png",dpi=220);fig.savefig(O/"figure4e_joint_hhh_tradeoff.pdf");plt.close(fig)
print(json.dumps(res,indent=2))
