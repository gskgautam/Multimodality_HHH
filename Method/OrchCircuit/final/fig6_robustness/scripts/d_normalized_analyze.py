#!/usr/bin/env python3
import csv,json,importlib.util,re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd())); E=R/"Method/OrchCircuit/final"; F=E; O=F/"fig6_robustness/c_source_strata";O.mkdir(parents=True,exist_ok=True)
s=importlib.util.spec_from_file_location("cv",E/"run_selector_cv.py");cv=importlib.util.module_from_spec(s);s.loader.exec_module(cv)
def jl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
by,orders,dim,per,wr,ss,tr,info=cv.labels_and_orders(); oi={o:i for i,o in enumerate(orders)}
grp,_=cv.make_groups(by,orders); rec=json.loads((cv.DATA/"sample.json").read_text())["records"]
MERGE={"reffered vision":"referred vision comprehension","gui":"gui comprehension","handwriting":"handwriting comprehension","racial":"racial discrimination","racialdiscrimination":"racial discrimination"}
src={int(x["sample_order"]):x.get("source","unknown").strip().lower().replace("_"," ").replace("-"," ").replace("/"," ") for x in rec}
src={o:MERGE.get(v,v) for o,v in src.items()}
seeds=[20261111,20261112,20261113]; preds={}
for sd in seeds:
 p=F/"fig6_robustness/b_seed_repeatability"/"runs"/f"full_seed{sd}"/"oof_selected_candidates.jsonl"
 if not p.exists():p=F/"fig6_robustness/a_ablations"/"runs"/f"full_seed{sd}_matched"/"oof_selected_candidates.jsonl"
 preds[sd]={int(x["sample_order"]):x for x in jl(p)}
res=json.loads((E/"runs/fullset_group5fold/results.json").read_text())
base={m:res["best_fixed_on_oof_cohort"][m]["model"] for m in ("WR","SS","TI")}
def vals(metric,o,slug):
 j=cv.MODELS.index(slug);i=oi[o]
 if metric=="WR":return int(wr[i,j])
 if metric=="SS":return int(ss[i,j])
 return int(tr[i,j]),int(info[i,j])
def score(metric,v):
 if metric=="WR":return 100*sum(x==0 for x in v)/len(v)
 if metric=="SS":return 100*sum(v)/len(v)
 return 100*np.mean([x[0]*x[1] for x in v])
def adv(metric,a,b):return score(metric,b)-score(metric,a) if metric=="SS" else score(metric,a)-score(metric,b)
out=[];rng=np.random.default_rng(20261113)
for metric in ("WR","SS","TI"):
 bmodel=base[metric];field={"WR":"wr_selected_candidate","SS":"ss_selected_candidate","TI":"ti_selected_candidate"}[metric]
 buckets=defaultdict(list)
 for o in per[metric]:buckets[src[o]].append(o)
 for source,osrc in sorted(buckets.items()):
  bygrp=defaultdict(list)
  for o in osrc:bygrp[grp[o]].append(o)
  keys=list(bygrp);nclusters=len(keys);seed_delta=[]; seed_sel=[]
  for sd in seeds:
   p=preds[sd];a=[];b=[]
   for o in osrc:
    slug=p[o].get(field) or p[o]["selected_candidate"];a.append(vals(metric,o,slug));b.append(vals(metric,o,bmodel))
   seed_delta.append(adv(metric,a,b));seed_sel.append(score(metric,a))
  ci=[None,None]
  if nclusters>=5:
   draws=np.empty(2000)
   for k in range(len(draws)):
    chosen=rng.choice(keys,len(keys),replace=True);ds=[]
    for sd in seeds:
     p=preds[sd];a=[];b=[]
     for g in chosen:
      for o in bygrp[g]:
       slug=p[o].get(field) or p[o]["selected_candidate"];a.append(vals(metric,o,slug));b.append(vals(metric,o,bmodel))
     ds.append(adv(metric,a,b))
    draws[k]=np.mean(ds)
   ci=[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]
  out.append({"source":source,"metric":metric,"n":len(osrc),"n_connected_groups":nclusters,"baseline_model":bmodel,"orch_mean_pct":float(np.mean(seed_sel)),"baseline_pct":score(metric,[vals(metric,o,bmodel) for o in osrc]),"delta_pp":float(np.mean(seed_delta)),"ci_low":ci[0],"ci_high":ci[1],"seed_deltas":";".join(f"{a}:{b:+.2f}" for a,b in zip(seeds,seed_delta))})
with open(O/"source_metric_strata.csv","w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
fig,axs=plt.subplots(1,3,figsize=(17,10),constrained_layout=True)
for ax,m,title,color in zip(axs,("WR","SS","TI"),("WR · Helpful","SS · Harmless","TI · Honest"),("#3c6e91","#bb6b35","#4c8a72")):
 rows=sorted([x for x in out if x["metric"]==m and x["n"]>=12 and x["n_connected_groups"]>=5],key=lambda x:x["delta_pp"])
 if rows:
  y=np.arange(len(rows));xx=np.array([x["delta_pp"] for x in rows]); lo=xx-np.array([x["ci_low"] for x in rows]);hi=np.array([x["ci_high"] for x in rows])-xx
  ax.errorbar(xx,y,xerr=np.vstack([lo,hi]),fmt="o",color=color,capsize=2)
  ax.set_yticks(y,[f'{x["source"]} (n={x["n"]}, groups={x["n_connected_groups"]})' for x in rows],fontsize=7)
 ax.axvline(0,color="black",ls="--",lw=.8);ax.set_title(title);ax.set_xlabel("Favorable delta (pp; >0 favors OrchCircuit)");ax.grid(axis="x",alpha=.2)
fig.suptitle("Figure 4d — Cross-question analysis by Ch3EF source (normalized labels; joint TI)",fontsize=14)
fig.savefig(O/"figure4d_source_strata.png",dpi=220);fig.savefig(O/"figure4d_source_strata.pdf");plt.close(fig)
md=["# Figure 4d — Cross-question source strata","","Exploratory paired subgroup analysis by the original Ch3EF source field. Labels were normalized for case/underscore/hyphen and four spelling duplicates were merged by label text before outcomes were computed (reffered vision, GUI, handwriting, Racial/RacialDiscrimination). TI is the joint truthful-AND-informative rate. The fixed comparator comes from the full-set OOF common-valid comparison. Positive deltas favor OrchCircuit; SS is oriented as baseline unsafe rate minus OrchCircuit unsafe rate.","","| Metric | Source | n | groups | Orch mean % | Baseline | Baseline % | Favorable delta pp (95% cluster bootstrap) |","|---|---|---:|---:|---:|---|---:|---:|"]
for x in out:md.append(f'| {x["metric"]} | {x["source"]} | {x["n"]} | {x["n_connected_groups"]} | {x["orch_mean_pct"]:.2f} | {x["baseline_model"]} | {x["baseline_pct"]:.2f} | {x["delta_pp"]:+.2f} [{x["ci_low"]:+.2f}, {x["ci_high"]:+.2f}] |' if x["ci_low"] is not None else f'| {x["metric"]} | {x["source"]} | {x["n"]} | {x["n_connected_groups"]} | {x["orch_mean_pct"]:.2f} | {x["baseline_model"]} | {x["baseline_pct"]:.2f} | {x["delta_pp"]:+.2f} [not estimable; <5 groups] |')
md+=["","Intervals resample connected query/image groups and average paired deltas over three training seeds. They are exploratory and unadjusted across source strata; small groups are included in the CSV; plotted strata require n≥12 and at least five connected query/image groups. CIs are omitted for strata with fewer than five groups. The data were used in method development; this is not a fresh confirmatory test."]
(O/"RESULTS.md").write_text("\n".join(md)+"\n")
print(json.dumps({"source_families":len(set(x["source"] for x in out)),"rows":len(out),"plotted":sum(x["n"]>=12 and x["n_connected_groups"]>=5 for x in out),"files":[str(O/"RESULTS.md"),str(O/"source_metric_strata.csv"),str(O/"figure4d_source_strata.png")]},indent=2))
