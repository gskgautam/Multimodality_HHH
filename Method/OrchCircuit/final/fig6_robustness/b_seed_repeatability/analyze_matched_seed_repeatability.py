#!/usr/bin/env python3
"""Figure 4c seed repeatability using matched-init full runs from Figure 4a."""
import csv,importlib.util,json,shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()))
EXP=ROOT/"Method/OrchCircuit/final";F4=EXP
BASE=EXP/"runs/fullset_group5fold";SEEDS=[20261111,20261112,20261113];BOOT=10000
spec=importlib.util.spec_from_file_location("runner",F4/"fig6_robustness/a_ablations/run_selector_ablation.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
_,orders,_,per_metric,wr,ss,truth,info=mod.labels_and_orders()
mi={m:i for i,m in enumerate(mod.MODELS)}
basefolds=json.loads((BASE/"folds.json").read_text())
group_of={int(k):int(v) for k,v in basefolds["group_ids"].items()}
idx={o:i for i,o in enumerate(orders)}
def loadpred(seed):
 p=F4/"fig6_robustness/a_ablations"/"runs"/f"full_seed{seed}_matched"/"oof_selected_candidates.jsonl"
 return {int(x["sample_order"]):x for x in (json.loads(z) for z in p.read_text().splitlines() if z.strip())}
def vals_for(seed,metric):
 pred=loadpred(seed);key={"WR":"wr_selected_candidate","SS":"ss_selected_candidate","TI":"ti_selected_candidate"}[metric]
 out={}
 for o in per_metric[metric]:
  m=pred[o].get(key)
  if m not in mi: raise ValueError(f"missing/invalid selected candidate: {seed=} {metric=} {o=}")
  j=mi[m];i=idx[o]
  if metric=="WR":out[o]=int(wr[i,j])
  elif metric=="SS":out[o]=int(ss[i,j])
  else:out[o]=(int(truth[i,j]),int(info[i,j]))
 return out
def point(metric,v):
 n=len(v)
 if metric=="WR":return 100*sum(x==0 for x in v.values())/n
 if metric=="SS":return 100*sum(v.values())/n
 return 100*(sum(x[0] for x in v.values())/n)*(sum(x[1] for x in v.values())/n)
def boot(metric,v,seed):
 grouped=defaultdict(list)
 for o in per_metric[metric]:grouped[group_of[o]].append(o)
 keys=sorted(grouped);counts=np.array([len(grouped[g]) for g in keys],float)
 if metric=="TI":
  a=np.array([sum(v[o][0] for o in grouped[g]) for g in keys],float)
  b=np.array([sum(v[o][1] for o in grouped[g]) for g in keys],float)
 else:
  a=np.array([sum((v[o]==0) if metric=="WR" else v[o] for o in grouped[g]) for g in keys],float)
 rng=np.random.default_rng(seed);draws=np.empty(BOOT,float)
 for start in range(0,BOOT,250):
  ix=rng.integers(0,len(keys),size=(min(250,BOOT-start),len(keys)))
  den=counts[ix].sum(axis=1)
  if metric=="TI":draws[start:start+len(ix)]=100*(a[ix].sum(axis=1)/den)*(b[ix].sum(axis=1)/den)
  else:draws[start:start+len(ix)]=100*a[ix].sum(axis=1)/den
 return tuple(map(float,np.quantile(draws,[.025,.975])))
rows=[]
for si,seed in enumerate(SEEDS):
 for mi_,metric in enumerate(("WR","SS","TI")):
  v=vals_for(seed,metric);lo,hi=boot(metric,v,seed+mi_*101)
  rows.append({"training_seed":seed,"metric":metric,"n":len(v),"score_pct":point(metric,v),"group_bootstrap_ci_low_pct":lo,"group_bootstrap_ci_high_pct":hi})
ranges={}
for metric in ("WR","SS","TI"):
 xs=[r["score_pct"] for r in rows if r["metric"]==metric]
 ranges[metric]={"min_pct":min(xs),"max_pct":max(xs),"range_pp":max(xs)-min(xs)}
out=F4/"fig6_robustness/b_seed_repeatability"/"results";out.mkdir(parents=True,exist_ok=True)
archive=out/"initial_native_init"
archive.mkdir(exist_ok=True)
for name in ("RESULTS.md","seed_repeatability.csv","seed_repeatability.json","figure4c_seed_repeatability.png","figure4c_seed_repeatability.pdf"):
 src=out/name;dst=archive/name
 if src.exists() and not dst.exists():shutil.copy2(src,dst)
with (out/"seed_repeatability.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/"seed_repeatability.json").write_text(json.dumps({"status":"exploratory; Ch3Ef development data and automated judge labels","training_seeds":SEEDS,"fixed_split_seed":20261111,"initialization":"matched module-wise initialization; full-selector runs shared with the Figure 4a matched ablation protocol","bootstrap_replicates":BOOT,"interpretation":"per-seed group-bootstrap intervals describe sampling uncertainty; seed range is descriptive training variation, not an interval over independent test sets","source_runs":[str(F4/"fig6_robustness/a_ablations"/"runs"/f"full_seed{s}_matched") for s in SEEDS],"seed_level_results":rows,"seed_range":ranges},indent=2)+"\n")
lines=["# Figure 4c — training-seed repeatability (matched initialization)","","Exploratory full-selector results on previously used Ch3Ef common-valid records with automated-judge labels. The five outer folds and grouped validation assignments are fixed. The three runs use matched module-wise initialization and are the same full-selector runs used as the reference in the corrected Figure 4a ablations.","","Per-seed intervals are group-bootstrap sampling uncertainty. The across-seed range is descriptive variation over these three training runs and is not a confidence interval over new datasets.","","| Training seed | Metric | n | Score (%) | Group-bootstrap 95% CI (%) |","|---:|---|---:|---:|---:|"]
for r in rows:lines.append(f"| {r['training_seed']} | {r['metric']} | {r['n']} | {r['score_pct']:.2f} | [{r['group_bootstrap_ci_low_pct']:.2f}, {r['group_bootstrap_ci_high_pct']:.2f}] |")
lines += ["","## Across-seed range","","| Metric | Minimum (%) | Maximum (%) | Range (pp) |","|---|---:|---:|---:|"]
for m,r in ranges.items():lines.append(f"| {m} | {r['min_pct']:.2f} | {r['max_pct']:.2f} | {r['range_pp']:.2f} |")
(out/"RESULTS.md").write_text("\n".join(lines)+"\n")
fig,axs=plt.subplots(1,3,figsize=(12,4.2))
for ax,metric in zip(axs,("WR","SS","TI")):
 sub=[r for r in rows if r["metric"]==metric];x=np.arange(len(sub));y=np.array([r["score_pct"] for r in sub])
 lo=np.array([r["group_bootstrap_ci_low_pct"] for r in sub]);hi=np.array([r["group_bootstrap_ci_high_pct"] for r in sub])
 ax.errorbar(x,y,yerr=[np.maximum(0,y-lo),np.maximum(0,hi-y)],fmt="o-",capsize=4,color="#405f8c")
 ax.set_xticks(x,[str(r["training_seed"]) for r in sub],rotation=30);ax.set_title(metric);ax.set_ylabel("Score (%)");ax.grid(axis="y",alpha=.2)
fig.tight_layout();fig.savefig(out/"figure4c_seed_repeatability.png",dpi=300,bbox_inches="tight");fig.savefig(out/"figure4c_seed_repeatability.pdf",bbox_inches="tight");plt.close(fig)
print("\n".join(lines))
