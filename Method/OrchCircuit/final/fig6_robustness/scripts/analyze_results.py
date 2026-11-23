#!/usr/bin/env python3
"""Summarize Figure 4a component ablations and Figure 4c seed repeatability."""
import csv,importlib.util,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()))
EXP=ROOT/"Method/OrchCircuit/final"; F4=EXP; BASE=EXP/"runs/fullset_group5fold"
SEED0=20261111; SEEDS=[20261111,20261112,20261113]
VARIANTS=["no_set_attention","no_candidate_identity","mean_token_pool"]; BOOT=10000
spec=importlib.util.spec_from_file_location("orch_runner",F4/"fig6_robustness/a_ablations/run_selector_ablation.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
_,orders,_,per_metric,wr,ss,truth,info=mod.labels_and_orders()
mi={m:i for i,m in enumerate(mod.MODELS)}
folds=json.loads((BASE/"folds.json").read_text())
group_of={int(k):int(v) for k,v in folds["group_ids"].items()}
order_ix={o:i for i,o in enumerate(orders)}
def loadpred(path):
    return {int(r["sample_order"]):r for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())}
def chosen(pred,metric):
    key={"WR":"wr_selected_candidate","SS":"ss_selected_candidate","TI":"ti_selected_candidate"}[metric]
    vals={}
    for o in per_metric[metric]:
        m=pred[o].get(key)
        if m not in mi: raise ValueError(f"invalid {key} at {o}: {m}")
        j=mi[m];i=order_ix[o]
        if metric=="WR": vals[o]=int(wr[i,j])
        elif metric=="SS": vals[o]=int(ss[i,j])
        else: vals[o]=(int(truth[i,j]),int(info[i,j]))
    return vals
def point(metric,v):
    n=len(v)
    if metric=="WR": return 100*sum(x==0 for x in v.values())/n
    if metric=="SS": return 100*sum(v.values())/n
    return 100*(sum(x[0] for x in v.values())/n)*(sum(x[1] for x in v.values())/n)
def boot(metric,a,b,family,seed):
    grouped=defaultdict(list)
    for o in per_metric[metric]: grouped[group_of[o]].append(o)
    keys=sorted(grouped); n=len(keys); counts=np.array([len(grouped[g]) for g in keys],float)
    if metric=="TI":
        aa=np.array([sum(a[o][0] for o in grouped[g]) for g in keys],float)
        ab=np.array([sum(a[o][1] for o in grouped[g]) for g in keys],float)
        ba=np.array([sum(b[o][0] for o in grouped[g]) for g in keys],float)
        bb=np.array([sum(b[o][1] for o in grouped[g]) for g in keys],float)
    else:
        aa=np.array([sum((a[o]==0) if metric=="WR" else a[o] for o in grouped[g]) for g in keys],float)
        ba=np.array([sum((b[o]==0) if metric=="WR" else b[o] for o in grouped[g]) for g in keys],float)
    rng=np.random.default_rng(seed); draws=np.empty(BOOT,float)
    for start in range(0,BOOT,250):
        bsz=min(250,BOOT-start); ix=rng.integers(0,n,size=(bsz,n)); den=counts[ix].sum(axis=1)
        if metric=="TI":
            xa=100*(aa[ix].sum(axis=1)/den)*(ab[ix].sum(axis=1)/den)
            xb=100*(ba[ix].sum(axis=1)/den)*(bb[ix].sum(axis=1)/den)
        else:
            xa=100*aa[ix].sum(axis=1)/den; xb=100*ba[ix].sum(axis=1)/den
        draws[start:start+bsz]=xa-xb
    alpha=.05/family
    return tuple(map(float,np.quantile(draws,[alpha/2,1-alpha/2])))
def runpred(panel,variant,seed):
    return loadpred(F4/panel/"runs"/f"{variant}_seed{seed}"/"oof_selected_candidates.jsonl")
full=loadpred(BASE/"oof_selected_candidates.jsonl")
# Figure 4a: three single-component removals vs the saved full selector.
a_rows=[]
for vi,variant in enumerate(VARIANTS):
    p=runpred("a",variant,SEED0)
    for dj,metric in enumerate(("WR","SS","TI")):
        av=chosen(p,metric); bv=chosen(full,metric)
        lo,hi=boot(metric,av,bv,9,SEED0+vi*31+dj)
        a_rows.append({"variant":variant,"metric":metric,"n":len(av),
            "full_pct":point(metric,bv),"ablation_pct":point(metric,av),
            "delta_ablation_minus_full_pp":point(metric,av)-point(metric,bv),
            "bonferroni_ci_low_pp":lo,"bonferroni_ci_high_pp":hi})
ao=F4/"fig6_robustness/a_ablations/results";ao.mkdir(parents=True,exist_ok=True)
with (ao/"ablation_effects.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(a_rows[0]));w.writeheader();w.writerows(a_rows)
(ao/"ablation_effects.json").write_text(json.dumps({"status":"exploratory; reused Ch3Ef development data and automated labels","training_seed":SEED0,"split_seed":SEED0,"bootstrap_replicates":BOOT,"intervals":"paired query/image-group bootstrap; Bonferroni over 9 contrasts","delta":"ablation minus full in percentage points; lower SS is better","results":a_rows},indent=2)+"\n")
lines=["# Figure 4a — component ablation results","","Exploratory grouped out-of-fold ablations on previously used Ch3Ef records. Outcomes are automated-judge-relative, not human-verified correctness. Each variant changes one implemented component and shares the full model's grouped folds and training seed.","","Delta is ablation minus full in percentage points; higher WR/TI and lower SS are favorable. Intervals are paired group-cluster bootstrap intervals, Bonferroni-adjusted across 9 contrasts.","","| Variant | Metric | n | Full (%) | Ablation (%) | Delta (pp) | Adjusted 95% CI (pp) |","|---|---|---:|---:|---:|---:|---:|"]
for r in a_rows: lines.append(f"| {r['variant']} | {r['metric']} | {r['n']} | {r['full_pct']:.2f} | {r['ablation_pct']:.2f} | {r['delta_ablation_minus_full_pp']:+.2f} | [{r['bonferroni_ci_low_pp']:+.2f}, {r['bonferroni_ci_high_pp']:+.2f}] |")
(ao/"RESULTS.md").write_text("\n".join(lines)+"\n")

# Figure 4c: three training seeds on fixed folds; show all seed-level estimates.
c_rows=[]
for seed in SEEDS:
    p=full if seed==SEED0 else runpred("c","full",seed)
    for dj,metric in enumerate(("WR","SS","TI")):
        v=chosen(p,metric); zero={o:((0,0) if metric=="TI" else (1 if metric=="WR" else 0)) for o in per_metric[metric]}
        lo,hi=boot(metric,v,zero,1,seed+dj*101)
        c_rows.append({"training_seed":seed,"metric":metric,"n":len(v),"score_pct":point(metric,v),
            "group_bootstrap_ci_low_pct":lo,"group_bootstrap_ci_high_pct":hi})
ranges={}
for metric in ("WR","SS","TI"):
    x=[r["score_pct"] for r in c_rows if r["metric"]==metric]
    ranges[metric]={"min_pct":min(x),"max_pct":max(x),"range_pp":max(x)-min(x)}
co=F4/"fig6_robustness/b_seed_repeatability/results";co.mkdir(parents=True,exist_ok=True)
with (co/"seed_repeatability.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(c_rows[0]));w.writeheader();w.writerows(c_rows)
(co/"seed_repeatability.json").write_text(json.dumps({"status":"exploratory; same Ch3Ef groups and automated labels across seeds","training_seeds":SEEDS,"fixed_split_seed":SEED0,"bootstrap_replicates":BOOT,"interpretation":"per-seed group-bootstrap intervals describe sampling uncertainty; seed range is descriptive training variation, not a confidence interval over independent test sets","seed_level_results":c_rows,"seed_range":ranges},indent=2)+"\n")
cl=["# Figure 4c — training-seed repeatability","","Exploratory full-selector results on the same previously used Ch3Ef common-valid records. The five outer folds and grouped inner-validation assignments are fixed; only training RNG seed changes. Per-seed intervals describe group-bootstrap sampling uncertainty. Seed range is descriptive, not a confidence interval over independent datasets.","","| Training seed | Metric | n | Score (%) | Group-bootstrap 95% CI (%) |","|---:|---|---:|---:|---:|"]
for r in c_rows: cl.append(f"| {r['training_seed']} | {r['metric']} | {r['n']} | {r['score_pct']:.2f} | [{r['group_bootstrap_ci_low_pct']:.2f}, {r['group_bootstrap_ci_high_pct']:.2f}] |")
cl += ["","## Across-seed range","","| Metric | Minimum (%) | Maximum (%) | Range (pp) |","|---|---:|---:|---:|"]
for m,r in ranges.items(): cl.append(f"| {m} | {r['min_pct']:.2f} | {r['max_pct']:.2f} | {r['range_pp']:.2f} |")
(co/"RESULTS.md").write_text("\n".join(cl)+"\n")

fig,axs=plt.subplots(1,3,figsize=(13,4.2))
labels=["No set\nattention","No candidate\nidentity","Mean token\npool"]
for ax,m in zip(axs,("WR","SS","TI")):
    sub=[r for r in a_rows if r["metric"]==m];x=np.arange(len(sub))
    y=np.array([r["delta_ablation_minus_full_pp"] for r in sub])
    lo=np.array([r["bonferroni_ci_low_pp"] for r in sub]);hi=np.array([r["bonferroni_ci_high_pp"] for r in sub])
    ax.axhline(0,color="#555",lw=1,ls="--")
    ax.errorbar(x,y,yerr=[np.maximum(0, y-lo),np.maximum(0, hi-y)],fmt="o",capsize=4,color="#405f8c")
    ax.set_xticks(x,labels);ax.set_title(m);ax.set_ylabel("Ablation − full (pp)");ax.grid(axis="y",alpha=.2)
fig.tight_layout();fig.savefig(ao/"figure4a_ablation_effects.png",dpi=300,bbox_inches="tight");fig.savefig(ao/"figure4a_ablation_effects.pdf",bbox_inches="tight");plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(12,4.2))
for ax,m in zip(axs,("WR","SS","TI")):
    sub=[r for r in c_rows if r["metric"]==m];x=np.arange(len(sub));y=np.array([r["score_pct"] for r in sub])
    lo=np.array([r["group_bootstrap_ci_low_pct"] for r in sub]);hi=np.array([r["group_bootstrap_ci_high_pct"] for r in sub])
    ax.errorbar(x,y,yerr=[np.maximum(0, y-lo),np.maximum(0, hi-y)],fmt="o-",capsize=4,color="#405f8c")
    ax.set_xticks(x,[str(r["training_seed"]) for r in sub],rotation=30);ax.set_title(m);ax.set_ylabel("Score (%)");ax.grid(axis="y",alpha=.2)
fig.tight_layout();fig.savefig(co/"figure4c_seed_repeatability.png",dpi=300,bbox_inches="tight");fig.savefig(co/"figure4c_seed_repeatability.pdf",bbox_inches="tight");plt.close(fig)
print(json.dumps({"figure4a":str(ao),"figure4c":str(co),"seed_ranges":ranges},indent=2))
