#!/usr/bin/env python3
"""Paired three-seed, matched-initialization analysis for Figure 4a."""
import csv, importlib.util, json, shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()))
EXP=ROOT/"Method/OrchCircuit/final"
F4=EXP/"figure4"
SEEDS=[20261111,20261112,20261113]
VARIANTS=["no_set_attention","no_candidate_identity","mean_token_pool"]
BOOT=10000
spec=importlib.util.spec_from_file_location("orch_runner",F4/"a/run_selector_ablation.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
_,orders,_,per_metric,wr,ss,truth,info=mod.labels_and_orders()
mi={m:i for i,m in enumerate(mod.MODELS)}
fingerprint_checks=[]
def loadpred(path):
    return {int(r["sample_order"]):r for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())}
def check_manifests():
    reference=None
    for seed in SEEDS:
        for variant in ["full"]+VARIANTS:
            path=F4/"a"/"runs"/f"{variant}_seed{seed}_matched"
            manifest=json.loads((path/"folds.json").read_text())
            if manifest["init_scheme"]!="matched": raise ValueError(f"unmatched init: {path}")
            if reference is None: reference=manifest["orders_by_fold"]
            elif manifest["orders_by_fold"]!=reference: raise ValueError(f"fold mismatch: {path}")
            for fold in range(5):
                row=json.loads((path/f"fold{fold}.json").read_text())
                if row["init_scheme"]!="matched": raise ValueError(f"fold init mismatch: {path}/fold{fold}.json")
                fingerprint_checks.append((seed,variant,fold,row["initialization_fingerprints"]))
    for seed in SEEDS:
        for fold in range(5):
            group=[x for x in fingerprint_checks if x[0]==seed and x[2]==fold]
            by={variant:fps for _,variant,_,fps in group}
            for name in ("input_projection","wr_head","ss_head","truth_head","info_head"):
                if len({by[v][name] for v in by})!=1: raise ValueError(f"initialization mismatch: {seed=} {fold=} {name=}")
            for v in ("full","no_candidate_identity","mean_token_pool"):
                if by[v]["set_layer"]!=by["full"]["set_layer"]: raise ValueError(f"set layer mismatch: {seed=} {fold=} {v=}")
            for v in ("full","no_set_attention","no_candidate_identity"):
                if by[v]["token_attention"]!=by["full"]["token_attention"]: raise ValueError(f"token attention mismatch: {seed=} {fold=} {v=}")
    return reference
def selected_values(pred,metric):
    key={"WR":"wr_selected_candidate","SS":"ss_selected_candidate","TI":"ti_selected_candidate"}[metric]
    vals={}; order_index={o:i for i,o in enumerate(orders)}
    for o in per_metric[metric]:
        slug=pred[o].get(key)
        if slug not in mi: raise ValueError(f"invalid/missing selected model for {metric} order {o}: {slug}")
        j=mi[slug];i=order_index[o]
        if metric=="WR": vals[o]=int(wr[i,j])
        elif metric=="SS": vals[o]=int(ss[i,j])
        else: vals[o]=(int(truth[i,j]),int(info[i,j]))
    return vals
def score(metric,vals):
    n=len(vals)
    if metric=="WR":return 100*sum(v==0 for v in vals.values())/n
    if metric=="SS":return 100*sum(vals.values())/n
    return 100*(sum(v[0] for v in vals.values())/n)*(sum(v[1] for v in vals.values())/n)
def grouped_counts(metric,vals,grouped,group_keys):
    count=np.array([len(grouped[g]) for g in group_keys],float)
    if metric=="WR":
        a=np.array([sum(vals[o]==0 for o in grouped[g]) for g in group_keys],float)
        return count,a,None
    if metric=="SS":
        a=np.array([sum(vals[o] for o in grouped[g]) for g in group_keys],float)
        return count,a,None
    a=np.array([sum(vals[o][0] for o in grouped[g]) for g in group_keys],float)
    b=np.array([sum(vals[o][1] for o in grouped[g]) for g in group_keys],float)
    return count,a,b
folds=json.loads((EXP/"runs/fullset_group5fold/folds.json").read_text())
group_of={int(k):int(v) for k,v in folds["group_ids"].items()}
rows=[]
check_manifests()
for metric in ("WR","SS","TI"):
    grouped=defaultdict(list)
    for o in per_metric[metric]:grouped[group_of[o]].append(o)
    group_keys=sorted(grouped)
    pred_by_seed={};counts_by_seed={}
    for seed in SEEDS:
        pred_by_seed[seed]={};counts_by_seed[seed]={}
        for variant in ["full"]+VARIANTS:
            path=F4/"a"/"runs"/f"{variant}_seed{seed}_matched"/"oof_selected_candidates.jsonl"
            vals=selected_values(loadpred(path),metric)
            pred_by_seed[seed][variant]=vals
            counts_by_seed[seed][variant]=grouped_counts(metric,vals,grouped,group_keys)
    for vi,variant in enumerate(VARIANTS):
        per_seed={seed:score(metric,pred_by_seed[seed][variant])-score(metric,pred_by_seed[seed]["full"]) for seed in SEEDS}
        point_delta=float(np.mean(list(per_seed.values())))
        rng=np.random.default_rng(20261111+vi*37+("WR","SS","TI").index(metric)*101)
        draws=np.empty(BOOT,float)
        for start in range(0,BOOT,250):
            bsz=min(250,BOOT-start)
            ix=rng.integers(0,len(group_keys),size=(bsz,len(group_keys)))
            seed_deltas=[]
            for seed in SEEDS:
                den=counts_by_seed[seed]["full"][0][ix].sum(axis=1)
                if metric=="TI":
                    fa,fb=counts_by_seed[seed]["full"][1:]
                    va,vb=counts_by_seed[seed][variant][1:]
                    fullpct=100*(fa[ix].sum(axis=1)/den)*(fb[ix].sum(axis=1)/den)
                    varpct=100*(va[ix].sum(axis=1)/den)*(vb[ix].sum(axis=1)/den)
                else:
                    fa=counts_by_seed[seed]["full"][1]
                    va=counts_by_seed[seed][variant][1]
                    fullpct=100*fa[ix].sum(axis=1)/den
                    varpct=100*va[ix].sum(axis=1)/den
                seed_deltas.append(varpct-fullpct)
            draws[start:start+bsz]=np.mean(np.stack(seed_deltas,axis=0),axis=0)
        alpha=.05/9
        lo,hi=np.quantile(draws,[alpha/2,1-alpha/2])
        full_mean=float(np.mean([score(metric,pred_by_seed[s]["full"]) for s in SEEDS]))
        var_mean=float(np.mean([score(metric,pred_by_seed[s][variant]) for s in SEEDS]))
        rows.append({"variant":variant,"metric":metric,"n":len(per_metric[metric]),
          "full_mean_pct":full_mean,"ablation_mean_pct":var_mean,"mean_delta_pp":point_delta,
          "seed_deltas_pp":"; ".join(f"{s}:{per_seed[s]:+.2f}" for s in SEEDS),
          "bonferroni_group_bootstrap_low_pp":float(lo),"bonferroni_group_bootstrap_high_pp":float(hi)})
out=F4/"a"/"results";out.mkdir(parents=True,exist_ok=True)
legacy=out/"initial_native_init_results.md"
if not legacy.exists() and (out/"RESULTS.md").exists():shutil.copy2(out/"RESULTS.md",legacy)
with (out/"matched_ablation_effects.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/"matched_ablation_effects.json").write_text(json.dumps({
 "status":"exploratory; repeated Ch3Ef development cohort and automated judge labels",
 "initialization":"module-wise deterministic initialization; matching shared parameter fingerprints across variants per seed and fold",
 "training_seeds":SEEDS,"split_seed":20261111,"bootstrap_replicates":BOOT,
 "interval":"paired connected query/image-group bootstrap over groups; calculate paired variant-minus-full difference within each seed and average across three seeds per draw; Bonferroni over 9 variant-metric contrasts",
 "note":"Intervals are conditional on the three selected training seeds; no result is selected based on direction.",
 "results":rows},indent=2)+"\n")
lines=["# Figure 4a — matched-initialization component ablations","","Exploratory grouped out-of-fold ablations on previously used Ch3Ef records. Labels are automated-judge outputs, not human-verified correctness. All four model variants were rerun under three training seeds with identical outer/inner folds and deterministic module-wise matched initialization for shared components. Initialization fingerprints were checked across every seed and fold.","","Delta is ablation minus full in percentage points; higher WR/TI and lower SS are favorable. The intervals are paired connected-group bootstrap intervals for the three-seed mean delta, Bonferroni-adjusted across the nine planned comparisons. The interval conditions on these three training seeds and does not estimate uncertainty on a new dataset.","","| Variant | Metric | n | Full mean (%) | Ablation mean (%) | Mean delta (pp) | Per-seed deltas (pp) | Adjusted 95% CI (pp) |","|---|---|---:|---:|---:|---|---:|"]
for r in rows:lines.append(f"| {r['variant']} | {r['metric']} | {r['n']} | {r['full_mean_pct']:.2f} | {r['ablation_mean_pct']:.2f} | {r['mean_delta_pp']:+.2f} | {r['seed_deltas_pp']} | [{r['bonferroni_group_bootstrap_low_pp']:+.2f}, {r['bonferroni_group_bootstrap_high_pp']:+.2f}] |")
lines += ["","The prior single-seed results used native whole-model initialization, which changed initial values of shared downstream modules when a component was removed. Those initial results are retained in initial_native_init_results.md for audit but are superseded by this matched-initialization analysis."]
(out/"RESULTS.md").write_text("\n".join(lines)+"\n")
fig,axs=plt.subplots(1,3,figsize=(13,4.2))
labels=["No set\nattention","No candidate\nidentity","Mean token\npool"]
for ax,metric in zip(axs,("WR","SS","TI")):
    sub=[r for r in rows if r["metric"]==metric];x=np.arange(len(sub))
    y=np.array([r["mean_delta_pp"] for r in sub])
    lo=np.array([r["bonferroni_group_bootstrap_low_pp"] for r in sub]);hi=np.array([r["bonferroni_group_bootstrap_high_pp"] for r in sub])
    ax.axhline(0,color="#555",lw=1,ls="--")
    ax.errorbar(x,y,yerr=[np.maximum(0,y-lo),np.maximum(0,hi-y)],fmt="o",capsize=4,color="#405f8c")
    ax.set_xticks(x,labels);ax.set_title(metric);ax.set_ylabel("Ablation − full (pp)");ax.grid(axis="y",alpha=.2)
fig.tight_layout();fig.savefig(out/"figure4a_ablation_effects.png",dpi=300,bbox_inches="tight");fig.savefig(out/"figure4a_ablation_effects.pdf",bbox_inches="tight");plt.close(fig)
print("Matched-init Figure 4a analysis complete:",out)
print("\n".join(lines))
