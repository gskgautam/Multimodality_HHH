#!/usr/bin/env python3
"""Reproduce exploratory Figure 3e/3f using frozen grouped-OOF outputs.

No model calls are made. The 600 prompts were used in prior OrchCircuit
development, so these analyses are exploratory, not confirmatory.
"""
import argparse, csv, importlib.util, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MODELS = [
    "lfm3-vl-2b","qwen3.5-2b","gemma4-e2b","qwen2.5-vl-7b","qwen3-vl-8b",
    "minicpm-v-4.5-8b","gemma3-12b","gemma-4-12b","pixtral-12b","gemma3-27b",
    "qwen3.6-27b","qwen3.8-27b","qwen3.5-35b-a3b","qwen3.6-35b-a3b",
    "minicpm-v-4.7-35b-a3b","gpt-6.1-terra","gpt-6.1-sol","gpt-6-luna",
    "gemini-4-argon","gemini-3.8-flash","gemini-3.6-flash","claude-opus-5-5",
    "claude-sonnet-5-5","claude-haiku-5-5"]
ANCHOR = "qwen2.5-vl-7b"
SEED = 20261113

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--bootstrap",type=int,default=5000)
    a=ap.parse_args()
    root=a.root
    exp=root/"Method/OrchCircuit/final"
    figroot=exp/"figure3"
    private=root/"Datasets/Ch3EF/data/ch3ef_full1416/joint_hhh_labels_600_PRIVATE"
    sys.path.insert(0,str(exp))
    spec=importlib.util.spec_from_file_location("selector_cv",exp/"run_selector_cv.py")
    cv=importlib.util.module_from_spec(spec);spec.loader.exec_module(cv)

    manifest=json.loads((private/"sample_manifest.json").read_text())
    orders=[int(x) for x in manifest["orders"]]
    if len(orders)!=600 or len(set(orders))!=600: raise ValueError("Expected 600 unique prompt orders")
    data=root/"Datasets/Ch3EF/data/ch3ef_full1416"
    records=json.loads((data/"sample.json").read_text())["records"]
    by={int(r["sample_order"]):r for r in records}
    order_ix={o:i for i,o in enumerate(orders)}
    mi={m:i for i,m in enumerate(MODELS)}
    wr=np.full((600,24),-1,dtype=np.int8)
    ss=np.full((600,24),-1,dtype=np.int8)
    tr=np.full((600,24),-1,dtype=np.int8)
    inf=np.full((600,24),-1,dtype=np.int8)
    gpt=read_jsonl(private/"gpt4o_joint_labels_v2.jsonl")
    label_rows={(int(r["sample_order"]),r["task"]):r for r in gpt if not r.get("error") and isinstance(r.get("labels"),dict)}
    if any((o,t) not in label_rows for o in orders for t in ("wr","ti")):
        raise ValueError("GPT-4o WR/TI labels incomplete")
    for o in orders:
        i=order_ix[o]
        for task in ("wr","ti"):
            row=label_rows[(o,task)]
            if set(row["candidate_map"].values())!=set(MODELS): raise ValueError(f"candidate coverage mismatch at {o}/{task}")
            for code,model in row["candidate_map"].items():
                lab=row["labels"].get(code)
                if lab is None: raise ValueError(f"missing label at {o}/{task}/{code}")
                j=mi[model]
                if task=="wr": wr[i,j]=int(lab["preference"]=="win")
                else:
                    tr[i,j]=int(bool(lab["truthful"]))
                    inf[i,j]=int(bool(lab["informative"]))
    ss_rows=read_jsonl(private/"beaver_dam_ss_600x24.jsonl")
    for row in ss_rows:
        o=int(row["sample_order"])
        if o in order_ix: ss[order_ix[o],mi[row["model"]]]=int(bool(row["unsafe"]))
    if any(np.any(a<0) for a in (wr,ss,tr,inf)): raise ValueError("Incomplete 600 x 24 joint labels")
    preds=[json.loads(x) for x in (private/"orchcircuit_joint_oof_predictions.jsonl").read_text().splitlines() if x.strip()]
    if len(preds)!=600: raise ValueError("Expected 600 OrchCircuit OOF rows")
    pred_by={int(r["sample_order"]):r for r in preds}
    if set(pred_by)!=set(orders): raise ValueError("OOF predictions do not match manifest")
    fold={o:int(pred_by[o]["fold"]) for o in orders}
    mi={m:i for i,m in enumerate(MODELS)}
    # Existing WR metric is candidate wins against the Qwen anchor divided by all prompts; ties are not wins.
    w=wr.astype(float); s=ss.astype(float); t=tr.astype(float); q=inf.astype(float)
    orch_idx=np.asarray([mi[pred_by[o]["chosen_model"]] for o in orders])
    row_ix=np.arange(len(orders))
    systems={}
    kinds={}
    systems["OrchCircuit"]=(w[row_ix,orch_idx],s[row_ix,orch_idx],t[row_ix,orch_idx],q[row_ix,orch_idx])
    kinds["OrchCircuit"]="proposed method"
    for j,m in enumerate(MODELS):
        systems[m]=(w[:,j],s[:,j],t[:,j],q[:,j])
        kinds[m]="open-weight" if j<15 else "closed-source"

    # Fold-specific fixed selectors choose on outer-training prompts only.
    best_wr_idx=np.zeros(600,dtype=int); constrained_idx=np.zeros(600,dtype=int)
    fold_choices={}
    ai=mi[ANCHOR]
    for f in sorted(set(fold.values())):
        test=np.asarray([i for i,o in enumerate(orders) if fold[o]==f])
        train=np.asarray([i for i,o in enumerate(orders) if fold[o]!=f])
        mw=w[train].mean(axis=0); ms=s[train].mean(axis=0); mt=t[train].mean(axis=0)*q[train].mean(axis=0)
        jwr=int(np.argmax(mw))
        safe_floor=float(ms[ai]); ti_floor=float(mt[ai])
        feasible=(ms<=safe_floor+1e-12)&(mt>=ti_floor-1e-12)
        if feasible.any():
            pool=np.flatnonzero(feasible); jc=int(pool[np.argmax(mw[pool])])
        else:
            violation=np.maximum(ms-safe_floor,0)/max(safe_floor,1e-8)+np.maximum(ti_floor-mt,0)/max(ti_floor,1e-8)
            pool=np.flatnonzero(np.isclose(violation,violation.min())); jc=int(pool[np.argmax(mw[pool])])
        best_wr_idx[test]=jwr; constrained_idx[test]=jc
        fold_choices[str(f)]={"test_n":int(len(test)),"best_WR_model":MODELS[jwr],
             "constrained_fixed_model":MODELS[jc],"training_anchor_unsafe_rate":safe_floor,
             "training_anchor_TI_product":ti_floor}
    systems["Fixed best-WR model"]=(w[row_ix,best_wr_idx],s[row_ix,best_wr_idx],t[row_ix,best_wr_idx],q[row_ix,best_wr_idx])
    kinds["Fixed best-WR model"]="fixed selector"
    systems["Fixed constrained model"]=(w[row_ix,constrained_idx],s[row_ix,constrained_idx],t[row_ix,constrained_idx],q[row_ix,constrained_idx])
    kinds["Fixed constrained model"]="fixed selector"
    systems["Qwen anchor"]=systems[ANCHOR]; kinds["Qwen anchor"]="fixed selector"
    # Mean candidate outcome is the expected metric vector for a uniform random valid candidate.
    systems["Uniform random candidate"]=(w.mean(axis=1),s.mean(axis=1),t.mean(axis=1),q.mean(axis=1))
    kinds["Uniform random candidate"]="random-choice expectation"

    def score(x):
        return {"n":int(len(x[0])),"WR":float(100*np.mean(x[0])),"SS":float(100*np.mean(x[1])),
                "truthful":float(100*np.mean(x[2])),"informative":float(100*np.mean(x[3])),
                "TI":float(100*np.mean(x[2])*np.mean(x[3]))}
    points={name:score(x) for name,x in systems.items()}

    # Match the selector CV definition: connect records sharing normalized question or image hash.
    group_of,_=cv.make_groups(by,orders)
    group_keys=sorted(set(group_of[o] for o in orders)); gi={g:k for k,g in enumerate(group_keys)}
    group_n=np.zeros(len(group_keys),dtype=int)
    names=list(systems); mats=[np.stack([systems[n][k] for n in names],axis=1) for k in range(4)]
    grouped=[np.zeros((len(group_keys),len(names))) for _ in range(4)]
    for i,o in enumerate(orders):
        g=gi[group_of[o]]; group_n[g]+=1
        for k in range(4): grouped[k][g]+=mats[k][i]
    rng=np.random.default_rng(SEED)
    counts=rng.multinomial(len(group_keys),np.full(len(group_keys),1/len(group_keys)),size=a.bootstrap)
    denom=counts@group_n
    bmeans=[(counts@totals)/denom[:,None] for totals in grouped]
    bti=bmeans[2]*bmeans[3]
    orch_col=names.index("OrchCircuit")
    comparison=[]
    for name in names:
        if name=="OrchCircuit": continue
        j=names.index(name)
        diffs={"WR":100*(bmeans[0][:,orch_col]-bmeans[0][:,j]),
               "SS":100*(bmeans[1][:,j]-bmeans[1][:,orch_col]),
               "TI":100*(bti[:,orch_col]-bti[:,j])}
        pts={"WR":points["OrchCircuit"]["WR"]-points[name]["WR"],
             "SS":points[name]["SS"]-points["OrchCircuit"]["SS"],
             "TI":points["OrchCircuit"]["TI"]-points[name]["TI"]}
        for metric in ("WR","SS","TI"):
            lo,hi=np.percentile(diffs[metric],[2.5,97.5])
            comparison.append({"comparator":name,"type":kinds[name],"metric":metric,"n":600,
                "orchcircuit_value":points["OrchCircuit"][metric],"comparator_value":points[name][metric],
                "difference_pp_positive_favors_orchcircuit":float(pts[metric]),
                "ci95_low_pp":float(lo),"ci95_high_pp":float(hi)})

    # Pareto status uses WR and TI maximization, SS-unsafe minimization; no composite score.
    frontier={}
    for x in systems:
        px=points[x]; dominated=False
        for y in systems:
            if x==y: continue
            py=points[y]
            if (py["WR"]>=px["WR"] and py["SS"]<=px["SS"] and py["TI"]>=px["TI"] and
                (py["WR"]>px["WR"] or py["SS"]<px["SS"] or py["TI"]>px["TI"])):
                dominated=True; break
        frontier[x]=not dominated

    out_e=figroot/"exploratory_600prompt_oof"/"e"; out_f=figroot/"exploratory_600prompt_oof"/"f"
    out_e.mkdir(parents=True,exist_ok=True); out_f.mkdir(parents=True,exist_ok=True)
    summary={"status":"exploratory grouped out-of-fold development analysis; not confirmatory",
        "n":600,"group_count":len(group_keys),"fold_sizes":{str(f):sum(v==f for v in fold.values()) for f in sorted(set(fold.values()))},
        "fold_choices":fold_choices,
        "label_provenance":{"WR":"GPT-4o image-present pairwise preferences against Qwen2.5-VL 7B; ties are not wins",
             "SS":"PKU Beaver-Dam 7B text-only unsafe flags, threshold 0.5",
             "TI":"GPT-4o image-present truthful and informative labels; TI is product of marginal rates"},
        "systems":points,"paired_group_bootstrap":{"replicates":a.bootstrap,"seed":SEED,
             "interval":"paired percentile 95% over connected query/image groups",
             "sign":"positive differences favor OrchCircuit for all three; SS uses comparator unsafe rate minus OrchCircuit"},
        "pareto_frontier":[x for x in systems if frontier[x]],
        "limits":["These prompts were used during earlier OrchCircuit development and variant comparisons; OOF is not a fresh confirmatory test.",
             "Automated labels are not ground truth. Figure 3a-d found limited agreement for some automated judge outcomes.",
             "Results do not establish improved human-rated answer correctness or general HHH performance."]}
    (out_e/"comparison_results.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (out_e/"paired_comparisons.csv").open("w",newline="") as f:
        cw=csv.DictWriter(f,fieldnames=list(comparison[0]));cw.writeheader();cw.writerows(comparison)
    with (out_f/"tradeoff_metrics.csv").open("w",newline="") as f:
        cw=csv.writer(f);cw.writerow(["system","type","n","WR_win_pct","SS_unsafe_pct","TI_product_pct","pareto_nondominated"])
        for name in systems: cw.writerow([name,kinds[name],600,points[name]["WR"],points[name]["SS"],points[name]["TI"],frontier[name]])

    tradeoff_json={"status":summary["status"],"n":600,"group_count":len(group_keys),
        "directions":{"WR":"maximize","SS":"minimize unsafe flags","TI":"maximize product of truthful and informative rates"},
        "systems":points,"pareto_frontier":summary["pareto_frontier"],"caveats":summary["limits"]}
    (out_f/"tradeoff_results.json").write_text(json.dumps(tradeoff_json,indent=2)+"\n")
    (out_f/"pareto_front.json").write_text(json.dumps({"nondominated_systems":summary["pareto_frontier"],
        "definition":"non-dominated on WR wins (higher), SS unsafe flags (lower), and TI product (higher), using point estimates"},indent=2)+"\n")
    # Figure 3e: paired method-minus-comparator differences and group-bootstrap intervals.
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,"axes.labelsize":11,
         "axes.labelweight":"bold","figure.facecolor":"white","axes.facecolor":"white","svg.fonttype":"none",
         "axes.edgecolor":"#555555","grid.color":"#dddddd","grid.linestyle":":","grid.linewidth":.7})
    display=list(reversed(MODELS+["Uniform random candidate","Qwen anchor","Fixed best-WR model","Fixed constrained model"]))
    colors={"open-weight":"#76a892","closed-source":"#819ac9","fixed selector":"#d5a448","random-choice expectation":"#c88562"}
    table={(r["comparator"],r["metric"]):r for r in comparison}
    fig,axes=plt.subplots(1,3,figsize=(19,12),sharey=True)
    for ax,metric,xlabel in zip(axes,("WR","SS","TI"),("WR win-rate difference (pp)","Reduction in SS unsafe flags (pp)","TI product difference (pp)")):
        for y,name in enumerate(display):
            r=table[(name,metric)];c=colors[r["type"]];v=r["difference_pp_positive_favors_orchcircuit"]
            ax.errorbar(v,y,xerr=[[v-r["ci95_low_pp"]],[r["ci95_high_pp"]-v]],fmt="o",ms=5,color=c,ecolor=c,capsize=2.4,elinewidth=1.1)
        ax.axvline(0,color="#222222",linestyle="--",linewidth=1);ax.grid(axis="x")
        ax.set_xlabel(xlabel);ax.spines[["top","right","left"]].set_visible(False);ax.tick_params(axis="y",length=0)
    axes[0].set_yticks(np.arange(len(display)),display,fontsize=8.5)
    legend=[Line2D([0],[0],marker="o",color="none",markerfacecolor=colors[k],markersize=7,label=v)
        for k,v in [("open-weight","Open-weight model"),("closed-source","Closed-source model"),
                    ("fixed selector","Fixed selector"),("random-choice expectation","Uniform random expectation")]]
    fig.legend(handles=legend,loc="upper center",ncol=4,frameon=False,bbox_to_anchor=(.5,1.01))
    fig.subplots_adjust(left=.25,right=.99,bottom=.08,top=.95,wspace=.20)
    fig.savefig(out_e/"figure3e.png",dpi=300,bbox_inches="tight");fig.savefig(out_e/"figure3e.svg",bbox_inches="tight");plt.close(fig)

    # Figure 3f: Pareto view; axes retain three distinct outcomes.
    fig,ax=plt.subplots(figsize=(12.5,8.5))
    vmin=min(p["TI"] for p in points.values());vmax=max(p["TI"] for p in points.values())
    for typ,marker,label in [("open-weight","o","Open-weight"),("closed-source","D","Closed-source")]:
        ns=[n for n in MODELS if kinds[n]==typ]
        coll=ax.scatter([points[n]["WR"] for n in ns],[points[n]["SS"] for n in ns],
              c=[points[n]["TI"] for n in ns],cmap="viridis",vmin=vmin,vmax=vmax,
              marker=marker,s=95,edgecolor="white",linewidth=.8,label=label)
    for name in ("Fixed best-WR model","Fixed constrained model","Qwen anchor","Uniform random candidate"):
        p=points[name];ax.scatter(p["WR"],p["SS"],c=[p["TI"]],cmap="viridis",vmin=vmin,vmax=vmax,
              marker="P",s=125,edgecolor="#333333",linewidth=.7,label=name)
    orch_point=points["OrchCircuit"]
    ax.scatter(orch_point["WR"],orch_point["SS"],c=[orch_point["TI"]],cmap="viridis",vmin=vmin,vmax=vmax,
              marker="*",s=300,edgecolor="#a51f2b",linewidth=1.4,label="OrchCircuit",zorder=5)
    for name in systems:
        if frontier[name] and name not in ("OrchCircuit","Fixed best-WR model","Fixed constrained model","Qwen anchor","Uniform random candidate"):
            p=points[name];ax.scatter(p["WR"],p["SS"],s=155,facecolors="none",edgecolors="#161616",linewidths=1.4,zorder=4)
    ax.annotate("OrchCircuit",(orch_point["WR"],orch_point["SS"]),xytext=(-78,18),textcoords="offset points",fontsize=10.5,fontweight="bold",
                color="#9b1c29",arrowprops={"arrowstyle":"-","color":"#9b1c29","lw":1})
    ax.set_xlabel("WR wins (%)  ↑");ax.set_ylabel("SS unsafe flags (%)  ↓");ax.invert_yaxis()
    ax.grid(alpha=.7);ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="upper left",frameon=False,ncol=2,fontsize=8.5)
    cb=fig.colorbar(coll,ax=ax,pad=.025);cb.set_label("TI product (%)  ↑",fontweight="bold")
    fig.subplots_adjust(left=.11,right=.90,bottom=.12,top=.98)
    fig.savefig(out_f/"figure3f.png",dpi=300,bbox_inches="tight");fig.savefig(out_f/"figure3f.svg",bbox_inches="tight");plt.close(fig)

    best_wr=max((points[n]["WR"],n) for n in systems if n!="OrchCircuit")
    best_ss=min((points[n]["SS"],n) for n in systems if n!="OrchCircuit")
    best_ti=max((points[n]["TI"],n) for n in systems if n!="OrchCircuit")
    out_e.joinpath("results.md").write_text(
        "# Figure 3e — Paired grouped out-of-fold comparison\n\n"
        "**Exploratory development analysis, not a confirmatory test.** The panel uses 600 prompts in five outer folds "
        "with query/image-linked records grouped together. OrchCircuit is scored out of fold. Simple fixed selectors "
        "choose their model using only the other four folds; every system is scored on the same held-out fold records and candidate responses.\n\n"
        "WR is each answer's GPT-4o judged win against the Qwen2.5-VL 7B anchor divided by all 600 prompts (ties and losses are not wins); it is not a direct OrchCircuit-versus-each-baseline head-to-head judgment. SS is Beaver-Dam's text-only unsafe-flag rate; TI is the product of GPT-4o truthfulness and informativeness marginal rates. "
        "The label sources are automated judges, not ground truth. Differences are paired by query/image-connected groups, with "
        f"{a.bootstrap:,} percentile-bootstrap resamples and unadjusted 95% intervals across comparisons. "
        "Positive values favor OrchCircuit; for SS the difference is comparator unsafe flags minus OrchCircuit unsafe flags.\n\n"
        f"OrchCircuit: WR {points['OrchCircuit']['WR']:.2f}%, SS flags {points['OrchCircuit']['SS']:.2f}%, TI product {points['OrchCircuit']['TI']:.2f}%. "
        f"Best comparator point estimates: WR {best_wr[0]:.2f}% ({best_wr[1]}), SS flags {best_ss[0]:.2f}% ({best_ss[1]}), "
        f"TI product {best_ti[0]:.2f}% ({best_ti[1]}). Full paired differences and intervals are in paired_comparisons.csv.\n\n"
        "Although fold-held-out rows were not used to fit that fold's OrchCircuit model, this 600-prompt set was used in prior method development and variant comparisons. "
        "These results therefore do not establish performance on a genuinely untouched test set or human-rated HHH correctness.\n\n![Figure 3e](figure3e.png)\n")
    control_names=["OrchCircuit","Fixed best-WR model","Fixed constrained model","Qwen anchor","Uniform random candidate"]
    control_table=["","| System | WR wins (%) | SS unsafe flags (%) | TI product (%) |","|---|---:|---:|---:|"]
    for name in control_names:
        p=points[name]
        control_table.append(f"| {name} | {p['WR']:.2f} | {p['SS']:.2f} | {p['TI']:.2f} |")
    with (out_e/"results.md").open("a",encoding="utf-8") as report:
        report.write("\n".join(control_table)+"\n")
    (out_f/"tradeoff_plot.png").write_bytes((out_f/"figure3f.png").read_bytes())
    frontier=", ".join(summary["pareto_frontier"])
    out_f.joinpath("results.md").write_text(
        "# Figure 3f — Three-objective trade-off\n\n"
        "**Exploratory development analysis on the same 600 grouped out-of-fold prompts as Figure 3e.** "
        "WR wins and TI product are maximized; SS unsafe flags are minimized. The three outcomes remain separate: "
        "the scatter uses WR on x, SS on y (reversed so fewer flags plot higher), and color for TI product. "
        "Open-weight and closed-source baselines use different markers; simple fixed rules and OrchCircuit are distinguished in the legend. "
        "Outlined model points are nondominated under the three point estimates. Exact scores and Pareto labels are in tradeoff_metrics.csv.\n\n"
        f"Point-estimate nondominated systems: {frontier}. Nondominance is descriptive, not evidence of statistical superiority. "
        "Scores are automated-judge outputs and do not establish human-rated correctness or generalization to a fresh test set.\n\n![Figure 3f](figure3f.png)\n")
    print(json.dumps({"groups":len(group_keys),"systems":len(systems),"orchcircuit":points["OrchCircuit"],
                      "pareto_frontier":summary["pareto_frontier"],"figure3e":str(out_e/"figure3e.png"),
                      "figure3f":str(out_f/"figure3f.png")},indent=2))

if __name__=="__main__": main()

