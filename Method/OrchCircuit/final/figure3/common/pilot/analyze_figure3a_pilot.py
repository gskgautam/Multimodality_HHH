#!/usr/bin/env python3
"""Descriptive Figure 3a analysis of the Ch3Ef human rubric pilot.

This is a development-only rubric pilot (45 cases), not confirmatory evidence.
Consumes only de-identified rating fields and the private model-order key.
"""
import csv, json
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = (next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()))
PILOT = ROOT / "Datasets/Ch3EF/data/ch3ef_full1416/human_rubric_pilot_20261112"
INFILE = PILOT / "ratings_labels_only_PRIVATE.csv"
KEYFILE = PILOT / "identity_key_PRIVATE.csv"
OUTDIR = ROOT / "Method/OrchCircuit/final/figure3/a"
OUTDIR.mkdir(parents=True, exist_ok=True)

with INFILE.open(newline="", encoding="utf-8") as f:
    ratings = list(csv.DictReader(f))
with KEYFILE.open(newline="", encoding="utf-8-sig") as f:
    keyrows = list(csv.DictReader(f))
key = {(r["annotator_code"], r["case_code"]): r for r in keyrows}
assert len(ratings) == 135, f"Expected 135 rater-case rows, got {len(ratings)}"
assert len(keyrows) == 135, f"Expected 135 identity rows, got {len(keyrows)}"
annotators = sorted({r["annotator_code"] for r in ratings})
cases = sorted({r["case_code"] for r in ratings})
assert len(annotators) == 3 and len(cases) == 45
assert all((a,c) in key for a in annotators for c in cases)
rating_by = {(r["annotator_code"], r["case_code"]): r for r in ratings}
dim_by_case = {}
for c in cases:
    dims = {rating_by[(a,c)]["target_dimension"] for a in annotators}
    assert len(dims) == 1, (c, dims)
    dim_by_case[c] = dims.pop()
assert Counter(dim_by_case.values()) == {"Helpful":15,"Harmless":15,"Honest":15}

def reliability(unit_rows):
    n_raters = 3
    assert unit_rows and all(len(x)==n_raters for x in unit_rows)
    pair_agree = sum(sum(x[i] == x[j] for i in range(n_raters) for j in range(i+1,n_raters)) for x in unit_rows) / (len(unit_rows)*3)
    categories = sorted({v for row in unit_rows for v in row})
    counts = [[row.count(cat) for cat in categories] for row in unit_rows]
    p = [sum(row[j] for row in counts)/(len(unit_rows)*n_raters) for j in range(len(categories))]
    p_obs = sum((sum(v*v for v in row)-n_raters)/(n_raters*(n_raters-1)) for row in counts)/len(unit_rows)
    p_exp = sum(v*v for v in p)
    kappa = (p_obs-p_exp)/(1-p_exp) if p_exp < 1 else None
    return {"units":len(unit_rows),"unanimous_units":sum(len(set(row))==1 for row in unit_rows),
            "pairwise_agreement":pair_agree,"fleiss_kappa":kappa,
            "label_counts":dict(Counter(v for row in unit_rows for v in row)),
            "kappa_note":"undefined: no label variation" if kappa is None else None}

helpful_units=[]
for c in cases:
    if dim_by_case[c] != "Helpful": continue
    labels=[]
    for a in annotators:
        r=rating_by[(a,c)]; k=key[(a,c)]
        value=(r.get("helpfulness_preference_A_B_tie_uncertain") or "").strip().lower()
        if value in {"tie","uncertain"}:
            labels.append(value); continue
        if value not in {"a","b"}: raise ValueError(f"Missing/invalid helpful label at {a}/{c}: {value}")
        chosen_model=k[f"{value.upper()}_model"]
        if chosen_model == k["selected_model"]: labels.append("selected")
        elif chosen_model == k["baseline_model"]: labels.append("baseline")
        else: raise ValueError(f"Pair mapping mismatch at {a}/{c}")
    helpful_units.append(labels)

def response_units(dimension, field_stem):
    units=defaultdict(dict)
    for c in cases:
        if dim_by_case[c] != dimension: continue
        for a in annotators:
            r=rating_by[(a,c)]; k=key[(a,c)]
            for letter in ("A","B"):
                model=k[f"{letter}_model"]
                val=(r.get(f"{field_stem}_{letter}") or "").strip().lower()
                if not val: raise ValueError(f"Missing {field_stem} label at {a}/{c}/{letter}")
                if a in units[(c,model)]: raise ValueError(f"Duplicate model response mapping at {a}/{c}/{model}")
                units[(c,model)][a]=val
    result=[]
    for (c,model), by_annotator in units.items():
        assert set(by_annotator)==set(annotators), (c,model,by_annotator)
        result.append([by_annotator[a] for a in annotators])
    return result

results={
    "status":"development-only rubric pilot; not confirmatory; not a held-out test",
    "source":"existing Ch3Ef development cases","case_count":len(cases),"annotator_count":len(annotators),
    "cases_per_dimension":dict(Counter(dim_by_case.values())),
    "rating_units":{"helpful_pairwise_comparisons":15,"harmless_response_units":30,"honest_response_units":30},
    "human_agreement":{
        "helpfulness_pairwise_selected_vs_baseline":reliability(helpful_units),
        "safety":reliability(response_units("Harmless","safety")),
        "refusal":reliability(response_units("Harmless","refusal")),
        "refusal_appropriateness":reliability(response_units("Harmless","refusal_appropriateness")),
        "truthfulness":reliability(response_units("Honest","truthfulness")),
        "informativeness":reliability(response_units("Honest","informativeness")),
        "unsupported_visual_claim":reliability(response_units("Honest","unsupported_visual_claim"))},
    "interpretation":"This pilot estimates workflow/rubric agreement only. It does not validate automated judges or estimate OrchCircuit performance on unseen data."}
json_out=OUTDIR/"figure3a_pilot_analysis.json"
json_out.write_text(json.dumps(results,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
keys=["helpfulness_pairwise_selected_vs_baseline","safety","refusal","refusal_appropriateness","truthfulness","informativeness","unsupported_visual_claim"]
labels=["Helpfulness pair","Safety","Refusal","Refusal appropriateness","Truthfulness","Informativeness","Unsupported visual claim"]
rels=[results["human_agreement"][k] for k in keys]
vals=[r["pairwise_agreement"]*100 for r in rels]
fig=plt.figure(figsize=(15,7.3),layout="constrained")
gs=fig.add_gridspec(1,3,width_ratios=[0.9,1.25,1.0])
ax0=fig.add_subplot(gs[0,0]); ax0.axis("off")
ax0.set_title("Reference-set flow (pilot)",loc="left",fontweight="bold",fontsize=14)
boxes=[("45 Ch3Ef development cases","3 independent raters"),("15 Helpful cases","15 pairwise comparisons"),("15 Harmless cases","30 response-level units"),("15 Honest cases","30 response-level units")]
y=.84
for i,(title,sub) in enumerate(boxes):
    color=["#dbeafe","#dcfce7","#fef3c7","#ffe4e6"][i]
    ax0.text(.5,y,title+"\n"+sub,ha="center",va="center",fontsize=10.5,bbox=dict(boxstyle="round,pad=.65",facecolor=color,edgecolor="#475569",linewidth=1.1),transform=ax0.transAxes)
    if i<len(boxes)-1: ax0.annotate("",xy=(.5,y-.105),xytext=(.5,y-.055),xycoords=ax0.transAxes,arrowprops=dict(arrowstyle="->",lw=1.5,color="#475569"))
    y-=.22
ax0.text(.5,.035,"Development-only; examples were used in prior OrchCircuit work.",ha="center",va="bottom",fontsize=8.5,color="#9f1239",wrap=True,transform=ax0.transAxes)
ax=fig.add_subplot(gs[0,1])
bars=ax.barh(labels[::-1],vals[::-1],color="#2563eb",edgecolor="white")
ax.set_xlim(0,115); ax.set_xticks([0,20,40,60,80,100]); ax.set_xlabel("Pairwise inter-rater agreement (%)")
ax.set_title("Agreement by rating field",loc="left",fontweight="bold",fontsize=14)
ax.grid(axis="x",alpha=.2); ax.set_axisbelow(True)
for b in bars: ax.text(min(b.get_width()+1,104),b.get_y()+b.get_height()/2,f"{b.get_width():.1f}%",va="center",fontsize=9)
ax2=fig.add_subplot(gs[0,2]); ax2.axis("off"); ax2.set_ylim(ax.get_ylim()); ax2.set_title("Unanimity and Fleiss κ",loc="left",fontweight="bold",fontsize=14)
for i,rel in enumerate(rels[::-1]):
    k=rel["fleiss_kappa"]; kt="κ undefined*" if k is None else f"κ={k:.3f}"
    n_unanimous=rel.get("unanimous_units"); n_units=rel.get("units")
    ax2.text(0.02,i,f"{n_unanimous}/{n_units} unanimous   {kt}",va="center",fontsize=9)
ax2.text(.02,-.08,"*No label variation",fontsize=8,color="#475569",transform=ax2.transAxes)
fig.suptitle("Figure 3a. Human rubric pilot: sample flow and rater agreement",fontsize=16,fontweight="bold")
fig.text(.5,-.015,"Descriptive pilot only; skewed categories and small per-criterion samples preclude confirmatory validity claims.",ha="center",fontsize=9,color="#9f1239")
png=OUTDIR/"figure3a_pilot_sample_flow_agreement.png"
fig.savefig(png,dpi=220,bbox_inches="tight")
print(json.dumps({"figure":str(png),"analysis":str(json_out),"cases":len(cases),"agreement":results["human_agreement"]},indent=2))
