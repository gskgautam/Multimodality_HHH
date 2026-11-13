#!/usr/bin/env python3
"""Post-hoc tie-adjusted sensitivity for Figure 3e; primary analysis remains unchanged."""
import ast,csv,hashlib,json,re,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()) / "Method/OrchCircuit/final/figure3")
FORM_DIR=ROOT/"e/private/ratings_received"
MAP_PATH=ROOT/"e/private/reviewer_forms/blind_mapping_PRIVATE.json"
OUT=ROOT/"e"
RATERS=["R01","R02","R03"]
SELECTORS=["sel_wr","sel_ss","sel_ti"]
CONTROLS=["control_wr","control_ss","control_ti"]
N_BOOT=10000
SEED=20261113
NS={"m":"http://schemas.openxmlformats.org/spreadsheetml/2006/main","r":"http://schemas.openxmlformats.org/officeDocument/2006/relationships"}

# Reuse the exact workbook reader from the frozen main analysis without executing that script.
main=ROOT/"common/scripts/analyze_human_ratings_3e_f_PRIVATE.py"
tree=ast.parse(main.read_text())
reader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="read_xlsx")
exec(compile(ast.Module(body=[reader],type_ignores=[]),str(main),"exec"),globals())

mapping=json.loads(MAP_PATH.read_text())
votes=defaultdict(dict); hashes={}
for r in RATERS:
    form=FORM_DIR/f"{r}_Figure3e_3f_Blinded_Ratings.xlsx"
    hashes[r]=hashlib.sha256(form.read_bytes()).hexdigest()
    rows={str(x["Pair ID"]):x for x in read_xlsx(form)["WR pairs"]}
    meta=next(x for x in mapping["reviewers"] if x["rater"]==r)
    if len(rows)!=1800 or set(rows)!=set(meta["wr"]): raise ValueError(f"{r}: workbook/map IDs differ")
    for pid,info in meta["wr"].items():
        choice=rows[pid]["WR choice"]
        if choice=="Tie": outcome="Tie"
        elif choice=="Uncertain": outcome="Uncertain"
        else:
            side=info["sideA"] if choice=="A" else info["sideB"] if choice=="B" else None
            if side is None: raise ValueError(f"invalid choice {choice}")
            outcome="Policy win" if side==info["selector"] else "Control win"
        votes[(info["case_id"],info["selector"],info["control"])][r]=outcome
cases=sorted({k[0] for k in votes})
if len(cases)!=200 or len(votes)!=1800 or any(set(v)!=set(RATERS) for v in votes.values()):
    raise ValueError("Expected 200 prompts x 3 selectors x 3 controls x 3 raters")

def majority(values):
    c=Counter(values); top=max(c.values()); winners=[v for v,n in c.items() if n==top]
    return winners[0] if len(winners)==1 else "No majority"
consensus={k:majority([v[r] for r in RATERS]) for k,v in votes.items()}
rng=np.random.default_rng(SEED)
boot=rng.integers(0,len(cases),size=(N_BOOT,len(cases)))
rows=[]
for sel in SELECTORS:
    case_scores=[]; totals=Counter()
    for case in cases:
        score=0.0
        for ctrl in CONTROLS:
            outcome=consensus[(case,sel,ctrl)]; totals[outcome]+=1
            score += 1.0 if outcome=="Policy win" else 0.5 if outcome=="Tie" else 0.0
        case_scores.append(score/len(CONTROLS))
    values=np.asarray(case_scores)
    ci=np.percentile(values[boot].mean(axis=1)*100,[2.5,97.5]).tolist()
    rows.append({"policy":sel,"policy_name":{"sel_wr":"OrchCircuit-WR","sel_ss":"OrchCircuit-SS","sel_ti":"OrchCircuit-TI"}[sel],
      "wins":totals["Policy win"],"control_wins":totals["Control win"],"ties":totals["Tie"],
      "uncertain":totals["Uncertain"],"no_majority":totals["No majority"],"n_pairings":600,
      "tie_adjusted_preference_pct":float(values.mean()*100),"ci95_low_pct":float(ci[0]),"ci95_high_pct":float(ci[1])})
OUT.mkdir(parents=True,exist_ok=True)
with open(OUT/"tie_adjusted_preference.csv","w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
meta={"analysis":"post-hoc sensitivity; does not replace primary WR net-preference analysis",
 "n_prompts":200,"pairings_per_policy":600,"reviewers":3,"weights":{"policy_win":1,"tie":0.5,"control_win_uncertain_no_majority":0},
 "bootstrap":{"replicates":N_BOOT,"seed":SEED,"unit":"prompt"},"input_hashes":hashes,"results":rows}
(OUT/"tie_adjusted_preference.json").write_text(json.dumps(meta,indent=2))

# Secondary horizontal bar graph, centred on neutral preference (50%).
fig,ax=plt.subplots(figsize=(8.8,4.8),constrained_layout=True)
labels=[x["policy_name"] for x in rows]; y=np.arange(len(rows))
v=np.array([x["tie_adjusted_preference_pct"] for x in rows])
lo=np.array([x["ci95_low_pct"] for x in rows]); hi=np.array([x["ci95_high_pct"] for x in rows])
err=np.maximum(0,np.vstack((v-lo,hi-v)))
colors=["#25835b" if x>50 else "#bd4b4b" if x<50 else "#9299a1" for x in v]
ax.barh(y,v-50,left=50,color=colors,alpha=.84,height=.6,zorder=2)
ax.errorbar(v,y,xerr=err,fmt="none",ecolor="#343a40",capsize=4,zorder=3)
ax.axvline(50,color="#20252a",lw=1.1,linestyle="--")
ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlim(40,60)
ax.set_xlabel("Tie-adjusted human preference (%)")
ax.set_title("Figure 3e: tie-adjusted WR sensitivity",fontweight="bold")
ax.grid(axis="x",alpha=.22,zorder=0)
for yy,x,u in zip(y,v,hi): ax.text(u+.25,yy,f"{x:.1f}%",va="center",fontsize=9)
ax.text(.5,-.18,"Post hoc secondary view: win = 1, tie = 0.5, loss/uncertain/no-majority = 0; 95% prompt-bootstrap intervals.",
        transform=ax.transAxes,ha="center",fontsize=8.5)
fig.savefig(OUT/"figure_3e_tie_adjusted.png",dpi=300,bbox_inches="tight")
fig.savefig(OUT/"figure_3e_tie_adjusted.pdf",bbox_inches="tight"); plt.close(fig)

# Preserve the current report before appending this sensitivity result.
arc=ROOT/"archive_PRIVATE/pre_tie_adjusted_2026-11-13_PRIVATE/3e"
arc.mkdir(parents=True,exist_ok=True)
report=OUT/"RESULTS.md"
if report.exists() and not (arc/"RESULTS.md").exists(): shutil.copy2(report,arc/"RESULTS.md")
text=report.read_text()
marker="## Exploratory tie-adjusted WR sensitivity (post hoc)"
if marker in text: text=text[:text.index(marker)].rstrip()+"\n"
section=[marker,"",
"Secondary sensitivity only; it does not replace the primary nine pairwise net-preference results. Each selector is pooled across its three fixed controls (600 pairings). A selector win contributes 1, a tie 0.5, and a control win, uncertain, or no-majority outcome 0. The 95% interval resamples the 200 prompts as clusters (10,000 replicates, seed 20261113).","",
"| OrchCircuit policy | Wins | Control wins | Ties | Uncertain | No majority | Tie-adjusted preference / 600 (95% CI) |",
"|---|---:|---:|---:|---:|---:|---:|"]
for x in rows:
    section.append(f"| {x['policy_name']} | {x['wins']} | {x['control_wins']} | {x['ties']} | {x['uncertain']} | {x['no_majority']} | {x['tie_adjusted_preference_pct']:.1f}% ({x['ci95_low_pct']:.1f}–{x['ci95_high_pct']:.1f}) |")
section += ["","All three point estimates are near 50%; this is a sensitivity analysis, not evidence of a large or established helpfulness advantage.",""]
report.write_text(text.rstrip()+"\n\n"+"\n".join(section))
print(json.dumps({"results":rows,"hashes":hashes,"graph":str(OUT/"figure_3e_tie_adjusted.png")},indent=2))
