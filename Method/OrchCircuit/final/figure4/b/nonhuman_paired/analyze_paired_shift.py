from pathlib import Path
import json,collections,csv
import numpy as np
ROOT=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()))
SRC=ROOT/"Baselines/Judge/image-access-audit/ch3ef-judge-image-access/judge_calls.jsonl"
FOLDS=ROOT/"Method/OrchCircuit/final/runs/fullset_group5fold/folds.json"
OUT=Path(__file__).resolve().parent
MODELS=("orchcircuit","qwen2.5-vl-7b"); TASKS=("WR","SS","TI"); JUDGES=("anthropic","google")
B=10000; SEED=20261113
def score(r,task):
 o=r["judge_output"]
 if task=="SS": return {"unsafe":int(bool(o["unsafe"]))}
 if task=="TI": return {"truthful":int(bool(o["truthful"])),"informative":int(bool(o["informative"]))}
 p=str(o["preference"]).lower()
 if p in ("tie","t"): return None
 return int(p==("a" if bool(r.get("candidate_first")) else "b"))
def agg(v,task):
 if task=="SS": return 100*np.mean([x["unsafe"] for x in v])
 if task=="TI": return 100*np.mean([x["truthful"] for x in v])*np.mean([x["informative"] for x in v])
 z=[x for x in v if x is not None]; return 100*np.mean(z)
def main():
 latest={}
 for line in SRC.read_text().splitlines():
  if not line.strip():continue
  r=json.loads(line)
  if r.get("status")!="success" or not isinstance(r.get("judge_output"),dict):continue
  if r.get("model_slug") not in MODELS or r.get("judge_provider") not in JUDGES or r.get("task") not in TASKS:continue
  if r.get("condition") not in ("image_present","image_omitted") or r.get("sample_order") is None:continue
  k=(r["model_slug"],int(r["sample_order"]),r["judge_provider"],r["task"],r["condition"])
  if k not in latest or str(r.get("completed_utc",""))>str(latest[k].get("completed_utc","")):latest[k]=r
 group=json.loads(FOLDS.read_text())["group_ids"];rng=np.random.default_rng(SEED);out=[]
 for task in TASKS:
  for judge in JUDGES:
   by={}
   for model in MODELS:
    tmp=collections.defaultdict(dict)
    for (m,o,j,t,c),r in latest.items():
     if (m,j,t)!=(model,judge,task):continue
     try:v=score(r,task)
     except (KeyError,TypeError):continue
     tmp[o][c]=v
    by[model]={o:x for o,x in tmp.items() if "image_present" in x and "image_omitted" in x}
   orders=sorted(set(by[MODELS[0]])&set(by[MODELS[1]]))
   if len(orders)<20:continue
   vals={}
   for model in MODELS:
    vals[model]={c:[by[model][o][c] for o in orders] for c in ("image_present","image_omitted")}
    vals[model]["shift"]=agg(vals[model]["image_present"],task)-agg(vals[model]["image_omitted"],task)
   gids=[int(group[str(o)]) for o in orders];ug=sorted(set(gids));members={g:[i for i,x in enumerate(gids) if x==g] for g in ug}
   pos={o:i for i,o in enumerate(orders)};boot=np.empty(B)
   for b in range(B):
    selected=rng.choice(ug,size=len(ug),replace=True);ix=[i for g in selected for i in members[int(g)]]
    shifts={}
    for model in MODELS:
     shifts[model]=agg([vals[model]["image_present"][i] for i in ix],task)-agg([vals[model]["image_omitted"][i] for i in ix],task)
    boot[b]=shifts["orchcircuit"]-shifts["qwen2.5-vl-7b"]
   out.append({"metric":task,"judge":judge,"n_common_prompts":len(orders),"n_query_image_groups":len(ug),"OrchCircuit_image_minus_omitted_pp":vals["orchcircuit"]["shift"],"Qwen2.5_VL_image_minus_omitted_pp":vals["qwen2.5-vl-7b"]["shift"],"difference_in_shifts_Orch_minus_Qwen_pp":vals["orchcircuit"]["shift"]-vals["qwen2.5-vl-7b"]["shift"],"cluster_bootstrap_95_low":float(np.quantile(boot,.025)),"cluster_bootstrap_95_high":float(np.quantile(boot,.975)),"bootstrap_replicates":B})
 with (OUT/"paired_shift_contrast.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 md=["# Figure 4b - Paired image-access sensitivity contrast (no human ratings)","","For each metric and judge, compare the image-present minus image-omitted score shift for OrchCircuit's selected answer with the same shift for Qwen2.5-VL 7B on the exact same prompt/image records. Qwen is the common reference because the saved audit has paired coverage across all three metrics and both judges. The estimand is the difference-in-differences: OrchCircuit shift minus Qwen shift. For SS, the score is unsafe rate, so a positive shift means more judged unsafe.","","This tests relative automated-judge sensitivity to image access. It does not test answer correctness, human agreement, or whether OrchCircuit uses visual evidence better.","","| Metric | Judge | Common prompts | Query/image groups | OrchCircuit shift (pp) | Qwen shift (pp) | Difference in shifts (pp; 95% group-bootstrap CI) |","|---|---|---:|---:|---:|---:|---:|"]
 for x in out:md.append(f'| {x["metric"]} | {x["judge"]} | {x["n_common_prompts"]} | {x["n_query_image_groups"]} | {x["OrchCircuit_image_minus_omitted_pp"]:+.2f} | {x["Qwen2.5_VL_image_minus_omitted_pp"]:+.2f} | {x["difference_in_shifts_Orch_minus_Qwen_pp"]:+.2f} [{x["cluster_bootstrap_95_low"]:+.2f}, {x["cluster_bootstrap_95_high"]:+.2f}] |')
 md+=["","WR is candidate win rate among decisive non-tied judgments, SS is unsafe rate, and TI is the product of truthful and informative marginal rates, matching the audit definitions. Intervals use 10,000 bootstrap resamples of connected query/image groups.","","This is a post-hoc analysis of the existing 80-item-per-dimension audit and development-exposed Ch3EF records; it is exploratory, not confirmatory. Qwen is a common reference, not the strongest baseline on every metric. Judge effects should not be averaged as interchangeable measurements. This can supplement, but cannot replace, human validation.","","Reproduce by running python3 Method/OrchCircuit/final/figure4/b/nonhuman_paired/analyze_paired_shift.py. It reads existing judge calls and grouped-fold metadata; it makes no model or API calls."]
 (OUT/"RESULTS.md").write_text("\n".join(md)+"\n")
 print("\n".join(md))
if __name__=="__main__":main()
