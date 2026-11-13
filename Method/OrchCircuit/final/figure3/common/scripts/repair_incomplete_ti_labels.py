#!/usr/bin/env python3
"""Repair the two incomplete TI outputs using small strict-schema candidate batches."""
import json,sys,time
from pathlib import Path
ROOT=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()));sys.path.insert(0,str(ROOT/"Method/OrchCircuit/final/figure3"))
import generate_joint_gpt4o_labels_v2 as g
import run_selector_cv as cv
CHUNK=4
def main():
 manifest=json.load(open(g.MANIFEST)); records={int(x["sample_order"]):x for x in json.load(open(g.DATA/"sample.json"))["records"]}
 maps={}
 for model in cv.MODELS:
  fam="Open" if model in cv.OPEN else "Closed";p=ROOT/"Baselines"/fam/model/"runs/ch3ef-full1416/responses.jsonl"
  maps[model]={int(x["sample_order"]):x for x in g.readjl(p) if x.get("sample_order") is not None}
 journal=g.readjl(g.OUT); done={(int(x["sample_order"]),x["task"]) for x in journal if not x.get("error")}
 unlogged=sum(max(0,len(x.get("attempt_errors",[]))-len(x.get("attempt_usages",[]))) for x in journal if x.get("error"))
 spent=g.SPENT_PRIOR_BOUND+sum(g.row_usage(x) for x in journal)+unlogged*g.MAX_PER_CALL
 missing=[o for o in manifest["orders"] if (int(o),"ti") not in done]
 print(json.dumps({"missing_ti_orders":missing,"total_estimated_usd":round(spent,6),"cap_usd":g.CAP}),flush=True)
 if len(missing)>2: raise SystemExit("Unexpectedly many incomplete TI rows; inspect journal first.")
 key=g.load_key(); codes=[chr(65+i) for i in range(24)]
 with g.OUT.open("a",encoding="utf-8") as f:
  for order in missing:
   order=int(order);rec=records[order];mapping,texts,anchor=g.build_candidates(order,rec,maps);images=g.image_content(rec)
   all_labels={};usages=[];errors=[]
   for start in range(0,len(codes),CHUNK):
    subset=codes[start:start+CHUNK];prompt,expected=g.prompt_for("ti",rec,texts,anchor,subset);last=None
    for attempt in range(3):
     if spent+g.MAX_PER_CALL>g.CAP: raise SystemExit("Spend cap reserve reached before repair request.")
     usage=None
     try:
      parsed,usage=g.call(key,"You are an impartial multimodal evaluator. Follow the task rubric exactly.",prompt,images,"ti",expected)
      labels=g.normalize("ti",parsed,expected,anchor)
      if set(labels)!=set(expected): raise ValueError("chunk output mismatch")
      all_labels.update(labels);spent+=usage["estimated_cost_usd"];usages.append(usage);last=usage;break
     except Exception as e:
      if usage is None: usage=getattr(e,"usage",None)
      if usage is None: usage={"estimated_cost_usd":g.MAX_PER_CALL,"accounting":"conservative_request_reserve_no_usage"}
      spent+=usage["estimated_cost_usd"];usages.append(usage);errors.append(str(e)[:220])
      if attempt<2: time.sleep(2**attempt)
    if last is None: raise SystemExit(f"Repair chunk failed for order={order}, ids={subset}: {errors[-1]}")
   if set(all_labels)!=set(codes): raise SystemExit(f"Still missing TI IDs for order {order}")
   row={"sample_order":order,"record_id":rec["id"],"dimension":rec["dimension"],"task":"ti","candidate_map":mapping,"anchor_id":anchor,"labels":all_labels,"usage":usages[-1],"attempt_usages":usages[:-1],"repair_errors":errors,"judge_model":g.MODEL,"temperature":0,"prompt_version":"joint-dev-v3-chunked-repair"}
   f.write(json.dumps(row,ensure_ascii=False)+"\n");f.flush()
   print(json.dumps({"repaired_order":order,"labels":len(all_labels),"total_estimated_usd":round(spent,6)}),flush=True)
 print("REPAIRED_ALL",flush=True)
if __name__=="__main__":main()
