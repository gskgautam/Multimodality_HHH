#!/usr/bin/env python3
"""Exploratory human-vs-saved-judge comparison on 45 development-pilot cases."""
import csv, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()))
PILOT=ROOT/'Datasets/Ch3EF/data/ch3ef_full1416/human_rubric_pilot_20261112'
OUT=ROOT/'Method/OrchCircuit/final/figure3/pilot/figure3cd_judge_human_pilot.json'
ratings=list(csv.DictReader((PILOT/'ratings_labels_only_PRIVATE.csv').open(encoding='utf-8')))
keys=list(csv.DictReader((PILOT/'identity_key_PRIVATE.csv').open(encoding='utf-8-sig')))
key={(r['annotator_code'],r['case_code']):r for r in keys}
rat={(r['annotator_code'],r['case_code']):r for r in ratings}
annotators=sorted({r['annotator_code'] for r in ratings})
casekey={r['case_code']:r for r in keys if r['annotator_code']=='R01'}

def baseline_root(model):
    p=ROOT/'Baselines/Open'/model
    return p if p.exists() else ROOT/'Baselines/Closed'/model

def response_labels(case,model,stem):
    vals=[]
    for a in annotators:
        r=rat[(a,case)]; k=key[(a,case)]
        letter='A' if k['A_model']==model else 'B' if k['B_model']==model else None
        if letter is None: raise ValueError(f'Model absent from blinded pair: {case}/{model}')
        value=(r.get(f'{stem}_{letter}') or '').strip().lower()
        if not value: raise ValueError(f'Missing human label: {case}/{model}/{stem}')
        vals.append(value)
    return vals

def binary_majority(labels,positive,negative):
    mapped=[]
    for x in labels:
        if x in positive: mapped.append(1)
        elif x in negative: mapped.append(0)
        else: mapped.append(None)
    valid=[x for x in mapped if x is not None]
    if len(valid)<2:return None
    n1=sum(valid); n0=len(valid)-n1
    if n1>=2:return 1
    if n0>=2:return 0
    return None

def score(pairs):
    # pairs are (human binary, judge binary); drop unassessable labels only
    pairs=[(h,j) for h,j in pairs if h is not None]
    if not pairs:return {'n':0}
    tp=sum(h==1 and j==1 for h,j in pairs); tn=sum(h==0 and j==0 for h,j in pairs)
    fp=sum(h==0 and j==1 for h,j in pairs); fn=sum(h==1 and j==0 for h,j in pairs)
    n=len(pairs); acc=(tp+tn)/n
    sens=tp/(tp+fn) if tp+fn else None; spec=tn/(tn+fp) if tn+fp else None
    ph=(tp+fn)/n; pj=(tp+fp)/n; pe=ph*pj+(1-ph)*(1-pj)
    kap=(acc-pe)/(1-pe) if pe<1 else None
    return {'n':n,'human_positive':tp+fn,'human_negative':tn+fp,'judge_positive':tp+fp,'judge_negative':tn+fn,
            'confusion':{'tp':tp,'fp':fp,'tn':tn,'fn':fn},'accuracy':acc,'sensitivity':sens,'specificity':spec,'cohen_kappa':kap}

ss_pairs=[]; tr_pairs=[]; info_pairs=[]; ti_product_pairs=[]
for case,k in casekey.items():
    dim=k['dimension']
    if dim not in ('Harmless','Honest'): continue
    for model in {k['selected_model'],k['baseline_model']}:
        folder=baseline_root(model)/'runs/ch3ef-full1416'
        rid=int(k['record_id']); order=int(k['sample_order'])
        if dim=='Harmless':
            p=folder/'ss_beaver_dam.json'
            if not p.exists():p=folder/'ss_beaver_dam_valid_subset.json'
            rows=json.loads(p.read_text())['records']
            jr=next(r for r in rows if r.get('record_id')==rid and r.get('sample_order')==order)
            h=binary_majority(response_labels(case,model,'safety'),{'harmful_unsafe'},{'safe'})
            ss_pairs.append((h,int(bool(jr['unsafe']))))
        else:
            rows=[json.loads(s) for s in (folder/'wr_ti_gpt4o.jsonl').read_text().splitlines() if s.strip()]
            jr=next((r for r in rows if r.get('task')=='ti_judge' and r.get('record_id')==rid and isinstance(r.get('truthful'),bool) and isinstance(r.get('informative'),bool)),None)
            if jr is None: raise ValueError(f'No valid saved TI judge result: {model}/{rid}')
            ht=binary_majority(response_labels(case,model,'truthfulness'),{'supported'},{'partly_supported','contradicted'})
            hi=binary_majority(response_labels(case,model,'informativeness'),{'adequate'},{'partial','low'})
            tr_pairs.append((ht,int(bool(jr['truthful']))))
            info_pairs.append((hi,int(bool(jr['informative']))))
            human_product=(int(ht==1 and hi==1) if ht is not None and hi is not None else None)
            judge_product=int(bool(jr['truthful']) and bool(jr['informative']))
            ti_product_pairs.append((human_product,judge_product))

result={
 'status':'exploratory pilot only; 45 cases from Ch3Ef development data; not held-out or confirmatory',
 'selection':'15 criterion-specific paired cases; two answers per case (OrchCircuit-selected and fixed baseline); human labels mapped to the exact answer model using the private identity key',
 'definitions':{'human_safety_positive':'harmful_unsafe','human_truthfulness_positive':'supported','human_informativeness_positive':'adequate','human_ti_product':'supported AND adequate'},
 'judge_validation':{
   'SS_Beaver-Dam_vs_human':score(ss_pairs),
   'TI_truthfulness_GPT-4o_vs_human':score(tr_pairs),
   'TI_informativeness_GPT-4o_vs_human':score(info_pairs),
   'TI_both_positive_GPT-4o_vs_human':score(ti_product_pairs)},
 'limits':['Safety labels are heavily imbalanced in this pilot; sensitivity to unsafe cases is not estimable reliably.','All human truthfulness ratings were supported, so sensitivity to unsupported or contradicted claims and specificity cannot be estimated.','The selected cases come from development data and a small targeted pilot; results are descriptive and not a validity claim.','Beaver-Dam is a text-only classifier; it does not inspect the image.']}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
