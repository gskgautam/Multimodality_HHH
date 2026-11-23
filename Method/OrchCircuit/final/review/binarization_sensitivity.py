import json
from pathlib import Path
import numpy as np
E=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()) / 'Method/OrchCircuit/final'); ns={}
src=(E/'review/inter_judge_agreement.py').read_text(); exec(compile(src.split("out = {'judge_vs_human_prompt_cluster'")[0],'ija','exec'),ns)
L,maj,kc,LAB=ns['L'],ns['maj'],ns['kappa_ci'],ns['LAB']
alts={'TI_truth|partly=positive':('TI_truth',{c:('pos' if h in('supported','partly_supported') else 'neg') for c,h in maj['TT'].items() if h in('supported','contradicted','partly_supported')}),
      'SS|concerning=unsafe':('SS',{c:('pos' if h in('harmful_unsafe','concerning') else 'neg') for c,h in maj['SS'].items() if h in('harmful_unsafe','safe','concerning')}),
      'TI_info|partial=positive':('TI_info',{c:('pos' if h in('adequate','partial') else 'neg') for c,h in maj['TF'].items()})}
out={}
for name,(crit,H) in alts.items():
    for j,d in L[crit].items():
        keys=[k for k in H if k in d]
        if len(keys)<len(H)-3: continue
        r=kc(H,d,['pos','neg'],keys); out[f'{name}|{j}']=r
        print(name,j,r['n'],round(r['kappa'],3),[round(x,3) for x in r['ci95']])
(E/'fig2_judge_validity/f_frontier_judges/binarization_sensitivity.json').write_text(json.dumps(out,indent=1)+'\n')
