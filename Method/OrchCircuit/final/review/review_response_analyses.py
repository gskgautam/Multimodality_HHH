"""Review-response analyses (2026-11-23). Saved labels only; no API calls, no training.

1. Trivial routers on the identical grouped 5-fold OOF split as Table 1 (reviewer R7):
   (i)  in-fold best single model per criterion (chosen on the 4 training folds),
   (ii) uniform random among the in-fold top-3 models (exact expectation).
2. Self-preference split of the OrchCircuit image-access audit (reviewer R4):
   OrchCircuit rows split by whether the selected answer comes from the judge's own family.
Writes review/review_response_analyses.json and prints a summary.
"""
import json, sys, collections
from pathlib import Path
import numpy as np

EXP = (next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()) / 'Method/OrchCircuit/final')
sys.path.insert(0, str(EXP))
import run_selector_cv as rs  # noqa: E402  (reuses the exact Table 1 label loader)

by, orders, dim, per_metric, wr, ss, truth, info = rs.labels_and_orders()
oi = {o: i for i, o in enumerate(orders)}
M = rs.MODELS
folds = json.loads((EXP / 'runs/fullset_group5fold/folds.json').read_text())['orders_by_fold']
fold_of = {int(o): f for f, fo in enumerate(folds) for o in fo}


def per_model(metric, idx):
    """Score of every model on the item indices idx (higher is better for ranking)."""
    if metric == 'WR':
        return (wr[idx] == 0).mean(0)
    if metric == 'SS':
        return -(ss[idx] == 1).mean(0)
    return truth[idx].mean(0) * info[idx].mean(0)


out = {'routers': {}}
for metric in ('WR', 'SS', 'TI'):
    items = [o for o in per_metric[metric]]
    best_pick = {}; top3 = {}
    for f in range(len(folds)):
        train = np.array([oi[o] for o in items if fold_of[o] != f])
        sc = per_model(metric, train)
        order = np.argsort(-sc)
        best_pick[f] = int(order[0]); top3[f] = [int(x) for x in order[:3]]
    idx = np.array([oi[o] for o in items]); n = len(idx)
    fsel = np.array([fold_of[o] for o in items])
    res = {}
    for name in ('best_single_infold', 'random_top3_infold'):
        if name == 'best_single_infold':
            w = np.zeros((n, len(M)))
            for k, f in enumerate(fsel):
                w[k, best_pick[f]] = 1
        else:
            w = np.zeros((n, len(M)))
            for k, f in enumerate(fsel):
                w[k, top3[f]] = 1 / 3
        if metric == 'WR':
            res[name] = {'wins': float((w * (wr[idx] == 0)).sum()), 'ties': float((w * (wr[idx] == 1)).sum()),
                         'losses': float((w * (wr[idx] == 2)).sum())}
            res[name]['pct'] = 100 * res[name]['wins'] / n
            res[name]['net_pct'] = 100 * (res[name]['wins'] - res[name]['losses']) / n
        elif metric == 'SS':
            res[name] = {'unsafe': float((w * (ss[idx] == 1)).sum())}; res[name]['pct'] = 100 * res[name]['unsafe'] / n
        else:
            t = float((w * truth[idx]).sum()); i_ = float((w * info[idx]).sum()); b = float((w * (truth[idx] * info[idx])).sum())
            res[name] = {'truthful': t, 'informative': i_, 'both': b, 'product_pct': 100 * t * i_ / n / n, 'joint_pct': 100 * b / n}
        res[name]['picks_by_fold'] = {str(f): ([M[best_pick[f]]] if name == 'best_single_infold' else [M[x] for x in top3[f]]) for f in range(len(folds))}
    res['n'] = n
    out['routers'][metric] = res

# ---- self-preference split of the image-access audit ----
sel = {}
for line in (EXP / 'runs/fullset_group5fold/oof_selected_candidates.jsonl').read_text().splitlines():
    r = json.loads(line); sel[int(r['sample_order'])] = r['selected_candidate']
fam = lambda m: 'anthropic' if 'claude' in m else 'google' if 'gemini' in m else 'openai' if 'gpt' in m else 'open'
SRC = (next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()) / 'Baselines/Judge/image-access-audit/ch3ef-judge-image-access/judge_calls.jsonl')
latest = {}
for line in SRC.read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get('model_slug') != 'orchcircuit' or r.get('status') != 'success' or not isinstance(r.get('judge_output'), dict):
        continue
    if r.get('condition') not in ('image_present', 'image_omitted') or r.get('sample_order') is None:
        continue
    k = (int(r['sample_order']), r['judge_provider'], r['task'], r['condition'])
    if k not in latest or str(r.get('completed_utc', '')) > str(latest[k].get('completed_utc', '')):
        latest[k] = r
cells = collections.defaultdict(lambda: collections.defaultdict(list))
for (o, j, t, c), r in latest.items():
    if t != 'WR' or o not in sel:
        continue
    p = str(r['judge_output'].get('preference')).lower()
    v = 0 if p in ('tie', 't') else (1 if p == ('a' if r.get('candidate_first') else 'b') else -1)
    own = 'same_family' if fam(sel[o]) == j else 'other_family'
    cells[(j, own)][(o, c)] = v
sp = {}
for (j, own), d in cells.items():
    os_ = sorted({o for (o, c) in d if (o, 'image_present') in d and (o, 'image_omitted') in d})
    if not os_:
        continue
    img = np.array([d[(o, 'image_present')] for o in os_]); om = np.array([d[(o, 'image_omitted')] for o in os_])
    sp[f'{j}|{own}'] = {'n': len(os_), 'wr_all_image_pct': 100 * float((img == 1).mean()), 'wr_all_omitted_pct': 100 * float((om == 1).mean()),
                        'net_image_pct': 100 * float(img.mean()), 'net_omitted_pct': 100 * float(om.mean())}
out['self_preference_WR_image_audit'] = sp
out['selected_family_WR'] = dict(collections.Counter(fam(sel[o]) for o in per_metric['WR']))
(EXP / 'review').mkdir(exist_ok=True)
(EXP / 'review/review_response_analyses.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({k: {kk: {x: y for x, y in vv.items() if x != 'picks_by_fold'} if isinstance(vv, dict) else vv for kk, vv in v.items()} for k, v in out['routers'].items()}, indent=1))
print(json.dumps({m: out['routers'][m]['best_single_infold']['picks_by_fold'] for m in out['routers']}))
print(json.dumps({m: out['routers'][m]['random_top3_infold']['picks_by_fold'] for m in out['routers']}))
print(json.dumps(sp, indent=1))
