"""Resolve open items O1 (WR reference / position bias) and O5 (joint TI as primary).

Saved labels and saved OOF selections only; no API calls, no training.
Writes review/resolve_o1_o5.json.
"""
import json, random, sys
from pathlib import Path
import numpy as np

R = (next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd())); E = R / 'Method/OrchCircuit/final'
sys.path.insert(0, str(E))
import run_selector_cv as rs  # exact Table 1 label loader

by, orders, dim, per, wr, ss, tr, inf = rs.labels_and_orders()
oi = {o: i for i, o in enumerate(orders)}; M = rs.MODELS; mi = {m: j for j, m in enumerate(M)}
group = json.loads((E / 'runs/fullset_group5fold/folds.json').read_text())['group_ids']
out = {}
jl = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

# ---------------- O1: WR reference and position bias ----------------
# All WR judge scripts build candidates=[(candidate, ans), ("gpt4o_baseline", gpt4o_ans)] and shuffle with
# random.Random(record_id + 90210); the order depends only on record_id, so it is recoverable.
def cand_first(rid):
    c = ['cand', 'ref']; random.Random(int(rid) + 90210).shuffle(c); return c[0] == 'cand'

wo = per['WR']; idx = np.array([oi[o] for o in wo])
first = np.array([cand_first(by[o]['id']) for o in wo])
W = wr[idx]  # 0 win, 1 tie, 2 loss ; shape (498, 24)
def rates(mask):
    x = W[mask]; return {'n_items': int(mask.sum()), 'win': 100 * float((x == 0).mean()), 'tie': 100 * float((x == 1).mean()), 'loss': 100 * float((x == 2).mean())}
pos = {'candidate_first': rates(first), 'candidate_second': rates(~first)}
gids = np.array([int(group[str(o)]) for o in wo]); ug = np.unique(gids); mem = {g: np.where(gids == g)[0] for g in ug}
rng = np.random.default_rng(20261123); bd = []; bl = []
for _ in range(10000):
    ix = np.concatenate([mem[g] for g in rng.choice(ug, len(ug))]); f = first[ix]; x = W[ix]
    if f.all() or (~f).all():
        continue
    bd.append(100 * ((x[f] == 0).mean() - (x[~f] == 0).mean())); bl.append(100 * ((x[f] == 2).mean() - (x[~f] == 2).mean()))
pos['win_first_minus_second_pp'] = pos['candidate_first']['win'] - pos['candidate_second']['win']
pos['win_diff_ci95'] = [float(np.quantile(bd, .025)), float(np.quantile(bd, .975))]
pos['loss_first_minus_second_pp'] = pos['candidate_first']['loss'] - pos['candidate_second']['loss']
pos['loss_diff_ci95'] = [float(np.quantile(bl, .025)), float(np.quantile(bl, .975))]
# order-balanced WR: average of the first-position and second-position win rates (removes position imbalance)
bal = {m: 50 * float((W[first, j] == 0).mean() + (W[~first, j] == 0).mean()) for m, j in mi.items()}
sel = {int(x['sample_order']): x['selected_candidate'] for x in jl(E / 'runs/fullset_group5fold/oof_selected_candidates.jsonl')}
oc = np.array([mi[sel[o]] for o in wo]); ow = W[np.arange(len(wo)), oc]
bal['OrchCircuit'] = 50 * float((ow[first] == 0).mean() + (ow[~first] == 0).mean())
pos['order_balanced_wr_pct'] = dict(sorted(bal.items(), key=lambda kv: -kv[1])[:6])
gpt_fam = [m for m in M if m.startswith('gpt')]
pos['gpt_family_wr_pct'] = {m: 100 * float((W[:, mi[m]] == 0).mean()) for m in gpt_fam}
out['O1'] = pos

# ---------------- O5: joint TI everywhere ----------------
ti = per['TI']; tix = np.array([oi[o] for o in ti]); J = tr[tix] * inf[tix]
tg = np.array([int(group[str(o)]) for o in ti]); tug = np.unique(tg); tmem = {g: np.where(tg == g)[0] for g in tug}
def joint_of(run_dir):
    s = {int(x['sample_order']): x.get('ti_selected_candidate') or x['selected_candidate'] for x in jl(run_dir / 'oof_selected_candidates.jsonl')}
    return J[np.arange(len(ti)), np.array([mi[s[o]] for o in ti])]
A = E / 'fig6_robustness/a_ablations/runs'; seeds = [20261111, 20261112, 20261113]
full = {s: joint_of(A / f'full_seed{s}_matched') for s in seeds}
out['O5_4c_joint_by_seed_pct'] = {str(s): 100 * float(full[s].mean()) for s in seeds}
abl = {}
alpha = 0.05 / 9  # same family size as the 9 Figure 4a comparisons
for v in ('no_set_attention', 'no_candidate_identity', 'mean_token_pool'):
    va = {s: joint_of(A / f'{v}_seed{s}_matched') for s in seeds}
    d = np.mean([va[s].mean() - full[s].mean() for s in seeds]) * 100
    boots = []; rg = np.random.default_rng(20261113)
    for _ in range(10000):
        ix = np.concatenate([tmem[g] for g in rg.choice(tug, len(tug))])
        boots.append(100 * np.mean([va[s][ix].mean() - full[s][ix].mean() for s in seeds]))
    abl[v] = {'full_mean_pct': 100 * float(np.mean([full[s].mean() for s in seeds])), 'ablation_mean_pct': 100 * float(np.mean([va[s].mean() for s in seeds])),
              'delta_pp': float(d), 'adj_ci': [float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))]}
out['O5_4a_joint'] = abl
out['O5_table1_joint'] = {m: int(J[:, j].sum()) for m, j in mi.items()}
out['O5_table1_joint']['OrchCircuit'] = int(joint_of(E / 'runs/fullset_group5fold').sum())

# 4e joint on the 600-record development cohort (same loader as e_analyze_joint_hhh.py)
P = R / 'Datasets/Ch3EF/data/ch3ef_full1416/joint_hhh_labels_600_PRIVATE'
o6 = list(map(int, json.loads((P / 'sample_manifest.json').read_text())['orders'])); ix6 = {o: i for i, o in enumerate(o6)}; N = len(o6)
w6 = np.full((N, 24), -1); s6 = w6.copy(); t6 = w6.copy(); i6 = w6.copy()
lr = {(int(x['sample_order']), x['task']): x for x in jl(P / 'gpt4o_joint_labels_v2.jsonl') if not x.get('error') and isinstance(x.get('labels'), dict)}
for o in o6:
    for task in ('wr', 'ti'):
        x = lr[(o, task)]
        for c, m in x['candidate_map'].items():
            v = x['labels'][c]; j = mi[m]
            if task == 'wr':
                w6[ix6[o], j] = int(v['preference'] == 'win')
            else:
                t6[ix6[o], j] = int(bool(v['truthful'])); i6[ix6[o], j] = int(bool(v['informative']))
for x in jl(P / 'beaver_dam_ss_600x24.jsonl'):
    o = int(x['sample_order'])
    if o in ix6:
        s6[ix6[o], mi[x['model']]] = int(bool(x['unsafe']))
assert all(np.all(a >= 0) for a in (w6, s6, t6, i6))
ch = np.array([mi[sel[o]] for o in o6]); r6 = np.arange(N)
sysv = {m: (100 * w6[:, j].mean(), 100 * s6[:, j].mean(), 100 * (t6[:, j] * i6[:, j]).mean()) for m, j in mi.items()}
sysv['OrchCircuit'] = (100 * w6[r6, ch].mean(), 100 * s6[r6, ch].mean(), 100 * (t6[r6, ch] * i6[r6, ch]).mean())
front = [a for a, va in sysv.items() if not any(vb[0] >= va[0] and vb[1] <= va[1] and vb[2] >= va[2] and vb != va for b, vb in sysv.items() if b != a)]
g6 = np.array([str(group[str(o)]) for o in o6]); gk = np.unique(g6); gm = {g: np.where(g6 == g)[0] for g in gk}
jg = mi['gemini-4-argon']; rg = np.random.default_rng(20261113); bt = []
for _ in range(10000):
    ix = np.concatenate([gm[g] for g in rg.choice(gk, len(gk))])
    bt.append(100 * ((t6[ix, ch[ix]] * i6[ix, ch[ix]]).mean() - (t6[ix, jg] * i6[ix, jg]).mean()))
out['O5_4e_joint'] = {'joint_pct': {m: v[2] for m, v in sysv.items()}, 'pareto_with_joint': front,
                      'orch_minus_gemini_joint_pp': sysv['OrchCircuit'][2] - sysv['gemini-4-argon'][2], 'ci95': [float(np.quantile(bt, .025)), float(np.quantile(bt, .975))],
                      'orch_joint_count': int((t6[r6, ch] * i6[r6, ch]).sum()), 'gemini_joint_count': int((t6[:, jg] * i6[:, jg]).sum())}
(E / 'review/resolve_o1_o5.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out, indent=1))
