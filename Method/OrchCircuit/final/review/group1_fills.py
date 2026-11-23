"""Group 1 fills (saved data only; no API calls, no training). Writes review/group1_fills.json.
(a) Fig. 2a: joint-TI paired group-bootstrap CIs vs each of the 24 models, Bonferroni over 72 (same family as Table 1).
(b) Fig. 3c: joint-TI image-present vs image-omitted per model x judge (image-access audit).
(c) Fig. 3d: joint-TI difference-in-differences OrchCircuit vs Qwen2.5-VL 7B (same pairing/groups/seed as 4b script).
(d) Fig. 4e: order-balanced WR with 95% group-bootstrap CIs for all 25 systems.
"""
import json, random, sys, collections
from pathlib import Path
import numpy as np

R = (next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd())); E = R / 'Method/OrchCircuit/final'
sys.path.insert(0, str(E)); import run_selector_cv as rs  # noqa
by, orders, dim, per, wr, ss, tr, inf = rs.labels_and_orders(); oi = {o: i for i, o in enumerate(orders)}; M = rs.MODELS
group = json.loads((E / 'runs/fullset_group5fold/folds.json').read_text())['group_ids']
jl = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
sel = {int(x['sample_order']): x for x in jl(E / 'runs/fullset_group5fold/oof_selected_candidates.jsonl')}
out = {}; B = 20000


def gboot(n_items, gids, stat, reps=B, seed=20261123):
    ug = np.unique(gids); mem = {g: np.where(gids == g)[0] for g in ug}; rng = np.random.default_rng(seed); v = np.empty(reps)
    for b in range(reps):
        ix = np.concatenate([mem[g] for g in rng.choice(ug, len(ug))]); v[b] = stat(ix)
    return v

# (a) joint TI vs all 24
ti = per['TI']; tix = np.array([oi[o] for o in ti]); J = tr[tix] * inf[tix]
oc = np.array([M.index(sel[o].get('ti_selected_candidate') or sel[o]['selected_candidate']) for o in ti]); oj = J[np.arange(len(ti)), oc]
tg = np.array([int(group[str(o)]) for o in ti]); a = 0.05 / 72
ug = np.unique(tg); mem = {g: np.where(tg == g)[0] for g in ug}; rng = np.random.default_rng(20261111)
draws = np.empty((B, 24))
for b in range(B):
    ix = np.concatenate([mem[g] for g in rng.choice(ug, len(ug))]); draws[b] = 100 * (oj[ix].mean() - J[ix].mean(0))
res = {}
for j, m in enumerate(M):
    lo, hi = np.quantile(draws[:, j], [a / 2, 1 - a / 2])
    res[m] = {'delta_pp': float(100 * (oj.mean() - J[:, j].mean())), 'adj_ci': [float(lo), float(hi)], 'beats': bool(lo > 0)}
out['a_fig2a_joint_ti_vs_24'] = {'orch_joint_pct': float(100 * oj.mean()), 'n_beaten': sum(v['beats'] for v in res.values()),
                                 'not_beaten': [m for m, v in res.items() if not v['beats']], 'per_model': res}

# (b)+(c) image-access joint TI
SRC = R / 'Baselines/Judge/image-access-audit/ch3ef-judge-image-access/judge_calls.jsonl'
latest = {}
for line in SRC.read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get('task') != 'TI' or r.get('status') != 'success' or not isinstance(r.get('judge_output'), dict) or r.get('sample_order') is None:
        continue
    k = (r['model_slug'], int(r['sample_order']), r['judge_provider'], r['condition'])
    if k not in latest or str(r.get('completed_utc', '')) > str(latest[k].get('completed_utc', '')):
        latest[k] = r
cell = collections.defaultdict(dict)
for (m, o, j, c), r in latest.items():
    try:
        cell[(m, j)].setdefault(o, {})[c] = int(bool(r['judge_output']['truthful']) and bool(r['judge_output']['informative']))
    except (KeyError, TypeError):
        pass
fig3c = {}
for (m, j), d in sorted(cell.items()):
    os_ = [o for o, x in d.items() if 'image_present' in x and 'image_omitted' in x]
    if not os_:
        continue
    p = np.mean([d[o]['image_present'] for o in os_]); q = np.mean([d[o]['image_omitted'] for o in os_])
    fig3c[f'{m}|{j}'] = {'pairs': len(os_), 'joint_image_pct': 100 * p, 'joint_omitted_pct': 100 * q, 'delta_pp': 100 * (p - q)}
out['b_fig3c_joint_ti'] = fig3c
fig3d = {}
for j in ('anthropic', 'google'):
    A = cell.get(('orchcircuit', j), {}); Q = cell.get(('qwen2.5-vl-7b', j), {})
    os_ = sorted(o for o in A if o in Q and len(A[o]) == 2 and len(Q[o]) == 2)
    a_p = np.array([A[o]['image_present'] for o in os_]); a_o = np.array([A[o]['image_omitted'] for o in os_])
    q_p = np.array([Q[o]['image_present'] for o in os_]); q_o = np.array([Q[o]['image_omitted'] for o in os_])
    gids = np.array([int(group[str(o)]) for o in os_])
    st = lambda ix: 100 * ((a_p[ix].mean() - a_o[ix].mean()) - (q_p[ix].mean() - q_o[ix].mean()))
    v = gboot(len(os_), gids, st, reps=10000, seed=20261113)
    fig3d[j] = {'common_prompts': len(os_), 'orch_shift_pp': 100 * (a_p.mean() - a_o.mean()), 'qwen_shift_pp': 100 * (q_p.mean() - q_o.mean()),
                'did_pp': float(st(np.arange(len(os_)))), 'ci95': [float(np.quantile(v, .025)), float(np.quantile(v, .975))]}
out['c_fig3d_joint_ti_did'] = fig3d

# (d) order-balanced WR with CIs
def cand_first(rid):
    c = ['cand', 'ref']; random.Random(int(rid) + 90210).shuffle(c); return c[0] == 'cand'
wo = per['WR']; widx = np.array([oi[o] for o in wo]); W = (wr[widx] == 0).astype(float)
first = np.array([cand_first(by[o]['id']) for o in wo]); ow = W[np.arange(len(wo)), np.array([M.index(sel[o]['selected_candidate']) for o in wo])]
wg = np.array([int(group[str(o)]) for o in wo]); cols = {m: W[:, j] for j, m in enumerate(M)}; cols['OrchCircuit'] = ow
bal = lambda x, ix: 50 * (x[ix][first[ix]].mean() + x[ix][~first[ix]].mean())
fig4e = {}
for m, x in cols.items():
    v = gboot(len(wo), wg, lambda ix: bal(x, ix), reps=5000, seed=20261123)
    fig4e[m] = {'order_balanced_wr_pct': bal(x, np.arange(len(wo))), 'ci95': [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))]}
out['d_fig4e_order_balanced_wr'] = dict(sorted(fig4e.items(), key=lambda kv: -kv[1]['order_balanced_wr_pct']))
(E / 'review/group1_fills.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps({'a': {k: v for k, v in out['a_fig2a_joint_ti_vs_24'].items() if k != 'per_model'},
                  'gemini_opus': {m: out['a_fig2a_joint_ti_vs_24']['per_model'][m] for m in ('gemini-4-argon', 'claude-opus-5-5', 'qwen3.8-27b')},
                  'c': fig3d, 'd_top6': dict(list(out['d_fig4e_order_balanced_wr'].items())[:6])}, indent=1))
print('fig3c rows', len(fig3c))
