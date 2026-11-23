"""Round-2 reviewer M1/minor 1-2: agreement among judges, judge-vs-human with prompt-cluster bootstrap,
and WR A/B rates, on the 569 Fig. 4 human-rated items. Saved labels only; no API calls.
Writes fig2_judge_validity/f_frontier_judges/inter_judge_agreement.json."""
import collections, csv, importlib.util, itertools, json
from pathlib import Path
import numpy as np

R = (next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd())); E = R / 'Method/OrchCircuit/final'
P = R / 'Datasets/Ch3EF/data/ch3ef_full1416/human_validation_figure3a_d_20261112_PRIVATE'
OUT = E / 'fig2_judge_validity/f_frontier_judges'
sp = importlib.util.spec_from_file_location('hv', E / 'shared/human_rating_pipeline/scripts/analyze_figure3a_d_human_validation.py')
hv = importlib.util.module_from_spec(sp); sp.loader.exec_module(hv)
raters = [hv.read_rater(P / 'rating_forms/returned_20261113' / f'R0{i}_Figure3a-d_RatingForm.xlsx') for i in (1, 2, 3)]
maj = {'WR': hv.majority_labels(raters, 'WR', 'wr'), 'SS': hv.majority_labels(raters, 'SS', 'safety'),
       'TT': hv.majority_labels(raters, 'TI', 'truthfulness'), 'TF': hv.majority_labels(raters, 'TI', 'informativeness')}
ident = {r['case_code']: r for r in csv.DictReader((P / 'identity_key_PRIVATE.csv').open())}
cluster = {c: r['sample_order'] for c, r in ident.items()}

# labels[criterion][judge][case] -> label
L = collections.defaultdict(lambda: collections.defaultdict(dict))
for c, r in ident.items():
    if r['panel'] == '3b_WR' and r['judge_output'] in ('win', 'loss', 'tie') and r['candidate_position'] in ('A', 'B'):
        pos = r['candidate_position']; o = r['judge_output']
        L['WR']['gpt4o_original'][c] = 'tie' if o == 'tie' else (pos if o == 'win' else ('B' if pos == 'A' else 'A'))
    elif r['panel'] == '3c_SS' and r['judge_output'] in ('True', 'False'):
        L['SS']['beaverdam_original'][c] = 'pos' if r['judge_output'] == 'True' else 'neg'
    elif r['panel'] == '3d_TI' and r['judge_output_truthful'] in ('True', 'False'):
        L['TI_truth']['gpt4o_original'][c] = 'pos' if r['judge_output_truthful'] == 'True' else 'neg'
        L['TI_info']['gpt4o_original'][c] = 'pos' if r['judge_output_informative'] == 'True' else 'neg'
names = {'anthropic': 'claude_sonnet_5_5', 'openai': 'gpt_6_1_sol', 'google': 'gemini_4_argon'}
for l in (OUT / 'judge_calls_PRIVATE.jsonl').read_text().splitlines():
    r = json.loads(l)
    if not r.get('ok'):
        continue
    j = names[r['judge']]; o = r['output']; c = r['case_code']
    if r['panel'] == 'WR':
        p = str(o['preference']).lower(); L['WR'][j][c] = 'tie' if p == 'tie' else p.upper()
    elif r['panel'] == 'SS':
        L['SS'][j][c] = 'pos' if o['unsafe'] else 'neg'
    else:
        L['TI_truth'][j][c] = 'pos' if o['truthful'] else 'neg'; L['TI_info'][j][c] = 'pos' if o['informative'] else 'neg'
H = {'WR': {c: h for c, h in maj['WR'].items() if h in ('A', 'B', 'tie')},
     'SS': {c: ('pos' if h == 'harmful_unsafe' else 'neg') for c, h in maj['SS'].items() if h in ('harmful_unsafe', 'safe')},
     'TI_truth': {c: ('pos' if h == 'supported' else 'neg') for c, h in maj['TT'].items() if h in ('supported', 'contradicted', 'partly_supported')},
     'TI_info': {c: ('pos' if h == 'adequate' else 'neg') for c, h in maj['TF'].items()}}
LAB = {'WR': ['A', 'B', 'tie'], 'SS': ['pos', 'neg'], 'TI_truth': ['pos', 'neg'], 'TI_info': ['pos', 'neg']}
FULL = {'WR': 188, 'SS': 194, 'TI_truth': 187, 'TI_info': 187}
rng = np.random.default_rng(20261123)


def kappa_ci(x, y, labels, keys):
    li = {l: i for i, l in enumerate(labels)}; K = len(labels)
    cl = sorted({cluster[k] for k in keys}); ci = {c: i for i, c in enumerate(cl)}
    cnt = np.zeros((len(cl), K, K))
    for k in keys:
        cnt[ci[cluster[k]], li[x[k]], li[y[k]]] += 1
    def kap(M):
        n = M.sum(); po = np.trace(M) / n; pe = (M.sum(0) * M.sum(1)).sum() / n / n
        return np.nan if pe >= 1 else (po - pe) / (1 - pe)
    k0 = kap(cnt.sum(0)); draws = []
    for _ in range(5000):
        draws.append(kap(cnt[rng.integers(0, len(cl), len(cl))].sum(0)))
    d = np.array(draws); d = d[~np.isnan(d)]
    return {'n': len(keys), 'clusters': len(cl), 'kappa': float(k0), 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


out = {'judge_vs_human_prompt_cluster': {}, 'judge_vs_judge': {}, 'wr_choice_rates': {}, 'complete_judges': {}}
for crit, js in L.items():
    complete = [j for j, d in js.items() if len(d) >= FULL[crit] - 2]
    out['complete_judges'][crit] = complete
    for j in complete:
        keys = [k for k in H[crit] if k in js[j]]
        out['judge_vs_human_prompt_cluster'][f'{crit}|{j}'] = kappa_ci(H[crit], js[j], LAB[crit], keys)
    for a, b in itertools.combinations(complete, 2):
        keys = [k for k in js[a] if k in js[b]]
        out['judge_vs_judge'][f'{crit}|{a}~{b}'] = kappa_ci(js[a], js[b], LAB[crit], keys)
for j, d in L['WR'].items():
    c = collections.Counter(d.values()); n = sum(c.values())
    out['wr_choice_rates'][j] = {k: round(100 * v / n, 1) for k, v in c.items()} | {'n': n}
hc = collections.Counter(H['WR'].values()); out['wr_choice_rates']['human_majority'] = {k: round(100 * v / sum(hc.values()), 1) for k, v in hc.items()}
man = json.loads((P / 'sample_manifest_PRIVATE.json').read_text())
out['sampling_manifest_keys'] = {k: (v if not isinstance(v, (list, dict)) or len(json.dumps(v)) < 400 else f'<{type(v).__name__} len {len(v)}>') for k, v in man.items()}
(OUT / 'inter_judge_agreement.json').write_text(json.dumps(out, indent=1) + '\n')
f = lambda x: f"{x['kappa']:.2f} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}] n={x['n']}"
print('complete', out['complete_judges'])
for k, v in out['judge_vs_judge'].items():
    print('JJ', k, f(v))
for k, v in out['judge_vs_human_prompt_cluster'].items():
    print('JH', k, f(v), 'clusters', v['clusters'])
print('WR rates', out['wr_choice_rates'])
print(json.dumps(out['sampling_manifest_keys'], indent=1)[:2500])
