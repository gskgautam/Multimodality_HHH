"""Objective endpoint: accuracy vs the HumaniBench answer key on the Fig. 5 sample (200) and the
O3 sample (300). No judges, no raters. Also: single-rater truthfulness vs key on Fig. 5 (round-3 minor 3) and
image-size / item-format descriptives for Ch3EF (Fig. 4) vs HumaniBench (Fig. 5) (round-3 M3, minor 4)."""
import ast, collections, json, re
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

R = (next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd())); E = R / 'Method/OrchCircuit/final'
HB = R / 'Datasets/HumaniBench/confirmatory_figure3ef_PRIVATE'
SAMPLES = {'fig5': HB / 'sample_200_task3', 'o3': R / 'Datasets/HumaniBench/confirmatory_v2_PRIVATE'}
CONTROLS = {'control_wr': 'gemini-4-argon', 'control_ss': 'gemma3-12b', 'control_ti': 'claude-opus-5-5'}
POL = {'OrchCircuit-WR': 'wr', 'OrchCircuit-SS': 'ss', 'OrchCircuit-TI': 'ti'}
jl = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
OUT = E / 'fig5_human_transfer/f_objective_key_accuracy'; OUT.mkdir(parents=True, exist_ok=True)
par = pd.read_parquet(HB / 'task3.parquet'); par_key = {str(r.ID): str(r.Answer).strip()[0].upper() for r in par.itertuples()}


def load(name, S):
    recs = json.loads((S / 'inference_prompts.json').read_text())['records']
    if name == 'fig5':
        key = {r['external_case_id']: r['source_answer'].strip()[0].upper() for r in json.load(open(S / 'answer_key_PRIVATE.json'))}
    else:
        key = {r['external_case_id']: par_key[str(r['source_record_id'])] for r in recs}
    ch = {}
    for p in (S / 'candidate_outputs_PRIVATE').glob('*.jsonl'):
        for r in jl(p):
            if r.get('candidate_text') and not r.get('error'):
                ch[(p.stem, r['external_case_id'])] = r.get('choice_id') if r.get('valid_choice') else None
    sel = {x['external_case_id']: x['selection'] for x in jl(S / 'selector_outputs_PRIVATE/selections.jsonl')}
    cases = sorted(key); models = sorted({m for m, _ in ch})
    M = {m: np.array([int(ch.get((m, c)) == key[c]) for c in cases]) for m in models}
    for p, k in POL.items():
        M[p] = np.array([int(ch.get((sel[c][k]['model_slug'], c)) == key[c]) for c in cases])
    return cases, M


rng = np.random.default_rng(20261113); res = {}
pooled = collections.defaultdict(list)
for name, S in SAMPLES.items():
    cases, M = load(name, S); n = len(cases); idx = rng.integers(0, n, (10000, n))
    acc = {m: round(100 * float(v.mean()), 1) for m, v in M.items()}
    con = {}
    for p in POL:
        for c, slug in CONTROLS.items():
            d = (M[p] - M[slug]).astype(float); b = d[idx].mean(1) * 100
            con[f'{p} vs {slug}'] = {'diff_pp': round(100 * float(d.mean()), 1), 'ci95': [round(float(np.percentile(b, 2.5)), 1), round(float(np.percentile(b, 97.5)), 1)]}
    best = max((m for m in M if m not in POL), key=lambda m: M[m].mean())
    for p in POL:
        d = (M[p] - M[best]).astype(float); b = d[idx].mean(1) * 100
        con[f'{p} vs best generator ({best}, chosen on this sample)'] = {'diff_pp': round(100 * float(d.mean()), 1), 'ci95': [round(float(np.percentile(b, 2.5)), 1), round(float(np.percentile(b, 97.5)), 1)]}
    res[name] = {'n': n, 'accuracy_pct': dict(sorted(acc.items(), key=lambda kv: -kv[1])), 'contrasts': con}
    for m, v in M.items():
        pooled[m].append(v)
P = {m: np.concatenate(v) for m, v in pooled.items() if len(v) == 2}; n = len(next(iter(P.values()))); idx = rng.integers(0, n, (10000, n))
res['pooled_500'] = {'accuracy_pct': dict(sorted({m: round(100 * float(v.mean()), 1) for m, v in P.items()}.items(), key=lambda kv: -kv[1])), 'contrasts': {}}
for p in POL:
    for c, slug in CONTROLS.items():
        d = (P[p] - P[slug]).astype(float); b = d[idx].mean(1) * 100
        res['pooled_500']['contrasts'][f'{p} vs {slug}'] = {'diff_pp': round(100 * float(d.mean()), 1), 'ci95': [round(float(np.percentile(b, 2.5)), 1), round(float(np.percentile(b, 97.5)), 1)]}

# single-rater truthfulness vs key (Fig. 5 reported round)
au = {}; src = (E / 'review/fig5_answer_key_audit.py').read_text(); exec(compile(src.split('res = {')[0], 'audit', 'exec'), au)
g = au['analyzer_globals'](None, OUT / '_scratch'); single = {}
for r in g['RATERS']:
    rows = []
    for (case, system), votes in g['card_votes'].items():
        v = votes[r]['Truthfulness']; chs = au['system_choice'](case, system)
        if v == 'Not assessable' or chs is None:
            continue
        rows.append((int(chs == au['key'][case]), int(v == 'Supported')))
    M2 = np.zeros((2, 2))
    for a, b in rows:
        M2[a, b] += 1
    nn = M2.sum(); po = np.trace(M2) / nn; pe = (M2.sum(0) * M2.sum(1)).sum() / nn / nn
    single[r] = {'n': len(rows), 'kappa_vs_key': round(float((po - pe) / (1 - pe)), 3),
                 'supported_given_correct_pct': round(100 * M2[1, 1] / M2[1].sum(), 1), 'supported_given_incorrect_pct': round(100 * M2[0, 1] / M2[0].sum(), 1)}
res['fig5_single_rater_truth_vs_key'] = single

# image descriptives: Fig. 4 rater packet vs Fig. 5 sample
def dims(paths):
    s = [Image.open(p).size for p in paths]; mx = np.array([max(a) for a in s]); asp = np.array([max(a) / min(a) for a in s])
    return {'n': len(s), 'median_long_side_px': int(np.median(mx)), 'share_long_side_over_2000px_pct': round(100 * float((mx > 2000).mean()), 1),
            'share_aspect_over_2_pct': round(100 * float((asp > 2).mean()), 1)}
P4 = R / 'Datasets/Ch3EF/data/ch3ef_full1416/human_validation_figure3a_d_20261112_PRIVATE/images'
res['image_descriptives'] = {'fig4_ch3ef_rater_packet_montages': dims(sorted(P4.glob('*'))), 'fig5_humanibench': dims(sorted((SAMPLES['fig5'] / 'images').glob('*')))}
ch3 = json.loads((R / 'Datasets/Ch3EF/data/ch3ef_full1416/sample.json').read_text())['records']
res['image_descriptives']['ch3ef_multi_image_item_share_pct'] = round(100 * float(np.mean([len(r['image']) > 1 for r in ch3])), 1)
(OUT / 'results.json').write_text(json.dumps(res, indent=1) + '\n')
print(json.dumps({k: (v if k != 'fig5' and k != 'o3' else {'n': v['n'], 'top': dict(list(v['accuracy_pct'].items())[:6]), 'pol': {p: v['accuracy_pct'][p] for p in POL}, 'contrasts': v['contrasts']}) for k, v in res.items()}, indent=1))
