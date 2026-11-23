"""Fig. 5e checks (saved data only): (1) judges vs key on exactly the raters' assessable cards;
(2) judge-vs-human truthfulness kappa on the Fig. 5 cards (reported round); (3) WR on the raters' decisive one-correct pairs."""
import collections, json, sys
from pathlib import Path
import numpy as np

E = (next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()) / 'Method/OrchCircuit/final'); sys.argv = ['x', 'analyze']
ns = {}; exec(compile((E / 'review/fig5_judges_vs_key.py').read_text().split("if __name__ == '__main__':")[0], 'jk', 'exec'), ns)
au = {}; src = (E / 'review/fig5_answer_key_audit.py').read_text(); exec(compile(src.split('res = {')[0], 'audit', 'exec'), au)
OUT = E / 'fig5_human_transfer/e_answer_key_audit'
g = au['analyzer_globals'](None, OUT / '_scratch_lfl'); cm = g['card_metrics']; wrv = g['wr_votes']; maj = g['majority']; RATERS = g['RATERS']
rng = np.random.default_rng(20261123)


def boot(rows, stat, reps=5000):
    cases = sorted({r[0] for r in rows}); by = collections.defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
    v = []
    for _ in range(reps):
        s = [r for c in rng.choice(cases, len(cases)) for r in by[c]]
        x = stat(s)
        if not np.isnan(x):
            v.append(x)
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def kap(rows):
    M = np.zeros((2, 2))
    for _, a, b in rows:
        M[a, b] += 1
    n = M.sum(); po = np.trace(M) / n; pe = (M.sum(0) * M.sum(1)).sum() / n / n
    return np.nan if pe >= 1 else (po - pe) / (1 - pe)


diff = lambda rows: np.mean([t for _, k, t in rows if k]) - np.mean([t for _, k, t in rows if not k])
res = {}
for j in ns['JUDGES']:
    L = {r['key']: r for r in ns['jl'](ns['OUT'] / f'{j}.jsonl') if r.get('ok')}
    sub, jh = [], []
    for (case, system), m in cm.items():
        if np.isnan(m['truth']):
            continue
        t, ch = ns['answer'](case, system); r = L.get(f"TI|{case}|{ns['h'](t.strip())}")
        if r is None or ch is None:
            continue
        jt = int(bool(r['output']['truthful']))
        sub.append((case, int(ch == ns['key'][case]), jt)); jh.append((case, int(m['truth'] == 1.0), jt))
    wr = []
    for (case, s, c), votes in wrv.items():
        out = maj([votes[x] for x in RATERS]); r = L.get(f'WR|{case}|{s}|{c}')
        if r is None or out not in ('Policy win', 'Control win'):
            continue
        p = str(r['output']['preference']).lower()
        if p == 'tie':
            continue
        jsel = (p == 'a') == r['selector_is_A']
        wr.append((case, int(jsel == r['selector_correct']), 1))
    res[j] = {'subset_n': len(sub), 'key_diff_pp': round(100 * float(diff(sub)), 1), 'key_diff_ci95_pp': [round(100 * x, 1) for x in boot(sub, diff)],
              'key_kappa': round(float(kap(sub)), 3), 'key_kappa_ci95': boot(sub, kap),
              'judge_vs_human_truth_kappa': round(float(kap(jh)), 3), 'judge_vs_human_truth_kappa_ci95': boot(jh, kap),
              'judge_truthful_rate_pct': round(100 * float(np.mean([t for *_, t in jh])), 1), 'human_supported_rate_pct': round(100 * float(np.mean([h for _, h, _ in jh])), 1),
              'WR_on_raters_decisive_one_correct_n': len(wr), 'WR_prefers_correct_pct': round(100 * float(np.mean([x for _, x, _ in wr])), 1) if wr else None}
(OUT / 'like_for_like_results.json').write_text(json.dumps(res, indent=1) + '\n')
for j, v in res.items():
    print(j, v)
