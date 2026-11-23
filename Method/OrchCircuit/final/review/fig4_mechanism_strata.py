"""Fig. 4 mechanism check (saved labels only): does judge-human agreement on the Ch3EF items depend on item type?
Strata: single- vs multi-image items (Ch3EF record image count), and small vs large rater image (montage long side,
median split). For each criterion and stratum: kappa of every judge vs the human majority, and the judge-averaged
kappa with a prompt-cluster bootstrap CI; plus the multi-minus-single and large-minus-small differences.
Writes fig2_judge_validity/f_frontier_judges/mechanism_strata.json."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

E = (next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()) / 'Method/OrchCircuit/final'); R = E.parents[2]
ns = {}; src = (E / 'review/inter_judge_agreement.py').read_text(); exec(compile(src.split("out = {'judge_vs_human_prompt_cluster'")[0], 'ija', 'exec'), ns)
L, H, LAB, ident, P = ns['L'], ns['H'], ns['LAB'], ns['ident'], ns['P']
recs = {int(r['sample_order']): r for r in json.loads((R / 'Datasets/Ch3EF/data/ch3ef_full1416/sample.json').read_text())['records']}
pk = json.loads((P / 'packet_data_FOR_RATERS_PRIVATE.json').read_text())['sheets']
img = {x['case_code']: P / x['image_file'] for sh in pk.values() for x in sh}
nimg = {c: len(recs[int(r['sample_order'])]['image']) for c, r in ident.items()}
side = {c: max(Image.open(p).size) for c, p in img.items()}
med = float(np.median(list(side.values())))
STRATA = {'single_image': lambda c: nimg[c] == 1, 'multi_image': lambda c: nimg[c] > 1,
          'small_image': lambda c: side[c] <= med, 'large_image': lambda c: side[c] > med}


def kap(M):
    n = M.sum(); po = np.trace(M) / n; pe = (M.sum(0) * M.sum(1)).sum() / n / n
    return np.nan if pe >= 1 or n == 0 else (po - pe) / (1 - pe)


rng = np.random.default_rng(20261123); B = 5000; out = {'median_long_side_px': med, 'criteria': {}}
for crit, js in L.items():
    judges = [j for j in js if len(js[j]) >= len(H[crit]) - 2]
    labs = LAB[crit]; li = {l: i for i, l in enumerate(labs)}; K = len(labs)
    keys = [c for c in H[crit] if all(c in js[j] for j in judges)]
    cnt = {j: {c: np.zeros((K, K)) for c in keys} for j in judges}
    for j in judges:
        for c in keys:
            cnt[j][c][li[H[crit][c]], li[js[j][c]]] += 1
    res = {}; draws = {}
    for sname, f in STRATA.items():
        ks = [c for c in keys if f(c)]
        if len(ks) < 20:
            res[sname] = {'n': len(ks), 'status': 'too few items'}; continue
        per = {j: float(kap(sum(cnt[j][c] for c in ks))) for j in judges}
        arr = np.stack([[cnt[j][c] for c in ks] for j in judges])          # judges x items x K x K
        bs = []
        for _ in range(B):
            ix = rng.integers(0, len(ks), len(ks)); M = arr[:, ix].sum(1)
            bs.append(np.nanmean([kap(m) for m in M]))
        bs = np.array(bs); draws[sname] = bs
        res[sname] = {'n': len(ks), 'kappa_by_judge': {j: round(v, 3) for j, v in per.items()}, 'mean_kappa': round(float(np.nanmean(list(per.values()))), 3),
                      'mean_kappa_ci95': [round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]}
    for a, b in (('multi_image', 'single_image'), ('large_image', 'small_image')):
        if a in draws and b in draws:
            d = draws[a] - draws[b]
            res[f'{a}_minus_{b}'] = {'diff': round(res[a]['mean_kappa'] - res[b]['mean_kappa'], 3),
                                     'ci95_independent_bootstrap': [round(float(np.nanpercentile(d, 2.5)), 3), round(float(np.nanpercentile(d, 97.5)), 3)]}
    out['criteria'][crit] = res
(E / 'fig2_judge_validity/f_frontier_judges/mechanism_strata.json').write_text(json.dumps(out, indent=1) + '\n')
print('median long side', med)
for crit, res in out['criteria'].items():
    for s, v in res.items():
        print(crit, s, {k: v[k] for k in v if k in ('n', 'mean_kappa', 'mean_kappa_ci95', 'diff', 'ci95_independent_bootstrap', 'status')})
