"""Fig. 5 post-rating correctness audit against the HumaniBench answer key (pre-specified in
FINAL_3E_3F_PROTOCOL.md: "Use the source answer only for a separate post-rating correctness audit").
No new raters, no API calls. Human labels come from the frozen Fig. 5 analyzer's majority rule
(executed in memory with its output directory redirected to a scratch folder; no existing output is touched).
Runs on the reported and the superseded rating rounds. Writes fig5_human_transfer/e_answer_key_audit/.
"""
import collections, json, re
from pathlib import Path
import numpy as np

R = (next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd())); E = R / 'Method/OrchCircuit/final'
S = R / 'Datasets/HumaniBench/confirmatory_figure3ef_PRIVATE/sample_200_task3'
OUT = E / 'fig5_human_transfer/e_answer_key_audit'; OUT.mkdir(parents=True, exist_ok=True)
SRC = (E / 'shared/human_rating_pipeline/scripts/analyze_human_ratings_3e_f_PRIVATE.py').read_text()
CONTROLS = {'control_wr': 'gemini-4-argon', 'control_ss': 'gemma3-12b', 'control_ti': 'claude-opus-5-5'}
jl = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

key = {r['external_case_id']: r['source_answer'].strip()[0].upper() for r in json.load(open(S / 'answer_key_PRIVATE.json'))}
choice = {}
for p in (S / 'candidate_outputs_PRIVATE').glob('*.jsonl'):
    for r in jl(p):
        if r.get('candidate_text') and not r.get('error'):
            choice[(p.stem, r['external_case_id'])] = (r.get('choice_id') if r.get('valid_choice') else None)
sel = {x['external_case_id']: x['selection'] for x in jl(S / 'selector_outputs_PRIVATE/selections.jsonl')}


def system_choice(case, system):
    slug = sel[case][system.split('_')[1]]['model_slug'] if system.startswith('sel_') else CONTROLS[system]
    return choice.get((slug, case))


def analyzer_globals(form_dir, scratch):
    src = SRC.replace("OUT = BASE / 'Method/OrchCircuit/final'", f"OUT = Path({str(scratch)!r})")
    if form_dir:
        src = src.replace("FORM_DIR = BASE / 'Method/OrchCircuit/final/fig5_human_transfer/abc_pairwise_human/private/ratings_received'", f"FORM_DIR = Path({str(form_dir)!r})")
    # stop after the vote/label construction: everything needed is defined before the bootstrap section
    cut = src.index('# Bootstrap confidence intervals')
    g = {'__name__': 'fig5_audit'}; exec(compile(src[:cut], 'analyzer', 'exec'), g); return g


def kappa(pairs):
    M = np.zeros((2, 2))
    for a, b in pairs:
        M[a, b] += 1
    n = M.sum(); po = np.trace(M) / n; pe = (M.sum(0) * M.sum(1)).sum() / n / n
    return (po - pe) / (1 - pe)


def boot(items, stat, reps=5000, seed=20261123):
    cases = sorted({c for c, *_ in items}); by = collections.defaultdict(list)
    for it in items:
        by[it[0]].append(it)
    rng = np.random.default_rng(seed); v = []
    for _ in range(reps):
        smp = [it for c in rng.choice(cases, len(cases)) for it in by[c]]
        try:
            v.append(stat(smp))
        except (ZeroDivisionError, ValueError):
            pass
    v = np.array(v); v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


res = {'answer_key_n': len(key)}
rounds = {'reported': None, 'superseded': E / 'fig5_human_transfer/abc_pairwise_human/private/superseded_rating_round'}
for name, fd in rounds.items():
    g = analyzer_globals(fd, OUT / f'_scratch_{name}')
    cm = g['card_metrics']; wrv = g['wr_votes']; maj = g['majority']; RATERS = g['RATERS']
    # Truthfulness vs correctness
    items = []
    for (case, system), m in cm.items():
        ch = system_choice(case, system)
        if ch is None or np.isnan(m['truth']):
            continue
        items.append((case, int(ch == key[case]), int(m['truth'] == 1.0)))
    sup = lambda xs, c: np.mean([t for _, k_, t in xs if k_ == c])
    r = {'n_cards': len(items), 'n_correct': sum(k_ for _, k_, _ in items),
         'P_supported_given_correct': float(sup(items, 1)), 'P_supported_given_incorrect': float(sup(items, 0)),
         'kappa_truth_vs_key': float(kappa([(k_, t) for _, k_, t in items]))}
    r['kappa_ci95'] = boot(items, lambda xs: kappa([(k_, t) for _, k_, t in xs]))
    r['diff_pp'] = 100 * (r['P_supported_given_correct'] - r['P_supported_given_incorrect'])
    r['diff_ci95_pp'] = [100 * x for x in boot(items, lambda xs: sup(xs, 1) - sup(xs, 0))]
    # WR: when exactly one side is correct, does the human majority prefer it?
    wr = []
    for (case, s, c), votes in wrv.items():
        cs, cc = system_choice(case, s), system_choice(case, c)
        if cs is None or cc is None or (cs == key[case]) == (cc == key[case]):
            continue
        out = maj([votes[x] for x in RATERS])
        if out in ('Policy win', 'Control win'):
            wr.append((case, int((out == 'Policy win') == (cs == key[case])), 1))
    r['wr_one_side_correct_decisive_n'] = len(wr)
    r['wr_prefers_correct_pct'] = 100 * float(np.mean([x for _, x, _ in wr])) if wr else None
    r['wr_prefers_correct_ci95'] = [100 * x for x in boot(wr, lambda xs: np.mean([x for _, x, _ in xs]))] if wr else None
    res[name] = r
acc = collections.defaultdict(list)
for case in key:
    for system in ['sel_wr', 'sel_ss', 'sel_ti'] + list(CONTROLS):
        ch = system_choice(case, system)
        acc[system].append(int(ch == key[case]) if ch else 0)
res['system_accuracy_vs_key_pct'] = {k: 100 * float(np.mean(v)) for k, v in acc.items()}
(OUT / 'results.json').write_text(json.dumps(res, indent=1) + '\n')
print(json.dumps(res, indent=1))
