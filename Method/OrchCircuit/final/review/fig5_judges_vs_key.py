"""Fig. 5e: automated judges vs the HumaniBench answer key on the Fig. 5 answers (same items the raters rated).
Judges: gpt4o (original TI judge, audit call path), sonnet, gemini, gpt61sol (frontier). Rubric: the image-access audit
TI and WR prompts, image present. TI on every unique (case, answer text) among the 6 rated systems; WR on the unique
selector/control pairs where exactly one answer matches the key (balanced deterministic A/B orientation).
  python3 fig5_judges_vs_key.py run --cap-usd 20      (resumable; hard cap across all judges)
  python3 fig5_judges_vs_key.py analyze
"""
import argparse, base64, collections, hashlib, importlib.util, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

R = (next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd())); E = R / 'Method/OrchCircuit/final'
S = R / 'Datasets/HumaniBench/confirmatory_figure3ef_PRIVATE/sample_200_task3'
OUT = E / 'fig5_human_transfer/e_answer_key_audit/judges'; OUT.mkdir(parents=True, exist_ok=True)
SYSTEMS = {'sel_wr': ('sel', 'wr'), 'sel_ss': ('sel', 'ss'), 'sel_ti': ('sel', 'ti'),
           'control_wr': ('ctl', 'gemini-4-argon'), 'control_ss': ('ctl', 'gemma3-12b'), 'control_ti': ('ctl', 'claude-opus-5-5')}
jl = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []
h = lambda s: hashlib.sha256(s.encode()).hexdigest()


def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


audit = load(R / 'Baselines/Judge/scripts/run_judge_image_access_audit.py', 'audit')
frontier = load(E / 'review/frontier_judges_vs_humans.py', 'frontier')
recs = {r['external_case_id']: r for r in json.loads((S / 'inference_prompts.json').read_text())['records']}
key = {r['external_case_id']: r['source_answer'].strip()[0].upper() for r in json.load(open(S / 'answer_key_PRIVATE.json'))}
cand = {}
for p in (S / 'candidate_outputs_PRIVATE').glob('*.jsonl'):
    for r in jl(p):
        if r.get('candidate_text') and not r.get('error'):
            cand[(p.stem, r['external_case_id'])] = r
sel = {x['external_case_id']: x['selection'] for x in jl(S / 'selector_outputs_PRIVATE/selections.jsonl')}


def answer(case, system):
    kind, k = SYSTEMS[system]
    slug = sel[case][k]['model_slug'] if kind == 'sel' else k
    row = cand[(slug, case)]
    return row['candidate_text'], (row.get('choice_id') if row.get('valid_choice') else None)


def items():
    out = []; seen = set()
    for case in sorted(recs):
        rec = {'query': recs[case]['question'], 'options': recs[case]['options']}; img = S / recs[case]['image_file']
        for system in SYSTEMS:
            t, ch = answer(case, system); tk = f'TI|{case}|{h(t.strip())}'
            if tk not in seen:
                seen.add(tk); out.append({'key': tk, 'task': 'TI', 'case': case, 'prompt': audit.prompt_for('TI', rec, t, None, True), 'image': img})
        for s in ('sel_wr', 'sel_ss', 'sel_ti'):
            for c in ('control_wr', 'control_ss', 'control_ti'):
                (ts, cs), (tc, cc) = answer(case, s), answer(case, c)
                if cs is None or cc is None or (cs == key[case]) == (cc == key[case]):
                    continue
                k_ = f'WR|{case}|{s}|{c}'; sel_a = int(h(k_), 16) % 2 == 0; a, b = (ts, tc) if sel_a else (tc, ts)
                out.append({'key': k_, 'task': 'WR', 'case': case, 'selector_is_A': sel_a, 'selector_correct': cs == key[case],
                            'prompt': audit.prompt_for('WR', rec, a, b, True), 'image': img})
    return out


def call(judge, it):
    if judge == 'gpt4o':
        raw = it['image'].read_bytes(); mime = 'image/png' if raw.startswith(b'\x89PNG') else 'image/jpeg'
        parsed, meta = audit.call_judge('openai', it['prompt'], [(mime, base64.b64encode(raw).decode())], it['task'])
        if not audit.valid_judgment(it['task'], parsed):
            raise RuntimeError('invalid judgment fields')
        return parsed, meta['estimated_cost_usd']
    return frontier.call({'sonnet': 'anthropic', 'gemini': 'google', 'gpt61sol': 'openai'}[judge], {'panel': it['task'], 'prompt': it['prompt'], 'image': it['image']})


LOCK = threading.Lock(); SPENT = [0.0]; JUDGES = ['gpt4o', 'sonnet', 'gpt61sol', 'gemini']


def run(cap):
    its = items(); print(f"items TI={sum(i['task'] == 'TI' for i in its)} WR={sum(i['task'] == 'WR' for i in its)}", flush=True)
    for j in JUDGES:
        SPENT[0] += sum(r.get('cost_usd', 0) for r in jl(OUT / f'{j}.jsonl'))
    print(f'prior spend {SPENT[0]:.3f} cap {cap}', flush=True)

    def work(j):
        p = OUT / f'{j}.jsonl'; done = {r['key'] for r in jl(p) if r.get('ok')}; n = 0
        for it in its:
            if it['key'] in done:
                continue
            with LOCK:
                if SPENT[0] >= cap:
                    print(f'{j} CAP REACHED', flush=True); return
            row = {'judge': j, 'key': it['key'], 'task': it['task'], 'case': it['case']}
            if it['task'] == 'WR':
                row.update(selector_is_A=it['selector_is_A'], selector_correct=it['selector_correct'])
            for a in range(3):
                try:
                    o, c = call(j, it); row.update(ok=True, output=o, cost_usd=c); break
                except Exception as e:
                    row.update(ok=False, error=str(e)[:200], cost_usd=0.0); time.sleep(2 ** a)
            with LOCK:
                SPENT[0] += row['cost_usd']
                with p.open('a') as f:
                    f.write(json.dumps(row) + '\n')
            n += 1
            if n % 100 == 0:
                print(f'{j} {n} spent={SPENT[0]:.3f}', flush=True)
    with ThreadPoolExecutor(len(JUDGES)) as ex:
        list(ex.map(work, JUDGES))
    print(f'done spent={SPENT[0]:.3f}', flush=True)


def analyze():
    rng = np.random.default_rng(20261123)

    def boot(rows, stat, reps=5000):
        cases = sorted({r[0] for r in rows}); by = collections.defaultdict(list)
        for r in rows:
            by[r[0]].append(r)
        v = []
        for _ in range(reps):
            smp = [r for c in rng.choice(cases, len(cases)) for r in by[c]]
            try:
                x = stat(smp)
                if not np.isnan(x):
                    v.append(x)
            except (ZeroDivisionError, ValueError):
                pass
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    def kap(rows):
        M = np.zeros((2, 2))
        for _, k_, t in rows:
            M[k_, t] += 1
        n = M.sum(); po = np.trace(M) / n; pe = (M.sum(0) * M.sum(1)).sum() / n / n
        return (po - pe) / (1 - pe)
    diff = lambda rows: np.mean([t for _, k_, t in rows if k_]) - np.mean([t for _, k_, t in rows if not k_])
    res = {}
    for j in JUDGES:
        L = {r['key']: r for r in jl(OUT / f'{j}.jsonl') if r.get('ok')}
        ti = []
        for case in sorted(recs):
            for system in SYSTEMS:
                t, ch = answer(case, system); r = L.get(f'TI|{case}|{h(t.strip())}')
                if r is None or ch is None:
                    continue
                ti.append((case, int(ch == key[case]), int(bool(r['output']['truthful']))))
        wr = [(r['case'], int((str(r['output']['preference']).lower() == 'a') == r['selector_is_A']) if str(r['output']['preference']).lower() != 'tie' else None, r['selector_correct'])
              for r in L.values() if r['task'] == 'WR']
        dec = [(c, int(pick_sel == corr), 1) for c, pick_sel, corr in wr if pick_sel is not None]
        res[j] = {'TI_n': len(ti), 'P_truthful_given_correct': float(np.mean([t for _, k_, t in ti if k_])), 'P_truthful_given_incorrect': float(np.mean([t for _, k_, t in ti if not k_])),
                  'diff_pp': 100 * float(diff(ti)), 'diff_ci95_pp': [100 * x for x in boot(ti, diff)], 'kappa_vs_key': float(kap(ti)), 'kappa_ci95': boot(ti, kap),
                  'WR_pairs': len(wr), 'WR_tie_pct': 100 * float(np.mean([p is None for _, p, _ in wr])) if wr else None,
                  'WR_decisive_n': len(dec), 'WR_prefers_correct_pct': 100 * float(np.mean([x for _, x, _ in dec])) if dec else None,
                  'WR_prefers_correct_ci95': [100 * x for x in boot(dec, lambda xs: np.mean([x for _, x, _ in xs]))] if dec else None}
    (OUT.parent / 'judges_vs_key_results.json').write_text(json.dumps(res, indent=1) + '\n')
    for j, v in res.items():
        print(j, {k: (round(x, 3) if isinstance(x, float) else ([round(y, 2) for y in x] if isinstance(x, list) else x)) for k, x in v.items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['run', 'analyze']); ap.add_argument('--cap-usd', type=float, default=20.0)
    a = ap.parse_args(); run(a.cap_usd) if a.mode == 'run' else analyze()
