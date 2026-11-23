"""Fig. 4 extension: current frontier judges vs the existing human majority labels (569 items).

Judges: Claude Sonnet 5.5 (anthropic), Gemini 4 Argon (google), GPT-6.1 Sol (openai).
Prompts: the exact WR/SS/TI rubric prompts of the image-access audit (run_judge_image_access_audit.py),
image present. Inputs: the same question, options, response text(s) and image the raters saw
(packet human_validation_figure3a_d_20261112_PRIVATE). WR uses the displayed A/B order, as the raters saw it.
Resumable journal; hard spend cap. Writes fig2_judge_validity/f_frontier_judges/.
  python3 frontier_judges_vs_humans.py run   [--cap-usd 15]
  python3 frontier_judges_vs_humans.py analyze
"""
import argparse, base64, collections, importlib.util, json, stat, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

R = (next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd())); E = R / 'Method/OrchCircuit/final'
P = R / 'Datasets/Ch3EF/data/ch3ef_full1416/human_validation_figure3a_d_20261112_PRIVATE'
OUT = E / 'fig2_judge_validity/f_frontier_judges'; OUT.mkdir(parents=True, exist_ok=True)
JOURNAL = OUT / 'judge_calls_PRIVATE.jsonl'
KEYDIR = Path.home() / '.config/multimodality'
JUDGES = {'anthropic': {'model': 'claude-sonnet-5-5', 'key': KEYDIR / 'anthropic_api_key', 'rates': (2.0, 10.0)},
          'google': {'model': 'gemini-4-argon', 'key': KEYDIR / 'google_gemini_api_key', 'rates': (2.0, 10.0)},
          'openai': {'model': 'gpt-6.1-sol', 'key': KEYDIR / 'openai_api_key', 'rates': (1.0, 5.0)}}
spec = importlib.util.spec_from_file_location('audit', R / 'Baselines/Judge/scripts/run_judge_image_access_audit.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
LOCK = threading.Lock(); SPENT = [0.0]


def items():
    sheets = json.loads((P / 'packet_data_FOR_RATERS_PRIVATE.json').read_text())['sheets']; out = []
    for panel in ('WR', 'SS', 'TI'):
        for x in sheets[panel]:
            opts = [l.split('. ', 1)[1] if '. ' in l[:4] else l for l in x['answer_options'].split('\n') if l.strip()]
            rec = {'query': x['question'], 'options': opts}
            if panel == 'WR':
                prompt = audit.prompt_for('WR', rec, x['response_A'], x['response_B'], True)
            else:
                prompt = audit.prompt_for(panel, rec, x['candidate_response'], None, True)
            out.append({'case_code': x['case_code'], 'panel': panel, 'prompt': prompt, 'image': P / x['image_file']})
    return out


def blocks(provider, prompt, img):
    raw = img.read_bytes(); mime = 'image/png' if raw.startswith(b'\x89PNG') else 'image/jpeg'
    b64 = base64.b64encode(raw).decode()
    if provider == 'openai':
        return [{'type': 'input_text', 'text': prompt}, {'type': 'input_image', 'image_url': f'data:{mime};base64,{b64}', 'detail': 'high'}]
    return audit.content_blocks(provider, prompt, [(mime, b64)])


def call(provider, it):
    cfg = JUDGES[provider]
    if stat.S_IMODE(cfg['key'].stat().st_mode) & 0o077:
        raise RuntimeError('key file permissions too broad')
    key = cfg['key'].read_text().strip(); task = it['panel']
    sysmsg = 'You are an impartial evaluator. Return only the requested JSON.'
    if provider == 'openai':
        body = {'model': cfg['model'], 'instructions': sysmsg, 'input': [{'role': 'user', 'content': blocks(provider, it['prompt'], it['image'])}],
                'max_output_tokens': 600, 'text': {'format': {'type': 'json_object'}}}
        url, hdr = 'https://api.openai.com/v1/responses', {'Authorization': 'Bearer ' + key}
    elif provider == 'google':
        body = {'contents': [{'role': 'user', 'parts': blocks(provider, it['prompt'], it['image'])}],
                'generationConfig': {'temperature': 0, 'maxOutputTokens': 600, 'responseMimeType': 'application/json', 'thinkingConfig': {'thinkingLevel': 'low'}}}
        url, hdr = f'https://generativelanguage.googleapis.com/v1beta/models/{cfg["model"]}:generateContent', {'x-goog-api-key': key}
    else:
        props = {'WR': ({'preference': {'type': 'string', 'enum': ['A', 'B', 'tie']}}, ['preference']),
                 'SS': ({'unsafe': {'type': 'boolean'}, 'reason': {'type': 'string'}}, ['unsafe', 'reason']),
                 'TI': ({'truthful': {'type': 'boolean'}, 'informative': {'type': 'boolean'}}, ['truthful', 'informative'])}[task]
        body = {'model': cfg['model'], 'max_tokens': 600, 'thinking': {'type': 'adaptive'}, 'system': sysmsg,
                'messages': [{'role': 'user', 'content': blocks(provider, it['prompt'], it['image'])}],
                'output_config': {'effort': 'low', 'format': {'type': 'json_schema', 'schema': {'type': 'object', 'properties': props[0], 'required': props[1], 'additionalProperties': False}}}}
        url, hdr = 'https://api.anthropic.com/v1/messages', {'x-api-key': key, 'anthropic-version': '2023-06-01'}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={**hdr, 'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code}: {e.read().decode(errors="replace")[:200]}') from None
    if provider == 'openai':
        raw = '\n'.join(c.get('text', '') for o in d.get('output', []) for c in o.get('content', []) if c.get('type') == 'output_text')
        u = d.get('usage', {}); i, o = u.get('input_tokens', 0), u.get('output_tokens', 0)
    elif provider == 'google':
        c = (d.get('candidates') or [{}])[0]; raw = '\n'.join(p.get('text', '') for p in c.get('content', {}).get('parts', []))
        u = d.get('usageMetadata', {}); i, o = u.get('promptTokenCount', 0), u.get('candidatesTokenCount', 0) + u.get('thoughtsTokenCount', 0)
    else:
        raw = '\n'.join(x.get('text', '') for x in d.get('content', []) if x.get('type') == 'text')
        u = d.get('usage', {}); i, o = u.get('input_tokens', 0), u.get('output_tokens', 0)
    parsed = json.loads(raw)
    if not audit.valid_judgment(task, parsed):
        raise RuntimeError('invalid judgment fields')
    return parsed, (i * cfg['rates'][0] + o * cfg['rates'][1]) / 1e6


def run(cap):
    done = set()
    if JOURNAL.exists():
        for l in JOURNAL.read_text().splitlines():
            r = json.loads(l); SPENT[0] += r.get('cost_usd', 0)
            if r.get('ok'):
                done.add((r['judge'], r['case_code']))
    its = items(); print(f'items={len(its)} judges={len(JUDGES)} done={len(done)} spent={SPENT[0]:.3f} cap={cap}', flush=True)

    def work(provider):
        n = 0
        for it in its:
            if (provider, it['case_code']) in done:
                continue
            with LOCK:
                if SPENT[0] >= cap:
                    print('CAP REACHED', flush=True); return n
            row = {'judge': provider, 'model': JUDGES[provider]['model'], 'case_code': it['case_code'], 'panel': it['panel']}
            for attempt in range(3):
                try:
                    parsed, cost = call(provider, it); row.update(ok=True, output=parsed, cost_usd=cost); break
                except Exception as e:
                    row.update(ok=False, error=str(e)[:200], cost_usd=0.0); time.sleep(2 ** attempt)
            with LOCK:
                SPENT[0] += row['cost_usd']
                with JOURNAL.open('a') as f:
                    f.write(json.dumps(row) + '\n')
            n += 1
            if n % 50 == 0:
                print(f'{provider} {n} spent={SPENT[0]:.3f}', flush=True)
        return n
    with ThreadPoolExecutor(3) as ex:
        print(dict(zip(JUDGES, ex.map(work, JUDGES))), f'spent={SPENT[0]:.3f}', flush=True)


def kappa(pairs, labels):
    n = len(pairs); idx = {l: i for i, l in enumerate(labels)}; m = np.zeros((len(labels),) * 2)
    for h, j in pairs:
        m[idx[h], idx[j]] += 1
    po = np.trace(m) / n; pe = (m.sum(1) * m.sum(0)).sum() / n / n
    return (po - pe) / (1 - pe), m


def analyze():
    hv = importlib.util.spec_from_file_location('hv', E / 'shared/human_rating_pipeline/scripts/analyze_figure3a_d_human_validation.py')
    m = importlib.util.module_from_spec(hv); hv.loader.exec_module(m)
    raters = [m.read_rater(P / 'rating_forms/returned_20261113' / f'R0{i}_Figure3a-d_RatingForm.xlsx') for i in (1, 2, 3)]
    maj = {'WR': m.majority_labels(raters, 'WR', 'wr'), 'SS': m.majority_labels(raters, 'SS', 'safety'),
           'TT': m.majority_labels(raters, 'TI', 'truthfulness'), 'TF': m.majority_labels(raters, 'TI', 'informativeness')}
    ident = {r['case_code']: r for r in __import__('csv').DictReader((P / 'identity_key_PRIVATE.csv').open())}
    fam = lambda s: 'anthropic' if 'claude' in s else 'google' if 'gemini' in s else 'openai' if 'gpt' in s else 'open'
    calls = {}
    for l in JOURNAL.read_text().splitlines():
        r = json.loads(l)
        if r.get('ok'):
            calls[(r['judge'], r['case_code'])] = r['output']
    res = {}
    rng = np.random.default_rng(20261123)
    for judge in JUDGES:
        res[judge] = {'model': JUDGES[judge]['model']}
        for excl in (False, True):
            tag = 'excl_same_family' if excl else 'all'
            def keep(cc):
                k = ident.get(cc, {}); fams = {fam(k.get('candidate_model_id', '')), fam(k.get('anchor_model_id', ''))}
                return not (excl and judge in fams)
            spec_ = {
                'WR': ([(h, str(calls[(judge, c)]['preference']).lower().replace('a', 'A').replace('b', 'B')) for c, h in maj['WR'].items()
                        if h in ('A', 'B', 'tie') and (judge, c) in calls and keep(c)], ['A', 'B', 'tie']),
                'SS': ([('pos' if h == 'harmful_unsafe' else 'neg', 'pos' if calls[(judge, c)]['unsafe'] else 'neg') for c, h in maj['SS'].items()
                        if h in ('harmful_unsafe', 'safe') and (judge, c) in calls and keep(c)], ['pos', 'neg']),
                'TI_truth': ([('pos' if h == 'supported' else 'neg', 'pos' if calls[(judge, c)]['truthful'] else 'neg') for c, h in maj['TT'].items()
                              if h in ('supported', 'contradicted', 'partly_supported') and (judge, c) in calls and keep(c)], ['pos', 'neg']),
                'TI_info': ([('pos' if h == 'adequate' else 'neg', 'pos' if calls[(judge, c)]['informative'] else 'neg') for c, h in maj['TF'].items()
                             if (judge, c) in calls and keep(c)], ['pos', 'neg'])}
            for crit, (pairs, labels) in spec_.items():
                if len(pairs) < 10:
                    continue
                k, mat = kappa(pairs, labels)
                bs = []
                for _ in range(5000):
                    s = [pairs[i] for i in rng.integers(0, len(pairs), len(pairs))]
                    try:
                        bs.append(kappa(s, labels)[0])
                    except ZeroDivisionError:
                        pass
                row = {'n': len(pairs), 'kappa': float(k), 'kappa_ci95': [float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))],
                       'accuracy': float(np.trace(mat) / mat.sum()), 'confusion_rows_human_cols_judge': mat.astype(int).tolist(), 'labels': labels}
                if labels == ['pos', 'neg']:
                    tp, fn, fp, tn = mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1]
                    row.update(sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)), judge_pos_rate=float((tp + fp) / mat.sum()), human_pos_rate=float((tp + fn) / mat.sum()))
                res[judge].setdefault(tag, {})[crit] = row
    (OUT / 'results.json').write_text(json.dumps(res, indent=1) + '\n')
    for j, v in res.items():
        print(j, v['model'], {c: (x['n'], round(x['kappa'], 3), [round(y, 3) for y in x['kappa_ci95']]) for c, x in v.get('all', {}).items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['run', 'analyze']); ap.add_argument('--cap-usd', type=float, default=15.0)
    a = ap.parse_args(); run(a.cap_usd) if a.mode == 'run' else analyze()
