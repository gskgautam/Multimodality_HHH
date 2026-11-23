"""O3 deviation 1 (logged): one recovery request per Claude model for items whose image exceeds
Anthropic's 8,000-pixel side limit (HTTP 400 image_dimensions_too_large). The image is downscaled
(aspect ratio kept, longest side 7,680 px) for that provider only; prompt, model and decoding are
unchanged. Precedent: the Fig. 5 run's single logged recovery request for HB3-0082.
Appends one row per recovered item with a deviation note; the original failed row stays in the journal."""
import importlib.util, io, json
from pathlib import Path
from PIL import Image

E = (next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()) / 'Method/OrchCircuit/final')
sp = importlib.util.spec_from_file_location('gen', E / 'review/o3_run_candidates_v2.py'); g = importlib.util.module_from_spec(sp); sp.loader.exec_module(g)
recs = {r['external_case_id']: r for r in json.loads(g.INPUT.read_text())['records']}; t = g.template()
for slug in ('claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5'):
    op = g.OUTPUT_DIR / (slug + '.jsonl'); rows = g.read_jsonl(op)
    ok = {r['external_case_id'] for r in rows if r.get('candidate_text') and not r.get('error')}
    for cid in sorted({r['external_case_id'] for r in rows} - ok):
        if 'image_dimensions_too_large' not in next(r for r in rows if r['external_case_id'] == cid).get('error', ''):
            print('skip (different error)', slug, cid); continue
        r = recs[cid]; ip = g.SAMPLE_DIR / r['image_file']; im = Image.open(ip); w, h = im.size
        s = 7680 / max(w, h); im = im.convert('RGB').resize((round(w * s), round(h * s)), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, 'JPEG', quality=95); ib = buf.getvalue()
        fam, provider, model, ep, key, meta = g.load_config(slug); prompt = g.make_prompt(t, r)
        raw, u, rid = g.invoke(provider, model, ep, prompt, ib, 'image/jpeg', key)
        parsed, perr, choice, sel, rat, ans = g.parse(raw, r['options'])
        it, ot = int(u.get('input_tokens', 0) or 0), int(u.get('output_tokens', 0) or 0); ri, ro = g.RATES[provider]
        row = {'external_case_id': cid, 'source_record_id': r['source_record_id'], 'model_slug': slug, 'model_id': model, 'provider': provider,
               'attribute_stratum': r['attribute_stratum'], 'prompt_sha256': g.hashlib.sha256(prompt.encode()).hexdigest(), 'image_sha256': r['source_image_sha256'],
               'attempts': 1, 'attempt_errors': [], 'request_id': rid, 'raw_response': raw, 'parsed_response': parsed, 'parse_error': perr, 'choice_id': choice,
               'choice_text': sel, 'rationale': rat, 'candidate_text': ans, 'valid_choice': choice is not None, 'error': None if raw.strip() else 'empty',
               'deviation': f'O3 deviation 1: image downscaled from {w}x{h} to {im.size[0]}x{im.size[1]} for the Anthropic 8000-px side limit; one recovery request',
               'usage': {'input_tokens': it, 'output_tokens': ot, 'estimated_cost_usd': (it * ri + ot * ro) / 1e6, 'rate_input_per_million': ri, 'rate_output_per_million': ro}}
        with op.open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print('recovered', slug, cid, 'choice', choice, 'valid', row['valid_choice'])
