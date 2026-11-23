"""O3: draw and freeze an untouched confirmatory sample (no model calls; answers never read).

Revision 2 (2026-11-23, before any generation): all-MCQ design chosen by the PI. 300 HumaniBench
task3 items, 60 per attribute (Age, Ethnicity, Gender, Occupation, Sport). The open-ended task7
stratum from revision 1 is dropped because the selector, prompt template and human rubric are MCQ-only.

Exclusions: every record/image/question used in the Figure 3e/f (now Fig. 5) sample, every image whose
SHA-256 appears in Ch3EF, the single SES item, duplicate images and duplicate questions.
Writes Datasets/HumaniBench/confirmatory_v2_PRIVATE/{manifest.json, inference_prompts.json, images/, freeze_hashes.json}
in the same packet format as sample_200_task3, so the existing pipeline scripts can read it.
"""
import ast, hashlib, json, random, re
from pathlib import Path
import pandas as pd

R = (next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd())); HB = R / 'Datasets/HumaniBench/confirmatory_figure3ef_PRIVATE'
E = R / 'Method/OrchCircuit/final'; OUT = R / 'Datasets/HumaniBench/confirmatory_v2_PRIVATE'
SEED = 20261123; PER = 60
sha = lambda b: hashlib.sha256(b).hexdigest()
norm = lambda q: re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', str(q).lower())).strip()
assert not (OUT / 'candidate_outputs_PRIVATE').exists(), 'generation already started; do not re-freeze'
OUT.mkdir(mode=0o700, exist_ok=True); (OUT / 'images').mkdir(mode=0o700, exist_ok=True)
for f in ('manifest.json', 'freeze_hashes.json'):
    if (OUT / f).exists() and '"revision": 2' not in (OUT / f).read_text() and 'revision 2' not in (OUT / f).read_text():
        (OUT / f).rename(OUT / f.replace('.json', '_rev1_superseded.json'))
for old in (OUT / 'images').glob('*'):
    old.unlink()

used = json.loads((HB / 'sample_200_task3/inference_prompts.json').read_text())['records']
used_ids = {str(r['source_record_id']) for r in used}
used_img = {r['source_image_sha256'] for r in used}
used_q = {norm(r['question']) for r in used}
ch3ef_img = {h for v in json.loads((E / 'runs/fullset_group5fold/folds.json').read_text())['image_sha256_by_order'].values() for h in (v if isinstance(v, list) else [v])}


def parse_options(s):
    opts = ast.literal_eval(s) if isinstance(s, str) else list(s)
    return [re.sub(r'^[A-Z]\.\s*', '', str(o)).strip() for o in opts]


df = pd.read_parquet(HB / 'task3.parquet')
pool = []
for _, r in df.iterrows():
    b = r['image']['bytes']; ih = sha(b); q = norm(r['Question'])
    if str(r['ID']) in used_ids or ih in used_img or ih in ch3ef_img or q in used_q or r['Attribute'] == 'SES':
        continue
    try:
        opts = parse_options(r['Options'])
    except Exception:
        continue
    if not 2 <= len(opts) <= 6 or len(set(opts)) != len(opts):
        continue
    pool.append({'id': str(r['ID']), 'attribute': r['Attribute'], 'question': str(r['Question']).strip(), 'options': opts, 'img': b, 'ih': ih, 'qh': sha(q.encode())})

rng = random.Random(SEED); seen_i, seen_q = set(), set(); chosen = []
for att in sorted({p['attribute'] for p in pool}):
    cand = [p for p in pool if p['attribute'] == att]; rng.shuffle(cand); got = 0
    for p in cand:
        if got == PER:
            break
        if p['ih'] in seen_i or p['qh'] in seen_q:
            continue
        seen_i.add(p['ih']); seen_q.add(p['qh']); chosen.append(p); got += 1
    assert got == PER, (att, got)
rng.shuffle(chosen)


def ext(b):
    return '.png' if b.startswith(b'\x89PNG') else '.webp' if b[:4] == b'RIFF' and b[8:12] == b'WEBP' else '.jpg'


records = []
for i, p in enumerate(chosen, 1):
    fn = f'images/{i:04d}{ext(p["img"])}'; (OUT / fn).write_bytes(p['img'])
    records.append({'external_case_id': f'CV2-{i:04d}', 'source_record_id': p['id'], 'attribute_stratum': p['attribute'],
                    'question': p['question'], 'options': p['options'], 'image_file': fn, 'source_image_sha256': p['ih'],
                    'prompt_sha256': sha(json.dumps({'q': p['question'], 'o': p['options']}, ensure_ascii=False).encode()), 'answer_hidden': True})
strat = {a: PER for a in sorted({p['attribute'] for p in chosen})}
packet = {'dataset': 'HumaniBench task3_Multiple_Choice_VQA test', 'license': 'CC BY-NC-SA 4.0', 'seed': SEED, 'stratification': strat, 'records': records}
ptxt = json.dumps(packet, indent=1, ensure_ascii=False) + '\n'; (OUT / 'inference_prompts.json').write_text(ptxt)
man = {'purpose': 'Confirmatory, untouched human-rated test (open item O3), revision 2: all-MCQ', 'seed': SEED, 'n': len(records),
       'strata': strat, 'eligible_pool_task3': len(pool), 'answers_read': False,
       'exclusions': ['Fig. 5 (old Figure 3e/f) record IDs, images and normalized questions', 'any image SHA-256 present in Ch3EF',
                      'SES attribute (single item)', 'duplicate images/questions within sample', 'unparseable or duplicate options'],
       'records': [{k: r[k] for k in ('external_case_id', 'source_record_id', 'attribute_stratum', 'source_image_sha256')} for r in records]}
mtxt = json.dumps(man, indent=1) + '\n'; (OUT / 'manifest.json').write_text(mtxt)
frozen = {'revision': 2, 'manifest.json': sha(mtxt.encode()), 'inference_prompts.json': sha(ptxt.encode())}
for p in [E / 'shared/human_rating_pipeline/scripts/analyze_figure3e_f.py', E / 'shared/human_rating_pipeline/scripts/apply_humanibench_selector.py',
          E / 'shared/human_rating_pipeline/scripts/run_humanibench_candidates.py', E / 'review/PREREGISTRATION_CONFIRMATORY_V2.md',
          E / 'review/o3_run_candidates_v2.py']:
    frozen[str(p.relative_to(R))] = sha(p.read_bytes()) if p.exists() else 'MISSING'
frozen['selector_checkpoints'] = {str(p.relative_to(R)): sha(p.read_bytes()) for p in sorted(R.glob('Datasets/**/orchcircuit_selector.pt'))}
(OUT / 'freeze_hashes.json').write_text(json.dumps(frozen, indent=1) + '\n')
print(json.dumps({k: v for k, v in man.items() if k != 'records'}, indent=1)); print(json.dumps(frozen, indent=1))
