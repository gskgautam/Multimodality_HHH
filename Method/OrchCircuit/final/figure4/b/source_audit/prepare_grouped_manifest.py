#!/usr/bin/env python3
"""Build the MM-SafetyBench++ external-test manifest and verify source pairing."""
import collections, glob, hashlib, json, urllib.request
from pathlib import Path

ROOT=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()))
CACHE=Path.home()/'.cache/multimodality/figure3_source_metadata'
OUT=ROOT/'Method/OrchCircuit/final/figure4/b/source_audit'
CATEGORY_FILES={
 'Illegal_Activity':'01-Illegal_Activity',
 'HateSpeech':'02-HateSpeech',
 'Malware_Generation':'03-Malware_Generation',
 'Physical_Harm':'04-Physical_Harm',
 'Fraud':'06-Fraud',
 'Sex':'07-Sex'
}
rows=[]
for path in glob.glob(str(CACHE/'EchoSafe-MLLM__MM-SafetyBench-plus-plus__test_*.json')):
 block=json.loads(Path(path).read_text())
 rows.extend((int(x['row_idx']),x['row']) for x in block.get('rows',[]))
rows.sort(key=lambda x:x[0])
if len(rows)!=2844 or [i for i,_ in rows]!=list(range(2844)):
 raise SystemExit(f'Expected 2844 contiguous rows; got {len(rows)}')

def norm(x): return ' '.join(str(x or '').lower().split())
by=collections.defaultdict(list)
for idx,row in rows: by[(str(row['category']),str(row['label']))].append((idx,row))
manifest=[]; category_counts={}; source_file_hashes={}; checks=0
for category,filename in CATEGORY_FILES.items():
 for label,folder in [('unsafe','processed_questions'),('safe','processed_questions_safe')]:
  url=f'https://raw.githubusercontent.com/EchoSafe-MLLM/EchoSafe/master/{folder}/{filename}.json'
  request=urllib.request.Request(url,headers={'User-Agent':'OrchCircuit external transfer audit'})
  raw=urllib.request.urlopen(request,timeout=45).read()
  source_file_hashes[f'{folder}/{filename}.json']=hashlib.sha256(raw).hexdigest()
  source=json.loads(raw)
  items=by[(category,label)]
  if len(items)!=2*len(source):
   raise SystemExit(f'Count mismatch for {category}/{label}: {len(items)} HF rows vs {len(source)} source items')
  modes=[row['mode'] for _,row in items]
  if any(modes[i:i+2]!=['Gen','GenOCR'] for i in range(0,len(modes),2)):
   raise SystemExit(f'Gen/GenOCR adjacency mismatch for {category}/{label}')
  for pos in range(len(source)):
   source_item=source[str(pos)]
   expected=norm(source_item.get('Rephrased Question(SD)'))
   for off in (0,1):
    idx,row=items[2*pos+off]
    if norm(row.get('question'))!=expected:
     raise SystemExit(f'Source question alignment mismatch at {category}/{label}/{pos}/{row["mode"]}')
    checks+=1
    manifest.append({
     'dataset':'EchoSafe-MLLM/MM-SafetyBench-plus-plus',
     'dataset_revision':'128ea182089b677a8d48d97c9f86425713351ac2',
     'split':'test','dataset_row_idx':idx,
     'pair_id':f'{category}:{pos:04d}',
     'category':category,'intent_label':label,
     'mode':row['mode'],
     'question_sha256':hashlib.sha256(expected.encode()).hexdigest(),
     'image_ref':f'test-row:{idx}'
    })
  category_counts.setdefault(category,{})[label]={'source_items':len(source),'export_rows':len(items)}
pair_rows=collections.defaultdict(list)
for row in manifest: pair_rows[row['pair_id']].append(row)
for pair_id,group in pair_rows.items():
 if len(group)!=4 or {x['intent_label'] for x in group}!={'safe','unsafe'} or {x['mode'] for x in group}!={'Gen','GenOCR'}:
  raise SystemExit(f'Incomplete matched group: {pair_id}')
OUT.mkdir(parents=True,exist_ok=True)
manifest.sort(key=lambda r:r['dataset_row_idx'])
mp=OUT/'mm_safetybench_external_test_manifest.jsonl'
mp.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in manifest))
pairing={
 'audit':'source-alignment verification against EchoSafe authors\' processed_questions and processed_questions_safe files',
 'source_repo':'https://github.com/EchoSafe-MLLM/EchoSafe',
 'source_files_sha256':source_file_hashes,
 'source_row_question_alignments_verified':checks,
 'hf_rows':len(manifest),'matched_source_items':len(pair_rows),
 'pair_key':'category + original source dictionary key position; each source item has Gen and GenOCR exports on each intent side',
 'validation':'PASS: all 2,844 exported question/mode rows align to source files; every source key has one safe and one unsafe counterpart and both modes.',
 'caveat':'The Hugging Face export itself has no pair_id. Pair groups are reconstructed from exact source-prompt alignment and category/label ordering; the benchmark paper describes each safe item as a minimally modified counterpart of an unsafe item.'
}
(OUT/'mm_safetybench_pairing_audit.json').write_text(json.dumps(pairing,indent=2)+'\n')
summary={
 'dataset':'EchoSafe-MLLM/MM-SafetyBench-plus-plus',
 'dataset_revision':'128ea182089b677a8d48d97c9f86425713351ac2',
 'split':'test','rows':len(manifest),'matched_source_pairs':len(pair_rows),
 'category_counts':category_counts,
 'pair_cluster':'one source item with unsafe/safe intent x Gen/GenOCR; four records per cluster',
 'pair_mapping_validation':'PASS; see mm_safetybench_pairing_audit.json',
 'test_handling':'All external groups remain test-only. Do not train, tune, calibrate, or select models on this split.',
 'bootstrap_unit':'pair_id',
 'image_urls_saved':False,
 'manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest()
}
(OUT/'mm_safetybench_manifest_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
