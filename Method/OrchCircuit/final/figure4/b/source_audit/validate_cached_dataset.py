#!/usr/bin/env python3
"""Offline integrity check for the pinned MM-SafetyBench++ test split."""
import collections, hashlib, json
from pathlib import Path
from datasets import load_dataset
REV='128ea182089b677a8d48d97c9f86425713351ac2'
OUT=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()) / "Method/OrchCircuit/final/figure4/b/source_audit/mm_safetybench_data_integrity.json")
ds=load_dataset('EchoSafe-MLLM/MM-SafetyBench-plus-plus',split='test',revision=REV,cache_dir='${HOME}/.cache/huggingface/datasets')
if len(ds)!=2844: raise SystemExit(f'Expected 2844 rows, got {len(ds)}')
expected={'image','question','category','label','mode'}
if set(ds.column_names)!=expected: raise SystemExit(f'Unexpected columns: {ds.column_names}')
sizes=[];bad=[];counts=collections.Counter()
for i,row in enumerate(ds):
 try:
  image=row['image']
  if image is None: raise ValueError('null image')
  w,h=image.size
  if w<1 or h<1: raise ValueError(f'invalid dimensions {(w,h)}')
  sizes.append((w,h))
 except Exception as exc: bad.append({'row':i,'error':type(exc).__name__+': '+str(exc)})
 counts['category:'+str(row['category'])]+=1
 counts['intent:'+str(row['label'])]+=1
 counts['mode:'+str(row['mode'])]+=1
if bad: raise SystemExit(json.dumps({'bad_images':bad[:20],'count':len(bad)},indent=2))
summary={
 'dataset':'EchoSafe-MLLM/MM-SafetyBench-plus-plus','revision':REV,'split':'test',
 'rows':len(ds),'columns':ds.column_names,'decoded_images':len(sizes),'image_decode_failures':len(bad),
 'width_range':[min(w for w,h in sizes),max(w for w,h in sizes)],
 'height_range':[min(h for w,h in sizes),max(h for w,h in sizes)],
 'source_content_sha256':hashlib.sha256(''.join(f'{w}x{h};' for w,h in sizes).encode()).hexdigest(),
 'counts':dict(sorted(counts.items())),
 'cached_copy':'Hugging Face cache outside repository; source images are not added to Git.'
}
OUT.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
