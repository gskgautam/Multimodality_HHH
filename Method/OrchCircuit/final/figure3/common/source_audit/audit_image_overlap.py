#!/usr/bin/env python3
"""Transient public-image duplicate audit; persists hashes only, never image bytes."""
import hashlib,io,json,time,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from PIL import Image
ROOT=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd())); CACHE=Path.home()/'.cache/multimodality/figure3_source_metadata'
OUT=ROOT/'Method/OrchCircuit/final/figure3/common/source_audit/image_overlap_audit.json'
REV='cf0cb9857e5f0969a7fc2c8efb3ae9d4816ccc1a'
def load_rows(pattern):
 rows=[]
 for p in sorted(CACHE.glob(pattern)): rows.extend(x.get('row',x) for x in json.load(open(p)).get('rows',[]))
 return rows
def d_hash(data):
 im=Image.open(io.BytesIO(data)).convert('L').resize((9,8),Image.Resampling.LANCZOS); px=list(im.getdata()); bits=0
 for y in range(8):
  for x in range(8): bits=(bits<<1)|(px[y*9+x]>px[y*9+x+1])
 return f'{bits:016x}'
def norm_pixel_hash(data):
 im=Image.open(io.BytesIO(data)).convert('RGB'); im.thumbnail((128,128)); c=Image.new('RGB',(128,128),'white'); c.paste(im,((128-im.width)//2,(128-im.height)//2))
 return hashlib.sha256(c.tobytes()).hexdigest()
def fingerprints(data): return hashlib.sha256(data).hexdigest(),d_hash(data),norm_pixel_hash(data),len(data)
def fetch(url):
 last=None
 for attempt in range(5):
  try:
   req=urllib.request.Request(url,headers={'User-Agent':'OrchCircuit image overlap audit/0.2'})
   with urllib.request.urlopen(req,timeout=45) as r: b=r.read()
   if len(b)<100: raise ValueError('image payload too small')
   return b
  except Exception as e:
   last=e
   if attempt<4: time.sleep(min(20,2**attempt))
 raise RuntimeError(f'{type(last).__name__}: {last}')
def local_refs():
 records=json.load(open(ROOT/'Datasets/Ch3EF/data/ch3ef_full1416/sample.json'))['records']; out=[]
 for ri,row in enumerate(records):
  refs=row.get('image',[])
  if isinstance(refs,str): refs=[refs]
  for ref in refs:
   p=ROOT/'Datasets/Ch3EF/data/ch3ef_full1416'/ref
   if p.exists():
    b=p.read_bytes(); sha,dh,ph,n=fingerprints(b); out.append({'record_index':ri,'path':str(p),'sha256':sha,'dhash':dh,'normalized_pixel_sha256':ph})
 return out
def run(name,rows,url_fn,local):
 jobs=[(i,row,url_fn(row)) for i,row in enumerate(rows) if url_fn(row)]; result=[]; failures=[]
 print(f'{name}: fetching {len(jobs)} images',flush=True)
 with ThreadPoolExecutor(max_workers=8) as pool:
  fs={pool.submit(fetch,u):(i,row) for i,row,u in jobs}
  for n,f in enumerate(as_completed(fs),1):
   i,row=fs[f]
   try:
    raw=f.result(); sha,dh,ph,size=fingerprints(raw)
    image=row.get('image') or {}
    result.append({'row_index':i,'image_id':image.get('image_id'),'file_name':image.get('file_name'),'sha256':sha,'dhash':dh,'normalized_pixel_sha256':ph,'bytes':size})
   except Exception as e: failures.append({'row_index':i,'image_id':(row.get('image') or {}).get('image_id'),'file_name':(row.get('image') or {}).get('file_name'),'error':str(e)[:240]})
   if n%100==0 or n==len(jobs): print(f'{name}: {n}/{len(jobs)}',flush=True)
 local_sha={x['sha256'] for x in local}; local_pixel={x['normalized_pixel_sha256'] for x in local}; local_dh={x['dhash'] for x in local}
 for x in result:
  x['exact_file_overlap']=x['sha256'] in local_sha
  x['normalized_pixel_overlap']=x['normalized_pixel_sha256'] in local_pixel
  x['dhash_distance_0_candidate']=x['dhash'] in local_dh
 from collections import Counter,defaultdict
 def dup_summary(key):
  g=defaultdict(list)
  for x in result:g[x[key]].append(x['row_index'])
  return {'unique_values':len(g),'duplicate_groups':sum(len(v)>1 for v in g.values()),'duplicate_rows':sum(len(v)-1 for v in g.values() if len(v)>1),'groups':{k:v for k,v in g.items() if len(v)>1}}
 return {'metadata_rows':len(rows),'images_attempted':len(jobs),'images_hashed':len(result),'failures':failures,'exact_file_overlap_count':sum(x['exact_file_overlap'] for x in result),'normalized_pixel_overlap_count':sum(x['normalized_pixel_overlap'] for x in result),'dhash_distance_0_count':sum(x['dhash_distance_0_candidate'] for x in result),'unique_content_sha256':len({x['sha256'] for x in result}),'unique_normalized_pixel_sha256':len({x['normalized_pixel_sha256'] for x in result}),'duplicate_file_hashes':dup_summary('sha256'),'duplicate_normalized_pixels':dup_summary('normalized_pixel_sha256'),'rows':result}
vista=load_rows('mozhina__VistaQA__train_*.json'); safe=load_rows('EchoSafe-MLLM__MM-SafetyBench-plus-plus__test_*.json'); local=local_refs()
print(f'Ch3Ef local image instances={len(local)}; unique file hashes={len({x["sha256"] for x in local})}; unique normalized pixels={len({x["normalized_pixel_sha256"] for x in local})}',flush=True)
def vu(r):
 n=(r.get('image') or {}).get('file_name')
 return f'https://huggingface.co/datasets/mozhina/VistaQA/resolve/{REV}/VistaQA/{n}' if n else None
def su(r): return (r.get('image') or {}).get('src')
out={'method':'Transient HTTP fetch. Exact-file SHA-256, 128x128 white-padded RGB pixel SHA-256, and 64-bit dHash. No public image bytes saved.','ch3ef_local_image_instances':len(local),'ch3ef_unique_file_hashes':len({x['sha256'] for x in local}),'ch3ef_unique_normalized_pixel_hashes':len({x['normalized_pixel_sha256'] for x in local}),'ch3ef_images':local,'vistaqa':run('VISTAQA',vista,vu,local),'mm_safetybench_plus_plus':run('MM-SafetyBench++',safe,su,local)}
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
for k in ['vistaqa','mm_safetybench_plus_plus']:
 v=out[k]; print(k,json.dumps({x:v[x] for x in ['metadata_rows','images_attempted','images_hashed','failures','exact_file_overlap_count','normalized_pixel_overlap_count','dhash_distance_0_count','unique_content_sha256','unique_normalized_pixel_sha256','duplicate_file_hashes','duplicate_normalized_pixels']},ensure_ascii=False),flush=True)
print('Saved',OUT,flush=True)
