#!/usr/bin/env python3
"""Metadata-only schema, stratum, exact-text overlap, and duplicate audit."""
import collections, hashlib, json, re, time, urllib.parse, urllib.request, urllib.error
from pathlib import Path
ROOT=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()))
OUT=ROOT/'Method/OrchCircuit/final/figure3/common/source_audit/source_frame_audit.json'
CACHE=Path.home()/'.cache/multimodality/figure3_source_metadata'
CACHE.mkdir(parents=True,exist_ok=True)

def norm(x):return ' '.join(str(x or '').lower().split())
def get_rows(dataset,config,split):
    rows=[]; total=None; offset=0; length=100
    slug=dataset.replace('/','__')+'__'+split
    while total is None or offset<total:
        cache_path=CACHE/f"{slug}_{offset}.json"
        if cache_path.exists():
            obj=json.loads(cache_path.read_text())
        else:
            query=urllib.parse.urlencode({'dataset':dataset,'config':config,'split':split,'offset':offset,'length':length})
            req=urllib.request.Request('https://datasets-server.huggingface.co/rows?'+query,headers={'User-Agent':'OrchCircuit metadata audit/0.1'})
            for attempt in range(8):
                try:
                    with urllib.request.urlopen(req,timeout=60) as res: obj=json.loads(res.read())
                    cache_path.write_text(json.dumps(obj,separators=(',',':')))
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code not in {429,502,503,504} or attempt==7: raise
                    wait=int(exc.headers.get('Retry-After','0') or 0) or min(60,3*(2**attempt))
                    print(f"temporary HTTP {exc.code} on {dataset} offset {offset}; retrying after {wait}s",flush=True)
                    time.sleep(wait)
            time.sleep(1.5)
        total=obj.get('num_rows_total',len(obj.get('rows',[])))
        part=[x.get('row',x) for x in obj.get('rows',[])]
        if not part:break
        rows.extend(part);offset+=len(part)
        print(f"metadata {slug}: {min(offset,total)}/{total}",flush=True)
    return rows

def image_key(image):
    if isinstance(image,dict):
        for k in ('image_id','id','path','file_name','src','url'):
            v=image.get(k)
            if v is not None:return str(k)+':'+str(v)
    return None

ch3=json.loads((ROOT/'Datasets/Ch3EF/data/ch3ef_full1416/sample.json').read_text())['records']
ch3q={norm(r.get('query')) for r in ch3}
ch3_image_hashes=set()
for r in ch3:
    ims=r.get('image',[])
    if isinstance(ims,str):ims=[ims]
    for ref in ims:
        p=(ROOT/'Datasets/Ch3EF/data/ch3ef_full1416'/ref)
        if p.exists():ch3_image_hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())

vista=get_rows('mozhina/VistaQA','default','train')
safety=get_rows('EchoSafe-MLLM/MM-SafetyBench-plus-plus','default','test')
assert len(vista)==1157, f'VistaQA row-count changed: {len(vista)}'
assert len(safety)==2844, f'MM-SafetyBench++ row-count changed: {len(safety)}'
vista_q=[norm(r.get('question')) for r in vista]
safe_q=[norm(r.get('question')) for r in safety]
vista_images=[image_key(r.get('image')) for r in vista]
safe_images=[image_key(r.get('image')) for r in safety]

def question_group_summary(rows, question_field, dimensions):
    groups=collections.defaultdict(list)
    for r in rows:groups[norm(r.get(question_field))].append(r)
    out={'unique_exact_questions':len(groups),'repeated_question_groups':sum(len(v)>1 for v in groups.values()),
         'max_rows_in_one_exact_question_group':max(map(len,groups.values()))}
    for name,field in dimensions.items():
        out[f'groups_spanning_multiple_{name}']=sum(len({str(r.get(field)) for r in values})>1 for values in groups.values())
    return out

def duplicate_summary(rows,questions,images):
    by_pair=collections.Counter((images[i],questions[i]) for i in range(len(rows)))
    by_img=collections.Counter(x for x in images if x is not None)
    by_q=collections.Counter(q for q in questions if q)
    return {'rows':len(rows),'exact_image_question_groups':len(by_pair),'duplicate_image_question_rows':sum(n-1 for n in by_pair.values() if n>1),
            'rows_with_repeated_image_key':sum(n for n in by_img.values() if n>1),'unique_nonempty_image_keys':len(by_img),
            'rows_with_repeated_exact_question':sum(n for n in by_q.values() if n>1),'unique_exact_questions':len(by_q),
            'rows_without_extractable_image_key':sum(x is None for x in images)}

safety_counts=collections.Counter((str(r.get('category')),str(r.get('label')),str(r.get('mode'))) for r in safety)
result={
 'audit_type':'metadata-only; no external image bytes downloaded; no external model outputs generated',
 'ch3ef_development':{'records':len(ch3),'unique_normalized_questions':len(ch3q),'local_image_sha256_count':len(ch3_image_hashes)},
 'vistaqa':{'dataset':'mozhina/VistaQA','split':'train','schema_keys':sorted(vista[0]),'rows':len(vista),
   'hallucination_flag_counts':dict(collections.Counter(str(r.get('hallucination')) for r in vista)),
   'task_counts':dict(collections.Counter(str(r.get('task_type')) for r in vista)),
   'domain_counts':dict(collections.Counter(str(r.get('task_domain')) for r in vista)),
   'exact_normalized_question_overlap_with_Ch3Ef':len(set(vista_q)&ch3q),
   'duplicates':duplicate_summary(vista,vista_q,vista_images),
   'exact_question_grouping':question_group_summary(vista,'question',{'hallucination_flags':'hallucination','task_types':'task_type','domains':'task_domain'})},
 'mm_safetybench_plus_plus':{'dataset':'EchoSafe-MLLM/MM-SafetyBench-plus-plus','split':'test','schema_keys':sorted(safety[0]),'rows':len(safety),
   'label_counts':dict(collections.Counter(str(r.get('label')) for r in safety)),
   'category_counts':dict(collections.Counter(str(r.get('category')) for r in safety)),
   'mode_counts':dict(collections.Counter(str(r.get('mode')) for r in safety)),
   'category_label_mode_counts':{'|'.join(k):v for k,v in sorted(safety_counts.items())},
   'explicit_pair_identifier_present':any(k in safety[0] for k in ('pair_id','pairid','pair_group','source_pair_id')),
   'exact_normalized_question_overlap_with_Ch3Ef':len(set(safe_q)&ch3q),
   'duplicates':duplicate_summary(safety,safe_q,safe_images),
   'exact_question_grouping':question_group_summary(safety,'question',{'labels':'label','modes':'mode','categories':'category'})},
 'image_overlap':'Not assessed at pixel level: public image bytes were intentionally not fetched. Exact prompt overlap is only a preliminary screen.',
 'rights_gate':'Dataset-card CC BY 4.0 entries were verified separately; check image-level upstream rights before test sampling or public redistribution.'}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,indent=2,ensure_ascii=False))
