#!/usr/bin/env python3
"""O3 confirmatory v2: resumable zero-shot candidate generation for the frozen 300-case HumaniBench test.
Generated from run_humanibench_candidates.py by review/make_o3_pipeline.py (sample path/size, per-GPU Ollama endpoint, per-worker manifest)."""
import argparse, base64, datetime, hashlib, json, mimetypes, os, stat, string, time, urllib.error, urllib.request
from pathlib import Path
ROOT=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()))
SAMPLE_DIR=ROOT/"Datasets/HumaniBench/confirmatory_v2_PRIVATE"
INPUT=SAMPLE_DIR/"inference_prompts.json"
OUTPUT_DIR=SAMPLE_DIR/"candidate_outputs_PRIVATE"
PROMPT_FILE=ROOT/"Datasets/Ch3EF/ch3ef-evaluation-prompt.md"
SLUGS=["lfm3-vl-2b","qwen3.5-2b","gemma4-e2b","qwen2.5-vl-7b","qwen3-vl-8b","minicpm-v-4.5-8b","gemma3-12b","gemma-4-12b","pixtral-12b","gemma3-27b","qwen3.6-27b","qwen3.8-27b","qwen3.5-35b-a3b","qwen3.6-35b-a3b","minicpm-v-4.7-35b-a3b","gpt-6.1-terra","gpt-6.1-sol","gpt-6-luna","gemini-4-argon","gemini-3.8-flash","gemini-3.6-flash","claude-opus-5-5","claude-sonnet-5-5","claude-haiku-5-5"]
OPEN=set(SLUGS[:15])
KEYS={"openai":ROOT.parent/".config/multimodality/openai_api_key","google":ROOT.parent/".config/multimodality/google_gemini_api_key","anthropic":ROOT.parent/".config/multimodality/anthropic_api_key"}
RATES={"openai":(1.0,5.0),"google":(2.0,10.0),"anthropic":(2.0,10.0)}
MAX_TOKENS=512
MAX_ATTEMPTS=3

def read_jsonl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def metadata(slug):
    family="Open" if slug in OPEN else "Closed"
    p=ROOT/"Baselines"/family/slug/"runs/ch3ef-full1416/run_metadata.json"
    return family,json.loads(p.read_text())
def template():
    t=PROMPT_FILE.read_text()
    return t.split("## Prompt\n",1)[1].split("\n## Run settings",1)[0].strip()
def mime(data,path):
    if data.startswith(b"\x89PNG\r\n\x1a\n"): return "image/png"
    if data.startswith(b"RIFF") and data[8:12]==b"WEBP": return "image/webp"
    if data.startswith(b"\xff\xd8\xff"): return "image/jpeg"
    return mimetypes.guess_type(str(path))[0] or "image/jpeg"
def make_prompt(t,r):
    opts=r["options"]
    if not isinstance(opts,list) or not all(isinstance(x,str) for x in opts): raise ValueError("Bad options "+r["external_case_id"])
    labels=list(string.ascii_uppercase[:len(opts)])
    choices="\n".join(f"{a}. {b}" for a,b in zip(labels,opts))
    return t.replace("{query}",r["question"]).replace("{options}",choices)
def post(url,body,headers,timeout=300):
    q=urllib.request.Request(url,data=json.dumps(body).encode(),headers={**headers,"Content-Type":"application/json"},method="POST")
    try:
        with urllib.request.urlopen(q,timeout=timeout) as x: return json.loads(x.read().decode()),dict(x.headers)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from None
def extract(provider,d):
    if provider=="ollama": return str(d.get("message",{}).get("content",""))
    if provider=="pixtral":
        c=d.get("choices") or []
        return str(c[0].get("message",{}).get("content","")) if c else ""
    if provider=="openai":
        return "\n".join(y.get("text","") for x in d.get("output",[]) for y in x.get("content",[]) if y.get("type") in ("output_text","refusal") and y.get("text"))
    if provider=="google":
        c=d.get("candidates") or []
        if c: return "\n".join(str(y.get("text","")) for y in c[0].get("content",{}).get("parts",[]) if y.get("text"))
        return "[provider_blocked:"+str(d.get("promptFeedback",{}).get("blockReason","unspecified"))+"]"
    if provider=="anthropic": return "\n".join(str(y.get("text","")) for y in d.get("content",[]) if y.get("type")=="text")
    return ""
def parse(raw,opts):
    txt=raw.strip()
    parsed=None;err=None
    try:
        parsed=json.loads(txt)
        if not isinstance(parsed,dict): raise ValueError("not an object")
    except Exception as e: err=str(e)
    labels=list(string.ascii_uppercase[:len(opts)])
    choice=parsed.get("choice_id") if isinstance(parsed,dict) else None
    rationale=str(parsed.get("rationale") or "") if isinstance(parsed,dict) else ""
    if isinstance(choice,int) and not isinstance(choice,bool) and 1<=choice<=len(opts): choice=labels[choice-1]
    if isinstance(choice,str):
        choice=choice.strip().upper()
        if choice not in labels:
            matches=[i for i,x in enumerate(opts) if choice.casefold()==x.strip().casefold()]
            choice=labels[matches[0]] if len(matches)==1 else None
    selected=opts[labels.index(choice)] if choice in labels else None
    if selected is not None: answer=selected+("\n"+rationale.strip() if rationale.strip() else "")
    elif isinstance(parsed,dict): answer=str(parsed.get("answer") or parsed.get("response") or parsed.get("refusal") or raw).strip()
    else: answer=raw.strip()
    return parsed,err,choice,selected,rationale,answer
def invoke(provider,model,endpoint,prompt,ib,mt,key):
    b64=base64.b64encode(ib).decode()
    if provider=="ollama":
        msgs=[]
        if model.startswith("gemma4:"): msgs.append({"role":"system","content":"<|channel>thought\n<channel|>Answer directly with only the requested JSON."})
        msgs.append({"role":"user","content":prompt,"images":[b64]})
        opt={"temperature":0}
        if model.startswith("gemma4:"): opt["num_predict"]=512
        d,_=post(endpoint,{"model":model,"messages":msgs,"stream":False,"think":False,"format":"json","options":opt},{},timeout=900)
        raw=extract(provider,d);u={"input_tokens":d.get("prompt_eval_count",0),"output_tokens":d.get("eval_count",0)};rid=None
    elif provider=="pixtral":
        uri=f"data:{mt};base64,{b64}"
        body={"model":model,"messages":[{"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":uri}}]}],"temperature":0,"max_tokens":MAX_TOKENS,"response_format":{"type":"json_object"}}
        d,_=post(endpoint,body,{},timeout=900);raw=extract(provider,d);u=d.get("usage",{});rid=d.get("id")
        u={"input_tokens":u.get("prompt_tokens",0),"output_tokens":u.get("completion_tokens",0)}
    elif provider=="openai":
        body={"model":model,"input":[{"role":"user","content":[{"type":"input_text","text":prompt},{"type":"input_image","image_url":f"data:{mt};base64,{b64}","detail":"high"}]}],"max_output_tokens":MAX_TOKENS}
        d,_=post("https://api.openai.com/v1/responses",body,{"Authorization":"Bearer "+key});raw=extract(provider,d);u=d.get("usage",{});rid=d.get("id")
        u={"input_tokens":u.get("input_tokens",0),"output_tokens":u.get("output_tokens",0)}
    elif provider=="google":
        conf={"maxOutputTokens":MAX_TOKENS,"responseMimeType":"application/json","thinkingConfig":{"thinkingLevel":"low"}}
        if model!="gemini-3.8-flash": conf["temperature"]=0
        body={"contents":[{"role":"user","parts":[{"text":prompt},{"inline_data":{"mime_type":mt,"data":b64}}]}],"generationConfig":conf}
        d,_=post("https://generativelanguage.googleapis.com/v1beta/models/"+model+":generateContent",body,{"x-goog-api-key":key});raw=extract(provider,d);u=d.get("usageMetadata",{});rid=d.get("responseId")
        u={"input_tokens":u.get("promptTokenCount",0),"output_tokens":u.get("candidatesTokenCount",0)+u.get("thoughtsTokenCount",0)}
    else:
        body={"model":model,"max_tokens":MAX_TOKENS,"system":"Return only the requested JSON object.","messages":[{"role":"user","content":[{"type":"text","text":prompt},{"type":"image","source":{"type":"base64","media_type":mt,"data":b64}}]}]}
        d,_=post("https://api.anthropic.com/v1/messages",body,{"x-api-key":key,"anthropic-version":"2023-06-01"});raw=extract(provider,d);u=d.get("usage",{});rid=d.get("id")
        u={"input_tokens":u.get("input_tokens",0),"output_tokens":u.get("output_tokens",0)}
    return raw,u,rid
def load_config(slug):
    fam,m=metadata(slug);model=str(m["model"]);ep=str(m.get("endpoint",""))
    if fam=="Open" and os.environ.get("OLLAMA_OVERRIDE") and "127.0.0.1:11435" in ep: ep=ep.replace("127.0.0.1:11435",os.environ["OLLAMA_OVERRIDE"])
    if fam=="Open":
        provider="pixtral" if "8001" in ep else "ollama"
        return fam,provider,model,("http://127.0.0.1:8001/v1/chat/completions" if provider=="pixtral" else ep),None,m
    provider="openai" if slug.startswith("gpt-") else "google" if slug.startswith("gemini-") else "anthropic"
    p=KEYS[provider];mode=stat.S_IMODE(p.stat().st_mode)
    if mode&0o077: raise RuntimeError("API key file must have permission 600: "+str(p))
    return fam,provider,model,ep,p.read_text().strip(),m
def spend(outdir):
    total=0.0
    for p in outdir.glob("*.jsonl"):
        for r in read_jsonl(p): total+=float(r.get("usage",{}).get("estimated_cost_usd",0) or 0)
    return total
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--models",help="comma-separated slugs; default runs all 24")
    ap.add_argument("--budget-usd",type=float,default=12.0)
    a=ap.parse_args()
    OUTPUT_DIR.mkdir(mode=0o700,parents=True,exist_ok=True);os.chmod(SAMPLE_DIR,0o700)
    data=json.loads(INPUT.read_text());records=data["records"]
    if len(records)!=300: raise SystemExit("Expected frozen 300 cases")
    t=template();models=SLUGS if not a.models else [x.strip() for x in a.models.split(",") if x.strip()]
    if set(models)-set(SLUGS): raise SystemExit("Unknown slug(s): "+str(sorted(set(models)-set(SLUGS))))
    mrun={}
    for slug in models:
        fam,p,m,ep,k,meta=load_config(slug)
        mrun[slug]={"family":fam,"provider":p,"model":m,"endpoint":ep,"runtime":meta.get("runtime")}
    manifest={"dataset":"HumaniBench task3_Multiple_Choice_VQA test (O3 confirmatory v2)","ollama_endpoint_override":os.environ.get("OLLAMA_OVERRIDE"),"sample_sha256":sha(INPUT),"prompt_template_sha256":hashlib.sha256(t.encode()).hexdigest(),"model_runs":mrun,"budget_cap_usd":a.budget_usd,"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"answer_and_reasoning_sent":False}
    mp=OUTPUT_DIR/("run_manifest_"+os.environ.get("O3_WORKER","main")+".json")
    if not mp.exists(): mp.write_text(json.dumps(manifest,indent=2)+"\n")
    elif json.loads(mp.read_text()).get("sample_sha256")!=manifest["sample_sha256"]: raise SystemExit("Manifest/sample mismatch")
    print(f"records={len(records)} models={len(models)} prior_spend_usd={spend(OUTPUT_DIR):.4f} cap_usd={a.budget_usd:.2f}",flush=True)
    for slug in models:
        fam,provider,model,endpoint,key,meta=load_config(slug);op=OUTPUT_DIR/(slug+".jsonl")
        done={r.get("external_case_id") for r in read_jsonl(op) if r.get("candidate_text") and not r.get("error")}
        print(f"START {slug} {provider}/{model} resumed={len(done)}/{len(records)}",flush=True)
        with op.open("a",encoding="utf-8") as stream:
            for j,r in enumerate(records,1):
                if r["external_case_id"] in done: continue
                if spend(OUTPUT_DIR)>=a.budget_usd: raise SystemExit(f"Generation cap reached at USD {spend(OUTPUT_DIR):.4f}")
                prompt=make_prompt(t,r);ip=SAMPLE_DIR/r["image_file"];ib=ip.read_bytes();mt=mime(ib,ip)
                raw="";u={};rid=None;errs=[]
                for attempt in range(1,MAX_ATTEMPTS+1):
                    try:
                        raw,u,rid=invoke(provider,model,endpoint,prompt,ib,mt,key)
                        if raw.strip(): break
                        raise RuntimeError("empty response")
                    except Exception as e:
                        errs.append(str(e))
                        if attempt<MAX_ATTEMPTS: time.sleep(2**attempt)
                row={"external_case_id":r["external_case_id"],"source_record_id":r["source_record_id"],"model_slug":slug,"model_id":model,"provider":provider,"attribute_stratum":r["attribute_stratum"],"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),"image_sha256":r["source_image_sha256"],"attempts":min(len(errs)+1,MAX_ATTEMPTS),"attempt_errors":errs,"request_id":rid,"raw_response":raw}
                if raw.strip():
                    parsed,perr,choice,selected,rationale,answer=parse(raw,r["options"])
                    row.update({"parsed_response":parsed,"parse_error":perr,"choice_id":choice,"choice_text":selected,"rationale":rationale,"candidate_text":answer,"valid_choice":choice is not None,"error":None})
                else: row.update({"candidate_text":"","valid_choice":False,"error":errs[-1] if errs else "no response"})
                it=int(u.get("input_tokens",0) or 0);ot=int(u.get("output_tokens",0) or 0);ri,ro=RATES.get(provider,(0.0,0.0))
                cost=(it*ri+ot*ro)/1_000_000 if provider not in ("ollama","pixtral") else 0.0
                row["usage"]={"input_tokens":it,"output_tokens":ot,"estimated_cost_usd":cost,"rate_input_per_million":ri,"rate_output_per_million":ro}
                stream.write(json.dumps(row,ensure_ascii=False)+"\n");stream.flush()
                if j%20==0 or j==len(records): print(f"PROGRESS {slug} {j}/{len(records)} spend_usd={spend(OUTPUT_DIR):.4f}",flush=True)
        print(f"DONE {slug} spend_usd={spend(OUTPUT_DIR):.4f}",flush=True)
    print(f"ALL_DONE spend_usd={spend(OUTPUT_DIR):.4f}",flush=True)
if __name__=="__main__": main()

