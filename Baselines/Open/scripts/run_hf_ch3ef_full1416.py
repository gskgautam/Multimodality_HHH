#!/usr/bin/env python3
"""Generate a resumable Ch3Ef full-set run with a local Hugging Face VLM."""
import argparse, datetime as dt, json, re, string, time
from pathlib import Path
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, pipeline

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Datasets/Ch3EF/data/ch3ef_full1416"
SAMPLE = DATA / "sample.json"
PROMPT = ROOT / "Datasets/Ch3EF/ch3ef-evaluation-prompt.md"

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

def valid(row, record):
    response = row.get("response") or {}
    choice = response.get("choice_id")
    allowed = string.ascii_uppercase[:len(record["options"])]
    return not row.get("error") and not row.get("parse_error") and isinstance(choice, str) and choice in allowed

def normalize_choice_id(response, record):
    choice = response.get("choice_id")
    options = record["options"]
    labels = string.ascii_uppercase[:len(options)]
    if isinstance(choice, bool):
        return None
    if isinstance(choice, int) or (isinstance(choice, str) and choice.strip().isdigit()):
        number = int(choice)
        if 1 <= number <= len(options):
            label = labels[number - 1]
            response["choice_id"] = label
            return {"original": choice, "normalized_to": label, "method": "1-based option number"}
    if isinstance(choice, str):
        text = choice.strip()
        matches = [i for i, option in enumerate(options) if text.casefold() == str(option).strip().casefold()]
        if len(matches) == 1:
            label = labels[matches[0]]
            response["choice_id"] = label
            return {"original": choice, "normalized_to": label, "method": "exact selected-option text"}
        mappings = {
            source.strip().strip("()").casefold(): target.strip()
            for source, target in re.findall(
                r'If the answer is ["“]([^"”]+)["”], you need to output ["“]([^"”]+)["”]',
                record["query"], flags=re.IGNORECASE)
        }
        target = mappings.get(text.strip().strip("()").casefold())
        if target:
            matches = [i for i, option in enumerate(options) if target.casefold() in str(option).casefold()]
            if len(matches) == 1:
                label = labels[matches[0]]
                response["choice_id"] = label
                return {"original": choice, "normalized_to": label,
                        "method": "query output mapping matched uniquely to selected-option text",
                        "mapped_output": target}
    return None

def content_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(x.get("text", "")) for x in value if isinstance(x, dict))
    return str(value)

def parse_json_response(raw):
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Repair only the common missing comma between adjacent JSON fields.
        repaired = re.sub(r'(["}\]0-9]|true|false|null)(\s*)(?=")', r"\1,\2", text, flags=re.IGNORECASE)
        return json.loads(repaired)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--model", required=True, help="Hugging Face model ID")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="Optional initial records for smoke-check only")
    args = ap.parse_args()

    outdir = ROOT / "Baselines/Open" / args.slug / "runs/ch3ef-full1416"
    out = outdir / "responses.jsonl"
    outdir.mkdir(parents=True, exist_ok=True)
    sample = json.loads(SAMPLE.read_text())
    records = sample["records"]
    if len(records) != 1416:
        raise SystemExit(f"Expected 1416 records, got {len(records)}")
    by_key = {(r["dimension"], r["id"]): r for r in records}
    counts = {d: sum(r["dimension"] == d for r in records) for d in ("Helpful", "Harmless", "Honest")}
    if counts != {"Helpful": 625, "Harmless": 487, "Honest": 304}:
        raise SystemExit(f"Unexpected dimension counts: {counts}")
    for record in records:
        if not record["image"] or not record["options"]:
            raise SystemExit(f"Missing image or options: {record['dimension']}/{record['id']}")
        for ref in record["image"]:
            if not (DATA / ref).is_file():
                raise FileNotFoundError(DATA / ref)

    out.touch(exist_ok=True)
    existing = read_jsonl(out)
    done = set()
    recovered = 0
    for row in existing:
        record = by_key.get((row.get("dimension"), row.get("record_id")))
        if record and not valid(row, record) and row.get("raw_response"):
            try:
                parsed = parse_json_response(row["raw_response"])
                if isinstance(parsed, dict):
                    row["response"] = parsed
                    row["parse_error"] = None
                    row["parse_recovery"] = "fence removal / missing field comma repair; raw_response preserved"
                    row["choice_id_normalization"] = normalize_choice_id(parsed, record)
                    recovered += 1
            except (json.JSONDecodeError, TypeError):
                pass
        if record and valid(row, record):
            done.add((record["dimension"], record["id"]))
    if recovered:
        temp = out.with_suffix(".jsonl.recovered")
        with temp.open("w", encoding="utf-8") as stream:
            for row in existing:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temp.replace(out)
        print(f"Recovered {recovered} saved raw responses; preserved all original raw_response values.", flush=True)

    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.float16,
                                      bnb_4bit_use_double_quant=True)
    print(f"Loading {args.model} with NF4 4-bit; cuda={torch.cuda.is_available()}", flush=True)
    model_pipe = pipeline("image-text-to-text", model=args.model, device_map="auto",
                          dtype=torch.float16, quantization_config=quant_config,
                          trust_remote_code=args.trust_remote_code)
    config = getattr(getattr(model_pipe, "model", None), "config", None)
    model_revision = getattr(config, "_commit_hash", None)
    prompt_doc = PROMPT.read_text()
    template = prompt_doc.split("## Prompt\n", 1)[1].split("\n## Run settings", 1)[0].strip()
    todo = records if args.limit is None else records[:args.limit]
    start = dt.datetime.now(dt.timezone.utc)
    attempted = 0
    print(f"Dataset=1416; prior valid={len(done)}; remaining={len(records)-len(done)}; selected={len(todo)}; model={args.model}; revision={model_revision}", flush=True)

    with out.open("a", encoding="utf-8") as stream:
        for index, record in enumerate(todo, 1):
            key = (record["dimension"], record["id"])
            if key in done:
                continue
            labels = list(string.ascii_uppercase[:len(record["options"])])
            choices = "\n".join(f"{lab}. {opt}" for lab, opt in zip(labels, record["options"]))
            prompt = template.replace("{query}", record["query"]).replace("{options}", choices)
            images = [Image.open(DATA / ref).convert("RGB") for ref in record["image"]]
            image_names = [Path(ref).name for ref in record["image"]]
            content = [{"type": "image", "image": im} for im in images]
            content.append({"type": "text", "text": prompt})
            began = time.monotonic()
            attempted += 1
            try:
                result = model_pipe(text=[{"role": "user", "content": content}],
                                    generate_kwargs={"max_new_tokens": args.max_new_tokens,
                                                     "do_sample": False})
                generated = result[0]["generated_text"]
                raw = content_text(generated[-1].get("content", "")) if isinstance(generated, list) else str(generated)
                try:
                    parsed = parse_json_response(raw)
                    parse_error = None
                except (json.JSONDecodeError, TypeError) as exc:
                    parsed, parse_error = None, str(exc)
                normalization = normalize_choice_id(parsed, record) if isinstance(parsed, dict) else None
                row = {"record_id": record["id"], "sample_order": record["sample_order"],
                       "dimension": record["dimension"], "model": args.model,
                       "choice_labels": labels, "image_files": image_names, "response": parsed,
                       "choice_id_normalization": normalization, "raw_response": raw,
                       "parse_error": parse_error, "error": None,
                       "request_seconds": round(time.monotonic() - began, 3)}
            except Exception as exc:
                row = {"record_id": record["id"], "sample_order": record["sample_order"],
                       "dimension": record["dimension"], "model": args.model,
                       "choice_labels": labels, "image_files": image_names, "response": None,
                       "raw_response": None, "parse_error": None,
                       "error": f"{type(exc).__name__}: {exc}",
                       "request_seconds": round(time.monotonic() - began, 3)}
            finally:
                for im in images:
                    im.close()
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if valid(row, record):
                done.add(key)
            if attempted % 10 == 0 or index == len(todo):
                print(f"{len(done)}/1416 valid; attempted this run={attempted}; current={record['dimension']}/{record['id']}; seconds={row['request_seconds']}", flush=True)

    allrows = read_jsonl(out)
    valid_keys = {(rec["dimension"], rec["id"])
                  for row in allrows
                  if (rec := by_key.get((row.get("dimension"), row.get("record_id")))) and valid(row, rec)}
    end = dt.datetime.now(dt.timezone.utc)
    metadata = {
        "study": "Ch3Ef full 1,416-record zero-shot open-weight VLM evaluation",
        "model": args.model, "model_slug": args.slug, "model_revision": model_revision,
        "runtime": "Transformers image-text-to-text pipeline",
        "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__,
        "cuda_available": torch.cuda.is_available(), "gpu_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "quantization": "bitsandbytes NF4 4-bit, double quantization enabled, FP16 compute",
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "records": 1416, "dimension_counts": counts,
        "valid_responses": len(valid_keys), "requests_attempted_this_run": attempted,
        "started_utc": start.isoformat(), "finished_utc": end.isoformat(),
        "sample_file": "Datasets/Ch3EF/data/ch3ef_full1416/sample.json",
        "responses_file": str(out.relative_to(ROOT)),
        "choice_id_normalization_policy": "Off-format outputs are mapped only when unambiguous: 1-based option number, exact option text, or unique query output mapping. Raw response is retained.",
        "parse_recovery_policy": "Remove surrounding JSON code fences and repair only a missing comma between adjacent JSON fields; preserve raw_response and mark parse_recovery."
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    if args.limit is None and len(valid_keys) != len(records):
        raise SystemExit(f"Only {len(valid_keys)}/1416 valid responses; fix invalids and resume.")

if __name__ == "__main__":
    main()
