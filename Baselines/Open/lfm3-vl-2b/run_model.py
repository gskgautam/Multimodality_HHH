#!/usr/bin/env python3
"""Run one complete Ch3Ef VLM generation pass against the fixed 1,416-record sample."""
import argparse
import base64
import datetime as dt
import json
import re
import string
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Datasets/Ch3EF/data/ch3ef_full1416"
SAMPLE = DATA / "sample.json"
PROMPT = ROOT / "Datasets/Ch3EF/ch3ef-evaluation-prompt.md"
DEFAULT_API = "http://127.0.0.1:11435/api/chat"


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
                record["query"],
                flags=re.IGNORECASE,
            )
        }
        target = mappings.get(text.strip().strip("()").casefold())
        if target:
            target_norm = target.casefold()
            matches = [i for i, option in enumerate(options) if target_norm in str(option).casefold()]
            if len(matches) == 1:
                label = labels[matches[0]]
                response["choice_id"] = label
                return {
                    "original": choice,
                    "normalized_to": label,
                    "method": "query output mapping matched uniquely to selected-option text",
                    "mapped_output": target,
                }
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True, help="Model folder name under evaluations/")
    parser.add_argument("--model", required=True, help="Exact Ollama model tag")
    parser.add_argument("--api", default=DEFAULT_API)
    args = parser.parse_args()

    outdir = ROOT / "evaluations" / args.slug / "ch3ef-full1416"
    out = outdir / "responses.jsonl"
    sample = json.loads(SAMPLE.read_text())
    records = sample["records"]
    if len(records) != 1416:
        raise SystemExit(f"Expected 1416 records, got {len(records)}")
    by_key = {(r["dimension"], r["id"]): r for r in records}
    if len(by_key) != len(records):
        raise SystemExit("Duplicate (dimension,id) in full sample")
    counts = {d: sum(r["dimension"] == d for r in records) for d in ("Helpful", "Harmless", "Honest")}
    if counts != {"Helpful": 625, "Harmless": 487, "Honest": 304}:
        raise SystemExit(f"Unexpected dimension counts: {counts}")
    for record in records:
        if not record["image"] or not record["options"]:
            raise SystemExit(f"Missing image or options: {record['dimension']}/{record['id']}")
        for ref in record["image"]:
            if not (DATA / ref).is_file():
                raise FileNotFoundError(DATA / ref)

    outdir.mkdir(parents=True, exist_ok=True)
    out.touch(exist_ok=True)
    existing = read_jsonl(out)
    done = set()
    for row in existing:
        record = by_key.get((row.get("dimension"), row.get("record_id")))
        if record and valid(row, record):
            done.add((record["dimension"], record["id"]))

    version_request = urllib.request.Request(args.api.rsplit("/api/chat", 1)[0] + "/api/version")
    with urllib.request.urlopen(version_request, timeout=15) as response:
        ollama_version = json.loads(response.read().decode()).get("version")
    prompt_doc = PROMPT.read_text()
    template = prompt_doc.split("## Prompt\n", 1)[1].split("\n## Run settings", 1)[0].strip()
    start = dt.datetime.now(dt.timezone.utc)
    attempted = 0
    print(f"Dataset: {len(records)}; prior valid: {len(done)}; remaining: {len(records)-len(done)}; model={args.model}; Ollama={ollama_version}", flush=True)

    with out.open("a", encoding="utf-8") as stream:
        for index, record in enumerate(records, 1):
            key = (record["dimension"], record["id"])
            if key in done:
                continue
            labels = list(string.ascii_uppercase[:len(record["options"])])
            choices = "\n".join(f"{label}. {option}" for label, option in zip(labels, record["options"]))
            prompt = template.replace("{query}", record["query"]).replace("{options}", choices)
            images = [base64.b64encode((DATA / ref).read_bytes()).decode("ascii") for ref in record["image"]]
            image_names = [Path(ref).name for ref in record["image"]]
            messages = [{"role": "user", "content": prompt, "images": images}]
            if args.model.startswith("gemma4:"):
                # Gemma 4 can enter an unbounded thought-channel loop despite think=false.
                # An empty thought block follows Google's no-thinking prompt guidance.
                messages.insert(0, {"role": "system", "content": "<|channel>thought\n<channel|>Answer directly with only the requested JSON."})
            options = {"temperature": 0}
            if args.model.startswith("gemma4:"):
                options["num_predict"] = 512
            payload = {
                "model": args.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "format": "json",
                "options": options,
            }
            request = urllib.request.Request(args.api, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
            began = time.monotonic()
            attempted += 1
            try:
                with urllib.request.urlopen(request, timeout=900) as response:
                    body = json.loads(response.read().decode())
                raw = body.get("message", {}).get("content", "")
                try:
                    parsed = json.loads(raw)
                    parse_error = None
                except (json.JSONDecodeError, TypeError) as exc:
                    parsed, parse_error = None, str(exc)
                normalization = normalize_choice_id(parsed, record) if isinstance(parsed, dict) else None
                row = {
                    "record_id": record["id"], "sample_order": record["sample_order"],
                    "dimension": record["dimension"], "model": body.get("model", args.model),
                    "choice_labels": labels, "image_files": image_names, "response": parsed,
                    "choice_id_normalization": normalization,
                    "raw_response": raw, "parse_error": parse_error, "done_reason": body.get("done_reason"),
                    "prompt_eval_count": body.get("prompt_eval_count"), "eval_count": body.get("eval_count"),
                    "total_duration_ns": body.get("total_duration"), "load_duration_ns": body.get("load_duration"),
                    "request_seconds": round(time.monotonic() - began, 3), "error": None,
                }
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                row = {
                    "record_id": record["id"], "sample_order": record["sample_order"],
                    "dimension": record["dimension"], "model": args.model, "choice_labels": labels,
                    "image_files": image_names, "response": None, "raw_response": None,
                    "parse_error": None, "request_seconds": round(time.monotonic() - began, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if valid(row, record):
                done.add(key)
            if index % 10 == 0 or index == len(records):
                print(f"{index}/1416; valid={len(done)}; current={record['dimension']}/{record['id']}; seconds={row['request_seconds']}", flush=True)

    end = dt.datetime.now(dt.timezone.utc)
    allrows = read_jsonl(out)
    valid_keys = {
        (record["dimension"], record["id"])
        for row in allrows
        if (record := by_key.get((row.get("dimension"), row.get("record_id")))) and valid(row, record)
    }
    valid_n = len(valid_keys)
    metadata = {
        "study": "Ch3Ef complete automated HHH evaluation",
        "model": args.model, "model_slug": args.slug,
        "runtime": "user-level Ollama service on SSH server",
        "ollama_version": ollama_version, "endpoint": args.api, "temperature": 0, "think": False,
        "model_specific_generation_settings": ({"empty_thought_block": True, "num_predict": 512} if args.model.startswith("gemma4:") else {}),
        "records": len(records), "dimension_counts": counts, "valid_responses": valid_n,
        "requests_attempted_this_run": attempted, "started_utc": start.isoformat(),
        "finished_utc": end.isoformat(),
        "sample_file": "Datasets/Ch3EF/data/ch3ef_full1416/sample.json",
        "responses_file": str(out.relative_to(ROOT)),
        "existing_rows_before_this_run": len(existing),
        "normalized_choice_ids": sum(bool(row.get("choice_id_normalization")) for row in allrows),
        "choice_id_normalization_policy": "Off-format outputs are mapped only when unambiguous: 1-based option number, exact option text, or a unique query output mapping that matches exactly one outer option. The raw_response field preserves the original model output.",
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    if valid_n != len(records):
        raise SystemExit(f"Only {valid_n}/{len(records)} valid responses, expected {len(records)}")


if __name__ == "__main__":
    main()
