#!/usr/bin/env python3
"""Post-hoc paired image-access audit for OrchCircuit OOF selections."""
import argparse
import ast
import concurrent.futures
import datetime as dt
import importlib.util
import json
import threading
import time
from pathlib import Path

ROOT = (next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()))
DATA = ROOT / "Datasets/Ch3EF/data/ch3ef_full1416"
AUDIT = ROOT / "Baselines/Judge/image-access-audit/ch3ef-judge-image-access"
RUNNER = ROOT / "Baselines/Judge/scripts/run_judge_image_access_audit.py"
OOF = ROOT / "Method/OrchCircuit/final/runs/fullset_group5fold/oof_selected_candidates.jsonl"
COST_CAP_USD = 32.0
RESERVE_PER_CALL_USD = 0.03
PROVIDERS = ("google", "anthropic")
MODEL_SLUG = "orchcircuit"
FIELD_BY_DIM = {
    "Helpful": "wr_selected_candidate",
    "Harmless": "ss_selected_candidate",
    "Honest": "ti_selected_candidate",
}
LOCK = threading.Lock()
SPENT = 0.0
RESERVED = 0.0
BUDGET_HIT = False

spec = importlib.util.spec_from_file_location("judge_audit", RUNNER)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.DATA = DATA

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def parse_response(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return ast.literal_eval(value)
    return {}

def load_source_rows(slug, cache):
    if slug in cache:
        return cache[slug]
    paths = [
        ROOT / "Baselines/Open" / slug / "runs/ch3ef-full1416/responses.jsonl",
        ROOT / "Baselines/Closed" / slug / "runs/ch3ef-full1416/responses.jsonl",
    ]
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        cache[slug] = {}
        return cache[slug]
    rows = {}
    for row in read_jsonl(path):
        if row.get("sample_order") is None:
            continue
        key = (row.get("dimension"), int(row["sample_order"]))
        response = parse_response(row.get("response"))
        row["response"] = response
        if audit.valid_response(row):
            rows[key] = row
    cache[slug] = rows
    return rows

def append_row(row):
    with LOCK:
        with (AUDIT / "judge_calls.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

def main():
    global SPENT, RESERVED, BUDGET_HIT
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    AUDIT.mkdir(parents=True, exist_ok=True)
    manifest_path = AUDIT / "sample_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sample = json.loads((DATA / "sample.json").read_text())["records"]
    records = {(r["dimension"], str(r["id"])): r for r in sample}
    by_order = {(r["dimension"], int(r["sample_order"])): r for r in sample}
    oof = {(r["dimension"], int(r["sample_order"])): r for r in read_jsonl(OOF)}
    reference_rows = read_jsonl(ROOT / "Baselines/Open/qwen2.5-vl-7b/runs/ch3ef-full1416/wr_ti_gpt4o.jsonl")
    references = {
        str(r.get("record_id")): r.get("answer", "")
        for r in reference_rows
        if r.get("dimension") == "Helpful" and r.get("task") == "baseline" and not r.get("error")
    }
    response_cache = {}
    eligible = []
    missing = {dim: [] for dim in FIELD_BY_DIM}
    for dim, entries in manifest["dimensions"].items():
        field = FIELD_BY_DIM[dim]
        for item in entries:
            order = int(item["sample_order"])
            record = records.get((dim, str(item["record_id"]))) or by_order.get((dim, order))
            selected = oof.get((dim, order), {}).get(field)
            if not record or not selected:
                missing[dim].append(order)
                continue
            source = load_source_rows(selected, response_cache).get((dim, order))
            if not source:
                missing[dim].append(order)
                continue
            baseline = references.get(str(record["id"]), "") if dim == "Helpful" else ""
            if dim == "Helpful" and not baseline:
                missing[dim].append(order)
                continue
            eligible.append((dim, item, record, selected, source, baseline))
    eligible_by_dim = {d: sum(1 for x in eligible if x[0] == d) for d in FIELD_BY_DIM}
    extension_manifest = {
        "study": "post-hoc paired image-access audit of OrchCircuit out-of-fold selections",
        "model_slug": MODEL_SLUG,
        "selection_source": str(OOF),
        "selector_field_by_dimension": FIELD_BY_DIM,
        "sample_manifest": str(manifest_path),
        "frozen_original_sample_seed": manifest["selection_seed"],
        "frozen_records_per_dimension": manifest["records_per_dimension"],
        "judges": list(PROVIDERS),
        "conditions": ["image_present", "image_omitted"],
        "eligible_records_by_dimension": eligible_by_dim,
        "unavailable_sample_orders_by_dimension": missing,
        "cumulative_estimated_cost_cap_usd": COST_CAP_USD,
        "pairing_note": "Google and Anthropic were selected because this is the most common existing two-judge pairing; the same two judges score both access conditions.",
    }
    (AUDIT / "orchcircuit_extension_manifest.json").write_text(json.dumps(extension_manifest, indent=2) + "\n")
    journal_path = AUDIT / "judge_calls.jsonl"
    old = read_jsonl(journal_path)
    SPENT = sum(r.get("usage", {}).get("estimated_cost_usd", 0.0) for r in old if r.get("usage")) + RESERVE_PER_CALL_USD * sum(r.get("status") == "error" for r in old)
    completed = {
        (r.get("model_slug"), str(r.get("record_id")), r.get("dimension"), r.get("condition"), r.get("judge_provider"))
        for r in old if r.get("status") == "success"
    }
    tasks = []
    for dim, item, record, selected, source, baseline in eligible:
        rid = record["id"]
        response = audit.answer_text(record, source)
        task_name = "WR" if dim == "Helpful" else ("SS" if dim == "Harmless" else "TI")
        candidate_first = ((int(rid) + int(manifest["selection_seed"]) + len(manifest["models"])) % 2 == 0)
        for condition in ("image_present", "image_omitted"):
            for provider in PROVIDERS:
                key = (MODEL_SLUG, str(rid), dim, condition, provider)
                if key in completed:
                    continue
                tasks.append({
                    "model_slug": MODEL_SLUG,
                    "record_id": rid,
                    "dimension": dim,
                    "sample_order": int(item["sample_order"]),
                    "condition": condition,
                    "judge_provider": provider,
                    "judge_model": audit.JUDGES[provider]["model"],
                    "task": task_name,
                    "response": response,
                    "baseline": baseline,
                    "candidate_first": candidate_first,
                    "record": record,
                })
    print(json.dumps({
        "eligible_records_by_dimension": eligible_by_dim,
        "unavailable_records_by_dimension": {d: len(v) for d, v in missing.items()},
        "pending_calls": len(tasks),
        "judges": list(PROVIDERS),
        "estimated_cost_already_reserved_usd": round(SPENT, 4),
        "remaining_cumulative_cap_usd": round(COST_CAP_USD - SPENT, 4),
    }), flush=True)
    if args.dry_run:
        print("Dry run only; no judge requests were sent.", flush=True)
        return
    if not tasks:
        print("No pending calls.", flush=True)
        return
    def worker(task):
        global SPENT, RESERVED, BUDGET_HIT
        last_error = None
        for attempt in range(1, 3):
            with LOCK:
                if SPENT + RESERVED + RESERVE_PER_CALL_USD > COST_CAP_USD:
                    BUDGET_HIT = True
                    return "budget"
                RESERVED += RESERVE_PER_CALL_USD
            images = audit.load_images(task["record"]) if task["condition"] == "image_present" else []
            prompt = audit.prompt_for(task["task"], task["record"], task["response"], task["baseline"], task["candidate_first"])
            result = None
            try:
                parsed, usage = audit.call_judge(task["judge_provider"], prompt, images, task["task"])
                status = "success" if audit.valid_judgment(task["task"], parsed) else "invalid_json"
                result = {"status": status, "parsed": parsed, "usage": usage}
                if status == "invalid_json":
                    last_error = "Judge response missing required label fields."
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            with LOCK:
                RESERVED -= RESERVE_PER_CALL_USD
                if result and result.get("usage"):
                    SPENT += result["usage"].get("estimated_cost_usd", 0.0)
                else:
                    SPENT += RESERVE_PER_CALL_USD
            row = {k: v for k, v in task.items() if k not in {"record", "response", "baseline"}}
            row.update({
                "status": result["status"] if result else "error",
                "attempts": attempt,
                "error": last_error,
                "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            })
            if result:
                row.update({"judge_output": result["parsed"], "usage": result["usage"]})
            append_row(row)
            if row["status"] == "success":
                return "success"
            time.sleep(1)
        return "error"
    counts = {"success": 0, "error": 0, "budget": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, task) for task in tasks]
        for f in concurrent.futures.as_completed(futures):
            result = f.result()
            counts[result] = counts.get(result, 0) + 1
            if sum(counts.values()) % 50 == 0:
                print(json.dumps({"completed": sum(counts.values()), "counts": counts, "estimated_total_usd": round(SPENT, 4)}), flush=True)
    all_rows = read_jsonl(journal_path)
    success_cost = sum(r.get("usage", {}).get("estimated_cost_usd", 0.0) for r in all_rows if r.get("usage"))
    error_count = sum(r.get("status") == "error" for r in all_rows)
    total_est = success_cost + RESERVE_PER_CALL_USD * error_count
    summary_path = AUDIT / "usage_summary.json"
    usage = json.loads(summary_path.read_text())
    by_provider = {}
    for provider in ("openai", "google", "anthropic"):
        rows = [r for r in all_rows if r.get("judge_provider") == provider]
        by_provider[provider] = {
            "journal_rows": len(rows),
            "valid": sum(r.get("status") == "success" for r in rows),
            "estimated_cost_usd": round(sum(r.get("usage", {}).get("estimated_cost_usd", 0.0) for r in rows if r.get("usage")), 6),
            "errors": sum(r.get("status") == "error" for r in rows),
        }
    usage.update({
        "journal_rows_total": len(all_rows),
        "estimated_total_cost_usd": round(total_est, 6),
        "cap_usd": COST_CAP_USD,
        "budget_hit": BUDGET_HIT,
        "by_provider": by_provider,
        "orchcircuit_extension": {
            "eligible_records_by_dimension": eligible_by_dim,
            "calls_pending_before_run": len(tasks),
            "completed_outcomes": counts,
            "judges": list(PROVIDERS),
        },
    })
    summary_path.write_text(json.dumps(usage, indent=2) + "\n")
    extension_manifest["estimated_total_cost_after_run_usd"] = round(total_est, 6)
    extension_manifest["budget_hit"] = BUDGET_HIT
    extension_manifest["outcomes"] = counts
    (AUDIT / "orchcircuit_extension_manifest.json").write_text(json.dumps(extension_manifest, indent=2) + "\n")
    print(json.dumps({
        "completed": counts,
        "eligible_records_by_dimension": eligible_by_dim,
        "estimated_total_cost_usd": round(total_est, 6),
        "cap_usd": COST_CAP_USD,
        "budget_hit": BUDGET_HIT,
    }, indent=2), flush=True)

if __name__ == "__main__":
    main()
