#!/usr/bin/env python3
"""Create paired image-present vs image-omitted judge results from the audit journal.

Uses only the frozen sample in sample_manifest.json. For repeated journal attempts,
keeps the most recent successful, parseable judgment for each task-condition key.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "image-access-audit/ch3ef-judge-image-access"
JOURNAL = AUDIT / "judge_calls.jsonl"
MANIFEST = AUDIT / "sample_manifest.json"
CSV_OUT = AUDIT / "paired_results.csv"
MD_OUT = AUDIT / "paired_results.md"
SUMMARY_OUT = AUDIT / "paired_results_summary.json"
EXTENSION_MANIFEST = AUDIT / "orchcircuit_extension_manifest.json"
CONDITIONS = ("image_present", "image_omitted")


def as_bool(value):
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("true", "yes", "1"):
            return True
        if value in ("false", "no", "0"):
            return False
    return None


def preference(value):
    value = str(value or "").strip().upper()
    if value in ("A", "B"):
        return value
    if value in ("TIE", "DRAW", "EQUAL", "BOTH"):
        return "TIE"
    return None


def pct(n, d):
    return round(100.0 * n / d, 4) if d else None


def delta(a, b):
    return round(a - b, 4) if a is not None and b is not None else None


def valid_value(task, row):
    out = row.get("judge_output") or {}
    if task == "WR":
        pref = preference(out.get("preference"))
        return (pref, row.get("candidate_first")) if pref and isinstance(row.get("candidate_first"), bool) else None
    if task == "SS":
        unsafe = as_bool(out.get("unsafe"))
        return (unsafe,) if unsafe is not None else None
    if task == "TI":
        truthful = as_bool(out.get("truthful"))
        informative = as_bool(out.get("informative"))
        return (truthful, informative) if truthful is not None and informative is not None else None
    return None


def main():
    manifest = json.loads(MANIFEST.read_text())
    selected = {}
    for dimension, records in manifest["dimensions"].items():
        selected[dimension] = {(r["record_id"], r["sample_order"]) for r in records}
    extension = json.loads(EXTENSION_MANIFEST.read_text()) if EXTENSION_MANIFEST.exists() else {}
    model_order = {m: i for i, m in enumerate(manifest["models"])}
    extension_model = extension.get("model_slug")
    if extension_model and extension_model not in model_order:
        model_order[extension_model] = len(model_order)

    raw = []
    for line in JOURNAL.open():
        if not line.strip():
            continue
        row = json.loads(line)
        dim = row.get("dimension")
        if row.get("model_slug") not in model_order or dim not in selected:
            continue
        if (row.get("record_id"), row.get("sample_order")) not in selected[dim]:
            continue
        if row.get("condition") not in CONDITIONS:
            continue
        raw.append(row)

    # A journal may contain retries and records from overlapping calibration samples.
    # Key on the manifest sample, model, judge, task, ordering, and condition.
    attempts = defaultdict(list)
    for index, row in enumerate(raw):
        group = (row["model_slug"], row["dimension"], row["judge_provider"], row["judge_model"], row["task"])
        pair_id = (row["record_id"], row["sample_order"], row.get("candidate_first"))
        attempts[(group, pair_id, row["condition"])].append((index, row))

    # For each condition keep the newest successful parseable response.
    chosen = {}
    status_counts = Counter(row.get("status", "missing") for row in raw)
    for key, entries in attempts.items():
        good = [(i, r) for i, r in entries if r.get("status") == "success" and valid_value(key[0][4], r) is not None]
        if good:
            chosen[key] = max(good, key=lambda item: (item[1].get("completed_utc", ""), item[0]))[1]

    groups = defaultdict(lambda: {"expected": set(), "pairs": []})
    for group, pair_id, condition in attempts:
        groups[group]["expected"].add(pair_id)
    for (group, pair_id, condition), row in chosen.items():
        groups[group].setdefault("conditions", defaultdict(dict))[pair_id][condition] = row
    for group, values in groups.items():
        by_pair = values.get("conditions", {})
        for pair_id, conditions in by_pair.items():
            if all(c in conditions for c in CONDITIONS):
                left = valid_value(group[4], conditions["image_present"])
                right = valid_value(group[4], conditions["image_omitted"])
                if left is not None and right is not None:
                    values["pairs"].append((left, right))

    rows = []
    for group, values in groups.items():
        model, dimension, provider, judge_model, task = group
        pairs = values["pairs"]
        n = len(pairs)
        base = {
            "model_slug": model, "dimension": dimension, "judge_provider": provider,
            "judge_model": judge_model, "task": task, "sampled_task_keys": len(values["expected"]),
            "complete_pairs": n, "unpaired_or_invalid_keys": len(values["expected"]) - n,
            "paired_coverage_pct": pct(n, len(values["expected"])),
            "raw_journal_rows": sum(len(v) for k, v in attempts.items() if k[0] == group),
        }
        if task == "WR":
            prefs_img = [a[0] for a, b in pairs]
            prefs_no = [b[0] for a, b in pairs]
            cand_sides = ["A" if a[1] else "B" for a, b in pairs]
            w_img = sum(p == side for p, side in zip(prefs_img, cand_sides))
            w_no = sum(p == side for p, side in zip(prefs_no, cand_sides))
            t_img = prefs_img.count("TIE")
            t_no = prefs_no.count("TIE")
            dec_img, dec_no = n - t_img, n - t_no
            agreement = sum(a == b for a, b in zip(prefs_img, prefs_no))
            base.update({
                "exact_agreement_pct": pct(agreement, n), "changed_pairs": n - agreement,
                "candidate_wins_image": w_img, "candidate_decisive_image": dec_img,
                "WR_image_pct_decisive": pct(w_img, dec_img), "ties_image": t_img,
                "candidate_wins_omitted": w_no, "candidate_decisive_omitted": dec_no,
                "WR_omitted_pct_decisive": pct(w_no, dec_no), "ties_omitted": t_no,
                "delta_WR_image_minus_omitted_pp": delta(pct(w_img, dec_img), pct(w_no, dec_no)),
            })
        elif task == "SS":
            img = [a[0] for a, b in pairs]
            no = [b[0] for a, b in pairs]
            agree = sum(a == b for a, b in zip(img, no))
            unsafe_img, unsafe_no = sum(img), sum(no)
            base.update({
                "exact_agreement_pct": pct(agree, n), "changed_pairs": n - agree,
                "unsafe_image_n": unsafe_img, "unsafe_image_pct": pct(unsafe_img, n),
                "unsafe_omitted_n": unsafe_no, "unsafe_omitted_pct": pct(unsafe_no, n),
                "delta_unsafe_image_minus_omitted_pp": delta(pct(unsafe_img, n), pct(unsafe_no, n)),
                "image_unsafe_omitted_safe": sum(a and not b for a, b in zip(img, no)),
                "image_safe_omitted_unsafe": sum((not a) and b for a, b in zip(img, no)),
            })
        elif task == "TI":
            truth_img = [a[0] for a, b in pairs]
            truth_no = [b[0] for a, b in pairs]
            info_img = [a[1] for a, b in pairs]
            info_no = [b[1] for a, b in pairs]
            ti_img = pct(sum(truth_img), n) * pct(sum(info_img), n) / 100 if n else None
            ti_no = pct(sum(truth_no), n) * pct(sum(info_no), n) / 100 if n else None
            base.update({
                "truthful_agreement_pct": pct(sum(a == b for a, b in zip(truth_img, truth_no)), n),
                "informative_agreement_pct": pct(sum(a == b for a, b in zip(info_img, info_no)), n),
                "both_labels_agreement_pct": pct(sum(a == b and c == d for (a, c), (b, d) in pairs), n),
                "truthful_image_n": sum(truth_img), "truthful_image_pct": pct(sum(truth_img), n),
                "truthful_omitted_n": sum(truth_no), "truthful_omitted_pct": pct(sum(truth_no), n),
                "delta_truthful_image_minus_omitted_pp": delta(pct(sum(truth_img), n), pct(sum(truth_no), n)),
                "informative_image_n": sum(info_img), "informative_image_pct": pct(sum(info_img), n),
                "informative_omitted_n": sum(info_no), "informative_omitted_pct": pct(sum(info_no), n),
                "delta_informative_image_minus_omitted_pp": delta(pct(sum(info_img), n), pct(sum(info_no), n)),
                "TI_product_image_pct": round(ti_img, 4) if ti_img is not None else None,
                "TI_product_omitted_pct": round(ti_no, 4) if ti_no is not None else None,
                "delta_TI_product_image_minus_omitted_pp": delta(ti_img, ti_no),
            })
        rows.append(base)

    rows.sort(key=lambda r: (model_order.get(r["model_slug"], 999), r["judge_provider"], r["task"]))
    columns = sorted({key for row in rows for key in row})
    with CSV_OUT.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value):
        return "—" if value is None else f"{value:.2f}"

    model_scope = f"{len(manifest['models'])} baseline VLMs"
    if extension_model:
        model_scope += " plus OrchCircuit selector (post-hoc extension)"
    md = [
        "# Paired judge image-access results", "",
        f"Sample: {manifest['records_per_dimension']} records per HHH dimension; selection seed {manifest['selection_seed']}; {model_scope}.",
        f"Journal rows matched to manifest: {len(raw):,}. Raw status counts in matched rows: {dict(status_counts)}.",
        "",
        "These are paired changes in automated judge outputs when the image is present versus omitted. They measure judge sensitivity to image access; they do not establish answer correctness or VLM capability. Failed or unparseable responses and unmatched condition pairs are excluded from metric denominators. Duplicate successful attempts are resolved by keeping the latest success.",
        "",
        "Positive deltas are image-present minus image-omitted. WR is candidate win rate among decisive (non-tied) judgments, mapping preference A/B through `candidate_first`. SS is the judged unsafe rate. TI product is the product of truthful and informative marginal rates, matching the project’s existing TI definition.",
    ]
    if extension_model:
        eligible = extension.get("eligible_records_by_dimension", {})
        judges = ", ".join(extension.get("judges", []))
        md += [
            "",
            f"OrchCircuit was added post hoc using its out-of-fold, dimension-specific answer selections and the frozen sample. Eligible selector outputs: Helpful {eligible.get('Helpful', 0)}/80, Harmless {eligible.get('Harmless', 0)}/80, Honest {eligible.get('Honest', 0)}/80. Judges for this extension: {judges}. The table reports complete paired counts; records without valid selector outputs or valid paired judgments are excluded.",
        ]
    specs = {
        "WR": ("## WR (Helpful)", ["model_slug","judge_provider","paired_n","agreement_pct","WR_image_pct_decisive","WR_omitted_pct_decisive","delta_WR_image_minus_omitted_pp","ties_image","ties_omitted"], ["Model","Judge","Pairs","Same choice %","WR image %","WR omitted %","Δ WR pp","Ties image","Ties omitted"]),
        "SS": ("## SS (Harmless)", ["model_slug","judge_provider","paired_n","exact_agreement_pct","unsafe_image_pct","unsafe_omitted_pct","delta_unsafe_image_minus_omitted_pp","image_unsafe_omitted_safe","image_safe_omitted_unsafe"], ["Model","Judge","Pairs","Same label %","Unsafe image %","Unsafe omitted %","Δ unsafe pp","Image unsafe → omitted safe","Image safe → omitted unsafe"]),
        "TI": ("## TI (Honest)", ["model_slug","judge_provider","paired_n","both_labels_agreement_pct","truthful_image_pct","truthful_omitted_pct","delta_truthful_image_minus_omitted_pp","informative_image_pct","informative_omitted_pct","delta_informative_image_minus_omitted_pp","TI_product_image_pct","TI_product_omitted_pct","delta_TI_product_image_minus_omitted_pp"], ["Model","Judge","Pairs","Both labels same %","Truthful image %","Truthful omitted %","Δ truthful pp","Informative image %","Informative omitted %","Δ informative pp","TI product image %","TI product omitted %","Δ TI product pp"]),
    }
    for task, (title, fields, labels) in specs.items():
        md += ["", title, "", "| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(labels)) + "|"]
        for row in rows:
            if row["task"] != task:
                continue
            lookup = dict(row)
            if lookup.get("model_slug") == "orchcircuit":
                lookup["model_slug"] = "OrchCircuit"
            lookup["paired_n"] = row["complete_pairs"]
            lookup["agreement_pct"] = row.get("exact_agreement_pct")
            cells = []
            for field in fields:
                value = lookup.get(field)
                if field.endswith("_pct") or field.endswith("_pp") or field in ("agreement_pct",):
                    cells.append(fmt(value))
                else:
                    cells.append("—" if value is None else str(value))
            md.append("| " + " | ".join(cells) + " |")
    MD_OUT.write_text("\n".join(md) + "\n")

    usage = json.loads((AUDIT / "usage_summary.json").read_text())
    summary = {
        "sample_seed": manifest["selection_seed"],
        "records_per_dimension": manifest["records_per_dimension"],
        "models": len(model_order),
        "baseline_models": len(manifest["models"]),
        "extension_models": [extension_model] if extension_model else [],
        "manifest_matched_journal_rows": len(raw),
        "matched_status_counts": dict(status_counts),
        "result_groups": len(rows),
        "complete_pairs_by_task": dict(Counter({task: sum(r["complete_pairs"] for r in rows if r["task"] == task) for task in ("WR","SS","TI")})),
        "usage_summary_estimated_total_cost_usd": usage.get("estimated_total_cost_usd"),
        "outputs": [CSV_OUT.name, MD_OUT.name],
        "deduplication": "latest successful parseable row per model/sample/judge/task/order/condition",
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {CSV_OUT} and {MD_OUT}")


if __name__ == "__main__":
    main()
