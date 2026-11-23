"""Figure 4b WR sensitivity: tie-inclusive WR definitions (review response, 2026-11-23).

The original analyze_paired_shift.py scores WR as wins among decisive (non-tied)
judgments, so the image-present and image-omitted WR are computed on different
item subsets whenever tie counts change. This sibling script keeps the identical
record selection, pairing, groups, bootstrap and seed, and re-scores WR two ways
that keep every paired item in the denominator:
  WR_all = wins / all paired items   (Table 1 definition: ties stay in denominator)
  WR_net = (wins - losses) / all paired items
It writes separate outputs and does not modify the original results.
"""
from pathlib import Path
import json, collections, csv
import numpy as np

ROOT = (next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()))
SRC = ROOT / "Baselines/Judge/image-access-audit/ch3ef-judge-image-access/judge_calls.jsonl"
FOLDS = ROOT / "Method/OrchCircuit/final/runs/fullset_group5fold/folds.json"
OUT = Path(__file__).resolve().parent
MODELS = ("orchcircuit", "qwen2.5-vl-7b"); JUDGES = ("anthropic", "google")
B = 10000; SEED = 20261113


def score(r):
    p = str(r["judge_output"]["preference"]).lower()
    if p in ("tie", "t"):
        return 0
    return 1 if p == ("a" if bool(r.get("candidate_first")) else "b") else -1


def agg(v, mode):
    v = np.asarray(v)
    return 100 * (np.mean(v == 1) if mode == "all" else np.mean(v))


def main():
    latest = {}
    for line in SRC.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") != "success" or not isinstance(r.get("judge_output"), dict):
            continue
        if r.get("model_slug") not in MODELS or r.get("judge_provider") not in JUDGES or r.get("task") != "WR":
            continue
        if r.get("condition") not in ("image_present", "image_omitted") or r.get("sample_order") is None:
            continue
        k = (r["model_slug"], int(r["sample_order"]), r["judge_provider"], r["condition"])
        if k not in latest or str(r.get("completed_utc", "")) > str(latest[k].get("completed_utc", "")):
            latest[k] = r
    group = json.loads(FOLDS.read_text())["group_ids"]
    out = []
    for mode in ("all", "net"):
        rng = np.random.default_rng(SEED)
        for judge in JUDGES:
            by = {}
            for model in MODELS:
                tmp = collections.defaultdict(dict)
                for (m, o, j, c), r in latest.items():
                    if (m, j) != (model, judge):
                        continue
                    try:
                        tmp[o][c] = score(r)
                    except (KeyError, TypeError):
                        continue
                by[model] = {o: x for o, x in tmp.items() if "image_present" in x and "image_omitted" in x}
            orders = sorted(set(by[MODELS[0]]) & set(by[MODELS[1]]))
            vals = {m: {c: [by[m][o][c] for o in orders] for c in ("image_present", "image_omitted")} for m in MODELS}
            shift = {m: agg(vals[m]["image_present"], mode) - agg(vals[m]["image_omitted"], mode) for m in MODELS}
            gids = [int(group[str(o)]) for o in orders]; ug = sorted(set(gids))
            members = {g: [i for i, x in enumerate(gids) if x == g] for g in ug}
            boot = np.empty(B)
            for b in range(B):
                ix = [i for g in rng.choice(ug, size=len(ug), replace=True) for i in members[int(g)]]
                s = {m: agg([vals[m]["image_present"][i] for i in ix], mode) - agg([vals[m]["image_omitted"][i] for i in ix], mode) for m in MODELS}
                boot[b] = s["orchcircuit"] - s["qwen2.5-vl-7b"]
            out.append({"wr_definition": "wins/all" if mode == "all" else "(wins-losses)/all", "judge": judge,
                        "n_common_prompts": len(orders), "n_groups": len(ug),
                        "orch_shift_pp": shift["orchcircuit"], "qwen_shift_pp": shift["qwen2.5-vl-7b"],
                        "did_pp": shift["orchcircuit"] - shift["qwen2.5-vl-7b"],
                        "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975))})
    with (OUT / "paired_shift_wr_tie_inclusive.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    md = ["# Figure 4b WR sensitivity - tie-inclusive definitions", "",
          "Same records, pairing, groups, 10,000 group-bootstrap replicates and seed as analyze_paired_shift.py; only the WR scoring changes so every paired item stays in both condition denominators.", "",
          "| WR definition | Judge | Common prompts | Groups | Orch shift pp | Qwen shift pp | Difference pp (95% group-bootstrap CI) |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for x in out:
        md.append(f'| {x["wr_definition"]} | {x["judge"]} | {x["n_common_prompts"]} | {x["n_groups"]} | {x["orch_shift_pp"]:+.2f} | {x["qwen_shift_pp"]:+.2f} | {x["did_pp"]:+.2f} [{x["ci_low"]:+.2f}, {x["ci_high"]:+.2f}] |')
    (OUT / "RESULTS_WR_TIE_INCLUSIVE.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
