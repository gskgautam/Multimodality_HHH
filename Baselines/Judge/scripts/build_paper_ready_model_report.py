#!/usr/bin/env python3
"""Build paper-ready metric summaries for one completed Ch3Ef model run."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True)
    args = parser.parse_args()
    model_dir = ROOT / "evaluations" / args.slug
    run_dir = model_dir / "ch3ef-full1416"
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    wrti = json.loads((run_dir / "wr_ti_gpt4o_summary.json").read_text())
    ss = json.loads((run_dir / "ss_beaver_dam.json").read_text())
    if meta.get("valid_responses") != 1416:
        raise SystemExit(f"Generation is incomplete: {meta.get('valid_responses')}/1416 valid")
    if ss.get("n") != 487:
        raise SystemExit(f"SS is incomplete: {ss.get('n')}/487")
    if wrti.get("wr", {}).get("n") not in (624, 625) or wrti.get("ti", {}).get("n") != 304:
        raise SystemExit("GPT-4o WR/TI judging is incomplete")

    wr = wrti["wr"]
    ti = wrti["ti"]
    lines = [
        f"# Paper-ready results: {args.slug} on Ch3Ef",
        "",
        "This report summarizes automated model evaluation. WR and TI were judged by GPT-4o; SS was scored by PKU-Alignment/Beaver-Dam 7B.",
        "",
        "## Final metrics",
        "",
        f"- **WR:** {wr['percent']:.2f}% ({wr['wins']}/{wr['n']} valid pairwise judgments; {wr['ties']} ties).",
        f"- **SS:** {ss['ss_percent']:.2f}% ({ss['unsafe_count']}/{ss['n']} outputs flagged unsafe at threshold {ss['threshold']}). Lower is safer.",
        f"- **TI product:** {ti['ti_product_percent']:.2f}%.",
        f"- **Truthful:** {ti['truthful_percent']:.2f}% ({ti['truthful_count']}/{ti['n']}).",
        f"- **Informative:** {ti['informative_percent']:.2f}% ({ti['informative_count']}/{ti['n']}).",
        f"- **Both truthful and informative:** {ti['both_true_and_informative_percent']:.2f}%.",
        "",
        "Any off-format choice IDs are normalized only when they unambiguously match the dataset option encoding; raw model outputs are retained in responses.jsonl and the normalization count/rule are in run_metadata.json.",
        "WR is candidate wins divided by valid pairwise judgments; ties are shown separately and remain in the denominator. TI product is the product of the truthful and informative marginal rates. SS is the Beaver-Dam unsafe rate and uses text-only moderation.",
        "",
        "## Source files",
        "",
        f"- Responses: {(run_dir / 'responses.jsonl').relative_to(ROOT)}",
        f"- Run metadata: {(run_dir / 'run_metadata.json').relative_to(ROOT)}",
        f"- WR/TI judge journal: {(run_dir / 'wr_ti_gpt4o.jsonl').relative_to(ROOT)}",
        f"- WR/TI summary: {(run_dir / 'wr_ti_gpt4o_summary.json').relative_to(ROOT)}",
        f"- SS summary: {(run_dir / 'ss_beaver_dam.json').relative_to(ROOT)}",
        "",
        "## Interpretation limits",
        "",
        "These metrics describe this model's outputs under the Ch3Ef prompt and the automated evaluators used here. They are not accuracy against human-verified correct answers, and the SS classifier is text-only even though the candidate model receives images.",
        "",
    ]
    judge_rows = [
        json.loads(line)
        for line in (run_dir / "wr_ti_gpt4o.jsonl").read_text().splitlines()
        if line.strip()
    ]
    recovery_rows = [
        row for row in judge_rows
        if row.get("task") == "ti_judge" and row.get("judge_prompt_variant") not in (None, "", "original")
    ]
    if recovery_rows:
        items = ", ".join(
            f"{dimension} record {record_id}"
            for dimension, record_id in sorted({(row["dimension"], row["record_id"]) for row in recovery_rows})
        )
        note = (
            f"{len(recovery_rows)} TI judgment(s) ({items}) used neutral research-annotation wording after GPT-4o refused the original prompt. "
            "The image, candidate response, and truthfulness/informativeness criteria were unchanged; the prompt wording differed from the default."
        )
        index = lines.index("## Interpretation limits")
        lines[index:index] = ["## Prompt recovery", "", note, ""]
    report = "\n".join(lines)
    (run_dir / "summary-report.md").write_text(report)
    (model_dir / "paper-ready-results.md").write_text(report)
    print(f"Wrote {run_dir / 'summary-report.md'} and {model_dir / 'paper-ready-results.md'}")


if __name__ == "__main__":
    main()
