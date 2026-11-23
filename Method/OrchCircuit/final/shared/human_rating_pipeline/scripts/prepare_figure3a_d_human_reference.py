#!/usr/bin/env python3
"""Prepare a blinded, model-balanced human reference packet for Figure 3a-d.

Run from the multimodality repository root. All outputs are written under the
git-ignored Ch3Ef data directory and must remain private.
"""

from __future__ import annotations

import csv
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


ROOT = (next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()))
DATA = ROOT / "Datasets/Ch3EF/data/ch3ef_full1416"
LABELS = DATA / "joint_hhh_labels_600_PRIVATE"
PILOT = DATA / "human_rubric_pilot_20261112"
OUTPUT = DATA / "human_validation_figure3a_d_20261112_PRIVATE"
SEED = 2026111202
ANCHOR_MODEL = "qwen2.5-vl-7b"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def response_text(response: dict[str, Any], options: list[str]) -> str:
    choice_id = str(response.get("choice_id", "")).strip()
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    selected = ""
    if choice_id.upper() in letters[: len(options)]:
        idx = letters.index(choice_id.upper())
        selected = options[idx]
    else:
        try:
            idx = int(choice_id) - 1
            if 0 <= idx < len(options):
                selected = options[idx]
        except (TypeError, ValueError):
            pass
    parts = [f"Selected option {choice_id or '(not specified)'}: {selected}".rstrip()]
    rationale = response.get("rationale")
    if rationale:
        parts.append(f"Rationale: {str(rationale).strip()}")
    uncertainty = response.get("uncertain")
    if uncertainty not in (None, "", "None"):
        is_uncertain = str(uncertainty).lower() in ("true", "1", "yes")
        parts.append(f"Model uncertainty marker: {'yes' if is_uncertain else 'no'}")
    return "\n".join(parts)


def format_options(options: list[str]) -> str:
    return "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options))


def balanced_assignment(records: list[dict[str, Any]], models: list[str], rng: random.Random) -> list[tuple[dict[str, Any], str]]:
    shuffled_records = list(records)
    rng.shuffle(shuffled_records)
    assignments = (models * ((len(records) + len(models) - 1) // len(models)))[: len(records)]
    rng.shuffle(assignments)
    return list(zip(shuffled_records, assignments))


def image_montage(source_paths: list[Path], target: Path) -> None:
    """Stack every source image without cropping; downscale only above 1600 px wide."""
    images: list[Image.Image] = []
    for source in source_paths:
        with Image.open(source) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
            if img.width > 1600:
                new_h = round(img.height * 1600 / img.width)
                img = img.resize((1600, new_h), Image.Resampling.LANCZOS)
            images.append(img.copy())
    if not images:
        raise ValueError(f"No images for {target.name}")
    gap = 24
    width = max(im.width for im in images)
    height = sum(im.height for im in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        x = (width - image.width) // 2
        canvas.paste(image, (x, y))
        y += image.height + gap
    canvas.save(target.with_suffix(".jpg"), format="JPEG", quality=90, optimize=False)


def valid_response(row: dict[str, Any]) -> bool:
    response = row.get("response")
    no_error = lambda value: value in (None, "", "None", "null", "False")
    return (
        isinstance(response, dict)
        and bool(response.get("choice_id"))
        and no_error(row.get("error"))
        and no_error(row.get("parse_error"))
    )


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing private output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    image_dir = OUTPUT / "images"
    image_dir.mkdir()

    dataset = json.loads((DATA / "sample.json").read_text(encoding="utf-8"))
    records_by_order = {int(row["sample_order"]): row for row in dataset["records"]}
    selection = json.loads((LABELS / "sample_manifest.json").read_text(encoding="utf-8"))
    selected_rows = selection["record_ids"]

    with (PILOT / "identity_key_PRIVATE.csv").open(encoding="utf-8-sig", newline="") as stream:
        pilot_keys = {
            (row["dimension"], int(row["sample_order"]))
            for row in csv.DictReader(stream)
        }

    model_ids = list(selection["candidate_models"])
    response_maps: dict[str, dict[int, dict[str, Any]]] = {}
    for model in model_ids:
        family = "Open" if (ROOT / "Baselines/Open" / model).exists() else "Closed"
        path = ROOT / "Baselines" / family / model / "runs/ch3ef-full1416/responses.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        # Match the existing GPT-4o label-generation join: the last row for a
        # sample_order in the canonical model response file is authoritative.
        rows = read_jsonl(path)
        response_maps[model] = {
            int(row["sample_order"]): row
            for row in rows
            if row.get("sample_order") is not None
        }

    gpt_rows = read_jsonl(LABELS / "gpt4o_joint_labels_v2.jsonl")
    gpt_by_task: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in gpt_rows:
        if not row.get("error") and row.get("labels"):
            key = (row["dimension"], row["task"], int(row["sample_order"]))
            gpt_by_task[key] = row

    beaver_rows = read_jsonl(LABELS / "beaver_dam_ss_600x24.jsonl")
    beaver_by_model_order = {
        (str(row["model"]), int(row["sample_order"])): row
        for row in beaver_rows
    }

    rng = random.Random(SEED)
    packet: dict[str, Any] = {
        "title": "Figure 3a-d human reference ratings",
        "status": "Development-set judge validation; not an untouched test.",
        "sampling_seed": SEED,
        "sheets": {},
    }
    key_rows: list[dict[str, Any]] = []
    manifest_panels: dict[str, Any] = {}

    by_dimension: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected_rows:
        by_dimension[item["dimension"]].append(item)

    # 3b: all eligible GPT-4o pairwise judgments, with the Qwen2.5-VL 7B
    # response retained as the fixed anchor.
    helpful = [
        row for row in by_dimension["Helpful"]
        if ("Helpful", int(row["sample_order"])) not in pilot_keys
    ]
    helpful_available = len(helpful)
    rng.shuffle(helpful)
    wr_rows: list[dict[str, Any]] = []
    wr_candidates = [model for model in model_ids if model != ANCHOR_MODEL]
    for record, candidate_model in balanced_assignment(helpful, wr_candidates, rng):
        order = int(record["sample_order"])
        judge = gpt_by_task.get(("Helpful", "wr", order))
        if not judge:
            raise ValueError(f"Missing successful GPT-4o WR judgment at sample_order={order}")
        mapping = judge["candidate_map"]
        inverse = {model: label for label, model in mapping.items()}
        anchor_model = mapping[judge["anchor_id"]]
        if anchor_model != ANCHOR_MODEL:
            raise ValueError(f"Unexpected WR anchor {anchor_model} at sample_order={order}")
        candidate_label = inverse[candidate_model]
        judge_preference = judge["labels"][candidate_label]["preference"]
        if judge_preference not in {"win", "loss", "tie"}:
            raise ValueError(f"Unexpected WR output {judge_preference}")
        candidate_resp = response_maps[candidate_model][order]
        anchor_resp = response_maps[anchor_model][order]
        if not valid_response(candidate_resp) or not valid_response(anchor_resp):
            raise ValueError(f"Invalid WR candidate or anchor response at sample_order={order}")
        item = records_by_order[order]
        code = f"WR-{rng.getrandbits(32):08X}"
        candidate_text = response_text(candidate_resp["response"], item["options"])
        anchor_text = response_text(anchor_resp["response"], item["options"])
        if rng.choice((True, False)):
            response_a, response_b, candidate_position = candidate_text, anchor_text, "A"
        else:
            response_a, response_b, candidate_position = anchor_text, candidate_text, "B"
        image_file = f"images/{code}.jpg"
        image_sources = [DATA / path for path in item["image"]]
        for path in image_sources:
            if not path.exists():
                raise FileNotFoundError(path)
        image_montage(image_sources, OUTPUT / image_file)
        wr_rows.append({
            "case_code": code,
            "image_file": image_file,
            "question": item["query"],
            "answer_options": format_options(item["options"]),
            "response_A": response_a,
            "response_B": response_b,
            "rating": "",
            "evidence_note": "",
        })
        key_rows.append({
            "case_code": code, "panel": "3b_WR", "sample_order": order,
            "record_id": item["id"], "dimension": "Helpful",
            "candidate_model_id": candidate_model, "anchor_model_id": anchor_model,
            "candidate_position": candidate_position, "judge_output": judge_preference,
            "judge_model": judge.get("judge_model"),
            "prompt_version": judge.get("prompt_version"),
        })

    rng.shuffle(wr_rows)
    packet["sheets"]["WR"] = wr_rows
    manifest_panels["3b_WR"] = {
        "n": len(wr_rows), "unique_prompts": len(wr_rows),
        "available_after_prior_pilot_exclusion": helpful_available,
        "prior_pilot_excluded": 200 - helpful_available,
        "eligible_not_sampled": 0,
        "models": dict(Counter(row["candidate_model_id"] for row in key_rows if row["panel"] == "3b_WR")),
        "judge_outputs": dict(Counter(row["judge_output"] for row in key_rows if row["panel"] == "3b_WR")),
        "anchor_model": ANCHOR_MODEL,
    }

    # 3c/3d: one response per remaining record, with generator identities
    # balanced across the 24 systems and labels hidden from raters.
    for dimension, sheet_name, panel in [
        ("Harmless", "SS", "3c_SS"),
        ("Honest", "TI", "3d_TI"),
    ]:
        source_records = [
            row for row in by_dimension[dimension]
            if (dimension, int(row["sample_order"])) not in pilot_keys
        ]
        available_after_pilot = len(source_records)
        rng.shuffle(source_records)
        panel_rows: list[dict[str, Any]] = []
        panel_key_start = len(key_rows)
        for record, model in balanced_assignment(source_records, model_ids, rng):
            order = int(record["sample_order"])
            model_response = response_maps[model][order]
            if not valid_response(model_response):
                raise ValueError(f"Invalid {dimension} response for {model} at sample_order={order}")
            item = records_by_order[order]
            code = f"{sheet_name}-{rng.getrandbits(32):08X}"
            image_file = f"images/{code}.jpg"
            image_sources = [DATA / path for path in item["image"]]
            for path in image_sources:
                if not path.exists():
                    raise FileNotFoundError(path)
            image_montage(image_sources, OUTPUT / image_file)
            candidate_text = response_text(model_response["response"], item["options"])
            judge_output: dict[str, Any]
            if dimension == "Harmless":
                judge = beaver_by_model_order.get((model, order))
                if judge is None:
                    raise ValueError(f"Missing Beaver-Dam result for {model} at {order}")
                judge_output = {"unsafe": bool(judge["unsafe"])}
                panel_rows.append({
                    "case_code": code, "image_file": image_file,
                    "question": item["query"], "answer_options": format_options(item["options"]),
                    "candidate_response": candidate_text, "safety_rating": "",
                    "refusal": "", "evidence_note": "",
                })
                key_rows.append({
                    "case_code": code, "panel": panel, "sample_order": order,
                    "record_id": item["id"], "dimension": dimension,
                    "candidate_model_id": model, "judge_output": judge_output["unsafe"],
                    "judge_model": "PKU-Alignment/beaver-dam-7b", "judge_threshold": judge.get("threshold"),
                })
            else:
                judge = gpt_by_task.get(("Honest", "ti", order))
                if not judge:
                    raise ValueError(f"Missing successful GPT-4o TI judgment at sample_order={order}")
                mapping = judge["candidate_map"]
                inverse = {candidate_model: label for label, candidate_model in mapping.items()}
                if model not in inverse:
                    raise ValueError(f"Missing candidate {model} in GPT-4o TI labels at {order}")
                label = judge["labels"][inverse[model]]
                judge_output = {
                    "truthful": bool(label["truthful"]),
                    "informative": bool(label["informative"]),
                }
                panel_rows.append({
                    "case_code": code, "image_file": image_file,
                    "question": item["query"], "answer_options": format_options(item["options"]),
                    "candidate_response": candidate_text, "truthfulness": "",
                    "informativeness": "", "evidence_note": "",
                })
                key_rows.append({
                    "case_code": code, "panel": panel, "sample_order": order,
                    "record_id": item["id"], "dimension": dimension,
                    "candidate_model_id": model,
                    "judge_output_truthful": judge_output["truthful"],
                    "judge_output_informative": judge_output["informative"],
                    "judge_model": judge.get("judge_model"),
                    "prompt_version": judge.get("prompt_version"),
                })
        rng.shuffle(panel_rows)
        packet["sheets"][sheet_name] = panel_rows
        rows_key = key_rows[panel_key_start:]
        manifest_panels[panel] = {
            "n": len(panel_rows), "unique_prompts": len(panel_rows),
            "available_after_prior_pilot_exclusion": available_after_pilot,
            "prior_pilot_excluded": 200 - available_after_pilot,
            "eligible_not_sampled": 0,
            "models": dict(Counter(row["candidate_model_id"] for row in rows_key)),
            "judge_outputs": (
                dict(Counter(str(row["judge_output"]) for row in rows_key))
                if dimension == "Harmless"
                else {
                    "truthful": dict(Counter(str(row["judge_output_truthful"]) for row in rows_key)),
                    "informative": dict(Counter(str(row["judge_output_informative"]) for row in rows_key)),
                }
            ),
        }

    (OUTPUT / "packet_data_FOR_RATERS_PRIVATE.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    key_path = OUTPUT / "identity_key_PRIVATE.csv"
    fieldnames = sorted({field for row in key_rows for field in row})
    with key_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(key_rows)

    manifest = {
        "purpose": "Blinded independent human validation of automated HHH judges and human label reliability.",
        "evidence_status": "Development-set validation; the same 600 prompts were used for automated labels and OrchCircuit development, so this is not a confirmatory method test.",
        "dataset": "Ch3Ef; 200 randomly sampled all-24-valid prompts per HHH dimension from the 600-prompt joint-label development set.",
        "source_manifest": "Datasets/Ch3EF/data/ch3ef_full1416/joint_hhh_labels_600_PRIVATE/sample_manifest.json",
        "prior_pilot_exclusion": "Exclude matching (dimension, sample_order) cases from the 45-case rubric pilot.",
        "sampling_seed": SEED,
        "selection": "Use every eligible prompt in the 600-prompt development set after excluding the prior 45-case pilot. Select one candidate response per prompt, with generator models balanced as evenly as possible. WR compares one non-anchor model response against the Qwen2.5-VL 7B anchor, matching the existing GPT-4o WR label.",
        "raters": ["R01", "R02", "R03"],
        "independence": "All three raters receive the same cases in the same randomized order, rate independently, and do not see model identities or automated judge labels.",
        "panels": manifest_panels,
        "data_handling": "Human-facing workbook and image assets contain prompts and model responses and must remain private. The identity key must never be given to raters or committed to GitHub.",
    }
    (OUTPUT / "sample_manifest_PRIVATE.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # The reviewer archive deliberately omits identity_key_PRIVATE.csv.
    archive_path = OUTPUT / "review_packet_assets_PRIVATE.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(OUTPUT / "packet_data_FOR_RATERS_PRIVATE.json", "packet_data.json")
        for image in sorted(image_dir.glob("*.jpg")):
            archive.write(image, f"images/{image.name}")

    print(json.dumps({
        "output": str(OUTPUT),
        "panels": manifest_panels,
        "image_assets": len(list(image_dir.glob("*.jpg"))),
        "archive_bytes": archive_path.stat().st_size,
        "reviewer_identity_key_in_archive": False,
    }, indent=2))


if __name__ == "__main__":
    main()
