#!/usr/bin/env python3
"""Create publication-style Figure 3a-d plots from aggregate result JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

PALETTE = {
    # Figure 3a: coordinated blue-purple reliability palette.
    "wr": "#9b8ac7",
    "safety": "#7d9ed1",
    "refusal": "#b3a7d8",
    "truth": "#7185b8",
    "info": "#c4b9e5",
    "edge": "#4b4b4b",
    "grid": "#d8d8d8",
    "ink": "#171717",
}


def style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 16,
        "axes.labelweight": "bold",
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "text.color": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "axes.edgecolor": PALETTE["edge"],
        "axes.linewidth": 1.1,
        "grid.color": PALETTE["grid"],
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        "svg.fonttype": "none",
    })


def save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def panel_a(data, out):
    rel = data["interrater_reliability"]
    order = [
        ("WR_helpfulness", "WR · Helpfulness", PALETTE["wr"]),
        ("SS_safety", "SS · Safety", PALETTE["safety"]),
        ("SS_refusal", "SS · Refusal", PALETTE["refusal"]),
        ("TI_truthfulness", "TI · Truthfulness", PALETTE["truth"]),
        ("TI_informativeness", "TI · Informativeness", PALETTE["info"]),
    ]
    fig, ax = plt.subplots(figsize=(10, 6.4))
    ys = np.arange(len(order))[::-1]
    for y, (key, label, color) in zip(ys, order):
        r = rel[key]
        k = r["fleiss_kappa"]
        lo, hi = r["fleiss_kappa_bootstrap_95"]
        ax.hlines(y, lo, hi, color=color, linewidth=7, alpha=0.82, zorder=2)
        ax.plot(k, y, marker="o", ms=10, color=color, mec=PALETTE["edge"], mew=1.2, zorder=3)
        ax.text(1.04, y, f"κ={k:.2f}  ·  unanimous {r['unanimous_cases']}/{r['n_cases']}",
                transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=11.5, fontweight="bold", clip_on=False)
    ax.set_yticks(ys, [x[1] for x in order])
    ax.set_xlim(0, 1.04)
    ax.set_ylim(-0.65, len(order)-0.35)
    ax.set_xlabel("Fleiss’ κ (2,000-case-bootstrap 95% CI)")
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="x", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.23, right=0.68, bottom=0.10, top=0.97)
    save(fig, out / "figure3a")


def draw_matrix(ax, matrix, labels, title, n, subtitle=None, xlabels=None, ylabels=None,
                show_note=True, diag_color="#acd9c5", off_color="#f3aaaa"):
    arr = np.asarray(matrix, dtype=int)
    row_sums = arr.sum(axis=1)
    ax.set_xlim(0, len(labels)); ax.set_ylim(len(labels), 0)
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = int(arr[i, j])
            pct = 100 * count / row_sums[i] if row_sums[i] else 0
            face = diag_color if i == j else off_color
            alpha = 0.36 + 0.52 * min(pct / 100, 1)
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face, alpha=alpha,
                                   edgecolor="white", linewidth=2.5))
            ax.text(j + 0.5, i + 0.44, f"{count}", ha="center", va="center",
                    fontsize=17 if len(labels) == 2 else 15, fontweight="bold")
            ax.text(j + 0.5, i + 0.70, f"{pct:.1f}%", ha="center", va="center",
                    fontsize=10.5, color="#333333")
    ax.set_xticks(np.arange(len(labels)) + 0.5, xlabels or labels, fontweight="bold")
    ax.set_yticks(np.arange(len(labels)) + 0.5, ylabels or labels, fontweight="bold")
    ax.tick_params(length=0, pad=8)
    ax.set_xlabel("Automated judge", labelpad=10)
    ax.set_ylabel("Human majority", labelpad=10)
    if title:
        ax.set_title(title, fontfamily="DejaVu Serif", pad=14)
    if show_note:
        ax.text(0.5, -0.25, f"n={n}" + (f"  ·  {subtitle}" if subtitle else ""),
                transform=ax.transAxes, ha="center", va="top", fontsize=11.5, fontweight="bold")
    for spine in ax.spines.values(): spine.set_visible(False)


def panel_b(data, out):
    r = data["automated_judge_comparison"]["WR"]
    labels = ["A", "B", "Tie"]
    cm = [[r["human_by_judge_confusion"][h][j.lower() if j == "Tie" else j]
           for j in labels] for h in ("A", "B", "tie")]
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    draw_matrix(ax, cm, labels, "", r["eligible_exact_match_n"], show_note=False,
                diag_color="#c9b8e8", off_color="#e9b5c2")
    ax.set_xlabel("GPT-4o WR choice", labelpad=10)
    ax.set_ylabel("Human WR majority", labelpad=10)
    fig.subplots_adjust(left=0.23, right=0.96, top=0.97, bottom=0.18)
    save(fig, out / "figure3b")


def panel_c(data, out):
    r = data["automated_judge_comparison"]["SS"]["beaver_unsafe_confusion"]
    cm = [[r["tp"], r["fn"]], [r["fp"], r["tn"]]]
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    draw_matrix(ax, cm, ["x", "y"], "", r["n"], show_note=False,
                xlabels=["Flagged unsafe", "Not flagged"], ylabels=["Harmful/unsafe", "Safe"],
                diag_color="#a7d8d0", off_color="#f0b19e")
    ax.set_xlabel("Beaver-Dam unsafe flag", labelpad=10)
    ax.set_ylabel("Human safety majority", labelpad=10)
    fig.subplots_adjust(left=0.25, right=0.96, top=0.97, bottom=0.18)
    save(fig, out / "figure3c")


def panel_d(data, out):
    ti = data["automated_judge_comparison"]["TI"]
    t = ti["truthfulness"]["gpt4o_truthful_confusion"]
    i = ti["informativeness"]["gpt4o_informative_confusion"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2))
    draw_matrix(axes[0], [[t["tp"], t["fn"]], [t["fp"], t["tn"]]],
                ["Positive", "Negative"], "", t["n"], show_note=False,
                diag_color="#a9c5ea", off_color="#dfb5d5")
    draw_matrix(axes[1], [[i["tp"], i["fn"]], [i["fp"], i["tn"]]],
                ["Positive", "Negative"], "", i["n"], show_note=False,
                diag_color="#f1c99d", off_color="#e7a7a5")
    axes[0].set_xlabel("GPT-4o truthfulness", labelpad=10)
    axes[0].set_ylabel("Human truthfulness majority", labelpad=10)
    axes[1].set_xlabel("GPT-4o informativeness", labelpad=10)
    axes[1].set_ylabel("Human informativeness majority", labelpad=10)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.16, wspace=0.35)
    save(fig, out / "figure3d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--figure3-root", type=Path, required=True)
    args = ap.parse_args()
    style()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    panel_a(data, args.figure3_root / "a")
    panel_b(data, args.figure3_root / "b")
    panel_c(data, args.figure3_root / "c")
    panel_d(data, args.figure3_root / "d")
    print("Wrote figure3a-d PNG/SVG charts.")


if __name__ == "__main__":
    main()
