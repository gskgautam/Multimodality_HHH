"""Collect the analysis outputs that make_figures.py reads into one flat folder.

Usage (from Method/OrchCircuit/final):
    python3 figures/collect_figure_data.py figdata
    python3 figures/make_figures.py figdata figures/rendered
"""
import shutil, sys
from pathlib import Path
FINAL = Path(__file__).resolve().parents[1]
MAP = {
 "fig2_judge_validity_f_frontier_judges_inter_judge_agreement.json": "fig2_judge_validity/f_frontier_judges/inter_judge_agreement.json",
 "fig2_judge_validity_f_frontier_judges_mechanism_strata.json": "fig2_judge_validity/f_frontier_judges/mechanism_strata.json",
 "fig2_judge_validity_f_frontier_judges_results.json": "fig2_judge_validity/f_frontier_judges/results.json",
 "review_group1_fills.json": "review/group1_fills.json",
 "review_review_response_analyses.json": "review/review_response_analyses.json",
 "runs_fullset_group5fold_results.json": "runs/fullset_group5fold/results.json",
 "fig5_human_transfer_archive_PRIVATE_pre_updated_human_ratings_2026-11-13_PRIVATE_3e_RESULTS.md": "fig5_human_transfer/c_superseded_round/3e/RESULTS.md",
 "fig5_human_transfer_e_answer_key_audit_like_for_like_results.json": "fig5_human_transfer/e_answer_key_audit/like_for_like_results.json",
 "fig5_human_transfer_e_answer_key_audit_results.json": "fig5_human_transfer/e_answer_key_audit/results.json",
 "fig5_human_transfer_f_objective_key_accuracy_results.json": "fig5_human_transfer/f_objective_key_accuracy/results.json",
 "fig6_robustness_c_source_strata_source_metric_strata.csv": "fig6_robustness/c_source_strata/source_metric_strata.csv",
 "review_resolve_o1_o5.json": "review/resolve_o1_o5.json",
 "figure_tables.md": "figures/figure_tables.md"
}
out = Path(sys.argv[1] if len(sys.argv) > 1 else 'figdata'); out.mkdir(parents=True, exist_ok=True)
for flat, rel in MAP.items():
    shutil.copy2(FINAL / rel, out / flat)
print(f'copied {len(MAP)} files to {out}')
