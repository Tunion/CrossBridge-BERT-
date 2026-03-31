from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pipeline with multiple seeds and aggregate manual-eval metrics.")
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--max-slices", type=int, default=20000)
    p.add_argument("--epochs-dual", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--prototype-k", type=int, default=8)
    p.add_argument("--dual-mechanism-invariance-enable", dest="dual_mechanism_invariance_enable", action="store_true")
    p.add_argument("--no-dual-mechanism-invariance-enable", dest="dual_mechanism_invariance_enable", action="store_false")
    p.add_argument("--dual-lambda-mechanism-invariance", type=float, default=0.20)
    p.add_argument("--dual-mechanism-invariance-edge-drop-rate", type=float, default=0.12)
    p.add_argument("--dual-mechanism-invariance-node-drop-rate", type=float, default=0.00)
    p.add_argument("--dual-mechanism-invariance-max-hops", type=int, default=4)
    p.add_argument("--dual-mechanism-invariance-skeleton-scale", type=float, default=0.50)
    p.add_argument("--dual-mechanism-invariance-min-context-degree", type=int, default=2)
    p.add_argument("--unsup-energy-head-enable", dest="unsup_energy_head_enable", action="store_true")
    p.add_argument("--no-unsup-energy-head-enable", dest="unsup_energy_head_enable", action="store_false")
    p.add_argument("--unsup-energy-head-mode", type=str, default="proto_hybrid", choices=["full", "proto_hybrid", "proto_free_energy"])
    p.add_argument("--unsup-energy-head-free-energy-temp", type=float, default=0.35)
    p.add_argument("--unsup-energy-head-free-energy-prior", type=str, default="uniform", choices=["uniform", "support", "sqrt_support"])
    p.add_argument("--unsup-energy-head-hidden-dim", type=int, default=64)
    p.add_argument("--unsup-energy-head-epochs", type=int, default=40)
    p.add_argument("--unsup-energy-head-batch-size", type=int, default=256)
    p.add_argument("--unsup-energy-head-lr", type=float, default=1e-3)
    p.add_argument("--unsup-energy-head-weight-decay", type=float, default=1e-4)
    p.add_argument("--unsup-energy-head-noise-std", type=float, default=0.05)
    p.add_argument("--unsup-energy-head-dropout", type=float, default=0.10)
    p.add_argument("--unsup-energy-head-out", type=str, default="outputs/models/energy_head.pt")
    p.add_argument("--unsup-energy-head-loss-mode", type=str, default="proto_margin", choices=["softplus_neg", "proto_margin"])
    p.add_argument("--unsup-energy-head-pos-quantile", type=float, default=0.30)
    p.add_argument("--unsup-energy-head-neg-quantile", type=float, default=0.85)
    p.add_argument("--unsup-energy-head-rank-margin", type=float, default=0.35)
    p.add_argument("--unsup-energy-head-rank-weight", type=float, default=1.0)
    p.add_argument("--unsup-energy-head-calib-weight", type=float, default=0.50)
    p.add_argument("--unsup-energy-head-anchor-cls-weight", type=float, default=0.50)
    p.add_argument("--unsup-energy-head-stable-train", dest="unsup_energy_head_stable_train", action="store_true")
    p.add_argument("--no-unsup-energy-head-stable-train", dest="unsup_energy_head_stable_train", action="store_false")
    p.add_argument("--prebuilt-graph-dir", type=str, default="")
    p.add_argument("--graph-report-out", type=str, default="outputs/results/graph_semantic_coverage.json")
    p.add_argument("--implicit-mechanism-edges", dest="implicit_mechanism_edges", action="store_true")
    p.add_argument("--no-implicit-mechanism-edges", dest="implicit_mechanism_edges", action="store_false")
    p.add_argument("--unsup-mechanism-risk-weight", type=float, default=0.0)
    p.add_argument("--unsup-mechanism-top-quantile", type=float, default=1.0)
    p.add_argument("--unsup-mechanism-semantic-gate", action="store_true")
    p.add_argument("--unsup-mechanism-semantic-min-pairs", type=int, default=2)
    p.add_argument("--unsup-mechanism-semantic-max-hops", type=int, default=6)
    p.add_argument("--unsup-family-aware-weight", type=float, default=0.30)
    p.add_argument("--unsup-family-top-quantile", type=float, default=1.0)
    p.add_argument("--unsup-family-min-samples", type=int, default=8)
    p.add_argument("--unsup-family-boundary-quantile", type=float, default=0.9)
    p.add_argument("--unsup-family-shrinkage-tau", type=float, default=16.0)
    p.add_argument("--unsup-family-ramp-width", type=int, default=6)
    p.add_argument("--unsup-family-global-mix-floor", type=float, default=0.20)
    p.add_argument("--unsup-family-radius-min-ratio", type=float, default=0.60)
    p.add_argument("--unsup-family-radius-max-ratio", type=float, default=1.80)
    p.add_argument("--unsup-adaptive-boundary-refine", dest="unsup_adaptive_boundary_refine", action="store_true")
    p.add_argument("--no-unsup-adaptive-boundary-refine", dest="unsup_adaptive_boundary_refine", action="store_false")
    p.add_argument("--unsup-adaptive-boundary-focus-quantile", type=float, default=0.65)
    p.add_argument("--unsup-adaptive-boundary-min-scale", type=float, default=0.85)
    p.add_argument("--unsup-adaptive-boundary-max-scale", type=float, default=1.20)
    p.add_argument("--unsup-adaptive-boundary-outside-scale", type=float, default=1.00)
    p.add_argument(
        "--risk-rerank-mode",
        type=str,
        default="function_coverage",
        choices=["none", "coverage", "function_coverage", "family_hybrid"],
    )
    p.add_argument("--build-curated-standard", dest="build_curated_standard", action="store_true")
    p.add_argument("--no-build-curated-standard", dest="build_curated_standard", action="store_false")
    p.add_argument("--train-list", type=str, default="processed/dataset_non_overlap_local.txt")
    p.add_argument("--curated-output-list", type=str, default="processed/dataset_non_overlap_curated_standard.txt")
    p.add_argument("--curated-output-report", type=str, default="processed/dataset_non_overlap_curated_standard_report.json")
    p.add_argument("--curated-max-per-bridge", type=int, default=120)
    p.add_argument("--curated-target-size", type=int, default=2500)
    p.add_argument("--curated-exclude-bridges", type=str, default="__NONE__")
    p.add_argument("--curated-drop-common-libs", dest="curated_drop_common_libs", action="store_true")
    p.add_argument("--no-curated-drop-common-libs", dest="curated_drop_common_libs", action="store_false")
    p.add_argument("--manual-scope", choices=["label-files", "all-sol"], default="label-files")
    p.add_argument("--manual-max-files", type=int, default=0)
    p.add_argument("--manual-mechanism-risk-weight", type=float, default=0.0)
    p.add_argument("--manual-mechanism-top-quantile", type=float, default=1.0)
    p.add_argument("--manual-mechanism-semantic-gate", action="store_true")
    p.add_argument("--manual-mechanism-semantic-min-pairs", type=int, default=2)
    p.add_argument("--manual-mechanism-semantic-max-hops", type=int, default=6)
    p.add_argument("--manual-family-aware-weight", type=float, default=0.30)
    p.add_argument("--manual-family-top-quantile", type=float, default=1.0)
    p.add_argument("--manual-family-min-samples", type=int, default=8)
    p.add_argument("--manual-family-boundary-quantile", type=float, default=0.9)
    p.add_argument("--manual-family-shrinkage-tau", type=float, default=16.0)
    p.add_argument("--manual-family-ramp-width", type=int, default=6)
    p.add_argument("--manual-family-global-mix-floor", type=float, default=0.20)
    p.add_argument("--manual-family-radius-min-ratio", type=float, default=0.60)
    p.add_argument("--manual-family-radius-max-ratio", type=float, default=1.80)
    p.add_argument("--manual-adaptive-boundary-refine", dest="manual_adaptive_boundary_refine", action="store_true")
    p.add_argument("--no-manual-adaptive-boundary-refine", dest="manual_adaptive_boundary_refine", action="store_false")
    p.add_argument("--manual-adaptive-boundary-focus-quantile", type=float, default=0.65)
    p.add_argument("--manual-adaptive-boundary-min-scale", type=float, default=0.85)
    p.add_argument("--manual-adaptive-boundary-max-scale", type=float, default=1.20)
    p.add_argument("--manual-adaptive-boundary-outside-scale", type=float, default=1.00)
    p.add_argument("--manual-line-risk-closure-boost", type=float, default=0.30)
    p.add_argument("--manual-line-risk-stage2-weight", type=float, default=0.15)
    p.add_argument("--manual-line-risk-stage2-top-functions", type=int, default=640)
    p.add_argument("--manual-line-risk-stage2-support-weight", type=float, default=0.60)
    p.add_argument("--manual-line-risk-stage2-max-file-share", type=float, default=0.0)
    p.add_argument("--manual-score-head-method", type=str, default="iforest", choices=["none", "iforest"])
    p.add_argument("--manual-score-head-fit-scope", type=str, default="low_risk", choices=["all", "low_risk", "normal"])
    p.add_argument("--manual-binary-head-method", type=str, default="energy_head", choices=["none", "ocsvm", "energy_head"])
    p.add_argument("--manual-binary-head-fit-scope", type=str, default="all", choices=["all", "low_risk", "normal"])
    p.add_argument("--manual-binary-head-contamination", type=float, default=0.08)
    p.add_argument("--manual-energy-head-ckpt", type=str, default="outputs/models/energy_head.pt")
    p.add_argument("--manual-triage-low-quantile", type=float, default=0.0)
    p.add_argument("--manual-triage-high-quantile", type=float, default=0.90)
    p.add_argument("--manual-triage-gray-threshold", type=float, default=0.28)
    p.add_argument("--manual-triage-family-aware-gray", dest="manual_triage_family_aware_gray", action="store_true")
    p.add_argument("--no-manual-triage-family-aware-gray", dest="manual_triage_family_aware_gray", action="store_false")
    p.add_argument("--manual-triage-family-min-samples", type=int, default=32)
    p.add_argument("--manual-triage-adaptive-gray", dest="manual_triage_adaptive_gray", action="store_true")
    p.add_argument("--no-manual-triage-adaptive-gray", dest="manual_triage_adaptive_gray", action="store_false")
    p.add_argument("--manual-triage-adaptive-gray-source", type=str, default="outputs/results/unsup_final_risk_scores.csv")
    p.add_argument("--manual-triage-adaptive-gray-column", type=str, default="energy_head_score")
    p.add_argument("--manual-triage-adaptive-gray-group-by", type=str, default="family", choices=["family", "prototype"])
    p.add_argument("--manual-binary-target-mode", type=str, default="file_function", choices=["file_function", "row_window2", "union_window2"])
    p.add_argument("--manual-binary-target-window", type=int, default=2)
    p.add_argument("--manual-binary-eval-topn", type=int, default=0)
    p.add_argument("--slice-hops", type=int, default=2)
    p.add_argument("--slice-forward-hops", type=int, default=0, help="0 means use --slice-hops")
    p.add_argument("--slice-backward-hops", type=int, default=0, help="0 means use --slice-hops")
    p.add_argument("--slice-formal-propagation", dest="slice_formal_propagation", action="store_true")
    p.add_argument("--no-slice-formal-propagation", dest="slice_formal_propagation", action="store_false")
    p.add_argument("--slice-mechanism-path-max-len", type=int, default=4)
    p.add_argument("--slice-min-nodes", type=int, default=6)
    p.add_argument("--slice-include-contains", dest="slice_include_contains", action="store_true")
    p.add_argument("--no-slice-include-contains", dest="slice_include_contains", action="store_false")
    p.add_argument("--slice-function-fallback-mode", choices=["none", "no-role", "all"], default="none")
    p.add_argument("--slice-function-fallback-min-nodes", type=int, default=3)
    p.add_argument("--slice-dedup-line-jaccard", type=float, default=0.95)
    p.add_argument("--slice-max-slices-per-function", type=int, default=120)
    p.add_argument("--slice-stats-out", type=str, default="outputs/results/state_slice_stats.json")
    p.add_argument("--dual-skeleton-mode", choices=["legacy", "formal", "dynamic"], default="formal")
    p.add_argument("--dual-mechanism-path-max-len", type=int, default=4)
    p.add_argument("--dual-dynamic-aux-keep-prob", type=float, default=0.35)
    p.add_argument("--dual-dynamic-seed", type=int, default=42)
    p.add_argument("--manual-slice-hops", type=int, default=2)
    p.add_argument("--manual-slice-forward-hops", type=int, default=0, help="0 means use --manual-slice-hops")
    p.add_argument("--manual-slice-backward-hops", type=int, default=0, help="0 means use --manual-slice-hops")
    p.add_argument("--manual-slice-formal-propagation", dest="manual_slice_formal_propagation", action="store_true")
    p.add_argument("--no-manual-slice-formal-propagation", dest="manual_slice_formal_propagation", action="store_false")
    p.add_argument("--manual-slice-mechanism-path-max-len", type=int, default=4)
    p.add_argument("--manual-min-slice-nodes", type=int, default=6)
    p.add_argument("--manual-slice-include-contains", dest="manual_slice_include_contains", action="store_true")
    p.add_argument("--no-manual-slice-include-contains", dest="manual_slice_include_contains", action="store_false")
    p.add_argument("--manual-slice-function-fallback-mode", choices=["none", "no-role", "all"], default="none")
    p.add_argument("--manual-slice-function-fallback-min-nodes", type=int, default=3)
    p.add_argument("--manual-slice-dedup-line-jaccard", type=float, default=0.95)
    p.add_argument("--manual-slice-max-slices-per-function", type=int, default=120)
    p.add_argument("--manual-slice-stats-out", type=str, default="outputs/results/manual_state_slice_stats.json")
    p.add_argument("--manual-dual-skeleton-mode", choices=["legacy", "formal", "dynamic"], default="legacy")
    p.add_argument("--manual-dual-mechanism-path-max-len", type=int, default=4)
    p.add_argument("--manual-dual-dynamic-aux-keep-prob", type=float, default=0.35)
    p.add_argument("--manual-dual-dynamic-seed", type=int, default=42)
    p.add_argument("--run-baseline", action="store_true")
    p.add_argument("--run-ablation", action="store_true")
    p.add_argument("--run-generalization-split", action="store_true")
    p.add_argument("--generalization-topk", type=str, default="20,50,100,200")
    p.add_argument("--generalization-line-window", type=int, default=2)
    p.add_argument(
        "--generalization-out-pattern",
        type=str,
        default="outputs/results/generalization_split_seed{seed}.json",
        help="Per-seed generalization split output path pattern.",
    )
    p.add_argument("--line-window", type=int, default=2, help="Window used for aggregated row recall summary.")
    p.add_argument("--out", type=str, default="outputs/results/multiseed_pipeline_summary.json")
    p.add_argument("--artifact-dir", type=str, default="outputs/results/multiseed_runs")
    p.set_defaults(
        build_curated_standard=True,
        curated_drop_common_libs=True,
        slice_formal_propagation=True,
        slice_include_contains=True,
        manual_slice_formal_propagation=True,
        manual_slice_include_contains=False,
        implicit_mechanism_edges=False,
        unsup_adaptive_boundary_refine=True,
        manual_adaptive_boundary_refine=True,
        unsup_energy_head_enable=True,
        unsup_energy_head_stable_train=False,
        dual_mechanism_invariance_enable=False,
        manual_triage_adaptive_gray=False,
    )
    return p.parse_args()


def parse_seeds(s: str) -> List[int]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("No valid seeds")
    return vals


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def topk_metric(summary: Dict[str, Any], topk: int, line_window: int = 0) -> Dict[str, Any]:
    if line_window == 0:
        arr = summary["recall_summary"]["topk_metrics"]
    else:
        arr = summary["recall_summary_by_line_window"][str(line_window)]["topk_metrics"]
    for x in arr:
        if int(x["top_k"]) == int(topk):
            return x
    return arr[-1]


def topk_metric_from_list(arr: List[Dict[str, Any]], topk: int) -> Dict[str, Any]:
    for x in arr:
        if int(x.get("top_k", -1)) == int(topk):
            return x
    return arr[-1] if arr else {}


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    py = sys.executable
    out_path = (ROOT / args.out).resolve()
    artifact_dir = (ROOT / args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    t0 = time.time()

    for seed in seeds:
        summary_rel = f"outputs/results/pipeline_run_seed{seed}.json"
        gen_out_rel = str(args.generalization_out_pattern).format(seed=seed)
        cmd = [
            py,
            "run_pipeline.py",
            "--seed",
            str(seed),
            "--max-files",
            str(args.max_files),
            "--max-slices",
            str(args.max_slices),
            "--epochs-dual",
            str(args.epochs_dual),
            "--hidden-dim",
            str(args.hidden_dim),
            "--num-layers",
            str(args.num_layers),
            "--device",
            args.device,
            "--prototype-k",
            str(args.prototype_k),
            "--dual-lambda-mechanism-invariance",
            str(args.dual_lambda_mechanism_invariance),
            "--dual-mechanism-invariance-edge-drop-rate",
            str(args.dual_mechanism_invariance_edge_drop_rate),
            "--dual-mechanism-invariance-node-drop-rate",
            str(args.dual_mechanism_invariance_node_drop_rate),
            "--dual-mechanism-invariance-max-hops",
            str(args.dual_mechanism_invariance_max_hops),
            "--dual-mechanism-invariance-skeleton-scale",
            str(args.dual_mechanism_invariance_skeleton_scale),
            "--dual-mechanism-invariance-min-context-degree",
            str(args.dual_mechanism_invariance_min_context_degree),
            "--unsup-energy-head-mode",
            str(args.unsup_energy_head_mode),
            "--unsup-energy-head-free-energy-temp",
            str(args.unsup_energy_head_free_energy_temp),
            "--unsup-energy-head-free-energy-prior",
            str(args.unsup_energy_head_free_energy_prior),
            "--unsup-energy-head-hidden-dim",
            str(args.unsup_energy_head_hidden_dim),
            "--unsup-energy-head-epochs",
            str(args.unsup_energy_head_epochs),
            "--unsup-energy-head-batch-size",
            str(args.unsup_energy_head_batch_size),
            "--unsup-energy-head-lr",
            str(args.unsup_energy_head_lr),
            "--unsup-energy-head-weight-decay",
            str(args.unsup_energy_head_weight_decay),
            "--unsup-energy-head-noise-std",
            str(args.unsup_energy_head_noise_std),
            "--unsup-energy-head-dropout",
            str(args.unsup_energy_head_dropout),
            "--unsup-energy-head-out",
            str(args.unsup_energy_head_out),
            "--unsup-energy-head-loss-mode",
            str(args.unsup_energy_head_loss_mode),
            "--unsup-energy-head-pos-quantile",
            str(args.unsup_energy_head_pos_quantile),
            "--unsup-energy-head-neg-quantile",
            str(args.unsup_energy_head_neg_quantile),
            "--unsup-energy-head-rank-margin",
            str(args.unsup_energy_head_rank_margin),
            "--unsup-energy-head-rank-weight",
            str(args.unsup_energy_head_rank_weight),
            "--unsup-energy-head-calib-weight",
            str(args.unsup_energy_head_calib_weight),
            "--unsup-energy-head-anchor-cls-weight",
            str(args.unsup_energy_head_anchor_cls_weight),
            "--graph-report-out",
            str(args.graph_report_out),
            "--unsup-mechanism-risk-weight",
            str(args.unsup_mechanism_risk_weight),
            "--unsup-mechanism-top-quantile",
            str(args.unsup_mechanism_top_quantile),
            "--unsup-mechanism-semantic-min-pairs",
            str(args.unsup_mechanism_semantic_min_pairs),
            "--unsup-mechanism-semantic-max-hops",
            str(args.unsup_mechanism_semantic_max_hops),
            "--unsup-family-aware-weight",
            str(args.unsup_family_aware_weight),
            "--unsup-family-top-quantile",
            str(args.unsup_family_top_quantile),
            "--unsup-family-min-samples",
            str(args.unsup_family_min_samples),
            "--unsup-family-boundary-quantile",
            str(args.unsup_family_boundary_quantile),
            "--unsup-family-shrinkage-tau",
            str(args.unsup_family_shrinkage_tau),
            "--unsup-family-ramp-width",
            str(args.unsup_family_ramp_width),
            "--unsup-family-global-mix-floor",
            str(args.unsup_family_global_mix_floor),
            "--unsup-family-radius-min-ratio",
            str(args.unsup_family_radius_min_ratio),
            "--unsup-family-radius-max-ratio",
            str(args.unsup_family_radius_max_ratio),
            "--unsup-adaptive-boundary-focus-quantile",
            str(args.unsup_adaptive_boundary_focus_quantile),
            "--unsup-adaptive-boundary-min-scale",
            str(args.unsup_adaptive_boundary_min_scale),
            "--unsup-adaptive-boundary-max-scale",
            str(args.unsup_adaptive_boundary_max_scale),
            "--unsup-adaptive-boundary-outside-scale",
            str(args.unsup_adaptive_boundary_outside_scale),
            "--risk-rerank-mode",
            str(args.risk_rerank_mode),
            "--train-list",
            args.train_list,
            "--curated-output-list",
            args.curated_output_list,
            "--curated-output-report",
            args.curated_output_report,
            "--curated-max-per-bridge",
            str(args.curated_max_per_bridge),
            "--curated-target-size",
            str(args.curated_target_size),
            "--curated-exclude-bridges",
            args.curated_exclude_bridges,
            "--manual-scope",
            args.manual_scope,
            "--manual-max-files",
            str(args.manual_max_files),
            "--manual-mechanism-risk-weight",
            str(args.manual_mechanism_risk_weight),
            "--manual-mechanism-top-quantile",
            str(args.manual_mechanism_top_quantile),
            "--manual-mechanism-semantic-min-pairs",
            str(args.manual_mechanism_semantic_min_pairs),
            "--manual-mechanism-semantic-max-hops",
            str(args.manual_mechanism_semantic_max_hops),
            "--manual-family-aware-weight",
            str(args.manual_family_aware_weight),
            "--manual-family-top-quantile",
            str(args.manual_family_top_quantile),
            "--manual-family-min-samples",
            str(args.manual_family_min_samples),
            "--manual-family-boundary-quantile",
            str(args.manual_family_boundary_quantile),
            "--manual-family-shrinkage-tau",
            str(args.manual_family_shrinkage_tau),
            "--manual-family-ramp-width",
            str(args.manual_family_ramp_width),
            "--manual-family-global-mix-floor",
            str(args.manual_family_global_mix_floor),
            "--manual-family-radius-min-ratio",
            str(args.manual_family_radius_min_ratio),
            "--manual-family-radius-max-ratio",
            str(args.manual_family_radius_max_ratio),
            "--manual-adaptive-boundary-focus-quantile",
            str(args.manual_adaptive_boundary_focus_quantile),
            "--manual-adaptive-boundary-min-scale",
            str(args.manual_adaptive_boundary_min_scale),
            "--manual-adaptive-boundary-max-scale",
            str(args.manual_adaptive_boundary_max_scale),
            "--manual-adaptive-boundary-outside-scale",
            str(args.manual_adaptive_boundary_outside_scale),
            "--manual-line-risk-closure-boost",
            str(args.manual_line_risk_closure_boost),
            "--manual-line-risk-stage2-weight",
            str(args.manual_line_risk_stage2_weight),
            "--manual-line-risk-stage2-top-functions",
            str(args.manual_line_risk_stage2_top_functions),
            "--manual-line-risk-stage2-support-weight",
            str(args.manual_line_risk_stage2_support_weight),
            "--manual-line-risk-stage2-max-file-share",
            str(args.manual_line_risk_stage2_max_file_share),
            "--manual-score-head-method",
            str(args.manual_score_head_method),
            "--manual-score-head-fit-scope",
            str(args.manual_score_head_fit_scope),
            "--manual-binary-head-method",
            str(args.manual_binary_head_method),
            "--manual-binary-head-fit-scope",
            str(args.manual_binary_head_fit_scope),
            "--manual-binary-head-contamination",
            str(args.manual_binary_head_contamination),
            "--manual-energy-head-ckpt",
            str(args.manual_energy_head_ckpt),
            "--manual-triage-low-quantile",
            str(args.manual_triage_low_quantile),
            "--manual-triage-high-quantile",
            str(args.manual_triage_high_quantile),
            "--manual-triage-gray-threshold",
            str(args.manual_triage_gray_threshold),
            "--manual-triage-family-min-samples",
            str(args.manual_triage_family_min_samples),
            "--manual-triage-adaptive-gray-source",
            str(args.manual_triage_adaptive_gray_source),
            "--manual-triage-adaptive-gray-column",
            str(args.manual_triage_adaptive_gray_column),
            "--manual-triage-adaptive-gray-group-by",
            str(args.manual_triage_adaptive_gray_group_by),
            "--manual-binary-target-mode",
            str(args.manual_binary_target_mode),
            "--manual-binary-target-window",
            str(args.manual_binary_target_window),
            "--manual-binary-eval-topn",
            str(args.manual_binary_eval_topn),
            "--slice-hops",
            str(args.slice_hops),
            "--slice-forward-hops",
            str(args.slice_forward_hops),
            "--slice-backward-hops",
            str(args.slice_backward_hops),
            "--slice-mechanism-path-max-len",
            str(args.slice_mechanism_path_max_len),
            "--slice-min-nodes",
            str(args.slice_min_nodes),
            "--slice-function-fallback-mode",
            args.slice_function_fallback_mode,
            "--slice-function-fallback-min-nodes",
            str(args.slice_function_fallback_min_nodes),
            "--slice-dedup-line-jaccard",
            str(args.slice_dedup_line_jaccard),
            "--slice-max-slices-per-function",
            str(args.slice_max_slices_per_function),
            "--slice-stats-out",
            str(args.slice_stats_out),
            "--dual-skeleton-mode",
            args.dual_skeleton_mode,
            "--dual-mechanism-path-max-len",
            str(args.dual_mechanism_path_max_len),
            "--dual-dynamic-aux-keep-prob",
            str(args.dual_dynamic_aux_keep_prob),
            "--dual-dynamic-seed",
            str(args.dual_dynamic_seed),
            "--manual-slice-hops",
            str(args.manual_slice_hops),
            "--manual-slice-forward-hops",
            str(args.manual_slice_forward_hops),
            "--manual-slice-backward-hops",
            str(args.manual_slice_backward_hops),
            "--manual-slice-mechanism-path-max-len",
            str(args.manual_slice_mechanism_path_max_len),
            "--manual-min-slice-nodes",
            str(args.manual_min_slice_nodes),
            "--manual-slice-function-fallback-mode",
            args.manual_slice_function_fallback_mode,
            "--manual-slice-function-fallback-min-nodes",
            str(args.manual_slice_function_fallback_min_nodes),
            "--manual-slice-dedup-line-jaccard",
            str(args.manual_slice_dedup_line_jaccard),
            "--manual-slice-max-slices-per-function",
            str(args.manual_slice_max_slices_per_function),
            "--manual-slice-stats-out",
            str(args.manual_slice_stats_out),
            "--manual-dual-skeleton-mode",
            args.manual_dual_skeleton_mode,
            "--manual-dual-mechanism-path-max-len",
            str(args.manual_dual_mechanism_path_max_len),
            "--manual-dual-dynamic-aux-keep-prob",
            str(args.manual_dual_dynamic_aux_keep_prob),
            "--manual-dual-dynamic-seed",
            str(args.manual_dual_dynamic_seed),
            "--run-manual-eval",
            "--generalization-topk",
            str(args.generalization_topk),
            "--generalization-line-window",
            str(args.generalization_line_window),
            "--generalization-out",
            gen_out_rel,
            "--summary-out",
            summary_rel,
        ]
        if bool(args.manual_triage_family_aware_gray):
            cmd.append("--manual-triage-family-aware-gray")
        else:
            cmd.append("--no-manual-triage-family-aware-gray")
        if bool(args.manual_triage_adaptive_gray):
            cmd.append("--manual-triage-adaptive-gray")
        else:
            cmd.append("--no-manual-triage-adaptive-gray")
        if args.unsup_energy_head_enable:
            cmd.append("--unsup-energy-head-enable")
        else:
            cmd.append("--no-unsup-energy-head-enable")
        if args.dual_mechanism_invariance_enable:
            cmd.append("--dual-mechanism-invariance-enable")
        else:
            cmd.append("--no-dual-mechanism-invariance-enable")
        if args.unsup_energy_head_stable_train:
            cmd.append("--unsup-energy-head-stable-train")
        else:
            cmd.append("--no-unsup-energy-head-stable-train")
        if args.prebuilt_graph_dir:
            cmd.extend(["--prebuilt-graph-dir", args.prebuilt_graph_dir])
        if args.implicit_mechanism_edges:
            cmd.append("--implicit-mechanism-edges")
        if args.build_curated_standard:
            cmd.append("--build-curated-standard")
        if args.curated_drop_common_libs:
            cmd.append("--curated-drop-common-libs")
        if args.run_baseline:
            cmd.append("--run-baseline")
        if args.run_ablation:
            cmd.append("--run-ablation")
        if args.run_generalization_split:
            cmd.append("--run-generalization-split")
        if args.slice_include_contains:
            cmd.append("--slice-include-contains")
        if args.slice_formal_propagation:
            cmd.append("--slice-formal-propagation")
        if args.unsup_mechanism_semantic_gate:
            cmd.append("--unsup-mechanism-semantic-gate")
        if args.unsup_adaptive_boundary_refine:
            cmd.append("--unsup-adaptive-boundary-refine")
        if args.manual_slice_include_contains:
            cmd.append("--manual-slice-include-contains")
        if args.manual_slice_formal_propagation:
            cmd.append("--manual-slice-formal-propagation")
        if args.manual_mechanism_semantic_gate:
            cmd.append("--manual-mechanism-semantic-gate")
        if args.manual_adaptive_boundary_refine:
            cmd.append("--manual-adaptive-boundary-refine")

        print("[run_multiseed_pipeline] RUN:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT), check=True)

        # Capture per-seed artifacts before next run overwrites default outputs.
        manual_summary = ROOT / "outputs/results/manual_eval_summary.json"
        dual_summary = ROOT / "outputs/results/dual_view_summary.json"
        unsup_summary = ROOT / "outputs/results/unsup_final_summary.json"
        seed_manual = artifact_dir / f"manual_eval_summary_seed{seed}.json"
        seed_dual = artifact_dir / f"dual_view_summary_seed{seed}.json"
        seed_unsup = artifact_dir / f"unsup_final_summary_seed{seed}.json"
        shutil.copy2(manual_summary, seed_manual)
        shutil.copy2(dual_summary, seed_dual)
        shutil.copy2(unsup_summary, seed_unsup)

        m = load_json(seed_manual)
        d = load_json(seed_dual)
        u = load_json(seed_unsup)
        gen_summary_path = (ROOT / gen_out_rel).resolve()

        strict200 = topk_metric(m, topk=200, line_window=0)
        win200 = topk_metric(m, topk=200, line_window=args.line_window)
        rec = {
            "seed": seed,
            "pipeline_summary": str((ROOT / summary_rel).resolve()),
            "manual_eval_summary": str(seed_manual),
            "dual_view_summary": str(seed_dual),
            "unsup_final_summary": str(seed_unsup),
            "top200_strict_row88_recall": float(strict200["row88_recall"]),
            "top200_strict_file_function_recall": float(strict200["file_function_recall"]),
            "top200_window_row88_recall": float(win200["row88_recall"]),
            "top200_window": int(args.line_window),
            "dual_embed_mean_std": float(d["embedding_diagnostics"]["z_joint"]["mean_std"]),
            "unsup_high_risk_count": int(u["status_counts"]["high_risk"]),
            "generalization_split_summary": str(gen_summary_path) if gen_summary_path.exists() else "",
        }
        if args.run_generalization_split:
            rec["top200_filefn_seen_recall"] = 0.0
            rec["top200_filefn_unseen_recall"] = 0.0
            rec["top200_row_seen_recall"] = 0.0
            rec["top200_row_unseen_recall"] = 0.0
        if args.run_generalization_split and gen_summary_path.exists():
            g = load_json(gen_summary_path)
            ff200 = topk_metric_from_list(g.get("file_function_metrics", []), 200)
            rw200 = topk_metric_from_list(g.get("row_metrics", []), 200)
            rec["top200_filefn_seen_recall"] = float(ff200.get("strict_file_function_seen_recall", 0.0))
            rec["top200_filefn_unseen_recall"] = float(ff200.get("strict_file_function_unseen_recall", 0.0))
            rec["top200_row_seen_recall"] = float(rw200.get("window_row_seen_recall", 0.0))
            rec["top200_row_unseen_recall"] = float(rw200.get("window_row_unseen_recall", 0.0))
        runs.append(rec)

    def mean_std(key: str) -> Dict[str, float]:
        arr = np.array([float(r[key]) for r in runs], dtype=np.float32)
        return {"mean": float(arr.mean()), "std": float(arr.std())}

    summary = {
        "seeds": seeds,
        "line_window_for_aggregate": int(args.line_window),
        "runs": runs,
        "aggregate": {
            "top200_strict_row88_recall": mean_std("top200_strict_row88_recall"),
            "top200_strict_file_function_recall": mean_std("top200_strict_file_function_recall"),
            "top200_window_row88_recall": mean_std("top200_window_row88_recall"),
            "dual_embed_mean_std": mean_std("dual_embed_mean_std"),
            "unsup_high_risk_count": mean_std("unsup_high_risk_count"),
            **(
                {
                    "top200_filefn_seen_recall": mean_std("top200_filefn_seen_recall"),
                    "top200_filefn_unseen_recall": mean_std("top200_filefn_unseen_recall"),
                    "top200_row_seen_recall": mean_std("top200_row_seen_recall"),
                    "top200_row_unseen_recall": mean_std("top200_row_unseen_recall"),
                }
                if args.run_generalization_split
                else {}
            ),
        },
        "elapsed_sec_total": float(time.time() - t0),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[run_multiseed_pipeline] summary={out_path}")


if __name__ == "__main__":
    main()
