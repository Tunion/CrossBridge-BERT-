from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-click pipeline runner for unsupervised cross-chain vulnerability detection.")
    p.add_argument("--quick", action="store_true", help="Use lightweight settings for sanity check.")
    p.add_argument("--max-files", type=int, default=0, help="Max training files for normalization stage. 0 means all.")
    p.add_argument("--max-slices", type=int, default=20000, help="Max slices for dual-view training. 0 means all.")
    p.add_argument("--keep-comments", action="store_true", help="Keep comment lines as statements during normalization.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs-baseline", type=int, default=10)
    p.add_argument("--epochs-dual", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dual-global-local-enable", action="store_true")
    p.add_argument("--dual-global-graph-dir", type=str, default="data/graphs/tagged")
    p.add_argument("--dual-lambda-global-align", type=float, default=0.20)
    p.add_argument("--dual-lambda-global-risk", type=float, default=0.20)
    p.add_argument("--dual-mechanism-invariance-enable", dest="dual_mechanism_invariance_enable", action="store_true")
    p.add_argument("--no-dual-mechanism-invariance-enable", dest="dual_mechanism_invariance_enable", action="store_false")
    p.add_argument("--dual-lambda-mechanism-invariance", type=float, default=0.20)
    p.add_argument("--dual-mechanism-invariance-edge-drop-rate", type=float, default=0.12)
    p.add_argument("--dual-mechanism-invariance-node-drop-rate", type=float, default=0.00)
    p.add_argument("--dual-mechanism-invariance-max-hops", type=int, default=4)
    p.add_argument("--dual-mechanism-invariance-skeleton-scale", type=float, default=0.50)
    p.add_argument("--dual-mechanism-invariance-min-context-degree", type=int, default=2)
    p.add_argument("--dual-edge-soft-weight-enable", dest="dual_edge_soft_weight_enable", action="store_true")
    p.add_argument("--no-dual-edge-soft-weight-enable", dest="dual_edge_soft_weight_enable", action="store_false")
    p.add_argument("--dual-edge-weight-local-call-summary", type=float, default=0.35)
    p.add_argument("--dual-edge-weight-shared-object-summary", type=float, default=0.45)
    p.add_argument("--dual-edge-weight-cross-function-state-flow", type=float, default=0.55)
    p.add_argument("--dual-edge-weight-local-call", type=float, default=0.75)
    p.add_argument("--dual-edge-weight-state-summary", type=float, default=0.85)
    p.add_argument("--prototype-k", type=int, default=8)
    p.add_argument(
        "--unsup-embedding-key",
        type=str,
        default="z_joint",
        choices=["z_joint", "z_fused", "z_global", "z_full", "z_skeleton", "z_hybrid"],
    )
    p.add_argument("--unsup-hybrid-alpha", type=float, default=0.50)
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
    p.add_argument("--unsup-mechanism-head-enable", dest="unsup_mechanism_head_enable", action="store_true")
    p.add_argument("--no-unsup-mechanism-head-enable", dest="unsup_mechanism_head_enable", action="store_false")
    p.add_argument("--unsup-mechanism-head-out", type=str, default="outputs/models/mechanism_head.pkl")
    p.add_argument("--unsup-mechanism-head-max-hops", type=int, default=6)
    p.add_argument("--unsup-mechanism-head-gate-quantile", type=float, default=0.55)
    p.add_argument("--unsup-mechanism-head-slice-quantile", type=float, default=0.95)
    p.add_argument("--unsup-mechanism-head-override-final-risk", dest="unsup_mechanism_head_override_final_risk", action="store_true")
    p.add_argument("--no-unsup-mechanism-head-override-final-risk", dest="unsup_mechanism_head_override_final_risk", action="store_false")
    p.add_argument("--unsup-energy-head-loss-mode", type=str, default="proto_margin", choices=["softplus_neg", "proto_margin"])
    p.add_argument("--unsup-energy-head-pos-quantile", type=float, default=0.30)
    p.add_argument("--unsup-energy-head-neg-quantile", type=float, default=0.85)
    p.add_argument("--unsup-energy-head-rank-margin", type=float, default=0.35)
    p.add_argument("--unsup-energy-head-rank-weight", type=float, default=1.0)
    p.add_argument("--unsup-energy-head-calib-weight", type=float, default=0.50)
    p.add_argument("--unsup-energy-head-anchor-cls-weight", type=float, default=0.50)
    p.add_argument("--unsup-energy-head-family-aware-anchor", dest="unsup_energy_head_family_aware_anchor", action="store_true")
    p.add_argument("--no-unsup-energy-head-family-aware-anchor", dest="unsup_energy_head_family_aware_anchor", action="store_false")
    p.add_argument("--unsup-energy-head-anchor-family-min-samples", type=int, default=64)
    p.add_argument("--unsup-energy-head-include-closure-features", dest="unsup_energy_head_include_closure_features", action="store_true")
    p.add_argument("--no-unsup-energy-head-include-closure-features", dest="unsup_energy_head_include_closure_features", action="store_false")
    p.add_argument("--unsup-energy-head-stable-train", dest="unsup_energy_head_stable_train", action="store_true")
    p.add_argument("--no-unsup-energy-head-stable-train", dest="unsup_energy_head_stable_train", action="store_false")
    p.add_argument(
        "--unsup-mode",
        type=str,
        default="offline",
        choices=["offline", "e2e"],
        help="offline: train_dual + train_unsup; e2e: end-to-end multi-prototype SVDD training.",
    )
    p.add_argument("--epochs-e2e", type=int, default=8)
    p.add_argument("--warmup-epochs-e2e", type=int, default=2)
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
    p.add_argument("--risk-rerank-mode", type=str, default="function_coverage", choices=["none", "coverage", "function_coverage", "family_hybrid"])
    p.add_argument("--risk-rerank-topn", type=int, default=2000)
    p.add_argument("--risk-rerank-file-penalty", type=float, default=0.06)
    p.add_argument("--risk-rerank-filefn-penalty", type=float, default=0.12)
    p.add_argument("--risk-rerank-fn-novelty-bonus", type=float, default=0.04)
    p.add_argument("--risk-rerank-fn-overlap-penalty", type=float, default=0.01)
    p.add_argument("--risk-rerank-max-per-filefn", type=int, default=0)
    p.add_argument("--risk-rerank-max-per-file", type=int, default=0)
    p.add_argument("--risk-rerank-file-repeat-power", type=float, default=1.0)
    p.add_argument("--risk-rerank-family-penalty", type=float, default=0.05)
    p.add_argument("--risk-rerank-family-novelty-bonus", type=float, default=0.03)
    p.add_argument("--risk-rerank-aux-weight", type=float, default=0.20)
    p.add_argument("--risk-rerank-max-per-family", type=int, default=0)
    p.add_argument("--risk-rerank-warmup-topk", type=int, default=0)
    p.add_argument("--risk-rerank-warmup-file-cap", type=int, default=3)
    p.add_argument("--risk-rerank-warmup-filefn-cap", type=int, default=1)
    p.add_argument(
        "--train-list",
        type=str,
        default="processed/dataset_non_overlap_local.txt",
        help="Input solidity list for training pipeline normalization stage.",
    )
    p.add_argument(
        "--prebuilt-graph-dir",
        type=str,
        default="",
        help=(
            "Reuse an existing graph directory as Stage-1 output and skip normalize/build_graph. "
            "If empty and data/graphs/full_ast_full_v2 exists, it is auto-selected."
        ),
    )
    p.add_argument("--disable-auto-prebuilt", action="store_true", help="Disable auto fallback to data/graphs/full_ast_full_v2.")
    p.add_argument("--graph-report-out", type=str, default="outputs/results/graph_semantic_coverage.json")
    p.add_argument("--implicit-mechanism-edges", dest="implicit_mechanism_edges", action="store_true")
    p.add_argument("--no-implicit-mechanism-edges", dest="implicit_mechanism_edges", action="store_false")
    p.add_argument("--build-curated-standard", dest="build_curated_standard", action="store_true", help="Build and use curated standard training list.")
    p.add_argument("--no-build-curated-standard", dest="build_curated_standard", action="store_false", help="Disable curated standard training list build.")
    p.add_argument("--curated-output-list", type=str, default="processed/dataset_non_overlap_curated_standard.txt")
    p.add_argument("--curated-output-report", type=str, default="processed/dataset_non_overlap_curated_standard_report.json")
    p.add_argument("--curated-max-per-bridge", type=int, default=120)
    p.add_argument("--curated-target-size", type=int, default=2500)
    p.add_argument(
        "--curated-bridge-whitelist",
        type=str,
        default="",
        help="Pipe-separated bridge whitelist for curated list. Empty keeps all bridge groups.",
    )
    p.add_argument(
        "--curated-exclude-bridges",
        type=str,
        default="__NONE__",
        help="Pipe-separated bridge keywords to exclude from curated training set.",
    )
    p.add_argument("--curated-drop-common-libs", dest="curated_drop_common_libs", action="store_true")
    p.add_argument("--no-curated-drop-common-libs", dest="curated_drop_common_libs", action="store_false")
    p.add_argument("--run-baseline", action="store_true", help="Run Stage-1 baseline training.")
    p.add_argument("--run-ablation", action="store_true", help="Run full/skeleton/dual ablation after dual training.")
    p.add_argument("--run-manual-eval", dest="run_manual_eval", action="store_true", help="Run formal manual-set evaluation after unsup scoring.")
    p.add_argument("--no-run-manual-eval", dest="run_manual_eval", action="store_false", help="Skip formal manual-set evaluation after unsup scoring.")
    p.add_argument("--run-generalization-split", action="store_true", help="Run seen-vs-unseen family split evaluation after manual eval.")
    p.add_argument("--generalization-topk", type=str, default="20,50,100,200")
    p.add_argument("--generalization-line-window", type=int, default=2)
    p.add_argument("--generalization-out", type=str, default="outputs/results/generalization_split_summary.json")
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
    p.add_argument("--manual-binary-head-method", type=str, default="mechanism_head", choices=["none", "ocsvm", "energy_head", "mechanism_head"])
    p.add_argument("--manual-binary-head-fit-scope", type=str, default="all", choices=["all", "low_risk", "normal"])
    p.add_argument("--manual-binary-head-contamination", type=float, default=0.08)
    p.add_argument("--manual-energy-head-ckpt", type=str, default="outputs/models/energy_head.pt")
    p.add_argument("--manual-mechanism-head-ckpt", type=str, default="outputs/models/mechanism_head.pkl")
    p.add_argument("--manual-mechanism-head-override-final-risk", dest="manual_mechanism_head_override_final_risk", action="store_true")
    p.add_argument("--no-manual-mechanism-head-override-final-risk", dest="manual_mechanism_head_override_final_risk", action="store_false")
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
    p.add_argument("--manual-proto-resp-gray-adjust-enable", dest="manual_proto_resp_gray_adjust_enable", action="store_true")
    p.add_argument("--no-manual-proto-resp-gray-adjust-enable", dest="manual_proto_resp_gray_adjust_enable", action="store_false")
    p.add_argument("--manual-proto-resp-gray-adjust-mode", type=str, default="negent", choices=["negent", "respmax", "mix", "sharp"])
    p.add_argument("--manual-proto-resp-gray-adjust-beta", type=float, default=0.04)
    p.add_argument("--manual-proto-resp-gray-adjust-gated", dest="manual_proto_resp_gray_adjust_gated", action="store_true")
    p.add_argument("--no-manual-proto-resp-gray-adjust-gated", dest="manual_proto_resp_gray_adjust_gated", action="store_false")
    p.add_argument("--manual-proto-resp-gray-adjust-gate-center", type=float, default=0.50)
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
    p.add_argument("--slice-cross-function-propagation-mode", choices=["all", "same_function", "anchor_function", "typed_mechanism"], default="all")
    p.add_argument("--slice-cross-function-context-mode", choices=["all", "same_function", "none"], default="all")
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
    p.add_argument("--manual-slice-cross-function-propagation-mode", choices=["all", "same_function", "anchor_function", "typed_mechanism"], default="all")
    p.add_argument("--manual-slice-cross-function-context-mode", choices=["all", "same_function", "none"], default="all")
    p.add_argument("--manual-slice-function-fallback-mode", choices=["none", "no-role", "all"], default="none")
    p.add_argument("--manual-slice-function-fallback-min-nodes", type=int, default=3)
    p.add_argument("--manual-slice-dedup-line-jaccard", type=float, default=0.95)
    p.add_argument("--manual-slice-max-slices-per-function", type=int, default=120)
    p.add_argument("--manual-slice-stats-out", type=str, default="outputs/results/manual_state_slice_stats.json")
    p.add_argument("--manual-dual-skeleton-mode", choices=["legacy", "formal", "dynamic"], default="legacy")
    p.add_argument("--manual-dual-mechanism-path-max-len", type=int, default=4)
    p.add_argument("--manual-dual-dynamic-aux-keep-prob", type=float, default=0.35)
    p.add_argument("--manual-dual-dynamic-seed", type=int, default=42)
    p.add_argument("--summary-out", type=str, default="outputs/results/pipeline_run_summary.json")
    p.set_defaults(
        build_curated_standard=True,
        curated_drop_common_libs=True,
        run_manual_eval=True,
        slice_formal_propagation=True,
        slice_include_contains=True,
        manual_slice_formal_propagation=True,
        manual_slice_include_contains=False,
        implicit_mechanism_edges=False,
        unsup_adaptive_boundary_refine=True,
        manual_adaptive_boundary_refine=True,
        unsup_energy_head_enable=False,
        unsup_mechanism_head_enable=True,
        unsup_mechanism_head_override_final_risk=True,
        unsup_energy_head_stable_train=True,
        unsup_energy_head_include_closure_features=False,
        unsup_energy_head_family_aware_anchor=False,
        dual_mechanism_invariance_enable=False,
        dual_edge_soft_weight_enable=False,
        manual_mechanism_head_override_final_risk=True,
        manual_triage_adaptive_gray=False,
        manual_proto_resp_gray_adjust_enable=False,
        manual_proto_resp_gray_adjust_gated=True,
    )
    return p.parse_args()


def run_cmd(cmd: Sequence[str], cwd: Path, stage: str, logs: List[Dict[str, object]]) -> None:
    print(f"[run_pipeline] [{stage}] RUN: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, cwd=str(cwd), check=True)
    dt = time.time() - t0
    logs.append({"stage": stage, "command": " ".join(cmd), "elapsed_sec": dt})


def _resolve_input_path(root: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (root / p).resolve()


def clean_path(path: Path) -> None:
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    py = sys.executable
    logs: List[Dict[str, object]] = []
    t_start = time.time()

    if args.quick:
        if args.max_files <= 0:
            args.max_files = 200
        if args.max_slices <= 0:
            args.max_slices = 1000
        if args.epochs_baseline <= 0:
            args.epochs_baseline = 3
        if args.epochs_dual <= 0:
            args.epochs_dual = 3
        if args.epochs_e2e <= 0:
            args.epochs_e2e = 3
        if args.warmup_epochs_e2e < 0:
            args.warmup_epochs_e2e = 1

    if not args.prebuilt_graph_dir and not args.disable_auto_prebuilt:
        default_prebuilt = ROOT / "data/graphs/full_ast_full_v2"
        if default_prebuilt.exists():
            args.prebuilt_graph_dir = "data/graphs/full_ast_full_v2"

    train_list = args.train_list
    use_prebuilt_graph = bool(args.prebuilt_graph_dir)
    graph_dir_for_pipeline = args.prebuilt_graph_dir if use_prebuilt_graph else "data/graphs/full"

    if args.build_curated_standard and not use_prebuilt_graph:
        curated_cmd = [
            py,
            "tools/build_curated_training_list.py",
            "--project-root",
            ".",
            "--input-list",
            args.train_list,
            "--output-list",
            args.curated_output_list,
            "--output-report",
            args.curated_output_report,
            "--max-per-bridge",
            str(args.curated_max_per_bridge),
            "--target-size",
            str(args.curated_target_size),
            "--bridge-whitelist",
            args.curated_bridge_whitelist,
            "--exclude-bridge-keywords",
            args.curated_exclude_bridges,
        ]
        if args.curated_drop_common_libs:
            curated_cmd.append("--drop-common-lib-filenames")
        run_cmd(
            curated_cmd,
            cwd=ROOT,
            stage="build_curated_standard",
            logs=logs,
        )
        train_list = args.curated_output_list
    elif args.build_curated_standard and use_prebuilt_graph:
        logs.append(
            {
                "stage": "build_curated_standard_skipped_prebuilt_graph",
                "command": "",
                "elapsed_sec": 0.0,
            }
        )

    # Clean stage artifacts to avoid mixing with previous runs.
    clean_targets = [ROOT / "data/graphs/tagged", ROOT / "data/slices/slices.jsonl", ROOT / "data/slices/dual_views.jsonl"]
    if not use_prebuilt_graph:
        clean_targets.extend(
            [
                ROOT / "data/processed/contracts_normalized.jsonl",
                ROOT / "data/graphs/full",
                ROOT / "data/graphs/graph_index.jsonl",
            ]
        )
    for p in clean_targets:
        clean_path(p)

    # Stage-1: normalize + hetero graph (or reuse prebuilt graph).
    if use_prebuilt_graph:
        prebuilt_dir = _resolve_input_path(ROOT, args.prebuilt_graph_dir)
        if not prebuilt_dir.exists():
            raise FileNotFoundError(f"prebuilt graph dir not found: {prebuilt_dir}")
        logs.append(
            {
                "stage": "use_prebuilt_graph",
                "command": str(prebuilt_dir),
                "elapsed_sec": 0.0,
            }
        )
    else:
        normalize_cmd = [
            py,
            "parser/normalize_or_expand.py",
            "--project-root",
            ".",
            "--dataset-root",
            "DataSet",
            "--input-list",
            train_list,
            "--output",
            "data/processed/contracts_normalized.jsonl",
            "--max-files",
            str(args.max_files),
            "--expand-modifiers",
        ]
        if args.keep_comments:
            normalize_cmd.append("--keep-comments")
        run_cmd(
            normalize_cmd,
            cwd=ROOT,
            stage="normalize",
            logs=logs,
        )
        run_cmd(
            [
                py,
                "parser/build_hetero_graph.py",
                "--input",
                "data/processed/contracts_normalized.jsonl",
                "--graph-dir",
                "data/graphs/full",
                "--index-out",
                "data/graphs/graph_index.jsonl",
                "--report-out",
                str(args.graph_report_out),
                *(["--implicit-mechanism-edges"] if args.implicit_mechanism_edges else []),
            ],
            cwd=ROOT,
            stage="build_graph",
            logs=logs,
        )

    # Stage-2: role tag + slicing.
    run_cmd(
        [
            py,
            "slicing/role_tagging.py",
            "--graph-dir",
            graph_dir_for_pipeline,
            "--output-dir",
            "data/graphs/tagged",
        ],
        cwd=ROOT,
        stage="role_tagging",
        logs=logs,
    )
    state_slice_cmd = [
        py,
        "slicing/state_influence_slice.py",
        "--graph-dir",
        "data/graphs/tagged",
        "--output",
        "data/slices/slices.jsonl",
        "--hops",
        str(args.slice_hops),
        "--forward-hops",
        str(args.slice_forward_hops),
        "--backward-hops",
        str(args.slice_backward_hops),
        "--min-slice-nodes",
        str(args.slice_min_nodes),
        "--mechanism-path-max-len",
        str(args.slice_mechanism_path_max_len),
        "--cross-function-propagation-mode",
        args.slice_cross_function_propagation_mode,
        "--cross-function-context-mode",
        args.slice_cross_function_context_mode,
        "--function-fallback-mode",
        args.slice_function_fallback_mode,
        "--function-fallback-min-nodes",
        str(args.slice_function_fallback_min_nodes),
        "--dedup-line-jaccard",
        str(args.slice_dedup_line_jaccard),
        "--max-slices-per-function",
        str(args.slice_max_slices_per_function),
    ]
    if args.slice_stats_out:
        state_slice_cmd.extend(["--stats-out", args.slice_stats_out])
    if args.slice_include_contains:
        state_slice_cmd.append("--include-contains")
    if args.slice_formal_propagation:
        state_slice_cmd.append("--formal-propagation")
    run_cmd(
        state_slice_cmd,
        cwd=ROOT,
        stage="state_slice",
        logs=logs,
    )

    # Stage-3: dual view.
    run_cmd(
        [
            py,
            "slicing/dual_view_builder.py",
            "--input",
            "data/slices/slices.jsonl",
            "--output",
            "data/slices/dual_views.jsonl",
            "--max-slices",
            str(args.max_slices),
            "--skeleton-mode",
            args.dual_skeleton_mode,
            "--mechanism-path-max-len",
            str(args.dual_mechanism_path_max_len),
            "--dynamic-aux-keep-prob",
            str(args.dual_dynamic_aux_keep_prob),
            "--dynamic-seed",
            str(args.dual_dynamic_seed),
        ],
        cwd=ROOT,
        stage="dual_view_build",
        logs=logs,
    )

    if args.run_baseline:
        run_cmd(
            [
                py,
                "trainers/train_baseline.py",
                "--graph-dir",
                graph_dir_for_pipeline,
                "--max-graphs",
                str(args.max_files if args.max_files > 0 else 0),
                "--epochs",
                str(args.epochs_baseline),
                "--seed",
                str(args.seed),
                "--hidden-dim",
                str(args.hidden_dim),
                "--num-layers",
                str(args.num_layers),
                "--device",
                args.device,
                "--model-out",
                "outputs/models/baseline_single_view.pt",
                "--risk-out",
                "outputs/results/baseline_risk_scores.csv",
                "--summary-out",
                "outputs/results/baseline_summary.json",
            ],
            cwd=ROOT,
            stage="train_baseline",
            logs=logs,
        )

    # Stage-3 training dual-view.
    if args.unsup_mode == "offline" or args.run_ablation:
        dual_cmd = [
                py,
                "trainers/train_dual_view.py",
                "--input",
                "data/slices/dual_views.jsonl",
                "--mode",
                "dual-view",
                "--max-slices",
                str(args.max_slices),
                "--epochs",
                str(args.epochs_dual),
                "--seed",
                str(args.seed),
                "--hidden-dim",
                str(args.hidden_dim),
                "--num-layers",
                str(args.num_layers),
                "--device",
                args.device,
                "--model-out",
                "outputs/models/dual_view.pt",
                "--risk-out",
                "outputs/results/dual_view_risk_scores.csv",
                "--summary-out",
                "outputs/results/dual_view_summary.json",
                "--embedding-out",
                "outputs/results/dual_view_embeddings.npz",
                "--global-graph-dir",
                str(args.dual_global_graph_dir),
                "--lambda-global-align",
                str(args.dual_lambda_global_align),
                "--lambda-global-risk",
                str(args.dual_lambda_global_risk),
                "--lambda-mechanism-invariance",
                str(args.dual_lambda_mechanism_invariance),
                "--mechanism-invariance-edge-drop-rate",
                str(args.dual_mechanism_invariance_edge_drop_rate),
                "--mechanism-invariance-node-drop-rate",
                str(args.dual_mechanism_invariance_node_drop_rate),
                "--mechanism-invariance-max-hops",
                str(args.dual_mechanism_invariance_max_hops),
                "--mechanism-invariance-skeleton-scale",
                str(args.dual_mechanism_invariance_skeleton_scale),
                "--mechanism-invariance-min-context-degree",
                str(args.dual_mechanism_invariance_min_context_degree),
                "--edge-weight-local-call-summary",
                str(args.dual_edge_weight_local_call_summary),
                "--edge-weight-shared-object-summary",
                str(args.dual_edge_weight_shared_object_summary),
                "--edge-weight-cross-function-state-flow",
                str(args.dual_edge_weight_cross_function_state_flow),
                "--edge-weight-local-call",
                str(args.dual_edge_weight_local_call),
                "--edge-weight-state-summary",
                str(args.dual_edge_weight_state_summary),
        ]
        if args.dual_global_local_enable:
            dual_cmd.append("--global-local-enable")
        if args.dual_mechanism_invariance_enable:
            dual_cmd.append("--mechanism-invariance-enable")
        if args.dual_edge_soft_weight_enable:
            dual_cmd.append("--edge-soft-weight-enable")
        else:
            dual_cmd.append("--no-edge-soft-weight-enable")
        run_cmd(dual_cmd, cwd=ROOT, stage="train_dual", logs=logs)
    else:
        logs.append(
            {
                "stage": "train_dual_skipped_e2e_mode",
                "command": "",
                "elapsed_sec": 0.0,
            }
        )

    if args.run_ablation:
        run_cmd(
            [
                py,
                "trainers/train_dual_view.py",
                "--input",
                "data/slices/dual_views.jsonl",
                "--mode",
                "full-only",
                "--max-slices",
                str(args.max_slices),
                "--epochs",
                str(max(2, args.epochs_dual // 2)),
                "--seed",
                str(args.seed),
                "--hidden-dim",
                str(args.hidden_dim),
                "--num-layers",
                str(args.num_layers),
                "--device",
                args.device,
                "--model-out",
                "outputs/models/full_only.pt",
                "--risk-out",
                "outputs/results/full_only_risk.csv",
                "--summary-out",
                "outputs/results/full_only_summary.json",
                "--embedding-out",
                "outputs/results/full_only_emb.npz",
            ],
            cwd=ROOT,
            stage="train_full_only",
            logs=logs,
        )
        run_cmd(
            [
                py,
                "trainers/train_dual_view.py",
                "--input",
                "data/slices/dual_views.jsonl",
                "--mode",
                "skeleton-only",
                "--max-slices",
                str(args.max_slices),
                "--epochs",
                str(max(2, args.epochs_dual // 2)),
                "--seed",
                str(args.seed),
                "--hidden-dim",
                str(args.hidden_dim),
                "--num-layers",
                str(args.num_layers),
                "--device",
                args.device,
                "--model-out",
                "outputs/models/skeleton_only.pt",
                "--risk-out",
                "outputs/results/skeleton_only_risk.csv",
                "--summary-out",
                "outputs/results/skeleton_only_summary.json",
                "--embedding-out",
                "outputs/results/skeleton_only_emb.npz",
            ],
            cwd=ROOT,
            stage="train_skeleton_only",
            logs=logs,
        )
        run_cmd(
            [
                py,
                "eval/ablation.py",
                "--full",
                "outputs/results/full_only_risk.csv",
                "--skeleton",
                "outputs/results/skeleton_only_risk.csv",
                "--dual",
                "outputs/results/dual_view_risk_scores.csv",
                "--top-k",
                "100",
                "--out",
                "outputs/results/ablation_summary.json",
            ],
            cwd=ROOT,
            stage="ablation_eval",
            logs=logs,
        )

    # Stage-4/5: prototype + boundary refine.
    risk_rerank_mode_for_train = str(args.risk_rerank_mode)
    if args.unsup_mode == "e2e" and risk_rerank_mode_for_train == "family_hybrid":
        risk_rerank_mode_for_train = "function_coverage"
        logs.append(
            {
                "stage": "risk_rerank_mode_downgrade_for_e2e",
                "command": "family_hybrid -> function_coverage",
                "elapsed_sec": 0.0,
            }
        )

    if args.unsup_mode == "offline":
        unsup_cmd = [
            py,
            "trainers/train_unsup.py",
            "--dual-view-input",
            "data/slices/dual_views.jsonl",
            "--embedding-npz",
            "outputs/results/dual_view_embeddings.npz",
            "--embedding-key",
            str(args.unsup_embedding_key),
            "--hybrid-alpha",
            str(args.unsup_hybrid_alpha),
            "--prototype-k",
            str(args.prototype_k),
            "--mechanism-risk-weight",
            str(args.unsup_mechanism_risk_weight),
            "--mechanism-top-quantile",
            str(args.unsup_mechanism_top_quantile),
            "--mechanism-semantic-min-pairs",
            str(args.unsup_mechanism_semantic_min_pairs),
            "--mechanism-semantic-max-hops",
            str(args.unsup_mechanism_semantic_max_hops),
            "--family-aware-weight",
            str(args.unsup_family_aware_weight),
            "--family-top-quantile",
            str(args.unsup_family_top_quantile),
            "--family-min-samples",
            str(args.unsup_family_min_samples),
            "--family-boundary-quantile",
            str(args.unsup_family_boundary_quantile),
            "--family-shrinkage-tau",
            str(args.unsup_family_shrinkage_tau),
            "--family-ramp-width",
            str(args.unsup_family_ramp_width),
            "--family-global-mix-floor",
            str(args.unsup_family_global_mix_floor),
            "--family-radius-min-ratio",
            str(args.unsup_family_radius_min_ratio),
            "--family-radius-max-ratio",
            str(args.unsup_family_radius_max_ratio),
            "--adaptive-boundary-focus-quantile",
            str(args.unsup_adaptive_boundary_focus_quantile),
            "--adaptive-boundary-min-scale",
            str(args.unsup_adaptive_boundary_min_scale),
            "--adaptive-boundary-max-scale",
            str(args.unsup_adaptive_boundary_max_scale),
            "--adaptive-boundary-outside-scale",
            str(args.unsup_adaptive_boundary_outside_scale),
            "--risk-rerank-mode",
            risk_rerank_mode_for_train,
            "--risk-rerank-topn",
            str(args.risk_rerank_topn),
            "--risk-rerank-file-penalty",
            str(args.risk_rerank_file_penalty),
            "--risk-rerank-filefn-penalty",
            str(args.risk_rerank_filefn_penalty),
            "--risk-rerank-fn-novelty-bonus",
            str(args.risk_rerank_fn_novelty_bonus),
            "--risk-rerank-fn-overlap-penalty",
            str(args.risk_rerank_fn_overlap_penalty),
            "--risk-rerank-max-per-filefn",
            str(args.risk_rerank_max_per_filefn),
            "--risk-rerank-max-per-file",
            str(args.risk_rerank_max_per_file),
            "--risk-rerank-file-repeat-power",
            str(args.risk_rerank_file_repeat_power),
            "--risk-rerank-family-penalty",
            str(args.risk_rerank_family_penalty),
            "--risk-rerank-family-novelty-bonus",
            str(args.risk_rerank_family_novelty_bonus),
            "--risk-rerank-aux-weight",
            str(args.risk_rerank_aux_weight),
            "--risk-rerank-max-per-family",
            str(args.risk_rerank_max_per_family),
            "--risk-rerank-warmup-topk",
            str(args.risk_rerank_warmup_topk),
            "--risk-rerank-warmup-file-cap",
            str(args.risk_rerank_warmup_file_cap),
            "--risk-rerank-warmup-filefn-cap",
            str(args.risk_rerank_warmup_filefn_cap),
            "--seed",
            str(args.seed),
            "--boundary-quantile",
            "0.9",
            "--risk-out",
            "outputs/results/unsup_final_risk_scores.csv",
            "--summary-out",
            "outputs/results/unsup_final_summary.json",
            "--prototype-out",
            "outputs/models/prototypes.npz",
        ]
        if args.unsup_energy_head_enable:
            unsup_cmd.append("--energy-head-enable")
        else:
            unsup_cmd.append("--no-energy-head-enable")
        unsup_cmd.extend(
            [
                "--energy-head-mode",
                str(args.unsup_energy_head_mode),
                "--energy-head-free-energy-temp",
                str(args.unsup_energy_head_free_energy_temp),
                "--energy-head-free-energy-prior",
                str(args.unsup_energy_head_free_energy_prior),
                "--energy-head-hidden-dim",
                str(args.unsup_energy_head_hidden_dim),
                "--energy-head-epochs",
                str(args.unsup_energy_head_epochs),
                "--energy-head-batch-size",
                str(args.unsup_energy_head_batch_size),
                "--energy-head-lr",
                str(args.unsup_energy_head_lr),
                "--energy-head-weight-decay",
                str(args.unsup_energy_head_weight_decay),
                "--energy-head-noise-std",
                str(args.unsup_energy_head_noise_std),
                "--energy-head-dropout",
                str(args.unsup_energy_head_dropout),
                "--energy-head-out",
                str(args.unsup_energy_head_out),
                "--energy-head-loss-mode",
                str(args.unsup_energy_head_loss_mode),
                "--energy-head-pos-quantile",
                str(args.unsup_energy_head_pos_quantile),
                "--energy-head-neg-quantile",
                str(args.unsup_energy_head_neg_quantile),
                "--energy-head-rank-margin",
                str(args.unsup_energy_head_rank_margin),
                "--energy-head-rank-weight",
                str(args.unsup_energy_head_rank_weight),
                "--energy-head-calib-weight",
                str(args.unsup_energy_head_calib_weight),
                "--energy-head-anchor-cls-weight",
                str(args.unsup_energy_head_anchor_cls_weight),
                "--energy-head-anchor-family-min-samples",
                str(args.unsup_energy_head_anchor_family_min_samples),
            ]
        )
        if args.unsup_energy_head_include_closure_features:
            unsup_cmd.append("--energy-head-include-closure-features")
        else:
            unsup_cmd.append("--no-energy-head-include-closure-features")
        if args.unsup_energy_head_stable_train:
            unsup_cmd.append("--energy-head-stable-train")
        else:
            unsup_cmd.append("--no-energy-head-stable-train")
        if args.unsup_energy_head_family_aware_anchor:
            unsup_cmd.append("--energy-head-family-aware-anchor")
        else:
            unsup_cmd.append("--no-energy-head-family-aware-anchor")
        if args.unsup_mechanism_head_enable:
            unsup_cmd.append("--mechanism-head-enable")
        else:
            unsup_cmd.append("--no-mechanism-head-enable")
        unsup_cmd.extend(
            [
                "--mechanism-head-out",
                str(args.unsup_mechanism_head_out),
                "--mechanism-head-max-hops",
                str(args.unsup_mechanism_head_max_hops),
                "--mechanism-head-gate-quantile",
                str(args.unsup_mechanism_head_gate_quantile),
                "--mechanism-head-slice-quantile",
                str(args.unsup_mechanism_head_slice_quantile),
            ]
        )
        if args.unsup_mechanism_head_override_final_risk:
            unsup_cmd.append("--mechanism-head-override-final-risk")
        else:
            unsup_cmd.append("--no-mechanism-head-override-final-risk")
    else:
        unsup_cmd = [
            py,
            "trainers/train_e2e_unsup.py",
            "--input",
            "data/slices/dual_views.jsonl",
            "--max-slices",
            str(args.max_slices),
            "--epochs",
            str(args.epochs_e2e),
            "--warmup-epochs",
            str(args.warmup_epochs_e2e),
            "--seed",
            str(args.seed),
            "--hidden-dim",
            str(args.hidden_dim),
            "--num-layers",
            str(args.num_layers),
            "--device",
            args.device,
            "--prototype-k",
            str(args.prototype_k),
            "--mechanism-risk-weight",
            str(args.unsup_mechanism_risk_weight),
            "--mechanism-top-quantile",
            str(args.unsup_mechanism_top_quantile),
            "--mechanism-semantic-min-pairs",
            str(args.unsup_mechanism_semantic_min_pairs),
            "--mechanism-semantic-max-hops",
            str(args.unsup_mechanism_semantic_max_hops),
            "--risk-rerank-mode",
            risk_rerank_mode_for_train,
            "--risk-rerank-topn",
            str(args.risk_rerank_topn),
            "--risk-rerank-file-penalty",
            str(args.risk_rerank_file_penalty),
            "--risk-rerank-filefn-penalty",
            str(args.risk_rerank_filefn_penalty),
            "--risk-rerank-fn-novelty-bonus",
            str(args.risk_rerank_fn_novelty_bonus),
            "--risk-rerank-fn-overlap-penalty",
            str(args.risk_rerank_fn_overlap_penalty),
            "--risk-rerank-max-per-filefn",
            str(args.risk_rerank_max_per_filefn),
            "--risk-rerank-max-per-file",
            str(args.risk_rerank_max_per_file),
            "--risk-rerank-file-repeat-power",
            str(args.risk_rerank_file_repeat_power),
            "--risk-rerank-warmup-topk",
            str(args.risk_rerank_warmup_topk),
            "--risk-rerank-warmup-file-cap",
            str(args.risk_rerank_warmup_file_cap),
            "--risk-rerank-warmup-filefn-cap",
            str(args.risk_rerank_warmup_filefn_cap),
            "--model-out",
            "outputs/models/dual_view.pt",
            "--prototype-out",
            "outputs/models/prototypes.npz",
            "--embedding-out",
            "outputs/results/dual_view_embeddings.npz",
            "--risk-out",
            "outputs/results/unsup_final_risk_scores.csv",
            "--summary-out",
            "outputs/results/unsup_final_summary.json",
        ]

    if args.unsup_mechanism_semantic_gate:
        unsup_cmd.append("--mechanism-semantic-gate")
    if args.unsup_adaptive_boundary_refine:
        unsup_cmd.append("--adaptive-boundary-refine")
    run_cmd(
        unsup_cmd,
        cwd=ROOT,
        stage="unsup_refine",
        logs=logs,
    )

    # Metrics with two label modes.
    run_cmd(
        [
            py,
            "eval/metrics.py",
            "--pred",
            "outputs/results/unsup_final_risk_scores.csv",
            "--label-file",
            "processed/label_standard_local.csv",
            "--label-mode",
            "row88",
            "--score-col",
            "final_risk",
            "--top-k",
            "200",
            "--out",
            "outputs/results/metrics_row88.json",
        ],
        cwd=ROOT,
        stage="metrics_row88",
        logs=logs,
    )
    run_cmd(
        [
            py,
            "eval/metrics.py",
            "--pred",
            "outputs/results/unsup_final_risk_scores.csv",
            "--label-file",
            "processed/label_standard_local.csv",
            "--label-mode",
            "file_function",
            "--score-col",
            "final_risk",
            "--top-k",
            "200",
            "--out",
            "outputs/results/metrics_file_function.json",
        ],
        cwd=ROOT,
        stage="metrics_file_function",
        logs=logs,
    )

    if args.run_manual_eval:
        manual_eval_cmd = [
            py,
            "eval/evaluate_manual_set.py",
            "--project-root",
            ".",
            "--manual-root",
            "manually-labeled dataset/Real_attack_dataset_format",
            "--label-csv",
            "processed/label_standard_local.csv",
            "--scope",
            args.manual_scope,
            "--max-files",
            str(args.manual_max_files),
            "--work-dir",
            "outputs/manual_eval_artifacts",
            "--dual-model",
            "outputs/models/dual_view.pt",
            "--prototype-npz",
            "outputs/models/prototypes.npz",
            "--device",
            args.device,
            "--hops",
            str(args.manual_slice_hops),
            "--slice-forward-hops",
            str(args.manual_slice_forward_hops),
            "--slice-backward-hops",
            str(args.manual_slice_backward_hops),
            "--min-slice-nodes",
            str(args.manual_min_slice_nodes),
            "--slice-mechanism-path-max-len",
            str(args.manual_slice_mechanism_path_max_len),
            "--slice-function-fallback-mode",
            args.manual_slice_function_fallback_mode,
            "--slice-function-fallback-min-nodes",
            str(args.manual_slice_function_fallback_min_nodes),
            "--slice-dedup-line-jaccard",
            str(args.manual_slice_dedup_line_jaccard),
            "--slice-max-slices-per-function",
            str(args.manual_slice_max_slices_per_function),
            "--dual-skeleton-mode",
            args.manual_dual_skeleton_mode,
            "--dual-mechanism-path-max-len",
            str(args.manual_dual_mechanism_path_max_len),
            "--dual-dynamic-aux-keep-prob",
            str(args.manual_dual_dynamic_aux_keep_prob),
            "--dual-dynamic-seed",
            str(args.manual_dual_dynamic_seed),
            "--risk-out",
            "outputs/results/manual_eval_risk_scores.csv",
            "--binary-out",
            "outputs/results/manual_eval_binary_predictions.csv",
            "--summary-out",
            "outputs/results/manual_eval_summary.json",
            "--line-risk-closure-boost",
            str(args.manual_line_risk_closure_boost),
            "--line-risk-stage2-weight",
            str(args.manual_line_risk_stage2_weight),
            "--line-risk-stage2-top-functions",
            str(args.manual_line_risk_stage2_top_functions),
            "--line-risk-stage2-support-weight",
            str(args.manual_line_risk_stage2_support_weight),
            "--line-risk-stage2-max-file-share",
            str(args.manual_line_risk_stage2_max_file_share),
            "--mechanism-risk-weight",
            str(args.manual_mechanism_risk_weight),
            "--mechanism-top-quantile",
            str(args.manual_mechanism_top_quantile),
            "--mechanism-semantic-min-pairs",
            str(args.manual_mechanism_semantic_min_pairs),
            "--mechanism-semantic-max-hops",
            str(args.manual_mechanism_semantic_max_hops),
            "--family-aware-weight",
            str(args.manual_family_aware_weight),
            "--family-top-quantile",
            str(args.manual_family_top_quantile),
            "--family-min-samples",
            str(args.manual_family_min_samples),
            "--family-boundary-quantile",
            str(args.manual_family_boundary_quantile),
            "--family-shrinkage-tau",
            str(args.manual_family_shrinkage_tau),
            "--family-ramp-width",
            str(args.manual_family_ramp_width),
            "--family-global-mix-floor",
            str(args.manual_family_global_mix_floor),
            "--family-radius-min-ratio",
            str(args.manual_family_radius_min_ratio),
            "--family-radius-max-ratio",
            str(args.manual_family_radius_max_ratio),
            "--adaptive-boundary-focus-quantile",
            str(args.manual_adaptive_boundary_focus_quantile),
            "--adaptive-boundary-min-scale",
            str(args.manual_adaptive_boundary_min_scale),
            "--adaptive-boundary-max-scale",
            str(args.manual_adaptive_boundary_max_scale),
            "--adaptive-boundary-outside-scale",
            str(args.manual_adaptive_boundary_outside_scale),
            "--risk-rerank-mode",
            str(args.risk_rerank_mode),
            "--risk-rerank-topn",
            str(args.risk_rerank_topn),
            "--risk-rerank-file-penalty",
            str(args.risk_rerank_file_penalty),
            "--risk-rerank-filefn-penalty",
            str(args.risk_rerank_filefn_penalty),
            "--risk-rerank-fn-novelty-bonus",
            str(args.risk_rerank_fn_novelty_bonus),
            "--risk-rerank-fn-overlap-penalty",
            str(args.risk_rerank_fn_overlap_penalty),
            "--risk-rerank-max-per-filefn",
            str(args.risk_rerank_max_per_filefn),
            "--risk-rerank-max-per-file",
            str(args.risk_rerank_max_per_file),
            "--risk-rerank-file-repeat-power",
            str(args.risk_rerank_file_repeat_power),
            "--risk-rerank-family-penalty",
            str(args.risk_rerank_family_penalty),
            "--risk-rerank-family-novelty-bonus",
            str(args.risk_rerank_family_novelty_bonus),
            "--risk-rerank-aux-weight",
            str(args.risk_rerank_aux_weight),
            "--risk-rerank-max-per-family",
            str(args.risk_rerank_max_per_family),
            "--risk-rerank-warmup-topk",
            str(args.risk_rerank_warmup_topk),
            "--risk-rerank-warmup-file-cap",
            str(args.risk_rerank_warmup_file_cap),
            "--risk-rerank-warmup-filefn-cap",
            str(args.risk_rerank_warmup_filefn_cap),
            "--score-head-method",
            str(args.manual_score_head_method),
            "--score-head-fit-scope",
            str(args.manual_score_head_fit_scope),
            "--binary-head-method",
            str(args.manual_binary_head_method),
            "--binary-head-fit-scope",
            str(args.manual_binary_head_fit_scope),
            "--binary-head-contamination",
            str(args.manual_binary_head_contamination),
            "--energy-head-ckpt",
            str(args.manual_energy_head_ckpt),
            "--mechanism-head-ckpt",
            str(args.manual_mechanism_head_ckpt),
            "--triage-low-quantile",
            str(args.manual_triage_low_quantile),
            "--triage-high-quantile",
            str(args.manual_triage_high_quantile),
            "--triage-gray-threshold",
            str(args.manual_triage_gray_threshold),
            "--triage-family-min-samples",
            str(args.manual_triage_family_min_samples),
            "--triage-adaptive-gray-source",
            str(args.manual_triage_adaptive_gray_source),
            "--triage-adaptive-gray-column",
            str(args.manual_triage_adaptive_gray_column),
            "--triage-adaptive-gray-group-by",
            str(args.manual_triage_adaptive_gray_group_by),
            "--proto-resp-gray-adjust-mode",
            str(args.manual_proto_resp_gray_adjust_mode),
            "--proto-resp-gray-adjust-beta",
            str(args.manual_proto_resp_gray_adjust_beta),
            "--proto-resp-gray-adjust-gate-center",
            str(args.manual_proto_resp_gray_adjust_gate_center),
            "--binary-target-mode",
            str(args.manual_binary_target_mode),
            "--binary-target-window",
            str(args.manual_binary_target_window),
            "--binary-eval-topn",
            str(args.manual_binary_eval_topn),
        ]
        if args.manual_mechanism_head_override_final_risk:
            manual_cmd.append("--mechanism-head-override-final-risk")
        else:
            manual_cmd.append("--no-mechanism-head-override-final-risk")
        if bool(args.manual_triage_family_aware_gray):
            manual_eval_cmd.append("--triage-family-aware-gray")
        else:
            manual_eval_cmd.append("--no-triage-family-aware-gray")
        if bool(args.manual_triage_adaptive_gray):
            manual_eval_cmd.append("--triage-adaptive-gray")
        else:
            manual_eval_cmd.append("--no-triage-adaptive-gray")
        if bool(args.manual_proto_resp_gray_adjust_enable):
            manual_eval_cmd.append("--proto-resp-gray-adjust-enable")
        else:
            manual_eval_cmd.append("--no-proto-resp-gray-adjust-enable")
        if bool(args.manual_proto_resp_gray_adjust_gated):
            manual_eval_cmd.append("--proto-resp-gray-adjust-gated")
        else:
            manual_eval_cmd.append("--no-proto-resp-gray-adjust-gated")
        if args.manual_slice_stats_out:
            manual_eval_cmd.extend(["--slice-stats-out", args.manual_slice_stats_out])
        if args.manual_slice_cross_function_propagation_mode:
            manual_eval_cmd.extend(
                [
                    "--slice-cross-function-propagation-mode",
                    args.manual_slice_cross_function_propagation_mode,
                    "--slice-cross-function-context-mode",
                    args.manual_slice_cross_function_context_mode,
                ]
            )
        if args.manual_mechanism_semantic_gate:
            manual_eval_cmd.append("--mechanism-semantic-gate")
        if args.manual_adaptive_boundary_refine:
            manual_eval_cmd.append("--adaptive-boundary-refine")
        if args.keep_comments:
            manual_eval_cmd.append("--keep-comments")
        if args.manual_slice_include_contains:
            manual_eval_cmd.append("--slice-include-contains")
        if args.manual_slice_formal_propagation:
            manual_eval_cmd.append("--slice-formal-propagation")
        run_cmd(
            manual_eval_cmd,
            cwd=ROOT,
            stage="manual_eval",
            logs=logs,
        )
        if args.run_generalization_split:
            run_cmd(
                [
                    py,
                    "eval/generalization_split.py",
                    "--risk-csv",
                    "outputs/results/manual_eval_risk_scores.csv",
                    "--label-csv",
                    "processed/label_standard_local.csv",
                    "--train-list",
                    str(train_list),
                    "--topk",
                    str(args.generalization_topk),
                    "--line-window",
                    str(args.generalization_line_window),
                    "--out",
                    str(args.generalization_out),
                ],
                cwd=ROOT,
                stage="generalization_split_eval",
                logs=logs,
            )

    summary = {
        "quick": bool(args.quick),
        "args": vars(args),
        "stages": logs,
        "elapsed_sec_total": float(time.time() - t_start),
    }
    summary_out = (ROOT / args.summary_out).resolve()
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[run_pipeline] summary={summary_out}")


if __name__ == "__main__":
    main()
