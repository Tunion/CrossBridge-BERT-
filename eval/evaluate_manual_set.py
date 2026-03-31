from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.energy_head_utils import (
    align_feature_matrix_columns,
    build_energy_feature_matrix,
    normalize01 as normalize01_energy,
    standardize_transform,
)
from common.closure_feature_utils import CLOSURE_FEATURE_NAMES, aggregate_closure_features, compute_graph_closure_features
from common.group_score_calibration_utils import (
    apply_group_score_calibration,
    fit_group_score_calibration_from_csv,
)
from common.hierarchical_prototype_utils import score_hierarchical_prototypes, unpack_local_bank
from common.graph_utils import EDGE_TYPES, build_soft_edge_weight_config
from common.io_utils import load_jsonl, save_json
from common.mechanism_head_utils import (
    extract_mechanism_head_feature_rows,
    load_mechanism_head_model,
)
from common.metamorphic_feature_utils import (
    compute_metamorphic_instability_features,
    embedding_array_from_map,
)
from common.metamorphic_utils import build_dual_metamorphic_variants
from common.score_head_utils import (
    apply_iforest_score_head,
    build_score_feature_matrix as build_score_feature_matrix_common,
)
from models.boundary_refine import BoundaryRefiner, RefineWeights
from models.dual_view_gnn import DualViewModel, GlobalLocalDualViewModel, dual_view_to_tensors
from models.energy_head import EnergyHeadMLP
from models.single_view_gnn import graph_to_tensor
from trainers.train_unsup import (
    adaptive_mechanism_score,
    blend_family_risk,
    blend_mechanism_risk,
    compute_family_aware_risk,
    compute_mechanism_risk,
    semantic_closed_loop_mask,
    top_quantile_mask,
)


OLD_PREFIX = "/mnt/3.6TB-DATA/zhy/zhy_dir/data/FSE24-SmartAxe-main/FSE24-SmartAxe-main/"
MANUAL_PREFIX = "manually-labeled dataset/Real_attack_dataset_format/"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Formal evaluation on manually-labeled dataset using trained dual-view model and prototype boundaries."
        )
    )
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--manual-root", type=str, default="manually-labeled dataset/Real_attack_dataset_format")
    p.add_argument("--label-csv", type=str, default="processed/label_standard_local.csv")
    p.add_argument("--scope", choices=["label-files", "all-sol"], default="label-files")
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--work-dir", type=str, default="outputs/manual_eval_artifacts")
    p.add_argument(
        "--prebuilt-artifacts-dir",
        type=str,
        default="",
        help="Optional existing manual-eval artifact dir containing graphs_tagged/slices.jsonl/dual_views.jsonl.",
    )
    p.add_argument("--graph-report-out", type=str, default="")
    p.add_argument("--implicit-mechanism-edges", dest="implicit_mechanism_edges", action="store_true")
    p.add_argument("--no-implicit-mechanism-edges", dest="implicit_mechanism_edges", action="store_false")
    p.add_argument("--dual-model", type=str, default="outputs/models/dual_view.pt")
    p.add_argument("--prototype-npz", type=str, default="outputs/models/prototypes.npz")
    p.add_argument(
        "--embedding-key",
        type=str,
        default="auto",
        choices=["auto", "z_joint", "z_graph_joint", "z_text", "z_fused", "z_global", "z_full", "z_skeleton", "z_hybrid"],
        help="Embedding key used during prototype fitting. 'auto' reads it from prototype npz when available.",
    )
    p.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.50,
        help="When embedding-key=z_hybrid, use alpha*z_fused + (1-alpha)*z_joint.",
    )
    p.add_argument("--function-bag-enable", dest="function_bag_enable", action="store_true")
    p.add_argument("--no-function-bag-enable", dest="function_bag_enable", action="store_false")
    p.add_argument(
        "--function-bag-agg",
        type=str,
        default="mean",
        choices=["mean", "max", "meanmaxmix"],
        help="How to aggregate slice embeddings into function-level bag embeddings before prototype fitting.",
    )
    p.add_argument(
        "--function-bag-eval-level",
        type=str,
        default="slice",
        choices=["slice", "function"],
        help="When function-bag is enabled, evaluate on expanded slice rows or direct function-bag rows.",
    )
    p.add_argument(
        "--manual-global-graph-dir",
        type=str,
        default="",
        help="Optional override for manual-eval whole-graph directory when dual model uses global-local fusion.",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--adaptive-boundary-refine", dest="adaptive_boundary_refine", action="store_true")
    p.add_argument("--no-adaptive-boundary-refine", dest="adaptive_boundary_refine", action="store_false")
    p.add_argument("--adaptive-boundary-focus-quantile", type=float, default=0.65)
    p.add_argument("--adaptive-boundary-min-scale", type=float, default=0.85)
    p.add_argument("--adaptive-boundary-max-scale", type=float, default=1.20)
    p.add_argument("--adaptive-boundary-outside-scale", type=float, default=1.00)
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--keep-comments", action="store_true", help="Keep comment lines as statements during normalization.")
    p.add_argument("--min-slice-nodes", type=int, default=6)
    p.add_argument("--slice-forward-hops", type=int, default=0, help="0 means use --hops")
    p.add_argument("--slice-backward-hops", type=int, default=0, help="0 means use --hops")
    p.add_argument("--slice-formal-propagation", dest="slice_formal_propagation", action="store_true")
    p.add_argument("--no-slice-formal-propagation", dest="slice_formal_propagation", action="store_false")
    p.add_argument("--slice-mechanism-path-max-len", type=int, default=4)
    p.add_argument("--slice-include-contains", dest="slice_include_contains", action="store_true")
    p.add_argument("--no-slice-include-contains", dest="slice_include_contains", action="store_false")
    p.add_argument("--slice-cross-function-propagation-mode", choices=["all", "same_function", "anchor_function", "typed_mechanism"], default="all")
    p.add_argument("--slice-cross-function-context-mode", choices=["all", "same_function", "none"], default="all")
    p.add_argument("--slice-function-fallback-mode", choices=["none", "no-role", "all"], default="none")
    p.add_argument("--slice-function-fallback-min-nodes", type=int, default=3)
    p.add_argument("--slice-dedup-line-jaccard", type=float, default=0.95)
    p.add_argument("--slice-max-slices-per-function", type=int, default=120)
    p.add_argument("--slice-stats-out", type=str, default="")
    p.set_defaults(implicit_mechanism_edges=False)
    p.add_argument(
        "--dual-skeleton-mode",
        choices=["legacy", "formal", "dynamic"],
        default="legacy",
        help="Default aligned to the preserved better manual-eval artifact lineage.",
    )
    p.add_argument("--dual-mechanism-path-max-len", type=int, default=4)
    p.add_argument("--dual-dynamic-aux-keep-prob", type=float, default=0.35)
    p.add_argument("--dual-dynamic-seed", type=int, default=42)
    p.add_argument("--topk", type=str, default="20,50,100,200")
    p.add_argument(
        "--mechanism-risk-weight",
        type=float,
        default=0.0,
        help="Blend weight for mechanism-consistency risk in manual-set scoring.",
    )
    p.add_argument(
        "--mechanism-top-quantile",
        type=float,
        default=1.0,
        help="Only apply mechanism correction on samples whose base risk is in top-q quantile. 1.0 disables.",
    )
    p.add_argument(
        "--mechanism-semantic-gate",
        action="store_true",
        help="Apply mechanism correction only on semantic closed-loop slices.",
    )
    p.add_argument(
        "--mechanism-semantic-min-pairs",
        type=int,
        default=2,
        help="Minimum connected role pairs required by semantic closed-loop gate.",
    )
    p.add_argument(
        "--mechanism-semantic-max-hops",
        type=int,
        default=6,
        help="Max hops used when checking role-pair connectivity for semantic gate.",
    )
    p.add_argument(
        "--family-aware-weight",
        type=float,
        default=0.30,
        help="Blend weight for family-aware risk channel in manual-set scoring.",
    )
    p.add_argument(
        "--family-top-quantile",
        type=float,
        default=1.0,
        help="Only apply family-aware correction on samples in top-q base risk quantile. 1.0 disables gating.",
    )
    p.add_argument(
        "--family-min-samples",
        type=int,
        default=8,
        help="Minimum samples required to fit a family-local boundary; smaller families use global fallback.",
    )
    p.add_argument(
        "--family-boundary-quantile",
        type=float,
        default=0.9,
        help="Quantile used as family-local boundary radius when computing family-aware risk.",
    )
    p.add_argument(
        "--family-shrinkage-tau",
        type=float,
        default=16.0,
        help="Larger tau means slower trust in family-local boundary (more global stabilization).",
    )
    p.add_argument(
        "--family-ramp-width",
        type=int,
        default=6,
        help="Smooth transition width around family_min_samples for local-boundary activation.",
    )
    p.add_argument(
        "--family-global-mix-floor",
        type=float,
        default=0.20,
        help="Minimum global component kept in family score blending (stability guard).",
    )
    p.add_argument(
        "--family-radius-min-ratio",
        type=float,
        default=0.60,
        help="Lower clip ratio of local family radius against global radius.",
    )
    p.add_argument(
        "--family-radius-max-ratio",
        type=float,
        default=1.80,
        help="Upper clip ratio of local family radius against global radius.",
    )
    p.add_argument(
        "--line-match-windows",
        type=str,
        default="0,1,2,3",
        help="Comma-separated line tolerance windows for row-level matching. 0 means exact line.",
    )
    p.add_argument("--line-align-enable", dest="line_align_enable", action="store_true")
    p.add_argument("--no-line-align-enable", dest="line_align_enable", action="store_false")
    p.add_argument(
        "--line-align-max-shift",
        type=int,
        default=24,
        help="Max allowed shift when mapping manual labels to nearest statement lines in the same file/function.",
    )
    p.add_argument("--risk-out", type=str, default="outputs/results/manual_eval_risk_scores.csv")
    p.add_argument("--line-risk-out", type=str, default="outputs/results/manual_eval_line_risk_scores.csv")
    p.add_argument("--line-risk-topn", type=int, default=2000, help="Aggregate line risk using top-N slice risks.")
    p.add_argument(
        "--line-risk-closure-boost",
        type=float,
        default=0.30,
        help="Boost line contribution from slices with stronger auth-external-state closure.",
    )
    p.add_argument(
        "--line-risk-stage2-weight",
        type=float,
        default=0.15,
        help="Stage-2 function-internal line localization weight. <=0 disables stage-2.",
    )
    p.add_argument(
        "--line-risk-stage2-top-functions",
        type=int,
        default=640,
        help="Only run stage-2 localization on top-N risky (file,function) buckets.",
    )
    p.add_argument(
        "--line-risk-stage2-support-weight",
        type=float,
        default=0.60,
        help="Extra weight on within-function line support frequency in stage-2.",
    )
    p.add_argument(
        "--line-risk-stage2-max-file-share",
        type=float,
        default=0.0,
        help="Optional per-file cap on stage-2 bonus share against total stage-2 bonus budget. 0 disables.",
    )
    p.add_argument(
        "--line-risk-topk",
        type=str,
        default="100,200,400,800",
        help="Comma-separated K list for line-level recall@K evaluation.",
    )
    p.add_argument("--summary-out", type=str, default="outputs/results/manual_eval_summary.json")
    p.add_argument(
        "--risk-rerank-mode",
        type=str,
        default="function_coverage",
        choices=["none", "coverage", "function_coverage", "family_hybrid"],
        help=(
            "Optional post-ranking strategy on final_risk. "
            "'coverage' uses duplicate penalties; 'function_coverage' rewards new file+function coverage; "
            "'family_hybrid' adds family-level diversity and auxiliary risk signals."
        ),
    )
    p.add_argument(
        "--risk-rerank-topn",
        type=int,
        default=2000,
        help="Only rerank top-N rows by final_risk; remainder keeps original order.",
    )
    p.add_argument(
        "--risk-rerank-file-penalty",
        type=float,
        default=0.06,
        help="Penalty multiplied by already-selected count in the same file during rerank.",
    )
    p.add_argument(
        "--risk-rerank-filefn-penalty",
        type=float,
        default=0.12,
        help="Penalty multiplied by already-selected count in the same file+function bucket during rerank.",
    )
    p.add_argument(
        "--risk-rerank-fn-novelty-bonus",
        type=float,
        default=0.04,
        help="Only for function_coverage mode: bonus per newly covered function in the same file.",
    )
    p.add_argument(
        "--risk-rerank-fn-overlap-penalty",
        type=float,
        default=0.01,
        help="Only for function_coverage mode: penalty per already-covered function overlap.",
    )
    p.add_argument(
        "--risk-rerank-max-per-filefn",
        type=int,
        default=0,
        help="Only for function_coverage mode: hard cap per (file,function) in rerank head. <=0 disables.",
    )
    p.add_argument(
        "--risk-rerank-max-per-file",
        type=int,
        default=0,
        help="Hard cap per file in rerank head. <=0 disables.",
    )
    p.add_argument(
        "--risk-rerank-file-repeat-power",
        type=float,
        default=1.0,
        help="Amplify repeated-file penalty by count^power during rerank. 1.0 keeps linear penalty.",
    )
    p.add_argument(
        "--risk-rerank-warmup-topk",
        type=int,
        default=0,
        help="Optional diversity warmup on top-K rows after rerank. 0 disables.",
    )
    p.add_argument(
        "--risk-rerank-warmup-file-cap",
        type=int,
        default=3,
        help="Max rows per file inside rerank warmup window.",
    )
    p.add_argument(
        "--risk-rerank-warmup-filefn-cap",
        type=int,
        default=1,
        help="Max rows per file+function bucket inside rerank warmup window.",
    )
    p.add_argument(
        "--risk-rerank-family-penalty",
        type=float,
        default=0.05,
        help="Only for family_hybrid mode: penalty for repeated selections from the same family.",
    )
    p.add_argument(
        "--risk-rerank-family-novelty-bonus",
        type=float,
        default=0.03,
        help="Only for family_hybrid mode: bonus for first hit from an unseen family.",
    )
    p.add_argument(
        "--risk-rerank-aux-weight",
        type=float,
        default=0.20,
        help="Only for family_hybrid mode: weight of auxiliary risk channel in rerank score.",
    )
    p.add_argument(
        "--risk-rerank-max-per-family",
        type=int,
        default=0,
        help="Only for family_hybrid mode: hard cap per family in rerank head. <=0 disables.",
    )
    p.add_argument(
        "--score-head-method",
        type=str,
        default="iforest",
        choices=["none", "iforest"],
        help="Optional unsupervised score head used before rerank. 2026-03-26 stable line uses iforest.",
    )
    p.add_argument(
        "--score-head-fit-scope",
        type=str,
        default="low_risk",
        choices=["all", "low_risk", "normal"],
        help="Fit scope for score head. 'low_risk' maps to current normal-status rows.",
    )
    p.add_argument(
        "--score-head-random-state",
        type=int,
        default=42,
        help="Random seed for score head.",
    )
    p.add_argument(
        "--binary-head-method",
        type=str,
        default="mechanism_head",
        choices=["none", "ocsvm", "selftrain_lr", "model_head", "energy_head", "gmm_head", "mechanism_head"],
        help="Optional unsupervised binary head. 2026-03-26 stable line uses ocsvm.",
    )
    p.add_argument(
        "--binary-head-fit-scope",
        type=str,
        default="all",
        choices=["all", "low_risk", "normal"],
        help="Fit scope for binary head. 'all' matches the stable line.",
    )
    p.add_argument(
        "--binary-head-contamination",
        type=float,
        default=0.08,
        help="Equivalent nu / contamination for binary head. Stable line uses 0.08.",
    )
    p.add_argument(
        "--binary-head-pseudo-neg-quantile",
        type=float,
        default=0.45,
        help="For selftrain_lr: low-risk quantile used to sample pseudo negatives from normal rows.",
    )
    p.add_argument(
        "--binary-head-pseudo-pos-quantile",
        type=float,
        default=0.92,
        help="For selftrain_lr: high-risk quantile used to sample pseudo positives from high-risk rows.",
    )
    p.add_argument(
        "--binary-head-random-state",
        type=int,
        default=42,
        help="Random seed for classifier-style binary head variants.",
    )
    p.add_argument(
        "--energy-head-ckpt",
        type=str,
        default="outputs/models/energy_head.pt",
        help="Optional learned energy-head checkpoint produced by train_unsup.py.",
    )
    p.add_argument(
        "--gmm-head-ckpt",
        type=str,
        default="outputs/models/gmm_head.pkl",
        help="Optional GMM density-head checkpoint produced by train_unsup.py.",
    )
    p.add_argument(
        "--mechanism-head-ckpt",
        type=str,
        default="outputs/models/mechanism_head.pkl",
        help="Optional mechanism-slice head checkpoint produced by train_unsup.py.",
    )
    p.add_argument(
        "--mechanism-head-override-final-risk",
        dest="mechanism_head_override_final_risk",
        action="store_true",
        help="When binary-head-method=mechanism_head, replace slice ranking score with mechanism_head_score before rerank/line risk.",
    )
    p.add_argument(
        "--no-mechanism-head-override-final-risk",
        dest="mechanism_head_override_final_risk",
        action="store_false",
    )
    p.add_argument(
        "--binary-head-score-calibration-enable",
        dest="binary_head_score_calibration_enable",
        action="store_true",
        help="Apply post-hoc calibration to energy-head score using training-side normal score distributions.",
    )
    p.add_argument(
        "--no-binary-head-score-calibration-enable",
        dest="binary_head_score_calibration_enable",
        action="store_false",
    )
    p.add_argument(
        "--binary-head-score-calibration-source",
        type=str,
        default="",
        help="Optional training risk CSV used to fit score calibration when checkpoint does not already store it.",
    )
    p.add_argument(
        "--binary-head-score-calibration-group-by",
        type=str,
        default="prototype",
        choices=["family", "prototype"],
        help="Grouping used by score calibration. Prototype routing transfers to unseen bridge families better than exact family IDs.",
    )
    p.add_argument(
        "--binary-head-score-calibration-fit-scope",
        type=str,
        default="normal",
        choices=["all", "normal", "low_risk", "boundary", "high_risk"],
        help="Which training rows are used to fit score calibration when reading from CSV.",
    )
    p.add_argument("--binary-head-score-calibration-alpha", type=float, default=0.40)
    p.add_argument("--binary-head-score-calibration-min-samples", type=int, default=32)
    p.add_argument("--binary-head-score-calibration-shrinkage-tau", type=float, default=16.0)
    p.add_argument("--binary-head-score-calibration-global-mix-floor", type=float, default=0.20)
    p.add_argument("--binary-head-score-calibration-quantile-bins", type=int, default=257)
    p.add_argument("--proto-resp-gray-adjust-enable", dest="proto_resp_gray_adjust_enable", action="store_true")
    p.add_argument("--no-proto-resp-gray-adjust-enable", dest="proto_resp_gray_adjust_enable", action="store_false")
    p.add_argument(
        "--proto-resp-gray-adjust-mode",
        type=str,
        default="negent",
        choices=["negent", "respmax", "mix", "sharp"],
        help="Responsibility-confidence signal used to suppress low-confidence gray-zone positives.",
    )
    p.add_argument(
        "--proto-resp-gray-adjust-beta",
        type=float,
        default=0.04,
        help="Maximum gray-zone score penalty applied to low-confidence prototype assignments.",
    )
    p.add_argument("--proto-resp-gray-adjust-gated", dest="proto_resp_gray_adjust_gated", action="store_true")
    p.add_argument("--no-proto-resp-gray-adjust-gated", dest="proto_resp_gray_adjust_gated", action="store_false")
    p.add_argument(
        "--proto-resp-gray-adjust-gate-center",
        type=float,
        default=0.50,
        help="When gated, penalty decays to zero as binary score approaches this value inside the gray zone.",
    )
    p.add_argument(
        "--triage-low-quantile",
        type=float,
        default=0.0,
        help="Low-risk gate quantile for triage. Stable line uses 0.0.",
    )
    p.add_argument(
        "--triage-high-quantile",
        type=float,
        default=0.90,
        help="High-risk gate quantile for triage. Stable line uses 0.90.",
    )
    p.add_argument(
        "--triage-gray-threshold",
        type=float,
        default=0.28,
        help="Normalized gray-zone threshold for binary head. Current official prototype-energy mainline uses 0.28.",
    )
    p.add_argument("--triage-family-aware-gray", dest="triage_family_aware_gray", action="store_true")
    p.add_argument("--no-triage-family-aware-gray", dest="triage_family_aware_gray", action="store_false")
    p.add_argument("--triage-family-min-samples", type=int, default=32)
    p.add_argument("--triage-adaptive-gray", dest="triage_adaptive_gray", action="store_true")
    p.add_argument("--no-triage-adaptive-gray", dest="triage_adaptive_gray", action="store_false")
    p.add_argument(
        "--triage-adaptive-gray-source",
        type=str,
        default="outputs/results/unsup_final_risk_scores.csv",
        help="Training-side risk CSV used to derive adaptive gray thresholds.",
    )
    p.add_argument(
        "--triage-adaptive-gray-column",
        type=str,
        default="energy_head_score",
        help="Score column inside adaptive-gray source CSV.",
    )
    p.add_argument(
        "--triage-adaptive-gray-group-by",
        type=str,
        default="family",
        choices=["family", "prototype"],
        help="Grouping key used when deriving local adaptive gray thresholds.",
    )
    p.add_argument("--triage-closure-boost-enable", dest="triage_closure_boost_enable", action="store_true")
    p.add_argument("--no-triage-closure-boost-enable", dest="triage_closure_boost_enable", action="store_false")
    p.add_argument(
        "--triage-closure-boost-value",
        type=float,
        default=0.10,
        help="In two-stage cascade, add this boost to gray-zone binary score when the same function already has enough high-risk slices.",
    )
    p.add_argument(
        "--triage-closure-boost-min-high",
        type=int,
        default=2,
        help="Minimum count of high-risk slices in a function required to trigger closure boost on its gray slices.",
    )
    p.add_argument(
        "--triage-closure-boost-cap",
        type=float,
        default=1.0,
        help="Upper cap after closure boost is applied to gray-zone binary score.",
    )
    p.add_argument(
        "--binary-target-mode",
        type=str,
        default="file_function",
        choices=["file_function", "row_window2", "union_window2"],
        help="Pseudo label construction used for binary evaluation.",
    )
    p.add_argument(
        "--binary-target-window",
        type=int,
        default=2,
        help="Line window used when binary-target-mode includes row matching.",
    )
    p.add_argument(
        "--binary-eval-topn",
        type=int,
        default=0,
        help="Evaluate binary metrics on top-N reranked rows only. 0 means all rows.",
    )
    p.add_argument(
        "--binary-out",
        type=str,
        default="outputs/results/manual_eval_binary_predictions.csv",
        help="Binary prediction output csv reconstructed for the stable line.",
    )
    p.add_argument("--metamorphic-gray-enable", dest="metamorphic_gray_enable", action="store_true")
    p.add_argument("--no-metamorphic-gray-enable", dest="metamorphic_gray_enable", action="store_false")
    p.add_argument("--metamorphic-feature-enable", dest="metamorphic_feature_enable", action="store_true")
    p.add_argument("--no-metamorphic-feature-enable", dest="metamorphic_feature_enable", action="store_false")
    p.add_argument(
        "--metamorphic-feature-max-rows",
        type=int,
        default=5000,
        help="Maximum rows used to compute metamorphic instability features for energy head inference. 0 means all selected rows.",
    )
    p.add_argument(
        "--metamorphic-feature-select",
        type=str,
        default="boundary",
        choices=["boundary", "all"],
        help="Which rows receive metamorphic instability feature computation for energy head inference.",
    )
    p.add_argument(
        "--metamorphic-feature-boundary-quantile",
        type=float,
        default=0.80,
        help="Within selected rows, keep only samples whose boundary_proximity is in top-q quantile for metamorphic features.",
    )
    p.add_argument(
        "--metamorphic-feature-switch-min",
        type=float,
        default=0.25,
        help="After variant evaluation, zero metamorphic features when proto switch rate is below this threshold.",
    )
    p.add_argument(
        "--metamorphic-feature-gate-mode",
        type=str,
        default="hard",
        choices=["none", "hard", "soft"],
        help="How to gate metamorphic instability features after variant evaluation.",
    )
    p.add_argument(
        "--metamorphic-gray-max-rows",
        type=int,
        default=2500,
        help="Maximum number of gray-zone rows to evaluate with metamorphic variants. 0 means all gray rows.",
    )
    p.add_argument(
        "--metamorphic-gray-score-weight",
        type=float,
        default=0.20,
        help="Additive weight applied to binary score using metamorphic instability for gray rows.",
    )
    p.add_argument(
        "--metamorphic-gray-risk-weight",
        type=float,
        default=0.05,
        help="Additive weight applied to risk score using metamorphic instability for gray rows.",
    )
    p.add_argument(
        "--metamorphic-max-hops",
        type=int,
        default=4,
        help="Role closure search depth used when building semantic-preserving metamorphic variants.",
    )
    p.set_defaults(
        adaptive_boundary_refine=True,
        slice_formal_propagation=True,
        slice_include_contains=False,
        line_align_enable=True,
        function_bag_enable=False,
        mechanism_head_override_final_risk=True,
        binary_head_score_calibration_enable=False,
        proto_resp_gray_adjust_enable=False,
        proto_resp_gray_adjust_gated=True,
        metamorphic_feature_enable=False,
        metamorphic_gray_enable=False,
        triage_closure_boost_enable=False,
    )
    return p.parse_args()


def run_cmd(cmd: Sequence[str], cwd: Path) -> None:
    print("[evaluate_manual_set] RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def normalize_path(s: str) -> str:
    return s.replace("\\", "/")


def to_label_file_key(path_str: str) -> str:
    p = normalize_path(path_str)
    if MANUAL_PREFIX in p:
        p = p.split(MANUAL_PREFIX, 1)[1]
    return p


def map_old_path_to_local(project_root: Path, old_path: str) -> Optional[Path]:
    if not isinstance(old_path, str) or not old_path.startswith(OLD_PREFIX):
        return None
    rel = old_path[len(OLD_PREFIX) :]
    p = project_root / rel.replace("/", "\\")
    return p.resolve()


def collect_manual_files(
    project_root: Path,
    manual_root: Path,
    label_df: pd.DataFrame,
    scope: str,
    max_files: int,
) -> List[Path]:
    files: List[Path] = []
    if scope == "all-sol":
        files = sorted([p.resolve() for p in manual_root.rglob("*.sol") if p.is_file()], key=lambda x: str(x))
    else:
        if "local_abs_path" in label_df.columns:
            for v in label_df["local_abs_path"].dropna().tolist():
                p = Path(str(v))
                if p.exists():
                    files.append(p.resolve())
        elif "full_path" in label_df.columns:
            for v in label_df["full_path"].dropna().tolist():
                p = map_old_path_to_local(project_root, str(v))
                if p is not None and p.exists():
                    files.append(p.resolve())
        else:
            for v in label_df["file"].dropna().tolist():
                p = manual_root / str(v).replace("/", "\\")
                if p.exists():
                    files.append(p.resolve())
    files = sorted(set(files), key=lambda x: str(x))
    if max_files and max_files > 0:
        files = files[:max_files]
    return files


def parse_topk(s: str) -> List[int]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    vals = sorted(set(v for v in vals if v > 0))
    return vals if vals else [100]


def parse_windows(s: str) -> List[int]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    vals = sorted(set(v for v in vals if v >= 0))
    if 0 not in vals:
        vals = [0] + vals
    return vals


def load_dual_model(ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model_type = str(ckpt.get("model_type", "dual-view")).strip().lower()
    model_cls = GlobalLocalDualViewModel if model_type == "global-local-dual-view" else DualViewModel
    state = ckpt["model_state_dict"]
    rel_alpha_key = "full_model.encoder.rel_alpha"
    if rel_alpha_key in state:
        relation_count = int(state[rel_alpha_key].shape[1])
    else:
        stats_key = "full_model.stats_proj.0.weight"
        stats_dim = int(state[stats_key].shape[1]) if stats_key in state else (len(EDGE_TYPES) + 10)
        relation_count = int(max(1, stats_dim - len(("function", "statement", "state_var", "call", "condition", "data_object")) - 4))
    edge_types = tuple(EDGE_TYPES[:relation_count])
    if model_cls is GlobalLocalDualViewModel:
        model = model_cls(
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
            classifier_enable=bool(cfg.get("global_local_classifier_enable", False)),
            classifier_hidden_dim=int(cfg.get("global_local_classifier_hidden_dim", 64)),
            edge_types=edge_types,
            edge_weight_config=build_soft_edge_weight_config(
                enable=bool(cfg.get("edge_soft_weight_enable", False)),
                local_call_summary=float(cfg.get("edge_weight_local_call_summary", 0.35)),
                shared_object_summary=float(cfg.get("edge_weight_shared_object_summary", 0.45)),
                cross_function_state_flow=float(cfg.get("edge_weight_cross_function_state_flow", 0.55)),
                local_call=float(cfg.get("edge_weight_local_call", 0.75)),
                state_summary=float(cfg.get("edge_weight_state_summary", 0.85)),
            ),
        ).to(device)
    else:
        model = model_cls(
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
            edge_types=edge_types,
            edge_weight_config=build_soft_edge_weight_config(
                enable=bool(cfg.get("edge_soft_weight_enable", False)),
                local_call_summary=float(cfg.get("edge_weight_local_call_summary", 0.35)),
                shared_object_summary=float(cfg.get("edge_weight_shared_object_summary", 0.45)),
                cross_function_state_flow=float(cfg.get("edge_weight_cross_function_state_flow", 0.55)),
                local_call=float(cfg.get("edge_weight_local_call", 0.75)),
                state_summary=float(cfg.get("edge_weight_state_summary", 0.85)),
            ),
            text_view_enable=bool(cfg.get("text_view_enable", False)),
            text_fuse_enable=bool(cfg.get("text_view_fuse_enable", True)),
            text_vocab_size=int(cfg.get("text_vocab_size", 8192)),
            text_max_tokens=int(cfg.get("text_max_tokens", 96)),
            text_mix_weight=float(cfg.get("text_view_mix_weight", 0.20)),
            text_dropout=float(cfg.get("text_dropout", 0.10)),
            gate_enable=bool(cfg.get("slice_gate_enable", False)),
            gate_hidden_dim=int(cfg.get("slice_gate_hidden_dim", 32)),
            gate_temperature=float(cfg.get("slice_gate_temperature", 1.0)),
            gate_role_target=float(cfg.get("slice_gate_role_target", 0.35)),
        ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def load_global_graph_cache(graph_dir: Path, rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    graph_map: Dict[str, Dict[str, Any]] = {}
    need_ids = {str(r.get("contract_id", "")).strip() for r in rows if str(r.get("contract_id", "")).strip()}
    for cid in need_ids:
        path = graph_dir / f"{cid}.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            graph_map[cid] = json.load(f)
    return graph_map


def compute_embeddings(
    model: torch.nn.Module,
    rows: List[Dict[str, Any]],
    device: torch.device,
    global_graph_dir: Optional[Path] = None,
) -> Tuple[Dict[str, np.ndarray], List[Set[int]]]:
    zf_list: List[np.ndarray] = []
    zs_list: List[np.ndarray] = []
    zj_list: List[np.ndarray] = []
    zgj_list: List[np.ndarray] = []
    zt_list: List[np.ndarray] = []
    zg_list: List[np.ndarray] = []
    zfused_list: List[np.ndarray] = []
    model_binary_score_list: List[float] = []
    line_sets: List[Set[int]] = []
    global_graph_map: Dict[str, Dict[str, Any]] = {}
    global_emb_cache: Dict[str, torch.Tensor] = {}
    use_global_local = isinstance(model, GlobalLocalDualViewModel)
    if use_global_local and global_graph_dir is not None and global_graph_dir.exists():
        global_graph_map = load_global_graph_cache(global_graph_dir, rows)
    with torch.no_grad():
        for row in rows:
            full_graph = row["full_graph"]
            skeleton_graph = row["skeleton_graph"]
            edge_types = getattr(model, "edge_types", EDGE_TYPES)
            edge_weight_config = getattr(model, "edge_weight_config", None)
            full_t, skel_t = dual_view_to_tensors(
                full_graph,
                skeleton_graph,
                device=device,
                edge_types=edge_types,
                edge_weight_config=edge_weight_config,
            )
            if use_global_local:
                cid = str(row.get("contract_id", "")).strip()
                global_graph = global_graph_map.get(cid, full_graph)
                if cid not in global_emb_cache:
                    global_emb_cache[cid] = model.global_model(
                        graph_to_tensor(
                            global_graph,
                            device=device,
                            edge_types=edge_types,
                            edge_weight_config=edge_weight_config,
                        )
                    )
                zf = model.full_model(full_t)
                zs = model.skeleton_model(skel_t)
                zj = 0.5 * (zf + zs)
                zfused, _ = model.fuse(zj, global_emb_cache[cid])
                cls_logit = model.classify(zfused, zj, global_emb_cache[cid])
                zg_list.append(global_emb_cache[cid].detach().cpu().numpy())
                zfused_list.append(zfused.detach().cpu().numpy())
                model_binary_score_list.append(float(torch.sigmoid(cls_logit).item()) if cls_logit is not None else 0.0)
            else:
                out = model(full_t, skel_t, mechanism_text=str(row.get("mechanism_text", "") or ""))  # type: ignore[misc]
                zf = out.z_full
                zs = out.z_skeleton
                zj = out.z_joint
            zf_list.append(zf.detach().cpu().numpy())
            zs_list.append(zs.detach().cpu().numpy())
            zj_list.append(zj.detach().cpu().numpy())
            if not use_global_local and out.z_graph_joint is not None:
                zgj_list.append(out.z_graph_joint.detach().cpu().numpy())
            if not use_global_local and out.z_text is not None:
                zt_list.append(out.z_text.detach().cpu().numpy())
            lines: Set[int] = set()
            for n in full_graph.get("nodes", []):
                if n.get("type") != "statement":
                    continue
                line = n.get("line")
                if not isinstance(line, int):
                    continue
                end_line = n.get("end_line", line)
                if not isinstance(end_line, int):
                    end_line = line
                lo = min(int(line), int(end_line))
                hi = max(int(line), int(end_line))
                if hi - lo <= 4:
                    for ln in range(lo, hi + 1):
                        lines.add(int(ln))
                else:
                    lines.add(int(lo))
                    lines.add(int(hi))
            line_sets.append(lines)
    emb = {
        "z_full": np.stack(zf_list, axis=0).astype(np.float32) if zf_list else np.zeros((0, model.embed_dim), dtype=np.float32),
        "z_skeleton": np.stack(zs_list, axis=0).astype(np.float32) if zs_list else np.zeros((0, model.embed_dim), dtype=np.float32),
        "z_joint": np.stack(zj_list, axis=0).astype(np.float32) if zj_list else np.zeros((0, model.embed_dim), dtype=np.float32),
    }
    if zgj_list:
        emb["z_graph_joint"] = np.stack(zgj_list, axis=0).astype(np.float32)
    if zt_list:
        emb["z_text"] = np.stack(zt_list, axis=0).astype(np.float32)
    if zg_list:
        emb["z_global"] = np.stack(zg_list, axis=0).astype(np.float32)
    if zfused_list:
        emb["z_fused"] = np.stack(zfused_list, axis=0).astype(np.float32)
    if model_binary_score_list:
        emb["model_binary_score"] = np.asarray(model_binary_score_list, dtype=np.float32)
    return emb, line_sets


def nearest_to_centers(emb: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dmat = np.linalg.norm(emb[:, None, :] - centers[None, :, :], axis=2)
    assign = np.argmin(dmat, axis=1)
    nearest = dmat[np.arange(len(emb)), assign]
    return assign.astype(np.int64), nearest.astype(np.float32)


def build_eval_rows(
    dual_rows: List[Dict[str, Any]],
    line_sets: List[Set[int]],
    assign: np.ndarray,
    nearest: np.ndarray,
    margin: np.ndarray,
    view_gap: np.ndarray,
    refined: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    out = []
    for i, row in enumerate(dual_rows):
        rel = str(row.get("relative_source_path", ""))
        file_key = to_label_file_key(rel)
        fn_names = sorted({str(x).strip() for x in row.get("function_names", []) if str(x).strip()})
        closure_feat = compute_graph_closure_features(
            row.get("full_graph", {}) or {},
            row.get("skeleton_graph", {}) or {},
        )
        out.append(
            {
                "slice_id": row.get("slice_id"),
                "contract_id": row.get("contract_id"),
                "relative_source_path": rel,
                "file": file_key,
                "function_names": "|".join(fn_names),
                "line_candidates": "|".join(str(x) for x in sorted(line_sets[i])),
                "prototype_id": int(assign[i]),
                "nearest_distance": float(nearest[i]),
                "boundary_margin": float(margin[i]),
                "view_gap": float(view_gap[i]),
                "global_gap": float(refined.get("global_gap", np.zeros_like(refined["final_risk"]))[i]),
                "fused_gap": float(refined.get("fused_gap", np.zeros_like(refined["final_risk"]))[i]),
                "proto_score": float(refined["proto_score"][i]),
                "view_score": float(refined["view_score"][i]),
                "density_risk": float(refined["density_risk"][i]),
                "instability": float(refined["instability"][i]),
                "boundary_proximity": float(refined.get("boundary_proximity", np.zeros_like(refined["final_risk"]))[i]),
                "adaptive_local_scale": float(refined.get("adaptive_local_scale", np.ones_like(refined["final_risk"]))[i]),
                "adaptive_focus_mask": int(
                    refined.get("adaptive_focus_mask", np.ones_like(refined["final_risk"], dtype=np.bool_))[i]
                ),
                "base_final_risk": float(refined.get("base_final_risk", refined["final_risk"])[i]),
                "mechanism_risk": float(refined.get("mechanism_risk", np.zeros_like(refined["final_risk"]))[i]),
                "family_risk": float(refined.get("family_risk", np.zeros_like(refined["final_risk"]))[i]),
                "final_risk": float(refined["final_risk"][i]),
                "status": str(refined["status"][i]),
                **closure_feat,
            }
        )
    out.sort(key=lambda x: x["final_risk"], reverse=True)
    return out


def build_function_bag_eval_rows(
    bag_rows: List[Dict[str, Any]],
    bag_line_sets: List[Set[int]],
    assign: np.ndarray,
    nearest: np.ndarray,
    margin: np.ndarray,
    view_gap: np.ndarray,
    refined: Dict[str, np.ndarray],
    bag_global_gap: np.ndarray,
    bag_fused_gap: np.ndarray,
    bag_closure_features: Optional[List[Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    out = []
    zeros = np.zeros_like(refined["final_risk"])
    ones = np.ones_like(refined["final_risk"])
    for i, row in enumerate(bag_rows):
        rel = str(row.get("relative_source_path", ""))
        file_key = to_label_file_key(rel)
        fn_names = sorted({str(x).strip() for x in row.get("function_names", []) if str(x).strip()})
        closure_feat = (
            dict(bag_closure_features[i])
            if bag_closure_features is not None and i < len(bag_closure_features)
            else {}
        )
        out.append(
            {
                "slice_id": row.get("slice_id"),
                "contract_id": row.get("contract_id"),
                "relative_source_path": rel,
                "file": file_key,
                "function_names": "|".join(fn_names),
                "function_bag_enable": 1,
                "function_bag_size": int(row.get("function_bag_size", 1)),
                "line_candidates": "|".join(str(x) for x in sorted(bag_line_sets[i])),
                "prototype_id": int(assign[i]),
                "nearest_distance": float(nearest[i]),
                "boundary_margin": float(margin[i]),
                "view_gap": float(view_gap[i]),
                "global_gap": float(bag_global_gap[i]) if len(bag_global_gap) else 0.0,
                "fused_gap": float(bag_fused_gap[i]) if len(bag_fused_gap) else 0.0,
                "proto_score": float(refined["proto_score"][i]),
                "view_score": float(refined["view_score"][i]),
                "density_risk": float(refined["density_risk"][i]),
                "instability": float(refined["instability"][i]),
                "boundary_proximity": float(refined.get("boundary_proximity", zeros)[i]),
                "adaptive_local_scale": float(refined.get("adaptive_local_scale", ones)[i]),
                "adaptive_focus_mask": int(refined.get("adaptive_focus_mask", np.ones_like(refined["final_risk"], dtype=np.bool_))[i]),
                "base_final_risk": float(refined.get("base_final_risk", refined["final_risk"])[i]),
                "mechanism_risk": float(refined.get("mechanism_risk", zeros)[i]),
                "family_risk": float(refined.get("family_risk", zeros)[i]),
                "final_risk": float(refined["final_risk"][i]),
                "status": str(refined["status"][i]),
                **closure_feat,
            }
        )
    out.sort(key=lambda x: x["final_risk"], reverse=True)
    return out


def parse_fn_set(fn_pipe: str) -> Set[str]:
    return {x.strip().lower() for x in fn_pipe.split("|") if x.strip()}


def parse_line_set(line_pipe: str) -> Set[int]:
    out: Set[int] = set()
    for x in line_pipe.split("|"):
        x = x.strip()
        if not x:
            continue
        try:
            out.add(int(x))
        except ValueError:
            pass
    return out


def _triage_function_group_key(row: Dict[str, Any]) -> str:
    file_key = normalize_path(
        str(
            row.get("file")
            or row.get("relative_source_path")
            or row.get("source_path")
            or row.get("contract_id")
            or "<unknown_file>"
        )
    ).lower()
    raw_fn = row.get("function_names", "")
    if isinstance(raw_fn, list):
        fn_text = "|".join(str(x) for x in raw_fn if str(x).strip())
    else:
        fn_text = str(raw_fn or "")
    fn_key = _canonical_fn(fn_text)
    return f"{file_key}:::{fn_key}"


def _file_key_from_eval_row(row: Dict[str, Any]) -> str:
    return str(row.get("relative_source_path") or row.get("file") or row.get("contract_id") or "<unknown_file>")


def _function_bag_id_eval(row: Dict[str, Any]) -> str:
    file_key = _file_key_from_eval_row(row)
    fn_key = _canonical_fn(str("|".join(row.get("function_names", [])) if isinstance(row.get("function_names"), list) else row.get("function_names", "")))
    return f"{file_key}:::{fn_key}"


def _aggregate_group_embeddings_eval(arr: np.ndarray, groups: List[np.ndarray], mode: str) -> np.ndarray:
    if len(groups) <= 0:
        return np.zeros((0, arr.shape[1] if arr.ndim == 2 else 0), dtype=np.float32)
    out: List[np.ndarray] = []
    agg_mode = str(mode or "mean").strip().lower()
    for idx in groups:
        x = np.asarray(arr[idx], dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        mean_v = np.mean(x, axis=0).astype(np.float32)
        if agg_mode == "mean":
            out.append(mean_v)
            continue
        max_v = np.max(x, axis=0).astype(np.float32)
        if agg_mode == "max":
            out.append(max_v)
        else:
            out.append((0.7 * mean_v + 0.3 * max_v).astype(np.float32))
    return np.stack(out, axis=0).astype(np.float32)


def _aggregate_group_scalar_eval(arr: np.ndarray, groups: List[np.ndarray], reduce: str = "mean") -> np.ndarray:
    if len(groups) <= 0:
        return np.zeros((0,), dtype=np.float32)
    src = np.asarray(arr, dtype=np.float32)
    out: List[float] = []
    red = str(reduce or "mean").strip().lower()
    for idx in groups:
        vals = src[idx]
        if red == "max":
            out.append(float(np.max(vals)))
        elif red == "min":
            out.append(float(np.min(vals)))
        else:
            out.append(float(np.mean(vals)))
    return np.asarray(out, dtype=np.float32)


def build_function_bag_view_eval(
    dual_rows: List[Dict[str, Any]],
    emb_main: np.ndarray,
    emb_full: np.ndarray,
    emb_skel: np.ndarray,
    mechanism_raw: np.ndarray,
    agg_mode: str = "mean",
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    group_to_indices: Dict[str, List[int]] = {}
    ordered_keys: List[str] = []
    for i, row in enumerate(dual_rows):
        gid = _function_bag_id_eval(row)
        if gid not in group_to_indices:
            group_to_indices[gid] = []
            ordered_keys.append(gid)
        group_to_indices[gid].append(i)
    groups = [np.asarray(group_to_indices[k], dtype=np.int64) for k in ordered_keys]
    row_to_group = np.zeros((len(dual_rows),), dtype=np.int64)
    bag_rows: List[Dict[str, Any]] = []
    for gid, idx in enumerate(groups):
        row_to_group[idx] = gid
        first = dual_rows[int(idx[0])]
        file_key = _file_key_from_eval_row(first)
        raw_fns = first.get("function_names", [])
        if isinstance(raw_fns, list):
            fn_join = "|".join(str(x) for x in raw_fns if str(x).strip())
        else:
            fn_join = str(raw_fns or "")
        fn_key = _canonical_fn(fn_join)
        bag_rows.append(
            {
                "slice_id": f"bag::{gid}",
                "contract_id": first.get("contract_id"),
                "relative_source_path": file_key,
                "file": to_label_file_key(file_key),
                "function_names": [fn_key],
                "function_bag_id": f"{file_key}:::{fn_key}",
                "function_bag_size": int(len(idx)),
            }
        )
    bag_main = _aggregate_group_embeddings_eval(emb_main, groups, agg_mode)
    bag_full = _aggregate_group_embeddings_eval(emb_full, groups, agg_mode)
    bag_skel = _aggregate_group_embeddings_eval(emb_skel, groups, agg_mode)
    bag_mech = _aggregate_group_scalar_eval(mechanism_raw, groups, reduce="max")
    bag_sizes = np.asarray([len(x) for x in groups], dtype=np.int32)
    return bag_rows, bag_main, bag_full, bag_skel, bag_mech, row_to_group, bag_sizes


def _iter_statement_lines(node: Dict[str, Any]) -> List[int]:
    line = node.get("line")
    if not isinstance(line, int):
        return []
    end_line = node.get("end_line", line)
    if not isinstance(end_line, int):
        end_line = line
    lo = min(int(line), int(end_line))
    hi = max(int(line), int(end_line))
    if hi - lo <= 4:
        return [int(x) for x in range(lo, hi + 1)]
    return [int(lo), int(hi)]


def build_manual_label_alignment(
    label_df: pd.DataFrame,
    dual_rows: Sequence[Dict[str, Any]],
    max_shift: int = 24,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    file_fn_lines: Dict[Tuple[str, str], Set[int]] = {}
    file_lines: Dict[str, Set[int]] = {}
    for row in dual_rows:
        rel = str(row.get("relative_source_path", "") or "")
        file_key = normalize_path(to_label_file_key(rel) if rel else str(row.get("file", "") or ""))
        if not file_key:
            continue
        graph = row.get("full_graph", {}) or {}
        for node in graph.get("nodes", []):
            if node.get("type") != "statement":
                continue
            lines = _iter_statement_lines(node)
            if not lines:
                continue
            file_lines.setdefault(file_key, set()).update(lines)
            fn_key = _canonical_fn(str(node.get("function", "") or ""))
            if fn_key != "<unknown_fn>":
                file_fn_lines.setdefault((file_key, fn_key), set()).update(lines)

    records: List[Dict[str, Any]] = []
    shifts: List[int] = []
    exact_count = 0
    aligned_count = 0
    unresolved = 0
    max_allowed = int(max(0, max_shift))
    for _, src in label_df.iterrows():
        rec = src.to_dict()
        file_key = normalize_path(str(src["file"]))
        fn_val = None if pd.isna(src.get("function")) else str(src.get("function", "")).strip().lower()
        raw_line = int(src["line"])
        candidates = set()
        align_mode = "none"
        if fn_val:
            candidates = set(file_fn_lines.get((file_key, fn_val), set()))
            if candidates:
                align_mode = "function"
        if not candidates:
            candidates = set(file_lines.get(file_key, set()))
            if candidates:
                align_mode = "file"
        aligned_line = raw_line
        abs_shift = 0
        if candidates:
            if raw_line in candidates:
                exact_count += 1
                align_mode = f"{align_mode}_exact" if align_mode != "none" else "exact"
            else:
                nearest = min(candidates, key=lambda ln: (abs(int(ln) - raw_line), int(ln)))
                abs_shift = int(abs(int(nearest) - raw_line))
                if abs_shift <= max_allowed:
                    aligned_line = int(nearest)
                    aligned_count += 1
                    shifts.append(abs_shift)
                    align_mode = f"{align_mode}_nearest"
                else:
                    unresolved += 1
                    align_mode = f"{align_mode}_unresolved"
        else:
            unresolved += 1
        rec["raw_line"] = int(raw_line)
        rec["line"] = int(aligned_line)
        rec["aligned_line"] = int(aligned_line)
        rec["line_shift"] = int(abs_shift)
        rec["alignment_mode"] = str(align_mode)
        rec["line_aligned"] = bool(aligned_line != raw_line)
        records.append(rec)

    aligned_df = pd.DataFrame.from_records(records)
    diag = {
        "enabled": True,
        "label_rows": int(len(label_df)),
        "candidate_file_functions": int(len(file_fn_lines)),
        "candidate_files": int(len(file_lines)),
        "exact_count": int(exact_count),
        "nearest_aligned_count": int(aligned_count),
        "resolved_count": int(exact_count + aligned_count),
        "unresolved_count": int(unresolved),
        "max_shift": int(max_allowed),
        "mean_shift": float(np.mean(shifts)) if shifts else 0.0,
        "median_shift": float(np.median(shifts)) if shifts else 0.0,
        "max_observed_shift": int(max(shifts)) if shifts else 0,
    }
    return aligned_df, diag


def _statement_node_importance(node: Dict[str, Any]) -> float:
    roles = set(node.get("roles", []) or [])
    has_state = int("state_change" in roles)
    has_ext = int("external_interaction" in roles)
    has_auth = int("auth_constraint" in roles)
    multi_role = max(0, has_state + has_ext + has_auth - 1)
    stmt_type = str(node.get("stmt_type", "")).lower()
    semantic = str(node.get("semantic_class", "")).lower()

    # Role-aware weighting to highlight cross-chain critical statements.
    w = (
        1.0
        + 0.55 * float(has_state)
        + 0.70 * float(has_ext)
        + 0.45 * float(has_auth)
        + 0.20 * float(multi_role)
    )
    if stmt_type in {"external_interaction", "call"}:
        w += 0.15
    if semantic in {"constraint_check", "state_change", "external_interaction"}:
        w += 0.10
    return float(max(0.05, w))


def build_slice_line_weight_map(full_graph: Dict[str, Any]) -> Dict[int, float]:
    line_score: Dict[int, float] = {}
    total = 0.0
    for node in full_graph.get("nodes", []):
        if node.get("type") != "statement":
            continue
        lines = _iter_statement_lines(node)
        if not lines:
            continue
        node_imp = _statement_node_importance(node)
        share = node_imp / float(max(1, len(lines)))
        for ln in lines:
            line_score[ln] = line_score.get(ln, 0.0) + share
            total += share
    if total <= 1e-8:
        return {}
    inv = 1.0 / total
    return {int(k): float(v * inv) for k, v in line_score.items() if v > 0.0}


def _normalize_map_values(x: Dict[Any, float]) -> Dict[Any, float]:
    if not x:
        return {}
    vals = np.asarray(list(x.values()), dtype=np.float32)
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi - lo <= 1e-8:
        return {k: 0.0 for k in x.keys()}
    inv = 1.0 / (hi - lo + 1e-8)
    return {k: float((float(v) - lo) * inv) for k, v in x.items()}


def _slice_closure_scale(dual_row: Dict[str, Any], closure_boost: float = 0.25) -> float:
    sk_meta = ((dual_row.get("skeleton_graph", {}) or {}).get("meta", {}) or {})
    links = float(sk_meta.get("mechanism_links", 0.0))
    mechanism_complete = bool(sk_meta.get("mechanism_complete", False))
    closure_strength = (0.60 if mechanism_complete else 0.0) + min(0.40, 0.15 * max(0.0, links))
    return float(1.0 + max(0.0, float(closure_boost)) * closure_strength)


def build_line_risk_rows(
    risk_rows: List[Dict[str, Any]],
    dual_rows: List[Dict[str, Any]],
    topn: int = 2000,
    closure_boost: float = 0.25,
    stage2_weight: float = 0.35,
    stage2_top_functions: int = 320,
    stage2_support_weight: float = 0.25,
    stage2_max_file_share: float = 0.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not risk_rows:
        return [], {
            "topn": int(topn),
            "closure_boost": float(closure_boost),
            "stage2_weight": float(stage2_weight),
            "stage2_top_functions": int(stage2_top_functions),
            "stage2_support_weight": float(stage2_support_weight),
            "stage2_max_file_share": float(stage2_max_file_share),
            "stage2_applied": False,
            "stage2_selected_functions": 0,
            "stage2_candidate_functions": 0,
            "stage2_bonus_sum": 0.0,
        }
    row_map: Dict[str, Dict[str, Any]] = {str(r.get("slice_id")): r for r in dual_rows}
    line_map_cache: Dict[str, Dict[int, float]] = {}
    for sid, drow in row_map.items():
        line_map_cache[sid] = build_slice_line_weight_map(drow.get("full_graph", {}) or {})

    k = int(max(1, min(int(topn), len(risk_rows))))
    agg: Dict[Tuple[str, int], Dict[str, float]] = {}
    fn_score: Dict[Tuple[str, str], float] = {}
    fn_line_score: Dict[Tuple[str, str, int], float] = {}
    fn_line_support: Dict[Tuple[str, str, int], int] = {}
    for rank, row in enumerate(risk_rows[:k], start=1):
        sid = str(row.get("slice_id", ""))
        file_key = normalize_path(str(row.get("file") or row.get("relative_source_path") or ""))
        base_risk = float(row.get("final_risk", 0.0))
        rank_decay = 1.0 / max(1.0, math.log2(float(rank) + 2.0))
        closure_scale = _slice_closure_scale(row_map.get(sid, {}), closure_boost=closure_boost)
        effective_risk = float(base_risk * rank_decay * closure_scale)
        lmap = dict(line_map_cache.get(sid, {}))
        if not lmap:
            fallback = sorted(parse_line_set(str(row.get("line_candidates", ""))))
            if fallback:
                u = 1.0 / float(len(fallback))
                lmap = {int(x): float(u) for x in fallback}
        if not lmap:
            continue
        fn_set = _function_key_set(str(row.get("function_names", "")))
        fn_share = 1.0 / float(max(1, len(fn_set)))
        for fn in fn_set:
            fnk = (file_key, str(fn))
            fn_score[fnk] = fn_score.get(fnk, 0.0) + float(effective_risk * fn_share)
        for ln, lw in lmap.items():
            key = (file_key, int(ln))
            rec = agg.setdefault(
                key,
                {
                    "file": file_key,
                    "line": int(ln),
                    "line_risk": 0.0,
                    "base_line_risk": 0.0,
                    "stage2_bonus": 0.0,
                    "support_slices": 0.0,
                    "max_slice_risk": 0.0,
                },
            )
            contrib = float(effective_risk * float(lw))
            rec["line_risk"] += contrib
            rec["base_line_risk"] += contrib
            rec["support_slices"] += 1.0
            rec["max_slice_risk"] = max(float(rec["max_slice_risk"]), base_risk)
            for fn in fn_set:
                lf_key = (file_key, str(fn), int(ln))
                fn_line_score[lf_key] = fn_line_score.get(lf_key, 0.0) + float(contrib * fn_share)
                fn_line_support[lf_key] = fn_line_support.get(lf_key, 0) + 1

    stage2_bonus_sum = 0.0
    stage2_bonus_scale = 1.0
    stage2_bonus_budget = 0.0
    stage2_file_scale = 1.0
    stage2_applied = bool(stage2_weight > 0.0 and fn_score and fn_line_score)
    selected_fn_count = 0
    if stage2_applied:
        fn_norm = _normalize_map_values(fn_score)
        sorted_fn = sorted(fn_score.items(), key=lambda kv: kv[1], reverse=True)
        fn_limit = int(max(1, min(int(stage2_top_functions), len(sorted_fn))))
        selected_fn = [x[0] for x in sorted_fn[:fn_limit]]
        selected_fn_count = len(selected_fn)
        support_w = float(max(0.0, stage2_support_weight))
        for fnk in selected_fn:
            file_key, fn_name = fnk
            fn_lines = {
                (f, fn, ln): v
                for (f, fn, ln), v in fn_line_score.items()
                if f == file_key and fn == fn_name
            }
            if not fn_lines:
                continue
            line_norm = _normalize_map_values(fn_lines)
            support_map = {
                lk: float(fn_line_support.get(lk, 0))
                for lk in fn_lines.keys()
            }
            support_norm = _normalize_map_values(support_map)
            fn_weight = float(0.25 + 0.75 * fn_norm.get(fnk, 0.0))
            for lk in fn_lines.keys():
                _, _, ln = lk
                local_strength = float(line_norm.get(lk, 0.0))
                local_support = float(support_norm.get(lk, 0.0))
                bonus = float(stage2_weight * fn_weight * (local_strength + support_w * local_support))
                if bonus <= 0.0:
                    continue
                gk = (file_key, int(ln))
                rec = agg.get(gk)
                if rec is None:
                    continue
                rec["line_risk"] += bonus
                rec["stage2_bonus"] += bonus
                stage2_bonus_sum += bonus

    # Keep stage-2 as a local correction: total bonus is capped by a budget
    # proportional to base line risk mass, so stage-2 cannot overwhelm stage-1.
    base_line_sum_pre = float(sum(float(rec.get("base_line_risk", 0.0)) for rec in agg.values()))
    stage2_bonus_budget = float(max(0.0, float(stage2_weight)) * max(0.0, base_line_sum_pre))
    if stage2_applied and stage2_bonus_sum > 0.0 and stage2_bonus_sum > stage2_bonus_budget + 1e-8:
        stage2_bonus_scale = float(stage2_bonus_budget / (stage2_bonus_sum + 1e-8))
        for rec in agg.values():
            b = float(rec.get("stage2_bonus", 0.0))
            if b <= 0.0:
                continue
            b_scaled = float(b * stage2_bonus_scale)
            rec["stage2_bonus"] = b_scaled
            rec["line_risk"] = float(rec.get("base_line_risk", 0.0)) + b_scaled
        stage2_bonus_sum = float(stage2_bonus_sum * stage2_bonus_scale)

    max_file_share = float(max(0.0, stage2_max_file_share))
    if stage2_applied and stage2_bonus_sum > 0.0 and max_file_share > 0.0:
        per_file_cap = float(max_file_share * stage2_bonus_budget)
        if per_file_cap > 0.0:
            file_bonus: Dict[str, float] = {}
            for rec in agg.values():
                b = float(rec.get("stage2_bonus", 0.0))
                if b <= 0.0:
                    continue
                file_key = str(rec.get("file", ""))
                file_bonus[file_key] = file_bonus.get(file_key, 0.0) + b
            changed = False
            for file_key, total_bonus in file_bonus.items():
                if total_bonus <= per_file_cap + 1e-8:
                    continue
                scale = float(per_file_cap / (total_bonus + 1e-8))
                stage2_file_scale = min(stage2_file_scale, scale)
                changed = True
                for rec in agg.values():
                    if str(rec.get("file", "")) != file_key:
                        continue
                    b = float(rec.get("stage2_bonus", 0.0))
                    if b <= 0.0:
                        continue
                    b_scaled = float(b * scale)
                    rec["stage2_bonus"] = b_scaled
                    rec["line_risk"] = float(rec.get("base_line_risk", 0.0)) + b_scaled
            if changed:
                stage2_bonus_sum = float(sum(float(rec.get("stage2_bonus", 0.0)) for rec in agg.values()))

    out = []
    for _, rec in agg.items():
        c = max(1.0, float(rec["support_slices"]))
        out.append(
            {
                "file": str(rec["file"]),
                "line": int(rec["line"]),
                "line_risk": float(rec["line_risk"]),
                "base_line_risk": float(rec["base_line_risk"]),
                "stage2_bonus": float(rec["stage2_bonus"]),
                "support_slices": int(rec["support_slices"]),
                "max_slice_risk": float(rec["max_slice_risk"]),
                "mean_contrib": float(rec["line_risk"] / c),
            }
        )
    out.sort(key=lambda x: x["line_risk"], reverse=True)
    diag = {
        "topn": int(k),
        "closure_boost": float(closure_boost),
        "stage2_weight": float(stage2_weight),
        "stage2_top_functions": int(stage2_top_functions),
        "stage2_support_weight": float(stage2_support_weight),
        "stage2_max_file_share": float(stage2_max_file_share),
        "stage2_applied": bool(stage2_applied),
        "stage2_candidate_functions": int(len(fn_score)),
        "stage2_selected_functions": int(selected_fn_count),
        "stage2_bonus_sum": float(stage2_bonus_sum),
        "stage2_bonus_budget": float(stage2_bonus_budget),
        "stage2_bonus_scale": float(stage2_bonus_scale),
        "stage2_file_scale": float(stage2_file_scale),
        "base_line_risk_sum": float(sum(float(x.get("base_line_risk", 0.0)) for x in out)),
        "final_line_risk_sum": float(sum(float(x.get("line_risk", 0.0)) for x in out)),
    }
    return out, diag


def _safe_ratio(num: float, den: float) -> float:
    den = float(den)
    if abs(den) <= 1e-8:
        return 0.0
    return float(num) / den


def _count_pipe_items(v: Any) -> int:
    return len([x for x in str(v).split("|") if str(x).strip()])


def _normalize01(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo + 1e-8)).astype(np.float32)


def _fit_scope_mask(rows: List[Dict[str, Any]], scope: str) -> np.ndarray:
    scope_key = str(scope or "all").strip().lower()
    if scope_key == "all":
        return np.ones(len(rows), dtype=bool)
    if scope_key in {"low_risk", "normal"}:
        return np.asarray([str(r.get("status", "")) == "normal" for r in rows], dtype=bool)
    return np.ones(len(rows), dtype=bool)


def build_score_feature_matrix(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    return build_score_feature_matrix_common(rows)


def build_binary_feature_matrix(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), []
    df = pd.DataFrame(rows)
    for key in ("normal", "boundary", "high_risk"):
        df[f"status_{key}"] = (df["status"].astype(str) == key).astype(np.float32)
    df["risk_calib_gap"] = (df["final_risk"] - df["base_final_risk"]).astype(np.float32)
    df["risk_refine_gap"] = (df["base_final_risk"] - df["proto_score"]).astype(np.float32)
    df["proto_view_gap_abs"] = (df["proto_score"] - df["view_score"]).abs().astype(np.float32)
    for col in ("global_gap", "fused_gap"):
        if col not in df.columns:
            df[col] = 0.0
    cols = [
        "final_risk",
        "base_final_risk",
        "boundary_margin",
        "view_gap",
        "global_gap",
        "fused_gap",
        "proto_score",
        "view_score",
        "density_risk",
        "instability",
        "boundary_proximity",
        "adaptive_local_scale",
        "mechanism_risk",
        "family_risk",
        "risk_calib_gap",
        "risk_refine_gap",
        "proto_view_gap_abs",
        "status_normal",
        "status_boundary",
        "status_high_risk",
    ]
    feat = (
        df[cols]
        .astype(np.float32)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    return feat, cols


def apply_score_head(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, Any]]:
    base = np.asarray([float(r.get("final_risk", 0.0)) for r in rows], dtype=np.float32)
    if not rows or str(args.score_head_method).lower() == "none":
        return base, {"enabled": False, "method": "none"}
    return apply_iforest_score_head(
        rows,
        fit_scope=str(args.score_head_fit_scope),
        random_state=int(args.score_head_random_state),
    )


def apply_mechanism_head_score(
    rows: List[Dict[str, Any]],
    dual_rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not rows:
        return np.zeros((0,), dtype=np.float32), {"enabled": False, "method": "mechanism_head", "fit_rows": 0}
    ckpt_path = str(getattr(args, "mechanism_head_ckpt", "") or "").strip()
    if not ckpt_path:
        base = np.asarray([float(r.get("final_risk", 0.0)) for r in rows], dtype=np.float32)
        return base, {
            "enabled": False,
            "method": "mechanism_head",
            "fallback": "missing_checkpoint",
            "fit_rows": int(len(rows)),
        }
    model = load_mechanism_head_model(Path(ckpt_path))
    feature_rows = extract_mechanism_head_feature_rows(
        dual_rows,
        max_hops=int(max(1, getattr(model, "max_hops", 6))),
    )
    sid_to_aux = {
        str(row.get("slice_id", "")): row
        for row in rows
        if str(row.get("slice_id", "")).strip()
    }
    for feature_row in feature_rows:
        sid = str(feature_row.get("slice_id", ""))
        aux = sid_to_aux.get(sid, {})
        feature_row["proto_score"] = float(aux.get("proto_score", 0.0) or 0.0)
        feature_row["view_score"] = float(aux.get("view_score", 0.0) or 0.0)
        feature_row["boundary_margin"] = float(aux.get("boundary_margin", 0.0) or 0.0)
    scored_feature_rows = model.score_rows(feature_rows)
    sid_to_row = {str(row.get("slice_id", "")): row for row in scored_feature_rows if str(row.get("slice_id", "")).strip()}
    score = np.zeros((len(rows),), dtype=np.float32)
    for i, row in enumerate(rows):
        sid = str(row.get("slice_id", ""))
        mrow = sid_to_row.get(sid)
        if mrow is None:
            score[i] = float(row.get("final_risk", 0.0))
            row["mechanism_head_score"] = float(score[i])
            continue
        score[i] = float(mrow.get("anomaly_score", 0.0))
        row["mechanism_head_score"] = float(score[i])
        row["mechanism_head_bridge_gate"] = float(mrow.get("bridge_gate", 0.0))
        row["mechanism_head_mechanism_norm"] = float(mrow.get("mechanism_norm", 0.0))
        row["mechanism_head_closure_norm"] = float(mrow.get("closure_norm", 0.0))
        row["mechanism_head_residual_norm"] = float(mrow.get("residual_norm", 0.0))
        row["mechanism_head_if_norm"] = float(mrow.get("if_norm", 0.0))
        row["mechanism_head_vp_evidence"] = float(mrow.get("vp_evidence", 0.0))
        row["mechanism_head_margin_sig"] = float(mrow.get("margin_sig", 0.0))
        row["mechanism_head_corroboration"] = float(mrow.get("corroboration", 0.0))
        row["mechanism_head_corroboration_factor"] = float(mrow.get("corroboration_factor", 0.0))
        row["mechanism_head_utility_surface"] = float(mrow.get("utility_surface", 0.0))
        row["mechanism_head_utility_corr"] = float(mrow.get("utility_corr", 0.0))
        row["mechanism_head_utility_discount"] = float(mrow.get("utility_discount", 0.0))
        row["mechanism_head_routine_bridge_surface"] = float(mrow.get("routine_bridge_surface", 0.0))
        row["mechanism_head_routine_bridge_discount"] = float(mrow.get("routine_bridge_discount", 1.0))
        row["mechanism_head_compat_wrapper_surface"] = float(mrow.get("compat_wrapper_surface", 0.0))
        row["mechanism_head_compat_wrapper_discount"] = float(mrow.get("compat_wrapper_discount", 1.0))
        row["mechanism_head_threshold"] = float(getattr(model, "slice_threshold", 0.5))
    diag = {
        "enabled": True,
        "method": "mechanism_head",
        "fit_scope": "mechanism_slice_head",
        "fit_rows": int(len(rows)),
        "feature_dim": int(len(getattr(model, "feature_names", []) or [])),
        "feature_names": list(getattr(model, "feature_names", []) or []),
        "slice_threshold": float(getattr(model, "slice_threshold", 0.5)),
        "corroboration_alpha": float(getattr(model, "corroboration_alpha", 0.40)),
        "corroboration_vp_weight": float(getattr(model, "corroboration_vp_weight", 0.80)),
        "corroboration_margin_weight": float(getattr(model, "corroboration_margin_weight", 0.20)),
        "utility_floor": float(getattr(model, "utility_floor", 0.60)),
        "utility_vp_weight": float(getattr(model, "utility_vp_weight", 0.40)),
        "utility_margin_weight": float(getattr(model, "utility_margin_weight", 0.60)),
        "routine_bridge_scale": float(getattr(model, "routine_bridge_scale", 0.55)),
        "routine_bridge_min_surface": float(getattr(model, "routine_bridge_min_surface", 0.60)),
        "routine_bridge_max_vp": float(getattr(model, "routine_bridge_max_vp", 0.18)),
        "routine_bridge_min_gate": float(getattr(model, "routine_bridge_min_gate", 0.70)),
        "routine_bridge_max_utility": float(getattr(model, "routine_bridge_max_utility", 0.25)),
        "compat_wrapper_scale": float(getattr(model, "compat_wrapper_scale", 0.60)),
        "compat_wrapper_min_surface": float(getattr(model, "compat_wrapper_min_surface", 0.70)),
        "compat_wrapper_max_vp": float(getattr(model, "compat_wrapper_max_vp", 0.18)),
        "compat_wrapper_min_utility": float(getattr(model, "compat_wrapper_min_utility", 0.75)),
        "score_mean": float(np.mean(score)) if len(score) else 0.0,
        "score_std": float(np.std(score)) if len(score) else 0.0,
        "train_summary": dict(getattr(model, "train_summary", {}) or {}),
    }
    return score.astype(np.float32), diag


def apply_binary_head(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    energy_embeddings: Optional[np.ndarray] = None,
    energy_graph_embeddings: Optional[np.ndarray] = None,
    energy_text_embeddings: Optional[np.ndarray] = None,
    prototype_centers: Optional[np.ndarray] = None,
    prototype_radii: Optional[np.ndarray] = None,
    prototype_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not rows or str(args.binary_head_method).lower() == "none":
        return np.zeros(len(rows), dtype=np.float32), {"enabled": False, "method": "none"}
    method = str(args.binary_head_method).lower()
    if method == "model_head":
        score = np.asarray([float(r.get("model_binary_score", 0.0)) for r in rows], dtype=np.float32)
        diag = {
            "enabled": True,
            "method": "model_head",
            "fit_scope": "training_model",
            "fit_rows": int(len(rows)),
            "feature_dim": 1,
            "feature_names": ["model_binary_score"],
            "score_mean": float(np.mean(score)) if len(score) else 0.0,
            "score_std": float(np.std(score)) if len(score) else 0.0,
        }
        return score, diag
    if method == "gmm_head":
        ckpt_path = str(getattr(args, "gmm_head_ckpt", "") or "").strip()
        if not ckpt_path:
            base = np.asarray([float(r.get("final_risk", 0.0)) for r in rows], dtype=np.float32)
            diag = {
                "enabled": True,
                "method": "gmm_head",
                "fallback": "missing_checkpoint",
                "fit_rows": 0,
                "feature_dim": 0,
                "score_mean": float(np.mean(base)) if len(base) else 0.0,
                "score_std": float(np.std(base)) if len(base) else 0.0,
            }
            return _normalize01(base).astype(np.float32), diag
        if energy_embeddings is None or len(energy_embeddings) != len(rows):
            base = np.asarray([float(r.get("final_risk", 0.0)) for r in rows], dtype=np.float32)
            diag = {
                "enabled": True,
                "method": "gmm_head",
                "fallback": "missing_embeddings",
                "fit_rows": int(len(rows)),
                "feature_dim": 0,
                "score_mean": float(np.mean(base)) if len(base) else 0.0,
                "score_std": float(np.std(base)) if len(base) else 0.0,
            }
            return _normalize01(base).astype(np.float32), diag
        with Path(ckpt_path).open("rb") as f:
            ckpt = pickle.load(f)
        cfg = dict(ckpt.get("config", {}) or {})
        input_dim = int(cfg.get("input_dim", 0))
        feat = np.asarray(energy_embeddings, dtype=np.float32)
        if feat.ndim != 2 or feat.shape[1] != input_dim or input_dim <= 0:
            base = np.asarray([float(r.get("final_risk", 0.0)) for r in rows], dtype=np.float32)
            diag = {
                "enabled": True,
                "method": "gmm_head",
                "fallback": "feature_dim_mismatch",
                "fit_rows": int(feat.shape[0]) if feat.ndim == 2 else 0,
                "feature_dim": int(feat.shape[1]) if feat.ndim == 2 else 0,
                "expected_dim": int(input_dim),
                "score_mean": float(np.mean(base)) if len(base) else 0.0,
                "score_std": float(np.std(base)) if len(base) else 0.0,
            }
            return _normalize01(base).astype(np.float32), diag
        feat_mean = np.asarray(cfg.get("feature_mean", np.zeros((input_dim,), dtype=np.float32)), dtype=np.float32)
        feat_scale = np.asarray(cfg.get("feature_scale", np.ones((input_dim,), dtype=np.float32)), dtype=np.float32)
        feat_std = standardize_transform(feat, feat_mean, feat_scale)
        gmm = ckpt.get("gmm")
        if gmm is None:
            base = np.asarray([float(r.get("final_risk", 0.0)) for r in rows], dtype=np.float32)
            diag = {
                "enabled": True,
                "method": "gmm_head",
                "fallback": "missing_model",
                "fit_rows": int(feat.shape[0]),
                "feature_dim": int(feat.shape[1]),
                "score_mean": float(np.mean(base)) if len(base) else 0.0,
                "score_std": float(np.std(base)) if len(base) else 0.0,
            }
            return _normalize01(base).astype(np.float32), diag
        neg_log_likelihood = (-gmm.score_samples(feat_std)).astype(np.float32)
        score = normalize01_energy(neg_log_likelihood).astype(np.float32)
        for i, row in enumerate(rows[: len(score)]):
            row["gmm_head_score"] = float(score[i])
            row["binary_score_base"] = float(score[i])
            row["binary_score_adjusted"] = float(score[i])
        diag = {
            "enabled": True,
            "method": "gmm_head",
            "fit_scope": "training_gmm_model",
            "fit_rows": int(len(rows)),
            "feature_dim": int(input_dim),
            "embedding_key": str(cfg.get("embedding_key", "z_joint")),
            "n_components": int(cfg.get("n_components", 0)),
            "covariance_type": str(cfg.get("covariance_type", "")),
            "reg_covar": float(cfg.get("reg_covar", 0.0)),
            "score_mean": float(np.mean(score)) if len(score) else 0.0,
            "score_std": float(np.std(score)) if len(score) else 0.0,
            "nll_mean": float(np.mean(neg_log_likelihood)) if len(neg_log_likelihood) else 0.0,
            "nll_std": float(np.std(neg_log_likelihood)) if len(neg_log_likelihood) else 0.0,
        }
        return score, diag
    if method == "mechanism_head":
        ckpt_path = str(getattr(args, "mechanism_head_ckpt", "") or "").strip()
        score = np.asarray(
            [float(r.get("mechanism_head_score", r.get("final_risk", 0.0))) for r in rows],
            dtype=np.float32,
        )
        if not ckpt_path:
            diag = {
                "enabled": True,
                "method": "mechanism_head",
                "fallback": "missing_checkpoint",
                "fit_rows": int(len(rows)),
                "feature_dim": 0,
                "slice_threshold": 0.5,
                "score_mean": float(np.mean(score)) if len(score) else 0.0,
                "score_std": float(np.std(score)) if len(score) else 0.0,
            }
            return score, diag
        model = load_mechanism_head_model(Path(ckpt_path))
        diag = {
            "enabled": True,
            "method": "mechanism_head",
            "fit_scope": "mechanism_slice_head",
            "fit_rows": int(len(rows)),
            "feature_dim": int(len(getattr(model, "feature_names", []) or [])),
            "feature_names": list(getattr(model, "feature_names", []) or []),
            "slice_threshold": float(getattr(model, "slice_threshold", 0.5)),
            "score_mean": float(np.mean(score)) if len(score) else 0.0,
            "score_std": float(np.std(score)) if len(score) else 0.0,
            "train_summary": dict(getattr(model, "train_summary", {}) or {}),
        }
        for i, row in enumerate(rows[: len(score)]):
            row["binary_score_base"] = float(score[i])
            row["binary_score_adjusted"] = float(score[i])
        return score, diag
    if method == "energy_head":
        ckpt_path = str(getattr(args, "energy_head_ckpt", "") or "").strip()
        if not ckpt_path:
            base = np.asarray([float(r.get("final_risk", 0.0)) for r in rows], dtype=np.float32)
            diag = {
                "enabled": True,
                "method": "energy_head",
                "fallback": "missing_checkpoint",
                "fit_rows": 0,
                "feature_dim": 0,
                "feature_names": [],
                "score_mean": float(np.mean(base)) if len(base) else 0.0,
                "score_std": float(np.std(base)) if len(base) else 0.0,
            }
            return _normalize01(base).astype(np.float32), diag
        ckpt = torch.load(Path(ckpt_path), map_location="cpu")
        cfg = dict(ckpt.get("config", {}) or {})
        input_dim = int(cfg.get("input_dim", 0))
        embed_dim = int(cfg.get("embedding_dim", 0))
        if energy_embeddings is None or len(energy_embeddings) != len(rows):
            energy_embeddings = np.zeros((len(rows), embed_dim), dtype=np.float32)
        graph_embeddings = None
        text_embeddings = None
        if energy_graph_embeddings is not None and len(energy_graph_embeddings) == len(rows):
            graph_embeddings = np.asarray(energy_graph_embeddings, dtype=np.float32)
        if energy_text_embeddings is not None and len(energy_text_embeddings) == len(rows):
            text_embeddings = np.asarray(energy_text_embeddings, dtype=np.float32)
        feat, cols, _ = build_energy_feature_matrix(
            rows,
            np.asarray(energy_embeddings, dtype=np.float32),
            mode=str(cfg.get("mode", "full")),
            include_metamorphic=bool(cfg.get("include_metamorphic_features", False)),
            include_score_head=bool(cfg.get("include_score_head_features", False)),
            include_text_aux_features=bool(cfg.get("include_text_aux_features", False)),
            include_closure_features=bool(cfg.get("include_closure_features", False)),
            text_embeddings=text_embeddings,
            graph_embeddings=graph_embeddings,
            prototype_centers=None if prototype_centers is None else np.asarray(prototype_centers, dtype=np.float32),
            prototype_radii=None if prototype_radii is None else np.asarray(prototype_radii, dtype=np.float32),
            prototype_weights=None if prototype_weights is None else np.asarray(prototype_weights, dtype=np.float32),
            free_energy_temp=float(cfg.get("free_energy_temp", 0.35)),
        )
        feat_diag_cols = {str(name): int(i) for i, name in enumerate(cols)}
        proto_resp_entropy = (
            np.asarray(feat[:, feat_diag_cols["proto_resp_entropy"]], dtype=np.float32)
            if "proto_resp_entropy" in feat_diag_cols
            else np.zeros((len(rows),), dtype=np.float32)
        )
        proto_resp_max = (
            np.asarray(feat[:, feat_diag_cols["proto_resp_max"]], dtype=np.float32)
            if "proto_resp_max" in feat_diag_cols
            else np.zeros((len(rows),), dtype=np.float32)
        )
        proto_resp_gap = (
            np.asarray(feat[:, feat_diag_cols["proto_resp_gap"]], dtype=np.float32)
            if "proto_resp_gap" in feat_diag_cols
            else np.zeros((len(rows),), dtype=np.float32)
        )
        proto_top2_norm_gap = (
            np.asarray(feat[:, feat_diag_cols["proto_top2_norm_gap"]], dtype=np.float32)
            if "proto_top2_norm_gap" in feat_diag_cols
            else np.zeros((len(rows),), dtype=np.float32)
        )
        proto_conf_respmax = normalize01_energy(proto_resp_max).astype(np.float32)
        proto_conf_negent = normalize01_energy(1.0 - proto_resp_entropy).astype(np.float32)
        proto_conf_gap = normalize01_energy(proto_resp_gap).astype(np.float32)
        proto_conf_mix = normalize01_energy(
            0.45 * proto_conf_respmax + 0.35 * proto_conf_negent + 0.20 * proto_conf_gap
        ).astype(np.float32)
        proto_conf_sharp = normalize01_energy(
            0.60 * proto_conf_negent + 0.40 * proto_conf_respmax
        ).astype(np.float32)
        for i, row in enumerate(rows[: len(feat)]):
            row["proto_resp_entropy"] = float(proto_resp_entropy[i])
            row["proto_resp_max"] = float(proto_resp_max[i])
            row["proto_resp_gap"] = float(proto_resp_gap[i])
            row["proto_top2_norm_gap"] = float(proto_top2_norm_gap[i])
            row["proto_conf_respmax"] = float(proto_conf_respmax[i])
            row["proto_conf_negent"] = float(proto_conf_negent[i])
            row["proto_conf_gap"] = float(proto_conf_gap[i])
            row["proto_conf_mix"] = float(proto_conf_mix[i])
            row["proto_conf_sharp"] = float(proto_conf_sharp[i])
        align_diag: Dict[str, Any] = {"enabled": False}
        target_cols = list(cfg.get("feature_names", []) or [])
        if feat.shape[0] > 0 and feat.shape[1] > 0 and int(feat.shape[1]) != input_dim and target_cols:
            feat_aligned, cols_aligned, align_info = align_feature_matrix_columns(feat, cols, target_cols)
            if int(feat_aligned.shape[1]) == input_dim:
                feat = feat_aligned
                cols = cols_aligned
                align_diag = {"enabled": True, **dict(align_info)}
        if feat.shape[0] <= 0 or feat.shape[1] <= 0 or int(feat.shape[1]) != input_dim:
            base = np.asarray([float(r.get("final_risk", 0.0)) for r in rows], dtype=np.float32)
            diag = {
                "enabled": True,
                "method": "energy_head",
                "fallback": "feature_dim_mismatch",
                "fit_rows": int(feat.shape[0]) if feat.ndim == 2 else 0,
                "feature_dim": int(feat.shape[1]) if feat.ndim == 2 else 0,
                "expected_dim": int(input_dim),
                "feature_names": cols,
                "feature_align": align_diag,
                "score_mean": float(np.mean(base)) if len(base) else 0.0,
                "score_std": float(np.std(base)) if len(base) else 0.0,
            }
            return _normalize01(base).astype(np.float32), diag
        feat_mean = np.asarray(cfg.get("feature_mean", np.zeros((input_dim,), dtype=np.float32)), dtype=np.float32)
        feat_scale = np.asarray(cfg.get("feature_scale", np.ones((input_dim,), dtype=np.float32)), dtype=np.float32)
        feat_std = standardize_transform(feat, feat_mean, feat_scale)
        model = EnergyHeadMLP(
            input_dim=input_dim,
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            dropout=float(cfg.get("dropout", 0.10)),
        )
        model.load_state_dict(dict(ckpt.get("model_state_dict", {}) or {}), strict=False)
        model.eval()
        with torch.no_grad():
            energy = model(torch.from_numpy(feat_std).float()).detach().cpu().numpy().astype(np.float32)
        score = normalize01_energy(energy).astype(np.float32)
        base_score = score.astype(np.float32, copy=True)
        score_calibration_diag: Dict[str, Any] = {"enabled": False}
        if bool(getattr(args, "binary_head_score_calibration_enable", False)):
            desired_group_by = str(getattr(args, "binary_head_score_calibration_group_by", "prototype"))
            desired_fit_scope = str(getattr(args, "binary_head_score_calibration_fit_scope", "normal"))
            calibration = None
            calibration_origin = ""
            ckpt_calibration = ckpt.get("score_calibration")
            if isinstance(ckpt_calibration, dict) and bool(ckpt_calibration.get("enabled", False)):
                ckpt_group = str(ckpt_calibration.get("group_by", ""))
                ckpt_scope = str(ckpt_calibration.get("fit_scope", ""))
                if ckpt_group == desired_group_by and ckpt_scope == desired_fit_scope:
                    calibration = ckpt_calibration
                    calibration_origin = "energy_head_ckpt"
            calibration_source = str(getattr(args, "binary_head_score_calibration_source", "") or "").strip()
            if calibration is None and calibration_source:
                cal_path = Path(calibration_source)
                if not cal_path.is_absolute():
                    cal_path = (ROOT / cal_path).resolve()
                calibration = fit_group_score_calibration_from_csv(
                    cal_path,
                    score_column="energy_head_score",
                    fit_scope=desired_fit_scope,
                    group_by=desired_group_by,
                    min_samples=int(getattr(args, "binary_head_score_calibration_min_samples", 32)),
                    shrinkage_tau=float(getattr(args, "binary_head_score_calibration_shrinkage_tau", 16.0)),
                    global_mix_floor=float(getattr(args, "binary_head_score_calibration_global_mix_floor", 0.20)),
                    quantile_bins=int(getattr(args, "binary_head_score_calibration_quantile_bins", 257)),
                )
                calibration_origin = str(calibration.get("diag", {}).get("source", cal_path))
            if calibration is None and isinstance(ckpt_calibration, dict) and bool(ckpt_calibration.get("enabled", False)):
                calibration = ckpt_calibration
                calibration_origin = "energy_head_ckpt:fallback"
            if calibration is not None and bool(calibration.get("enabled", False)):
                score, score_calibration_diag = apply_group_score_calibration(
                    rows[: len(base_score)],
                    base_score,
                    calibration,
                    alpha=float(getattr(args, "binary_head_score_calibration_alpha", 0.40)),
                    min_samples=int(getattr(args, "binary_head_score_calibration_min_samples", 32)),
                    shrinkage_tau=float(getattr(args, "binary_head_score_calibration_shrinkage_tau", 16.0)),
                    global_mix_floor=float(getattr(args, "binary_head_score_calibration_global_mix_floor", 0.20)),
                )
                score_calibration_diag = {
                    **dict(calibration.get("diag", {}) or {}),
                    **dict(score_calibration_diag or {}),
                    "origin": calibration_origin or "unknown",
                }
            else:
                score_calibration_diag = {
                    "enabled": False,
                    "origin": calibration_origin or "unavailable",
                    "missing_reason": ""
                    if calibration is None
                    else str(calibration.get("missing_reason", "")),
                }
        for i, row in enumerate(rows[: len(base_score)]):
            row["binary_score_base"] = float(base_score[i])
            row["binary_score_adjusted"] = float(score[i])
        diag = {
            "enabled": True,
            "method": "energy_head",
            "fit_scope": "training_energy_model",
            "fit_rows": int(len(rows)),
            "feature_dim": int(feat.shape[1]),
            "embedding_dim": int(embed_dim),
            "feature_names": cols,
            "feature_align": align_diag,
            "score_mean": float(np.mean(score)) if len(score) else 0.0,
            "score_std": float(np.std(score)) if len(score) else 0.0,
            "score_calibration": score_calibration_diag,
        }
        return score, diag
    feat, cols = build_binary_feature_matrix(rows)
    scaler = StandardScaler()
    feat_std = scaler.fit_transform(feat)
    fit_mask = _fit_scope_mask(rows, args.binary_head_fit_scope)
    if not np.any(fit_mask):
        fit_mask = np.ones(len(rows), dtype=bool)
    if method == "ocsvm":
        nu = float(min(0.5, max(1e-4, float(args.binary_head_contamination))))
        clf = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
        clf.fit(feat_std[fit_mask])
        score = -clf.decision_function(feat_std)
        score = _normalize01(score)
        diag = {
            "enabled": True,
            "method": "ocsvm",
            "fit_scope": str(args.binary_head_fit_scope),
            "fit_rows": int(np.sum(fit_mask)),
            "feature_dim": int(feat.shape[1]),
            "feature_names": cols,
            "contamination": float(args.binary_head_contamination),
            "score_mean": float(np.mean(score)) if len(score) else 0.0,
            "score_std": float(np.std(score)) if len(score) else 0.0,
        }
        return score.astype(np.float32), diag

    if method == "selftrain_lr":
        risk = np.asarray([float(r.get("final_risk", 0.0)) for r in rows], dtype=np.float32)
        status = np.asarray([str(r.get("status", "")) for r in rows], dtype=object)
        neg_q = float(min(0.9, max(0.0, args.binary_head_pseudo_neg_quantile)))
        pos_q = float(min(0.999, max(neg_q + 1e-3, args.binary_head_pseudo_pos_quantile)))
        neg_thr = float(np.quantile(risk, neg_q)) if len(risk) else 0.0
        pos_thr = float(np.quantile(risk, pos_q)) if len(risk) else 1.0
        neg_mask = (status == "normal") & (risk <= neg_thr)
        pos_mask = (status == "high_risk") & (risk >= pos_thr)
        # Fall back to pure quantile anchors when status-gated anchors are too sparse.
        if int(np.sum(neg_mask)) < 32:
            neg_mask = risk <= neg_thr
        if int(np.sum(pos_mask)) < 32:
            pos_mask = risk >= pos_thr
        train_mask = neg_mask | pos_mask
        y = np.zeros(int(np.sum(train_mask)), dtype=np.int64)
        if np.any(pos_mask):
            pos_idx = np.flatnonzero(train_mask & pos_mask)
            train_idx = np.flatnonzero(train_mask)
            pos_lookup = {int(idx): i for i, idx in enumerate(train_idx)}
            for idx in pos_idx:
                y[pos_lookup[int(idx)]] = 1
        train_x = feat_std[train_mask]
        unique = np.unique(y)
        if train_x.shape[0] < 64 or len(unique) < 2:
            diag = {
                "enabled": True,
                "method": "selftrain_lr",
                "fallback": "insufficient_pseudo_labels",
                "fit_rows": int(train_x.shape[0]),
                "positive_rows": int(np.sum(y == 1)),
                "negative_rows": int(np.sum(y == 0)),
                "feature_dim": int(feat.shape[1]),
                "feature_names": cols,
                "score_mean": float(np.mean(risk)) if len(risk) else 0.0,
                "score_std": float(np.std(risk)) if len(risk) else 0.0,
            }
            return _normalize01(risk).astype(np.float32), diag
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=int(args.binary_head_random_state),
        )
        clf.fit(train_x, y)
        score = clf.predict_proba(feat_std)[:, 1].astype(np.float32)
        diag = {
            "enabled": True,
            "method": "selftrain_lr",
            "fit_scope": "pseudo_labels",
            "fit_rows": int(train_x.shape[0]),
            "positive_rows": int(np.sum(y == 1)),
            "negative_rows": int(np.sum(y == 0)),
            "pseudo_neg_quantile": float(neg_q),
            "pseudo_pos_quantile": float(pos_q),
            "pseudo_neg_threshold": float(neg_thr),
            "pseudo_pos_threshold": float(pos_thr),
            "feature_dim": int(feat.shape[1]),
            "feature_names": cols,
            "score_mean": float(np.mean(score)) if len(score) else 0.0,
            "score_std": float(np.std(score)) if len(score) else 0.0,
        }
        return score, diag

    raise ValueError(f"Unsupported binary head method: {args.binary_head_method}")


def build_binary_targets(
    rows: List[Dict[str, Any]],
    label_df: pd.DataFrame,
    mode: str = "file_function",
    line_window: int = 2,
) -> np.ndarray:
    label_rows = []
    label_fn_map: Dict[str, Set[str]] = {}
    label_line_map: Dict[str, List[int]] = {}
    for _, r in label_df.iterrows():
        file_key = normalize_path(str(r["file"]))
        fn = None if pd.isna(r["function"]) else str(r["function"]).strip().lower()
        line = int(r["line"])
        label_rows.append({"file": file_key, "function": fn, "line": line})
        if fn:
            label_fn_map.setdefault(file_key, set()).add(fn)
        label_line_map.setdefault(file_key, []).append(line)

    mode_key = str(mode or "file_function").strip().lower()
    y = np.zeros(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        file_key = normalize_path(str(row.get("file", "")))
        fn_match = bool(parse_fn_set(str(row.get("function_names", ""))) & label_fn_map.get(file_key, set()))
        if mode_key == "file_function":
            y[i] = int(fn_match)
            continue
        line_set = parse_line_set(str(row.get("line_candidates", "")))
        row_match = False
        for ln in label_line_map.get(file_key, []):
            if line_window <= 0:
                if int(ln) in line_set:
                    row_match = True
                    break
            else:
                if any((int(ln) + d) in line_set for d in range(-int(line_window), int(line_window) + 1)):
                    row_match = True
                    break
        if mode_key == "row_window2":
            y[i] = int(row_match)
        else:
            y[i] = int(fn_match or row_match)
    return y


def apply_triage_predictions(
    rows: List[Dict[str, Any]],
    score_risk: np.ndarray,
    binary_score: np.ndarray,
    args: argparse.Namespace,
    binary_head_diag: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    if len(rows) == 0:
        return np.zeros(0, dtype=np.int64), [], {
            "enabled": False,
            "low_quantile": float(args.triage_low_quantile),
            "high_quantile": float(args.triage_high_quantile),
            "gray_threshold": float(args.triage_gray_threshold),
        }
    method = str(getattr(args, "binary_head_method", "none") or "none").strip().lower()
    if method == "mechanism_head":
        direct_thr = float((binary_head_diag or {}).get("slice_threshold", 0.5))
        pred = (np.asarray(binary_score, dtype=np.float32) >= direct_thr).astype(np.int64)
        region = ["high" if int(v) == 1 else "low" for v in pred.tolist()]
        return pred, region, {
            "enabled": True,
            "mode": "mechanism_head_threshold",
            "slice_threshold": float(direct_thr),
            "low_count": int(np.sum(pred == 0)),
            "high_count": int(np.sum(pred == 1)),
            "gray_count": 0,
        }
    low_q = float(min(1.0, max(0.0, args.triage_low_quantile)))
    high_q = float(min(1.0, max(low_q, args.triage_high_quantile)))
    low_thr = float(np.quantile(score_risk, low_q)) if len(score_risk) else 0.0
    high_thr = float(np.quantile(score_risk, high_q)) if len(score_risk) else 0.0
    gray_thr = float(min(1.0, max(0.0, args.triage_gray_threshold)))
    pred = np.zeros(len(rows), dtype=np.int64)
    region: List[str] = []
    low_cnt = 0
    gray_cnt = 0
    high_cnt = 0
    family_aware_gray = bool(getattr(args, "triage_family_aware_gray", False))
    family_min_samples = int(max(4, getattr(args, "triage_family_min_samples", 32)))
    adaptive_gray = bool(getattr(args, "triage_adaptive_gray", False))
    closure_boost_enable = bool(getattr(args, "triage_closure_boost_enable", False))
    closure_boost_value = float(max(0.0, getattr(args, "triage_closure_boost_value", 0.10)))
    closure_boost_min_high = int(max(1, getattr(args, "triage_closure_boost_min_high", 2)))
    closure_boost_cap = float(min(1.0, max(0.0, getattr(args, "triage_closure_boost_cap", 1.0))))
    for i in range(len(rows)):
        if float(score_risk[i]) < low_thr:
            region.append("low")
            pred[i] = 0
            low_cnt += 1
        elif float(score_risk[i]) > high_thr:
            region.append("high")
            pred[i] = 1
            high_cnt += 1
        else:
            region.append("gray")
            gray_cnt += 1
    gray_threshold_by_row = np.full((len(rows),), gray_thr, dtype=np.float32)
    family_local_count = 0
    family_fallback_rows = 0
    gray_quantile = 0.0
    adaptive_source = str(getattr(args, "triage_adaptive_gray_source", "") or "").strip()
    adaptive_column = str(getattr(args, "triage_adaptive_gray_column", "energy_head_score") or "energy_head_score").strip()
    adaptive_group_by = str(getattr(args, "triage_adaptive_gray_group_by", "family") or "family").strip().lower()
    adaptive_loaded = False
    adaptive_global_threshold = gray_thr
    adaptive_source_rows = 0
    adaptive_source_families = 0
    adaptive_source_missing = ""
    gray_idx = np.flatnonzero(np.asarray(region, dtype=object) == "gray")
    if len(gray_idx) > 0:
        gray_scores = np.asarray(binary_score[gray_idx], dtype=np.float32)
        fam_to_idx: Dict[str, List[int]] = {}
        for idx in gray_idx.tolist():
            grp = _adaptive_gray_group_key(rows[idx], adaptive_group_by)
            fam_to_idx.setdefault(grp, []).append(int(idx))
        if adaptive_gray and adaptive_source:
            adaptive = _load_adaptive_gray_distribution(
                source_csv=adaptive_source,
                score_column=adaptive_column,
                base_threshold=gray_thr,
                group_by=adaptive_group_by,
            )
            adaptive_loaded = bool(adaptive.get("loaded", False))
            adaptive_source_missing = str(adaptive.get("missing_reason", "") or "")
            adaptive_global_threshold = float(adaptive.get("global_threshold", gray_thr))
            adaptive_source_rows = int(adaptive.get("source_rows", 0))
            adaptive_source_families = int(adaptive.get("source_families", 0))
            gray_quantile = float(adaptive.get("quantile", 0.0))
            gray_threshold_by_row[:] = float(adaptive_global_threshold)
            if family_aware_gray and adaptive_loaded:
                family_thresholds = adaptive.get("family_thresholds", {})
                family_counts = adaptive.get("family_counts", {})
                for fam, idx_list in fam_to_idx.items():
                    fam_count = int(family_counts.get(fam, 0))
                    if fam_count < family_min_samples:
                        family_fallback_rows += int(len(idx_list))
                        continue
                    local_thr = float(family_thresholds.get(fam, adaptive_global_threshold))
                    gray_threshold_by_row[np.asarray(idx_list, dtype=np.int64)] = local_thr
                    family_local_count += 1
        else:
            gray_quantile = float(np.mean(gray_scores <= gray_thr))
            if family_aware_gray and len(gray_scores) > 0:
                for idx_list in fam_to_idx.values():
                    if len(idx_list) < family_min_samples:
                        family_fallback_rows += int(len(idx_list))
                        continue
                    idx_arr = np.asarray(idx_list, dtype=np.int64)
                    local_scores = np.asarray(binary_score[idx_arr], dtype=np.float32)
                    local_thr = float(np.quantile(local_scores, gray_quantile))
                    gray_threshold_by_row[idx_arr] = local_thr
                    family_local_count += 1
    gray_score_eff = np.asarray(binary_score, dtype=np.float32).copy()
    closure_boost_rows = 0
    closure_boost_groups = 0
    closure_boost_total_delta = 0.0
    closure_high_group_counts: Dict[str, int] = {}
    for i, tag in enumerate(region):
        if str(tag) != "high":
            continue
        fkey = _triage_function_group_key(rows[i])
        closure_high_group_counts[fkey] = closure_high_group_counts.get(fkey, 0) + 1
    if closure_boost_enable and len(gray_idx) > 0 and closure_boost_value > 0.0:
        eligible_groups = {k for k, v in closure_high_group_counts.items() if int(v) >= int(closure_boost_min_high)}
        closure_boost_groups = int(len(eligible_groups))
        if eligible_groups:
            for i in gray_idx.tolist():
                fkey = _triage_function_group_key(rows[i])
                if fkey not in eligible_groups:
                    continue
                old_score = float(gray_score_eff[i])
                new_score = float(min(closure_boost_cap, old_score + closure_boost_value))
                if new_score > old_score + 1e-12:
                    gray_score_eff[i] = np.float32(new_score)
                    closure_boost_rows += 1
                    closure_boost_total_delta += float(new_score - old_score)
    for i in gray_idx.tolist():
        pred[i] = int(float(gray_score_eff[i]) >= float(gray_threshold_by_row[i]))
    diag = {
        "enabled": True,
        "cascade_mode": "two_stage",
        "low_quantile": low_q,
        "high_quantile": high_q,
        "low_threshold": low_thr,
        "high_threshold": high_thr,
        "gray_threshold": gray_thr,
        "family_aware_gray": bool(family_aware_gray),
        "family_min_samples": int(family_min_samples),
        "adaptive_gray": bool(adaptive_gray),
        "adaptive_gray_source": adaptive_source,
        "adaptive_gray_column": adaptive_column,
        "adaptive_gray_group_by": adaptive_group_by,
        "adaptive_gray_loaded": bool(adaptive_loaded),
        "adaptive_gray_missing_reason": adaptive_source_missing,
        "adaptive_gray_global_threshold": float(adaptive_global_threshold),
        "adaptive_gray_source_rows": int(adaptive_source_rows),
        "adaptive_gray_source_families": int(adaptive_source_families),
        "gray_quantile": float(gray_quantile),
        "family_local_count": int(family_local_count),
        "family_fallback_rows": int(family_fallback_rows),
        "closure_boost_enable": bool(closure_boost_enable),
        "closure_boost_value": float(closure_boost_value),
        "closure_boost_min_high": int(closure_boost_min_high),
        "closure_boost_cap": float(closure_boost_cap),
        "closure_boost_groups": int(closure_boost_groups),
        "closure_boost_rows": int(closure_boost_rows),
        "closure_boost_mean_delta": float(closure_boost_total_delta / max(1, closure_boost_rows)),
        "region_counts": {"low": int(low_cnt), "gray": int(gray_cnt), "high": int(high_cnt)},
    }
    return pred, region, diag


def _load_adaptive_gray_distribution(
    source_csv: str,
    score_column: str,
    base_threshold: float,
    group_by: str = "family",
) -> Dict[str, Any]:
    path = Path(source_csv)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        return {
            "loaded": False,
            "missing_reason": f"missing_source:{path}",
            "global_threshold": float(base_threshold),
            "quantile": 0.0,
            "family_thresholds": {},
            "family_counts": {},
            "source_rows": 0,
            "source_families": 0,
        }
    scores: List[float] = []
    fam_scores: Dict[str, List[float]] = {}
    source_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if score_column not in (reader.fieldnames or []):
            return {
                "loaded": False,
                "missing_reason": f"missing_column:{score_column}",
                "global_threshold": float(base_threshold),
                "quantile": 0.0,
                "family_thresholds": {},
                "family_counts": {},
                "source_rows": 0,
                "source_families": 0,
            }
        for row in reader:
            try:
                score = float(row.get(score_column, ""))
            except Exception:
                continue
            if not np.isfinite(score):
                continue
            source_rows += 1
            scores.append(score)
            fam = _adaptive_gray_group_key(row, group_by)
            fam_scores.setdefault(fam, []).append(score)
    if not scores:
        return {
            "loaded": False,
            "missing_reason": "no_valid_scores",
            "global_threshold": float(base_threshold),
            "quantile": 0.0,
            "family_thresholds": {},
            "family_counts": {},
            "source_rows": int(source_rows),
            "source_families": 0,
        }
    score_arr = np.asarray(scores, dtype=np.float32)
    quantile = float(np.mean(score_arr <= float(base_threshold)))
    quantile = float(min(0.995, max(0.005, quantile)))
    global_threshold = float(np.quantile(score_arr, quantile))
    family_thresholds = {
        fam: float(np.quantile(np.asarray(vals, dtype=np.float32), quantile))
        for fam, vals in fam_scores.items()
        if vals
    }
    family_counts = {fam: int(len(vals)) for fam, vals in fam_scores.items()}
    return {
        "loaded": True,
        "missing_reason": "",
        "global_threshold": float(global_threshold),
        "quantile": float(quantile),
        "family_thresholds": family_thresholds,
        "family_counts": family_counts,
        "source_rows": int(source_rows),
        "source_families": int(len(family_thresholds)),
    }


def _adaptive_gray_group_key(row: Dict[str, Any], group_by: str) -> str:
    mode = str(group_by or "family").strip().lower()
    if mode == "prototype":
        pid = row.get("prototype_id")
        if pid is None or str(pid).strip() == "":
            return "<unknown_prototype>"
        return f"proto:{pid}"
    file_key = str(
        row.get("file")
        or row.get("relative_source_path")
        or row.get("source_path")
        or row.get("contract_id")
        or "<unknown_file>"
    )
    return _family_key_from_file(file_key)


def _embedding_array_from_map(
    emb_map: Dict[str, np.ndarray],
    embedding_key: str,
    hybrid_alpha: float,
) -> np.ndarray:
    key = str(embedding_key or "z_joint").strip().lower()
    if key == "z_hybrid":
        z_joint = np.asarray(emb_map.get("z_joint", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        z_fused = np.asarray(emb_map.get("z_fused", np.zeros_like(z_joint)), dtype=np.float32)
        if len(z_joint) and len(z_fused) == len(z_joint):
            alpha = float(min(1.0, max(0.0, hybrid_alpha)))
            return (alpha * z_fused + (1.0 - alpha) * z_joint).astype(np.float32)
        return z_joint
    if key in emb_map:
        return np.asarray(emb_map[key], dtype=np.float32)
    return np.asarray(emb_map.get("z_joint", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)


def apply_proto_resp_gray_adjustment(
    rows: List[Dict[str, Any]],
    region: Sequence[str],
    binary_score: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    base_score = np.asarray(binary_score, dtype=np.float32)
    if not bool(getattr(args, "proto_resp_gray_adjust_enable", False)) or len(rows) == 0:
        return base_score, {"enabled": False}

    gray_idx = np.asarray([i for i, tag in enumerate(region) if str(tag) == "gray"], dtype=np.int64)
    if gray_idx.size <= 0:
        return base_score, {"enabled": True, "selected_rows": 0, "adjusted_rows": 0}

    mode = str(getattr(args, "proto_resp_gray_adjust_mode", "negent") or "negent").strip().lower()
    beta = float(max(0.0, getattr(args, "proto_resp_gray_adjust_beta", 0.04)))
    gated = bool(getattr(args, "proto_resp_gray_adjust_gated", True))
    gate_center = float(max(1e-4, getattr(args, "proto_resp_gray_adjust_gate_center", 0.50)))

    if mode == "respmax":
        conf = np.asarray([float(r.get("proto_conf_respmax", 0.0)) for r in rows], dtype=np.float32)
    elif mode == "mix":
        conf = np.asarray([float(r.get("proto_conf_mix", 0.0)) for r in rows], dtype=np.float32)
    elif mode == "sharp":
        conf = np.asarray([float(r.get("proto_conf_sharp", 0.0)) for r in rows], dtype=np.float32)
    else:
        conf = np.asarray([float(r.get("proto_conf_negent", 0.0)) for r in rows], dtype=np.float32)
    conf = np.clip(conf, 0.0, 1.0).astype(np.float32)

    if float(np.max(conf[gray_idx]) - np.min(conf[gray_idx])) <= 1e-8:
        return base_score, {
            "enabled": True,
            "mode": mode,
            "selected_rows": int(gray_idx.size),
            "adjusted_rows": 0,
            "available": False,
        }

    penalty = beta * (1.0 - conf[gray_idx]).astype(np.float32)
    gate = np.ones_like(penalty, dtype=np.float32)
    if gated:
        gate = np.clip((gate_center - base_score[gray_idx]) / gate_center, 0.0, 1.0).astype(np.float32)
        penalty = penalty * gate

    out_score = base_score.copy()
    out_score[gray_idx] = np.clip(out_score[gray_idx] - penalty, 0.0, 1.0).astype(np.float32)
    adjusted_rows = int(np.sum(penalty > 1e-12))
    for idx in gray_idx.tolist():
        rows[idx]["binary_score_base"] = float(base_score[idx])
        rows[idx]["binary_score_adjusted"] = float(out_score[idx])
        rows[idx]["proto_confidence"] = float(conf[idx])
    diag = {
        "enabled": True,
        "mode": mode,
        "beta": float(beta),
        "gated": bool(gated),
        "gate_center": float(gate_center),
        "selected_rows": int(gray_idx.size),
        "adjusted_rows": int(adjusted_rows),
        "confidence_mean": float(np.mean(conf[gray_idx])) if gray_idx.size else 0.0,
        "confidence_std": float(np.std(conf[gray_idx])) if gray_idx.size else 0.0,
        "penalty_mean": float(np.mean(penalty)) if penalty.size else 0.0,
        "penalty_max": float(np.max(penalty)) if penalty.size else 0.0,
        "gate_mean": float(np.mean(gate)) if gate.size else 0.0,
        "score_mean_before": float(np.mean(base_score[gray_idx])) if gray_idx.size else 0.0,
        "score_mean_after": float(np.mean(out_score[gray_idx])) if gray_idx.size else 0.0,
    }
    return out_score.astype(np.float32), diag


def apply_metamorphic_gray_adjustment(
    rows: List[Dict[str, Any]],
    dual_rows: List[Dict[str, Any]],
    region: Sequence[str],
    score_risk: np.ndarray,
    binary_score: np.ndarray,
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    global_graph_dir: Path,
    centers: np.ndarray,
    radii: np.ndarray,
    embedding_key: str,
    hybrid_alpha: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    base_score = np.asarray(binary_score, dtype=np.float32)
    base_risk = np.asarray(score_risk, dtype=np.float32)
    if not bool(args.metamorphic_gray_enable) or len(rows) == 0:
        return base_score, base_risk, {"enabled": False}

    gray_idx = [i for i, x in enumerate(region) if str(x) == "gray"]
    if not gray_idx:
        return base_score, base_risk, {"enabled": True, "selected_rows": 0, "variant_rows": 0}

    gray_thr = float(min(1.0, max(0.0, args.triage_gray_threshold)))
    gray_idx.sort(key=lambda i: abs(float(base_score[i]) - gray_thr))
    max_rows = int(max(0, args.metamorphic_gray_max_rows))
    if max_rows > 0:
        gray_idx = gray_idx[: min(max_rows, len(gray_idx))]

    dual_row_map = {str(r.get("slice_id", "")): r for r in dual_rows}
    selected_rows: List[Tuple[int, Dict[str, Any]]] = []
    for i in gray_idx:
        sid = str(rows[i].get("slice_id", ""))
        drow = dual_row_map.get(sid)
        if drow is None:
            continue
        selected_rows.append((i, drow))
    if not selected_rows:
        return base_score, base_risk, {
            "enabled": True,
            "selected_rows": 0,
            "variant_rows": 0,
            "skipped_missing_dual_row": int(len(gray_idx)),
        }

    variant_rows: List[Dict[str, Any]] = []
    owner_idx: List[int] = []
    owner_kind: List[str] = []
    for row_idx, drow in selected_rows:
        full_graph = drow.get("full_graph", {}) or {}
        skel_graph = drow.get("skeleton_graph", {}) or {}
        try:
            variants = build_dual_metamorphic_variants(
                full_graph=full_graph,
                skeleton_graph=skel_graph,
                max_hops=int(max(1, args.metamorphic_max_hops)),
            )
        except Exception:
            variants = []
        for variant_name, full_v, skel_v in variants:
            vrow = dict(drow)
            vrow["slice_id"] = f"{drow.get('slice_id')}::meta::{variant_name}"
            vrow["full_graph"] = full_v
            vrow["skeleton_graph"] = skel_v
            variant_rows.append(vrow)
            owner_idx.append(int(row_idx))
            owner_kind.append(str(variant_name))

    if not variant_rows:
        return base_score, base_risk, {
            "enabled": True,
            "selected_rows": int(len(selected_rows)),
            "variant_rows": 0,
        }

    emb_map_var, _ = compute_embeddings(
        model,
        variant_rows,
        device=device,
        global_graph_dir=global_graph_dir,
    )
    proto_emb_var = _embedding_array_from_map(emb_map_var, embedding_key=embedding_key, hybrid_alpha=hybrid_alpha)
    if len(proto_emb_var) != len(variant_rows):
        return base_score, base_risk, {
            "enabled": True,
            "selected_rows": int(len(selected_rows)),
            "variant_rows": int(len(variant_rows)),
            "fallback": "embedding_count_mismatch",
        }

    assign_var, nearest_var = nearest_to_centers(proto_emb_var, centers)
    margin_var = nearest_var - radii[assign_var]
    zf_var = np.asarray(emb_map_var.get("z_full", np.zeros_like(proto_emb_var)), dtype=np.float32)
    zs_var = np.asarray(emb_map_var.get("z_skeleton", np.zeros_like(proto_emb_var)), dtype=np.float32)
    view_gap_var = np.linalg.norm(zf_var - zs_var, axis=1).astype(np.float32) if len(zf_var) else np.zeros((len(variant_rows),), dtype=np.float32)

    near_delta = np.zeros((len(selected_rows),), dtype=np.float32)
    margin_delta = np.zeros((len(selected_rows),), dtype=np.float32)
    view_delta = np.zeros((len(selected_rows),), dtype=np.float32)
    switch_rate = np.zeros((len(selected_rows),), dtype=np.float32)
    counts = np.zeros((len(selected_rows),), dtype=np.float32)
    pos_map = {int(row_idx): j for j, (row_idx, _) in enumerate(selected_rows)}

    for j, row_idx in enumerate(owner_idx):
        pos = pos_map[int(row_idx)]
        row = rows[int(row_idx)]
        near0 = float(row.get("nearest_distance", 0.0))
        margin0 = float(row.get("boundary_margin", 0.0))
        view0 = float(row.get("view_gap", 0.0))
        proto0 = int(row.get("prototype_id", -1))
        near_delta[pos] += abs(float(nearest_var[j]) - near0) / max(1e-6, abs(near0) + 0.10)
        margin_delta[pos] += abs(float(margin_var[j]) - margin0) / max(1e-6, abs(margin0) + 0.10)
        view_delta[pos] += abs(float(view_gap_var[j]) - view0) / max(1e-6, abs(view0) + 0.10)
        switch_rate[pos] += float(int(assign_var[j]) != proto0)
        counts[pos] += 1.0

    valid = counts > 0.0
    near_delta[valid] /= counts[valid]
    margin_delta[valid] /= counts[valid]
    view_delta[valid] /= counts[valid]
    switch_rate[valid] /= counts[valid]

    near_n = _normalize01(near_delta)
    margin_n = _normalize01(margin_delta)
    view_n = _normalize01(view_delta)
    switch_n = _normalize01(switch_rate)
    instability = (
        0.35 * near_n
        + 0.30 * margin_n
        + 0.20 * view_n
        + 0.15 * switch_n
    ).astype(np.float32)

    out_score = base_score.copy()
    out_risk = base_risk.copy()
    score_w = float(max(0.0, args.metamorphic_gray_score_weight))
    risk_w = float(max(0.0, args.metamorphic_gray_risk_weight))
    for j, (row_idx, _) in enumerate(selected_rows):
        i = int(row_idx)
        out_score[i] = float(min(1.0, max(0.0, out_score[i] + score_w * instability[j])))
        out_risk[i] = float(max(0.0, out_risk[i] + risk_w * instability[j]))
        rows[i]["metamorphic_instability"] = float(instability[j])
        rows[i]["metamorphic_nearest_delta"] = float(near_delta[j])
        rows[i]["metamorphic_margin_delta"] = float(margin_delta[j])
        rows[i]["metamorphic_view_delta"] = float(view_delta[j])
        rows[i]["metamorphic_proto_switch_rate"] = float(switch_rate[j])
        rows[i]["metamorphic_variant_count"] = int(counts[j])
    for i, r in enumerate(rows):
        if "metamorphic_instability" not in r:
            r["metamorphic_instability"] = 0.0
            r["metamorphic_nearest_delta"] = 0.0
            r["metamorphic_margin_delta"] = 0.0
            r["metamorphic_view_delta"] = 0.0
            r["metamorphic_proto_switch_rate"] = 0.0
            r["metamorphic_variant_count"] = 0
        r["binary_score_base"] = float(base_score[i])
        r["binary_score_adjusted"] = float(out_score[i])
        r["final_risk_base"] = float(base_risk[i])
        r["final_risk_adjusted"] = float(out_risk[i])

    diag = {
        "enabled": True,
        "selected_rows": int(len(selected_rows)),
        "variant_rows": int(len(variant_rows)),
        "score_weight": float(score_w),
        "risk_weight": float(risk_w),
        "max_hops": int(max(1, args.metamorphic_max_hops)),
        "instability_mean": float(np.mean(instability)) if len(instability) else 0.0,
        "instability_std": float(np.std(instability)) if len(instability) else 0.0,
        "nearest_delta_mean": float(np.mean(near_delta)) if len(near_delta) else 0.0,
        "margin_delta_mean": float(np.mean(margin_delta)) if len(margin_delta) else 0.0,
        "view_delta_mean": float(np.mean(view_delta)) if len(view_delta) else 0.0,
        "proto_switch_mean": float(np.mean(switch_rate)) if len(switch_rate) else 0.0,
    }
    return out_score, out_risk, diag


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    if len(y_true) == 0:
        return {
            "count": 0,
            "positive_rate": 0.0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "fpr": 0.0,
            "f1": 0.0,
        }
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    fpr = _safe_ratio(fp, fp + tn)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    return {
        "count": int(len(y_true)),
        "positive_rate": float(np.mean(y_true.astype(np.float32))) if len(y_true) else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "f1": float(f1),
    }


def compute_line_recall(
    line_rows: List[Dict[str, Any]],
    label_df: pd.DataFrame,
    topk_list: List[int],
    line_window: int = 0,
) -> Dict[str, Any]:
    label_rows = []
    for _, r in label_df.iterrows():
        file_key = normalize_path(str(r["file"]))
        line = int(r["line"])
        label_rows.append({"file": file_key, "line": line})

    output: Dict[str, Any] = {
        "label_counts": {"row88_total": int(len(label_rows))},
        "topk_metrics": [],
    }
    for k in topk_list:
        top = line_rows[: min(int(k), len(line_rows))]
        pred = [(normalize_path(str(x["file"])), int(x["line"])) for x in top]
        hit = 0
        for lab in label_rows:
            matched = False
            for pf, pl in pred:
                if pf != lab["file"]:
                    continue
                if abs(int(pl) - int(lab["line"])) <= int(line_window):
                    matched = True
                    break
            if matched:
                hit += 1
        rec = float(hit) / float(len(label_rows)) if label_rows else 0.0
        output["topk_metrics"].append(
            {
                "top_k": int(len(top)),
                "row88_hit": int(hit),
                "row88_recall": float(rec),
            }
        )
    return output


def _canonical_fn(function_names: str) -> str:
    toks = [x.strip().lower() for x in str(function_names).split("|") if x.strip()]
    return toks[0] if toks else "<unknown_fn>"


def _function_key_set(function_names: str) -> Set[str]:
    toks = {x.strip().lower() for x in str(function_names).split("|") if x.strip()}
    return toks if toks else {"<unknown_fn>"}


def _family_key_from_file(file_key: str) -> str:
    p = str(file_key).replace("\\", "/").strip().lower()
    toks = [t for t in p.split("/") if t and t not in {".", ".."}]
    if not toks:
        return "<unknown_family>"
    generic = {
        "data",
        "dataset",
        "datasets",
        "manually-labeled dataset",
        "real_attack_dataset_format",
        "manually-labeled",
        "manual",
        "manual_eval_artifacts",
        "contracts",
        "contract",
        "src",
        "source",
        "sol",
        "bridge",
        "bridges",
        "project",
        "projects",
        "mainnet",
        "testnet",
    }
    core = [t for t in toks[:-1] if not t.endswith(".sol") and t not in generic]
    if len(core) >= 2:
        return f"{core[0]}/{core[1]}"
    if len(core) == 1:
        return core[0]
    stem = toks[-1]
    if stem.endswith(".sol"):
        stem = stem[:-4]
    return stem or "<unknown_family>"


def coverage_rerank_rows(
    rows: List[Dict[str, Any]],
    score_key: str = "final_risk",
    topn: int = 2000,
    file_penalty: float = 0.15,
    filefn_penalty: float = 0.10,
    max_per_file: int = 0,
    file_repeat_power: float = 1.0,
) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    n = len(rows)
    k = int(max(0, min(topn, n)))
    if k <= 1:
        return rows

    head = list(rows[:k])
    tail = list(rows[k:])
    selected: List[Dict[str, Any]] = []
    used = [False] * k
    file_cnt: Dict[str, int] = {}
    file_fn_cnt: Dict[Tuple[str, str], int] = {}
    fp = float(max(0.0, file_penalty))
    ffp = float(max(0.0, filefn_penalty))
    file_cap = int(max_per_file)
    file_pow = float(max(1.0, file_repeat_power))

    for _ in range(k):
        best_i = -1
        best_v = -1e18
        for i, row in enumerate(head):
            if used[i]:
                continue
            file_key = str(row.get("file") or row.get("relative_source_path") or row.get("contract_id") or "<unknown_file>")
            fn_key = _canonical_fn(str(row.get("function_names", "")))
            group_key = (file_key, fn_key)
            base = float(row.get(score_key, 0.0))
            file_seen = float(file_cnt.get(file_key, 0))
            file_cost = float(math.pow(file_seen, file_pow)) if file_seen > 0 else 0.0
            cap_violation = int(file_cap > 0 and file_cnt.get(file_key, 0) >= file_cap)
            score = (
                base
                - fp * file_cost
                - ffp * float(file_fn_cnt.get(group_key, 0))
                - 1.0 * float(cap_violation)
            )
            if score > best_v:
                best_v = score
                best_i = i
        if best_i < 0:
            break
        used[best_i] = True
        row = head[best_i]
        selected.append(row)
        file_key = str(row.get("file") or row.get("relative_source_path") or row.get("contract_id") or "<unknown_file>")
        fn_key = _canonical_fn(str(row.get("function_names", "")))
        group_key = (file_key, fn_key)
        file_cnt[file_key] = file_cnt.get(file_key, 0) + 1
        file_fn_cnt[group_key] = file_fn_cnt.get(group_key, 0) + 1

    return selected + tail


def function_coverage_rerank_rows(
    rows: List[Dict[str, Any]],
    score_key: str = "final_risk",
    topn: int = 2000,
    file_penalty: float = 0.10,
    filefn_penalty: float = 0.15,
    novelty_bonus: float = 0.04,
    overlap_penalty: float = 0.01,
    max_per_filefn: int = 0,
    max_per_file: int = 0,
    file_repeat_power: float = 1.0,
) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    n = len(rows)
    k = int(max(0, min(topn, n)))
    if k <= 1:
        return rows

    head = list(rows[:k])
    tail = list(rows[k:])
    selected: List[Dict[str, Any]] = []
    used = [False] * k
    file_cnt: Dict[str, int] = {}
    file_fn_cnt: Dict[Tuple[str, str], int] = {}
    covered_fn_by_file: Dict[str, Set[str]] = {}

    meta: List[Tuple[str, Set[str], float]] = []
    for row in head:
        fk = str(row.get("file") or row.get("relative_source_path") or row.get("contract_id") or "<unknown_file>")
        fn_set = _function_key_set(str(row.get("function_names", "")))
        base = float(row.get(score_key, 0.0))
        meta.append((fk, fn_set, base))

    fp = float(max(0.0, file_penalty))
    ffp = float(max(0.0, filefn_penalty))
    n_bonus = float(max(0.0, novelty_bonus))
    o_pen = float(max(0.0, overlap_penalty))
    fn_cap = int(max_per_filefn)
    file_cap = int(max_per_file)
    file_pow = float(max(1.0, file_repeat_power))

    for _ in range(k):
        best_i = -1
        best_v = -1e18
        for i in range(k):
            if used[i]:
                continue
            fk, fn_set, base = meta[i]
            covered = covered_fn_by_file.get(fk, set())
            novelty = sum(1 for fn in fn_set if fn not in covered)
            overlap = len(fn_set) - novelty
            avg_group_cnt = float(sum(file_fn_cnt.get((fk, fn), 0) for fn in fn_set)) / float(max(1, len(fn_set)))
            cap_violation = 0
            if fn_cap > 0:
                cap_violation = sum(1 for fn in fn_set if file_fn_cnt.get((fk, fn), 0) >= fn_cap)
            file_seen = float(file_cnt.get(fk, 0))
            file_cost = float(math.pow(file_seen, file_pow)) if file_seen > 0 else 0.0
            file_cap_violation = int(file_cap > 0 and file_cnt.get(fk, 0) >= file_cap)
            score = (
                base
                - fp * file_cost
                - ffp * avg_group_cnt
                + n_bonus * float(novelty)
                - o_pen * float(overlap)
                - 1.0 * float(cap_violation)
                - 1.0 * float(file_cap_violation)
            )
            if score > best_v:
                best_v = score
                best_i = i
        if best_i < 0:
            break
        used[best_i] = True
        row = head[best_i]
        selected.append(row)
        fk, fn_set, _ = meta[best_i]
        file_cnt[fk] = file_cnt.get(fk, 0) + 1
        cov = covered_fn_by_file.setdefault(fk, set())
        for fn in fn_set:
            cov.add(fn)
            key = (fk, fn)
            file_fn_cnt[key] = file_fn_cnt.get(key, 0) + 1

    return selected + tail


def family_hybrid_rerank_rows(
    rows: List[Dict[str, Any]],
    score_key: str = "final_risk",
    topn: int = 2000,
    file_penalty: float = 0.06,
    filefn_penalty: float = 0.12,
    novelty_bonus: float = 0.02,
    overlap_penalty: float = 0.01,
    family_penalty: float = 0.05,
    family_novelty_bonus: float = 0.03,
    aux_weight: float = 0.20,
    max_per_filefn: int = 0,
    max_per_family: int = 0,
    max_per_file: int = 0,
    file_repeat_power: float = 1.0,
) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    n = len(rows)
    k = int(max(0, min(topn, n)))
    if k <= 1:
        return rows

    head = list(rows[:k])
    tail = list(rows[k:])
    selected: List[Dict[str, Any]] = []
    used = [False] * k
    file_cnt: Dict[str, int] = {}
    file_fn_cnt: Dict[Tuple[str, str], int] = {}
    family_cnt: Dict[str, int] = {}
    covered_fn_by_file: Dict[str, Set[str]] = {}

    base_raw: List[float] = []
    aux_raw: List[float] = []
    meta: List[Tuple[str, str, Set[str]]] = []
    for row in head:
        fk = str(row.get("file") or row.get("relative_source_path") or row.get("contract_id") or "<unknown_file>")
        fam = _family_key_from_file(fk)
        fn_set = _function_key_set(str(row.get("function_names", "")))
        base_raw.append(float(row.get(score_key, 0.0)))
        aux_val = (
            0.45 * float(row.get("view_score", 0.0))
            + 0.35 * float(row.get("mechanism_risk", 0.0))
            + 0.20 * float(row.get("proto_score", 0.0))
        )
        aux_raw.append(aux_val)
        meta.append((fk, fam, fn_set))

    def _norm(arr: List[float]) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float32)
        if len(a) == 0:
            return a
        lo = float(np.min(a))
        hi = float(np.max(a))
        if hi - lo < 1e-8:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo + 1e-8)

    base_norm = _norm(base_raw)
    aux_norm = _norm(aux_raw)

    fp = float(max(0.0, file_penalty))
    ffp = float(max(0.0, filefn_penalty))
    n_bonus = float(max(0.0, novelty_bonus))
    o_pen = float(max(0.0, overlap_penalty))
    fam_pen = float(max(0.0, family_penalty))
    fam_bonus = float(max(0.0, family_novelty_bonus))
    a_w = float(max(0.0, aux_weight))
    fn_cap = int(max_per_filefn)
    fam_cap = int(max_per_family)
    file_cap = int(max_per_file)
    file_pow = float(max(1.0, file_repeat_power))

    for _ in range(k):
        best_i = -1
        best_v = -1e18
        for i in range(k):
            if used[i]:
                continue
            fk, fam, fn_set = meta[i]
            covered = covered_fn_by_file.get(fk, set())
            novelty = sum(1 for fn in fn_set if fn not in covered)
            overlap = len(fn_set) - novelty
            avg_group_cnt = (
                float(sum(file_fn_cnt.get((fk, fn), 0) for fn in fn_set)) / float(max(1, len(fn_set)))
            )
            cap_violation = 0
            if fn_cap > 0:
                cap_violation = sum(1 for fn in fn_set if file_fn_cnt.get((fk, fn), 0) >= fn_cap)
            fam_seen = int(family_cnt.get(fam, 0) > 0)
            fam_cap_violation = int(fam_cap > 0 and family_cnt.get(fam, 0) >= fam_cap)
            file_seen = float(file_cnt.get(fk, 0))
            file_cost = float(math.pow(file_seen, file_pow)) if file_seen > 0 else 0.0
            file_cap_violation = int(file_cap > 0 and file_cnt.get(fk, 0) >= file_cap)
            score = (
                float(base_norm[i])
                + a_w * float(aux_norm[i])
                - fp * file_cost
                - ffp * avg_group_cnt
                - fam_pen * float(family_cnt.get(fam, 0))
                + n_bonus * float(novelty)
                - o_pen * float(overlap)
                + fam_bonus * float(1 - fam_seen)
                - 1.0 * float(cap_violation)
                - 1.0 * float(fam_cap_violation)
                - 1.0 * float(file_cap_violation)
            )
            if score > best_v:
                best_v = score
                best_i = i
        if best_i < 0:
            break
        used[best_i] = True
        row = head[best_i]
        selected.append(row)
        fk, fam, fn_set = meta[best_i]
        file_cnt[fk] = file_cnt.get(fk, 0) + 1
        family_cnt[fam] = family_cnt.get(fam, 0) + 1
        cov = covered_fn_by_file.setdefault(fk, set())
        for fn in fn_set:
            cov.add(fn)
            key = (fk, fn)
            file_fn_cnt[key] = file_fn_cnt.get(key, 0) + 1

    return selected + tail


def warmup_diversify_rows(
    rows: List[Dict[str, Any]],
    warmup_topk: int = 0,
    per_file_cap: int = 3,
    per_filefn_cap: int = 1,
) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    k = int(max(0, min(warmup_topk, len(rows))))
    if k <= 1:
        return rows
    file_cap = max(1, int(per_file_cap))
    filefn_cap = max(1, int(per_filefn_cap))

    head = list(rows[:k])
    tail = list(rows[k:])
    file_cnt: Dict[str, int] = {}
    file_fn_cnt: Dict[Tuple[str, str], int] = {}
    kept: List[Dict[str, Any]] = []
    delayed: List[Dict[str, Any]] = []
    for row in head:
        file_key = str(row.get("file") or row.get("relative_source_path") or row.get("contract_id") or "<unknown_file>")
        fn_key = _canonical_fn(str(row.get("function_names", "")))
        group_key = (file_key, fn_key)
        if file_cnt.get(file_key, 0) < file_cap and file_fn_cnt.get(group_key, 0) < filefn_cap:
            kept.append(row)
            file_cnt[file_key] = file_cnt.get(file_key, 0) + 1
            file_fn_cnt[group_key] = file_fn_cnt.get(group_key, 0) + 1
        else:
            delayed.append(row)
    return kept + delayed + tail


def compute_dual_recall(
    pred_rows: List[Dict[str, Any]],
    label_df: pd.DataFrame,
    topk_list: List[int],
    line_window: int = 0,
) -> Dict[str, Any]:
    # Normalize label table.
    label_rows = []
    for _, r in label_df.iterrows():
        file_key = normalize_path(str(r["file"]))
        fn = None if pd.isna(r["function"]) else str(r["function"]).strip().lower()
        line = int(r["line"])
        label_rows.append({"file": file_key, "function": fn, "line": line})

    ff_keys = sorted(
        {
            (x["file"], x["function"])
            for x in label_rows
            if x["function"] is not None and x["function"] != ""
        }
    )

    output: Dict[str, Any] = {
        "label_counts": {
            "row88_total": int(len(label_rows)),
            "file_function_unique": int(len(ff_keys)),
            "row_with_missing_function": int(sum(1 for x in label_rows if not x["function"])),
        },
        "topk_metrics": [],
    }

    for k in topk_list:
        top = pred_rows[: min(k, len(pred_rows))]
        # Preprocess predictions.
        pred_norm = []
        for r in top:
            pred_norm.append(
                {
                    "file": normalize_path(str(r["file"])),
                    "fn_set": parse_fn_set(str(r.get("function_names", ""))),
                    "line_set": parse_line_set(str(r.get("line_candidates", ""))),
                }
            )

        row_hits: Set[int] = set()
        ff_hits: Set[Tuple[str, str]] = set()
        for li, lab in enumerate(label_rows):
            for p in pred_norm:
                if p["file"] != lab["file"]:
                    continue
                fn_ok = False
                if not lab["function"]:
                    fn_ok = True
                elif lab["function"] in p["fn_set"]:
                    fn_ok = True
                    ff_hits.add((lab["file"], lab["function"]))
                if not fn_ok:
                    continue
                if line_window <= 0:
                    hit_line = lab["line"] in p["line_set"]
                else:
                    hit_line = any((lab["line"] + d) in p["line_set"] for d in range(-line_window, line_window + 1))
                if hit_line:
                    row_hits.add(li)

        row_recall = len(row_hits) / len(label_rows) if label_rows else 0.0
        ff_recall = len(ff_hits) / len(ff_keys) if ff_keys else 0.0
        output["topk_metrics"].append(
            {
                "top_k": int(len(top)),
                "row88_hit": int(len(row_hits)),
                "row88_recall": float(row_recall),
                "file_function_hit": int(len(ff_hits)),
                "file_function_recall": float(ff_recall),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    t0 = time.time()
    project_root = Path(args.project_root).resolve()
    manual_root = (project_root / args.manual_root).resolve()
    label_csv = (project_root / args.label_csv).resolve()
    work_dir = (project_root / args.work_dir).resolve()
    prebuilt_artifacts_dir = (
        (project_root / args.prebuilt_artifacts_dir).resolve()
        if str(args.prebuilt_artifacts_dir).strip()
        else None
    )
    risk_out = (project_root / args.risk_out).resolve()
    line_risk_out = (project_root / args.line_risk_out).resolve()
    binary_out = (project_root / args.binary_out).resolve()
    summary_out = (project_root / args.summary_out).resolve()
    dual_model_path = (project_root / args.dual_model).resolve()
    proto_npz = (project_root / args.prototype_npz).resolve()

    work_dir.mkdir(parents=True, exist_ok=True)
    risk_out.parent.mkdir(parents=True, exist_ok=True)
    line_risk_out.parent.mkdir(parents=True, exist_ok=True)
    binary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    label_df = pd.read_csv(label_csv)
    manual_files = collect_manual_files(
        project_root=project_root,
        manual_root=manual_root,
        label_df=label_df,
        scope=args.scope,
        max_files=args.max_files,
    )
    if not manual_files:
        raise RuntimeError("No manual files collected for evaluation.")

    manual_list_path = work_dir / "manual_eval_files.txt"
    manual_list_path.write_text("\n".join(str(p) for p in manual_files) + "\n", encoding="utf-8")

    artifact_dir = prebuilt_artifacts_dir if prebuilt_artifacts_dir is not None else work_dir
    normalized_path = artifact_dir / "contracts_normalized.jsonl"
    graph_full_dir = artifact_dir / "graphs_full"
    graph_tagged_dir = artifact_dir / "graphs_tagged"
    slices_path = artifact_dir / "slices.jsonl"
    dual_view_path = artifact_dir / "dual_views.jsonl"
    graph_index_path = artifact_dir / "graph_index.jsonl"

    py = sys.executable
    if prebuilt_artifacts_dir is None:
        normalize_cmd = [
            py,
            "parser/normalize_or_expand.py",
            "--project-root",
            ".",
            "--dataset-root",
            "DataSet",
            "--input-list",
            str(manual_list_path.relative_to(project_root)),
            "--output",
            str(normalized_path.relative_to(project_root)),
            "--max-files",
            "0",
            "--expand-modifiers",
        ]
        if args.keep_comments:
            normalize_cmd.append("--keep-comments")
        run_cmd(normalize_cmd, cwd=project_root)
        run_cmd(
            [
                py,
                "parser/build_hetero_graph.py",
                "--input",
                str(normalized_path.relative_to(project_root)),
                "--graph-dir",
                str(graph_full_dir.relative_to(project_root)),
                "--index-out",
                str(graph_index_path.relative_to(project_root)),
                *(
                    ["--report-out", str(args.graph_report_out)]
                    if str(args.graph_report_out).strip()
                    else []
                ),
                *(["--implicit-mechanism-edges"] if args.implicit_mechanism_edges else []),
            ],
            cwd=project_root,
        )
        run_cmd(
            [
                py,
                "slicing/role_tagging.py",
                "--graph-dir",
                str(graph_full_dir.relative_to(project_root)),
                "--output-dir",
                str(graph_tagged_dir.relative_to(project_root)),
            ],
            cwd=project_root,
        )
        slice_cmd = [
            py,
            "slicing/state_influence_slice.py",
            "--graph-dir",
            str(graph_tagged_dir.relative_to(project_root)),
            "--output",
            str(slices_path.relative_to(project_root)),
            "--hops",
            str(args.hops),
            "--forward-hops",
            str(args.slice_forward_hops),
            "--backward-hops",
            str(args.slice_backward_hops),
            "--min-slice-nodes",
            str(args.min_slice_nodes),
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
            slice_cmd.extend(["--stats-out", str(args.slice_stats_out)])
        if args.slice_include_contains:
            slice_cmd.append("--include-contains")
        if args.slice_formal_propagation:
            slice_cmd.append("--formal-propagation")
        run_cmd(slice_cmd, cwd=project_root)
        run_cmd(
            [
                py,
                "slicing/dual_view_builder.py",
                "--input",
                str(slices_path.relative_to(project_root)),
                "--output",
                str(dual_view_path.relative_to(project_root)),
                "--skeleton-mode",
                args.dual_skeleton_mode,
                "--mechanism-path-max-len",
                str(args.dual_mechanism_path_max_len),
                "--dynamic-aux-keep-prob",
                str(args.dual_dynamic_aux_keep_prob),
                "--dynamic-seed",
                str(args.dual_dynamic_seed),
            ],
            cwd=project_root,
        )
    else:
        required_paths = [
            graph_tagged_dir,
            slices_path,
            dual_view_path,
        ]
        missing_paths = [str(p) for p in required_paths if not p.exists()]
        if missing_paths:
            raise FileNotFoundError(
                "prebuilt manual-eval artifacts missing required paths: "
                + ", ".join(missing_paths)
            )
        print(f"[evaluate_manual_set] reuse prebuilt artifacts: {prebuilt_artifacts_dir}")

    dual_rows = load_jsonl(dual_view_path)
    if not dual_rows:
        raise RuntimeError("No dual-view slices generated from manual set.")

    aligned_label_df = label_df.copy()
    line_alignment_diag: Dict[str, Any] = {"enabled": False}
    if bool(getattr(args, "line_align_enable", True)):
        aligned_label_df, line_alignment_diag = build_manual_label_alignment(
            label_df=label_df,
            dual_rows=dual_rows,
            max_shift=int(getattr(args, "line_align_max_shift", 24)),
        )
        aligned_label_path = work_dir / "manual_label_alignment.csv"
        aligned_label_df.to_csv(aligned_label_path, index=False, encoding="utf-8")
        line_alignment_diag["output_csv"] = str(aligned_label_path)

    device = torch.device(args.device)
    model, model_ckpt = load_dual_model(dual_model_path, device=device)
    manual_global_graph_dir = (
        (project_root / args.manual_global_graph_dir).resolve()
        if str(args.manual_global_graph_dir).strip()
        else graph_tagged_dir.resolve()
    )
    emb_map, line_sets = compute_embeddings(
        model,
        dual_rows,
        device=device,
        global_graph_dir=manual_global_graph_dir,
    )
    z_full = emb_map["z_full"]
    z_skel = emb_map["z_skeleton"]
    z_global = emb_map.get("z_global")
    z_fused = emb_map.get("z_fused")

    proto_arr = np.load(proto_npz)
    centers = proto_arr["centers"].astype(np.float32)
    radii = proto_arr["radii"].astype(np.float32)
    weights = proto_arr["weights"].astype(np.float32) if "weights" in proto_arr else None
    hier_enable = False
    hier_local_bank: Dict[str, Dict[str, Any]] = {}
    hier_family_min_samples = 96
    hier_shrinkage_tau = 96.0
    hier_global_mix_floor = 0.25
    hier_match_conf_floor = 0.25
    if "hier_prototype_enable" in proto_arr:
        try:
            hier_enable = bool(np.asarray(proto_arr["hier_prototype_enable"]).reshape(-1)[0])
        except Exception:
            hier_enable = False
    if hier_enable:
        hier_local_bank = unpack_local_bank(proto_arr)
        if "hier_family_min_samples" in proto_arr:
            hier_family_min_samples = int(np.asarray(proto_arr["hier_family_min_samples"]).reshape(-1)[0])
        if "hier_shrinkage_tau" in proto_arr:
            hier_shrinkage_tau = float(np.asarray(proto_arr["hier_shrinkage_tau"]).reshape(-1)[0])
        if "hier_global_mix_floor" in proto_arr:
            hier_global_mix_floor = float(np.asarray(proto_arr["hier_global_mix_floor"]).reshape(-1)[0])
        if "hier_match_conf_floor" in proto_arr:
            hier_match_conf_floor = float(np.asarray(proto_arr["hier_match_conf_floor"]).reshape(-1)[0])
    proto_embedding_key = str(args.embedding_key).strip().lower()
    if proto_embedding_key == "auto":
        if "embedding_key" in proto_arr:
            proto_embedding_key = str(proto_arr["embedding_key"].tolist())
        elif bool(model_ckpt.get("config", {}).get("global_local_enable", False)) and "z_fused" in emb_map:
            proto_embedding_key = "z_fused"
        else:
            proto_embedding_key = "z_joint"
    if proto_embedding_key == "z_hybrid":
        if "z_joint" in emb_map and "z_fused" in emb_map:
            if "hybrid_alpha" in proto_arr:
                alpha = float(np.asarray(proto_arr["hybrid_alpha"]).reshape(-1)[0])
            else:
                alpha = float(args.hybrid_alpha)
            alpha = float(min(1.0, max(0.0, alpha)))
            emb_map["z_hybrid"] = (
                alpha * emb_map["z_fused"] + (1.0 - alpha) * emb_map["z_joint"]
            ).astype(np.float32)
        else:
            proto_embedding_key = "z_joint"
    if proto_embedding_key not in emb_map:
        proto_embedding_key = "z_joint"
    function_bag_enable = bool(args.function_bag_enable)
    if "function_bag_enable" in proto_arr:
        try:
            function_bag_enable = bool(np.asarray(proto_arr["function_bag_enable"]).reshape(-1)[0])
        except Exception:
            function_bag_enable = bool(args.function_bag_enable)
    function_bag_agg = str(args.function_bag_agg)
    if "function_bag_agg" in proto_arr:
        try:
            function_bag_agg = str(proto_arr["function_bag_agg"].tolist())
        except Exception:
            function_bag_agg = str(args.function_bag_agg)
    mechanism_raw_slice = compute_mechanism_risk(dual_rows)
    row_to_group = np.arange(len(dual_rows), dtype=np.int64)
    bag_sizes = np.ones((len(dual_rows),), dtype=np.int32)
    fit_rows = dual_rows
    z_proto_fit = emb_map[proto_embedding_key]
    z_full_fit = z_full
    z_skel_fit = z_skel
    mechanism_raw_fit = mechanism_raw_slice
    if function_bag_enable:
        (
            fit_rows,
            z_proto_fit,
            z_full_fit,
            z_skel_fit,
            mechanism_raw_fit,
            row_to_group,
            bag_sizes,
        ) = build_function_bag_view_eval(
            dual_rows=dual_rows,
            emb_main=emb_map[proto_embedding_key],
            emb_full=z_full,
            emb_skel=z_skel,
            mechanism_raw=mechanism_raw_slice,
            agg_mode=function_bag_agg,
        )
    assign, nearest = nearest_to_centers(z_proto_fit, centers)
    margin = nearest - radii[assign]
    local_assign = np.full_like(assign, -1)
    local_nearest = nearest.astype(np.float32)
    local_margin = margin.astype(np.float32)
    local_weight = np.zeros_like(nearest, dtype=np.float32)
    hier_diag: Dict[str, Any] = {
        "enabled": bool(hier_enable and bool(hier_local_bank)),
        "local_family_count": int(len(hier_local_bank)),
        "local_weight_mean": 0.0,
        "local_weight_std": 0.0,
        "local_weight_max": 0.0,
        "local_active_ratio": 0.0,
        "local_match_conf_mean": 0.0,
        "local_match_conf_std": 0.0,
        "local_route_exact_ratio": 0.0,
        "local_route_fallback_ratio": 0.0,
        "family_min_samples": int(hier_family_min_samples),
        "shrinkage_tau": float(hier_shrinkage_tau),
        "global_mix_floor": float(hier_global_mix_floor),
        "match_conf_floor": float(hier_match_conf_floor),
    }
    if hier_enable and hier_local_bank:
        hier_score = score_hierarchical_prototypes(
            rows=fit_rows,
            embeddings=z_proto_fit,
            global_assign=assign,
            global_nearest=nearest,
            global_margin=margin,
            local_bank=hier_local_bank,
            min_samples=int(max(4, hier_family_min_samples)),
            shrinkage_tau=float(hier_shrinkage_tau),
            global_mix_floor=float(hier_global_mix_floor),
            match_conf_floor=float(hier_match_conf_floor),
        )
        nearest = hier_score["combined_nearest"]
        margin = hier_score["combined_margin"]
        local_assign = hier_score["local_assign"]
        local_nearest = hier_score["local_nearest"]
        local_margin = hier_score["local_margin"]
        local_weight = hier_score["local_weight"]
        local_match_conf = hier_score.get("local_match_conf", np.ones_like(local_weight))
        local_route_exact = hier_score.get("local_route_exact", np.zeros_like(local_weight))
        local_route_fallback = hier_score.get("local_route_fallback_centroid", np.zeros_like(local_weight))
        hier_diag.update(
            {
                "local_weight_mean": float(np.mean(local_weight)) if len(local_weight) else 0.0,
                "local_weight_std": float(np.std(local_weight)) if len(local_weight) else 0.0,
                "local_weight_max": float(np.max(local_weight)) if len(local_weight) else 0.0,
                "local_active_ratio": float(np.mean((local_weight > 0).astype(np.float32))) if len(local_weight) else 0.0,
                "local_match_conf_mean": float(np.mean(local_match_conf)) if len(local_match_conf) else 0.0,
                "local_match_conf_std": float(np.std(local_match_conf)) if len(local_match_conf) else 0.0,
                "local_route_exact_ratio": float(np.mean(local_route_exact)) if len(local_route_exact) else 0.0,
                "local_route_fallback_ratio": float(np.mean(local_route_fallback)) if len(local_route_fallback) else 0.0,
            }
        )
    view_gap = np.linalg.norm(z_full_fit - z_skel_fit, axis=1).astype(np.float32)
    global_gap = (
        np.linalg.norm(emb_map["z_joint"] - z_global, axis=1).astype(np.float32)
        if z_global is not None and len(z_global) == len(emb_map["z_joint"])
        else np.zeros(len(emb_map["z_joint"]), dtype=np.float32)
    )
    fused_gap = (
        np.linalg.norm(z_fused - emb_map["z_joint"], axis=1).astype(np.float32)
        if z_fused is not None and len(z_fused) == len(emb_map["z_joint"])
        else np.zeros(len(emb_map["z_joint"]), dtype=np.float32)
    )
    bag_global_gap = np.zeros((len(fit_rows),), dtype=np.float32)
    bag_fused_gap = np.zeros((len(fit_rows),), dtype=np.float32)
    if function_bag_enable and len(fit_rows) > 0:
        bag_global_gap = _aggregate_group_scalar_eval(global_gap, [np.flatnonzero(row_to_group == i) for i in range(len(fit_rows))], reduce="mean")
        bag_fused_gap = _aggregate_group_scalar_eval(fused_gap, [np.flatnonzero(row_to_group == i) for i in range(len(fit_rows))], reduce="mean")

    refiner = BoundaryRefiner(
        RefineWeights(
            w_proto=0.45,
            w_view=0.25,
            w_density=0.15,
            w_stability=0.15,
        )
    )
    refined = refiner.refine(
        proto_margin=margin,
        view_gap=view_gap,
        embeddings=z_proto_fit,
        centers=centers,
        density_k=10,
        n_perturb=5,
        noise_std=0.02,
        adaptive_local=bool(args.adaptive_boundary_refine),
        adaptive_focus_quantile=args.adaptive_boundary_focus_quantile,
        adaptive_min_scale=args.adaptive_boundary_min_scale,
        adaptive_max_scale=args.adaptive_boundary_max_scale,
        adaptive_outside_scale=args.adaptive_boundary_outside_scale,
    )
    base_final = refined["final_risk"].astype(np.float32)
    family_score, family_diag = compute_family_aware_risk(
        rows=fit_rows,
        embeddings=z_proto_fit,
        proto_margin=margin,
        min_samples=args.family_min_samples,
        boundary_quantile=args.family_boundary_quantile,
        shrinkage_tau=args.family_shrinkage_tau,
        ramp_width=args.family_ramp_width,
        global_mix_floor=args.family_global_mix_floor,
        local_radius_min_ratio=args.family_radius_min_ratio,
        local_radius_max_ratio=args.family_radius_max_ratio,
    )
    w_family = float(min(1.0, max(0.0, args.family_aware_weight)))
    family_top_mask = top_quantile_mask(base_final, quantile=args.family_top_quantile)
    mechanism_raw = mechanism_raw_fit
    mechanism_score = adaptive_mechanism_score(mechanism_raw_fit, std_threshold=0.05)
    w_mech = float(min(1.0, max(0.0, args.mechanism_risk_weight)))
    mech_top_mask = top_quantile_mask(base_final, quantile=args.mechanism_top_quantile)
    if args.mechanism_semantic_gate:
        sem_mask = semantic_closed_loop_mask(
            dual_rows,
            min_connected_pairs=args.mechanism_semantic_min_pairs,
            max_hops=args.mechanism_semantic_max_hops,
        )
    else:
        sem_mask = np.ones_like(mech_top_mask, dtype=bool)
    mech_active_mask = np.logical_and(mech_top_mask, sem_mask)
    blended = blend_mechanism_risk(
        base_final,
        mechanism_score,
        weight=w_mech,
        active_mask=mech_active_mask,
    )
    blended = blend_family_risk(
        blended,
        family_score,
        weight=w_family,
        active_mask=family_top_mask,
    )
    q90 = float(np.quantile(blended, 0.9)) if len(blended) else 0.0
    q75 = float(np.quantile(blended, 0.75)) if len(blended) else 0.0
    status = np.array(
        [
            "high_risk" if x >= q90 else ("boundary" if x >= q75 else "normal")
            for x in blended
        ]
    )
    if function_bag_enable and str(args.function_bag_eval_level).strip().lower() == "function":
        bag_line_sets: List[Set[int]] = [set() for _ in range(len(fit_rows))]
        bag_closure_rows: List[List[Dict[str, float]]] = [[] for _ in range(len(fit_rows))]
        for i, ls in enumerate(line_sets):
            bag_line_sets[int(row_to_group[i])].update(ls)
            bag_closure_rows[int(row_to_group[i])].append(
                compute_graph_closure_features(
                    dual_rows[i].get("full_graph", {}) or {},
                    dual_rows[i].get("skeleton_graph", {}) or {},
                )
            )
        bag_closure_agg = [aggregate_closure_features(x, reduce="max") for x in bag_closure_rows]
        refined_bag = dict(refined)
        refined_bag["final_risk"] = blended
        refined_bag["status"] = status
        refined_bag["base_final_risk"] = base_final
        refined_bag["mechanism_risk"] = mechanism_score
        refined_bag["family_risk"] = family_score
        risk_rows = build_function_bag_eval_rows(
            bag_rows=fit_rows,
            bag_line_sets=bag_line_sets,
            assign=assign,
            nearest=nearest,
            margin=margin,
            view_gap=view_gap,
            refined=refined_bag,
            bag_global_gap=bag_global_gap,
            bag_fused_gap=bag_fused_gap,
            bag_closure_features=bag_closure_agg,
        )
    else:
        assign_expand = assign[row_to_group]
        nearest_expand = nearest[row_to_group]
        margin_expand = margin[row_to_group]
        local_assign_expand = local_assign[row_to_group]
        local_nearest_expand = local_nearest[row_to_group]
        local_margin_expand = local_margin[row_to_group]
        local_weight_expand = local_weight[row_to_group]
        view_gap_expand = view_gap[row_to_group]
        base_final_expand = base_final[row_to_group]
        mechanism_expand = mechanism_score[row_to_group]
        family_expand = family_score[row_to_group]
        blended_expand = blended[row_to_group]
        status_expand = status[row_to_group]
        refined["final_risk"] = blended_expand
        refined["status"] = status_expand
        refined["base_final_risk"] = base_final_expand
        refined["mechanism_risk"] = mechanism_expand
        refined["family_risk"] = family_expand
        refined["proto_score"] = refined["proto_score"][row_to_group]
        refined["view_score"] = refined["view_score"][row_to_group]
        refined["density_risk"] = refined["density_risk"][row_to_group]
        refined["instability"] = refined["instability"][row_to_group]
        refined["boundary_proximity"] = refined.get("boundary_proximity", np.zeros_like(base_final))[row_to_group]
        refined["adaptive_local_scale"] = refined.get("adaptive_local_scale", np.ones_like(base_final))[row_to_group]
        refined["adaptive_focus_mask"] = refined.get(
            "adaptive_focus_mask",
            np.ones_like(base_final, dtype=np.bool_),
        )[row_to_group]
        refined["global_gap"] = global_gap
        refined["fused_gap"] = fused_gap
        risk_rows = build_eval_rows(
            dual_rows=dual_rows,
            line_sets=line_sets,
            assign=assign_expand,
            nearest=nearest_expand,
            margin=margin_expand,
            view_gap=view_gap_expand,
            refined=refined,
        )
        if function_bag_enable:
            for i, row in enumerate(risk_rows):
                row["function_bag_enable"] = 1
                row["function_bag_size"] = int(bag_sizes[row_to_group[i]]) if len(bag_sizes) else 1
                row["local_prototype_id"] = int(local_assign_expand[i])
                row["local_nearest_distance"] = float(local_nearest_expand[i])
                row["local_boundary_margin"] = float(local_margin_expand[i])
                row["hier_local_weight"] = float(local_weight_expand[i])
        else:
            for i, row in enumerate(risk_rows):
                row["function_bag_enable"] = 0
                row["function_bag_size"] = 1
                row["local_prototype_id"] = int(local_assign_expand[i])
                row["local_nearest_distance"] = float(local_nearest_expand[i])
                row["local_boundary_margin"] = float(local_margin_expand[i])
                row["hier_local_weight"] = float(local_weight_expand[i])
    model_binary_score = emb_map.get("model_binary_score")
    if model_binary_score is not None and len(model_binary_score) == len(risk_rows):
        for i, row in enumerate(risk_rows):
            row["model_binary_score"] = float(model_binary_score[i])
    legacy_final_risk = np.asarray([float(r.get("final_risk", 0.0)) for r in risk_rows], dtype=np.float32)
    score_head_risk, score_head_diag = apply_score_head(risk_rows, args)
    if len(score_head_risk) == len(risk_rows):
        for i, row in enumerate(risk_rows):
            row["legacy_final_risk"] = float(legacy_final_risk[i])
            row["score_head_risk"] = float(score_head_risk[i])
            if str(args.score_head_method).lower() != "none":
                row["final_risk"] = float(score_head_risk[i])
    mechanism_head_pre_diag: Dict[str, Any] = {"enabled": False}
    if len(risk_rows) > 0 and (
        str(getattr(args, "binary_head_method", "none")).lower() == "mechanism_head"
        or bool(getattr(args, "mechanism_head_override_final_risk", False))
    ):
        mechanism_score_pre, mechanism_head_pre_diag = apply_mechanism_head_score(
            risk_rows,
            dual_rows,
            args,
        )
        if bool(getattr(args, "mechanism_head_override_final_risk", False)):
            for i, row in enumerate(risk_rows):
                row["final_risk"] = float(mechanism_score_pre[i])
    ranking_score_key = "final_risk"
    risk_rows.sort(key=lambda x: float(x.get(ranking_score_key, 0.0)), reverse=True)
    if args.risk_rerank_mode == "coverage":
        risk_rows = coverage_rerank_rows(
            risk_rows,
            score_key=ranking_score_key,
            topn=args.risk_rerank_topn,
            file_penalty=args.risk_rerank_file_penalty,
            filefn_penalty=args.risk_rerank_filefn_penalty,
            max_per_file=args.risk_rerank_max_per_file,
            file_repeat_power=args.risk_rerank_file_repeat_power,
        )
    elif args.risk_rerank_mode == "function_coverage":
        risk_rows = function_coverage_rerank_rows(
            risk_rows,
            score_key=ranking_score_key,
            topn=args.risk_rerank_topn,
            file_penalty=args.risk_rerank_file_penalty,
            filefn_penalty=args.risk_rerank_filefn_penalty,
            novelty_bonus=args.risk_rerank_fn_novelty_bonus,
            overlap_penalty=args.risk_rerank_fn_overlap_penalty,
            max_per_filefn=args.risk_rerank_max_per_filefn,
            max_per_file=args.risk_rerank_max_per_file,
            file_repeat_power=args.risk_rerank_file_repeat_power,
        )
    elif args.risk_rerank_mode == "family_hybrid":
        risk_rows = family_hybrid_rerank_rows(
            risk_rows,
            score_key=ranking_score_key,
            topn=args.risk_rerank_topn,
            file_penalty=args.risk_rerank_file_penalty,
            filefn_penalty=args.risk_rerank_filefn_penalty,
            novelty_bonus=args.risk_rerank_fn_novelty_bonus,
            overlap_penalty=args.risk_rerank_fn_overlap_penalty,
            family_penalty=args.risk_rerank_family_penalty,
            family_novelty_bonus=args.risk_rerank_family_novelty_bonus,
            aux_weight=args.risk_rerank_aux_weight,
            max_per_filefn=args.risk_rerank_max_per_filefn,
            max_per_family=args.risk_rerank_max_per_family,
            max_per_file=args.risk_rerank_max_per_file,
            file_repeat_power=args.risk_rerank_file_repeat_power,
        )
    risk_rows = warmup_diversify_rows(
        risk_rows,
        warmup_topk=args.risk_rerank_warmup_topk,
        per_file_cap=args.risk_rerank_warmup_file_cap,
        per_filefn_cap=args.risk_rerank_warmup_filefn_cap,
    )

    with risk_out.open("w", newline="", encoding="utf-8") as f:
        risk_fieldnames = [
            "slice_id",
            "contract_id",
            "relative_source_path",
            "file",
            "function_names",
            "function_bag_enable",
            "function_bag_size",
            "line_candidates",
            "prototype_id",
            "nearest_distance",
            "boundary_margin",
            "local_prototype_id",
            "local_nearest_distance",
            "local_boundary_margin",
            "hier_local_weight",
            "view_gap",
            "global_gap",
            "fused_gap",
            "proto_score",
            "view_score",
            "density_risk",
            "instability",
            "boundary_proximity",
            "adaptive_local_scale",
            "adaptive_focus_mask",
            "base_final_risk",
            "legacy_final_risk",
            "mechanism_risk",
            "family_risk",
            "score_head_risk",
            "mechanism_head_score",
            "mechanism_head_bridge_gate",
            "mechanism_head_mechanism_norm",
            "mechanism_head_closure_norm",
            "mechanism_head_residual_norm",
            "mechanism_head_if_norm",
            "mechanism_head_vp_evidence",
            "mechanism_head_margin_sig",
            "mechanism_head_corroboration",
            "mechanism_head_corroboration_factor",
            "mechanism_head_utility_surface",
            "mechanism_head_utility_corr",
            "mechanism_head_utility_discount",
            "mechanism_head_routine_bridge_surface",
            "mechanism_head_routine_bridge_discount",
            "mechanism_head_compat_wrapper_surface",
            "mechanism_head_compat_wrapper_discount",
            "mechanism_head_threshold",
            "final_risk",
            "status",
        ]
        risk_fieldnames.extend(list(CLOSURE_FEATURE_NAMES))
        if any("model_binary_score" in row for row in risk_rows):
            risk_fieldnames.append("model_binary_score")
        writer = csv.DictWriter(
            f,
            fieldnames=risk_fieldnames,
        )
        writer.writeheader()
        writer.writerows(risk_rows)

    line_risk_rows, line_risk_diag = build_line_risk_rows(
        risk_rows=risk_rows,
        dual_rows=dual_rows,
        topn=args.line_risk_topn,
        closure_boost=args.line_risk_closure_boost,
        stage2_weight=args.line_risk_stage2_weight,
        stage2_top_functions=args.line_risk_stage2_top_functions,
        stage2_support_weight=args.line_risk_stage2_support_weight,
        stage2_max_file_share=args.line_risk_stage2_max_file_share,
    )
    with line_risk_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "line",
                "line_risk",
                "base_line_risk",
                "stage2_bonus",
                "support_slices",
                "max_slice_risk",
                "mean_contrib",
            ],
        )
        writer.writeheader()
        writer.writerows(line_risk_rows)

    binary_eval_topn = int(max(0, args.binary_eval_topn))
    if binary_eval_topn > 0:
        binary_rows = list(risk_rows[: min(binary_eval_topn, len(risk_rows))])
    else:
        binary_rows = list(risk_rows)
    metamorphic_feature_diag: Dict[str, Any] = {"enabled": False}
    if bool(args.metamorphic_feature_enable) and len(binary_rows) > 0:
        dual_row_lookup = {str(r.get("slice_id", "")): r for r in dual_rows}
        meta_feat, metamorphic_feature_diag = compute_metamorphic_instability_features(
            eval_rows=binary_rows,
            dual_row_lookup=dual_row_lookup,
            model=model,
            device=device,
            global_graph_dir=manual_global_graph_dir,
            centers=centers,
            radii=radii,
            embedding_key=str(proto_embedding_key),
            hybrid_alpha=float(args.hybrid_alpha),
            max_hops=int(max(1, args.metamorphic_max_hops)),
            max_rows=int(max(0, args.metamorphic_feature_max_rows)),
            select_mode=str(args.metamorphic_feature_select),
            boundary_quantile=float(args.metamorphic_feature_boundary_quantile),
            switch_min=float(args.metamorphic_feature_switch_min),
            gate_mode=str(args.metamorphic_feature_gate_mode),
        )
        for i, row in enumerate(binary_rows):
            row["metamorphic_instability"] = float(meta_feat["metamorphic_instability"][i])
            row["metamorphic_nearest_delta"] = float(meta_feat["metamorphic_nearest_delta"][i])
            row["metamorphic_margin_delta"] = float(meta_feat["metamorphic_margin_delta"][i])
            row["metamorphic_view_delta"] = float(meta_feat["metamorphic_view_delta"][i])
            row["metamorphic_proto_switch_rate"] = float(meta_feat["metamorphic_proto_switch_rate"][i])
            row["metamorphic_variant_count"] = int(meta_feat["metamorphic_variant_count"][i])
    else:
        for row in binary_rows:
            row["metamorphic_instability"] = 0.0
            row["metamorphic_nearest_delta"] = 0.0
            row["metamorphic_margin_delta"] = 0.0
            row["metamorphic_view_delta"] = 0.0
            row["metamorphic_proto_switch_rate"] = 0.0
            row["metamorphic_variant_count"] = 0
    emb_lookup: Dict[str, np.ndarray] = {}
    graph_lookup: Dict[str, np.ndarray] = {}
    text_lookup: Dict[str, np.ndarray] = {}
    proto_dim = int(emb_map[proto_embedding_key].shape[1]) if proto_embedding_key in emb_map and emb_map[proto_embedding_key].ndim == 2 else 0
    graph_dim = int(emb_map["z_graph_joint"].shape[1]) if "z_graph_joint" in emb_map and emb_map["z_graph_joint"].ndim == 2 else 0
    text_dim = int(emb_map["z_text"].shape[1]) if "z_text" in emb_map and emb_map["z_text"].ndim == 2 else 0
    for i, row in enumerate(dual_rows):
        sid = str(row.get("slice_id", ""))
        if sid:
            emb_lookup[sid] = np.asarray(emb_map[proto_embedding_key][i], dtype=np.float32)
            if graph_dim > 0 and i < len(emb_map["z_graph_joint"]):
                graph_lookup[sid] = np.asarray(emb_map["z_graph_joint"][i], dtype=np.float32)
            if text_dim > 0 and i < len(emb_map["z_text"]):
                text_lookup[sid] = np.asarray(emb_map["z_text"][i], dtype=np.float32)
    if function_bag_enable and len(fit_rows) > 0:
        for i, row in enumerate(fit_rows):
            sid = str(row.get("slice_id", ""))
            if sid and sid.startswith("bag::"):
                emb_lookup[sid] = np.asarray(z_proto_fit[i], dtype=np.float32)
                if proto_dim <= 0:
                    proto_dim = int(z_proto_fit.shape[1]) if z_proto_fit.ndim == 2 else 0
    binary_energy_embeddings = np.zeros((len(binary_rows), max(0, proto_dim)), dtype=np.float32)
    binary_energy_graph_embeddings = np.zeros((len(binary_rows), max(0, graph_dim)), dtype=np.float32)
    binary_energy_text_embeddings = np.zeros((len(binary_rows), max(0, text_dim)), dtype=np.float32)
    if proto_dim > 0 and len(binary_rows) > 0:
        for i, row in enumerate(binary_rows):
            sid = str(row.get("slice_id", ""))
            vec = emb_lookup.get(sid)
            if vec is not None and len(vec) == proto_dim:
                binary_energy_embeddings[i] = np.asarray(vec, dtype=np.float32)
            if graph_dim > 0:
                gvec = graph_lookup.get(sid)
                if gvec is not None and len(gvec) == graph_dim:
                    binary_energy_graph_embeddings[i] = np.asarray(gvec, dtype=np.float32)
            if text_dim > 0:
                tvec = text_lookup.get(sid)
                if tvec is not None and len(tvec) == text_dim:
                    binary_energy_text_embeddings[i] = np.asarray(tvec, dtype=np.float32)
    binary_score, binary_head_diag = apply_binary_head(
        binary_rows,
        args,
        energy_embeddings=binary_energy_embeddings if len(binary_rows) > 0 else None,
        energy_graph_embeddings=binary_energy_graph_embeddings if (len(binary_rows) > 0 and graph_dim > 0) else None,
        energy_text_embeddings=binary_energy_text_embeddings if (len(binary_rows) > 0 and text_dim > 0) else None,
        prototype_centers=centers,
        prototype_radii=radii,
        prototype_weights=weights,
    )
    score_for_binary = np.asarray([float(r.get("final_risk", 0.0)) for r in binary_rows], dtype=np.float32)
    binary_pred, binary_region, triage_diag = apply_triage_predictions(
        binary_rows,
        score_risk=score_for_binary,
        binary_score=binary_score,
        args=args,
        binary_head_diag=binary_head_diag,
    )
    proto_resp_gray_diag: Dict[str, Any] = {"enabled": False}
    if bool(args.proto_resp_gray_adjust_enable) and len(binary_rows) > 0:
        binary_score_adj, proto_resp_gray_diag = apply_proto_resp_gray_adjustment(
            rows=binary_rows,
            region=binary_region,
            binary_score=binary_score,
            args=args,
        )
        binary_pred, binary_region, triage_diag = apply_triage_predictions(
            binary_rows,
            score_risk=score_for_binary,
            binary_score=binary_score_adj,
            args=args,
            binary_head_diag=binary_head_diag,
        )
        binary_score = binary_score_adj
    metamorphic_gray_diag: Dict[str, Any] = {"enabled": False}
    if bool(args.metamorphic_gray_enable) and len(binary_rows) > 0:
        binary_score_adj, score_risk_adj, metamorphic_gray_diag = apply_metamorphic_gray_adjustment(
            rows=binary_rows,
            dual_rows=dual_rows,
            region=binary_region,
            score_risk=score_for_binary,
            binary_score=binary_score,
            args=args,
            model=model,
            device=device,
            global_graph_dir=manual_global_graph_dir,
            centers=centers,
            radii=radii,
            embedding_key=proto_embedding_key,
            hybrid_alpha=float(args.hybrid_alpha),
        )
        binary_pred, binary_region, triage_diag = apply_triage_predictions(
            binary_rows,
            score_risk=score_risk_adj,
            binary_score=binary_score_adj,
            args=args,
            binary_head_diag=binary_head_diag,
        )
        binary_score = binary_score_adj
        score_for_binary = score_risk_adj
    binary_true = build_binary_targets(
        binary_rows,
        label_df=label_df,
        mode=args.binary_target_mode,
        line_window=args.binary_target_window,
    )
    binary_metrics = compute_binary_metrics(binary_true, binary_pred)
    with binary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "slice_id",
                "file",
                "function_names",
                "line_candidates",
                "final_risk",
                "legacy_final_risk",
                "score_head_risk",
                "final_risk_base",
                "final_risk_adjusted",
                "binary_score",
                "binary_score_base",
                "binary_score_adjusted",
                "binary_region",
                "mechanism_head_score",
                "mechanism_head_bridge_gate",
                "mechanism_head_mechanism_norm",
                "mechanism_head_closure_norm",
                "mechanism_head_residual_norm",
                "mechanism_head_if_norm",
                "mechanism_head_vp_evidence",
                "mechanism_head_margin_sig",
                "mechanism_head_corroboration",
                "mechanism_head_corroboration_factor",
                "mechanism_head_utility_surface",
                "mechanism_head_utility_corr",
                "mechanism_head_utility_discount",
                "mechanism_head_routine_bridge_surface",
                "mechanism_head_routine_bridge_discount",
                "mechanism_head_compat_wrapper_surface",
                "mechanism_head_compat_wrapper_discount",
                "mechanism_head_threshold",
                "proto_resp_entropy",
                "proto_resp_max",
                "proto_resp_gap",
                "proto_top2_norm_gap",
                "proto_confidence",
                "metamorphic_instability",
                "metamorphic_nearest_delta",
                "metamorphic_margin_delta",
                "metamorphic_view_delta",
                "metamorphic_proto_switch_rate",
                "metamorphic_variant_count",
                "binary_true",
                "binary_pred",
            ],
        )
        writer.writeheader()
        for i, row in enumerate(binary_rows):
            writer.writerow(
                {
                    "slice_id": row.get("slice_id"),
                    "file": row.get("file"),
                    "function_names": row.get("function_names"),
                    "line_candidates": row.get("line_candidates"),
                    "final_risk": float(row.get("final_risk", 0.0)),
                    "legacy_final_risk": float(row.get("legacy_final_risk", row.get("base_final_risk", 0.0))),
                    "score_head_risk": float(row.get("score_head_risk", row.get("final_risk", 0.0))),
                    "final_risk_base": float(row.get("final_risk_base", row.get("final_risk", 0.0))),
                    "final_risk_adjusted": float(row.get("final_risk_adjusted", row.get("final_risk", 0.0))),
                    "binary_score": float(binary_score[i]) if len(binary_score) > i else 0.0,
                    "binary_score_base": float(row.get("binary_score_base", binary_score[i] if len(binary_score) > i else 0.0)),
                    "binary_score_adjusted": float(row.get("binary_score_adjusted", binary_score[i] if len(binary_score) > i else 0.0)),
                    "binary_region": str(binary_region[i]) if len(binary_region) > i else "gray",
                    "mechanism_head_score": float(row.get("mechanism_head_score", 0.0)),
                    "mechanism_head_bridge_gate": float(row.get("mechanism_head_bridge_gate", 0.0)),
                    "mechanism_head_mechanism_norm": float(row.get("mechanism_head_mechanism_norm", 0.0)),
                    "mechanism_head_closure_norm": float(row.get("mechanism_head_closure_norm", 0.0)),
                    "mechanism_head_residual_norm": float(row.get("mechanism_head_residual_norm", 0.0)),
                    "mechanism_head_if_norm": float(row.get("mechanism_head_if_norm", 0.0)),
                    "mechanism_head_vp_evidence": float(row.get("mechanism_head_vp_evidence", 0.0)),
                    "mechanism_head_margin_sig": float(row.get("mechanism_head_margin_sig", 0.0)),
                    "mechanism_head_corroboration": float(row.get("mechanism_head_corroboration", 0.0)),
                    "mechanism_head_corroboration_factor": float(row.get("mechanism_head_corroboration_factor", 0.0)),
                    "mechanism_head_utility_surface": float(row.get("mechanism_head_utility_surface", 0.0)),
                    "mechanism_head_utility_corr": float(row.get("mechanism_head_utility_corr", 0.0)),
                    "mechanism_head_utility_discount": float(row.get("mechanism_head_utility_discount", 0.0)),
                    "mechanism_head_routine_bridge_surface": float(row.get("mechanism_head_routine_bridge_surface", 0.0)),
                    "mechanism_head_routine_bridge_discount": float(row.get("mechanism_head_routine_bridge_discount", 1.0)),
                    "mechanism_head_compat_wrapper_surface": float(row.get("mechanism_head_compat_wrapper_surface", 0.0)),
                    "mechanism_head_compat_wrapper_discount": float(row.get("mechanism_head_compat_wrapper_discount", 1.0)),
                    "mechanism_head_threshold": float(row.get("mechanism_head_threshold", 0.0)),
                    "proto_resp_entropy": float(row.get("proto_resp_entropy", 0.0)),
                    "proto_resp_max": float(row.get("proto_resp_max", 0.0)),
                    "proto_resp_gap": float(row.get("proto_resp_gap", 0.0)),
                    "proto_top2_norm_gap": float(row.get("proto_top2_norm_gap", 0.0)),
                    "proto_confidence": float(row.get("proto_confidence", 0.0)),
                    "metamorphic_instability": float(row.get("metamorphic_instability", 0.0)),
                    "metamorphic_nearest_delta": float(row.get("metamorphic_nearest_delta", 0.0)),
                    "metamorphic_margin_delta": float(row.get("metamorphic_margin_delta", 0.0)),
                    "metamorphic_view_delta": float(row.get("metamorphic_view_delta", 0.0)),
                    "metamorphic_proto_switch_rate": float(row.get("metamorphic_proto_switch_rate", 0.0)),
                    "metamorphic_variant_count": int(row.get("metamorphic_variant_count", 0)),
                    "binary_true": int(binary_true[i]) if len(binary_true) > i else 0,
                    "binary_pred": int(binary_pred[i]) if len(binary_pred) > i else 0,
                }
            )

    topk_list = parse_topk(args.topk)
    line_topk_list = parse_topk(args.line_risk_topk)
    line_windows = parse_windows(args.line_match_windows)
    recall_by_window: Dict[str, Any] = {}
    line_recall_by_window: Dict[str, Any] = {}
    recall_by_window_aligned: Dict[str, Any] = {}
    line_recall_by_window_aligned: Dict[str, Any] = {}
    for w in line_windows:
        recall_by_window[str(w)] = compute_dual_recall(
            risk_rows,
            label_df=label_df,
            topk_list=topk_list,
            line_window=w,
        )
        line_recall_by_window[str(w)] = compute_line_recall(
            line_rows=line_risk_rows,
            label_df=label_df,
            topk_list=line_topk_list,
            line_window=w,
        )
        if bool(getattr(args, "line_align_enable", True)):
            recall_by_window_aligned[str(w)] = compute_dual_recall(
                risk_rows,
                label_df=aligned_label_df,
                topk_list=topk_list,
                line_window=w,
            )
            line_recall_by_window_aligned[str(w)] = compute_line_recall(
                line_rows=line_risk_rows,
                label_df=aligned_label_df,
                topk_list=line_topk_list,
                line_window=w,
            )
    recall_summary = recall_by_window["0"]
    summary = {
        "scope": args.scope,
        "manual_files": len(manual_files),
        "generated_slices": len(dual_rows),
        "generated_function_bags": int(len(fit_rows)),
        "embedding_key": str(proto_embedding_key),
        "function_bag": {
            "enabled": bool(function_bag_enable),
            "agg": str(function_bag_agg),
            "eval_level": str(args.function_bag_eval_level),
            "num_bags": int(len(fit_rows)),
            "mean_bag_size": float(np.mean(bag_sizes)) if len(bag_sizes) else 0.0,
            "max_bag_size": int(np.max(bag_sizes)) if len(bag_sizes) else 0,
        },
        "manual_global_graph_dir": str(manual_global_graph_dir),
        "adaptive_boundary_refine": {
            "enabled": bool(args.adaptive_boundary_refine),
            "focus_quantile": float(args.adaptive_boundary_focus_quantile),
            "min_scale": float(args.adaptive_boundary_min_scale),
            "max_scale": float(args.adaptive_boundary_max_scale),
            "outside_scale": float(args.adaptive_boundary_outside_scale),
            "focus_mask_ratio": float(
                np.mean(refined.get("adaptive_focus_mask", np.ones_like(base_final, dtype=np.bool_)).astype(np.float32))
            )
            if len(base_final)
            else 0.0,
            "local_scale_mean": float(np.mean(refined.get("adaptive_local_scale", np.ones_like(base_final))))
            if len(base_final)
            else 0.0,
            "local_scale_std": float(np.std(refined.get("adaptive_local_scale", np.ones_like(base_final))))
            if len(base_final)
            else 0.0,
            "boundary_proximity_mean": float(np.mean(refined.get("boundary_proximity", np.zeros_like(base_final))))
            if len(base_final)
            else 0.0,
            "legacy_final_mean": float(np.mean(refined.get("legacy_final_risk", base_final))) if len(base_final) else 0.0,
            "adaptive_final_mean": float(np.mean(base_final)) if len(base_final) else 0.0,
        },
        "mechanism_risk": {
            "weight": w_mech,
            "top_quantile": float(args.mechanism_top_quantile),
            "semantic_gate": bool(args.mechanism_semantic_gate),
            "semantic_min_pairs": int(args.mechanism_semantic_min_pairs),
            "semantic_max_hops": int(args.mechanism_semantic_max_hops),
            "top_mask_ratio": float(np.mean(mech_top_mask.astype(np.float32))) if len(mech_top_mask) else 0.0,
            "semantic_mask_ratio": float(np.mean(sem_mask.astype(np.float32))) if len(sem_mask) else 0.0,
            "active_mask_ratio": float(np.mean(mech_active_mask.astype(np.float32))) if len(mech_active_mask) else 0.0,
            "raw_std": float(np.std(mechanism_raw)) if len(mechanism_raw) else 0.0,
            "score_std": float(np.std(mechanism_score)) if len(mechanism_score) else 0.0,
        },
        "family_aware": {
            "weight": w_family,
            "top_quantile": float(args.family_top_quantile),
            "top_mask_ratio": float(np.mean(family_top_mask.astype(np.float32))) if len(family_top_mask) else 0.0,
            "boundary_quantile": float(args.family_boundary_quantile),
            "min_samples": int(args.family_min_samples),
            "shrinkage_tau": float(args.family_shrinkage_tau),
            "ramp_width": int(args.family_ramp_width),
            "global_mix_floor": float(args.family_global_mix_floor),
            "radius_min_ratio": float(args.family_radius_min_ratio),
            "radius_max_ratio": float(args.family_radius_max_ratio),
            "num_families": float(family_diag.get("num_families", 0.0)),
            "small_family_ratio": float(family_diag.get("small_family_ratio", 0.0)),
            "global_fallback_ratio": float(family_diag.get("global_fallback_ratio", 0.0)),
            "raw_std": float(family_diag.get("raw_std", 0.0)),
            "score_std": float(family_diag.get("score_std", 0.0)),
            "score_mean": float(np.mean(family_score)) if len(family_score) else 0.0,
            "local_weight_mean": float(family_diag.get("local_weight_mean", 0.0)),
            "local_weight_std": float(family_diag.get("local_weight_std", 0.0)),
            "local_weight_min": float(family_diag.get("local_weight_min", 0.0)),
            "local_weight_max": float(family_diag.get("local_weight_max", 0.0)),
            "effective_local_ratio": float(family_diag.get("effective_local_ratio", 0.0)),
        },
        "hierarchical_prototypes": hier_diag,
        "score_head": score_head_diag,
        "mechanism_head": mechanism_head_pre_diag,
        "binary_head": binary_head_diag,
        "triage": triage_diag,
        "proto_resp_gray": proto_resp_gray_diag,
        "metamorphic_feature": metamorphic_feature_diag,
        "metamorphic_gray": metamorphic_gray_diag,
        "risk_rerank": {
            "mode": str(args.risk_rerank_mode),
            "topn": int(args.risk_rerank_topn),
            "file_penalty": float(args.risk_rerank_file_penalty),
            "filefn_penalty": float(args.risk_rerank_filefn_penalty),
            "fn_novelty_bonus": float(args.risk_rerank_fn_novelty_bonus),
            "fn_overlap_penalty": float(args.risk_rerank_fn_overlap_penalty),
            "max_per_filefn": int(args.risk_rerank_max_per_filefn),
            "max_per_file": int(args.risk_rerank_max_per_file),
            "file_repeat_power": float(args.risk_rerank_file_repeat_power),
            "family_penalty": float(args.risk_rerank_family_penalty),
            "family_novelty_bonus": float(args.risk_rerank_family_novelty_bonus),
            "aux_weight": float(args.risk_rerank_aux_weight),
            "max_per_family": int(args.risk_rerank_max_per_family),
            "warmup_topk": int(args.risk_rerank_warmup_topk),
            "warmup_file_cap": int(args.risk_rerank_warmup_file_cap),
            "warmup_filefn_cap": int(args.risk_rerank_warmup_filefn_cap),
        },
        "status_counts": {
            "high_risk": int(sum(1 for x in risk_rows if x["status"] == "high_risk")),
            "boundary": int(sum(1 for x in risk_rows if x["status"] == "boundary")),
            "normal": int(sum(1 for x in risk_rows if x["status"] == "normal")),
        },
        "binary_summary": {
            **binary_metrics,
            "target_mode": str(args.binary_target_mode),
            "target_window": int(args.binary_target_window),
            "eval_topn": int(binary_eval_topn),
        },
        "precision": float(binary_metrics.get("precision", 0.0)),
        "recall": float(binary_metrics.get("recall", 0.0)),
        "fpr": float(binary_metrics.get("fpr", 0.0)),
        "f1": float(binary_metrics.get("f1", 0.0)),
        "top5_risk": risk_rows[:5],
        "top5_line_risk": line_risk_rows[:5],
        "line_alignment": line_alignment_diag,
        "recall_summary": recall_summary,
        "recall_summary_by_line_window": recall_by_window,
        "line_risk_recall_summary_by_line_window": line_recall_by_window,
        "recall_summary_aligned_by_line_window": recall_by_window_aligned,
        "line_risk_recall_summary_aligned_by_line_window": line_recall_by_window_aligned,
        "line_match_windows": line_windows,
        "line_risk_topn": int(args.line_risk_topn),
        "line_risk_topk": line_topk_list,
        "line_risk_stage2": line_risk_diag,
        "artifacts": {
            "manual_list": str(manual_list_path),
            "normalized": str(normalized_path),
            "slices": str(slices_path),
            "dual_views": str(dual_view_path),
            "label_alignment_csv": str(line_alignment_diag.get("output_csv", "")),
            "risk_csv": str(risk_out),
            "line_risk_csv": str(line_risk_out),
            "binary_csv": str(binary_out),
        },
        "elapsed_sec": float(time.time() - t0),
    }
    save_json(summary_out, summary)
    print(f"[evaluate_manual_set] risk_csv={risk_out}")
    print(f"[evaluate_manual_set] binary_csv={binary_out}")
    print(f"[evaluate_manual_set] summary={summary_out}")


if __name__ == "__main__":
    main()
