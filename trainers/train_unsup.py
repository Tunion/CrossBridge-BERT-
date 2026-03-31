from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.energy_head_utils import (
    build_energy_feature_matrix,
    build_proto_margin_targets,
    normalize01,
    rank_normalize01,
    standardize_fit_transform,
)
from common.closure_feature_utils import CLOSURE_FEATURE_NAMES, compute_graph_closure_features
from common.group_score_calibration_utils import fit_group_score_calibration
from common.hierarchical_prototype_utils import (
    fit_hierarchical_local_prototypes,
    pack_local_bank,
    score_hierarchical_prototypes,
)
from common.io_utils import load_jsonl, save_json
from common.mechanism_head_utils import (
    MechanismSliceHeadModel,
    extract_mechanism_head_feature_rows,
    save_mechanism_head_model,
)
from common.metamorphic_feature_utils import (
    compute_metamorphic_instability_features,
    load_dual_model_ckpt,
)
from common.score_head_utils import apply_iforest_score_head
from models.boundary_refine import BoundaryRefiner, RefineWeights
from models.dual_view_gnn import GlobalLocalDualViewModel
from models.energy_head import EnergyHeadMLP
from models.prototype_head import MultiPrototypeHead

ROLE_KEYS = ("auth_constraint", "external_interaction", "state_change")
ROLE_PAIRS = (
    ("auth_constraint", "external_interaction"),
    ("external_interaction", "state_change"),
    ("auth_constraint", "state_change"),
)
CRITICAL_RELATIONS = {"constraint_on", "call_interaction", "dfg_dep", "data_dep", "cfg_next", "control_dep"}
MECHANISM_RELATIONS = CRITICAL_RELATIONS | {"contains", "ast_parent"}
KEY_NODE_TYPES = {"function", "statement", "condition", "call", "state_var", "data_object"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-4/5: prototype modeling + boundary refinement.")
    p.add_argument("--dual-view-input", type=str, default="data/slices/dual_views.jsonl")
    p.add_argument("--embedding-npz", type=str, default="outputs/results/dual_view_embeddings.npz")
    p.add_argument(
        "--embedding-key",
        type=str,
        default="z_joint",
        choices=["z_joint", "z_graph_joint", "z_text", "z_fused", "z_global", "z_full", "z_skeleton", "z_hybrid"],
        help="Embedding field used for prototype fitting and boundary refinement.",
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
        help="How to aggregate slice embeddings into a function-level bag embedding.",
    )
    p.set_defaults(function_bag_enable=False)
    p.add_argument("--prototype-k", type=int, default=8)
    p.add_argument("--hier-prototype-enable", dest="hier_prototype_enable", action="store_true")
    p.add_argument("--no-hier-prototype-enable", dest="hier_prototype_enable", action="store_false")
    p.add_argument("--hier-prototype-local-k", type=int, default=4)
    p.add_argument("--hier-prototype-family-min-samples", type=int, default=96)
    p.add_argument("--hier-prototype-shrinkage-tau", type=float, default=96.0)
    p.add_argument("--hier-prototype-global-mix-floor", type=float, default=0.25)
    p.add_argument("--hier-prototype-match-conf-floor", type=float, default=0.25)
    p.add_argument("--boundary-quantile", type=float, default=0.9)
    p.add_argument(
        "--mechanism-risk-weight",
        type=float,
        default=0.0,
        help="Blend weight for mechanism-consistency risk in final score. 0 keeps old behavior.",
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
        help="Blend weight for family-aware risk channel. 0 disables family-aware correction.",
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
    p.add_argument("--density-k", type=int, default=10)
    p.add_argument("--n-perturb", type=int, default=5)
    p.add_argument("--noise-std", type=float, default=0.02)
    p.add_argument(
        "--adaptive-boundary-refine",
        dest="adaptive_boundary_refine",
        action="store_true",
        help="Enable local adaptive boundary refinement near uncertain boundary band.",
    )
    p.add_argument(
        "--no-adaptive-boundary-refine",
        dest="adaptive_boundary_refine",
        action="store_false",
        help="Disable local adaptive boundary refinement.",
    )
    p.add_argument(
        "--adaptive-boundary-focus-quantile",
        type=float,
        default=0.65,
        help="Quantile on boundary-proximity used to select local refinement focus band.",
    )
    p.add_argument(
        "--adaptive-boundary-min-scale",
        type=float,
        default=0.85,
        help="Minimum local correction scale within adaptive boundary refinement.",
    )
    p.add_argument(
        "--adaptive-boundary-max-scale",
        type=float,
        default=1.20,
        help="Maximum local correction scale within adaptive boundary refinement.",
    )
    p.add_argument(
        "--adaptive-boundary-outside-scale",
        type=float,
        default=1.00,
        help="Correction scale outside adaptive focus band. 0 means strict band gating.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--risk-out", type=str, default="outputs/results/unsup_final_risk_scores.csv")
    p.add_argument("--summary-out", type=str, default="outputs/results/unsup_final_summary.json")
    p.add_argument("--prototype-out", type=str, default="outputs/models/prototypes.npz")
    p.add_argument("--energy-head-enable", dest="energy_head_enable", action="store_true")
    p.add_argument("--no-energy-head-enable", dest="energy_head_enable", action="store_false")
    p.add_argument("--energy-head-hidden-dim", type=int, default=96)
    p.add_argument(
        "--energy-head-mode",
        type=str,
        default="full",
        choices=["full", "proto_hybrid", "proto_free_energy"],
        help="Feature construction mode for learned energy head.",
    )
    p.add_argument("--energy-head-free-energy-temp", type=float, default=0.35)
    p.add_argument(
        "--energy-head-free-energy-prior",
        type=str,
        default="uniform",
        choices=["uniform", "support", "sqrt_support"],
        help="Prototype prior used by proto_free_energy. 'support' and 'sqrt_support' weight mixture terms by cluster mass.",
    )
    p.add_argument("--energy-head-epochs", type=int, default=40)
    p.add_argument("--energy-head-batch-size", type=int, default=256)
    p.add_argument("--energy-head-lr", type=float, default=1e-3)
    p.add_argument("--energy-head-weight-decay", type=float, default=1e-4)
    p.add_argument("--energy-head-noise-std", type=float, default=0.05)
    p.add_argument("--energy-head-dropout", type=float, default=0.10)
    p.add_argument("--energy-head-out", type=str, default="outputs/models/energy_head.pt")
    p.add_argument("--mechanism-head-enable", dest="mechanism_head_enable", action="store_true")
    p.add_argument("--no-mechanism-head-enable", dest="mechanism_head_enable", action="store_false")
    p.add_argument("--mechanism-head-out", type=str, default="outputs/models/mechanism_head.pkl")
    p.add_argument("--mechanism-head-max-hops", type=int, default=6)
    p.add_argument("--mechanism-head-gate-quantile", type=float, default=0.55)
    p.add_argument("--mechanism-head-slice-quantile", type=float, default=0.95)
    p.add_argument("--mechanism-head-override-final-risk", dest="mechanism_head_override_final_risk", action="store_true")
    p.add_argument("--no-mechanism-head-override-final-risk", dest="mechanism_head_override_final_risk", action="store_false")
    p.add_argument(
        "--energy-head-loss-mode",
        type=str,
        default="proto_margin",
        choices=["softplus_neg", "proto_margin"],
        help="Training objective for the learned energy head.",
    )
    p.add_argument("--energy-head-pos-quantile", type=float, default=0.30)
    p.add_argument("--energy-head-neg-quantile", type=float, default=0.85)
    p.add_argument("--energy-head-rank-margin", type=float, default=0.35)
    p.add_argument("--energy-head-rank-weight", type=float, default=1.0)
    p.add_argument("--energy-head-calib-weight", type=float, default=0.50)
    p.add_argument("--energy-head-anchor-cls-weight", type=float, default=0.50)
    p.add_argument(
        "--energy-head-family-aware-anchor",
        dest="energy_head_family_aware_anchor",
        action="store_true",
        help="Select proto-margin anchor positives/negatives within family groups when possible.",
    )
    p.add_argument(
        "--no-energy-head-family-aware-anchor",
        dest="energy_head_family_aware_anchor",
        action="store_false",
    )
    p.add_argument("--energy-head-anchor-family-min-samples", type=int, default=64)
    p.add_argument(
        "--energy-head-anchor-metamorphic-enable",
        dest="energy_head_anchor_metamorphic_enable",
        action="store_true",
        help="Use metamorphic instability signals only for proto-margin anchor construction.",
    )
    p.add_argument(
        "--no-energy-head-anchor-metamorphic-enable",
        dest="energy_head_anchor_metamorphic_enable",
        action="store_false",
    )
    p.add_argument(
        "--energy-head-anchor-metamorphic-weight",
        type=float,
        default=0.12,
        help="Additional weight of metamorphic instability in proto-margin anchor score.",
    )
    p.add_argument(
        "--energy-head-include-metamorphic-features",
        dest="energy_head_include_metamorphic_features",
        action="store_true",
        help="Include metamorphic columns directly in energy-head input features.",
    )
    p.add_argument(
        "--no-energy-head-include-metamorphic-features",
        dest="energy_head_include_metamorphic_features",
        action="store_false",
        help="Do not include metamorphic columns directly in energy-head input features.",
    )
    p.add_argument(
        "--energy-head-include-text-aux",
        dest="energy_head_include_text_aux",
        action="store_true",
        help="Include text-view auxiliary signals (graph-text gap/norm ratio) in energy-head input features.",
    )
    p.add_argument(
        "--no-energy-head-include-text-aux",
        dest="energy_head_include_text_aux",
        action="store_false",
    )
    p.add_argument(
        "--energy-head-include-closure-features",
        dest="energy_head_include_closure_features",
        action="store_true",
        help="Include closure-completeness features derived from full/skeleton role connectivity in energy-head input features.",
    )
    p.add_argument(
        "--no-energy-head-include-closure-features",
        dest="energy_head_include_closure_features",
        action="store_false",
    )
    p.add_argument(
        "--energy-head-protofree-extended-features",
        dest="energy_head_protofree_extended_features",
        action="store_true",
        help="Include proto_resp_max/proto_resp_gap/proto_top2_norm_gap in proto_free_energy features.",
    )
    p.add_argument(
        "--no-energy-head-protofree-extended-features",
        dest="energy_head_protofree_extended_features",
        action="store_false",
    )
    p.add_argument(
        "--energy-head-score-teacher-enable",
        dest="energy_head_score_teacher_enable",
        action="store_true",
        help="Use IsolationForest score-head output as auxiliary teacher signal for energy-head ranking.",
    )
    p.add_argument(
        "--no-energy-head-score-teacher-enable",
        dest="energy_head_score_teacher_enable",
        action="store_false",
    )
    p.add_argument(
        "--energy-head-score-teacher-feature-enable",
        dest="energy_head_score_teacher_feature_enable",
        action="store_true",
        help="Also append score-head channels into energy-head features when score teacher is enabled.",
    )
    p.add_argument(
        "--no-energy-head-score-teacher-feature-enable",
        dest="energy_head_score_teacher_feature_enable",
        action="store_false",
    )
    p.add_argument(
        "--energy-head-score-teacher-weight",
        type=float,
        default=0.35,
        help="Blend weight of score-head rank percentile when constructing proto-margin ranking anchors.",
    )
    p.add_argument(
        "--energy-head-score-fit-scope",
        type=str,
        default="low_risk",
        choices=["all", "low_risk", "normal"],
        help="Fit scope for the score-head teacher.",
    )
    p.add_argument(
        "--energy-head-score-random-state",
        type=int,
        default=42,
        help="Random seed for the score-head teacher.",
    )
    p.add_argument(
        "--energy-head-score-calibration-enable",
        dest="energy_head_score_calibration_enable",
        action="store_true",
        help="Fit score calibration statistics from normal training rows for later energy-head score post-processing.",
    )
    p.add_argument(
        "--no-energy-head-score-calibration-enable",
        dest="energy_head_score_calibration_enable",
        action="store_false",
    )
    p.add_argument(
        "--energy-head-score-calibration-group-by",
        type=str,
        default="prototype",
        choices=["family", "prototype"],
        help="Grouping used when fitting energy-head score calibration statistics.",
    )
    p.add_argument(
        "--energy-head-score-calibration-fit-scope",
        type=str,
        default="normal",
        choices=["all", "normal", "low_risk", "boundary", "high_risk"],
        help="Which training rows are used to fit score calibration statistics.",
    )
    p.add_argument("--energy-head-score-calibration-min-samples", type=int, default=32)
    p.add_argument("--energy-head-score-calibration-shrinkage-tau", type=float, default=16.0)
    p.add_argument("--energy-head-score-calibration-global-mix-floor", type=float, default=0.20)
    p.add_argument("--energy-head-score-calibration-quantile-bins", type=int, default=257)
    p.add_argument("--gmm-head-enable", dest="gmm_head_enable", action="store_true")
    p.add_argument("--no-gmm-head-enable", dest="gmm_head_enable", action="store_false")
    p.add_argument("--gmm-head-out", type=str, default="outputs/models/gmm_head.pkl")
    p.add_argument("--gmm-head-components", type=int, default=8)
    p.add_argument(
        "--gmm-head-covariance-type",
        type=str,
        default="diag",
        choices=["diag", "full", "tied", "spherical"],
    )
    p.add_argument("--gmm-head-reg-covar", type=float, default=1e-4)
    p.add_argument("--gmm-head-max-iter", type=int, default=200)
    p.add_argument("--dual-model-ckpt", type=str, default="outputs/models/dual_view.pt")
    p.add_argument(
        "--metamorphic-global-graph-dir",
        type=str,
        default="",
        help="Optional whole-graph directory used when the dual model is global-local and metamorphic features are enabled.",
    )
    p.add_argument("--metamorphic-feature-enable", dest="metamorphic_feature_enable", action="store_true")
    p.add_argument("--no-metamorphic-feature-enable", dest="metamorphic_feature_enable", action="store_false")
    p.add_argument(
        "--metamorphic-feature-max-rows",
        type=int,
        default=5000,
        help="Maximum rows used to compute metamorphic instability features. 0 means all selected rows.",
    )
    p.add_argument(
        "--metamorphic-feature-select",
        type=str,
        default="boundary",
        choices=["boundary", "all"],
        help="Which rows receive metamorphic instability feature computation.",
    )
    p.add_argument(
        "--metamorphic-feature-boundary-quantile",
        type=float,
        default=0.80,
        help="Within selected rows, keep only samples whose boundary_proximity is in top-q quantile.",
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
    p.add_argument("--metamorphic-max-hops", type=int, default=4)
    p.add_argument(
        "--energy-head-stable-train",
        dest="energy_head_stable_train",
        action="store_true",
        help="Use deterministic full-batch CPU training with fixed pseudo-negatives for the energy head.",
    )
    p.add_argument(
        "--no-energy-head-stable-train",
        dest="energy_head_stable_train",
        action="store_false",
        help="Disable deterministic full-batch energy-head training.",
    )
    p.add_argument("--topk", type=int, default=100)
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
    p.set_defaults(
        adaptive_boundary_refine=True,
        energy_head_enable=False,
        mechanism_head_enable=True,
        hier_prototype_enable=False,
        mechanism_head_override_final_risk=True,
        energy_head_stable_train=True,
        energy_head_family_aware_anchor=False,
        energy_head_anchor_metamorphic_enable=False,
        energy_head_include_metamorphic_features=False,
        energy_head_include_text_aux=False,
        energy_head_include_closure_features=False,
        energy_head_protofree_extended_features=True,
        energy_head_score_teacher_enable=False,
        energy_head_score_teacher_feature_enable=None,
        energy_head_score_calibration_enable=False,
        gmm_head_enable=False,
        metamorphic_feature_enable=False,
    )
    return p.parse_args()


def load_embeddings(npz_path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(npz_path)
    keys = set(arr.keys())
    z_keys = [k for k in keys if str(k).startswith("z_")]
    if {"z_full", "z_skeleton", "z_joint"}.issubset(keys):
        return {str(k): arr[str(k)].astype(np.float32) for k in z_keys}
    if "embedding" in keys:
        emb = arr["embedding"].astype(np.float32)
        return {"z_full": emb, "z_skeleton": emb, "z_joint": emb}
    raise ValueError(f"Unsupported embedding file keys: {sorted(keys)}")


def normalize01(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if len(arr) == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo + 1e-8)


def adaptive_mechanism_score(raw: np.ndarray, std_threshold: float = 0.05) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    if len(arr) == 0:
        return arr
    std = float(np.std(arr))
    if std >= float(std_threshold):
        return normalize01(arr)
    q1 = float(np.quantile(arr, 0.25))
    q3 = float(np.quantile(arr, 0.75))
    iqr = max(q3 - q1, 0.0)
    if iqr < 1e-6:
        return np.clip(arr, 0.0, 1.0)
    center = float(np.median(arr))
    scale = max(iqr, 0.02)
    z = np.clip((arr - center) / scale, -6.0, 6.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def top_quantile_mask(base: np.ndarray, quantile: float) -> np.ndarray:
    arr = np.asarray(base, dtype=np.float32)
    if len(arr) == 0:
        return np.zeros((0,), dtype=bool)
    q = float(quantile)
    if q <= 0.0 or q >= 1.0:
        return np.ones_like(arr, dtype=bool)
    thr = float(np.quantile(arr, q))
    return arr >= thr


def blend_mechanism_risk(
    base: np.ndarray,
    mechanism: np.ndarray,
    weight: float,
    active_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    base_arr = np.asarray(base, dtype=np.float32)
    mech_arr = np.asarray(mechanism, dtype=np.float32)
    if len(base_arr) == 0 or len(mech_arr) == 0:
        return base_arr
    w = float(min(1.0, max(0.0, weight)))
    if w <= 0.0:
        return base_arr
    if active_mask is None:
        mask = np.ones_like(base_arr, dtype=bool)
    else:
        mask = np.asarray(active_mask).astype(bool)
        if len(mask) != len(base_arr):
            mask = np.ones_like(base_arr, dtype=bool)
    center = float(np.median(mech_arr))
    delta = mech_arr - center
    # Gate correction by base risk confidence: low-base samples receive little positive lift.
    base_norm = normalize01(base_arr)
    gate = np.maximum(base_norm, 0.05)
    corr = w * gate * delta
    corr = np.where(mask, corr, 0.0).astype(np.float32)
    blended = base_arr + corr
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def blend_family_risk(
    base: np.ndarray,
    family_score: np.ndarray,
    weight: float,
    active_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    return blend_mechanism_risk(
        base=base,
        mechanism=family_score,
        weight=weight,
        active_mask=active_mask,
    )


def role_presence(graph: Dict[str, Any]) -> Dict[str, bool]:
    out = {k: False for k in ROLE_KEYS}
    for n in graph.get("nodes", []):
        for r in n.get("roles", []) or []:
            if r in out:
                out[r] = True
    return out


def pair_count_from_presence(pres: Dict[str, bool]) -> int:
    return sum(1 for a, b in ROLE_PAIRS if bool(pres.get(a, False)) and bool(pres.get(b, False)))


def role_node_sets(graph: Dict[str, Any]) -> Dict[str, Set[int]]:
    out: Dict[str, Set[int]] = {k: set() for k in ROLE_KEYS}
    for n in graph.get("nodes", []):
        nid = int(n.get("id", -1))
        if nid < 0:
            continue
        for r in n.get("roles", []) or []:
            if r in out:
                out[r].add(nid)
    return out


def node_role_flag_map(graph: Dict[str, Any]) -> Dict[int, bool]:
    out: Dict[int, bool] = {}
    for n in graph.get("nodes", []):
        nid = int(n.get("id", -1))
        if nid < 0:
            continue
        roles = set(n.get("roles", []) or [])
        out[nid] = any(r in roles for r in ROLE_KEYS)
    return out


def critical_edge_count(graph: Dict[str, Any]) -> int:
    role_map = node_role_flag_map(graph)
    cnt = 0
    for e in graph.get("edges", []):
        if e.get("type") not in CRITICAL_RELATIONS:
            continue
        s = int(e.get("src", -1))
        d = int(e.get("dst", -1))
        if role_map.get(s, False) or role_map.get(d, False):
            cnt += 1
    return cnt


def safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    if den <= 0:
        return float(default)
    return float(num) / float(den)


def build_undirected_adj(graph: Dict[str, Any], rel_types: Set[str]) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = collections.defaultdict(set)
    for e in graph.get("edges", []):
        if e.get("type") not in rel_types:
            continue
        s = int(e.get("src", -1))
        d = int(e.get("dst", -1))
        if s < 0 or d < 0:
            continue
        adj[s].add(d)
        adj[d].add(s)
    return adj


def shortest_path_len_between_sets(
    adj: Dict[int, Set[int]],
    src_set: Set[int],
    dst_set: Set[int],
    max_hops: int,
) -> Optional[int]:
    if not src_set or not dst_set:
        return None
    if src_set.intersection(dst_set):
        return 0
    q: collections.deque[Tuple[int, int]] = collections.deque((s, 0) for s in src_set)
    seen: Set[int] = set(src_set)
    while q:
        cur, dep = q.popleft()
        if dep >= max_hops:
            continue
        for nb in adj.get(cur, set()):
            if nb in seen:
                continue
            nd = dep + 1
            if nb in dst_set:
                return nd
            seen.add(nb)
            q.append((nb, nd))
    return None


def semantic_closed_loop_mask(
    rows: List[Dict[str, Any]],
    min_connected_pairs: int = 2,
    max_hops: int = 6,
) -> np.ndarray:
    out: List[bool] = []
    pair_need = max(1, int(min_connected_pairs))
    hop_cap = max(1, int(max_hops))
    for row in rows:
        full_graph = row.get("full_graph", {}) or {}
        rnodes = role_node_sets(full_graph)
        if not all(bool(rnodes.get(k)) for k in ROLE_KEYS):
            out.append(False)
            continue
        adj = build_undirected_adj(full_graph, MECHANISM_RELATIONS)
        connected = 0
        for a, b in ROLE_PAIRS:
            plen = shortest_path_len_between_sets(
                adj=adj,
                src_set=rnodes.get(a, set()),
                dst_set=rnodes.get(b, set()),
                max_hops=hop_cap,
            )
            if plen is not None:
                connected += 1
        out.append(connected >= pair_need)
    return np.asarray(out, dtype=bool)


def normalized_l1_hist_diff(a_hist: Dict[str, int], b_hist: Dict[str, int], keys: Set[str]) -> float:
    key_union = set(keys) | set(a_hist.keys()) | set(b_hist.keys())
    if not key_union:
        return 0.0
    a_total = float(sum(max(0, int(a_hist.get(k, 0))) for k in key_union))
    b_total = float(sum(max(0, int(b_hist.get(k, 0))) for k in key_union))
    if a_total <= 0 and b_total <= 0:
        return 0.0
    diff = 0.0
    for k in key_union:
        pa = safe_ratio(float(a_hist.get(k, 0)), a_total, default=0.0) if a_total > 0 else 0.0
        pb = safe_ratio(float(b_hist.get(k, 0)), b_total, default=0.0) if b_total > 0 else 0.0
        diff += abs(pa - pb)
    return float(min(1.0, max(0.0, 0.5 * diff)))


def node_type_hist(graph: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for n in graph.get("nodes", []):
        t = str(n.get("type", "unknown"))
        out[t] = out.get(t, 0) + 1
    return out


def edge_type_hist(graph: Dict[str, Any], keep_types: Set[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for e in graph.get("edges", []):
        t = str(e.get("type", "unknown"))
        if t not in keep_types:
            continue
        out[t] = out.get(t, 0) + 1
    return out


def compute_mechanism_risk(rows: List[Dict[str, Any]]) -> np.ndarray:
    vals: List[float] = []
    for row in rows:
        full_graph = row.get("full_graph", {}) or {}
        skel_graph = row.get("skeleton_graph", {}) or {}
        full_role_nodes = role_node_sets(full_graph)
        skel_role_nodes = role_node_sets(skel_graph)
        full_pres = {r: bool(full_role_nodes[r]) for r in ROLE_KEYS}
        skel_pres = {r: bool(skel_role_nodes[r]) for r in ROLE_KEYS}
        skel_meta = (skel_graph.get("meta", {}) or {})

        full_role_count = sum(1 for r in ROLE_KEYS if full_pres[r])
        has_role_signal = full_role_count > 0
        role_completeness_loss = (
            1.0 - safe_ratio(float(full_role_count), float(len(ROLE_KEYS)), default=0.0)
            if has_role_signal
            else 0.0
        )

        full_adj = build_undirected_adj(full_graph, MECHANISM_RELATIONS)
        expected_pairs = 0
        connected_pairs = 0
        length_penalties: List[float] = []
        max_hops = 6
        for a, b in ROLE_PAIRS:
            src = full_role_nodes[a]
            dst = full_role_nodes[b]
            if not src or not dst:
                continue
            expected_pairs += 1
            plen = shortest_path_len_between_sets(full_adj, src, dst, max_hops=max_hops)
            if plen is None:
                continue
            connected_pairs += 1
            length_penalties.append(min(1.0, float(plen) / float(max_hops)))
        has_pair_signal = expected_pairs > 0
        pair_disconnect_loss = (
            1.0 - safe_ratio(float(connected_pairs), float(expected_pairs), default=1.0)
            if has_pair_signal
            else 0.0
        )
        path_length_loss = float(np.mean(length_penalties)) if length_penalties else (1.0 if has_pair_signal else 0.0)

        role_node_count = sum(len(v) for v in full_role_nodes.values())
        full_nodes = max(1, int(len(full_graph.get("nodes", []))))
        role_focus_loss = (
            1.0 - min(1.0, safe_ratio(float(role_node_count), float(full_nodes), default=0.0))
            if has_role_signal
            else 0.0
        )

        role_counts = [float(len(full_role_nodes[r])) for r in ROLE_KEYS]
        total_role_nodes = sum(role_counts)
        if total_role_nodes <= 0 or not has_role_signal:
            role_balance_loss = 0.0
        else:
            probs = [c / total_role_nodes for c in role_counts if c > 0]
            if len(probs) <= 1:
                role_balance_loss = 1.0
            else:
                ent = -sum(p * math.log(max(p, 1e-8)) for p in probs)
                ent_max = math.log(float(len(probs)))
                role_balance_loss = 1.0 - safe_ratio(ent, ent_max, default=1.0)
        role_balance_loss = float(min(1.0, max(0.0, role_balance_loss)))

        role_retains: List[float] = []
        for r in ROLE_KEYS:
            full_cnt = len(full_role_nodes[r])
            if full_cnt <= 0:
                continue
            sk_cnt = len(skel_role_nodes[r])
            role_retains.append(min(1.0, safe_ratio(float(sk_cnt), float(full_cnt), default=0.0)))
        role_retain_loss = 1.0 - (float(np.mean(role_retains)) if role_retains else 1.0)
        if not has_pair_signal:
            role_retain_loss = 0.0

        full_node_hist = node_type_hist(full_graph)
        skel_node_hist = node_type_hist(skel_graph)
        node_type_drift = normalized_l1_hist_diff(full_node_hist, skel_node_hist, KEY_NODE_TYPES) if has_pair_signal else 0.0

        full_edge_hist = edge_type_hist(full_graph, MECHANISM_RELATIONS)
        skel_edge_hist = edge_type_hist(skel_graph, MECHANISM_RELATIONS)
        edge_type_drift = normalized_l1_hist_diff(full_edge_hist, skel_edge_hist, MECHANISM_RELATIONS) if has_pair_signal else 0.0

        if expected_pairs > 0:
            if isinstance(skel_meta.get("mechanism_links"), (int, float)):
                got_links = int(max(0, int(skel_meta["mechanism_links"])))
            else:
                got_links = pair_count_from_presence(skel_pres)
            link_cov = min(1.0, safe_ratio(float(got_links), float(expected_pairs), default=0.0))
            mechanism_link_loss = 1.0 - link_cov
        else:
            mechanism_link_loss = 0.0

        if has_pair_signal:
            full_edges = max(1, int(len(full_graph.get("edges", []))))
            skel_nodes = int(len(skel_graph.get("nodes", [])))
            skel_edges = int(len(skel_graph.get("edges", [])))
            node_ratio = min(1.0, safe_ratio(float(skel_nodes), float(full_nodes), default=1.0))
            edge_ratio = min(1.0, safe_ratio(float(skel_edges), float(full_edges), default=1.0))
            compression_loss = 0.5 * node_ratio + 0.5 * edge_ratio
        else:
            compression_loss = 0.0

        if has_pair_signal:
            full_crit = critical_edge_count(full_graph)
            skel_crit = critical_edge_count(skel_graph)
            edge_cov = min(1.0, safe_ratio(float(skel_crit), float(full_crit), default=1.0))
            critical_edge_loss = 1.0 - edge_cov
        else:
            critical_edge_loss = 0.0

        risk = (
            0.18 * role_completeness_loss
            + 0.15 * pair_disconnect_loss
            + 0.08 * path_length_loss
            + 0.14 * role_focus_loss
            + 0.07 * role_balance_loss
            + 0.10 * role_retain_loss
            + 0.06 * node_type_drift
            + 0.06 * edge_type_drift
            + 0.08 * mechanism_link_loss
            + 0.05 * compression_loss
            + 0.03 * critical_edge_loss
        )
        vals.append(float(min(1.0, max(0.0, risk))))
    return np.asarray(vals, dtype=np.float32)


def _canonical_fn(function_names: str) -> str:
    toks = [x.strip().lower() for x in str(function_names).split("|") if x.strip()]
    return toks[0] if toks else "<unknown_fn>"


def _function_key_set(function_names: str) -> Set[str]:
    toks = {x.strip().lower() for x in str(function_names).split("|") if x.strip()}
    return toks if toks else {"<unknown_fn>"}


def _file_key_from_row(row: Dict[str, Any]) -> str:
    return str(row.get("relative_source_path") or row.get("source_path") or row.get("contract_id") or "<unknown_file>")


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


def _function_bag_id(row: Dict[str, Any]) -> str:
    file_key = _file_key_from_row(row)
    fn_key = _canonical_fn(str(row.get("function_names", "")))
    return f"{file_key}:::{fn_key}"


def _aggregate_group_embeddings(arr: np.ndarray, groups: List[np.ndarray], mode: str) -> np.ndarray:
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


def _aggregate_group_scalar(arr: np.ndarray, groups: List[np.ndarray], reduce: str = "mean") -> np.ndarray:
    if len(groups) <= 0:
        return np.zeros((0,), dtype=np.float32)
    out: List[float] = []
    red = str(reduce or "mean").strip().lower()
    src = np.asarray(arr, dtype=np.float32)
    for idx in groups:
        vals = src[idx]
        if red == "max":
            out.append(float(np.max(vals)))
        elif red == "min":
            out.append(float(np.min(vals)))
        else:
            out.append(float(np.mean(vals)))
    return np.asarray(out, dtype=np.float32)


def build_function_bag_view(
    rows: List[Dict[str, Any]],
    emb_main: np.ndarray,
    emb_full: np.ndarray,
    emb_skel: np.ndarray,
    mechanism_raw: np.ndarray,
    agg_mode: str = "mean",
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    group_to_indices: Dict[str, List[int]] = collections.OrderedDict()
    for i, row in enumerate(rows):
        group_to_indices.setdefault(_function_bag_id(row), []).append(i)
    groups = [np.asarray(v, dtype=np.int64) for v in group_to_indices.values()]
    row_to_group = np.zeros((len(rows),), dtype=np.int64)
    bag_rows: List[Dict[str, Any]] = []
    for gid, idx in enumerate(groups):
        row_to_group[idx] = gid
        first = rows[int(idx[0])]
        fn_key = _canonical_fn(str(first.get("function_names", "")))
        file_key = _file_key_from_row(first)
        bag_rows.append(
            {
                "slice_id": f"bag::{gid}",
                "contract_id": first.get("contract_id"),
                "relative_source_path": file_key,
                "source_path": file_key,
                "function_names": [fn_key],
                "function_bag_id": f"{file_key}:::{fn_key}",
                "function_bag_size": int(len(idx)),
            }
        )
    bag_main = _aggregate_group_embeddings(emb_main, groups, agg_mode)
    bag_full = _aggregate_group_embeddings(emb_full, groups, agg_mode)
    bag_skel = _aggregate_group_embeddings(emb_skel, groups, agg_mode)
    bag_mech = _aggregate_group_scalar(mechanism_raw, groups, reduce="max")
    bag_view_gap = np.linalg.norm(bag_full - bag_skel, axis=1).astype(np.float32) if len(bag_full) else np.zeros((0,), dtype=np.float32)
    bag_sizes = np.asarray([len(x) for x in groups], dtype=np.int32)
    return bag_rows, bag_main, bag_full, bag_skel, bag_mech, row_to_group, bag_sizes


def compute_family_aware_risk(
    rows: List[Dict[str, Any]],
    embeddings: np.ndarray,
    proto_margin: np.ndarray,
    min_samples: int = 8,
    boundary_quantile: float = 0.9,
    shrinkage_tau: float = 16.0,
    ramp_width: int = 6,
    global_mix_floor: float = 0.20,
    local_radius_min_ratio: float = 0.60,
    local_radius_max_ratio: float = 1.80,
) -> Tuple[np.ndarray, Dict[str, float]]:
    n = int(len(rows))
    if n <= 0:
        return np.zeros((0,), dtype=np.float32), {
            "num_families": 0.0,
            "small_family_ratio": 0.0,
            "global_fallback_ratio": 0.0,
            "raw_std": 0.0,
            "score_std": 0.0,
            "local_weight_mean": 0.0,
            "local_weight_std": 0.0,
            "local_weight_min": 0.0,
            "local_weight_max": 0.0,
            "effective_local_ratio": 0.0,
        }
    z = np.asarray(embeddings, dtype=np.float32)
    margin = np.asarray(proto_margin, dtype=np.float32)
    if len(z) != n:
        m = min(n, len(z))
        n = m
        z = z[:m]
        margin = margin[:m]
        rows = rows[:m]

    fam_keys = [_family_key_from_file(_file_key_from_row(r)) for r in rows]
    fam_to_idx: Dict[str, List[int]] = collections.defaultdict(list)
    for i, fk in enumerate(fam_keys):
        fam_to_idx[fk].append(i)

    q = float(min(0.99, max(0.5, boundary_quantile)))
    g_center = np.mean(z, axis=0).astype(np.float32)
    g_dist = np.linalg.norm(z - g_center[None, :], axis=1).astype(np.float32)
    g_radius = float(np.quantile(g_dist, q)) if len(g_dist) else 1.0
    if g_radius < 1e-6:
        g_radius = 1.0

    g_med = float(np.median(margin)) if len(margin) else 0.0
    g_mad = float(np.median(np.abs(margin - g_med))) if len(margin) else 0.0
    g_scale = max(1e-6, 1.4826 * g_mad)
    g_margin_score = np.clip((margin - g_med) / g_scale, -3.0, 6.0)
    g_margin_score = (g_margin_score + 3.0) / 9.0
    g_margin_score = np.clip(g_margin_score, 0.0, 1.0).astype(np.float32)

    raw = np.zeros((n,), dtype=np.float32)
    min_n = max(1, int(min_samples))
    tau = max(1e-6, float(shrinkage_tau))
    width = max(1, int(ramp_width))
    g_floor = float(min(0.95, max(0.0, global_mix_floor)))
    rad_min_ratio = float(max(0.05, local_radius_min_ratio))
    rad_max_ratio = float(max(rad_min_ratio, local_radius_max_ratio))
    r_low = float(max(1e-6, g_radius * rad_min_ratio))
    r_high = float(max(r_low, g_radius * rad_max_ratio))

    small_fam_cnt = 0
    global_fallback_cnt = 0
    local_weight_arr = np.zeros((n,), dtype=np.float32)
    for _, idx_list in fam_to_idx.items():
        idx = np.asarray(idx_list, dtype=np.int64)
        nf = int(len(idx))
        if nf < min_n:
            small_fam_cnt += 1

        global_dist_score = g_dist[idx] / float(g_radius)
        global_margin_local = g_margin_score[idx]
        global_score = 0.7 * global_dist_score + 0.3 * global_margin_local

        if nf <= 1:
            raw[idx] = global_score
            global_fallback_cnt += nf
            continue

        zf = z[idx]
        mf = margin[idx]
        c_local = np.mean(zf, axis=0).astype(np.float32)
        dist_local = np.linalg.norm(zf - c_local[None, :], axis=1).astype(np.float32)
        r_local = float(np.quantile(dist_local, q)) if len(dist_local) else g_radius
        if r_local < 1e-6:
            r_local = g_radius
        r_local = float(np.clip(r_local, r_low, r_high))
        local_dist_score = dist_local / float(max(1e-6, r_local))

        m_med_local = float(np.median(mf)) if len(mf) else g_med
        m_mad_local = float(np.median(np.abs(mf - m_med_local))) if len(mf) else g_mad
        m_scale_local = max(1e-6, 1.4826 * m_mad_local)

        size_weight = float(nf) / float(nf + tau)
        ramp = float(np.clip((float(nf - min_n + 1)) / float(width), 0.0, 1.0))
        local_weight = float(np.clip(size_weight * ramp, 0.0, 1.0 - g_floor))

        m_med = (1.0 - local_weight) * g_med + local_weight * m_med_local
        m_scale = (1.0 - local_weight) * g_scale + local_weight * m_scale_local
        m_scale = max(1e-6, float(m_scale))

        local_margin_score = np.clip((mf - m_med) / m_scale, -3.0, 6.0)
        local_margin_score = (local_margin_score + 3.0) / 9.0
        local_margin_score = np.clip(local_margin_score, 0.0, 1.0).astype(np.float32)

        local_score = 0.7 * local_dist_score + 0.3 * local_margin_score
        raw[idx] = (1.0 - local_weight) * global_score + local_weight * local_score
        local_weight_arr[idx] = float(local_weight)
        if local_weight <= 1e-6:
            global_fallback_cnt += nf

    score = adaptive_mechanism_score(raw, std_threshold=0.05)
    local_nonzero_ratio = float(np.mean((local_weight_arr > 1e-6).astype(np.float32))) if len(local_weight_arr) else 0.0
    diag = {
        "num_families": float(len(fam_to_idx)),
        "small_family_ratio": float(safe_ratio(float(small_fam_cnt), float(max(1, len(fam_to_idx))), default=0.0)),
        "global_fallback_ratio": float(safe_ratio(float(global_fallback_cnt), float(max(1, n)), default=0.0)),
        "raw_std": float(np.std(raw)) if len(raw) else 0.0,
        "score_std": float(np.std(score)) if len(score) else 0.0,
        "local_weight_mean": float(np.mean(local_weight_arr)) if len(local_weight_arr) else 0.0,
        "local_weight_std": float(np.std(local_weight_arr)) if len(local_weight_arr) else 0.0,
        "local_weight_min": float(np.min(local_weight_arr)) if len(local_weight_arr) else 0.0,
        "local_weight_max": float(np.max(local_weight_arr)) if len(local_weight_arr) else 0.0,
        "effective_local_ratio": local_nonzero_ratio,
    }
    return score.astype(np.float32), diag


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
            file_key = _file_key_from_row(row)
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
        file_key = _file_key_from_row(row)
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
        fk = _file_key_from_row(row)
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

            avg_group_cnt = (
                float(sum(file_fn_cnt.get((fk, fn), 0) for fn in fn_set)) / float(max(1, len(fn_set)))
            )
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
        fk = _file_key_from_row(row)
        fam = _family_key_from_file(fk)
        fn_set = _function_key_set(str(row.get("function_names", "")))
        base_raw.append(float(row.get(score_key, 0.0)))
        # Auxiliary channel: view + mechanism + proto consistency.
        aux_val = (
            0.45 * float(row.get("view_score", 0.0))
            + 0.35 * float(row.get("mechanism_risk", 0.0))
            + 0.20 * float(row.get("proto_score", 0.0))
        )
        aux_raw.append(aux_val)
        meta.append((fk, fam, fn_set))

    base_norm = normalize01(np.asarray(base_raw, dtype=np.float32))
    aux_norm = normalize01(np.asarray(aux_raw, dtype=np.float32))

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
        file_key = _file_key_from_row(row)
        fn_key = _canonical_fn(str(row.get("function_names", "")))
        group_key = (file_key, fn_key)
        if file_cnt.get(file_key, 0) < file_cap and file_fn_cnt.get(group_key, 0) < filefn_cap:
            kept.append(row)
            file_cnt[file_key] = file_cnt.get(file_key, 0) + 1
            file_fn_cnt[group_key] = file_fn_cnt.get(group_key, 0) + 1
        else:
            delayed.append(row)
    return kept + delayed + tail


def train_energy_head_from_rows(
    rows: List[Dict[str, Any]],
    embeddings: np.ndarray,
    text_embeddings: Optional[np.ndarray],
    graph_embeddings: Optional[np.ndarray],
    prototype_centers: np.ndarray,
    prototype_radii: np.ndarray,
    prototype_weights: Optional[np.ndarray],
    args: argparse.Namespace,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], np.ndarray]:
    if not bool(getattr(args, "energy_head_enable", False)) or len(rows) <= 0:
        return None, {"enabled": False}, np.zeros((len(rows),), dtype=np.float32)

    score_teacher_enabled = bool(getattr(args, "energy_head_score_teacher_enable", False))
    score_teacher_feature_flag = getattr(args, "energy_head_score_teacher_feature_enable", None)
    include_score_teacher_features = (
        score_teacher_enabled if score_teacher_feature_flag is None else bool(score_teacher_feature_flag)
    )

    score_teacher_diag: Dict[str, Any] = {"enabled": False}
    if score_teacher_enabled:
        score_head_risk, score_teacher_diag = apply_iforest_score_head(
            rows,
            fit_scope=str(getattr(args, "energy_head_score_fit_scope", "low_risk")),
            random_state=int(getattr(args, "energy_head_score_random_state", args.seed)),
        )
        score_head_rank_pct = rank_normalize01(score_head_risk).astype(np.float32)
        for i, row in enumerate(rows[: len(score_head_risk)]):
            row["score_head_risk"] = float(score_head_risk[i])
            row["score_head_rank_pct"] = float(score_head_rank_pct[i])

    feat, cols, feat_diag = build_energy_feature_matrix(
        rows,
        embeddings,
        mode=str(getattr(args, "energy_head_mode", "full")),
        include_metamorphic=bool(getattr(args, "energy_head_include_metamorphic_features", False)),
        include_score_head=bool(include_score_teacher_features),
        include_text_aux_features=bool(getattr(args, "energy_head_include_text_aux", False)),
        include_closure_features=bool(getattr(args, "energy_head_include_closure_features", False)),
        include_protofree_extended_features=bool(getattr(args, "energy_head_protofree_extended_features", True)),
        text_embeddings=None if text_embeddings is None else np.asarray(text_embeddings, dtype=np.float32),
        graph_embeddings=None if graph_embeddings is None else np.asarray(graph_embeddings, dtype=np.float32),
        prototype_centers=np.asarray(prototype_centers, dtype=np.float32),
        prototype_radii=np.asarray(prototype_radii, dtype=np.float32),
        prototype_weights=None if prototype_weights is None else np.asarray(prototype_weights, dtype=np.float32),
        free_energy_temp=float(getattr(args, "energy_head_free_energy_temp", 0.35)),
    )
    if feat.shape[0] < 64 or feat.shape[1] <= 0:
        return None, {
            "enabled": False,
            "reason": "insufficient_rows_or_features",
            "num_rows": int(feat.shape[0]),
            "feature_dim": int(feat.shape[1]) if feat.ndim == 2 else 0,
        }, np.zeros((len(rows),), dtype=np.float32)

    feat_std, feat_mean, feat_scale = standardize_fit_transform(feat)
    n, d = feat_std.shape
    embed_dim = int(feat_diag.get("embedding_dim", 0))
    loss_mode = str(getattr(args, "energy_head_loss_mode", "softplus_neg")).strip().lower()
    stable_train = bool(getattr(args, "energy_head_stable_train", False))
    device = torch.device("cpu" if stable_train else ("cuda" if torch.cuda.is_available() else "cpu"))
    if stable_train:
        torch.manual_seed(int(args.seed))
    model = EnergyHeadMLP(
        input_dim=int(d),
        hidden_dim=int(args.energy_head_hidden_dim),
        dropout=0.0 if stable_train else float(args.energy_head_dropout),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.energy_head_lr),
        weight_decay=float(args.energy_head_weight_decay),
    )
    x_all = torch.from_numpy(feat_std).float().to(device)
    batch_size = int(max(32, args.energy_head_batch_size))
    rng = np.random.RandomState(int(args.seed))
    epochs = int(max(1, args.energy_head_epochs))
    noise_std = float(max(1e-5, args.energy_head_noise_std))
    loss_hist: List[float] = []
    model.train()
    anchor_diag: Dict[str, Any] = {}
    if loss_mode == "proto_margin":
        anchor_target_np, pos_mask_np, neg_mask_np, anchor_diag = build_proto_margin_targets(
            rows,
            n,
            pos_quantile=float(args.energy_head_pos_quantile),
            neg_quantile=float(args.energy_head_neg_quantile),
            family_aware=bool(getattr(args, "energy_head_family_aware_anchor", False)),
            family_min_samples=int(getattr(args, "energy_head_anchor_family_min_samples", 64)),
            metamorphic_weight=(
                float(getattr(args, "energy_head_anchor_metamorphic_weight", 0.0))
                if bool(getattr(args, "energy_head_anchor_metamorphic_enable", False))
                else 0.0
            ),
        )
        rank_anchor_np = anchor_target_np.astype(np.float32, copy=True)
        if score_teacher_enabled:
            score_rank_np = np.asarray([float(r.get("score_head_rank_pct", 0.0)) for r in rows[:n]], dtype=np.float32)
            teacher_weight = float(min(1.0, max(0.0, getattr(args, "energy_head_score_teacher_weight", 0.35))))
            rank_anchor_np = normalize01(
                (1.0 - teacher_weight) * anchor_target_np + teacher_weight * score_rank_np
            ).astype(np.float32)
            q_pos = float(np.quantile(rank_anchor_np, float(min(max(args.energy_head_pos_quantile, 0.01), 0.49))))
            q_neg = float(np.quantile(rank_anchor_np, float(min(max(args.energy_head_neg_quantile, 0.51), 0.99))))
            pos_mask_np = rank_anchor_np <= q_pos
            neg_mask_np = rank_anchor_np >= q_neg
            min_anchor = min(64, max(16, n // 50))
            if int(np.sum(pos_mask_np)) < min_anchor:
                order = np.argsort(rank_anchor_np)
                pos_mask_np = np.zeros((n,), dtype=bool)
                pos_mask_np[order[:min_anchor]] = True
            if int(np.sum(neg_mask_np)) < min_anchor:
                order = np.argsort(rank_anchor_np)
                neg_mask_np = np.zeros((n,), dtype=bool)
                neg_mask_np[order[-min_anchor:]] = True
            anchor_diag["score_teacher"] = {
                "enabled": True,
                "weight": teacher_weight,
                "score_mean": float(np.mean(score_rank_np)) if len(score_rank_np) else 0.0,
                "score_std": float(np.std(score_rank_np)) if len(score_rank_np) else 0.0,
                "rank_anchor_mean": float(np.mean(rank_anchor_np)) if len(rank_anchor_np) else 0.0,
                "rank_anchor_std": float(np.std(rank_anchor_np)) if len(rank_anchor_np) else 0.0,
                "pos_count": int(np.sum(pos_mask_np)),
                "neg_count": int(np.sum(neg_mask_np)),
                "feature_enable": bool(include_score_teacher_features),
            }
        anchor_target = torch.from_numpy(anchor_target_np).float().to(device)
        pos_idx_np = np.flatnonzero(pos_mask_np)
        neg_idx_np = np.flatnonzero(neg_mask_np)
        rank_margin = float(max(1e-4, args.energy_head_rank_margin))
        rank_weight = float(max(0.0, args.energy_head_rank_weight))
        calib_weight = float(max(0.0, args.energy_head_calib_weight))
        anchor_cls_weight = float(max(0.0, args.energy_head_anchor_cls_weight))
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            e_all = model(x_all)
            anchor_prob = torch.sigmoid(e_all)
            calib_loss = F.mse_loss(anchor_prob, anchor_target)

            m = int(min(len(pos_idx_np), len(neg_idx_np)))
            rank_loss = torch.zeros((), device=device)
            anchor_cls_loss = torch.zeros((), device=device)
            if m > 0:
                pos_pick = pos_idx_np[rng.permutation(len(pos_idx_np))[:m]]
                neg_pick = neg_idx_np[rng.permutation(len(neg_idx_np))[:m]]
                pos_idx = torch.from_numpy(pos_pick).long().to(device)
                neg_idx = torch.from_numpy(neg_pick).long().to(device)
                e_pos = e_all[pos_idx]
                e_neg = e_all[neg_idx]
                rank_loss = F.relu(rank_margin + e_pos - e_neg).mean()
                logits = torch.cat([e_pos, e_neg], dim=0)
                labels = torch.cat(
                    [torch.zeros_like(e_pos), torch.ones_like(e_neg)],
                    dim=0,
                )
                anchor_cls_loss = F.binary_cross_entropy_with_logits(logits, labels)

            reg_loss = 1e-4 * e_all.pow(2).mean()
            loss = rank_weight * rank_loss + calib_weight * calib_loss + anchor_cls_weight * anchor_cls_loss + reg_loss
            loss.backward()
            opt.step()
            loss_hist.append(float(loss.detach().cpu()))
    elif stable_train:
        xneg_perm = x_all.clone()
        perm_idx_np = rng.permutation(n)
        if embed_dim > 0:
            perm_idx = torch.from_numpy(perm_idx_np).long().to(device)
            xneg_perm[:, :embed_dim] = x_all[perm_idx, :embed_dim]
        noise_np = rng.normal(loc=0.0, scale=noise_std, size=feat_std.shape).astype(np.float32)
        xneg_noise = x_all + torch.from_numpy(noise_np).to(device)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            e_pos = model(x_all)
            e_neg_perm = model(xneg_perm)
            e_neg_noise = model(xneg_noise)
            loss = (
                F.softplus(e_pos).mean()
                + 0.5 * F.softplus(-e_neg_perm).mean()
                + 0.5 * F.softplus(-e_neg_noise).mean()
                + 1e-4 * (e_pos.pow(2).mean() + e_neg_perm.pow(2).mean() + e_neg_noise.pow(2).mean())
            )
            loss.backward()
            opt.step()
            loss_hist.append(float(loss.detach().cpu()))
    else:
        torch_gen = torch.Generator(device=device)
        torch_gen.manual_seed(int(args.seed))
        for _ in range(epochs):
            perm = rng.permutation(n)
            loss_sum = 0.0
            batch_cnt = 0
            for start in range(0, n, batch_size):
                idx_np = perm[start : start + batch_size]
                xb = x_all[idx_np]
                if xb.shape[0] <= 1:
                    continue
                opt.zero_grad(set_to_none=True)
                xneg_perm = xb.clone()
                perm_idx = torch.randperm(xb.shape[0], generator=torch_gen, device=device)
                if embed_dim > 0:
                    xneg_perm[:, :embed_dim] = xb[perm_idx, :embed_dim]
                xneg_noise = xb + noise_std * torch.randn(
                    xb.shape,
                    generator=torch_gen,
                    device=device,
                    dtype=xb.dtype,
                )

                e_pos = model(xb)
                e_neg_perm = model(xneg_perm)
                e_neg_noise = model(xneg_noise)
                loss = (
                    F.softplus(e_pos).mean()
                    + 0.5 * F.softplus(-e_neg_perm).mean()
                    + 0.5 * F.softplus(-e_neg_noise).mean()
                    + 1e-4 * (e_pos.pow(2).mean() + e_neg_perm.pow(2).mean() + e_neg_noise.pow(2).mean())
                )
                loss.backward()
                opt.step()
                loss_sum += float(loss.detach().cpu())
                batch_cnt += 1
            loss_hist.append(float(loss_sum / max(1, batch_cnt)))

    model.eval()
    with torch.no_grad():
        energy = model(x_all).detach().cpu().numpy().astype(np.float32)
    score = normalize01(energy)
    for i, row in enumerate(rows[: len(score)]):
        row["energy_head_score"] = float(score[i])

    score_calibration = {"enabled": False}
    if bool(getattr(args, "energy_head_score_calibration_enable", False)):
        score_calibration = fit_group_score_calibration(
            rows[: len(score)],
            score_column="energy_head_score",
            score_values=score,
            fit_scope=str(getattr(args, "energy_head_score_calibration_fit_scope", "normal")),
            group_by=str(getattr(args, "energy_head_score_calibration_group_by", "prototype")),
            min_samples=int(getattr(args, "energy_head_score_calibration_min_samples", 32)),
            shrinkage_tau=float(getattr(args, "energy_head_score_calibration_shrinkage_tau", 16.0)),
            global_mix_floor=float(getattr(args, "energy_head_score_calibration_global_mix_floor", 0.20)),
            quantile_bins=int(getattr(args, "energy_head_score_calibration_quantile_bins", 257)),
        )

    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {
            "input_dim": int(d),
            "hidden_dim": int(args.energy_head_hidden_dim),
            "dropout": float(args.energy_head_dropout),
            "mode": str(getattr(args, "energy_head_mode", "full")),
            "loss_mode": str(loss_mode),
            "embedding_dim": int(embed_dim),
            "include_metamorphic_features": bool(getattr(args, "energy_head_include_metamorphic_features", False)),
            "include_score_head_features": bool(include_score_teacher_features),
            "include_text_aux_features": bool(getattr(args, "energy_head_include_text_aux", False)),
            "include_closure_features": bool(getattr(args, "energy_head_include_closure_features", False)),
            "include_protofree_extended_features": bool(getattr(args, "energy_head_protofree_extended_features", True)),
            "free_energy_temp": float(getattr(args, "energy_head_free_energy_temp", 0.35)),
            "free_energy_prior": str(getattr(args, "energy_head_free_energy_prior", "uniform")),
            "rank_margin": float(args.energy_head_rank_margin),
            "rank_weight": float(args.energy_head_rank_weight),
            "calib_weight": float(args.energy_head_calib_weight),
            "anchor_cls_weight": float(args.energy_head_anchor_cls_weight),
            "feature_names": cols,
            "feature_mean": feat_mean.astype(np.float32),
            "feature_scale": feat_scale.astype(np.float32),
        },
        "score_calibration": score_calibration,
    }
    diag = {
        "enabled": True,
        "device": str(device),
        "num_rows": int(n),
        "feature_dim": int(d),
        "embedding_dim": int(embed_dim),
        "mode": str(getattr(args, "energy_head_mode", "full")),
        "loss_mode": str(loss_mode),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": float(args.energy_head_lr),
        "weight_decay": float(args.energy_head_weight_decay),
        "noise_std": float(noise_std),
        "stable_train": bool(stable_train),
        "free_energy_temp": float(getattr(args, "energy_head_free_energy_temp", 0.35)),
        "free_energy_prior": str(getattr(args, "energy_head_free_energy_prior", "uniform")),
        "rank_margin": float(args.energy_head_rank_margin),
        "rank_weight": float(args.energy_head_rank_weight),
        "calib_weight": float(args.energy_head_calib_weight),
        "anchor_cls_weight": float(args.energy_head_anchor_cls_weight),
        "score_teacher_feature_enable": bool(include_score_teacher_features),
        "include_closure_features": bool(getattr(args, "energy_head_include_closure_features", False)),
        "include_protofree_extended_features": bool(getattr(args, "energy_head_protofree_extended_features", True)),
        "feature_names": cols,
        "loss_last": float(loss_hist[-1]) if loss_hist else 0.0,
        "loss_hist_tail": loss_hist[-5:],
        "score_mean": float(np.mean(score)) if len(score) else 0.0,
        "score_std": float(np.std(score)) if len(score) else 0.0,
        "anchor_diag": anchor_diag,
        "score_teacher": score_teacher_diag,
        "bag_aux": {
            "num_function_groups": int(feat_diag.get("num_function_groups", 0)),
            "mean_bag_size": float(feat_diag.get("mean_bag_size", 0.0)),
            "max_bag_size": int(feat_diag.get("max_bag_size", 0)),
        },
        "anchor_metamorphic": {
            "enabled": bool(getattr(args, "energy_head_anchor_metamorphic_enable", False)),
            "weight": float(getattr(args, "energy_head_anchor_metamorphic_weight", 0.0)),
        },
        "text_aux": {
            "enabled": bool(getattr(args, "energy_head_include_text_aux", False)),
            "available": bool(feat_diag.get("text_aux_available", False)),
            "graph_dim": int(feat_diag.get("text_aux_graph_dim", 0)),
            "text_dim": int(feat_diag.get("text_aux_text_dim", 0)),
        },
        "closure_features": {
            "enabled": bool(getattr(args, "energy_head_include_closure_features", False)),
        },
        "score_calibration": dict(score_calibration.get("diag", {}) or {"enabled": False}),
    }
    return ckpt, diag, score.astype(np.float32)


def train_gmm_head_from_embeddings(
    rows: List[Dict[str, Any]],
    embeddings: np.ndarray,
    embedding_key: str,
    args: argparse.Namespace,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], np.ndarray]:
    if not bool(getattr(args, "gmm_head_enable", False)) or len(rows) <= 0:
        return None, {"enabled": False}, np.zeros((len(rows),), dtype=np.float32)

    feat = np.asarray(embeddings, dtype=np.float32)
    if feat.ndim != 2 or feat.shape[0] <= 0 or feat.shape[1] <= 0:
        return None, {"enabled": False, "reason": "invalid_embeddings"}, np.zeros((len(rows),), dtype=np.float32)

    feat_std, feat_mean, feat_scale = standardize_fit_transform(feat)
    n, d = feat_std.shape
    n_components = int(max(1, min(int(getattr(args, "gmm_head_components", 8)), n)))
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=str(getattr(args, "gmm_head_covariance_type", "diag")),
        reg_covar=float(max(1e-8, getattr(args, "gmm_head_reg_covar", 1e-4))),
        max_iter=int(max(10, getattr(args, "gmm_head_max_iter", 200))),
        random_state=int(args.seed),
    )
    gmm.fit(feat_std)
    neg_log_likelihood = (-gmm.score_samples(feat_std)).astype(np.float32)
    score = normalize01(neg_log_likelihood).astype(np.float32)
    for i, row in enumerate(rows[: len(score)]):
        row["gmm_head_score"] = float(score[i])

    ckpt = {
        "type": "gmm_head",
        "config": {
            "embedding_key": str(embedding_key),
            "input_dim": int(d),
            "n_components": int(n_components),
            "covariance_type": str(gmm.covariance_type),
            "reg_covar": float(gmm.reg_covar),
            "max_iter": int(gmm.max_iter),
            "feature_mean": feat_mean.astype(np.float32),
            "feature_scale": feat_scale.astype(np.float32),
        },
        "gmm": gmm,
    }
    diag = {
        "enabled": True,
        "input_dim": int(d),
        "embedding_key": str(embedding_key),
        "n_components": int(n_components),
        "covariance_type": str(gmm.covariance_type),
        "reg_covar": float(gmm.reg_covar),
        "max_iter": int(gmm.max_iter),
        "converged": bool(getattr(gmm, "converged_", False)),
        "n_iter": int(getattr(gmm, "n_iter_", 0)),
        "lower_bound": float(getattr(gmm, "lower_bound_", 0.0)),
        "score_mean": float(np.mean(score)) if len(score) else 0.0,
        "score_std": float(np.std(score)) if len(score) else 0.0,
        "nll_mean": float(np.mean(neg_log_likelihood)) if len(neg_log_likelihood) else 0.0,
        "nll_std": float(np.std(neg_log_likelihood)) if len(neg_log_likelihood) else 0.0,
    }
    return ckpt, diag, score


def topk_unique_coverage_stats(rows: List[Dict[str, Any]], topk: int = 200) -> Dict[str, int]:
    k = int(max(0, min(topk, len(rows))))
    file_set: Set[str] = set()
    file_fn_set: Set[Tuple[str, str]] = set()
    family_set: Set[str] = set()
    for row in rows[:k]:
        fk = _file_key_from_row(row)
        file_set.add(fk)
        family_set.add(_family_key_from_file(fk))
        for fn in _function_key_set(str(row.get("function_names", ""))):
            file_fn_set.add((fk, fn))
    return {
        f"top{k}_unique_files": int(len(file_set)),
        f"top{k}_unique_file_functions": int(len(file_fn_set)),
        f"top{k}_unique_families": int(len(family_set)),
    }


def main() -> None:
    args = parse_args()
    if not hasattr(args, "function_bag_enable"):
        args.function_bag_enable = False
    if not hasattr(args, "hier_prototype_enable"):
        args.hier_prototype_enable = False
    dual_view_input = (ROOT / args.dual_view_input).resolve()
    embedding_npz = (ROOT / args.embedding_npz).resolve()
    risk_out = (ROOT / args.risk_out).resolve()
    summary_out = (ROOT / args.summary_out).resolve()
    prototype_out = (ROOT / args.prototype_out).resolve()
    for p in [risk_out, summary_out, prototype_out]:
        p.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(dual_view_input)
    emb = load_embeddings(embedding_npz)
    embedding_key = str(args.embedding_key)
    if embedding_key == "z_hybrid":
        if "z_joint" in emb and "z_fused" in emb:
            alpha = float(min(1.0, max(0.0, args.hybrid_alpha)))
            emb["z_hybrid"] = (
                alpha * emb["z_fused"] + (1.0 - alpha) * emb["z_joint"]
            ).astype(np.float32)
        else:
            embedding_key = "z_joint"
    if embedding_key not in emb:
        if "z_joint" in emb:
            embedding_key = "z_joint"
        else:
            embedding_key = sorted(emb.keys())[0]
    z_joint = emb[embedding_key]
    z_full = emb["z_full"]
    z_skel = emb["z_skeleton"]
    if len(rows) != len(z_joint):
        n = min(len(rows), len(z_joint))
        rows = rows[:n]
        z_joint = z_joint[:n]
        z_full = z_full[:n]
        z_skel = z_skel[:n]

    mechanism_raw_slice = compute_mechanism_risk(rows)
    row_to_group = np.arange(len(rows), dtype=np.int64)
    bag_sizes = np.ones((len(rows),), dtype=np.int32)
    fit_rows = rows
    z_fit = z_joint
    z_full_fit = z_full
    z_skel_fit = z_skel
    mechanism_raw_fit = mechanism_raw_slice
    if bool(args.function_bag_enable):
        (
            fit_rows,
            z_fit,
            z_full_fit,
            z_skel_fit,
            mechanism_raw_fit,
            row_to_group,
            bag_sizes,
        ) = build_function_bag_view(
            rows=rows,
            emb_main=z_joint,
            emb_full=z_full,
            emb_skel=z_skel,
            mechanism_raw=mechanism_raw_slice,
            agg_mode=args.function_bag_agg,
        )

    view_gap = np.linalg.norm(z_full_fit - z_skel_fit, axis=1).astype(np.float32)

    proto = MultiPrototypeHead(
        n_prototypes=args.prototype_k,
        boundary_quantile=args.boundary_quantile,
        random_state=args.seed,
    )
    fit = proto.fit(z_fit)
    proto_scores = proto.score(z_fit)
    assign_global = proto_scores["assign"]
    nearest_global = proto_scores["nearest_distance"]
    margin_global = proto_scores["boundary_margin"]
    hier_local_bank: Dict[str, Dict[str, Any]] = {}
    hier_diag: Dict[str, Any] = {"enabled": False}
    if bool(getattr(args, "hier_prototype_enable", False)):
        hier_local_bank, fit_diag = fit_hierarchical_local_prototypes(
            rows=fit_rows,
            embeddings=z_fit,
            boundary_quantile=float(args.boundary_quantile),
            local_k=int(max(1, args.hier_prototype_local_k)),
            min_samples=int(max(4, args.hier_prototype_family_min_samples)),
            random_state=int(args.seed),
        )
        hier_score = score_hierarchical_prototypes(
            rows=fit_rows,
            embeddings=z_fit,
            global_assign=assign_global,
            global_nearest=nearest_global,
            global_margin=margin_global,
            local_bank=hier_local_bank,
            min_samples=int(max(4, args.hier_prototype_family_min_samples)),
            shrinkage_tau=float(args.hier_prototype_shrinkage_tau),
            global_mix_floor=float(args.hier_prototype_global_mix_floor),
            match_conf_floor=float(args.hier_prototype_match_conf_floor),
        )
        margin = hier_score["combined_margin"]
        nearest_for_rows = hier_score["combined_nearest"]
        local_assign_fit = hier_score["local_assign"]
        local_nearest_fit = hier_score["local_nearest"]
        local_margin_fit = hier_score["local_margin"]
        local_weight_fit = hier_score["local_weight"]
        local_match_conf_fit = hier_score.get("local_match_conf", np.ones_like(local_weight_fit))
        local_route_exact_fit = hier_score.get("local_route_exact", np.zeros_like(local_weight_fit))
        local_route_fallback_fit = hier_score.get("local_route_fallback_centroid", np.zeros_like(local_weight_fit))
        proto_scores["nearest_distance"] = nearest_for_rows
        proto_scores["boundary_margin"] = margin
        hier_diag = {
            **fit_diag,
            "enabled": True,
            "local_weight_mean": float(np.mean(local_weight_fit)) if len(local_weight_fit) else 0.0,
            "local_weight_std": float(np.std(local_weight_fit)) if len(local_weight_fit) else 0.0,
            "local_weight_max": float(np.max(local_weight_fit)) if len(local_weight_fit) else 0.0,
            "local_active_ratio": float(np.mean((local_weight_fit > 0).astype(np.float32))) if len(local_weight_fit) else 0.0,
            "local_match_conf_mean": float(np.mean(local_match_conf_fit)) if len(local_match_conf_fit) else 0.0,
            "local_match_conf_std": float(np.std(local_match_conf_fit)) if len(local_match_conf_fit) else 0.0,
            "local_route_exact_ratio": float(np.mean(local_route_exact_fit)) if len(local_route_exact_fit) else 0.0,
            "local_route_fallback_ratio": float(np.mean(local_route_fallback_fit)) if len(local_route_fallback_fit) else 0.0,
            "shrinkage_tau": float(args.hier_prototype_shrinkage_tau),
            "global_mix_floor": float(args.hier_prototype_global_mix_floor),
            "match_conf_floor": float(args.hier_prototype_match_conf_floor),
        }
    else:
        margin = margin_global
        nearest_for_rows = nearest_global
        local_assign_fit = np.full_like(assign_global, -1)
        local_nearest_fit = nearest_global.astype(np.float32)
        local_margin_fit = margin_global.astype(np.float32)
        local_weight_fit = np.zeros_like(nearest_global, dtype=np.float32)
    proto_counts = np.bincount(fit.labels.astype(np.int64), minlength=int(len(fit.radii))).astype(np.float32)
    proto_mass = (proto_counts / (np.sum(proto_counts) + 1e-8)).astype(np.float32)
    prior_mode = str(getattr(args, "energy_head_free_energy_prior", "uniform")).strip().lower()
    if prior_mode == "support":
        proto_weights = proto_mass.astype(np.float32)
    elif prior_mode == "sqrt_support":
        proto_weights = np.sqrt(np.maximum(proto_mass, 1e-8)).astype(np.float32)
        proto_weights = (proto_weights / (np.sum(proto_weights) + 1e-8)).astype(np.float32)
    else:
        proto_weights = np.full((len(fit.radii),), 1.0 / max(1, len(fit.radii)), dtype=np.float32)

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
        embeddings=z_fit,
        centers=fit.centers,
        density_k=args.density_k,
        n_perturb=args.n_perturb,
        noise_std=args.noise_std,
        random_state=args.seed,
        adaptive_local=bool(args.adaptive_boundary_refine),
        adaptive_focus_quantile=args.adaptive_boundary_focus_quantile,
        adaptive_min_scale=args.adaptive_boundary_min_scale,
        adaptive_max_scale=args.adaptive_boundary_max_scale,
        adaptive_outside_scale=args.adaptive_boundary_outside_scale,
    )
    base_final_risk = refined["final_risk"].astype(np.float32)
    family_score, family_diag = compute_family_aware_risk(
        rows=fit_rows,
        embeddings=z_fit,
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
    family_top_mask = top_quantile_mask(base_final_risk, quantile=args.family_top_quantile)
    mechanism_raw = mechanism_raw_fit
    mechanism_score = adaptive_mechanism_score(mechanism_raw_fit, std_threshold=0.05)
    w_mech = float(min(1.0, max(0.0, args.mechanism_risk_weight)))
    mech_top_mask = top_quantile_mask(base_final_risk, quantile=args.mechanism_top_quantile)
    if args.mechanism_semantic_gate:
        sem_mask = semantic_closed_loop_mask(
            rows,
            min_connected_pairs=args.mechanism_semantic_min_pairs,
            max_hops=args.mechanism_semantic_max_hops,
        )
    else:
        sem_mask = np.ones_like(mech_top_mask, dtype=bool)
    mech_active_mask = np.logical_and(mech_top_mask, sem_mask)
    final_risk = blend_mechanism_risk(
        base_final_risk,
        mechanism_score,
        weight=w_mech,
        active_mask=mech_active_mask,
    )
    final_risk = blend_family_risk(
        final_risk,
        family_score,
        weight=w_family,
        active_mask=family_top_mask,
    )

    q90 = float(np.quantile(final_risk, 0.9)) if len(final_risk) else 0.0
    q75 = float(np.quantile(final_risk, 0.75)) if len(final_risk) else 0.0
    status = np.array(
        [
            "high_risk" if x >= q90 else ("boundary" if x >= q75 else "normal")
            for x in final_risk
        ]
    )

    assign_expand = proto_scores["assign"][row_to_group]
    nearest_expand = nearest_for_rows[row_to_group]
    margin_expand = margin[row_to_group]
    local_assign_expand = local_assign_fit[row_to_group]
    local_nearest_expand = local_nearest_fit[row_to_group]
    local_margin_expand = local_margin_fit[row_to_group]
    local_weight_expand = local_weight_fit[row_to_group]
    view_gap_expand = view_gap[row_to_group]
    base_final_expand = base_final_risk[row_to_group]
    mechanism_expand = mechanism_score[row_to_group]
    family_expand = family_score[row_to_group]
    final_expand = final_risk[row_to_group]
    status_expand = status[row_to_group]
    proto_score_expand = refined["proto_score"][row_to_group]
    view_score_expand = refined["view_score"][row_to_group]
    density_expand = refined["density_risk"][row_to_group]
    instability_expand = refined["instability"][row_to_group]
    proximity_expand = refined.get("boundary_proximity", np.zeros_like(base_final_risk))[row_to_group]
    local_scale_expand = refined.get("adaptive_local_scale", np.ones_like(base_final_risk))[row_to_group]
    focus_mask_expand = refined.get(
        "adaptive_focus_mask",
        np.ones_like(base_final_risk, dtype=np.bool_),
    )[row_to_group]
    closure_feature_rows = [
        compute_graph_closure_features(
            row.get("full_graph", {}) or {},
            row.get("skeleton_graph", {}) or {},
            max_hops=int(getattr(args, "mechanism_semantic_max_hops", 6)),
        )
        for row in rows
    ]

    out_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        out_rows.append(
            {
                "slice_id": row.get("slice_id"),
                "contract_id": row.get("contract_id"),
                "relative_source_path": row.get("relative_source_path"),
                "function_names": "|".join(row.get("function_names", [])),
                "function_bag_enable": int(bool(args.function_bag_enable)),
                "function_bag_size": int(bag_sizes[row_to_group[i]]) if len(bag_sizes) else 1,
                "prototype_id": int(assign_expand[i]),
                "nearest_distance": float(nearest_expand[i]),
                "boundary_margin": float(margin_expand[i]),
                "local_prototype_id": int(local_assign_expand[i]),
                "local_nearest_distance": float(local_nearest_expand[i]),
                "local_boundary_margin": float(local_margin_expand[i]),
                "hier_local_weight": float(local_weight_expand[i]),
                "view_gap": float(view_gap_expand[i]),
                "proto_score": float(proto_score_expand[i]),
                "view_score": float(view_score_expand[i]),
                "density_risk": float(density_expand[i]),
                "instability": float(instability_expand[i]),
                "boundary_proximity": float(proximity_expand[i]),
                "adaptive_local_scale": float(local_scale_expand[i]),
                "adaptive_focus_mask": int(focus_mask_expand[i]),
                "base_final_risk": float(base_final_expand[i]),
                "mechanism_risk": float(mechanism_expand[i]),
                "family_risk": float(family_expand[i]),
                "final_risk": float(final_expand[i]),
                "status": str(status_expand[i]),
                **closure_feature_rows[i],
            }
        )

    metamorphic_feature_diag: Dict[str, Any] = {"enabled": False}
    if bool(getattr(args, "metamorphic_feature_enable", False)) and len(out_rows) > 0:
        try:
            dual_model_ckpt = (ROOT / args.dual_model_ckpt).resolve()
            global_graph_dir = None
            if str(getattr(args, "metamorphic_global_graph_dir", "") or "").strip():
                global_graph_dir = (ROOT / args.metamorphic_global_graph_dir).resolve()
            m_device = torch.device("cpu")
            dual_model, dual_ckpt = load_dual_model_ckpt(dual_model_ckpt, device=m_device)
            use_global_local = isinstance(dual_model, GlobalLocalDualViewModel)
            if use_global_local and global_graph_dir is None:
                fallback_dir = ROOT / "data/graphs/tagged"
                if fallback_dir.exists():
                    global_graph_dir = fallback_dir.resolve()
            dual_row_lookup = {str(r.get("slice_id", "")): r for r in rows}
            meta_feat, metamorphic_feature_diag = compute_metamorphic_instability_features(
                eval_rows=out_rows,
                dual_row_lookup=dual_row_lookup,
                model=dual_model,
                device=m_device,
                global_graph_dir=global_graph_dir,
                centers=fit.centers,
                radii=fit.radii,
                embedding_key=str(embedding_key),
                hybrid_alpha=float(args.hybrid_alpha),
                max_hops=int(max(1, args.metamorphic_max_hops)),
                max_rows=int(max(0, args.metamorphic_feature_max_rows)),
                select_mode=str(args.metamorphic_feature_select),
                boundary_quantile=float(args.metamorphic_feature_boundary_quantile),
                switch_min=float(args.metamorphic_feature_switch_min),
                gate_mode=str(args.metamorphic_feature_gate_mode),
            )
            metamorphic_feature_diag["model_type"] = str(dual_ckpt.get("model_type", "dual-view"))
            metamorphic_feature_diag["dual_model_ckpt"] = str(dual_model_ckpt)
            metamorphic_feature_diag["global_graph_dir"] = str(global_graph_dir) if global_graph_dir is not None else ""
            for i, row in enumerate(out_rows):
                row["metamorphic_instability"] = float(meta_feat["metamorphic_instability"][i])
                row["metamorphic_nearest_delta"] = float(meta_feat["metamorphic_nearest_delta"][i])
                row["metamorphic_margin_delta"] = float(meta_feat["metamorphic_margin_delta"][i])
                row["metamorphic_view_delta"] = float(meta_feat["metamorphic_view_delta"][i])
                row["metamorphic_proto_switch_rate"] = float(meta_feat["metamorphic_proto_switch_rate"][i])
                row["metamorphic_variant_count"] = int(meta_feat["metamorphic_variant_count"][i])
        except Exception as ex:
            metamorphic_feature_diag = {
                "enabled": False,
                "error": str(ex),
            }
            for row in out_rows:
                row["metamorphic_instability"] = 0.0
                row["metamorphic_nearest_delta"] = 0.0
                row["metamorphic_margin_delta"] = 0.0
                row["metamorphic_view_delta"] = 0.0
                row["metamorphic_proto_switch_rate"] = 0.0
                row["metamorphic_variant_count"] = 0
    else:
        for row in out_rows:
            row["metamorphic_instability"] = 0.0
            row["metamorphic_nearest_delta"] = 0.0
            row["metamorphic_margin_delta"] = 0.0
            row["metamorphic_view_delta"] = 0.0
            row["metamorphic_proto_switch_rate"] = 0.0
            row["metamorphic_variant_count"] = 0

    out_rows.sort(key=lambda x: x["final_risk"], reverse=True)
    if args.risk_rerank_mode == "coverage":
        out_rows = coverage_rerank_rows(
            out_rows,
            score_key="final_risk",
            topn=args.risk_rerank_topn,
            file_penalty=args.risk_rerank_file_penalty,
            filefn_penalty=args.risk_rerank_filefn_penalty,
            max_per_file=args.risk_rerank_max_per_file,
            file_repeat_power=args.risk_rerank_file_repeat_power,
        )
    elif args.risk_rerank_mode == "function_coverage":
        out_rows = function_coverage_rerank_rows(
            out_rows,
            score_key="final_risk",
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
        out_rows = family_hybrid_rerank_rows(
            out_rows,
            score_key="final_risk",
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
    out_rows = warmup_diversify_rows(
        out_rows,
        warmup_topk=args.risk_rerank_warmup_topk,
        per_file_cap=args.risk_rerank_warmup_file_cap,
        per_filefn_cap=args.risk_rerank_warmup_filefn_cap,
    )
    energy_head_diag: Dict[str, Any] = {"enabled": False}
    energy_head_out = (ROOT / args.energy_head_out).resolve()
    energy_ckpt = None
    energy_score = np.zeros((len(out_rows),), dtype=np.float32)
    mechanism_head_diag: Dict[str, Any] = {"enabled": False}
    mechanism_head_out = (ROOT / args.mechanism_head_out).resolve()
    mechanism_head_score = np.zeros((len(out_rows),), dtype=np.float32)
    gmm_head_diag: Dict[str, Any] = {"enabled": False}
    gmm_head_out = (ROOT / args.gmm_head_out).resolve()
    gmm_ckpt = None
    gmm_score = np.zeros((len(out_rows),), dtype=np.float32)
    sid_to_emb: Dict[str, np.ndarray] = {}
    sid_to_graph_emb: Dict[str, np.ndarray] = {}
    sid_to_text_emb: Dict[str, np.ndarray] = {}
    graph_aux = np.asarray(emb.get("z_graph_joint", z_joint), dtype=np.float32)
    text_aux_raw = emb.get("z_text")
    text_aux = np.asarray(text_aux_raw, dtype=np.float32) if text_aux_raw is not None else None
    for i, row in enumerate(rows):
        sid = str(row.get("slice_id", ""))
        if sid:
            sid_to_emb[sid] = np.asarray(z_joint[i], dtype=np.float32)
            if graph_aux.ndim == 2 and i < len(graph_aux):
                sid_to_graph_emb[sid] = np.asarray(graph_aux[i], dtype=np.float32)
            if text_aux is not None and text_aux.ndim == 2 and i < len(text_aux):
                sid_to_text_emb[sid] = np.asarray(text_aux[i], dtype=np.float32)
    ordered_emb: List[np.ndarray] = []
    ordered_graph_emb: List[np.ndarray] = []
    ordered_text_emb: List[np.ndarray] = []
    graph_dim = int(graph_aux.shape[1]) if graph_aux.ndim == 2 and graph_aux.shape[0] > 0 else int(z_joint.shape[1])
    text_dim = int(text_aux.shape[1]) if text_aux is not None and text_aux.ndim == 2 and text_aux.shape[0] > 0 else graph_dim
    for row in out_rows:
        sid = str(row.get("slice_id", ""))
        ordered_emb.append(np.asarray(sid_to_emb.get(sid, np.zeros((z_joint.shape[1],), dtype=np.float32)), dtype=np.float32))
        if bool(getattr(args, "energy_head_include_text_aux", False)):
            ordered_graph_emb.append(np.asarray(sid_to_graph_emb.get(sid, np.zeros((graph_dim,), dtype=np.float32)), dtype=np.float32))
            ordered_text_emb.append(np.asarray(sid_to_text_emb.get(sid, np.zeros((text_dim,), dtype=np.float32)), dtype=np.float32))
    if ordered_emb:
        ordered_emb_arr = np.stack(ordered_emb, axis=0).astype(np.float32)
        ordered_graph_arr = np.stack(ordered_graph_emb, axis=0).astype(np.float32) if ordered_graph_emb else None
        ordered_text_arr = np.stack(ordered_text_emb, axis=0).astype(np.float32) if ordered_text_emb else None
        energy_ckpt, energy_head_diag, energy_score = train_energy_head_from_rows(
            rows=out_rows,
            embeddings=ordered_emb_arr,
            text_embeddings=ordered_text_arr,
            graph_embeddings=ordered_graph_arr,
            prototype_centers=fit.centers,
            prototype_radii=fit.radii,
            prototype_weights=proto_weights,
            args=args,
        )
        if energy_ckpt is not None:
            energy_head_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(energy_ckpt, energy_head_out)
            for i, row in enumerate(out_rows):
                row["energy_head_score"] = float(energy_score[i])
        gmm_ckpt, gmm_head_diag, gmm_score = train_gmm_head_from_embeddings(
            rows=out_rows,
            embeddings=ordered_emb_arr,
            embedding_key=str(embedding_key),
            args=args,
        )
        if gmm_ckpt is not None:
            gmm_head_out.parent.mkdir(parents=True, exist_ok=True)
            with gmm_head_out.open("wb") as f:
                pickle.dump(gmm_ckpt, f)
            for i, row in enumerate(out_rows):
                row["gmm_head_score"] = float(gmm_score[i])
    if bool(getattr(args, "mechanism_head_enable", False)) and len(rows) > 0:
        sid_to_aux = {
            str(row.get("slice_id", "")): row
            for row in out_rows
            if str(row.get("slice_id", "")).strip()
        }
        mechanism_input_rows: List[Dict[str, Any]] = []
        for row in rows:
            sid = str(row.get("slice_id", ""))
            merged = dict(row)
            aux = sid_to_aux.get(sid, {})
            merged["proto_score"] = float(aux.get("proto_score", 0.0) or 0.0)
            merged["view_score"] = float(aux.get("view_score", 0.0) or 0.0)
            merged["boundary_margin"] = float(aux.get("boundary_margin", 0.0) or 0.0)
            mechanism_input_rows.append(merged)
        mechanism_feature_rows = extract_mechanism_head_feature_rows(
            mechanism_input_rows,
            max_hops=int(max(1, getattr(args, "mechanism_head_max_hops", 6))),
        )
        mechanism_model = MechanismSliceHeadModel(
            random_state=int(args.seed),
            max_hops=int(max(1, getattr(args, "mechanism_head_max_hops", 6))),
            gate_quantile=float(getattr(args, "mechanism_head_gate_quantile", 0.55)),
            slice_quantile=float(getattr(args, "mechanism_head_slice_quantile", 0.95)),
        )
        mechanism_model.fit(mechanism_feature_rows)
        scored_feature_rows = mechanism_model.score_rows(mechanism_feature_rows)
        sid_to_mech = {str(row.get("slice_id", "")): row for row in scored_feature_rows if str(row.get("slice_id", "")).strip()}
        mechanism_head_out.parent.mkdir(parents=True, exist_ok=True)
        save_mechanism_head_model(mechanism_head_out, mechanism_model)
        for i, row in enumerate(out_rows):
            sid = str(row.get("slice_id", ""))
            mrow = sid_to_mech.get(sid, {})
            score = float(mrow.get("anomaly_score", 0.0))
            mechanism_head_score[i] = score
            row["mechanism_head_score"] = score
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
            row["mechanism_head_threshold"] = float(mechanism_model.slice_threshold)
            if bool(getattr(args, "mechanism_head_override_final_risk", False)):
                row["legacy_final_risk"] = float(row.get("final_risk", 0.0))
                row["final_risk"] = score
        mechanism_head_diag = {
            "enabled": True,
            "fit_scope": "mechanism_slice_head",
            "feature_dim": int(len(mechanism_model.feature_names)),
            "feature_names": list(mechanism_model.feature_names),
            "slice_threshold": float(mechanism_model.slice_threshold),
            "override_final_risk": bool(getattr(args, "mechanism_head_override_final_risk", False)),
            "corroboration_alpha": float(getattr(mechanism_model, "corroboration_alpha", 0.40)),
            "corroboration_vp_weight": float(getattr(mechanism_model, "corroboration_vp_weight", 0.80)),
            "corroboration_margin_weight": float(getattr(mechanism_model, "corroboration_margin_weight", 0.20)),
            "utility_floor": float(getattr(mechanism_model, "utility_floor", 0.60)),
            "utility_vp_weight": float(getattr(mechanism_model, "utility_vp_weight", 0.40)),
            "utility_margin_weight": float(getattr(mechanism_model, "utility_margin_weight", 0.60)),
            "score_mean": float(np.mean(mechanism_head_score)) if len(mechanism_head_score) else 0.0,
            "score_std": float(np.std(mechanism_head_score)) if len(mechanism_head_score) else 0.0,
            **dict(mechanism_model.train_summary),
        }
    out_rows.sort(key=lambda x: float(x.get("final_risk", 0.0)), reverse=True)
    if args.risk_rerank_mode == "coverage":
        out_rows = coverage_rerank_rows(
            out_rows,
            score_key="final_risk",
            topn=args.risk_rerank_topn,
            file_penalty=args.risk_rerank_file_penalty,
            filefn_penalty=args.risk_rerank_filefn_penalty,
            max_per_file=args.risk_rerank_max_per_file,
            file_repeat_power=args.risk_rerank_file_repeat_power,
        )
    elif args.risk_rerank_mode == "function_coverage":
        out_rows = function_coverage_rerank_rows(
            out_rows,
            score_key="final_risk",
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
        out_rows = family_hybrid_rerank_rows(
            out_rows,
            score_key="final_risk",
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
    out_rows = warmup_diversify_rows(
        out_rows,
        warmup_topk=args.risk_rerank_warmup_topk,
        per_file_cap=args.risk_rerank_warmup_file_cap,
        per_filefn_cap=args.risk_rerank_warmup_filefn_cap,
    )
    with risk_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "slice_id",
                "contract_id",
                "relative_source_path",
                "function_names",
                "function_bag_enable",
                "function_bag_size",
                "prototype_id",
                "nearest_distance",
                "boundary_margin",
                "local_prototype_id",
                "local_nearest_distance",
                "local_boundary_margin",
                "hier_local_weight",
                "view_gap",
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
                "final_risk",
                "status",
                "metamorphic_instability",
                "metamorphic_nearest_delta",
                "metamorphic_margin_delta",
                "metamorphic_view_delta",
                "metamorphic_proto_switch_rate",
                "metamorphic_variant_count",
                "score_head_risk",
                "score_head_rank_pct",
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
                "mechanism_head_threshold",
                "energy_head_score",
                "gmm_head_score",
            ]
            + list(CLOSURE_FEATURE_NAMES),
        )
        writer.writeheader()
        writer.writerows(out_rows)

    np.savez(
        prototype_out,
        centers=fit.centers,
        radii=fit.radii,
        weights=proto_weights.astype(np.float32),
        counts=proto_counts.astype(np.float32),
        embedding_key=np.asarray(embedding_key),
        hybrid_alpha=np.asarray(float(args.hybrid_alpha)),
        function_bag_enable=np.asarray(bool(args.function_bag_enable)),
        function_bag_agg=np.asarray(str(args.function_bag_agg)),
        hier_prototype_enable=np.asarray(bool(getattr(args, "hier_prototype_enable", False))),
        hier_local_k=np.asarray(int(getattr(args, "hier_prototype_local_k", 4))),
        hier_family_min_samples=np.asarray(int(getattr(args, "hier_prototype_family_min_samples", 96))),
        hier_shrinkage_tau=np.asarray(float(getattr(args, "hier_prototype_shrinkage_tau", 96.0))),
        hier_global_mix_floor=np.asarray(float(getattr(args, "hier_prototype_global_mix_floor", 0.25))),
        hier_match_conf_floor=np.asarray(float(getattr(args, "hier_prototype_match_conf_floor", 0.25))),
        **pack_local_bank(hier_local_bank),
    )

    summary = {
        "num_samples": len(out_rows),
        "num_function_bags": int(len(fit_rows)),
        "seed": int(args.seed),
        "embedding_key": str(embedding_key),
        "hybrid_alpha": float(args.hybrid_alpha),
        "function_bag": {
            "enabled": bool(args.function_bag_enable),
            "agg": str(args.function_bag_agg),
            "num_bags": int(len(fit_rows)),
            "mean_bag_size": float(np.mean(bag_sizes)) if len(bag_sizes) else 0.0,
            "max_bag_size": int(np.max(bag_sizes)) if len(bag_sizes) else 0,
        },
        "energy_head": {
            **energy_head_diag,
            "out": str(energy_head_out) if bool(args.energy_head_enable) else "",
        },
        "mechanism_head": {
            **mechanism_head_diag,
            "out": str(mechanism_head_out) if bool(getattr(args, "mechanism_head_enable", False)) else "",
        },
        "gmm_head": {
            **gmm_head_diag,
            "out": str(gmm_head_out) if bool(args.gmm_head_enable) else "",
        },
        "metamorphic_feature": metamorphic_feature_diag,
        "prototype_k": int(len(fit.radii)),
        "prototype_prior": {
            "mode": str(prior_mode),
            "weights": proto_weights.astype(np.float32).tolist(),
            "counts": proto_counts.astype(np.float32).tolist(),
        },
        "hierarchical_prototypes": hier_diag,
        "boundary_quantile": args.boundary_quantile,
        "adaptive_boundary_refine": {
            "enabled": bool(args.adaptive_boundary_refine),
            "focus_quantile": float(args.adaptive_boundary_focus_quantile),
            "min_scale": float(args.adaptive_boundary_min_scale),
            "max_scale": float(args.adaptive_boundary_max_scale),
            "outside_scale": float(args.adaptive_boundary_outside_scale),
            "focus_mask_ratio": float(
                np.mean(refined.get("adaptive_focus_mask", np.ones_like(base_final_risk, dtype=np.bool_)).astype(np.float32))
            )
            if len(base_final_risk)
            else 0.0,
            "local_scale_mean": float(np.mean(refined.get("adaptive_local_scale", np.ones_like(base_final_risk))))
            if len(base_final_risk)
            else 0.0,
            "local_scale_std": float(np.std(refined.get("adaptive_local_scale", np.ones_like(base_final_risk))))
            if len(base_final_risk)
            else 0.0,
            "boundary_proximity_mean": float(np.mean(refined.get("boundary_proximity", np.zeros_like(base_final_risk))))
            if len(base_final_risk)
            else 0.0,
            "legacy_final_mean": float(np.mean(refined.get("legacy_final_risk", base_final_risk))) if len(base_final_risk) else 0.0,
            "adaptive_final_mean": float(np.mean(base_final_risk)) if len(base_final_risk) else 0.0,
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
            "blend_center_median": float(np.median(mechanism_score)) if len(mechanism_score) else 0.0,
            "raw_mean": float(np.mean(mechanism_raw)) if len(mechanism_raw) else 0.0,
            "raw_std": float(np.std(mechanism_raw)) if len(mechanism_raw) else 0.0,
            "score_mean": float(np.mean(mechanism_score)) if len(mechanism_score) else 0.0,
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
        "coverage_stats": topk_unique_coverage_stats(out_rows, topk=200),
        "status_counts": {
            "high_risk": int(sum(1 for r in out_rows if r["status"] == "high_risk")),
            "boundary": int(sum(1 for r in out_rows if r["status"] == "boundary")),
            "normal": int(sum(1 for r in out_rows if r["status"] == "normal")),
        },
        "topk_samples": out_rows[: args.topk],
    }
    save_json(summary_out, summary)
    print(f"[train_unsup] risk_csv={risk_out}")
    print(f"[train_unsup] summary={summary_out}")
    print(f"[train_unsup] prototypes={prototype_out}")
    if bool(args.energy_head_enable):
        print(f"[train_unsup] energy_head={energy_head_out}")
    if bool(getattr(args, "mechanism_head_enable", False)):
        print(f"[train_unsup] mechanism_head={mechanism_head_out}")
    if bool(args.gmm_head_enable):
        print(f"[train_unsup] gmm_head={gmm_head_out}")


if __name__ == "__main__":
    main()
