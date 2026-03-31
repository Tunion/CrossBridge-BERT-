from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def _count_pipe_items(v: Any) -> int:
    if isinstance(v, list):
        return int(sum(1 for x in v if str(x).strip()))
    text = str(v or "")
    if not text:
        return 0
    return int(sum(1 for x in text.split("|") if x.strip()))


def normalize01(v: Any) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo + 1e-8)).astype(np.float32)


def fit_scope_mask(rows: Sequence[Dict[str, Any]], scope: str) -> np.ndarray:
    scope_key = str(scope or "all").strip().lower()
    if scope_key == "all":
        return np.ones(len(rows), dtype=bool)
    if scope_key in {"low_risk", "normal"}:
        return np.asarray([str(r.get("status", "")) == "normal" for r in rows], dtype=bool)
    return np.ones(len(rows), dtype=bool)


def build_score_feature_matrix(rows: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), []
    df = pd.DataFrame(list(rows))
    if "function_names" not in df.columns:
        df["function_names"] = ""
    if "line_candidates" not in df.columns:
        df["line_candidates"] = ""
    df["fn_count"] = df["function_names"].map(_count_pipe_items).astype(np.float32)
    df["line_count"] = df["line_candidates"].map(_count_pipe_items).astype(np.float32)
    df["line_per_fn"] = df["line_count"] / np.maximum(df["fn_count"], 1.0)
    for col in (
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
    ):
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
        "fn_count",
        "line_count",
        "line_per_fn",
    ]
    feat = (
        df[cols]
        .astype(np.float32)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    return feat, cols


def apply_iforest_score_head(
    rows: Sequence[Dict[str, Any]],
    fit_scope: str = "low_risk",
    random_state: int = 42,
    n_estimators: int = 300,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    base = np.asarray([float(r.get("final_risk", 0.0)) for r in rows], dtype=np.float32)
    if len(rows) <= 0:
        return base, {"enabled": False, "method": "none"}
    feat, cols = build_score_feature_matrix(rows)
    if feat.shape[0] <= 0 or feat.shape[1] <= 0:
        return base, {"enabled": False, "method": "none", "reason": "empty_features"}
    scaler = StandardScaler()
    feat_std = scaler.fit_transform(feat)
    fit_mask = fit_scope_mask(rows, fit_scope)
    if not np.any(fit_mask):
        fit_mask = np.ones(len(rows), dtype=bool)
    clf = IsolationForest(
        n_estimators=int(max(50, n_estimators)),
        contamination="auto",
        random_state=int(random_state),
    )
    clf.fit(feat_std[fit_mask])
    score = -clf.score_samples(feat_std)
    score = normalize01(score)
    diag = {
        "enabled": True,
        "method": "iforest",
        "fit_scope": str(fit_scope),
        "fit_rows": int(np.sum(fit_mask)),
        "feature_dim": int(feat.shape[1]),
        "feature_names": cols,
        "score_mean": float(np.mean(score)) if len(score) else 0.0,
        "score_std": float(np.std(score)) if len(score) else 0.0,
        "n_estimators": int(max(50, n_estimators)),
    }
    return score.astype(np.float32), diag
