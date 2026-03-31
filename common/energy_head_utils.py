from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.closure_feature_utils import CLOSURE_FEATURE_NAMES


def _row_file_key(row: Dict[str, Any]) -> str:
    return str(
        row.get("relative_source_path")
        or row.get("file")
        or row.get("source_path")
        or row.get("contract_id")
        or "<unknown_file>"
    )


def _row_function_text(row: Dict[str, Any]) -> str:
    raw = row.get("function_names", "")
    if isinstance(raw, list):
        return "|".join(str(x) for x in raw if str(x).strip())
    return str(raw or "")


def _canonical_fn(function_names: str) -> str:
    toks = [x.strip().lower() for x in str(function_names).split("|") if x.strip()]
    return toks[0] if toks else "<unknown_fn>"


def _function_group_id(row: Dict[str, Any]) -> str:
    return f"{_row_file_key(row)}:::{_canonical_fn(_row_function_text(row))}"


def _family_key_from_row(row: Dict[str, Any]) -> str:
    file_key = _row_file_key(row).replace("\\", "/")
    toks = [t.strip().lower() for t in file_key.split("/") if t.strip()]
    if not toks:
        return "<unknown_family>"
    generic = {
        "dataset",
        "contract",
        "contracts",
        "manually-labeled dataset",
        "real_attack_dataset_format",
        "src",
        "source",
        "sources",
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
    stem = os.path.splitext(toks[-1])[0]
    return stem or "<unknown_family>"


def _safe_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def _build_group_stats(rows: Sequence[Dict[str, Any]], n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    used_rows = list(rows[:n])
    group_to_idx: Dict[str, List[int]] = {}
    for i, row in enumerate(used_rows):
        gid = _function_group_id(row)
        group_to_idx.setdefault(gid, []).append(i)

    bag_max_base = np.zeros((n,), dtype=np.float32)
    bag_mean_base = np.zeros((n,), dtype=np.float32)
    bag_max_proto = np.zeros((n,), dtype=np.float32)
    bag_size = np.ones((n,), dtype=np.float32)
    for idx_list in group_to_idx.values():
        base_vals = np.asarray([_safe_float(used_rows[i], "base_final_risk", 0.0) for i in idx_list], dtype=np.float32)
        proto_vals = np.asarray([_safe_float(used_rows[i], "proto_score", 0.0) for i in idx_list], dtype=np.float32)
        vmax = float(np.max(base_vals)) if len(base_vals) else 0.0
        vmean = float(np.mean(base_vals)) if len(base_vals) else 0.0
        pmax = float(np.max(proto_vals)) if len(proto_vals) else 0.0
        vsz = float(len(idx_list))
        for i in idx_list:
            bag_max_base[i] = vmax
            bag_mean_base[i] = vmean
            bag_max_proto[i] = pmax
            bag_size[i] = vsz
    return bag_max_base, bag_mean_base, bag_max_proto, bag_size


def compute_radius_normalized_free_energy(
    embeddings: np.ndarray,
    centers: Optional[np.ndarray],
    radii: Optional[np.ndarray],
    prototype_weights: Optional[np.ndarray] = None,
    temperature: float = 0.35,
) -> Dict[str, np.ndarray]:
    emb = np.asarray(embeddings, dtype=np.float32)
    n = int(emb.shape[0]) if emb.ndim == 2 else 0
    zero = np.zeros((n,), dtype=np.float32)
    if (
        emb.ndim != 2
        or centers is None
        or radii is None
        or len(centers) <= 0
        or len(radii) <= 0
    ):
        return {
            "proto_free_energy": zero,
            "proto_resp_entropy": zero,
            "proto_nearest_norm_distance": zero,
            "proto_resp_max": zero,
            "proto_resp_gap": zero,
            "proto_top2_norm_gap": zero,
        }
    c = np.asarray(centers, dtype=np.float32)
    r = np.asarray(radii, dtype=np.float32).reshape(-1)
    if c.ndim != 2 or c.shape[0] != r.shape[0] or c.shape[1] != emb.shape[1]:
        return {
            "proto_free_energy": zero,
            "proto_resp_entropy": zero,
            "proto_nearest_norm_distance": zero,
            "proto_resp_max": zero,
            "proto_resp_gap": zero,
            "proto_top2_norm_gap": zero,
        }

    temp = float(max(1e-4, temperature))
    safe_r = np.maximum(r.astype(np.float32), 1e-4)
    sq = np.sum((emb[:, None, :] - c[None, :, :]) ** 2, axis=2).astype(np.float32)
    norm_d = (sq / (safe_r[None, :] ** 2 + 1e-8)).astype(np.float32)
    logits = (-norm_d / temp).astype(np.float32)
    if prototype_weights is not None:
        w = np.asarray(prototype_weights, dtype=np.float32).reshape(-1)
        if w.shape[0] == c.shape[0]:
            w = np.maximum(w, 1e-8).astype(np.float32)
            w = (w / (np.sum(w) + 1e-8)).astype(np.float32)
            logits = (logits + np.log(w[None, :] + 1e-8)).astype(np.float32)
    max_logit = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logit).astype(np.float32)
    z = np.sum(exp_logits, axis=1, keepdims=True) + 1e-8
    probs = (exp_logits / z).astype(np.float32)
    lse = (max_logit + np.log(z)).reshape(-1).astype(np.float32)
    free_energy = (-temp * lse).astype(np.float32)
    nearest_norm = np.min(norm_d, axis=1).astype(np.float32)
    if norm_d.shape[1] >= 2:
        part = np.partition(norm_d, kth=1, axis=1)
        top2_norm_gap = (part[:, 1] - part[:, 0]).astype(np.float32)
    else:
        top2_norm_gap = np.zeros((norm_d.shape[0],), dtype=np.float32)
    denom = float(math.log(max(2, c.shape[0])))
    entropy = (-np.sum(probs * np.log(probs + 1e-8), axis=1) / denom).astype(np.float32)
    resp_sort = np.sort(probs, axis=1).astype(np.float32)
    resp_max = resp_sort[:, -1].astype(np.float32)
    if probs.shape[1] >= 2:
        resp_gap = (resp_sort[:, -1] - resp_sort[:, -2]).astype(np.float32)
    else:
        resp_gap = resp_max.copy()
    return {
        "proto_free_energy": free_energy,
        "proto_resp_entropy": entropy,
        "proto_nearest_norm_distance": nearest_norm,
        "proto_resp_max": resp_max,
        "proto_resp_gap": resp_gap,
        "proto_top2_norm_gap": top2_norm_gap,
    }


def build_proto_margin_targets(
    rows: Sequence[Dict[str, Any]],
    n: int,
    pos_quantile: float = 0.30,
    neg_quantile: float = 0.85,
    family_aware: bool = False,
    family_min_samples: int = 64,
    metamorphic_weight: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    used_rows = list(rows[:n])
    if n <= 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
            np.zeros((0,), dtype=bool),
            {"num_rows": 0},
        )

    bag_max_base, _, bag_max_proto, _ = _build_group_stats(used_rows, n)

    proto_score = np.asarray([_safe_float(r, "proto_score", 0.0) for r in used_rows], dtype=np.float32)
    nearest_distance = np.asarray([_safe_float(r, "nearest_distance", 0.0) for r in used_rows], dtype=np.float32)
    boundary_margin = np.asarray([_safe_float(r, "boundary_margin", 0.0) for r in used_rows], dtype=np.float32)
    boundary_proximity = np.asarray([_safe_float(r, "boundary_proximity", 0.0) for r in used_rows], dtype=np.float32)
    view_gap = np.asarray([_safe_float(r, "view_gap", 0.0) for r in used_rows], dtype=np.float32)
    family_risk = np.asarray([_safe_float(r, "family_risk", 0.0) for r in used_rows], dtype=np.float32)
    mechanism_risk = np.asarray([_safe_float(r, "mechanism_risk", 0.0) for r in used_rows], dtype=np.float32)

    proto_n = normalize01(proto_score)
    near_n = normalize01(nearest_distance)
    margin_n = normalize01(np.maximum(boundary_margin, 0.0))
    prox_n = normalize01(boundary_proximity)
    view_n = normalize01(view_gap)
    fam_n = normalize01(family_risk)
    mech_n = normalize01(mechanism_risk)
    bag_n = normalize01(bag_max_proto)
    bag_base_n = normalize01(bag_max_base)
    meta_inst = np.asarray([_safe_float(r, "metamorphic_instability", 0.0) for r in used_rows], dtype=np.float32)
    meta_margin = np.asarray([_safe_float(r, "metamorphic_margin_delta", 0.0) for r in used_rows], dtype=np.float32)
    meta_switch = np.asarray([_safe_float(r, "metamorphic_proto_switch_rate", 0.0) for r in used_rows], dtype=np.float32)
    meta_combo = (
        0.55 * normalize01(meta_inst)
        + 0.25 * normalize01(meta_switch)
        + 0.20 * normalize01(meta_margin)
    ).astype(np.float32)

    anchor = (
        0.32 * proto_n
        + 0.18 * near_n
        + 0.12 * margin_n
        + 0.12 * prox_n
        + 0.10 * view_n
        + 0.08 * bag_n
        + 0.04 * bag_base_n
        + 0.02 * fam_n
        + 0.02 * mech_n
    ).astype(np.float32)
    meta_w = float(max(0.0, metamorphic_weight))
    if meta_w > 0.0:
        anchor = (anchor + meta_w * meta_combo).astype(np.float32)
    anchor = normalize01(anchor)

    q_pos = float(np.quantile(anchor, float(min(max(pos_quantile, 0.01), 0.49))))
    q_neg = float(np.quantile(anchor, float(min(max(neg_quantile, 0.51), 0.99))))
    pos_mask = anchor <= q_pos
    neg_mask = anchor >= q_neg

    family_local_count = 0
    family_fallback_rows = 0
    if bool(family_aware):
        fam_to_idx: Dict[str, List[int]] = {}
        for i, row in enumerate(used_rows):
            fam_to_idx.setdefault(_family_key_from_row(row), []).append(i)
        pos_local = np.zeros((n,), dtype=bool)
        neg_local = np.zeros((n,), dtype=bool)
        for idx_list in fam_to_idx.values():
            idx = np.asarray(idx_list, dtype=np.int64)
            if len(idx) < int(max(4, family_min_samples)):
                family_fallback_rows += int(len(idx))
                continue
            fam_anchor = anchor[idx]
            fam_q_pos = float(np.quantile(fam_anchor, float(min(max(pos_quantile, 0.01), 0.49))))
            fam_q_neg = float(np.quantile(fam_anchor, float(min(max(neg_quantile, 0.51), 0.99))))
            pos_local[idx[fam_anchor <= fam_q_pos]] = True
            neg_local[idx[fam_anchor >= fam_q_neg]] = True
            family_local_count += 1
        if family_local_count > 0:
            fb = ~(pos_local | neg_local)
            pos_mask = pos_local | (fb & pos_mask)
            neg_mask = neg_local | (fb & neg_mask)

    min_anchor = min(64, max(16, n // 50))
    if int(np.sum(pos_mask)) < min_anchor:
        order = np.argsort(anchor)
        pos_mask = np.zeros((n,), dtype=bool)
        pos_mask[order[:min_anchor]] = True
    if int(np.sum(neg_mask)) < min_anchor:
        order = np.argsort(anchor)
        neg_mask = np.zeros((n,), dtype=bool)
        neg_mask[order[-min_anchor:]] = True

    diag = {
        "num_rows": int(n),
        "pos_quantile": float(pos_quantile),
        "neg_quantile": float(neg_quantile),
        "anchor_mean": float(np.mean(anchor)),
        "anchor_std": float(np.std(anchor)),
        "pos_count": int(np.sum(pos_mask)),
        "neg_count": int(np.sum(neg_mask)),
        "q_pos": q_pos,
        "q_neg": q_neg,
        "family_aware": bool(family_aware),
        "family_min_samples": int(max(4, family_min_samples)),
        "family_local_count": int(family_local_count),
        "family_fallback_rows": int(family_fallback_rows),
        "metamorphic_weight": float(meta_w),
        "metamorphic_mean": float(np.mean(meta_combo)) if len(meta_combo) else 0.0,
        "metamorphic_std": float(np.std(meta_combo)) if len(meta_combo) else 0.0,
    }
    return anchor.astype(np.float32), pos_mask.astype(bool), neg_mask.astype(bool), diag


def build_energy_feature_matrix(
    rows: Sequence[Dict[str, Any]],
    embeddings: np.ndarray,
    mode: str = "full",
    include_metamorphic: bool = False,
    include_score_head: bool = False,
    include_text_aux_features: bool = False,
    include_closure_features: bool = False,
    include_protofree_extended_features: bool = True,
    text_embeddings: Optional[np.ndarray] = None,
    graph_embeddings: Optional[np.ndarray] = None,
    prototype_centers: Optional[np.ndarray] = None,
    prototype_radii: Optional[np.ndarray] = None,
    prototype_weights: Optional[np.ndarray] = None,
    free_energy_temp: float = 0.35,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    n = min(len(rows), int(len(embeddings)))
    if n <= 0:
        return np.zeros((0, 0), dtype=np.float32), [], {"embedding_dim": 0}

    emb = np.asarray(embeddings[:n], dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError("embeddings must be rank-2")
    emb_dim = int(emb.shape[1])
    used_rows = list(rows[:n])
    bag_max_base, bag_mean_base, bag_max_proto, bag_size = _build_group_stats(used_rows, n)
    bag_size_log = np.log1p(bag_size).astype(np.float32)
    score_head_risk = np.asarray([_safe_float(r, "score_head_risk", 0.0) for r in used_rows], dtype=np.float32)
    score_head_rank_pct = rank_normalize01(score_head_risk).astype(np.float32)
    closure_cols = list(CLOSURE_FEATURE_NAMES)
    closure_feat = np.zeros((n, len(closure_cols)), dtype=np.float32)
    if bool(include_closure_features):
        for i, row in enumerate(used_rows):
            for j, key in enumerate(closure_cols):
                closure_feat[i, j] = _safe_float(row, key, 0.0)
    text_aux_cols = [
        "graph_text_gap_l2",
        "graph_text_gap_cos",
        "text_norm",
        "text_graph_norm_ratio",
    ]
    text_aux = np.zeros((n, len(text_aux_cols)), dtype=np.float32)
    text_aux_available = False
    text_aux_graph_dim = 0
    text_aux_text_dim = 0
    if bool(include_text_aux_features):
        g = None
        t = None
        if graph_embeddings is not None:
            g = np.asarray(graph_embeddings[:n], dtype=np.float32)
        if text_embeddings is not None:
            t = np.asarray(text_embeddings[:n], dtype=np.float32)
        if (
            g is not None
            and t is not None
            and g.ndim == 2
            and t.ndim == 2
            and g.shape[0] == n
            and t.shape[0] == n
            and g.shape[1] == t.shape[1]
            and g.shape[1] > 0
        ):
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            diff = (g - t).astype(np.float32)
            g_norm = np.linalg.norm(g, axis=1).astype(np.float32)
            t_norm = np.linalg.norm(t, axis=1).astype(np.float32)
            text_aux[:, 0] = np.linalg.norm(diff, axis=1).astype(np.float32)
            denom = np.maximum(g_norm * t_norm, 1e-6).astype(np.float32)
            cos_sim = np.sum(g * t, axis=1).astype(np.float32) / denom
            cos_sim = np.clip(cos_sim, -1.0, 1.0).astype(np.float32)
            text_aux[:, 1] = (1.0 - cos_sim).astype(np.float32)
            text_aux[:, 2] = t_norm
            text_aux[:, 3] = (t_norm / np.maximum(g_norm, 1e-6)).astype(np.float32)
            text_aux_available = True
            text_aux_graph_dim = int(g.shape[1])
            text_aux_text_dim = int(t.shape[1])
        else:
            if g is not None and g.ndim == 2 and g.shape[0] == n:
                text_aux_graph_dim = int(g.shape[1])
            if t is not None and t.ndim == 2 and t.shape[0] == n:
                text_aux_text_dim = int(t.shape[1])

    mode_key = str(mode or "full").strip().lower()
    if mode_key == "proto_hybrid":
        scalar_cols = [
            "nearest_distance",
            "boundary_margin",
            "view_gap",
            "proto_score",
            "view_score",
            "density_risk",
            "instability",
            "boundary_proximity",
            "mechanism_risk",
            "family_risk",
            "global_gap",
            "fused_gap",
        ]
        if include_metamorphic:
            scalar_cols.extend(
                [
                    "metamorphic_instability",
                    "metamorphic_nearest_delta",
                    "metamorphic_margin_delta",
                    "metamorphic_view_delta",
                    "metamorphic_proto_switch_rate",
                ]
            )
        extra_cols = 6 + (2 if include_score_head else 0)
        scalar = np.zeros((n, len(scalar_cols) + extra_cols), dtype=np.float32)
        for i, row in enumerate(used_rows):
            for j, key in enumerate(scalar_cols):
                scalar[i, j] = _safe_float(row, key, 0.0)
            proto_score = _safe_float(row, "proto_score", 0.0)
            view_score = _safe_float(row, "view_score", 0.0)
            base_final = _safe_float(row, "base_final_risk", 0.0)
            scalar[i, len(scalar_cols) + 0] = float(base_final - proto_score)
            scalar[i, len(scalar_cols) + 1] = float(abs(proto_score - view_score))
            scalar[i, len(scalar_cols) + 2] = bag_max_base[i]
            scalar[i, len(scalar_cols) + 3] = bag_mean_base[i]
            scalar[i, len(scalar_cols) + 4] = bag_max_proto[i]
            scalar[i, len(scalar_cols) + 5] = bag_size_log[i]
            if include_score_head:
                scalar[i, len(scalar_cols) + 6] = score_head_risk[i]
                scalar[i, len(scalar_cols) + 7] = score_head_rank_pct[i]
        feat = np.nan_to_num(scalar, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        cols = scalar_cols + [
            "risk_refine_gap",
            "proto_view_gap_abs",
            "bag_max_base_risk",
            "bag_mean_base_risk",
            "bag_max_proto_score",
            "bag_size_log",
        ]
        if include_score_head:
            cols.extend(["score_head_risk", "score_head_rank_pct"])
        if bool(include_closure_features):
            feat = np.concatenate([feat, closure_feat], axis=1).astype(np.float32)
            cols.extend(closure_cols)
        if bool(include_text_aux_features):
            feat = np.concatenate([feat, text_aux], axis=1).astype(np.float32)
            cols.extend(text_aux_cols)
    elif mode_key == "proto_free_energy":
        free_feat = compute_radius_normalized_free_energy(
            emb,
            centers=prototype_centers,
            radii=prototype_radii,
            prototype_weights=prototype_weights,
            temperature=float(free_energy_temp),
        )
        scalar_cols = [
            "proto_free_energy",
            "proto_resp_entropy",
            "proto_nearest_norm_distance",
        ]
        if bool(include_protofree_extended_features):
            scalar_cols.extend(
                [
                    "proto_resp_max",
                    "proto_resp_gap",
                    "proto_top2_norm_gap",
                ]
            )
        scalar_cols.extend(
            [
                "boundary_margin",
                "view_gap",
                "boundary_proximity",
                "mechanism_risk",
                "family_risk",
                "global_gap",
                "fused_gap",
            ]
        )
        extra_cols = 6 + (2 if include_score_head else 0)
        scalar = np.zeros((n, len(scalar_cols) + extra_cols), dtype=np.float32)
        for i, row in enumerate(used_rows):
            col_idx = 0
            scalar[i, col_idx] = float(free_feat["proto_free_energy"][i])
            col_idx += 1
            scalar[i, col_idx] = float(free_feat["proto_resp_entropy"][i])
            col_idx += 1
            scalar[i, col_idx] = float(free_feat["proto_nearest_norm_distance"][i])
            col_idx += 1
            if bool(include_protofree_extended_features):
                scalar[i, col_idx] = float(free_feat["proto_resp_max"][i])
                col_idx += 1
                scalar[i, col_idx] = float(free_feat["proto_resp_gap"][i])
                col_idx += 1
                scalar[i, col_idx] = float(free_feat["proto_top2_norm_gap"][i])
                col_idx += 1
            scalar[i, col_idx] = _safe_float(row, "boundary_margin", 0.0)
            col_idx += 1
            scalar[i, col_idx] = _safe_float(row, "view_gap", 0.0)
            col_idx += 1
            scalar[i, col_idx] = _safe_float(row, "boundary_proximity", 0.0)
            col_idx += 1
            scalar[i, col_idx] = _safe_float(row, "mechanism_risk", 0.0)
            col_idx += 1
            scalar[i, col_idx] = _safe_float(row, "family_risk", 0.0)
            col_idx += 1
            scalar[i, col_idx] = _safe_float(row, "global_gap", 0.0)
            col_idx += 1
            scalar[i, col_idx] = _safe_float(row, "fused_gap", 0.0)
            proto_score = _safe_float(row, "proto_score", 0.0)
            view_score = _safe_float(row, "view_score", 0.0)
            base_final = _safe_float(row, "base_final_risk", 0.0)
            scalar[i, len(scalar_cols) + 0] = float(base_final - proto_score)
            scalar[i, len(scalar_cols) + 1] = float(abs(proto_score - view_score))
            scalar[i, len(scalar_cols) + 2] = bag_max_base[i]
            scalar[i, len(scalar_cols) + 3] = bag_mean_base[i]
            scalar[i, len(scalar_cols) + 4] = bag_max_proto[i]
            scalar[i, len(scalar_cols) + 5] = bag_size_log[i]
            if include_score_head:
                scalar[i, len(scalar_cols) + 6] = score_head_risk[i]
                scalar[i, len(scalar_cols) + 7] = score_head_rank_pct[i]
        feat = np.nan_to_num(scalar, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        cols = scalar_cols + [
            "risk_refine_gap",
            "proto_view_gap_abs",
            "bag_max_base_risk",
            "bag_mean_base_risk",
            "bag_max_proto_score",
            "bag_size_log",
        ]
        if include_score_head:
            cols.extend(["score_head_risk", "score_head_rank_pct"])
        if bool(include_closure_features):
            feat = np.concatenate([feat, closure_feat], axis=1).astype(np.float32)
            cols.extend(closure_cols)
        if bool(include_text_aux_features):
            feat = np.concatenate([feat, text_aux], axis=1).astype(np.float32)
            cols.extend(text_aux_cols)
    else:
        scalar_cols = [
            "base_final_risk",
            "boundary_margin",
            "view_gap",
            "proto_score",
            "view_score",
            "density_risk",
            "instability",
            "boundary_proximity",
            "mechanism_risk",
            "family_risk",
            "global_gap",
            "fused_gap",
        ]
        if include_metamorphic:
            scalar_cols.extend(
                [
                    "metamorphic_instability",
                    "metamorphic_nearest_delta",
                    "metamorphic_margin_delta",
                    "metamorphic_view_delta",
                    "metamorphic_proto_switch_rate",
                ]
            )
        extra_cols = 3 + (2 if include_score_head else 0)
        scalar = np.zeros((n, len(scalar_cols) + extra_cols), dtype=np.float32)
        for i, row in enumerate(used_rows):
            for j, key in enumerate(scalar_cols):
                scalar[i, j] = _safe_float(row, key, 0.0)
            scalar[i, len(scalar_cols) + 0] = bag_max_base[i]
            scalar[i, len(scalar_cols) + 1] = bag_mean_base[i]
            scalar[i, len(scalar_cols) + 2] = bag_size_log[i]
            if include_score_head:
                scalar[i, len(scalar_cols) + 3] = score_head_risk[i]
                scalar[i, len(scalar_cols) + 4] = score_head_rank_pct[i]
        feat = np.concatenate([emb, scalar], axis=1).astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        cols = [f"embed_{i}" for i in range(emb_dim)] + scalar_cols + [
            "bag_max_base_risk",
            "bag_mean_base_risk",
            "bag_size_log",
        ]
        if include_score_head:
            cols.extend(["score_head_risk", "score_head_rank_pct"])
        if bool(include_closure_features):
            feat = np.concatenate([feat, closure_feat], axis=1).astype(np.float32)
            cols.extend(closure_cols)
        if bool(include_text_aux_features):
            feat = np.concatenate([feat, text_aux], axis=1).astype(np.float32)
            cols.extend(text_aux_cols)
    diag = {
        "mode": mode_key,
        "include_metamorphic": bool(include_metamorphic),
        "include_score_head": bool(include_score_head),
        "include_text_aux_features": bool(include_text_aux_features),
        "include_closure_features": bool(include_closure_features),
        "include_protofree_extended_features": bool(include_protofree_extended_features),
        "text_aux_available": bool(text_aux_available),
        "text_aux_graph_dim": int(text_aux_graph_dim),
        "text_aux_text_dim": int(text_aux_text_dim),
        "embedding_dim": emb_dim,
        "num_rows": n,
        "num_function_groups": int(len({_function_group_id(r) for r in used_rows})),
        "mean_bag_size": float(np.mean(bag_size)) if len(bag_size) else 0.0,
        "max_bag_size": int(np.max(bag_size)) if len(bag_size) else 0,
        "free_energy_temp": float(free_energy_temp),
    }
    return feat, cols, diag


def standardize_fit_transform(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("x must be rank-2")
    mean = np.mean(arr, axis=0).astype(np.float32)
    std = np.std(arr, axis=0).astype(np.float32)
    std = np.where(std <= 1e-6, 1.0, std).astype(np.float32)
    out = ((arr - mean) / std).astype(np.float32)
    return out, mean, std


def align_feature_matrix_columns(
    feat: np.ndarray,
    cols: Sequence[str],
    target_cols: Sequence[str],
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    arr = np.asarray(feat, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("feat must be rank-2")
    src_cols = [str(c) for c in cols]
    dst_cols = [str(c) for c in target_cols]
    src_index = {name: i for i, name in enumerate(src_cols)}
    out = np.zeros((arr.shape[0], len(dst_cols)), dtype=np.float32)
    matched = 0
    missing: List[str] = []
    for j, name in enumerate(dst_cols):
        idx = src_index.get(name)
        if idx is None:
            missing.append(name)
            continue
        out[:, j] = arr[:, idx]
        matched += 1
    diag = {
        "src_dim": int(arr.shape[1]),
        "target_dim": int(len(dst_cols)),
        "matched_cols": int(matched),
        "missing_cols": missing,
    }
    return out.astype(np.float32), dst_cols, diag


def standardize_transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    mu = np.asarray(mean, dtype=np.float32)
    sig = np.asarray(std, dtype=np.float32)
    sig = np.where(sig <= 1e-6, 1.0, sig).astype(np.float32)
    return ((arr - mu) / sig).astype(np.float32)


def normalize01(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size <= 0:
        return arr.astype(np.float32)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo + 1e-8)).astype(np.float32)


def robust_quantile01(x: np.ndarray, lo_q: float = 0.05, hi_q: float = 0.95) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size <= 0:
        return arr.astype(np.float32)
    lo = float(np.quantile(arr, lo_q))
    hi = float(np.quantile(arr, hi_q))
    if hi - lo <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    clipped = np.clip(arr, lo, hi)
    return ((clipped - lo) / (hi - lo + 1e-8)).astype(np.float32)


def rank_normalize01(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    n = int(arr.size)
    if n <= 0:
        return arr.astype(np.float32)
    if n == 1:
        return np.zeros_like(arr, dtype=np.float32)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.zeros((n,), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=n, endpoint=True, dtype=np.float32)
    return ranks


def robust_mix01(x: np.ndarray, rank_weight: float = 0.65) -> np.ndarray:
    rank_part = rank_normalize01(x)
    quant_part = robust_quantile01(x)
    w = float(min(max(rank_weight, 0.0), 1.0))
    return (w * rank_part + (1.0 - w) * quant_part).astype(np.float32)
