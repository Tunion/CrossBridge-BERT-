from __future__ import annotations

import collections
import math
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from common.closure_feature_utils import (
    MECHANISM_RELATIONS,
    ROLE_KEYS,
    ROLE_PAIRS,
    build_undirected_adj,
    role_node_sets,
    shortest_path_len_between_sets,
)


SUMMARY_EDGE_KINDS: Tuple[str, ...] = (
    "local_call_summary",
    "shared_object_summary",
    "cross_function_state_flow",
    "state_write_summary",
    "state_read_summary",
)

BRIDGE_KEYWORDS: Tuple[str, ...] = (
    "bridge",
    "deposit",
    "withdraw",
    "execute",
    "proposal",
    "verify",
    "header",
    "proof",
    "keeper",
    "validator",
    "relay",
    "mint",
    "burn",
    "swap",
    "unlock",
    "lock",
    "message",
    "chain",
    "sig",
)

MECHANISM_HEAD_FEATURE_NAMES: Tuple[str, ...] = (
    "full_node_count_log",
    "full_edge_count_log",
    "skeleton_node_ratio",
    "skeleton_edge_ratio",
    "statement_ratio",
    "condition_ratio",
    "data_object_ratio",
    "state_var_ratio",
    "role_presence_ratio",
    "auth_role_count_log",
    "ext_role_count_log",
    "state_role_count_log",
    "pair_connected_ratio",
    "pair_path_mean_norm",
    "pair_path_max_norm",
    "skeleton_role_retain",
    "skeleton_pair_retain",
    "skeleton_mechanism_complete",
    "skeleton_mechanism_links_norm",
    "function_count_log",
    "cross_function_statement_ratio",
    "summary_edge_ratio",
    "local_call_summary_ratio",
    "shared_object_summary_ratio",
    "cross_function_state_flow_ratio",
    "role_statement_ratio",
    "multi_role_statement_ratio",
    "auth_ext_edge_ratio",
    "ext_state_edge_ratio",
    "auth_state_edge_ratio",
    "keyword_hit_log",
    "mechanism_text_len_log",
    "bridge_intent_score",
    "helper_pollution_score",
)


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    if float(den) <= 0.0:
        return float(default)
    return float(num) / float(den)


def _log1p(value: float) -> float:
    return float(math.log1p(max(0.0, float(value))))


def _normalize_path(path_like: Any) -> str:
    return str(path_like or "").replace("\\", "/").strip()


def _row_file_key(row: Dict[str, Any]) -> str:
    return str(
        row.get("relative_source_path")
        or row.get("file")
        or row.get("source_path")
        or row.get("contract_id")
        or "<unknown_file>"
    )


def _function_name_list(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return sorted({str(x).strip() for x in raw if str(x).strip()})
    return sorted({x.strip() for x in str(raw or "").split("|") if x.strip()})


def _statement_nodes(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [node for node in graph.get("nodes", []) if node.get("type") == "statement"]


def _keyword_hits(texts: Iterable[str]) -> int:
    joined = " ".join(str(x or "") for x in texts).lower()
    if not joined:
        return 0
    hits = 0
    for kw in BRIDGE_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", joined):
            hits += 1
    return hits


def _line_candidates_from_graph(graph: Dict[str, Any]) -> List[int]:
    out: Set[int] = set()
    for node in graph.get("nodes", []):
        if node.get("type") != "statement":
            continue
        line = node.get("line")
        if not isinstance(line, int):
            continue
        end_line = node.get("end_line", line)
        if not isinstance(end_line, int):
            end_line = line
        lo = min(int(line), int(end_line))
        hi = max(int(line), int(end_line))
        if hi - lo <= 4:
            for ln in range(lo, hi + 1):
                out.add(int(ln))
        else:
            out.add(int(lo))
            out.add(int(hi))
    return sorted(out)


def extract_mechanism_head_feature_row(
    dual_row: Dict[str, Any],
    max_hops: int = 6,
) -> Dict[str, Any]:
    full_graph = dual_row.get("full_graph", {}) or {}
    skeleton_graph = dual_row.get("skeleton_graph", {}) or {}
    full_nodes = list(full_graph.get("nodes", []))
    skel_nodes = list(skeleton_graph.get("nodes", []))
    full_edges = list(full_graph.get("edges", []))
    skel_edges = list(skeleton_graph.get("edges", []))
    n_full = max(1, len(full_nodes))
    e_full = max(1, len(full_edges))

    stmt_nodes = _statement_nodes(full_graph)
    stmt_count = len(stmt_nodes)
    condition_count = sum(1 for node in full_nodes if node.get("type") == "condition")
    data_object_count = sum(1 for node in full_nodes if node.get("type") == "data_object")
    state_var_count = sum(1 for node in full_nodes if node.get("type") == "state_var")

    full_roles = role_node_sets(full_graph)
    skel_roles = role_node_sets(skeleton_graph)
    role_presence_count = sum(1 for key in ROLE_KEYS if full_roles.get(key))
    auth_role_count = len(full_roles["auth_constraint"])
    ext_role_count = len(full_roles["external_interaction"])
    state_role_count = len(full_roles["state_change"])

    full_adj = build_undirected_adj(full_graph, rel_types=MECHANISM_RELATIONS)
    pair_lengths: List[int] = []
    connected_pairs = 0
    for left, right in ROLE_PAIRS:
        plen = shortest_path_len_between_sets(
            full_adj,
            full_roles.get(left, set()),
            full_roles.get(right, set()),
            max_hops=max_hops,
        )
        if plen is None:
            continue
        connected_pairs += 1
        pair_lengths.append(int(plen))
    pair_connected_ratio = _safe_ratio(float(connected_pairs), float(len(ROLE_PAIRS)), default=0.0)
    pair_path_mean_norm = _safe_ratio(float(sum(pair_lengths)), float(max(1, len(pair_lengths)) * max_hops), default=1.0)
    pair_path_max_norm = _safe_ratio(float(max(pair_lengths) if pair_lengths else max_hops), float(max_hops), default=1.0)

    role_retain_terms: List[float] = []
    for key in ROLE_KEYS:
        full_cnt = len(full_roles.get(key, set()))
        if full_cnt <= 0:
            continue
        role_retain_terms.append(
            min(1.0, _safe_ratio(float(len(skel_roles.get(key, set()))), float(full_cnt), default=0.0))
        )
    skeleton_role_retain = float(sum(role_retain_terms) / len(role_retain_terms)) if role_retain_terms else 0.0

    skel_adj = build_undirected_adj(skeleton_graph, rel_types=MECHANISM_RELATIONS)
    skel_connected_pairs = 0
    for left, right in ROLE_PAIRS:
        plen = shortest_path_len_between_sets(
            skel_adj,
            skel_roles.get(left, set()),
            skel_roles.get(right, set()),
            max_hops=max_hops,
        )
        if plen is not None:
            skel_connected_pairs += 1
    skeleton_pair_retain = _safe_ratio(float(skel_connected_pairs), float(len(ROLE_PAIRS)), default=0.0)

    sk_meta = (skeleton_graph.get("meta", {}) or {}) if isinstance(skeleton_graph, dict) else {}
    mechanism_complete = 1.0 if bool(sk_meta.get("mechanism_complete", False)) else 0.0
    mechanism_links_norm = min(1.0, _safe_ratio(float(sk_meta.get("mechanism_links", 0.0)), 3.0, default=0.0))

    stmt_functions = sorted({str(node.get("function", "")).strip() for node in stmt_nodes if str(node.get("function", "")).strip()})
    function_count = len(stmt_functions)
    cross_function_stmt = 0
    if function_count > 1:
        first_fn = stmt_functions[0] if stmt_functions else ""
        for node in stmt_nodes:
            cur = str(node.get("function", "")).strip()
            if cur and cur != first_fn:
                cross_function_stmt += 1

    summary_counts = {key: 0 for key in SUMMARY_EDGE_KINDS}
    summary_total = 0
    role_edge_hits = {
        "auth_ext": 0,
        "ext_state": 0,
        "auth_state": 0,
    }
    node_role_map: Dict[int, Set[str]] = {}
    for node in full_nodes:
        node_role_map[int(node.get("id", -1))] = {
            str(role) for role in (node.get("roles", []) or []) if str(role)
        }
    for edge in full_edges:
        kind = str(edge.get("kind", "")).strip().lower()
        if kind in summary_counts:
            summary_counts[kind] += 1
            summary_total += 1
        src_roles = node_role_map.get(int(edge.get("src", -1)), set())
        dst_roles = node_role_map.get(int(edge.get("dst", -1)), set())
        union_roles = src_roles.union(dst_roles)
        if "auth_constraint" in union_roles and "external_interaction" in union_roles:
            role_edge_hits["auth_ext"] += 1
        if "external_interaction" in union_roles and "state_change" in union_roles:
            role_edge_hits["ext_state"] += 1
        if "auth_constraint" in union_roles and "state_change" in union_roles:
            role_edge_hits["auth_state"] += 1

    role_stmt_count = 0
    multi_role_stmt_count = 0
    stmt_texts: List[str] = []
    for node in stmt_nodes:
        roles = {str(role) for role in (node.get("roles", []) or []) if str(role)}
        if roles.intersection(ROLE_KEYS):
            role_stmt_count += 1
        if len(roles.intersection(ROLE_KEYS)) >= 2:
            multi_role_stmt_count += 1
        stmt_texts.append(str(node.get("text", "") or ""))

    fn_text = " ".join(_function_name_list(dual_row.get("function_names", [])))
    mech_text = str(dual_row.get("mechanism_text", "") or "")
    keyword_hits = _keyword_hits([fn_text, mech_text, *stmt_texts[:32]])
    mechanism_text_len = len(mech_text.split())

    role_presence_ratio = _safe_ratio(float(role_presence_count), 3.0, default=0.0)
    role_statement_ratio = _safe_ratio(float(role_stmt_count), float(max(1, stmt_count)), default=0.0)
    keyword_score = min(1.0, _safe_ratio(float(keyword_hits), 4.0, default=0.0))
    state_ext_bonus = 1.0 if (ext_role_count > 0 and state_role_count > 0) else 0.0
    bridge_intent_score = min(
        1.0,
        0.40 * role_presence_ratio
        + 0.25 * role_statement_ratio
        + 0.20 * keyword_score
        + 0.15 * state_ext_bonus,
    )
    helper_pollution_score = min(
        1.0,
        0.40 * _safe_ratio(float(summary_total), float(e_full), default=0.0)
        + 0.25 * _safe_ratio(float(data_object_count + condition_count), float(n_full), default=0.0)
        + 0.35 * (1.0 - bridge_intent_score),
    )

    line_candidates = _line_candidates_from_graph(full_graph)
    out: Dict[str, Any] = {
        "slice_id": str(dual_row.get("slice_id", "")),
        "relative_source_path": _normalize_path(_row_file_key(dual_row)),
        "file": _normalize_path(_row_file_key(dual_row)),
        "function_names": "|".join(_function_name_list(dual_row.get("function_names", []))),
        "line_candidates": "|".join(str(x) for x in line_candidates),
        "proto_score": float(dual_row.get("proto_score", 0.0) or 0.0),
        "view_score": float(dual_row.get("view_score", 0.0) or 0.0),
        "boundary_margin": float(dual_row.get("boundary_margin", 0.0) or 0.0),
        "full_node_count_log": _log1p(float(len(full_nodes))),
        "full_edge_count_log": _log1p(float(len(full_edges))),
        "skeleton_node_ratio": min(1.0, _safe_ratio(float(len(skel_nodes)), float(n_full), default=0.0)),
        "skeleton_edge_ratio": min(1.0, _safe_ratio(float(len(skel_edges)), float(e_full), default=0.0)),
        "statement_ratio": _safe_ratio(float(stmt_count), float(n_full), default=0.0),
        "condition_ratio": _safe_ratio(float(condition_count), float(n_full), default=0.0),
        "data_object_ratio": _safe_ratio(float(data_object_count), float(n_full), default=0.0),
        "state_var_ratio": _safe_ratio(float(state_var_count), float(n_full), default=0.0),
        "role_presence_ratio": role_presence_ratio,
        "auth_role_count_log": _log1p(float(auth_role_count)),
        "ext_role_count_log": _log1p(float(ext_role_count)),
        "state_role_count_log": _log1p(float(state_role_count)),
        "pair_connected_ratio": pair_connected_ratio,
        "pair_path_mean_norm": pair_path_mean_norm,
        "pair_path_max_norm": pair_path_max_norm,
        "skeleton_role_retain": skeleton_role_retain,
        "skeleton_pair_retain": skeleton_pair_retain,
        "skeleton_mechanism_complete": mechanism_complete,
        "skeleton_mechanism_links_norm": mechanism_links_norm,
        "function_count_log": _log1p(float(function_count)),
        "cross_function_statement_ratio": _safe_ratio(float(cross_function_stmt), float(max(1, stmt_count)), default=0.0),
        "summary_edge_ratio": _safe_ratio(float(summary_total), float(e_full), default=0.0),
        "local_call_summary_ratio": _safe_ratio(float(summary_counts["local_call_summary"]), float(e_full), default=0.0),
        "shared_object_summary_ratio": _safe_ratio(float(summary_counts["shared_object_summary"]), float(e_full), default=0.0),
        "cross_function_state_flow_ratio": _safe_ratio(float(summary_counts["cross_function_state_flow"]), float(e_full), default=0.0),
        "role_statement_ratio": role_statement_ratio,
        "multi_role_statement_ratio": _safe_ratio(float(multi_role_stmt_count), float(max(1, stmt_count)), default=0.0),
        "auth_ext_edge_ratio": _safe_ratio(float(role_edge_hits["auth_ext"]), float(e_full), default=0.0),
        "ext_state_edge_ratio": _safe_ratio(float(role_edge_hits["ext_state"]), float(e_full), default=0.0),
        "auth_state_edge_ratio": _safe_ratio(float(role_edge_hits["auth_state"]), float(e_full), default=0.0),
        "keyword_hit_log": _log1p(float(keyword_hits)),
        "mechanism_text_len_log": _log1p(float(mechanism_text_len)),
        "bridge_intent_score": bridge_intent_score,
        "helper_pollution_score": helper_pollution_score,
    }
    return out


def extract_mechanism_head_feature_rows(
    dual_rows: Sequence[Dict[str, Any]],
    max_hops: int = 6,
) -> List[Dict[str, Any]]:
    return [extract_mechanism_head_feature_row(row, max_hops=max_hops) for row in dual_rows]


def build_mechanism_head_feature_matrix(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(row.get(name, 0.0)) for name in MECHANISM_HEAD_FEATURE_NAMES] for row in rows],
        dtype=np.float32,
    )


def _qstat(values: np.ndarray, q_low: float = 0.10, q_high: float = 0.95) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 0:
        return {"lo": 0.0, "hi": 1.0}
    lo = float(np.quantile(arr, q_low))
    hi = float(np.quantile(arr, q_high))
    if hi <= lo + 1e-8:
        hi = lo + 1.0
    return {"lo": lo, "hi": hi}


def _normalize_by_stat(values: np.ndarray, stat: Dict[str, float]) -> np.ndarray:
    lo = float(stat.get("lo", 0.0))
    hi = float(stat.get("hi", lo + 1.0))
    if hi <= lo + 1e-8:
        hi = lo + 1.0
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


@dataclass
class MechanismSliceHeadModel:
    random_state: int = 42
    max_hops: int = 6
    gate_quantile: float = 0.55
    slice_quantile: float = 0.95
    function_quantile: float = 0.95
    threshold_std_factor: float = 1.05
    threshold_mad_factor: float = 1.50
    corroboration_alpha: float = 0.40
    corroboration_vp_weight: float = 0.80
    corroboration_margin_weight: float = 0.20
    feature_names: List[str] = field(default_factory=lambda: list(MECHANISM_HEAD_FEATURE_NAMES))
    feature_index: Dict[str, int] = field(default_factory=dict)
    scaler: RobustScaler | None = None
    kmeans: MiniBatchKMeans | None = None
    iforest: IsolationForest | None = None
    gate_floor: float = 0.25
    selected_count: int = 0
    residual_stat: Dict[str, float] = field(default_factory=dict)
    if_stat: Dict[str, float] = field(default_factory=dict)
    closure_stat: Dict[str, float] = field(default_factory=dict)
    mechanism_stat: Dict[str, float] = field(default_factory=dict)
    slice_threshold: float = 0.50
    function_threshold: float = 0.50
    train_summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.feature_index = {name: idx for idx, name in enumerate(self.feature_names)}

    def _col(self, x: np.ndarray, name: str) -> np.ndarray:
        return x[:, self.feature_index[name]]

    def _aux_col(self, feature_rows: Sequence[Dict[str, Any]], name: str) -> np.ndarray:
        return np.asarray([float(row.get(name, 0.0) or 0.0) for row in feature_rows], dtype=np.float32)

    def _closure_inconsistency(self, x: np.ndarray) -> np.ndarray:
        role_presence = self._col(x, "role_presence_ratio")
        pair_connected = self._col(x, "pair_connected_ratio")
        pair_path_mean = self._col(x, "pair_path_mean_norm")
        skel_role_retain = self._col(x, "skeleton_role_retain")
        skel_pair_retain = self._col(x, "skeleton_pair_retain")
        mech_complete = self._col(x, "skeleton_mechanism_complete")
        mech_links = self._col(x, "skeleton_mechanism_links_norm")
        cross_fn = self._col(x, "cross_function_statement_ratio")
        summary_ratio = self._col(x, "summary_edge_ratio")
        role_stmt = self._col(x, "role_statement_ratio")
        multi_role_stmt = self._col(x, "multi_role_statement_ratio")
        bridge_intent = self._col(x, "bridge_intent_score")

        gap = (
            0.16 * (1.0 - role_presence)
            + 0.18 * (1.0 - pair_connected)
            + 0.12 * pair_path_mean
            + 0.12 * (1.0 - skel_role_retain)
            + 0.10 * (1.0 - skel_pair_retain)
            + 0.08 * (1.0 - mech_complete)
            + 0.06 * (1.0 - mech_links)
            + 0.08 * np.maximum(0.0, cross_fn - 0.45)
            + 0.05 * np.maximum(0.0, summary_ratio - 0.18)
            + 0.03 * (1.0 - role_stmt)
            + 0.02 * (1.0 - multi_role_stmt)
        )
        gap = gap * (0.55 + 0.45 * bridge_intent)
        return np.clip(gap, 0.0, 2.0).astype(np.float32)

    def _mechanism_load(self, x: np.ndarray) -> np.ndarray:
        node_size = self._col(x, "full_node_count_log")
        edge_size = self._col(x, "full_edge_count_log")
        bridge_intent = self._col(x, "bridge_intent_score")
        keyword_hit = self._col(x, "keyword_hit_log")
        state_role = self._col(x, "state_role_count_log")
        ext_role = self._col(x, "ext_role_count_log")
        pair_connected = self._col(x, "pair_connected_ratio")
        cross_fn = self._col(x, "cross_function_statement_ratio")
        summary_ratio = self._col(x, "summary_edge_ratio")
        helper_pollution = self._col(x, "helper_pollution_score")
        return (
            0.16 * node_size
            + 0.16 * edge_size
            + 0.14 * bridge_intent
            + 0.12 * keyword_hit
            + 0.12 * state_role
            + 0.10 * ext_role
            + 0.08 * pair_connected
            + 0.08 * cross_fn
            + 0.08 * summary_ratio
            - 0.06 * helper_pollution
        ).astype(np.float32)

    def _pick_selected_mask(self, x: np.ndarray) -> np.ndarray:
        bridge = self._col(x, "bridge_intent_score")
        helper = self._col(x, "helper_pollution_score")
        mechanism = self._mechanism_load(x)
        floor = float(np.quantile(bridge, self.gate_quantile))
        floor = min(0.60, max(0.20, floor))
        self.gate_floor = floor
        mechanism_floor = float(np.quantile(mechanism, self.gate_quantile))
        mask = (bridge >= floor) & ((bridge >= (helper * 0.85)) | (mechanism >= mechanism_floor))
        if int(mask.sum()) < max(512, int(0.20 * len(x))):
            fallback_floor = float(np.quantile(bridge, max(0.20, self.gate_quantile - 0.15)))
            fallback_floor = min(0.55, max(0.15, fallback_floor))
            self.gate_floor = fallback_floor
            mask = bridge >= fallback_floor
        return mask

    def _auto_threshold(self, values: np.ndarray, fallback_quantile: float) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size < 128 or len(np.unique(np.round(arr, 6))) < 16:
            base = float(np.quantile(arr, fallback_quantile))
            mu = float(np.mean(arr))
            sigma = float(np.std(arr))
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            tail_guard = mu + float(self.threshold_std_factor) * sigma
            robust_guard = med + float(self.threshold_mad_factor) * 1.4826 * mad
            q995 = float(np.quantile(arr, 0.995))
            return float(min(max(base, tail_guard, robust_guard), q995))
        fallback = float(np.quantile(arr, fallback_quantile))
        try:
            gm = GaussianMixture(n_components=2, random_state=self.random_state)
            gm.fit(arr.reshape(-1, 1))
            means = gm.means_.reshape(-1)
            order = np.argsort(means)
            lo_idx = int(order[0])
            hi_idx = int(order[1])
            if float(means[hi_idx] - means[lo_idx]) < 0.05:
                return fallback
            grid = np.linspace(float(arr.min()), float(arr.max()), 1024, dtype=float)
            probs = gm.predict_proba(grid.reshape(-1, 1))
            region = (grid >= float(means[lo_idx])) & (grid <= float(means[hi_idx]))
            if not np.any(region):
                return fallback
            grid_mid = grid[region]
            probs_mid = probs[region]
            idx = int(np.argmin(np.abs(probs_mid[:, lo_idx] - probs_mid[:, hi_idx])))
            thr = float(grid_mid[idx])
            q80 = float(np.quantile(arr, 0.80))
            q995 = float(np.quantile(arr, 0.995))
            mu = float(np.mean(arr))
            sigma = float(np.std(arr))
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            tail_guard = mu + float(self.threshold_std_factor) * sigma
            robust_guard = med + float(self.threshold_mad_factor) * 1.4826 * mad
            if thr < q80:
                thr = q80
            thr = max(thr, tail_guard, robust_guard)
            if thr > q995:
                thr = min(fallback, q995)
            return thr
        except Exception:
            mu = float(np.mean(arr))
            sigma = float(np.std(arr))
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            tail_guard = mu + float(self.threshold_std_factor) * sigma
            robust_guard = med + float(self.threshold_mad_factor) * 1.4826 * mad
            q995 = float(np.quantile(arr, 0.995))
            return float(min(max(fallback, tail_guard, robust_guard), q995))

    def fit(self, feature_rows: Sequence[Dict[str, Any]]) -> "MechanismSliceHeadModel":
        x = build_mechanism_head_feature_matrix(feature_rows)
        mask = self._pick_selected_mask(x)
        self.selected_count = int(mask.sum())
        if self.selected_count <= 0:
            raise ValueError("No training slices selected for mechanism head")
        x_sel = x[mask]
        self.scaler = RobustScaler(quantile_range=(10.0, 90.0))
        x_sel_scaled = self.scaler.fit_transform(x_sel)
        n_selected = x_sel_scaled.shape[0]
        cluster_count = int(max(6, min(24, round(math.sqrt(max(64, n_selected)) / 2.5))))
        self.kmeans = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=self.random_state,
            batch_size=min(4096, max(512, n_selected)),
            n_init=10,
        )
        self.kmeans.fit(x_sel_scaled)
        self.iforest = IsolationForest(
            n_estimators=300,
            max_samples=min(4096, n_selected),
            contamination=0.03,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.iforest.fit(x_sel_scaled)

        residual = self._residual_raw(x_sel_scaled)
        if_raw = self._if_raw(x_sel_scaled)
        closure_raw = self._closure_inconsistency(x_sel)
        mechanism_raw = self._mechanism_load(x)
        self.residual_stat = _qstat(residual, q_low=0.20, q_high=0.95)
        self.if_stat = _qstat(if_raw, q_low=0.20, q_high=0.95)
        self.closure_stat = _qstat(closure_raw, q_low=0.20, q_high=0.95)
        self.mechanism_stat = _qstat(mechanism_raw, q_low=0.20, q_high=0.95)

        scored_rows = self.score_rows(feature_rows)
        slice_scores = np.asarray([float(row["anomaly_score"]) for row in scored_rows], dtype=float)
        self.slice_threshold = self._auto_threshold(slice_scores, self.slice_quantile) if slice_scores.size else 0.5
        self.train_summary = {
            "train_slice_count": int(len(feature_rows)),
            "selected_slice_count": int(self.selected_count),
            "selected_ratio": float(self.selected_count / max(1, len(feature_rows))),
            "cluster_count": int(cluster_count),
            "gate_floor": float(self.gate_floor),
            "slice_threshold": float(self.slice_threshold),
        }
        return self

    def _residual_raw(self, x_scaled: np.ndarray) -> np.ndarray:
        assert self.kmeans is not None
        dist = self.kmeans.transform(x_scaled)
        return np.min(dist, axis=1).astype(np.float32)

    def _if_raw(self, x_scaled: np.ndarray) -> np.ndarray:
        assert self.iforest is not None
        return (-self.iforest.score_samples(x_scaled)).astype(np.float32)

    def score_rows(self, feature_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.scaler is None or self.kmeans is None or self.iforest is None:
            raise ValueError("Mechanism head model not fitted")
        x = build_mechanism_head_feature_matrix(feature_rows)
        x_scaled = self.scaler.transform(x)

        residual_raw = self._residual_raw(x_scaled)
        if_raw = self._if_raw(x_scaled)
        closure_raw = self._closure_inconsistency(x)
        mechanism_raw = self._mechanism_load(x)

        residual_norm = _normalize_by_stat(residual_raw, self.residual_stat)
        if_norm = _normalize_by_stat(if_raw, self.if_stat)
        closure_norm = _normalize_by_stat(closure_raw, self.closure_stat)
        mechanism_norm = _normalize_by_stat(mechanism_raw, self.mechanism_stat)

        bridge_intent = self._col(x, "bridge_intent_score")
        helper_pollution = self._col(x, "helper_pollution_score")
        bridge_gate = np.clip((bridge_intent - self.gate_floor) / max(1e-6, 1.0 - self.gate_floor), 0.0, 1.0).astype(np.float32)
        helper_excess = np.maximum(0.0, helper_pollution - bridge_intent)
        helper_discount = np.clip(1.0 - 0.45 * helper_excess, 0.55, 1.0).astype(np.float32)

        vp_evidence = np.maximum(
            self._aux_col(feature_rows, "proto_score"),
            self._aux_col(feature_rows, "view_score"),
        )
        margin_sig = 1.0 / (1.0 + np.exp(-self._aux_col(feature_rows, "boundary_margin")))
        vp_w = float(min(1.0, max(0.0, self.corroboration_vp_weight)))
        margin_w = float(min(1.0, max(0.0, self.corroboration_margin_weight)))
        weight_sum = max(1e-6, vp_w + margin_w)
        corroboration = np.clip((vp_w * vp_evidence + margin_w * margin_sig) / weight_sum, 0.0, 1.0).astype(np.float32)
        corroboration_factor = (
            float(min(1.0, max(0.0, self.corroboration_alpha)))
            + (1.0 - float(min(1.0, max(0.0, self.corroboration_alpha)))) * corroboration
        )

        raw_score = 0.25 * residual_norm + 0.15 * if_norm + 0.20 * closure_norm + 0.40 * mechanism_norm
        scaled_score = (0.10 + 0.90 * bridge_gate) * helper_discount * corroboration_factor * raw_score
        anomaly_score = (1.0 - np.exp(-np.clip(scaled_score, 0.0, 6.0))).astype(np.float32)

        out: List[Dict[str, Any]] = []
        for idx, base_row in enumerate(feature_rows):
            row = dict(base_row)
            row.update(
                {
                    "bridge_gate": float(bridge_gate[idx]),
                    "residual_raw": float(residual_raw[idx]),
                    "residual_norm": float(residual_norm[idx]),
                    "if_raw": float(if_raw[idx]),
                    "if_norm": float(if_norm[idx]),
                    "closure_raw": float(closure_raw[idx]),
                    "closure_norm": float(closure_norm[idx]),
                    "mechanism_raw": float(mechanism_raw[idx]),
                    "mechanism_norm": float(mechanism_norm[idx]),
                    "helper_discount": float(helper_discount[idx]),
                    "vp_evidence": float(vp_evidence[idx]),
                    "margin_sig": float(margin_sig[idx]),
                    "corroboration": float(corroboration[idx]),
                    "corroboration_factor": float(corroboration_factor[idx]),
                    "anomaly_score": float(anomaly_score[idx]),
                    "slice_pred": int(float(anomaly_score[idx]) >= float(self.slice_threshold)),
                }
            )
            out.append(row)
        return out


def save_mechanism_head_model(path: str | Path, model: MechanismSliceHeadModel) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(model, f)


def load_mechanism_head_model(path: str | Path) -> MechanismSliceHeadModel:
    with Path(path).open("rb") as f:
        model = pickle.load(f)
    if not isinstance(model, MechanismSliceHeadModel):
        raise TypeError(f"Unexpected mechanism head model type: {type(model)!r}")
    if not getattr(model, "feature_index", None):
        model.feature_index = {name: idx for idx, name in enumerate(model.feature_names)}
    if not hasattr(model, "corroboration_alpha"):
        model.corroboration_alpha = 0.40
    if not hasattr(model, "corroboration_vp_weight"):
        model.corroboration_vp_weight = 0.80
    if not hasattr(model, "corroboration_margin_weight"):
        model.corroboration_margin_weight = 0.20
    return model
