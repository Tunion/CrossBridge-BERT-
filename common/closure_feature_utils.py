from __future__ import annotations

import collections
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


ROLE_KEYS: Tuple[str, ...] = (
    "auth_constraint",
    "external_interaction",
    "state_change",
)

ROLE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("auth_constraint", "external_interaction"),
    ("external_interaction", "state_change"),
    ("auth_constraint", "state_change"),
)

MECHANISM_RELATIONS: Set[str] = {
    "constraint_on",
    "call_interaction",
    "dfg_dep",
    "data_dep",
    "cfg_next",
    "control_dep",
    "contains",
    "ast_parent",
    "implicit_mechanism",
}

CLOSURE_FEATURE_NAMES: Tuple[str, ...] = (
    "closure_full_role_count",
    "closure_full_role_ratio",
    "closure_full_pair_count",
    "closure_full_pair_ratio",
    "closure_full_path_mean",
    "closure_full_role_focus",
    "closure_skeleton_role_retain",
    "closure_skeleton_pair_retain",
    "closure_skeleton_link_cov",
    "closure_mechanism_complete",
)


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    if float(den) <= 0.0:
        return float(default)
    return float(num) / float(den)


def role_node_sets(graph: Dict[str, Any]) -> Dict[str, Set[int]]:
    out: Dict[str, Set[int]] = {k: set() for k in ROLE_KEYS}
    for node in graph.get("nodes", []):
        nid = int(node.get("id", -1))
        if nid < 0:
            continue
        for role in node.get("roles", []) or []:
            if role in out:
                out[role].add(nid)
    return out


def build_undirected_adj(
    graph: Dict[str, Any],
    rel_types: Optional[Iterable[str]] = None,
) -> Dict[int, Set[int]]:
    keep = set(rel_types) if rel_types is not None else set(MECHANISM_RELATIONS)
    adj: Dict[int, Set[int]] = collections.defaultdict(set)
    for edge in graph.get("edges", []):
        if str(edge.get("type", "")) not in keep:
            continue
        src = int(edge.get("src", -1))
        dst = int(edge.get("dst", -1))
        if src < 0 or dst < 0:
            continue
        adj[src].add(dst)
        adj[dst].add(src)
    return adj


def shortest_path_len_between_sets(
    adj: Dict[int, Set[int]],
    src_set: Set[int],
    dst_set: Set[int],
    max_hops: int = 6,
) -> Optional[int]:
    if not src_set or not dst_set:
        return None
    if src_set.intersection(dst_set):
        return 0
    q: collections.deque[Tuple[int, int]] = collections.deque((nid, 0) for nid in src_set)
    seen: Set[int] = set(src_set)
    hop_cap = max(1, int(max_hops))
    while q:
        cur, dep = q.popleft()
        if dep >= hop_cap:
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


def compute_graph_closure_features(
    full_graph: Dict[str, Any],
    skeleton_graph: Optional[Dict[str, Any]] = None,
    max_hops: int = 6,
) -> Dict[str, float]:
    full = full_graph or {}
    skel = skeleton_graph or {}
    full_roles = role_node_sets(full)
    skel_roles = role_node_sets(skel)
    full_role_count = int(sum(1 for key in ROLE_KEYS if full_roles.get(key)))
    full_role_ratio = _safe_ratio(float(full_role_count), float(len(ROLE_KEYS)), default=0.0)

    full_adj = build_undirected_adj(full, rel_types=MECHANISM_RELATIONS)
    expected_pairs = 0
    connected_pairs = 0
    path_terms: List[float] = []
    hop_cap = max(1, int(max_hops))
    for left, right in ROLE_PAIRS:
        src = full_roles.get(left, set())
        dst = full_roles.get(right, set())
        if not src or not dst:
            continue
        expected_pairs += 1
        plen = shortest_path_len_between_sets(full_adj, src, dst, max_hops=hop_cap)
        if plen is None:
            continue
        connected_pairs += 1
        path_terms.append(min(1.0, float(plen) / float(hop_cap)))
    full_pair_ratio = _safe_ratio(float(connected_pairs), float(expected_pairs), default=0.0)
    full_path_mean = float(sum(path_terms) / len(path_terms)) if path_terms else 0.0

    full_role_nodes = float(sum(len(v) for v in full_roles.values()))
    full_nodes = float(max(1, len(full.get("nodes", []))))
    full_role_focus = min(1.0, _safe_ratio(full_role_nodes, full_nodes, default=0.0))

    role_retains: List[float] = []
    for key in ROLE_KEYS:
        full_cnt = len(full_roles.get(key, set()))
        if full_cnt <= 0:
            continue
        skel_cnt = len(skel_roles.get(key, set()))
        role_retains.append(min(1.0, _safe_ratio(float(skel_cnt), float(full_cnt), default=0.0)))
    skeleton_role_retain = float(sum(role_retains) / len(role_retains)) if role_retains else 0.0

    skel_adj = build_undirected_adj(skel, rel_types=MECHANISM_RELATIONS)
    skel_connected_pairs = 0
    for left, right in ROLE_PAIRS:
        src = skel_roles.get(left, set())
        dst = skel_roles.get(right, set())
        if not src or not dst:
            continue
        plen = shortest_path_len_between_sets(skel_adj, src, dst, max_hops=hop_cap)
        if plen is not None:
            skel_connected_pairs += 1
    skeleton_pair_retain = _safe_ratio(float(skel_connected_pairs), float(expected_pairs), default=0.0)

    skel_meta = (skel.get("meta", {}) or {}) if isinstance(skel, dict) else {}
    if isinstance(skel_meta.get("mechanism_links"), (int, float)):
        got_links = float(max(0.0, float(skel_meta.get("mechanism_links", 0.0))))
    else:
        got_links = float(skel_connected_pairs)
    skeleton_link_cov = min(1.0, _safe_ratio(got_links, float(max(1, expected_pairs)), default=0.0))
    mechanism_complete = 1.0 if bool(skel_meta.get("mechanism_complete", False)) else 0.0
    if mechanism_complete <= 0.0 and full_role_count == len(ROLE_KEYS) and connected_pairs >= 2:
        mechanism_complete = 1.0

    return {
        "closure_full_role_count": float(full_role_count),
        "closure_full_role_ratio": float(full_role_ratio),
        "closure_full_pair_count": float(connected_pairs),
        "closure_full_pair_ratio": float(full_pair_ratio),
        "closure_full_path_mean": float(full_path_mean),
        "closure_full_role_focus": float(full_role_focus),
        "closure_skeleton_role_retain": float(skeleton_role_retain),
        "closure_skeleton_pair_retain": float(skeleton_pair_retain),
        "closure_skeleton_link_cov": float(skeleton_link_cov),
        "closure_mechanism_complete": float(mechanism_complete),
    }


def closure_feature_defaults() -> Dict[str, float]:
    return {name: 0.0 for name in CLOSURE_FEATURE_NAMES}


def aggregate_closure_features(
    rows: Sequence[Dict[str, Any]],
    reduce: str = "max",
) -> Dict[str, float]:
    if not rows:
        return closure_feature_defaults()
    mode = str(reduce or "max").strip().lower()
    values: Dict[str, List[float]] = {name: [] for name in CLOSURE_FEATURE_NAMES}
    for row in rows:
        for name in CLOSURE_FEATURE_NAMES:
            try:
                values[name].append(float(row.get(name, 0.0)))
            except Exception:
                values[name].append(0.0)
    out: Dict[str, float] = {}
    for name in CLOSURE_FEATURE_NAMES:
        arr = values.get(name, [])
        if not arr:
            out[name] = 0.0
        elif mode == "mean":
            out[name] = float(sum(arr) / len(arr))
        else:
            out[name] = float(max(arr))
    return out
