from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


NODE_TYPES: Tuple[str, ...] = (
    "function",
    "statement",
    "state_var",
    "call",
    "condition",
    "data_object",
)

EDGE_TYPES: Tuple[str, ...] = (
    "contains",
    "ast_parent",
    "control_dep",
    "cfg_next",
    "data_dep",
    "dfg_dep",
    "call_interaction",
    "constraint_on",
    "implicit_mechanism",
)

ROLE_TYPES: Tuple[str, ...] = (
    "state_change",
    "external_interaction",
    "auth_constraint",
)


def build_soft_edge_weight_config(
    enable: bool = False,
    local_call_summary: float = 0.35,
    shared_object_summary: float = 0.45,
    cross_function_state_flow: float = 0.55,
    local_call: float = 0.75,
    state_summary: float = 0.85,
) -> Dict[str, Any]:
    return {
        "enable": bool(enable),
        "default_weight": 1.0,
        "kind_weights": {
            "local_call_summary": float(local_call_summary),
            "shared_object_summary": float(shared_object_summary),
            "cross_function_state_flow": float(cross_function_state_flow),
            "local_call": float(local_call),
            "state_write_summary": float(state_summary),
            "state_read_summary": float(state_summary),
        },
        "type_weights": {},
    }


def make_contract_id(source_path: str) -> str:
    return hashlib.md5(source_path.encode("utf-8")).hexdigest()


def init_graph(contract_id: str, source_path: str, file_name: str) -> Dict[str, Any]:
    return {
        "contract_id": contract_id,
        "source_path": source_path,
        "file_name": file_name,
        "nodes": [],
        "edges": [],
        "meta": {},
    }


def add_node(graph: Dict[str, Any], node_type: str, attrs: Dict[str, Any]) -> int:
    node_id = len(graph["nodes"])
    node = {"id": node_id, "type": node_type}
    node.update(attrs)
    graph["nodes"].append(node)
    return node_id


def add_edge(graph: Dict[str, Any], src: int, dst: int, rel_type: str, attrs: Dict[str, Any] | None = None) -> None:
    edge = {"src": src, "dst": dst, "type": rel_type}
    if attrs:
        edge.update(attrs)
    graph["edges"].append(edge)


def node_type_index(node_type: str) -> int:
    return NODE_TYPES.index(node_type) if node_type in NODE_TYPES else len(NODE_TYPES)


def edge_type_index(edge_type: str, edge_types: Optional[Sequence[str]] = None) -> int:
    et = tuple(edge_types) if edge_types is not None else EDGE_TYPES
    return et.index(edge_type) if edge_type in et else len(et)


def role_vector(node: Dict[str, Any]) -> np.ndarray:
    vec = np.zeros(len(ROLE_TYPES), dtype=np.float32)
    roles = node.get("roles", []) or []
    for r in roles:
        if r in ROLE_TYPES:
            vec[ROLE_TYPES.index(r)] = 1.0
    return vec


def _row_normalize(a: np.ndarray) -> np.ndarray:
    deg = np.maximum(a.sum(axis=1, keepdims=True), 1.0)
    return a / deg


def resolve_edge_soft_weight(
    edge: Dict[str, Any],
    edge_weight_config: Optional[Dict[str, Any]] = None,
) -> float:
    if not edge_weight_config or not bool(edge_weight_config.get("enable", False)):
        return 1.0
    kind = str(edge.get("kind", "") or "").strip().lower()
    etype = str(edge.get("type", "") or "").strip().lower()
    kind_weights = {
        str(k).strip().lower(): float(v)
        for k, v in dict(edge_weight_config.get("kind_weights", {}) or {}).items()
    }
    type_weights = {
        str(k).strip().lower(): float(v)
        for k, v in dict(edge_weight_config.get("type_weights", {}) or {}).items()
    }
    default_weight = float(edge_weight_config.get("default_weight", 1.0))
    if kind and kind in kind_weights:
        return float(max(0.0, kind_weights[kind]))
    if etype and etype in type_weights:
        return float(max(0.0, type_weights[etype]))
    return float(max(0.0, default_weight))


def graph_to_tensors(
    graph: Dict[str, Any],
    edge_types: Optional[Sequence[str]] = None,
    edge_weight_config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_types_seq = tuple(edge_types) if edge_types is not None else EDGE_TYPES
    n = len(graph["nodes"])
    x = np.zeros((max(n, 1), len(NODE_TYPES) + 3 + len(ROLE_TYPES)), dtype=np.float32)
    if n == 0:
        a = np.eye(1, dtype=np.float32)
        rel_a = np.zeros((len(edge_types_seq), 1, 1), dtype=np.float32)
        return x, a, rel_a

    for node in graph["nodes"]:
        i = int(node["id"])
        t_idx = node_type_index(str(node.get("type", "")))
        if t_idx < len(NODE_TYPES):
            x[i, t_idx] = 1.0
        semantic = str(node.get("semantic_class", ""))
        # 3 semantic indicators: constraint / external / state
        if "constraint" in semantic:
            x[i, len(NODE_TYPES) + 0] = 1.0
        if "external" in semantic:
            x[i, len(NODE_TYPES) + 1] = 1.0
        if "state" in semantic:
            x[i, len(NODE_TYPES) + 2] = 1.0
        rv = role_vector(node)
        x[i, len(NODE_TYPES) + 3 :] = rv

    a = np.zeros((n, n), dtype=np.float32)
    rel_a = np.zeros((len(edge_types_seq), n, n), dtype=np.float32)
    for e in graph["edges"]:
        s = int(e["src"])
        d = int(e["dst"])
        if 0 <= s < n and 0 <= d < n:
            w = float(resolve_edge_soft_weight(e, edge_weight_config=edge_weight_config))
            if w <= 0.0:
                continue
            a[s, d] += w
            a[d, s] += w
            et = str(e.get("type", ""))
            et_idx = edge_type_index(et, edge_types_seq)
            if et_idx < len(edge_types_seq):
                rel_a[et_idx, s, d] += w
                rel_a[et_idx, d, s] += w
    a += np.eye(n, dtype=np.float32)
    a = _row_normalize(a)
    for i in range(len(edge_types_seq)):
        rel_a[i] = _row_normalize(rel_a[i])
    return x, a, rel_a


def graph_stats_vector(graph: Dict[str, Any], edge_types: Optional[Sequence[str]] = None) -> np.ndarray:
    edge_types_seq = tuple(edge_types) if edge_types is not None else EDGE_TYPES
    node_counter = Counter(n.get("type", "unknown") for n in graph["nodes"])
    edge_counter = Counter(e.get("type", "unknown") for e in graph["edges"])
    vec: List[float] = []
    for t in NODE_TYPES:
        vec.append(float(node_counter.get(t, 0)))
    for t in edge_types_seq:
        vec.append(float(edge_counter.get(t, 0)))
    # Extra structural statistics
    n_nodes = len(graph["nodes"])
    n_edges = len(graph["edges"])
    vec.extend(
        [
            float(n_nodes),
            float(n_edges),
            float(n_edges / n_nodes) if n_nodes else 0.0,
            float(sum(len(n.get("roles", []) or []) for n in graph["nodes"])),
        ]
    )
    return np.asarray(vec, dtype=np.float32)


def dedup_edges(edges: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for e in edges:
        key = (e["src"], e["dst"], e["type"])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


@dataclass
class SliceRecord:
    slice_id: str
    contract_id: str
    source_path: str
    graph: Dict[str, Any]
    seed_node_ids: Sequence[int]
    seed_roles: Sequence[str]
