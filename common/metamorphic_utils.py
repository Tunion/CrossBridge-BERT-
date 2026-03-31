from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from common.graph_utils import dedup_edges


ROLE_SET = {"state_change", "external_interaction", "auth_constraint"}
ROLE_TRIPLETS: Sequence[Tuple[str, str]] = (
    ("auth_constraint", "external_interaction"),
    ("external_interaction", "state_change"),
    ("auth_constraint", "state_change"),
)
MECHANISM_EDGE_TYPES = {
    "control_dep",
    "cfg_next",
    "data_dep",
    "dfg_dep",
    "call_interaction",
    "constraint_on",
    "contains",
    "ast_parent",
}
SYNTAX_EDGE_TYPES = {"contains", "ast_parent"}
CONTEXT_EDGE_TYPES = {"control_dep", "constraint_on"}
LEAF_NODE_TYPES = {"data_object", "condition", "statement"}


def _build_adj(graph: Dict[str, Any]) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = {}
    for e in graph.get("edges", []):
        if e.get("type") not in MECHANISM_EDGE_TYPES:
            continue
        s = int(e["src"])
        d = int(e["dst"])
        adj.setdefault(s, set()).add(d)
        adj.setdefault(d, set()).add(s)
    return adj


def _role_map(graph: Dict[str, Any]) -> Dict[str, Set[int]]:
    out = {r: set() for r in ROLE_SET}
    for node in graph.get("nodes", []):
        nid = int(node["id"])
        for role in node.get("roles", []) or []:
            if role in out:
                out[role].add(nid)
    return out


def _shortest_path_between_sets(
    adj: Dict[int, Set[int]],
    src_set: Set[int],
    dst_set: Set[int],
    max_hops: int,
) -> Optional[List[int]]:
    if not src_set or not dst_set:
        return None
    inter = src_set.intersection(dst_set)
    if inter:
        return [next(iter(inter))]
    q: List[int] = list(src_set)
    parent: Dict[int, Optional[int]] = {s: None for s in src_set}
    dist: Dict[int, int] = {s: 0 for s in src_set}
    head = 0
    while head < len(q):
        cur = q[head]
        head += 1
        d = dist[cur]
        if d >= max_hops:
            continue
        for nb in adj.get(cur, set()):
            if nb in dist:
                continue
            dist[nb] = d + 1
            parent[nb] = cur
            if nb in dst_set:
                path = [nb]
                p = cur
                while p is not None:
                    path.append(p)
                    p = parent[p]
                path.reverse()
                return path
            q.append(nb)
    return None


def protected_nodes(graph: Dict[str, Any], max_hops: int = 4) -> Set[int]:
    rmap = _role_map(graph)
    keep: Set[int] = set().union(*rmap.values()) if rmap else set()
    adj = _build_adj(graph)
    for a, b in ROLE_TRIPLETS:
        path = _shortest_path_between_sets(adj, rmap.get(a, set()), rmap.get(b, set()), max_hops=max_hops)
        if path:
            keep.update(path)
    for e in graph.get("edges", []):
        if e.get("type") not in {"contains", "ast_parent"}:
            continue
        s = int(e["src"])
        d = int(e["dst"])
        if s in keep or d in keep:
            keep.add(s)
            keep.add(d)
    return keep


def _reindex_graph(graph: Dict[str, Any], keep_nodes: Set[int], kept_edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(keep_nodes)
    id_map = {old: new for new, old in enumerate(ordered)}
    nodes: List[Dict[str, Any]] = []
    for old in ordered:
        n = dict(graph["nodes"][old])
        n["id"] = id_map[old]
        nodes.append(n)
    edges: List[Dict[str, Any]] = []
    for e in kept_edges:
        s = int(e["src"])
        d = int(e["dst"])
        if s not in id_map or d not in id_map:
            continue
        x = dict(e)
        x["src"] = id_map[s]
        x["dst"] = id_map[d]
        edges.append(x)
    return {"nodes": nodes, "edges": dedup_edges(edges), "meta": dict(graph.get("meta", {}))}


def syntax_pruned_variant(graph: Dict[str, Any], max_hops: int = 4) -> Dict[str, Any]:
    protected = protected_nodes(graph, max_hops=max_hops)
    kept_edges: List[Dict[str, Any]] = []
    for e in graph.get("edges", []):
        s = int(e["src"])
        d = int(e["dst"])
        et = e.get("type")
        if et in SYNTAX_EDGE_TYPES and s not in protected and d not in protected:
            continue
        kept_edges.append(e)
    keep_nodes = {int(n["id"]) for n in graph.get("nodes", [])}
    out = _reindex_graph(graph, keep_nodes, kept_edges)
    out["meta"] = {**out.get("meta", {}), "metamorphic_variant": "syntax_pruned"}
    return out


def leaf_context_pruned_variant(graph: Dict[str, Any], max_hops: int = 4) -> Dict[str, Any]:
    base = syntax_pruned_variant(graph, max_hops=max_hops)
    protected = protected_nodes(base, max_hops=max_hops)
    deg: Dict[int, int] = {}
    for e in base.get("edges", []):
        s = int(e["src"])
        d = int(e["dst"])
        deg[s] = deg.get(s, 0) + 1
        deg[d] = deg.get(d, 0) + 1
    keep_nodes: Set[int] = set()
    for n in base.get("nodes", []):
        nid = int(n["id"])
        if nid in protected:
            keep_nodes.add(nid)
            continue
        ntype = str(n.get("type", ""))
        if ntype in LEAF_NODE_TYPES and deg.get(nid, 0) <= 1:
            continue
        keep_nodes.add(nid)
    kept_edges: List[Dict[str, Any]] = []
    for e in base.get("edges", []):
        s = int(e["src"])
        d = int(e["dst"])
        if s in keep_nodes and d in keep_nodes:
            if e.get("type") in CONTEXT_EDGE_TYPES and s not in protected and d not in protected and ((s + d) % 2 == 0):
                continue
            kept_edges.append(e)
    out = _reindex_graph(base, keep_nodes, kept_edges)
    out["meta"] = {**out.get("meta", {}), "metamorphic_variant": "leaf_context_pruned"}
    return out


def build_dual_metamorphic_variants(
    full_graph: Dict[str, Any],
    skeleton_graph: Dict[str, Any],
    max_hops: int = 4,
) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    return [
        (
            "syntax_pruned",
            syntax_pruned_variant(full_graph, max_hops=max_hops),
            syntax_pruned_variant(skeleton_graph, max_hops=max_hops),
        ),
        (
            "leaf_context_pruned",
            leaf_context_pruned_variant(full_graph, max_hops=max_hops),
            leaf_context_pruned_variant(skeleton_graph, max_hops=max_hops),
        ),
    ]
