from __future__ import annotations

import argparse
import collections
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.graph_utils import dedup_edges
from common.io_utils import load_jsonl, save_jsonl
from common.mechanism_text_view_utils import build_mechanism_text_view


KEY_REL = {
    "control_dep",
    "cfg_next",
    "data_dep",
    "dfg_dep",
    "call_interaction",
    "constraint_on",
    "implicit_mechanism",
    "contains",
    "ast_parent",
}
KEY_TYPES = {"call", "state_var", "condition", "data_object", "statement", "function"}
ROLE_SET = {"state_change", "external_interaction", "auth_constraint"}
ROLE_TRIPLETS: Sequence[Tuple[str, str]] = (
    ("auth_constraint", "external_interaction"),
    ("external_interaction", "state_change"),
    ("auth_constraint", "state_change"),
)
DYNAMIC_AUX_REL = {"contains", "ast_parent", "control_dep"}


def build_adj(graph: Dict[str, Any]) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = collections.defaultdict(set)
    for e in graph.get("edges", []):
        if e.get("type") not in KEY_REL:
            continue
        s = int(e["src"])
        d = int(e["dst"])
        adj[s].add(d)
        adj[d].add(s)
    return adj


def build_adj_and_edge_lookup(
    graph: Dict[str, Any],
) -> Tuple[Dict[int, Set[int]], Dict[Tuple[int, int], List[int]]]:
    adj: Dict[int, Set[int]] = collections.defaultdict(set)
    edge_lookup: Dict[Tuple[int, int], List[int]] = collections.defaultdict(list)
    for idx, e in enumerate(graph.get("edges", [])):
        if e.get("type") not in KEY_REL:
            continue
        s = int(e["src"])
        d = int(e["dst"])
        adj[s].add(d)
        adj[d].add(s)
        key = (s, d) if s <= d else (d, s)
        edge_lookup[key].append(idx)
    return adj, edge_lookup


def shortest_path_between_sets(
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

    q = collections.deque()
    parent: Dict[int, Optional[int]] = {}
    dist: Dict[int, int] = {}
    for s in src_set:
        q.append(s)
        parent[s] = None
        dist[s] = 0
    while q:
        cur = q.popleft()
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


def induced_subgraph(graph: Dict[str, Any], keep: Set[int]) -> Dict[str, Any]:
    id_map = {old: new for new, old in enumerate(sorted(keep))}
    nodes = []
    for old in sorted(keep):
        n = dict(graph["nodes"][old])
        n["id"] = id_map[old]
        nodes.append(n)
    edges = []
    for e in graph.get("edges", []):
        s = int(e["src"])
        d = int(e["dst"])
        if s in id_map and d in id_map:
            x = dict(e)
            x["src"] = id_map[s]
            x["dst"] = id_map[d]
            edges.append(x)
    return {"nodes": nodes, "edges": dedup_edges(edges)}


def subgraph_from_edge_ids(
    graph: Dict[str, Any],
    edge_ids: Set[int],
    extra_nodes: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    keep_nodes: Set[int] = set(extra_nodes or set())
    edges_src = graph.get("edges", [])
    for idx in edge_ids:
        if idx < 0 or idx >= len(edges_src):
            continue
        e = edges_src[idx]
        keep_nodes.add(int(e["src"]))
        keep_nodes.add(int(e["dst"]))
    if not keep_nodes:
        return {"nodes": [], "edges": []}
    id_map = {old: new for new, old in enumerate(sorted(keep_nodes))}
    nodes = []
    for old in sorted(keep_nodes):
        n = dict(graph["nodes"][old])
        n["id"] = id_map[old]
        nodes.append(n)
    edges = []
    for idx in sorted(edge_ids):
        if idx < 0 or idx >= len(edges_src):
            continue
        e = dict(edges_src[idx])
        s = int(e["src"])
        d = int(e["dst"])
        if s not in id_map or d not in id_map:
            continue
        e["src"] = id_map[s]
        e["dst"] = id_map[d]
        edges.append(e)
    return {"nodes": nodes, "edges": dedup_edges(edges)}


def filter_full_view_edges(graph: Dict[str, Any]) -> Dict[str, Any]:
    # Implicit mechanism edges are reserved for the skeleton branch in the
    # corresponding experiment so the full semantic branch stays close to the
    # physical code graph.
    edges = [dict(e) for e in graph.get("edges", []) if e.get("type") != "implicit_mechanism"]
    return {"nodes": list(graph.get("nodes", [])), "edges": dedup_edges(edges)}


def build_skeleton_view_legacy(slice_graph: Dict[str, Any]) -> Dict[str, Any]:
    nodes = slice_graph.get("nodes", [])
    adj = build_adj(slice_graph)
    role_nodes = {
        int(n["id"])
        for n in nodes
        if ROLE_SET.intersection(set(n.get("roles", []) or []))
    }
    keep: Set[int] = set(role_nodes)
    # 1-hop structural connectors around role nodes.
    for r in role_nodes:
        for nb in adj.get(r, set()):
            ntype = nodes[nb].get("type")
            if ntype in KEY_TYPES:
                keep.add(nb)
    # Keep parent function nodes if any contains edge exists.
    for e in slice_graph.get("edges", []):
        if e.get("type") == "contains":
            s = int(e["src"])
            d = int(e["dst"])
            if d in keep or s in keep:
                keep.add(s)
                keep.add(d)
    if not keep:
        keep = {int(n["id"]) for n in nodes[: min(8, len(nodes))]}
    return induced_subgraph(slice_graph, keep)


def role_map(nodes: List[Dict[str, Any]]) -> Dict[str, Set[int]]:
    out = {r: set() for r in ROLE_SET}
    for n in nodes:
        nid = int(n["id"])
        for r in n.get("roles", []) or []:
            if r in out:
                out[r].add(nid)
    return out


def build_skeleton_view_formal(
    slice_graph: Dict[str, Any],
    mechanism_path_max_len: int = 4,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    nodes = slice_graph.get("nodes", [])
    adj = build_adj(slice_graph)
    node_by_id = {int(n["id"]): n for n in nodes}
    rmap = role_map(nodes)
    keep: Set[int] = set().union(*rmap.values()) if rmap else set()
    mechanism_links = 0

    # Keep shortest semantic paths between key role pairs to preserve mechanism closure.
    for a, b in ROLE_TRIPLETS:
        path = shortest_path_between_sets(adj, rmap.get(a, set()), rmap.get(b, set()), max_hops=mechanism_path_max_len)
        if not path:
            continue
        mechanism_links += 1
        keep.update(path)

    # Pull one-hop connectors around kept nodes.
    for nid in list(keep):
        for nb in adj.get(nid, set()):
            nb_node = node_by_id.get(nb, {})
            if nb_node.get("type") in KEY_TYPES:
                keep.add(nb)

    # Ensure function context nodes are preserved.
    for e in slice_graph.get("edges", []):
        if e.get("type") not in {"contains", "ast_parent"}:
            continue
        s = int(e["src"])
        d = int(e["dst"])
        if s in keep or d in keep:
            keep.add(s)
            keep.add(d)

    if not keep:
        keep = {int(n["id"]) for n in nodes[: min(8, len(nodes))]}

    meta = {
        "mechanism_links": int(mechanism_links),
        "has_auth": bool(rmap.get("auth_constraint")),
        "has_external": bool(rmap.get("external_interaction")),
        "has_state": bool(rmap.get("state_change")),
        "mechanism_complete": bool(
            rmap.get("auth_constraint") and rmap.get("external_interaction") and rmap.get("state_change")
        ),
    }
    return induced_subgraph(slice_graph, keep), meta


def _stable_keep_edge(
    slice_id: str,
    edge: Dict[str, Any],
    seed: int,
    keep_prob: float,
) -> bool:
    if keep_prob >= 1.0:
        return True
    if keep_prob <= 0.0:
        return False
    payload = f"{slice_id}|{seed}|{edge.get('type','')}|{edge.get('src')}|{edge.get('dst')}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF
    return score <= keep_prob


def build_skeleton_view_dynamic(
    slice_graph: Dict[str, Any],
    slice_id: str,
    mechanism_path_max_len: int = 4,
    aux_keep_prob: float = 0.35,
    dynamic_seed: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    nodes = slice_graph.get("nodes", [])
    node_by_id = {int(n["id"]): n for n in nodes}
    adj, edge_lookup = build_adj_and_edge_lookup(slice_graph)
    rmap = role_map(nodes)

    core_nodes: Set[int] = set().union(*rmap.values()) if rmap else set()
    core_edge_ids: Set[int] = set()
    mechanism_links = 0

    for a, b in ROLE_TRIPLETS:
        path = shortest_path_between_sets(adj, rmap.get(a, set()), rmap.get(b, set()), max_hops=mechanism_path_max_len)
        if not path:
            continue
        mechanism_links += 1
        core_nodes.update(path)
        for u, v in zip(path, path[1:]):
            key = (u, v) if u <= v else (v, u)
            core_edge_ids.update(edge_lookup.get(key, []))

    expanded_nodes: Set[int] = set(core_nodes)
    for nid in list(core_nodes):
        for nb in adj.get(nid, set()):
            nb_node = node_by_id.get(nb, {})
            if nb_node.get("type") in KEY_TYPES:
                expanded_nodes.add(nb)

    edges_src = slice_graph.get("edges", [])
    for idx, e in enumerate(edges_src):
        etype = e.get("type")
        s = int(e["src"])
        d = int(e["dst"])
        if etype in {"contains", "ast_parent"} and (s in core_nodes or d in core_nodes):
            core_edge_ids.add(idx)
            expanded_nodes.add(s)
            expanded_nodes.add(d)
        elif etype in KEY_REL and s in core_nodes and d in core_nodes:
            core_edge_ids.add(idx)

    aux_candidates: List[int] = []
    selected_aux: Set[int] = set()
    for idx, e in enumerate(edges_src):
        if idx in core_edge_ids:
            continue
        etype = e.get("type")
        if etype not in DYNAMIC_AUX_REL:
            continue
        s = int(e["src"])
        d = int(e["dst"])
        if s not in expanded_nodes and d not in expanded_nodes:
            continue
        aux_candidates.append(idx)
        if _stable_keep_edge(slice_id, e, dynamic_seed, aux_keep_prob):
            selected_aux.add(idx)

    selected_edge_ids = set(core_edge_ids).union(selected_aux)
    keep_nodes = set(core_nodes)
    for idx in core_edge_ids:
        e = edges_src[idx]
        keep_nodes.add(int(e["src"]))
        keep_nodes.add(int(e["dst"]))
    for idx in selected_aux:
        e = edges_src[idx]
        keep_nodes.add(int(e["src"]))
        keep_nodes.add(int(e["dst"]))

    if not keep_nodes:
        keep_nodes = {int(n["id"]) for n in nodes[: min(8, len(nodes))]}

    meta = {
        "mechanism_links": int(mechanism_links),
        "has_auth": bool(rmap.get("auth_constraint")),
        "has_external": bool(rmap.get("external_interaction")),
        "has_state": bool(rmap.get("state_change")),
        "mechanism_complete": bool(
            rmap.get("auth_constraint") and rmap.get("external_interaction") and rmap.get("state_change")
        ),
        "dynamic_aux_keep_prob": float(aux_keep_prob),
        "dynamic_aux_candidates": int(len(aux_candidates)),
        "dynamic_aux_selected": int(len(selected_aux)),
        "dynamic_core_edges": int(len(core_edge_ids)),
    }
    return subgraph_from_edge_ids(slice_graph, selected_edge_ids, extra_nodes=keep_nodes), meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-3: build dual views from role-aware slices.")
    parser.add_argument("--input", type=str, default="data/slices/slices.jsonl")
    parser.add_argument("--output", type=str, default="data/slices/dual_views.jsonl")
    parser.add_argument("--max-slices", type=int, default=0)
    parser.add_argument("--skeleton-mode", choices=["legacy", "formal", "dynamic"], default="legacy")
    parser.add_argument("--mechanism-path-max-len", type=int, default=4)
    parser.add_argument("--dynamic-aux-keep-prob", type=float, default=0.35)
    parser.add_argument("--dynamic-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = (ROOT / args.input).resolve()
    out_path = (ROOT / args.output).resolve()
    rows = load_jsonl(in_path)
    if args.max_slices and args.max_slices > 0:
        rows = rows[: args.max_slices]

    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        base_graph = row["graph"]
        full_graph = filter_full_view_edges(base_graph)
        skeleton_meta: Dict[str, Any] = {}
        mechanism_text, mechanism_text_meta = build_mechanism_text_view(
            base_graph,
            function_names=row.get("function_names", []),
            seed_roles=row.get("seed_roles", []),
            mechanism_path_max_len=args.mechanism_path_max_len,
        )
        if args.skeleton_mode == "formal":
            skel_graph, skeleton_meta = build_skeleton_view_formal(
                base_graph,
                mechanism_path_max_len=args.mechanism_path_max_len,
            )
        elif args.skeleton_mode == "dynamic":
            skel_graph, skeleton_meta = build_skeleton_view_dynamic(
                base_graph,
                slice_id=str(row.get("slice_id", "")),
                mechanism_path_max_len=args.mechanism_path_max_len,
                aux_keep_prob=args.dynamic_aux_keep_prob,
                dynamic_seed=args.dynamic_seed,
            )
        else:
            skel_graph = build_skeleton_view_legacy(base_graph)
        out_rows.append(
            {
                "slice_id": row["slice_id"],
                "contract_id": row["contract_id"],
                "source_path": row.get("source_path"),
                "relative_source_path": row.get("relative_source_path"),
                "seed_roles": row.get("seed_roles", []),
                "function_names": row.get("function_names", []),
                "full_graph": {
                    "nodes": full_graph.get("nodes", []),
                    "edges": full_graph.get("edges", []),
                    "meta": {
                        "view_type": "full_semantic",
                        "node_count": len(full_graph.get("nodes", [])),
                        "edge_count": len(full_graph.get("edges", [])),
                    },
                },
                "skeleton_graph": {
                    "nodes": skel_graph.get("nodes", []),
                    "edges": skel_graph.get("edges", []),
                    "meta": {
                        "view_type": "security_skeleton",
                        "skeleton_mode": args.skeleton_mode,
                        "node_count": len(skel_graph.get("nodes", [])),
                        "edge_count": len(skel_graph.get("edges", [])),
                        **skeleton_meta,
                    },
                },
                "mechanism_text": mechanism_text,
                "mechanism_text_meta": mechanism_text_meta,
            }
        )
    save_jsonl(out_path, out_rows)
    print(f"[dual_view_builder] views={len(out_rows)} output={out_path}")


if __name__ == "__main__":
    main()
