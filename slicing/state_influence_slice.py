from __future__ import annotations

import argparse
import collections
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.graph_utils import ROLE_TYPES, dedup_edges
from common.io_utils import load_json, save_json, save_jsonl


# Keep implicit mechanism edges out of propagation by default.
# They remain available inside induced subgraphs when both endpoints are already
# selected, so downstream skeleton views can still consume them without letting
# them dominate slice expansion or the full view.
DEFAULT_RELATIONS = {"control_dep", "cfg_next", "data_dep", "dfg_dep", "call_interaction", "constraint_on"}

# Typed interprocedural filtering keeps bridge-mechanism context while dropping
# generic helper/math spillover. Default mode remains "all".
MECHANISM_FUNCTION_HINTS: Sequence[str] = (
    "cross",
    "bridge",
    "chain",
    "passport",
    "proposal",
    "relayer",
    "validator",
    "resource",
    "handler",
    "header",
    "verify",
    "merkle",
    "proof",
    "signature",
    "signer",
    "bookkeeper",
    "keeper",
    "checkpoint",
    "quorum",
    "root",
    "nonce",
    "message",
    "packet",
    "execute",
    "cancel",
    "vote",
    "genesis",
    "mint",
    "burn",
    "lock",
    "unlock",
    "claim",
    "release",
    "redeem",
)
MECHANISM_VAR_HINTS: Sequence[str] = (
    "nonce",
    "proposal",
    "proof",
    "signature",
    "sig",
    "merkle",
    "header",
    "hash",
    "root",
    "validator",
    "relayer",
    "bookkeeper",
    "keeper",
    "resource",
    "handler",
    "bridge",
    "chain",
    "message",
    "packet",
    "passport",
    "record",
    "checkpoint",
    "genesis",
)
MECHANISM_SHARED_OBJECT_KEYS: Set[str] = {"payload", "proof", "signature", "nonce", "chainid", "bridge"}

ROLE_TRIPLETS: Sequence[Tuple[str, str]] = (
    ("auth_constraint", "external_interaction"),
    ("external_interaction", "state_change"),
    ("auth_constraint", "state_change"),
)


def resolve_function_owner(
    node_id: int,
    node_by_id: Dict[int, Dict[str, Any]],
    contains_parent: Dict[int, int],
) -> Optional[int]:
    node = node_by_id.get(node_id, {})
    if node.get("type") == "function":
        return int(node_id)
    parent_fn = contains_parent.get(int(node_id))
    if parent_fn is None:
        return None
    return int(parent_fn)


def has_any_hint(text: Any, hints: Sequence[str]) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False
    return any(h in low for h in hints)


def is_mechanism_like_name(name: Any) -> bool:
    return has_any_hint(name, MECHANISM_FUNCTION_HINTS)


def is_mechanism_like_var(name: Any) -> bool:
    return has_any_hint(name, MECHANISM_VAR_HINTS)


def build_function_profiles(
    node_by_id: Dict[int, Dict[str, Any]],
    contains_children: Dict[int, Set[int]],
) -> Dict[int, Dict[str, Any]]:
    profiles: Dict[int, Dict[str, Any]] = {}
    for fn_id, node in node_by_id.items():
        if node.get("type") != "function":
            continue
        role_types: Set[str] = set()
        role_node_count = 0
        stmt_hint_hits = 0
        for child_id in contains_children.get(int(fn_id), set()):
            child = node_by_id.get(int(child_id), {})
            child_roles = {str(x) for x in (child.get("roles", []) or []) if isinstance(x, str)}
            if child_roles:
                role_types.update(child_roles)
                role_node_count += 1
            if child.get("type") == "statement" and has_any_hint(child.get("text"), MECHANISM_FUNCTION_HINTS):
                stmt_hint_hits += 1
        name_hint = is_mechanism_like_name(node.get("name"))
        modifier_hint = any(is_mechanism_like_name(x) for x in (node.get("modifiers", []) or []))
        hint_score = int(name_hint) + int(modifier_hint) + min(stmt_hint_hits, 2)
        profiles[int(fn_id)] = {
            "name": str(node.get("name") or ""),
            "role_types": role_types,
            "role_node_count": int(role_node_count),
            "stmt_hint_hits": int(stmt_hint_hits),
            "hint_score": int(hint_score),
            "mechanism_like": bool(hint_score > 0),
        }
    return profiles


def function_is_mechanism_like(function_id: Optional[int], function_profiles: Dict[int, Dict[str, Any]]) -> bool:
    if function_id is None:
        return False
    return bool(function_profiles.get(int(function_id), {}).get("mechanism_like", False))


def should_keep_typed_interproc_edge(
    edge: Dict[str, Any],
    node_by_id: Dict[int, Dict[str, Any]],
    contains_parent: Dict[int, int],
    function_profiles: Dict[int, Dict[str, Any]],
) -> bool:
    edge_type = str(edge.get("type", ""))
    edge_kind = str(edge.get("kind", ""))
    src = int(edge["src"])
    dst = int(edge["dst"])
    src_owner = resolve_function_owner(src, node_by_id, contains_parent)
    dst_owner = resolve_function_owner(dst, node_by_id, contains_parent)

    # Leave non-summary edges untouched; only explicit interprocedural summaries
    # and local-call links are filtered.
    if edge_type == "call_interaction" and edge_kind in {"local_call", "local_call_summary"}:
        caller_fn: Optional[int] = None
        callee_fn: Optional[int] = None
        if node_by_id.get(src, {}).get("type") == "function":
            caller_fn = src
        else:
            caller_fn = src_owner
        if node_by_id.get(dst, {}).get("type") == "function":
            callee_fn = dst
        else:
            callee_fn = dst_owner
        return function_is_mechanism_like(callee_fn, function_profiles)

    if edge_type == "dfg_dep" and edge_kind == "cross_function_state_flow":
        var_name = edge.get("var") or node_by_id.get(dst, {}).get("name") or node_by_id.get(src, {}).get("name")
        if is_mechanism_like_var(var_name):
            return True
        return function_is_mechanism_like(src_owner, function_profiles) and function_is_mechanism_like(dst_owner, function_profiles)

    if edge_type == "data_dep" and edge_kind in {"state_write_summary", "state_read_summary"}:
        fn_id: Optional[int] = None
        if node_by_id.get(src, {}).get("type") == "function":
            fn_id = src
        elif node_by_id.get(dst, {}).get("type") == "function":
            fn_id = dst
        else:
            fn_id = src_owner or dst_owner
        var_name = edge.get("var") or node_by_id.get(src, {}).get("name") or node_by_id.get(dst, {}).get("name")
        return is_mechanism_like_var(var_name) or function_is_mechanism_like(fn_id, function_profiles)

    if edge_type == "data_dep" and edge_kind == "shared_object_summary":
        obj_name = str(edge.get("object") or node_by_id.get(src, {}).get("name") or node_by_id.get(dst, {}).get("name") or "").lower()
        return obj_name in MECHANISM_SHARED_OBJECT_KEYS

    return True


def build_adjacency(
    graph: Dict[str, Any],
    rel_whitelist: Set[str],
    node_by_id: Dict[int, Dict[str, Any]],
    contains_parent: Dict[int, int],
    contains_children: Dict[int, Set[int]],
    cross_function_propagation_mode: str = "all",
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
    undirected: Dict[int, Set[int]] = collections.defaultdict(set)
    forward: Dict[int, Set[int]] = collections.defaultdict(set)
    backward: Dict[int, Set[int]] = collections.defaultdict(set)
    function_profiles = build_function_profiles(node_by_id=node_by_id, contains_children=contains_children)
    for e in graph.get("edges", []):
        if e.get("type") not in rel_whitelist:
            continue
        s = int(e["src"])
        d = int(e["dst"])
        if cross_function_propagation_mode == "same_function":
            src_owner = resolve_function_owner(s, node_by_id, contains_parent)
            dst_owner = resolve_function_owner(d, node_by_id, contains_parent)
            if src_owner is not None and dst_owner is not None and src_owner != dst_owner:
                continue
        if cross_function_propagation_mode == "typed_mechanism":
            if not should_keep_typed_interproc_edge(
                e,
                node_by_id=node_by_id,
                contains_parent=contains_parent,
                function_profiles=function_profiles,
            ):
                continue
        undirected[s].add(d)
        undirected[d].add(s)
        forward[s].add(d)
        backward[d].add(s)
    return (
        {k: sorted(v) for k, v in undirected.items()},
        {k: sorted(v) for k, v in forward.items()},
        {k: sorted(v) for k, v in backward.items()},
    )


def bfs_nodes(adj: Dict[int, List[int]], start: int, hops: int) -> Set[int]:
    visited = {start}
    q = collections.deque([(start, 0)])
    while q:
        cur, dep = q.popleft()
        if dep >= hops:
            continue
        for nxt in adj.get(cur, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            q.append((nxt, dep + 1))
    return visited


def directional_expand(
    forward_adj: Dict[int, List[int]],
    backward_adj: Dict[int, List[int]],
    start: int,
    forward_hops: int,
    backward_hops: int,
) -> Set[int]:
    out = {start}
    out.update(bfs_nodes(forward_adj, start, hops=max(0, forward_hops)))
    out.update(bfs_nodes(backward_adj, start, hops=max(0, backward_hops)))
    return out


def keep_anchor_function_scope(
    node_ids: Set[int],
    anchor_id: int,
    node_by_id: Dict[int, Dict[str, Any]],
    contains_parent: Dict[int, int],
) -> Set[int]:
    anchor_owner = resolve_function_owner(anchor_id, node_by_id, contains_parent)
    if anchor_owner is None:
        return set(node_ids)
    kept: Set[int] = set()
    for nid in node_ids:
        owner = resolve_function_owner(nid, node_by_id, contains_parent)
        if owner is None or owner == anchor_owner:
            kept.add(nid)
    kept.add(anchor_id)
    kept.add(anchor_owner)
    return kept


def shortest_path_between_sets(
    adj: Dict[int, List[int]],
    src_set: Set[int],
    dst_set: Set[int],
    max_hops: int,
) -> Optional[List[int]]:
    if not src_set or not dst_set:
        return None
    if src_set.intersection(dst_set):
        return [next(iter(src_set.intersection(dst_set)))]

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
        for nb in adj.get(cur, []):
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


def role_nodes_by_type(node_ids: Iterable[int], node_by_id: Dict[int, Dict[str, Any]]) -> Dict[str, Set[int]]:
    out = {r: set() for r in ROLE_TYPES}
    for nid in node_ids:
        for r in node_by_id.get(nid, {}).get("roles", []) or []:
            if r in out:
                out[r].add(nid)
    return out


def expand_cross_function_context(
    node_ids: Set[int],
    undirected_adj: Dict[int, List[int]],
    node_by_id: Dict[int, Dict[str, Any]],
    contains_parent: Dict[int, int],
    context_mode: str = "all",
    max_extra_nodes: int = 24,
) -> Tuple[Set[int], bool]:
    expanded = set(node_ids)
    anchor_ids = [
        nid
        for nid in sorted(node_ids)
        if node_by_id.get(nid, {}).get("type") in {"statement", "call", "condition", "state_var", "data_object"}
        or bool(node_by_id.get(nid, {}).get("roles"))
    ]

    for nid in anchor_ids:
        parent_fn = contains_parent.get(nid)
        if parent_fn is not None:
            expanded.add(parent_fn)

    if context_mode == "none":
        return expanded, len(expanded) > len(node_ids)

    anchor_parent_fns = {
        contains_parent.get(nid)
        for nid in anchor_ids
        if contains_parent.get(nid) is not None
    }
    allowed_neighbor_types = {"function", "state_var", "data_object", "call", "condition"}
    extra: List[int] = []
    seen_extra: Set[int] = set()
    for nid in anchor_ids:
        for nb in undirected_adj.get(nid, []):
            if nb in expanded or nb in seen_extra:
                continue
            if node_by_id.get(nb, {}).get("type") not in allowed_neighbor_types:
                continue
            if context_mode == "same_function":
                nb_owner = resolve_function_owner(nb, node_by_id, contains_parent)
                nb_type = node_by_id.get(nb, {}).get("type")
                if nb_type == "function" and nb not in anchor_parent_fns:
                    continue
                if nb_owner is not None and nb_owner not in anchor_parent_fns:
                    continue
            seen_extra.add(nb)
            extra.append(nb)
            if len(extra) >= max(0, int(max_extra_nodes)):
                break
        if len(extra) >= max(0, int(max_extra_nodes)):
            break

    expanded.update(extra)
    return expanded, len(expanded) > len(node_ids)


def apply_mechanism_closure(
    node_ids: Set[int],
    undirected_adj: Dict[int, List[int]],
    node_by_id: Dict[int, Dict[str, Any]],
    max_path_len: int,
) -> Tuple[Set[int], bool]:
    role_map = role_nodes_by_type(node_ids, node_by_id)
    added = False
    for a, b in ROLE_TRIPLETS:
        pa = role_map.get(a, set())
        pb = role_map.get(b, set())
        if not pa or not pb:
            continue
        path = shortest_path_between_sets(undirected_adj, pa, pb, max_hops=max_path_len)
        if not path:
            continue
        before = len(node_ids)
        node_ids.update(path)
        if len(node_ids) > before:
            added = True
    return node_ids, added


def make_subgraph(graph: Dict[str, Any], node_ids: Set[int]) -> Dict[str, Any]:
    id_map = {old: new for new, old in enumerate(sorted(node_ids))}
    nodes = []
    for old in sorted(node_ids):
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
    edges = dedup_edges(edges)
    return {"nodes": nodes, "edges": edges}


def stmt_line_set(subgraph: Dict[str, Any]) -> Set[int]:
    return {
        int(n.get("line"))
        for n in subgraph.get("nodes", [])
        if n.get("type") == "statement" and isinstance(n.get("line"), int)
    }


def line_jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2: role-aware state influence slicing.")
    parser.add_argument("--graph-dir", type=str, default="data/graphs/tagged")
    parser.add_argument("--output", type=str, default="data/slices/slices.jsonl")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--forward-hops", type=int, default=0, help="0 means use --hops")
    parser.add_argument("--backward-hops", type=int, default=0, help="0 means use --hops")
    parser.add_argument("--min-slice-nodes", type=int, default=6)
    parser.add_argument("--max-graphs", type=int, default=0)
    parser.add_argument(
        "--formal-propagation",
        action="store_true",
        help="Use role-aware directional propagation instead of undirected BFS.",
    )
    parser.add_argument(
        "--mechanism-path-max-len",
        type=int,
        default=4,
        help="Max path length used for auth/interaction/state mechanism closure in formal mode.",
    )
    parser.add_argument(
        "--include-contains",
        action="store_true",
        help="Include contains edges in adjacency (helps function-local coverage).",
    )
    parser.add_argument(
        "--cross-function-propagation-mode",
        choices=["all", "same_function", "anchor_function", "typed_mechanism"],
        default="all",
        help="Whether role propagation may traverse edges that connect different functions.",
    )
    parser.add_argument(
        "--cross-function-context-mode",
        choices=["all", "same_function", "none"],
        default="all",
        help="How much extra context may be added around seed nodes after propagation.",
    )
    parser.add_argument(
        "--function-fallback-mode",
        choices=["none", "no-role", "all"],
        default="none",
        help="Generate function-centric fallback slices.",
    )
    parser.add_argument("--function-fallback-min-nodes", type=int, default=3)
    parser.add_argument(
        "--roles",
        type=str,
        default="state_change,external_interaction,auth_constraint",
        help="Comma separated role seeds.",
    )
    parser.add_argument(
        "--dedup-line-jaccard",
        type=float,
        default=0.95,
        help="Drop near-duplicate slices in the same function when statement-line Jaccard >= threshold. <=0 disables.",
    )
    parser.add_argument(
        "--max-slices-per-function",
        type=int,
        default=120,
        help="Hard cap of kept slices per function within a contract. <=0 disables.",
    )
    parser.add_argument(
        "--stats-out",
        type=str,
        default="",
        help="Optional path for slicing statistics json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_dir = (ROOT / args.graph_dir).resolve()
    out_path = (ROOT / args.output).resolve()
    role_set = {x.strip() for x in args.roles.split(",") if x.strip()}
    graph_paths = sorted(graph_dir.glob("*.json"), key=lambda x: x.name)
    if args.max_graphs and args.max_graphs > 0:
        graph_paths = graph_paths[: args.max_graphs]

    rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "graphs_total": 0,
        "seed_nodes_total": 0,
        "candidate_slices_total": 0,
        "kept_slices_total": 0,
        "pruned_by_dedup_line_jaccard": 0,
        "pruned_by_max_slices_per_function": 0,
        "dedup_line_jaccard": float(args.dedup_line_jaccard),
        "max_slices_per_function": int(args.max_slices_per_function),
        "formal_propagation": bool(args.formal_propagation),
        "function_fallback_mode": str(args.function_fallback_mode),
        "include_contains": bool(args.include_contains),
        "cross_function_propagation_mode": str(args.cross_function_propagation_mode),
        "cross_function_context_mode": str(args.cross_function_context_mode),
        "graphs": [],
    }
    for gp in graph_paths:
        g = load_json(gp)
        graph_stat: Dict[str, Any] = {
            "graph_file": gp.name,
            "contract_id": g.get("contract_id"),
            "source_path": g.get("source_path"),
            "seed_nodes": 0,
            "candidate_slices": 0,
            "kept_slices": 0,
            "pruned_by_dedup_line_jaccard": 0,
            "pruned_by_max_slices_per_function": 0,
        }
        stats["graphs_total"] += 1
        rel_whitelist = set(DEFAULT_RELATIONS)
        if args.include_contains:
            rel_whitelist.add("contains")
        node_by_id = {int(n["id"]): n for n in g.get("nodes", [])}
        contains_children: Dict[int, Set[int]] = collections.defaultdict(set)
        for e in g.get("edges", []):
            if e.get("type") != "contains":
                continue
            s = int(e["src"])
            d = int(e["dst"])
            contains_children[s].add(d)
        contains_parent: Dict[int, int] = {}
        for fn_id, children in contains_children.items():
            if node_by_id.get(fn_id, {}).get("type") != "function":
                continue
            for child_id in children:
                contains_parent[child_id] = fn_id
        undirected_adj, forward_adj, backward_adj = build_adjacency(
            g,
            rel_whitelist,
            node_by_id=node_by_id,
            contains_parent=contains_parent,
            contains_children=contains_children,
            cross_function_propagation_mode=str(args.cross_function_propagation_mode),
        )
        seeds = [
            int(n["id"])
            for n in g.get("nodes", [])
            if role_set.intersection(set(n.get("roles", []) or []))
        ]
        graph_stat["seed_nodes"] = len(seeds)
        stats["seed_nodes_total"] += len(seeds)
        unique_slices: Dict[frozenset[int], Set[int]] = {}
        slice_strategy: Dict[frozenset[int], Set[str]] = {}
        kept_line_sets_by_fn: Dict[Tuple[str, ...], List[Set[int]]] = collections.defaultdict(list)
        kept_count_by_fn: Dict[Tuple[str, ...], int] = collections.defaultdict(int)
        f_hops = args.forward_hops if args.forward_hops > 0 else args.hops
        b_hops = args.backward_hops if args.backward_hops > 0 else args.hops
        for seed in seeds:
            if args.formal_propagation:
                nodes = directional_expand(
                    forward_adj=forward_adj,
                    backward_adj=backward_adj,
                    start=seed,
                    forward_hops=f_hops,
                    backward_hops=b_hops,
                )
                if args.cross_function_propagation_mode == "anchor_function":
                    nodes = keep_anchor_function_scope(
                        node_ids=nodes,
                        anchor_id=seed,
                        node_by_id=node_by_id,
                        contains_parent=contains_parent,
                    )
                nodes, closure_added = apply_mechanism_closure(
                    node_ids=nodes,
                    undirected_adj=undirected_adj,
                    node_by_id=node_by_id,
                    max_path_len=args.mechanism_path_max_len,
                )
                if args.cross_function_propagation_mode == "anchor_function":
                    nodes = keep_anchor_function_scope(
                        node_ids=nodes,
                        anchor_id=seed,
                        node_by_id=node_by_id,
                        contains_parent=contains_parent,
                    )
                nodes, context_added = expand_cross_function_context(
                    node_ids=nodes,
                    undirected_adj=undirected_adj,
                    node_by_id=node_by_id,
                    contains_parent=contains_parent,
                    context_mode=str(args.cross_function_context_mode),
                )
                strategy_name = "role_directional"
                if closure_added:
                    strategy_name = "role_directional+closure"
                if context_added:
                    strategy_name = f"{strategy_name}+ctx"
            else:
                nodes = bfs_nodes(undirected_adj, seed, hops=args.hops)
                strategy_name = "role_bfs"
            if len(nodes) < args.min_slice_nodes:
                continue
            key = frozenset(nodes)
            if key not in unique_slices:
                unique_slices[key] = set()
                slice_strategy[key] = set()
            unique_slices[key].add(seed)
            slice_strategy[key].add(strategy_name)

        if args.function_fallback_mode != "none":
            function_node_ids = [int(n["id"]) for n in g.get("nodes", []) if n.get("type") == "function"]
            for fn_id in function_node_ids:
                children = set(contains_children.get(fn_id, set()))
                if not children:
                    continue
                fn_scope = {fn_id} | children
                has_role = any(
                    role_set.intersection(set(node_by_id.get(nid, {}).get("roles", []) or []))
                    for nid in fn_scope
                )
                if args.function_fallback_mode == "no-role" and has_role:
                    continue

                fallback_nodes = set(fn_scope)
                # Pull one-hop semantic neighbors from function-local statements.
                for sid in children:
                    for nb in undirected_adj.get(sid, []):
                        nb_node = node_by_id.get(nb, {})
                        if nb_node.get("type") in {"call", "condition", "state_var", "data_object", "statement", "function"}:
                            fallback_nodes.add(nb)
                if len(fallback_nodes) < args.function_fallback_min_nodes:
                    continue
                key = frozenset(fallback_nodes)
                if key not in unique_slices:
                    unique_slices[key] = set()
                    slice_strategy[key] = set()
                unique_slices[key].add(fn_id)
                slice_strategy[key].add("function_fallback")

        graph_stat["candidate_slices"] = len(unique_slices)
        stats["candidate_slices_total"] += len(unique_slices)

        for idx, node_set in enumerate(sorted(unique_slices.keys(), key=lambda x: (len(x), tuple(sorted(x))))):
            node_ids = set(node_set)
            sub = make_subgraph(g, node_ids)
            fn_names = sorted(
                {
                    str(n.get("name") or n.get("function"))
                    for n in sub["nodes"]
                    if (n.get("type") == "function" and n.get("name"))
                    or (n.get("type") == "statement" and n.get("function"))
                }
            )
            fn_key = tuple(fn_names) if fn_names else ("<unknown_function>",)
            cur_stmt_lines = stmt_line_set(sub)

            if args.max_slices_per_function > 0 and kept_count_by_fn[fn_key] >= args.max_slices_per_function:
                stats["pruned_by_max_slices_per_function"] += 1
                graph_stat["pruned_by_max_slices_per_function"] += 1
                continue

            if args.dedup_line_jaccard > 0 and cur_stmt_lines:
                is_dup = any(
                    line_jaccard(cur_stmt_lines, prev_lines) >= args.dedup_line_jaccard
                    for prev_lines in kept_line_sets_by_fn[fn_key]
                )
                if is_dup:
                    stats["pruned_by_dedup_line_jaccard"] += 1
                    graph_stat["pruned_by_dedup_line_jaccard"] += 1
                    continue

            seed_roles = sorted(
                {
                    r
                    for n in g.get("nodes", [])
                    if int(n["id"]) in node_ids
                    for r in (n.get("roles", []) or [])
                    if r in role_set
                }
            )
            raw = f"{g['contract_id']}::{idx}::{len(node_ids)}"
            slice_id = hashlib.md5(raw.encode("utf-8")).hexdigest()
            role_kinds = sorted(
                {
                    r
                    for nid in node_ids
                    for r in (node_by_id.get(nid, {}).get("roles", []) or [])
                    if r in role_set
                }
            )
            rows.append(
                {
                    "slice_id": slice_id,
                    "contract_id": g["contract_id"],
                    "source_path": g.get("source_path"),
                    "relative_source_path": g.get("meta", {}).get("relative_source_path"),
                    "seed_roles": seed_roles,
                    "slice_role_kinds": role_kinds,
                    "seed_node_ids": sorted(unique_slices[node_set]),
                    "slice_strategy": sorted(slice_strategy.get(node_set, {"role_bfs"})),
                    "function_names": fn_names,
                    "function_count": int(len(fn_names)),
                    "node_count": len(sub["nodes"]),
                    "edge_count": len(sub["edges"]),
                    "graph": {
                        "nodes": sub["nodes"],
                        "edges": sub["edges"],
                        "meta": {
                            "from_graph": gp.name,
                            "slice_id": slice_id,
                            "contract_id": g["contract_id"],
                        },
                    },
                }
            )
            kept_count_by_fn[fn_key] += 1
            stats["kept_slices_total"] += 1
            graph_stat["kept_slices"] += 1
            if args.dedup_line_jaccard > 0 and cur_stmt_lines:
                kept_line_sets_by_fn[fn_key].append(cur_stmt_lines)
        stats["graphs"].append(graph_stat)
    save_jsonl(out_path, rows)
    if args.stats_out:
        stats_path = (ROOT / args.stats_out).resolve()
        save_json(stats_path, stats)
    print(
        f"[state_influence_slice] slices={len(rows)} "
        f"candidates={stats['candidate_slices_total']} "
        f"dedup_drop={stats['pruned_by_dedup_line_jaccard']} "
        f"cap_drop={stats['pruned_by_max_slices_per_function']} "
        f"output={out_path}"
    )


if __name__ == "__main__":
    main()
