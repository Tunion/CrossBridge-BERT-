from __future__ import annotations

import argparse
import csv
import json
import numpy as np
import random
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.io_utils import load_jsonl, save_json
from common.graph_utils import build_soft_edge_weight_config, dedup_edges
from common.seed import set_seed
from models.dual_view_gnn import DualViewModel, GlobalLocalDualViewModel, dual_view_to_tensors
from models.single_view_gnn import SingleViewModel, graph_to_tensor


ROLE_SET = {"state_change", "external_interaction", "auth_constraint"}
ROLE_TRIPLETS: Tuple[Tuple[str, str], ...] = (
    ("auth_constraint", "external_interaction"),
    ("external_interaction", "state_change"),
    ("auth_constraint", "state_change"),
)
PERTURBABLE_EDGE_TYPES = {"contains", "ast_parent", "control_dep", "constraint_on"}
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
PERTURBABLE_NODE_TYPES = {"statement", "condition", "data_object"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-3: train full-only/skeleton-only/dual-view unsupervised models.")
    p.add_argument("--input", type=str, default="data/slices/dual_views.jsonl")
    p.add_argument("--mode", type=str, default="dual-view", choices=["full-only", "skeleton-only", "dual-view"])
    p.add_argument("--max-slices", type=int, default=1000)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--var-target", type=float, default=1.0)
    p.add_argument("--lambda-contrast", type=float, default=1.0)
    p.add_argument("--lambda-invariance", type=float, default=0.25)
    p.add_argument("--lambda-variance", type=float, default=0.5)
    p.add_argument("--lambda-covariance", type=float, default=0.1)
    p.add_argument("--lambda-compact", type=float, default=0.05)
    p.add_argument("--lambda-gap-risk", type=float, default=0.3)
    p.add_argument("--text-view-enable", dest="text_view_enable", action="store_true")
    p.add_argument("--no-text-view-enable", dest="text_view_enable", action="store_false")
    p.add_argument("--text-view-fuse-enable", dest="text_view_fuse_enable", action="store_true")
    p.add_argument("--no-text-view-fuse-enable", dest="text_view_fuse_enable", action="store_false")
    p.add_argument("--text-vocab-size", type=int, default=8192)
    p.add_argument("--text-max-tokens", type=int, default=96)
    p.add_argument("--text-view-mix-weight", type=float, default=0.20)
    p.add_argument("--text-dropout", type=float, default=0.10)
    p.add_argument("--lambda-text-contrast", type=float, default=0.20)
    p.add_argument("--lambda-text-invariance", type=float, default=0.10)
    p.add_argument("--slice-gate-enable", dest="slice_gate_enable", action="store_true")
    p.add_argument("--no-slice-gate-enable", dest="slice_gate_enable", action="store_false")
    p.add_argument("--slice-gate-hidden-dim", type=int, default=32)
    p.add_argument("--slice-gate-temperature", type=float, default=1.0)
    p.add_argument("--slice-gate-role-target", type=float, default=0.35)
    p.add_argument("--lambda-slice-gate-sparsity", type=float, default=0.0)
    p.add_argument("--lambda-slice-gate-closure", type=float, default=0.0)
    p.add_argument("--edge-soft-weight-enable", dest="edge_soft_weight_enable", action="store_true")
    p.add_argument("--no-edge-soft-weight-enable", dest="edge_soft_weight_enable", action="store_false")
    p.add_argument("--edge-weight-local-call-summary", type=float, default=0.35)
    p.add_argument("--edge-weight-shared-object-summary", type=float, default=0.45)
    p.add_argument("--edge-weight-cross-function-state-flow", type=float, default=0.55)
    p.add_argument("--edge-weight-local-call", type=float, default=0.75)
    p.add_argument("--edge-weight-state-summary", type=float, default=0.85)
    p.add_argument("--mechanism-invariance-enable", action="store_true")
    p.add_argument("--lambda-mechanism-invariance", type=float, default=0.20)
    p.add_argument("--mechanism-invariance-edge-drop-rate", type=float, default=0.12)
    p.add_argument("--mechanism-invariance-node-drop-rate", type=float, default=0.00)
    p.add_argument("--mechanism-invariance-max-hops", type=int, default=4)
    p.add_argument("--mechanism-invariance-skeleton-scale", type=float, default=0.50)
    p.add_argument("--mechanism-invariance-min-context-degree", type=int, default=2)
    p.add_argument(
        "--global-local-enable",
        action="store_true",
        help="Enable optional global-local fusion branch using per-contract whole graphs.",
    )
    p.add_argument(
        "--global-graph-dir",
        type=str,
        default="data/graphs/tagged",
        help="Directory containing per-contract whole-graph JSON files keyed by contract_id.",
    )
    p.add_argument("--lambda-global-align", type=float, default=0.20)
    p.add_argument("--lambda-global-risk", type=float, default=0.20)
    p.add_argument("--global-local-classifier-enable", action="store_true")
    p.add_argument("--global-local-classifier-hidden-dim", type=int, default=64)
    p.add_argument("--lambda-global-classifier", type=float, default=0.25)
    p.add_argument("--global-local-classifier-neg-quantile", type=float, default=0.45)
    p.add_argument("--global-local-classifier-pos-quantile", type=float, default=0.92)
    p.add_argument("--global-local-classifier-warmup-epochs", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--model-out", type=str, default="outputs/models/dual_view.pt")
    p.add_argument("--risk-out", type=str, default="outputs/results/dual_view_risk_scores.csv")
    p.add_argument("--summary-out", type=str, default="outputs/results/dual_view_summary.json")
    p.add_argument("--embedding-out", type=str, default="outputs/results/dual_view_embeddings.npz")
    p.set_defaults(text_view_enable=False, text_view_fuse_enable=False, slice_gate_enable=False, edge_soft_weight_enable=False)
    return p.parse_args()


def load_global_graph_map(graph_dir: Path, rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    need_ids = {str(r.get("contract_id", "")).strip() for r in rows if str(r.get("contract_id", "")).strip()}
    graph_map: Dict[str, Dict[str, Any]] = {}
    for cid in need_ids:
        path = graph_dir / f"{cid}.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            graph_map[cid] = json.load(f)
    return graph_map


def resolve_global_graph(row: Dict[str, Any], global_graph_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    cid = str(row.get("contract_id", "")).strip()
    graph = global_graph_map.get(cid)
    if isinstance(graph, dict):
        return graph
    return row["full_graph"]


def split_rows(rows: List[Dict[str, Any]], train_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ids = list(range(len(rows)))
    random.Random(seed).shuffle(ids)
    n = int(len(rows) * train_ratio)
    return [rows[i] for i in ids[:n]], [rows[i] for i in ids[n:]]


def build_edge_soft_weight_config(args: argparse.Namespace) -> Dict[str, Any]:
    return build_soft_edge_weight_config(
        enable=bool(getattr(args, "edge_soft_weight_enable", False)),
        local_call_summary=float(getattr(args, "edge_weight_local_call_summary", 0.35)),
        shared_object_summary=float(getattr(args, "edge_weight_shared_object_summary", 0.45)),
        cross_function_state_flow=float(getattr(args, "edge_weight_cross_function_state_flow", 0.55)),
        local_call=float(getattr(args, "edge_weight_local_call", 0.75)),
        state_summary=float(getattr(args, "edge_weight_state_summary", 0.85)),
    )


def batch_iter(rows: List[Dict[str, Any]], batch_size: int, seed: int, shuffle: bool = True) -> Iterable[List[Dict[str, Any]]]:
    idx = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), batch_size):
        part = idx[i : i + batch_size]
        if not part:
            continue
        yield [rows[j] for j in part]


def _stable_u32(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


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


def _protected_nodes(graph: Dict[str, Any], max_hops: int) -> Set[int]:
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


def _node_degrees(graph: Dict[str, Any]) -> Dict[int, int]:
    deg: Dict[int, int] = {}
    for e in graph.get("edges", []):
        if e.get("type") not in MECHANISM_EDGE_TYPES:
            continue
        s = int(e["src"])
        d = int(e["dst"])
        deg[s] = deg.get(s, 0) + 1
        deg[d] = deg.get(d, 0) + 1
    return deg


def _induced_graph_from_kept(
    graph: Dict[str, Any],
    keep_nodes: Set[int],
    kept_edges: List[Dict[str, Any]],
) -> Dict[str, Any]:
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
    out = {
        "nodes": nodes,
        "edges": dedup_edges(edges),
        "meta": dict(graph.get("meta", {})),
    }
    return out


def perturb_graph_mechanism_preserving(
    graph: Dict[str, Any],
    args: argparse.Namespace,
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if len(nodes) <= 4 or not edges:
        return graph, {"nodes_dropped": 0.0, "edges_dropped": 0.0, "protected_nodes": float(len(nodes))}

    scale = float(args.mechanism_invariance_skeleton_scale) if graph.get("meta", {}).get("view_type") == "security_skeleton" else 1.0
    edge_drop_rate = max(0.0, min(0.95, float(args.mechanism_invariance_edge_drop_rate) * scale))
    node_drop_rate = max(0.0, min(0.50, float(args.mechanism_invariance_node_drop_rate) * scale))
    protected = _protected_nodes(graph, max_hops=int(args.mechanism_invariance_max_hops))
    deg = _node_degrees(graph)

    drop_nodes: Set[int] = set()
    for node in nodes:
        nid = int(node["id"])
        if nid in protected:
            continue
        if node.get("type") not in PERTURBABLE_NODE_TYPES:
            continue
        if deg.get(nid, 0) > int(args.mechanism_invariance_min_context_degree):
            continue
        if rng.random() < node_drop_rate:
            drop_nodes.add(nid)

    keep_nodes = {int(n["id"]) for n in nodes} - drop_nodes
    kept_edges: List[Dict[str, Any]] = []
    dropped_edges = 0
    for e in edges:
        s = int(e["src"])
        d = int(e["dst"])
        if s not in keep_nodes or d not in keep_nodes:
            continue
        et = e.get("type")
        if et in PERTURBABLE_EDGE_TYPES and s not in protected and d not in protected and rng.random() < edge_drop_rate:
            dropped_edges += 1
            continue
        kept_edges.append(e)

    if not kept_edges:
        return graph, {"nodes_dropped": 0.0, "edges_dropped": 0.0, "protected_nodes": float(len(protected))}

    out = _induced_graph_from_kept(graph, keep_nodes, kept_edges)
    out_meta = dict(out.get("meta", {}))
    out_meta.update(
        {
            "mechanism_perturbed": True,
            "mechanism_protected_nodes": int(len(protected)),
            "mechanism_nodes_dropped": int(len(drop_nodes)),
            "mechanism_edges_dropped": int(dropped_edges),
        }
    )
    out["meta"] = out_meta
    diag = {
        "nodes_dropped": float(len(drop_nodes)),
        "edges_dropped": float(dropped_edges),
        "protected_nodes": float(len(protected)),
    }
    return out, diag


def build_mechanism_perturbed_batch(
    batch: List[Dict[str, Any]],
    args: argparse.Namespace,
    epoch: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    out_batch: List[Dict[str, Any]] = []
    edge_drop_stats: List[float] = []
    node_drop_stats: List[float] = []
    protected_stats: List[float] = []
    for idx, row in enumerate(batch):
        sid = str(row.get("slice_id", f"row{idx}"))
        row_seed = int(args.seed) + 7919 * int(epoch) + _stable_u32(sid)
        rng = random.Random(row_seed)
        full_graph, full_diag = perturb_graph_mechanism_preserving(row["full_graph"], args, rng)
        skel_graph, skel_diag = perturb_graph_mechanism_preserving(row["skeleton_graph"], args, rng)
        out_row = dict(row)
        out_row["full_graph"] = full_graph
        out_row["skeleton_graph"] = skel_graph
        out_batch.append(out_row)
        edge_drop_stats.append(full_diag["edges_dropped"] + skel_diag["edges_dropped"])
        node_drop_stats.append(full_diag["nodes_dropped"] + skel_diag["nodes_dropped"])
        protected_stats.append(full_diag["protected_nodes"] + skel_diag["protected_nodes"])
    diag = {
        "edges_dropped": float(np.mean(edge_drop_stats)) if edge_drop_stats else 0.0,
        "nodes_dropped": float(np.mean(node_drop_stats)) if node_drop_stats else 0.0,
        "protected_nodes": float(np.mean(protected_stats)) if protected_stats else 0.0,
    }
    return out_batch, diag


def variance_loss(z: torch.Tensor, target: float = 1.0) -> torch.Tensor:
    if z.shape[0] <= 1:
        return torch.tensor(0.0, device=z.device)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.mean(F.relu(target - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] <= 1:
        return torch.tensor(0.0, device=z.device)
    z = z - z.mean(dim=0, keepdim=True)
    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag.pow(2).sum()) / d


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    if z1.shape[0] <= 1:
        return F.mse_loss(z1, z2)
    z1n = F.normalize(z1, dim=1)
    z2n = F.normalize(z2, dim=1)
    logits = torch.matmul(z1n, z2n.T) / max(temperature, 1e-6)
    labels = torch.arange(z1.shape[0], device=z1.device)
    l1 = F.cross_entropy(logits, labels)
    l2 = F.cross_entropy(logits.T, labels)
    return 0.5 * (l1 + l2)


def init_center_single(model: SingleViewModel, rows: List[Dict[str, Any]], mode: str, device: torch.device) -> torch.Tensor:
    zs = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            g = row["full_graph"] if mode == "full-only" else row["skeleton_graph"]
            z = model(graph_to_tensor(g, device=device, edge_weight_config=model.edge_weight_config))
            zs.append(z)
    if not zs:
        return torch.zeros(model.embed_dim, device=device)
    return torch.stack(zs, dim=0).mean(dim=0).detach()


def init_center_dual(model: DualViewModel, rows: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
    zs = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            full_t, skel_t = dual_view_to_tensors(
                row["full_graph"],
                row["skeleton_graph"],
                device=device,
                edge_weight_config=model.edge_weight_config,
            )
            out = model(full_t, skel_t, mechanism_text=str(row.get("mechanism_text", "") or ""))
            zs.append(out.z_joint)
    if not zs:
        return torch.zeros(model.embed_dim, device=device)
    return torch.stack(zs, dim=0).mean(dim=0).detach()


def init_center_global_local(
    model: GlobalLocalDualViewModel,
    rows: List[Dict[str, Any]],
    global_graph_map: Dict[str, Dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    zs = []
    global_emb_cache: Dict[str, torch.Tensor] = {}
    model.eval()
    with torch.no_grad():
        for row in rows:
            full_t, skel_t = dual_view_to_tensors(
                row["full_graph"],
                row["skeleton_graph"],
                device=device,
                edge_weight_config=model.edge_weight_config,
            )
            cid = str(row.get("contract_id", "")).strip()
            if cid not in global_emb_cache:
                global_t = graph_to_tensor(
                    resolve_global_graph(row, global_graph_map),
                    device=device,
                    edge_weight_config=model.edge_weight_config,
                )
                global_emb_cache[cid] = model.global_model(global_t)
            zf = model.full_model(full_t)
            zs_local = model.skeleton_model(skel_t)
            z_joint = 0.5 * (zf + zs_local)
            z_fused, _ = model.fuse(z_joint, global_emb_cache[cid])
            zs.append(z_fused)
    if not zs:
        return torch.zeros(model.embed_dim, device=device)
    return torch.stack(zs, dim=0).mean(dim=0).detach()


def build_glocal_classifier_targets(
    model: GlobalLocalDualViewModel,
    rows: List[Dict[str, Any]],
    global_graph_map: Dict[str, Dict[str, Any]],
    center: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    if not rows:
        return {}, {"fit_rows": 0, "positive_rows": 0, "negative_rows": 0}
    ids: List[str] = []
    risks: List[float] = []
    model.eval()
    with torch.no_grad():
        global_emb_cache: Dict[str, torch.Tensor] = {}
        for row in rows:
            sid = str(row.get("slice_id", "")).strip()
            if not sid:
                continue
            cid = str(row.get("contract_id", "")).strip()
            full_t, skel_t = dual_view_to_tensors(
                row["full_graph"],
                row["skeleton_graph"],
                device=device,
                edge_weight_config=model.edge_weight_config,
            )
            if cid not in global_emb_cache:
                global_t = graph_to_tensor(
                    resolve_global_graph(row, global_graph_map),
                    device=device,
                    edge_weight_config=model.edge_weight_config,
                )
                global_emb_cache[cid] = model.global_model(global_t)
            zf = model.full_model(full_t)
            zs = model.skeleton_model(skel_t)
            zj = 0.5 * (zf + zs)
            zg = global_emb_cache[cid]
            zfused, _ = model.fuse(zj, zg)
            dist = torch.sum((zfused - center) ** 2).item()
            gap = torch.norm(zf - zs, p=2).item()
            global_gap = torch.norm(zj - zg, p=2).item()
            risk = dist + args.lambda_gap_risk * gap + args.lambda_global_risk * global_gap
            ids.append(sid)
            risks.append(float(risk))
    if not risks:
        return {}, {"fit_rows": 0, "positive_rows": 0, "negative_rows": 0}
    risk_arr = np.asarray(risks, dtype=np.float32)
    neg_q = float(min(0.9, max(0.0, args.global_local_classifier_neg_quantile)))
    pos_q = float(min(0.999, max(neg_q + 1e-3, args.global_local_classifier_pos_quantile)))
    neg_thr = float(np.quantile(risk_arr, neg_q))
    pos_thr = float(np.quantile(risk_arr, pos_q))
    label_map: Dict[str, int] = {}
    for sid, risk in zip(ids, risk_arr):
        if float(risk) <= neg_thr:
            label_map[sid] = 0
        elif float(risk) >= pos_thr:
            label_map[sid] = 1
    diag = {
        "fit_rows": int(len(label_map)),
        "positive_rows": int(sum(1 for v in label_map.values() if v == 1)),
        "negative_rows": int(sum(1 for v in label_map.values() if v == 0)),
        "pseudo_neg_quantile": float(neg_q),
        "pseudo_pos_quantile": float(pos_q),
        "pseudo_neg_threshold": float(neg_thr),
        "pseudo_pos_threshold": float(pos_thr),
    }
    return label_map, diag


def encode_single_batch(model: SingleViewModel, batch: List[Dict[str, Any]], mode: str, device: torch.device) -> torch.Tensor:
    z_list = []
    for row in batch:
        g = row["full_graph"] if mode == "full-only" else row["skeleton_graph"]
        z = model(graph_to_tensor(g, device=device, edge_weight_config=model.edge_weight_config))
        z_list.append(z)
    return torch.stack(z_list, dim=0)


def encode_dual_batch(
    model: DualViewModel,
    batch: List[Dict[str, Any]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    zf_list = []
    zs_list = []
    zj_list = []
    zgj_list = []
    zt_list = []
    gate_sparsity_list = []
    gate_closure_list = []
    for row in batch:
        full_t, skel_t = dual_view_to_tensors(
            row["full_graph"],
            row["skeleton_graph"],
            device=device,
            edge_weight_config=model.edge_weight_config,
        )
        out = model(full_t, skel_t, mechanism_text=str(row.get("mechanism_text", "") or ""))
        zf_list.append(out.z_full)
        zs_list.append(out.z_skeleton)
        zj_list.append(out.z_joint)
        zgj_list.append(out.z_graph_joint if out.z_graph_joint is not None else out.z_joint)
        if out.z_text is not None:
            zt_list.append(out.z_text)
        if out.slice_gate_sparsity is not None:
            gate_sparsity_list.append(out.slice_gate_sparsity)
        if out.slice_gate_closure is not None:
            gate_closure_list.append(out.slice_gate_closure)
    z_text = torch.stack(zt_list, dim=0) if zt_list else None
    gate_sparsity = (
        torch.stack(gate_sparsity_list, dim=0).mean()
        if gate_sparsity_list
        else torch.tensor(0.0, dtype=torch.float32, device=device)
    )
    gate_closure = (
        torch.stack(gate_closure_list, dim=0).mean()
        if gate_closure_list
        else torch.tensor(0.0, dtype=torch.float32, device=device)
    )
    return (
        torch.stack(zf_list, dim=0),
        torch.stack(zs_list, dim=0),
        torch.stack(zj_list, dim=0),
        torch.stack(zgj_list, dim=0),
        z_text,
        gate_sparsity,
        gate_closure,
    )


def encode_dual_batch_joint_only(model: DualViewModel, batch: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
    zj_list = []
    for row in batch:
        full_t, skel_t = dual_view_to_tensors(
            row["full_graph"],
            row["skeleton_graph"],
            device=device,
            edge_weight_config=model.edge_weight_config,
        )
        out = model(full_t, skel_t, mechanism_text=str(row.get("mechanism_text", "") or ""))
        zj_list.append(out.z_joint)
    return torch.stack(zj_list, dim=0)


def encode_global_local_batch(
    model: GlobalLocalDualViewModel,
    batch: List[Dict[str, Any]],
    global_graph_map: Dict[str, Dict[str, Any]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zf_list = []
    zs_list = []
    zj_list = []
    zg_list = []
    zfused_list = []
    global_emb_cache: Dict[str, torch.Tensor] = {}
    for row in batch:
        full_t, skel_t = dual_view_to_tensors(
            row["full_graph"],
            row["skeleton_graph"],
            device=device,
            edge_weight_config=model.edge_weight_config,
        )
        cid = str(row.get("contract_id", "")).strip()
        if cid not in global_emb_cache:
            global_t = graph_to_tensor(
                resolve_global_graph(row, global_graph_map),
                device=device,
                edge_weight_config=model.edge_weight_config,
            )
            global_emb_cache[cid] = model.global_model(global_t)
        zf = model.full_model(full_t)
        zs = model.skeleton_model(skel_t)
        zj = 0.5 * (zf + zs)
        zg = global_emb_cache[cid]
        zfused, _ = model.fuse(zj, zg)
        zf_list.append(zf)
        zs_list.append(zs)
        zj_list.append(zj)
        zg_list.append(zg)
        zfused_list.append(zfused)
    return (
        torch.stack(zf_list, dim=0),
        torch.stack(zs_list, dim=0),
        torch.stack(zj_list, dim=0),
        torch.stack(zg_list, dim=0),
        torch.stack(zfused_list, dim=0),
    )


def encode_global_local_batch_fused_only(
    model: GlobalLocalDualViewModel,
    batch: List[Dict[str, Any]],
    global_graph_map: Dict[str, Dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    out_list = []
    global_emb_cache: Dict[str, torch.Tensor] = {}
    for row in batch:
        full_t, skel_t = dual_view_to_tensors(
            row["full_graph"],
            row["skeleton_graph"],
            device=device,
            edge_weight_config=model.edge_weight_config,
        )
        cid = str(row.get("contract_id", "")).strip()
        if cid not in global_emb_cache:
            global_t = graph_to_tensor(
                resolve_global_graph(row, global_graph_map),
                device=device,
                edge_weight_config=model.edge_weight_config,
            )
            global_emb_cache[cid] = model.global_model(global_t)
        zf = model.full_model(full_t)
        zs = model.skeleton_model(skel_t)
        zj = 0.5 * (zf + zs)
        zg = global_emb_cache[cid]
        zfused, _ = model.fuse(zj, zg)
        out_list.append(zfused)
    return torch.stack(out_list, dim=0)


def run_single_mode(args: argparse.Namespace, rows: List[Dict[str, Any]], train_rows: List[Dict[str, Any]], device: torch.device):
    edge_weight_config = build_edge_soft_weight_config(args)
    model = SingleViewModel(
        args.hidden_dim,
        args.num_layers,
        args.dropout,
        edge_weight_config=edge_weight_config,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    center = init_center_single(model, train_rows, args.mode, device)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_epoch = []
        comp_epoch = {"compact": [], "var": [], "cov": []}
        for batch in batch_iter(train_rows, args.batch_size, seed=args.seed + epoch, shuffle=True):
            z = encode_single_batch(model, batch, args.mode, device)
            compact = torch.mean((z - center.unsqueeze(0)) ** 2)
            var = variance_loss(z, target=args.var_target)
            cov = covariance_loss(z)
            loss = args.lambda_compact * compact + args.lambda_variance * var + args.lambda_covariance * cov
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_epoch.append(float(loss.item()))
            comp_epoch["compact"].append(float(compact.item()))
            comp_epoch["var"].append(float(var.item()))
            comp_epoch["cov"].append(float(cov.item()))
        center = 0.9 * center + 0.1 * init_center_single(model, train_rows, args.mode, device)
        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(loss_epoch)) if loss_epoch else 0.0,
            "compact": float(np.mean(comp_epoch["compact"])) if comp_epoch["compact"] else 0.0,
            "var": float(np.mean(comp_epoch["var"])) if comp_epoch["var"] else 0.0,
            "cov": float(np.mean(comp_epoch["cov"])) if comp_epoch["cov"] else 0.0,
        }
        history.append(rec)
        print(
            f"[train_dual_view:{args.mode}][epoch={epoch}] "
            f"loss={rec['train_loss']:.6f} compact={rec['compact']:.6f} var={rec['var']:.6f} cov={rec['cov']:.6f}"
        )

    model.eval()
    risk_rows = []
    embeds = []
    with torch.no_grad():
        for row in rows:
            g = row["full_graph"] if args.mode == "full-only" else row["skeleton_graph"]
            z = model(graph_to_tensor(g, device=device, edge_weight_config=model.edge_weight_config))
            dist = torch.sum((z - center) ** 2).item()
            embeds.append(z.detach().cpu().numpy())
            risk_rows.append(
                {
                    "slice_id": row["slice_id"],
                    "contract_id": row["contract_id"],
                    "relative_source_path": row.get("relative_source_path"),
                    "function_names": "|".join(row.get("function_names", [])),
                    "mode": args.mode,
                    "view_gap": 0.0,
                    "risk_score": float(dist),
                }
            )
    emb = np.stack(embeds, axis=0).astype(np.float32) if embeds else np.zeros((0, args.hidden_dim), dtype=np.float32)
    return model, center, history, risk_rows, emb


def run_dual_mode(args: argparse.Namespace, rows: List[Dict[str, Any]], train_rows: List[Dict[str, Any]], device: torch.device):
    edge_weight_config = build_edge_soft_weight_config(args)
    model = DualViewModel(
        args.hidden_dim,
        args.num_layers,
        args.dropout,
        edge_weight_config=edge_weight_config,
        text_view_enable=bool(args.text_view_enable),
        text_fuse_enable=bool(args.text_view_fuse_enable),
        text_vocab_size=int(args.text_vocab_size),
        text_max_tokens=int(args.text_max_tokens),
        text_mix_weight=float(args.text_view_mix_weight),
        text_dropout=float(args.text_dropout),
        gate_enable=bool(args.slice_gate_enable),
        gate_hidden_dim=int(args.slice_gate_hidden_dim),
        gate_temperature=float(args.slice_gate_temperature),
        gate_role_target=float(args.slice_gate_role_target),
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    center = init_center_dual(model, train_rows, device=device)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        stat = {
            "contrast": [],
            "invariance": [],
            "var": [],
            "cov": [],
            "compact": [],
            "text_contrast": [],
            "text_invariance": [],
            "gate_sparsity": [],
            "gate_closure": [],
            "mech_inv": [],
            "mech_edges_dropped": [],
            "mech_nodes_dropped": [],
            "mech_protected_nodes": [],
        }
        for batch in batch_iter(train_rows, args.batch_size, seed=args.seed + epoch, shuffle=True):
            zf, zs, zj, z_graph_joint, z_text, gate_sparsity, gate_closure = encode_dual_batch(model, batch, device)
            contrast = contrastive_loss(zf, zs, temperature=args.temperature)
            invariance = F.mse_loss(zf, zs)
            var = variance_loss(zf, target=args.var_target) + variance_loss(zs, target=args.var_target)
            cov = covariance_loss(zf) + covariance_loss(zs)
            compact = torch.mean((zj - center.unsqueeze(0)) ** 2)
            text_contrast = torch.tensor(0.0, device=device)
            text_invariance = torch.tensor(0.0, device=device)
            if z_text is not None:
                text_contrast = contrastive_loss(z_graph_joint, z_text, temperature=args.temperature)
                text_invariance = F.mse_loss(z_graph_joint, z_text)
            mech_inv = torch.tensor(0.0, device=device)
            if args.mechanism_invariance_enable:
                perturbed_batch, pert_diag = build_mechanism_perturbed_batch(batch, args, epoch)
                zj_cf = encode_dual_batch_joint_only(model, perturbed_batch, device)
                mech_inv = F.mse_loss(zj, zj_cf)
                stat["mech_edges_dropped"].append(float(pert_diag["edges_dropped"]))
                stat["mech_nodes_dropped"].append(float(pert_diag["nodes_dropped"]))
                stat["mech_protected_nodes"].append(float(pert_diag["protected_nodes"]))
            loss = (
                args.lambda_contrast * contrast
                + args.lambda_invariance * invariance
                + args.lambda_variance * var
                + args.lambda_covariance * cov
                + args.lambda_compact * compact
                + args.lambda_text_contrast * text_contrast
                + args.lambda_text_invariance * text_invariance
                + args.lambda_slice_gate_sparsity * gate_sparsity
                + args.lambda_slice_gate_closure * gate_closure
                + args.lambda_mechanism_invariance * mech_inv
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))
            stat["contrast"].append(float(contrast.item()))
            stat["invariance"].append(float(invariance.item()))
            stat["var"].append(float(var.item()))
            stat["cov"].append(float(cov.item()))
            stat["compact"].append(float(compact.item()))
            stat["text_contrast"].append(float(text_contrast.item()))
            stat["text_invariance"].append(float(text_invariance.item()))
            stat["gate_sparsity"].append(float(gate_sparsity.item()))
            stat["gate_closure"].append(float(gate_closure.item()))
            stat["mech_inv"].append(float(mech_inv.item()))

        center = 0.9 * center + 0.1 * init_center_dual(model, train_rows, device)
        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "contrast": float(np.mean(stat["contrast"])) if stat["contrast"] else 0.0,
            "invariance": float(np.mean(stat["invariance"])) if stat["invariance"] else 0.0,
            "var": float(np.mean(stat["var"])) if stat["var"] else 0.0,
            "cov": float(np.mean(stat["cov"])) if stat["cov"] else 0.0,
            "compact": float(np.mean(stat["compact"])) if stat["compact"] else 0.0,
            "text_contrast": float(np.mean(stat["text_contrast"])) if stat["text_contrast"] else 0.0,
            "text_invariance": float(np.mean(stat["text_invariance"])) if stat["text_invariance"] else 0.0,
            "gate_sparsity": float(np.mean(stat["gate_sparsity"])) if stat["gate_sparsity"] else 0.0,
            "gate_closure": float(np.mean(stat["gate_closure"])) if stat["gate_closure"] else 0.0,
            "mechanism_invariance": float(np.mean(stat["mech_inv"])) if stat["mech_inv"] else 0.0,
            "mechanism_edges_dropped": float(np.mean(stat["mech_edges_dropped"])) if stat["mech_edges_dropped"] else 0.0,
            "mechanism_nodes_dropped": float(np.mean(stat["mech_nodes_dropped"])) if stat["mech_nodes_dropped"] else 0.0,
            "mechanism_protected_nodes": float(np.mean(stat["mech_protected_nodes"])) if stat["mech_protected_nodes"] else 0.0,
        }
        history.append(rec)
        print(
            f"[train_dual_view:dual-view][epoch={epoch}] "
            f"loss={rec['train_loss']:.6f} contrast={rec['contrast']:.6f} "
            f"inv={rec['invariance']:.6f} textc={rec['text_contrast']:.6f} texti={rec['text_invariance']:.6f} "
            f"gsp={rec['gate_sparsity']:.6f} gcl={rec['gate_closure']:.6f} "
            f"mech_inv={rec['mechanism_invariance']:.6f} "
            f"var={rec['var']:.6f} cov={rec['cov']:.6f} compact={rec['compact']:.6f}"
        )

    model.eval()
    risk_rows = []
    z_full_list = []
    z_skel_list = []
    z_joint_list = []
    z_graph_joint_list = []
    z_text_list = []
    with torch.no_grad():
        for row in rows:
            full_t, skel_t = dual_view_to_tensors(
                row["full_graph"],
                row["skeleton_graph"],
                device=device,
                edge_weight_config=model.edge_weight_config,
            )
            out = model(full_t, skel_t, mechanism_text=str(row.get("mechanism_text", "") or ""))
            dist = torch.sum((out.z_joint - center) ** 2).item()
            gap = torch.norm(out.z_full - out.z_skeleton, p=2).item()
            risk = dist + args.lambda_gap_risk * gap
            risk_rows.append(
                {
                    "slice_id": row["slice_id"],
                    "contract_id": row["contract_id"],
                    "relative_source_path": row.get("relative_source_path"),
                    "function_names": "|".join(row.get("function_names", [])),
                    "mode": args.mode,
                    "view_gap": float(gap),
                    "risk_score": float(risk),
                }
            )
            z_full_list.append(out.z_full.cpu().numpy())
            z_skel_list.append(out.z_skeleton.cpu().numpy())
            z_joint_list.append(out.z_joint.cpu().numpy())
            if out.z_graph_joint is not None:
                z_graph_joint_list.append(out.z_graph_joint.cpu().numpy())
            if out.z_text is not None:
                z_text_list.append(out.z_text.cpu().numpy())
    emb = {
        "z_full": np.stack(z_full_list, axis=0).astype(np.float32) if z_full_list else np.zeros((0, args.hidden_dim), dtype=np.float32),
        "z_skeleton": np.stack(z_skel_list, axis=0).astype(np.float32) if z_skel_list else np.zeros((0, args.hidden_dim), dtype=np.float32),
        "z_joint": np.stack(z_joint_list, axis=0).astype(np.float32) if z_joint_list else np.zeros((0, args.hidden_dim), dtype=np.float32),
    }
    if z_graph_joint_list:
        emb["z_graph_joint"] = np.stack(z_graph_joint_list, axis=0).astype(np.float32)
    if z_text_list:
        emb["z_text"] = np.stack(z_text_list, axis=0).astype(np.float32)
    return model, center, history, risk_rows, emb


def run_global_local_mode(
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    train_rows: List[Dict[str, Any]],
    device: torch.device,
):
    graph_dir = (ROOT / args.global_graph_dir).resolve()
    global_graph_map = load_global_graph_map(graph_dir, rows)
    edge_weight_config = build_edge_soft_weight_config(args)
    model = GlobalLocalDualViewModel(
        args.hidden_dim,
        args.num_layers,
        args.dropout,
        classifier_enable=bool(args.global_local_classifier_enable),
        classifier_hidden_dim=int(args.global_local_classifier_hidden_dim),
        edge_weight_config=edge_weight_config,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    center = init_center_global_local(model, train_rows, global_graph_map=global_graph_map, device=device)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        stat = {
            "contrast": [],
            "invariance": [],
            "var": [],
            "cov": [],
            "compact": [],
            "global_align": [],
            "global_gap": [],
            "cls_loss": [],
            "cls_fit_rows": [],
            "cls_positive_rows": [],
            "cls_negative_rows": [],
            "mech_inv": [],
            "mech_edges_dropped": [],
            "mech_nodes_dropped": [],
            "mech_protected_nodes": [],
        }
        pseudo_label_map: Dict[str, int] = {}
        pseudo_diag: Dict[str, float] = {}
        if model.classifier_enable and epoch > int(args.global_local_classifier_warmup_epochs):
            pseudo_label_map, pseudo_diag = build_glocal_classifier_targets(
                model,
                train_rows,
                global_graph_map,
                center,
                args,
                device,
            )
        for batch in batch_iter(train_rows, args.batch_size, seed=args.seed + epoch, shuffle=True):
            zf, zs, zj, zg, zfused = encode_global_local_batch(model, batch, global_graph_map, device)
            contrast = contrastive_loss(zf, zs, temperature=args.temperature)
            invariance = F.mse_loss(zf, zs)
            var = variance_loss(zf, target=args.var_target) + variance_loss(zs, target=args.var_target) + 0.5 * variance_loss(zg, target=args.var_target)
            cov = covariance_loss(zf) + covariance_loss(zs) + 0.5 * covariance_loss(zg)
            compact = torch.mean((zfused - center.unsqueeze(0)) ** 2)
            global_align = F.mse_loss(zj, zg)
            mech_inv = torch.tensor(0.0, device=device)
            if args.mechanism_invariance_enable:
                perturbed_batch, pert_diag = build_mechanism_perturbed_batch(batch, args, epoch)
                zfused_cf = encode_global_local_batch_fused_only(model, perturbed_batch, global_graph_map, device)
                mech_inv = F.mse_loss(zfused, zfused_cf)
                stat["mech_edges_dropped"].append(float(pert_diag["edges_dropped"]))
                stat["mech_nodes_dropped"].append(float(pert_diag["nodes_dropped"]))
                stat["mech_protected_nodes"].append(float(pert_diag["protected_nodes"]))
            cls_loss = torch.tensor(0.0, device=device)
            if pseudo_label_map:
                idxs = []
                labels = []
                for i, row in enumerate(batch):
                    sid = str(row.get("slice_id", "")).strip()
                    if sid in pseudo_label_map:
                        idxs.append(i)
                        labels.append(float(pseudo_label_map[sid]))
                if idxs:
                    take = torch.tensor(idxs, dtype=torch.long, device=device)
                    logits = model.classify(zfused.index_select(0, take), zj.index_select(0, take), zg.index_select(0, take))
                    if logits is not None:
                        y = torch.tensor(labels, dtype=torch.float32, device=device)
                        cls_loss = F.binary_cross_entropy_with_logits(logits, y)
            loss = (
                args.lambda_contrast * contrast
                + args.lambda_invariance * invariance
                + args.lambda_variance * var
                + args.lambda_covariance * cov
                + args.lambda_compact * compact
                + args.lambda_global_align * global_align
                + args.lambda_global_classifier * cls_loss
                + args.lambda_mechanism_invariance * mech_inv
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))
            stat["contrast"].append(float(contrast.item()))
            stat["invariance"].append(float(invariance.item()))
            stat["var"].append(float(var.item()))
            stat["cov"].append(float(cov.item()))
            stat["compact"].append(float(compact.item()))
            stat["global_align"].append(float(global_align.item()))
            stat["global_gap"].append(float(torch.norm(zj - zg, dim=1).mean().item()))
            stat["cls_loss"].append(float(cls_loss.item()))
            stat["mech_inv"].append(float(mech_inv.item()))
        if pseudo_diag:
            stat["cls_fit_rows"].append(float(pseudo_diag.get("fit_rows", 0.0)))
            stat["cls_positive_rows"].append(float(pseudo_diag.get("positive_rows", 0.0)))
            stat["cls_negative_rows"].append(float(pseudo_diag.get("negative_rows", 0.0)))

        center = 0.9 * center + 0.1 * init_center_global_local(model, train_rows, global_graph_map, device)
        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "contrast": float(np.mean(stat["contrast"])) if stat["contrast"] else 0.0,
            "invariance": float(np.mean(stat["invariance"])) if stat["invariance"] else 0.0,
            "var": float(np.mean(stat["var"])) if stat["var"] else 0.0,
            "cov": float(np.mean(stat["cov"])) if stat["cov"] else 0.0,
            "compact": float(np.mean(stat["compact"])) if stat["compact"] else 0.0,
            "global_align": float(np.mean(stat["global_align"])) if stat["global_align"] else 0.0,
            "global_gap": float(np.mean(stat["global_gap"])) if stat["global_gap"] else 0.0,
            "cls_loss": float(np.mean(stat["cls_loss"])) if stat["cls_loss"] else 0.0,
            "cls_fit_rows": float(np.mean(stat["cls_fit_rows"])) if stat["cls_fit_rows"] else 0.0,
            "cls_positive_rows": float(np.mean(stat["cls_positive_rows"])) if stat["cls_positive_rows"] else 0.0,
            "cls_negative_rows": float(np.mean(stat["cls_negative_rows"])) if stat["cls_negative_rows"] else 0.0,
            "mechanism_invariance": float(np.mean(stat["mech_inv"])) if stat["mech_inv"] else 0.0,
            "mechanism_edges_dropped": float(np.mean(stat["mech_edges_dropped"])) if stat["mech_edges_dropped"] else 0.0,
            "mechanism_nodes_dropped": float(np.mean(stat["mech_nodes_dropped"])) if stat["mech_nodes_dropped"] else 0.0,
            "mechanism_protected_nodes": float(np.mean(stat["mech_protected_nodes"])) if stat["mech_protected_nodes"] else 0.0,
        }
        history.append(rec)
        print(
            f"[train_dual_view:global-local][epoch={epoch}] "
            f"loss={rec['train_loss']:.6f} contrast={rec['contrast']:.6f} "
            f"inv={rec['invariance']:.6f} mech_inv={rec['mechanism_invariance']:.6f} var={rec['var']:.6f} cov={rec['cov']:.6f} "
            f"compact={rec['compact']:.6f} galign={rec['global_align']:.6f} "
            f"ggap={rec['global_gap']:.6f} cls={rec['cls_loss']:.6f}"
        )

    model.eval()
    risk_rows = []
    z_full_list = []
    z_skel_list = []
    z_joint_list = []
    z_global_list = []
    z_fused_list = []
    gate_list = []
    cls_prob_list = []
    fallback_global = 0
    with torch.no_grad():
        global_emb_cache: Dict[str, torch.Tensor] = {}
        for row in rows:
            cid = str(row.get("contract_id", "")).strip()
            if cid not in global_graph_map:
                fallback_global += 1
            full_t, skel_t = dual_view_to_tensors(
                row["full_graph"],
                row["skeleton_graph"],
                device=device,
                edge_weight_config=model.edge_weight_config,
            )
            if cid not in global_emb_cache:
                global_t = graph_to_tensor(
                    resolve_global_graph(row, global_graph_map),
                    device=device,
                    edge_weight_config=model.edge_weight_config,
                )
                global_emb_cache[cid] = model.global_model(global_t)
            zf = model.full_model(full_t)
            zs = model.skeleton_model(skel_t)
            zj = 0.5 * (zf + zs)
            zg = global_emb_cache[cid]
            z_target, gate = model.fuse(zj, zg)
            cls_logit = model.classify(z_target, zj, zg)
            cls_prob = float(torch.sigmoid(cls_logit).item()) if cls_logit is not None else 0.0
            dist = torch.sum((z_target - center) ** 2).item()
            gap = torch.norm(zf - zs, p=2).item()
            global_gap = torch.norm(zj - zg, p=2).item()
            risk = dist + args.lambda_gap_risk * gap + args.lambda_global_risk * global_gap
            risk_rows.append(
                {
                    "slice_id": row["slice_id"],
                    "contract_id": row["contract_id"],
                    "relative_source_path": row.get("relative_source_path"),
                    "function_names": "|".join(row.get("function_names", [])),
                    "mode": "global-local",
                    "view_gap": float(gap),
                    "model_binary_score": float(cls_prob),
                    "risk_score": float(risk),
                }
            )
            z_full_list.append(zf.cpu().numpy())
            z_skel_list.append(zs.cpu().numpy())
            z_joint_list.append(zj.cpu().numpy())
            z_global_list.append(zg.cpu().numpy())
            z_fused_list.append(z_target.cpu().numpy())
            gate_list.append(float(gate.item()) if gate.dim() == 0 else float(gate.mean().item()))
            cls_prob_list.append(float(cls_prob))
    emb = {
        "z_full": np.stack(z_full_list, axis=0).astype(np.float32) if z_full_list else np.zeros((0, args.hidden_dim), dtype=np.float32),
        "z_skeleton": np.stack(z_skel_list, axis=0).astype(np.float32) if z_skel_list else np.zeros((0, args.hidden_dim), dtype=np.float32),
        "z_joint": np.stack(z_joint_list, axis=0).astype(np.float32) if z_joint_list else np.zeros((0, args.hidden_dim), dtype=np.float32),
        "z_global": np.stack(z_global_list, axis=0).astype(np.float32) if z_global_list else np.zeros((0, args.hidden_dim), dtype=np.float32),
        "z_fused": np.stack(z_fused_list, axis=0).astype(np.float32) if z_fused_list else np.zeros((0, args.hidden_dim), dtype=np.float32),
        "model_binary_score": np.asarray(cls_prob_list, dtype=np.float32),
    }
    diag = {
        "global_graph_dir": str(graph_dir),
        "available_global_graphs": int(len(global_graph_map)),
        "fallback_global_count": int(fallback_global),
        "gate_mean": float(np.mean(gate_list)) if gate_list else 0.0,
        "gate_std": float(np.std(gate_list)) if gate_list else 0.0,
        "classifier_enable": bool(model.classifier_enable),
    }
    return model, center, history, risk_rows, emb, diag


def embedding_diagnostics(emb: np.ndarray) -> Dict[str, float]:
    if emb.size == 0:
        return {"mean_std": 0.0, "min_std": 0.0, "max_std": 0.0, "avg_l2": 0.0}
    std = emb.std(axis=0)
    l2 = np.linalg.norm(emb, axis=1)
    return {
        "mean_std": float(np.mean(std)),
        "min_std": float(np.min(std)),
        "max_std": float(np.max(std)),
        "avg_l2": float(np.mean(l2)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    input_path = (ROOT / args.input).resolve()
    model_out = (ROOT / args.model_out).resolve()
    risk_out = (ROOT / args.risk_out).resolve()
    summary_out = (ROOT / args.summary_out).resolve()
    embedding_out = (ROOT / args.embedding_out).resolve()
    for p in [model_out, risk_out, summary_out, embedding_out]:
        p.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    if args.max_slices and args.max_slices > 0:
        rows = rows[: args.max_slices]
    if bool(args.text_view_enable) and bool(args.global_local_enable):
        raise RuntimeError("text_view_enable is currently only supported on the dual-view branch without global-local fusion.")
    train_rows, val_rows = split_rows(rows, args.train_ratio, args.seed)

    extra_summary: Dict[str, Any] = {}
    if args.mode == "dual-view" and args.global_local_enable:
        model, center, history, risk_rows, emb, glocal_diag = run_global_local_mode(args, rows, train_rows, device)
        extra_summary["global_local"] = glocal_diag
    elif args.mode == "dual-view":
        model, center, history, risk_rows, emb = run_dual_mode(args, rows, train_rows, device)
    else:
        model, center, history, risk_rows, emb = run_single_mode(args, rows, train_rows, device)

    risk_rows = sorted(risk_rows, key=lambda x: x["risk_score"], reverse=True)
    with risk_out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["slice_id", "contract_id", "relative_source_path", "function_names", "mode", "view_gap", "risk_score"]
        if any("model_binary_score" in row for row in risk_rows):
            fieldnames.append("model_binary_score")
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(risk_rows)

    state = {
        "mode": args.mode,
        "model_type": "global-local-dual-view" if (args.mode == "dual-view" and args.global_local_enable) else "dual-view",
        "model_state_dict": model.state_dict(),
        "center": center.detach().cpu(),
        "config": vars(args),
    }
    torch.save(state, model_out)

    if isinstance(emb, dict):
        np.savez(embedding_out, **emb)
        emb_diag = {
            "z_full": embedding_diagnostics(emb["z_full"]),
            "z_skeleton": embedding_diagnostics(emb["z_skeleton"]),
            "z_joint": embedding_diagnostics(emb["z_joint"]),
        }
        if "z_graph_joint" in emb:
            emb_diag["z_graph_joint"] = embedding_diagnostics(emb["z_graph_joint"])
        if "z_text" in emb:
            emb_diag["z_text"] = embedding_diagnostics(emb["z_text"])
        if "z_global" in emb:
            emb_diag["z_global"] = embedding_diagnostics(emb["z_global"])
        if "z_fused" in emb:
            emb_diag["z_fused"] = embedding_diagnostics(emb["z_fused"])
    else:
        np.savez(embedding_out, embedding=emb)
        emb_diag = {"embedding": embedding_diagnostics(emb)}

    summary = {
        "mode": args.mode,
        "global_local_enable": bool(args.global_local_enable),
        "num_rows": len(rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "history": history,
        "embedding_diagnostics": emb_diag,
        "top5_risk": risk_rows[:5],
    }
    summary.update(extra_summary)
    save_json(summary_out, summary)
    print(f"[train_dual_view] mode={args.mode} model={model_out}")
    print(f"[train_dual_view] risk_csv={risk_out}")
    print(f"[train_dual_view] embedding_npz={embedding_out}")


if __name__ == "__main__":
    main()
