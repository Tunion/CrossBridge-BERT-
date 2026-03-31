from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from common.graph_utils import build_soft_edge_weight_config
from common.metamorphic_utils import build_dual_metamorphic_variants
from models.dual_view_gnn import DualViewModel, GlobalLocalDualViewModel, dual_view_to_tensors
from models.single_view_gnn import graph_to_tensor


def load_dual_model_ckpt(ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model_type = str(ckpt.get("model_type", "dual-view")).strip().lower()
    model_cls = GlobalLocalDualViewModel if model_type == "global-local-dual-view" else DualViewModel
    if model_cls is GlobalLocalDualViewModel:
        model = model_cls(
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
            classifier_enable=bool(cfg.get("global_local_classifier_enable", False)),
            classifier_hidden_dim=int(cfg.get("global_local_classifier_hidden_dim", 64)),
            edge_weight_config=build_soft_edge_weight_config(
                enable=bool(cfg.get("edge_soft_weight_enable", False)),
                local_call_summary=float(cfg.get("edge_weight_local_call_summary", 0.35)),
                shared_object_summary=float(cfg.get("edge_weight_shared_object_summary", 0.45)),
                cross_function_state_flow=float(cfg.get("edge_weight_cross_function_state_flow", 0.55)),
                local_call=float(cfg.get("edge_weight_local_call", 0.75)),
                state_summary=float(cfg.get("edge_weight_state_summary", 0.85)),
            ),
        ).to(device)
    else:
        model = model_cls(
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
            edge_weight_config=build_soft_edge_weight_config(
                enable=bool(cfg.get("edge_soft_weight_enable", False)),
                local_call_summary=float(cfg.get("edge_weight_local_call_summary", 0.35)),
                shared_object_summary=float(cfg.get("edge_weight_shared_object_summary", 0.45)),
                cross_function_state_flow=float(cfg.get("edge_weight_cross_function_state_flow", 0.55)),
                local_call=float(cfg.get("edge_weight_local_call", 0.75)),
                state_summary=float(cfg.get("edge_weight_state_summary", 0.85)),
            ),
            text_view_enable=bool(cfg.get("text_view_enable", False)),
            text_fuse_enable=bool(cfg.get("text_view_fuse_enable", True)),
            text_vocab_size=int(cfg.get("text_vocab_size", 8192)),
            text_max_tokens=int(cfg.get("text_max_tokens", 96)),
            text_mix_weight=float(cfg.get("text_view_mix_weight", 0.20)),
            text_dropout=float(cfg.get("text_dropout", 0.10)),
            gate_enable=bool(cfg.get("slice_gate_enable", False)),
            gate_hidden_dim=int(cfg.get("slice_gate_hidden_dim", 32)),
            gate_temperature=float(cfg.get("slice_gate_temperature", 1.0)),
            gate_role_target=float(cfg.get("slice_gate_role_target", 0.35)),
        ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_global_graph_cache(graph_dir: Path, rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    graph_map: Dict[str, Dict[str, Any]] = {}
    need_ids = {str(r.get("contract_id", "")).strip() for r in rows if str(r.get("contract_id", "")).strip()}
    for cid in need_ids:
        path = graph_dir / f"{cid}.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            graph_map[cid] = json.load(f)
    return graph_map


def compute_embeddings_for_rows(
    model: torch.nn.Module,
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
    global_graph_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    zf_list: List[np.ndarray] = []
    zs_list: List[np.ndarray] = []
    zj_list: List[np.ndarray] = []
    zg_list: List[np.ndarray] = []
    zfused_list: List[np.ndarray] = []
    global_graph_map: Dict[str, Dict[str, Any]] = {}
    global_emb_cache: Dict[str, torch.Tensor] = {}
    use_global_local = isinstance(model, GlobalLocalDualViewModel)
    if use_global_local and global_graph_dir is not None and global_graph_dir.exists():
        global_graph_map = load_global_graph_cache(global_graph_dir, rows)
    with torch.no_grad():
        for row in rows:
            full_graph = row["full_graph"]
            skeleton_graph = row["skeleton_graph"]
            edge_weight_config = getattr(model, "edge_weight_config", None)
            full_t, skel_t = dual_view_to_tensors(
                full_graph,
                skeleton_graph,
                device=device,
                edge_weight_config=edge_weight_config,
            )
            if use_global_local:
                cid = str(row.get("contract_id", "")).strip()
                global_graph = global_graph_map.get(cid, full_graph)
                if cid not in global_emb_cache:
                    global_emb_cache[cid] = model.global_model(
                        graph_to_tensor(global_graph, device=device, edge_weight_config=edge_weight_config)
                    )
                zf = model.full_model(full_t)
                zs = model.skeleton_model(skel_t)
                zj = 0.5 * (zf + zs)
                zfused, _ = model.fuse(zj, global_emb_cache[cid])
                zg_list.append(global_emb_cache[cid].detach().cpu().numpy())
                zfused_list.append(zfused.detach().cpu().numpy())
            else:
                out = model(full_t, skel_t, mechanism_text=str(row.get("mechanism_text", "") or ""))  # type: ignore[misc]
                zf = out.z_full
                zs = out.z_skeleton
                zj = out.z_joint
            zf_list.append(zf.detach().cpu().numpy())
            zs_list.append(zs.detach().cpu().numpy())
            zj_list.append(zj.detach().cpu().numpy())
    emb = {
        "z_full": np.stack(zf_list, axis=0).astype(np.float32) if zf_list else np.zeros((0, model.embed_dim), dtype=np.float32),
        "z_skeleton": np.stack(zs_list, axis=0).astype(np.float32) if zs_list else np.zeros((0, model.embed_dim), dtype=np.float32),
        "z_joint": np.stack(zj_list, axis=0).astype(np.float32) if zj_list else np.zeros((0, model.embed_dim), dtype=np.float32),
    }
    if zg_list:
        emb["z_global"] = np.stack(zg_list, axis=0).astype(np.float32)
    if zfused_list:
        emb["z_fused"] = np.stack(zfused_list, axis=0).astype(np.float32)
    return emb


def embedding_array_from_map(
    emb_map: Dict[str, np.ndarray],
    embedding_key: str,
    hybrid_alpha: float,
) -> np.ndarray:
    key = str(embedding_key or "z_joint").strip().lower()
    if key == "z_hybrid":
        z_joint = np.asarray(emb_map.get("z_joint", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        z_fused = np.asarray(emb_map.get("z_fused", np.zeros_like(z_joint)), dtype=np.float32)
        if len(z_joint) and len(z_fused) == len(z_joint):
            alpha = float(min(1.0, max(0.0, hybrid_alpha)))
            return (alpha * z_fused + (1.0 - alpha) * z_joint).astype(np.float32)
        return z_joint
    if key in emb_map:
        return np.asarray(emb_map[key], dtype=np.float32)
    return np.asarray(emb_map.get("z_joint", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)


def nearest_to_centers(emb: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dmat = np.linalg.norm(emb[:, None, :] - centers[None, :, :], axis=2)
    assign = np.argmin(dmat, axis=1)
    nearest = dmat[np.arange(len(emb)), assign]
    return assign.astype(np.int64), nearest.astype(np.float32)


def _normalize01(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    if arr.size <= 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo + 1e-8)).astype(np.float32)


def select_metamorphic_candidate_indices(
    rows: Sequence[Dict[str, Any]],
    select_mode: str = "boundary",
    max_rows: int = 0,
    boundary_quantile: float = 0.0,
) -> List[int]:
    mode = str(select_mode or "boundary").strip().lower()
    idx = list(range(len(rows)))
    if mode == "boundary":
        idx = [i for i, r in enumerate(rows) if str(r.get("status", "")) == "boundary"]
    bq = float(min(1.0, max(0.0, boundary_quantile)))
    if idx and bq > 0.0:
        vals = np.asarray([float(rows[i].get("boundary_proximity", 0.0)) for i in idx], dtype=np.float32)
        thr = float(np.quantile(vals, bq))
        idx = [i for i in idx if float(rows[i].get("boundary_proximity", 0.0)) >= thr]
    idx.sort(
        key=lambda i: (
            float(rows[i].get("boundary_proximity", 0.0)),
            float(rows[i].get("final_risk", rows[i].get("base_final_risk", 0.0))),
        ),
        reverse=True,
    )
    limit = int(max(0, max_rows))
    if limit > 0:
        idx = idx[: min(limit, len(idx))]
    return idx


def compute_metamorphic_instability_features(
    eval_rows: Sequence[Dict[str, Any]],
    dual_row_lookup: Dict[str, Dict[str, Any]],
    model: torch.nn.Module,
    device: torch.device,
    global_graph_dir: Optional[Path],
    centers: np.ndarray,
    radii: np.ndarray,
    embedding_key: str,
    hybrid_alpha: float,
    max_hops: int = 4,
    max_rows: int = 0,
    select_mode: str = "boundary",
    boundary_quantile: float = 0.0,
    switch_min: float = 0.0,
    gate_mode: str = "hard",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    n = int(len(eval_rows))
    out = {
        "metamorphic_instability": np.zeros((n,), dtype=np.float32),
        "metamorphic_nearest_delta": np.zeros((n,), dtype=np.float32),
        "metamorphic_margin_delta": np.zeros((n,), dtype=np.float32),
        "metamorphic_view_delta": np.zeros((n,), dtype=np.float32),
        "metamorphic_proto_switch_rate": np.zeros((n,), dtype=np.float32),
        "metamorphic_variant_count": np.zeros((n,), dtype=np.int32),
    }
    if n <= 0:
        return out, {"enabled": False, "selected_rows": 0, "variant_rows": 0}

    cand_idx = select_metamorphic_candidate_indices(
        eval_rows,
        select_mode=select_mode,
        max_rows=max_rows,
        boundary_quantile=boundary_quantile,
    )
    selected_rows: List[Tuple[int, Dict[str, Any]]] = []
    for i in cand_idx:
        sid = str(eval_rows[i].get("slice_id", ""))
        drow = dual_row_lookup.get(sid)
        if drow is not None:
            selected_rows.append((i, drow))
    if not selected_rows:
        return out, {
            "enabled": True,
            "select_mode": str(select_mode),
            "selected_rows": 0,
            "variant_rows": 0,
        }

    variant_rows: List[Dict[str, Any]] = []
    owner_idx: List[int] = []
    for row_idx, drow in selected_rows:
        variants = build_dual_metamorphic_variants(
            full_graph=drow.get("full_graph", {}) or {},
            skeleton_graph=drow.get("skeleton_graph", {}) or {},
            max_hops=int(max(1, max_hops)),
        )
        for variant_name, full_v, skel_v in variants:
            vrow = dict(drow)
            vrow["slice_id"] = f"{drow.get('slice_id')}::meta::{variant_name}"
            vrow["full_graph"] = full_v
            vrow["skeleton_graph"] = skel_v
            variant_rows.append(vrow)
            owner_idx.append(int(row_idx))
    if not variant_rows:
        return out, {
            "enabled": True,
            "select_mode": str(select_mode),
            "selected_rows": int(len(selected_rows)),
            "variant_rows": 0,
        }

    emb_map_var = compute_embeddings_for_rows(
        model=model,
        rows=variant_rows,
        device=device,
        global_graph_dir=global_graph_dir,
    )
    proto_emb_var = embedding_array_from_map(emb_map_var, embedding_key=embedding_key, hybrid_alpha=hybrid_alpha)
    if len(proto_emb_var) != len(variant_rows):
        return out, {
            "enabled": True,
            "select_mode": str(select_mode),
            "selected_rows": int(len(selected_rows)),
            "variant_rows": int(len(variant_rows)),
            "fallback": "embedding_count_mismatch",
        }
    assign_var, nearest_var = nearest_to_centers(proto_emb_var, centers)
    margin_var = nearest_var - radii[assign_var]
    zf_var = np.asarray(emb_map_var.get("z_full", np.zeros_like(proto_emb_var)), dtype=np.float32)
    zs_var = np.asarray(emb_map_var.get("z_skeleton", np.zeros_like(proto_emb_var)), dtype=np.float32)
    view_gap_var = np.linalg.norm(zf_var - zs_var, axis=1).astype(np.float32) if len(zf_var) else np.zeros((len(variant_rows),), dtype=np.float32)

    pos_map = {int(row_idx): j for j, (row_idx, _) in enumerate(selected_rows)}
    near_delta = np.zeros((len(selected_rows),), dtype=np.float32)
    margin_delta = np.zeros((len(selected_rows),), dtype=np.float32)
    view_delta = np.zeros((len(selected_rows),), dtype=np.float32)
    switch_rate = np.zeros((len(selected_rows),), dtype=np.float32)
    counts = np.zeros((len(selected_rows),), dtype=np.float32)
    for j, row_idx in enumerate(owner_idx):
        pos = pos_map[int(row_idx)]
        row = eval_rows[int(row_idx)]
        near0 = float(row.get("nearest_distance", 0.0))
        margin0 = float(row.get("boundary_margin", 0.0))
        view0 = float(row.get("view_gap", 0.0))
        proto0 = int(row.get("prototype_id", -1))
        near_delta[pos] += abs(float(nearest_var[j]) - near0) / max(1e-6, abs(near0) + 0.10)
        margin_delta[pos] += abs(float(margin_var[j]) - margin0) / max(1e-6, abs(margin0) + 0.10)
        view_delta[pos] += abs(float(view_gap_var[j]) - view0) / max(1e-6, abs(view0) + 0.10)
        switch_rate[pos] += float(int(assign_var[j]) != proto0)
        counts[pos] += 1.0
    valid = counts > 0.0
    near_delta[valid] /= counts[valid]
    margin_delta[valid] /= counts[valid]
    view_delta[valid] /= counts[valid]
    switch_rate[valid] /= counts[valid]

    instability = (
        0.35 * _normalize01(near_delta)
        + 0.30 * _normalize01(margin_delta)
        + 0.20 * _normalize01(view_delta)
        + 0.15 * _normalize01(switch_rate)
    ).astype(np.float32)

    gate_key = str(gate_mode or "hard").strip().lower()
    sw_min = float(max(0.0, min(1.0, switch_min)))
    boundary_vals = np.asarray([float(eval_rows[int(row_idx)].get("boundary_proximity", 0.0)) for row_idx, _ in selected_rows], dtype=np.float32)
    gated_mask = np.ones_like(switch_rate, dtype=bool)
    gate_weight = np.ones_like(switch_rate, dtype=np.float32)
    if gate_key == "hard":
        gated_mask = switch_rate >= sw_min if sw_min > 0.0 else np.ones_like(switch_rate, dtype=bool)
        gate_weight = gated_mask.astype(np.float32)
    elif gate_key == "soft":
        boundary_gate = _normalize01(boundary_vals)
        switch_gate = _normalize01(switch_rate)
        gate_weight = np.sqrt(np.clip(boundary_gate * switch_gate, 0.0, 1.0)).astype(np.float32)
        gated_mask = gate_weight > 0.0
    else:
        gated_mask = np.ones_like(switch_rate, dtype=bool)
        gate_weight = np.ones_like(switch_rate, dtype=np.float32)

    instability = (instability * gate_weight).astype(np.float32)
    near_delta = (near_delta * gate_weight).astype(np.float32)
    margin_delta = (margin_delta * gate_weight).astype(np.float32)
    view_delta = (view_delta * gate_weight).astype(np.float32)
    switch_rate = (switch_rate * gate_weight).astype(np.float32)

    for j, (row_idx, _) in enumerate(selected_rows):
        i = int(row_idx)
        out["metamorphic_instability"][i] = float(instability[j])
        out["metamorphic_nearest_delta"][i] = float(near_delta[j])
        out["metamorphic_margin_delta"][i] = float(margin_delta[j])
        out["metamorphic_view_delta"][i] = float(view_delta[j])
        out["metamorphic_proto_switch_rate"][i] = float(switch_rate[j])
        out["metamorphic_variant_count"][i] = int(counts[j])

    diag = {
        "enabled": True,
        "select_mode": str(select_mode),
        "boundary_quantile": float(boundary_quantile),
        "selected_rows": int(len(selected_rows)),
        "variant_rows": int(len(variant_rows)),
        "max_hops": int(max(1, max_hops)),
        "gate_mode": str(gate_key),
        "switch_min": float(sw_min),
        "switch_kept_rows": int(np.sum(gated_mask.astype(np.int32))) if len(gated_mask) else 0,
        "gate_weight_mean": float(np.mean(gate_weight)) if len(gate_weight) else 0.0,
        "gate_weight_std": float(np.std(gate_weight)) if len(gate_weight) else 0.0,
        "instability_mean": float(np.mean(instability)) if len(instability) else 0.0,
        "instability_std": float(np.std(instability)) if len(instability) else 0.0,
        "nearest_delta_mean": float(np.mean(near_delta)) if len(near_delta) else 0.0,
        "margin_delta_mean": float(np.mean(margin_delta)) if len(margin_delta) else 0.0,
        "view_delta_mean": float(np.mean(view_delta)) if len(view_delta) else 0.0,
        "proto_switch_mean": float(np.mean(switch_rate)) if len(switch_rate) else 0.0,
    }
    return out, diag
