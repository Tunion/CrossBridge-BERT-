from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.io_utils import load_jsonl, save_json
from common.seed import set_seed
from models.boundary_refine import BoundaryRefiner, RefineWeights
from models.dual_view_gnn import DualViewModel, dual_view_to_tensors
from models.e2e_multi_proto_svdd import E2EMultiProtoSVDDHead
from trainers.train_unsup import (
    adaptive_mechanism_score,
    blend_mechanism_risk,
    compute_mechanism_risk,
    coverage_rerank_rows,
    function_coverage_rerank_rows,
    semantic_closed_loop_mask,
    top_quantile_mask,
    topk_unique_coverage_stats,
    warmup_diversify_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-4/5 (E2E): end-to-end multi-prototype SVDD unsupervised training.")
    p.add_argument("--input", type=str, default="data/slices/dual_views.jsonl")
    p.add_argument("--max-slices", type=int, default=0)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--warmup-epochs", type=int, default=2, help="Dual-view-only warmup epochs before enabling proto losses.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)

    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--var-target", type=float, default=1.0)
    p.add_argument("--lambda-contrast", type=float, default=1.0)
    p.add_argument("--lambda-invariance", type=float, default=0.25)
    p.add_argument("--lambda-variance", type=float, default=0.5)
    p.add_argument("--lambda-covariance", type=float, default=0.1)
    p.add_argument("--lambda-compact", type=float, default=0.05)

    p.add_argument("--prototype-k", type=int, default=8)
    p.add_argument("--boundary-quantile", type=float, default=0.9)
    p.add_argument("--proto-init-radius", type=float, default=1.0)
    p.add_argument("--proto-tau", type=float, default=0.2)
    p.add_argument("--proto-learnable-tau", action="store_true")
    p.add_argument("--proto-min-radius", type=float, default=1e-3)
    p.add_argument("--proto-max-radius", type=float, default=10.0)

    p.add_argument("--lambda-mp", type=float, default=1.0, help="Weight of multi-prototype margin loss.")
    p.add_argument("--lambda-sep", type=float, default=0.05, help="Weight of prototype center separation loss.")
    p.add_argument("--lambda-bal", type=float, default=0.05, help="Weight of assignment balance loss.")
    p.add_argument("--lambda-radius", type=float, default=0.01, help="Weight of radius regularization loss.")
    p.add_argument("--sep-scale", type=float, default=1.0)

    p.add_argument("--refine-w-proto", type=float, default=0.45)
    p.add_argument("--refine-w-view", type=float, default=0.25)
    p.add_argument("--refine-w-density", type=float, default=0.15)
    p.add_argument("--refine-w-stability", type=float, default=0.15)
    p.add_argument("--density-k", type=int, default=10)
    p.add_argument("--n-perturb", type=int, default=5)
    p.add_argument("--noise-std", type=float, default=0.02)

    p.add_argument(
        "--mechanism-risk-weight",
        type=float,
        default=0.0,
        help="Blend weight for mechanism-consistency risk in final score. 0 keeps base risk.",
    )
    p.add_argument(
        "--mechanism-top-quantile",
        type=float,
        default=1.0,
        help="Only apply mechanism correction on samples whose base risk is in top-q quantile. 1.0 disables.",
    )
    p.add_argument(
        "--mechanism-semantic-gate",
        action="store_true",
        help="Apply mechanism correction only on semantic closed-loop slices.",
    )
    p.add_argument(
        "--mechanism-semantic-min-pairs",
        type=int,
        default=2,
        help="Minimum connected role pairs required by semantic closed-loop gate.",
    )
    p.add_argument(
        "--mechanism-semantic-max-hops",
        type=int,
        default=6,
        help="Max hops used when checking role-pair connectivity for semantic gate.",
    )
    p.add_argument(
        "--risk-rerank-mode",
        type=str,
        default="none",
        choices=["none", "coverage", "function_coverage"],
        help=(
            "Optional post-ranking strategy on final_risk. "
            "'coverage' uses duplicate penalties; 'function_coverage' also rewards new file+function coverage."
        ),
    )
    p.add_argument(
        "--risk-rerank-topn",
        type=int,
        default=2000,
        help="Only rerank top-N rows by final_risk; remainder keeps original order.",
    )
    p.add_argument(
        "--risk-rerank-file-penalty",
        type=float,
        default=0.15,
        help="Penalty multiplied by already-selected count in the same file during rerank.",
    )
    p.add_argument(
        "--risk-rerank-filefn-penalty",
        type=float,
        default=0.10,
        help="Penalty multiplied by already-selected count in the same file+function bucket during rerank.",
    )
    p.add_argument(
        "--risk-rerank-fn-novelty-bonus",
        type=float,
        default=0.02,
        help="Only for function_coverage mode: bonus per newly covered function in the same file.",
    )
    p.add_argument(
        "--risk-rerank-fn-overlap-penalty",
        type=float,
        default=0.01,
        help="Only for function_coverage mode: penalty per already-covered function overlap.",
    )
    p.add_argument(
        "--risk-rerank-max-per-filefn",
        type=int,
        default=0,
        help="Only for function_coverage mode: hard cap per (file,function) in rerank head. <=0 disables.",
    )
    p.add_argument(
        "--risk-rerank-warmup-topk",
        type=int,
        default=0,
        help="Optional diversity warmup on top-K rows after rerank. 0 disables.",
    )
    p.add_argument(
        "--risk-rerank-warmup-file-cap",
        type=int,
        default=3,
        help="Max rows per file inside rerank warmup window.",
    )
    p.add_argument(
        "--risk-rerank-warmup-filefn-cap",
        type=int,
        default=1,
        help="Max rows per file+function bucket inside rerank warmup window.",
    )

    p.add_argument("--model-out", type=str, default="outputs/models/e2e_mp_svdd.pt")
    p.add_argument("--prototype-out", type=str, default="outputs/models/e2e_prototypes.npz")
    p.add_argument("--embedding-out", type=str, default="outputs/results/e2e_dual_view_embeddings.npz")
    p.add_argument("--risk-out", type=str, default="outputs/results/e2e_unsup_final_risk_scores.csv")
    p.add_argument("--summary-out", type=str, default="outputs/results/e2e_unsup_final_summary.json")
    p.add_argument("--topk", type=int, default=100)
    return p.parse_args()


def split_rows(rows: List[Dict[str, Any]], train_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ids = list(range(len(rows)))
    random.Random(seed).shuffle(ids)
    n = int(len(rows) * train_ratio)
    return [rows[i] for i in ids[:n]], [rows[i] for i in ids[n:]]


def batch_iter(rows: List[Dict[str, Any]], batch_size: int, seed: int, shuffle: bool = True) -> Iterable[List[Dict[str, Any]]]:
    idx = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), batch_size):
        part = idx[i : i + batch_size]
        if not part:
            continue
        yield [rows[j] for j in part]


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


def init_center_dual(model: DualViewModel, rows: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
    zs = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            full_t, skel_t = dual_view_to_tensors(row["full_graph"], row["skeleton_graph"], device=device)
            out = model(full_t, skel_t)
            zs.append(out.z_joint)
    if not zs:
        return torch.zeros(model.embed_dim, device=device)
    return torch.stack(zs, dim=0).mean(dim=0).detach()


def encode_dual_batch(model: DualViewModel, batch: List[Dict[str, Any]], device: torch.device):
    zf_list = []
    zs_list = []
    zj_list = []
    for row in batch:
        full_t, skel_t = dual_view_to_tensors(row["full_graph"], row["skeleton_graph"], device=device)
        out = model(full_t, skel_t)
        zf_list.append(out.z_full)
        zs_list.append(out.z_skeleton)
        zj_list.append(out.z_joint)
    return torch.stack(zf_list, dim=0), torch.stack(zs_list, dim=0), torch.stack(zj_list, dim=0)


def encode_joint_numpy(model: DualViewModel, rows: List[Dict[str, Any]], device: torch.device) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for row in rows:
            full_t, skel_t = dual_view_to_tensors(row["full_graph"], row["skeleton_graph"], device=device)
            z = model(full_t, skel_t).z_joint
            out.append(z.detach().cpu().numpy())
    if not out:
        return np.zeros((0, model.embed_dim), dtype=np.float32)
    return np.stack(out, axis=0).astype(np.float32)


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
    prototype_out = (ROOT / args.prototype_out).resolve()
    embedding_out = (ROOT / args.embedding_out).resolve()
    risk_out = (ROOT / args.risk_out).resolve()
    summary_out = (ROOT / args.summary_out).resolve()
    for p in [model_out, prototype_out, embedding_out, risk_out, summary_out]:
        p.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    if args.max_slices and args.max_slices > 0:
        rows = rows[: args.max_slices]
    if not rows:
        raise RuntimeError("No dual-view slices found.")

    train_rows, val_rows = split_rows(rows, args.train_ratio, args.seed)
    warmup_epochs = int(max(0, min(args.warmup_epochs, args.epochs)))

    model = DualViewModel(args.hidden_dim, args.num_layers, args.dropout).to(device)
    head = E2EMultiProtoSVDDHead(
        embed_dim=args.hidden_dim,
        n_prototypes=args.prototype_k,
        init_radius=args.proto_init_radius,
        tau=args.proto_tau,
        min_radius=args.proto_min_radius,
        max_radius=args.proto_max_radius,
        learnable_tau=args.proto_learnable_tau,
    ).to(device)
    optim = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    center = init_center_dual(model, train_rows, device=device)

    history: List[Dict[str, float]] = []
    proto_init_info: Dict[str, float] = {"used_k": 0.0, "radius_mean": float(head.radius().mean().item())}

    for epoch in range(1, args.epochs + 1):
        if epoch == warmup_epochs + 1:
            train_emb = encode_joint_numpy(model, train_rows, device=device)
            proto_init_info = head.initialize_from_embeddings(
                embeddings=train_emb,
                boundary_quantile=args.boundary_quantile,
                random_state=args.seed,
            )

        model.train()
        head.train()
        stat = {
            "loss": [],
            "contrast": [],
            "invariance": [],
            "var": [],
            "cov": [],
            "compact": [],
            "mp": [],
            "sep": [],
            "bal": [],
            "radius_reg": [],
            "assign_entropy": [],
        }
        for batch in batch_iter(train_rows, args.batch_size, seed=args.seed + epoch, shuffle=True):
            zf, zs, zj = encode_dual_batch(model, batch, device=device)
            contrast = contrastive_loss(zf, zs, temperature=args.temperature)
            invariance = F.mse_loss(zf, zs)
            var = variance_loss(zf, target=args.var_target) + variance_loss(zs, target=args.var_target)
            cov = covariance_loss(zf) + covariance_loss(zs)
            compact = torch.mean((zj - center.unsqueeze(0)) ** 2)

            if epoch <= warmup_epochs:
                mp = torch.tensor(0.0, device=device)
                sep = torch.tensor(0.0, device=device)
                bal = torch.tensor(0.0, device=device)
                rad = torch.tensor(0.0, device=device)
                assign_entropy = torch.tensor(0.0, device=device)
            else:
                hout = head(zj)
                mp = torch.mean(hout.sample_margin)
                sep = head.separation_loss(sep_scale=args.sep_scale)
                bal = head.balance_loss(hout.assign_prob)
                rad = torch.mean(head.radius() ** 2)
                p = torch.clamp(hout.assign_prob, min=1e-8)
                assign_entropy = -torch.mean(torch.sum(p * torch.log(p), dim=1))

            loss = (
                args.lambda_contrast * contrast
                + args.lambda_invariance * invariance
                + args.lambda_variance * var
                + args.lambda_covariance * cov
                + args.lambda_compact * compact
                + args.lambda_mp * mp
                + args.lambda_sep * sep
                + args.lambda_bal * bal
                + args.lambda_radius * rad
            )
            optim.zero_grad()
            loss.backward()
            optim.step()

            stat["loss"].append(float(loss.item()))
            stat["contrast"].append(float(contrast.item()))
            stat["invariance"].append(float(invariance.item()))
            stat["var"].append(float(var.item()))
            stat["cov"].append(float(cov.item()))
            stat["compact"].append(float(compact.item()))
            stat["mp"].append(float(mp.item()))
            stat["sep"].append(float(sep.item()))
            stat["bal"].append(float(bal.item()))
            stat["radius_reg"].append(float(rad.item()))
            stat["assign_entropy"].append(float(assign_entropy.item()))

        center = 0.9 * center + 0.1 * init_center_dual(model, train_rows, device=device)
        rec = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(stat["loss"])) if stat["loss"] else 0.0,
            "contrast": float(np.mean(stat["contrast"])) if stat["contrast"] else 0.0,
            "invariance": float(np.mean(stat["invariance"])) if stat["invariance"] else 0.0,
            "var": float(np.mean(stat["var"])) if stat["var"] else 0.0,
            "cov": float(np.mean(stat["cov"])) if stat["cov"] else 0.0,
            "compact": float(np.mean(stat["compact"])) if stat["compact"] else 0.0,
            "mp": float(np.mean(stat["mp"])) if stat["mp"] else 0.0,
            "sep": float(np.mean(stat["sep"])) if stat["sep"] else 0.0,
            "bal": float(np.mean(stat["bal"])) if stat["bal"] else 0.0,
            "radius_reg": float(np.mean(stat["radius_reg"])) if stat["radius_reg"] else 0.0,
            "assign_entropy": float(np.mean(stat["assign_entropy"])) if stat["assign_entropy"] else 0.0,
            "radius_mean": float(torch.mean(head.radius()).item()),
            "temperature": float(head.temperature().item()),
            "phase": "warmup" if epoch <= warmup_epochs else "joint",
        }
        history.append(rec)
        print(
            f"[train_e2e_unsup][epoch={epoch}] phase={rec['phase']} loss={rec['train_loss']:.6f} "
            f"mp={rec['mp']:.6f} sep={rec['sep']:.6f} bal={rec['bal']:.6f} "
            f"radius_mean={rec['radius_mean']:.4f} tau={rec['temperature']:.4f}"
        )

    model.eval()
    head.eval()
    z_full_list: List[np.ndarray] = []
    z_skel_list: List[np.ndarray] = []
    z_joint_list: List[np.ndarray] = []
    sample_margin_list: List[float] = []
    nearest_distance_list: List[float] = []
    nearest_idx_list: List[int] = []
    boundary_margin_list: List[float] = []
    rows_meta: List[Dict[str, Any]] = []

    with torch.no_grad():
        radii = head.radius()
        for row in rows:
            full_t, skel_t = dual_view_to_tensors(row["full_graph"], row["skeleton_graph"], device=device)
            out = model(full_t, skel_t)
            hout = head(out.z_joint.unsqueeze(0))
            proto_id = int(hout.nearest_idx.item())
            nearest_dist = float(torch.sqrt(torch.clamp(hout.nearest_dist2, min=1e-12)).item())
            rad = float(radii[proto_id].item())
            boundary_margin = nearest_dist - rad

            z_full_list.append(out.z_full.cpu().numpy())
            z_skel_list.append(out.z_skeleton.cpu().numpy())
            z_joint_list.append(out.z_joint.cpu().numpy())
            sample_margin_list.append(float(hout.sample_margin.item()))
            nearest_distance_list.append(nearest_dist)
            nearest_idx_list.append(proto_id)
            boundary_margin_list.append(float(boundary_margin))
            rows_meta.append(row)

    z_full = np.stack(z_full_list, axis=0).astype(np.float32) if z_full_list else np.zeros((0, args.hidden_dim), dtype=np.float32)
    z_skel = np.stack(z_skel_list, axis=0).astype(np.float32) if z_skel_list else np.zeros((0, args.hidden_dim), dtype=np.float32)
    z_joint = np.stack(z_joint_list, axis=0).astype(np.float32) if z_joint_list else np.zeros((0, args.hidden_dim), dtype=np.float32)
    view_gap = np.linalg.norm(z_full - z_skel, axis=1).astype(np.float32)
    proto_margin = np.asarray(sample_margin_list, dtype=np.float32)
    nearest_dist_arr = np.asarray(nearest_distance_list, dtype=np.float32)
    nearest_idx_arr = np.asarray(nearest_idx_list, dtype=np.int64)
    boundary_margin_arr = np.asarray(boundary_margin_list, dtype=np.float32)

    centers_np = head.centers.detach().cpu().numpy().astype(np.float32)
    radii_np = head.radius().detach().cpu().numpy().astype(np.float32)

    refiner = BoundaryRefiner(
        RefineWeights(
            w_proto=args.refine_w_proto,
            w_view=args.refine_w_view,
            w_density=args.refine_w_density,
            w_stability=args.refine_w_stability,
        )
    )
    refined = refiner.refine(
        proto_margin=proto_margin,
        view_gap=view_gap,
        embeddings=z_joint,
        centers=centers_np,
        density_k=args.density_k,
        n_perturb=args.n_perturb,
        noise_std=args.noise_std,
        random_state=args.seed,
    )

    base_final_risk = refined["final_risk"].astype(np.float32)
    mechanism_raw = compute_mechanism_risk(rows_meta)
    mechanism_score = adaptive_mechanism_score(mechanism_raw, std_threshold=0.05)
    w_mech = float(min(1.0, max(0.0, args.mechanism_risk_weight)))
    top_mask = top_quantile_mask(base_final_risk, quantile=args.mechanism_top_quantile)
    if args.mechanism_semantic_gate:
        sem_mask = semantic_closed_loop_mask(
            rows_meta,
            min_connected_pairs=args.mechanism_semantic_min_pairs,
            max_hops=args.mechanism_semantic_max_hops,
        )
    else:
        sem_mask = np.ones_like(top_mask, dtype=bool)
    active_mask = np.logical_and(top_mask, sem_mask)
    final_risk = blend_mechanism_risk(
        base_final_risk,
        mechanism_score,
        weight=w_mech,
        active_mask=active_mask,
    )

    q90 = float(np.quantile(final_risk, 0.9)) if len(final_risk) else 0.0
    q75 = float(np.quantile(final_risk, 0.75)) if len(final_risk) else 0.0
    status = np.array(["high_risk" if x >= q90 else ("boundary" if x >= q75 else "normal") for x in final_risk])

    out_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows_meta):
        out_rows.append(
            {
                "slice_id": row.get("slice_id"),
                "contract_id": row.get("contract_id"),
                "relative_source_path": row.get("relative_source_path"),
                "function_names": "|".join(row.get("function_names", [])),
                "prototype_id": int(nearest_idx_arr[i]),
                "nearest_distance": float(nearest_dist_arr[i]),
                "boundary_margin": float(boundary_margin_arr[i]),
                "view_gap": float(view_gap[i]),
                "proto_score": float(refined["proto_score"][i]),
                "view_score": float(refined["view_score"][i]),
                "density_risk": float(refined["density_risk"][i]),
                "instability": float(refined["instability"][i]),
                "e2e_margin": float(proto_margin[i]),
                "base_final_risk": float(base_final_risk[i]),
                "mechanism_risk": float(mechanism_score[i]),
                "final_risk": float(final_risk[i]),
                "status": str(status[i]),
            }
        )
    out_rows.sort(key=lambda x: x["final_risk"], reverse=True)

    if args.risk_rerank_mode == "coverage":
        out_rows = coverage_rerank_rows(
            out_rows,
            score_key="final_risk",
            topn=args.risk_rerank_topn,
            file_penalty=args.risk_rerank_file_penalty,
            filefn_penalty=args.risk_rerank_filefn_penalty,
        )
    elif args.risk_rerank_mode == "function_coverage":
        out_rows = function_coverage_rerank_rows(
            out_rows,
            score_key="final_risk",
            topn=args.risk_rerank_topn,
            file_penalty=args.risk_rerank_file_penalty,
            filefn_penalty=args.risk_rerank_filefn_penalty,
            novelty_bonus=args.risk_rerank_fn_novelty_bonus,
            overlap_penalty=args.risk_rerank_fn_overlap_penalty,
            max_per_filefn=args.risk_rerank_max_per_filefn,
        )
    out_rows = warmup_diversify_rows(
        out_rows,
        warmup_topk=args.risk_rerank_warmup_topk,
        per_file_cap=args.risk_rerank_warmup_file_cap,
        per_filefn_cap=args.risk_rerank_warmup_filefn_cap,
    )

    with risk_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "slice_id",
                "contract_id",
                "relative_source_path",
                "function_names",
                "prototype_id",
                "nearest_distance",
                "boundary_margin",
                "view_gap",
                "proto_score",
                "view_score",
                "density_risk",
                "instability",
                "e2e_margin",
                "base_final_risk",
                "mechanism_risk",
                "final_risk",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    state = {
        "model_state_dict": model.state_dict(),
        "head_state_dict": head.state_dict(),
        "center": center.detach().cpu(),
        "config": vars(args),
        "proto_init_info": proto_init_info,
    }
    torch.save(state, model_out)
    np.savez(prototype_out, centers=centers_np, radii=radii_np)
    np.savez(embedding_out, z_full=z_full, z_skeleton=z_skel, z_joint=z_joint)

    summary = {
        "seed": int(args.seed),
        "num_rows": int(len(rows)),
        "num_train": int(len(train_rows)),
        "num_val": int(len(val_rows)),
        "warmup_epochs": int(warmup_epochs),
        "history": history,
        "embedding_diagnostics": {
            "z_full": embedding_diagnostics(z_full),
            "z_skeleton": embedding_diagnostics(z_skel),
            "z_joint": embedding_diagnostics(z_joint),
        },
        "proto_init_info": proto_init_info,
        "prototype_stats": {
            "k": int(args.prototype_k),
            "radius_mean": float(np.mean(radii_np)) if len(radii_np) else 0.0,
            "radius_std": float(np.std(radii_np)) if len(radii_np) else 0.0,
            "radius_min": float(np.min(radii_np)) if len(radii_np) else 0.0,
            "radius_max": float(np.max(radii_np)) if len(radii_np) else 0.0,
            "center_min_pair_dist": float(np.min(np.linalg.norm(centers_np[:, None, :] - centers_np[None, :, :], axis=2) + np.eye(len(centers_np)) * 1e9))
            if len(centers_np) > 1
            else 0.0,
            "assign_usage": {
                str(i): float(np.mean(nearest_idx_arr == i)) for i in range(int(len(centers_np)))
            },
        },
        "mechanism_risk": {
            "weight": w_mech,
            "top_quantile": float(args.mechanism_top_quantile),
            "semantic_gate": bool(args.mechanism_semantic_gate),
            "semantic_min_pairs": int(args.mechanism_semantic_min_pairs),
            "semantic_max_hops": int(args.mechanism_semantic_max_hops),
            "top_mask_ratio": float(np.mean(top_mask.astype(np.float32))) if len(top_mask) else 0.0,
            "semantic_mask_ratio": float(np.mean(sem_mask.astype(np.float32))) if len(sem_mask) else 0.0,
            "active_mask_ratio": float(np.mean(active_mask.astype(np.float32))) if len(active_mask) else 0.0,
            "blend_center_median": float(np.median(mechanism_score)) if len(mechanism_score) else 0.0,
            "raw_mean": float(np.mean(mechanism_raw)) if len(mechanism_raw) else 0.0,
            "raw_std": float(np.std(mechanism_raw)) if len(mechanism_raw) else 0.0,
            "score_mean": float(np.mean(mechanism_score)) if len(mechanism_score) else 0.0,
            "score_std": float(np.std(mechanism_score)) if len(mechanism_score) else 0.0,
        },
        "risk_rerank": {
            "mode": str(args.risk_rerank_mode),
            "topn": int(args.risk_rerank_topn),
            "file_penalty": float(args.risk_rerank_file_penalty),
            "filefn_penalty": float(args.risk_rerank_filefn_penalty),
            "fn_novelty_bonus": float(args.risk_rerank_fn_novelty_bonus),
            "fn_overlap_penalty": float(args.risk_rerank_fn_overlap_penalty),
            "max_per_filefn": int(args.risk_rerank_max_per_filefn),
            "warmup_topk": int(args.risk_rerank_warmup_topk),
            "warmup_file_cap": int(args.risk_rerank_warmup_file_cap),
            "warmup_filefn_cap": int(args.risk_rerank_warmup_filefn_cap),
        },
        "coverage_stats": topk_unique_coverage_stats(out_rows, topk=200),
        "status_counts": {
            "high_risk": int(sum(1 for r in out_rows if r["status"] == "high_risk")),
            "boundary": int(sum(1 for r in out_rows if r["status"] == "boundary")),
            "normal": int(sum(1 for r in out_rows if r["status"] == "normal")),
        },
        "topk_samples": out_rows[: args.topk],
    }
    save_json(summary_out, summary)
    print(f"[train_e2e_unsup] model={model_out}")
    print(f"[train_e2e_unsup] prototypes={prototype_out}")
    print(f"[train_e2e_unsup] embedding_npz={embedding_out}")
    print(f"[train_e2e_unsup] risk_csv={risk_out}")
    print(f"[train_e2e_unsup] summary={summary_out}")


if __name__ == "__main__":
    main()

