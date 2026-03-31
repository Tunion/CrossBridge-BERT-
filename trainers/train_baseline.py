from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, save_json
from common.seed import set_seed
from models.single_view_gnn import SingleViewModel, graph_to_tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-1 baseline: single-view unsupervised graph training.")
    p.add_argument("--graph-dir", type=str, default="data/graphs/full")
    p.add_argument("--max-graphs", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--model-out", type=str, default="outputs/models/baseline_single_view.pt")
    p.add_argument("--risk-out", type=str, default="outputs/results/baseline_risk_scores.csv")
    p.add_argument("--summary-out", type=str, default="outputs/results/baseline_summary.json")
    return p.parse_args()


def load_graphs(graph_dir: Path, max_graphs: int) -> List[Dict[str, Any]]:
    paths = sorted(graph_dir.glob("*.json"), key=lambda x: x.name)
    if max_graphs and max_graphs > 0:
        paths = paths[:max_graphs]
    rows = []
    for p in paths:
        g = load_json(p)
        g["__graph_path"] = str(p)
        rows.append(g)
    return rows


def split_dataset(rows: List[Dict[str, Any]], ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    n_train = int(len(rows) * ratio)
    train = [rows[i] for i in idx[:n_train]]
    val = [rows[i] for i in idx[n_train:]]
    return train, val


def init_center(model: SingleViewModel, train_rows: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
    model.eval()
    zs = []
    with torch.no_grad():
        for row in train_rows:
            gt = graph_to_tensor(row, device=device)
            z = model(gt)
            zs.append(z)
    if not zs:
        return torch.zeros((model.embed_dim,), device=device)
    stack = torch.stack(zs, dim=0)
    return stack.mean(dim=0).detach()


def compute_risk(
    model: SingleViewModel,
    rows: List[Dict[str, Any]],
    center: torch.Tensor,
    device: torch.device,
) -> List[Dict[str, Any]]:
    model.eval()
    out: List[Dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            gt = graph_to_tensor(row, device=device)
            z = model(gt)
            dist = torch.sum((z - center) ** 2).item()
            out.append(
                {
                    "contract_id": row.get("contract_id"),
                    "source_path": row.get("source_path"),
                    "relative_source_path": row.get("meta", {}).get("relative_source_path"),
                    "graph_path": row.get("__graph_path", ""),
                    "risk_score": float(dist),
                }
            )
    out.sort(key=lambda x: x["risk_score"], reverse=True)
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    graph_dir = (ROOT / args.graph_dir).resolve()
    model_out = (ROOT / args.model_out).resolve()
    risk_out = (ROOT / args.risk_out).resolve()
    summary_out = (ROOT / args.summary_out).resolve()
    model_out.parent.mkdir(parents=True, exist_ok=True)
    risk_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    rows = load_graphs(graph_dir, max_graphs=args.max_graphs)
    train_rows, val_rows = split_dataset(rows, ratio=args.train_ratio, seed=args.seed)

    model = SingleViewModel(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    center = init_center(model, train_rows, device=device)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for row in train_rows:
            gt = graph_to_tensor(row, device=device)
            z = model(gt)
            loss = torch.mean((z - center) ** 2)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))
        # update center with current model (EMA-like refresh).
        new_center = init_center(model, train_rows, device=device)
        center = 0.9 * center + 0.1 * new_center
        train_loss = float(np.mean(losses)) if losses else 0.0
        history.append({"epoch": epoch, "train_loss": train_loss})
        print(f"[baseline][epoch={epoch}] train_loss={train_loss:.6f}")

    all_risks = compute_risk(model, rows, center=center, device=device)
    val_risks = compute_risk(model, val_rows, center=center, device=device)

    with risk_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["contract_id", "source_path", "relative_source_path", "graph_path", "risk_score"],
        )
        writer.writeheader()
        writer.writerows(all_risks)

    state = {
        "model_state_dict": model.state_dict(),
        "center": center.detach().cpu(),
        "config": vars(args),
    }
    torch.save(state, model_out)

    summary = {
        "num_graphs": len(rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "train_history": history,
        "top5_risk": all_risks[:5],
        "val_risk_mean": float(np.mean([r["risk_score"] for r in val_risks])) if val_risks else 0.0,
    }
    save_json(summary_out, summary)
    print(f"[train_baseline] model={model_out}")
    print(f"[train_baseline] risk_csv={risk_out}")
    print(f"[train_baseline] summary={summary_out}")


if __name__ == "__main__":
    main()

