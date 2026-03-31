from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.graph_utils import EDGE_TYPES, NODE_TYPES, ROLE_TYPES, graph_stats_vector, graph_to_tensors


@dataclass
class GraphTensor:
    x: torch.Tensor
    a: torch.Tensor
    rel_a: torch.Tensor
    stats: torch.Tensor


def graph_to_tensor(
    graph: Dict[str, Any],
    device: torch.device | str = "cpu",
    edge_types: Tuple[str, ...] | None = None,
    edge_weight_config: Optional[Dict[str, Any]] = None,
) -> GraphTensor:
    edge_types_seq = edge_types if edge_types is not None else EDGE_TYPES
    x_np, a_np, rel_a_np = graph_to_tensors(
        graph,
        edge_types=edge_types_seq,
        edge_weight_config=edge_weight_config,
    )
    stats = graph_stats_vector(graph, edge_types=edge_types_seq)
    x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
    a = torch.from_numpy(a_np).to(device=device, dtype=torch.float32)
    rel_a = torch.from_numpy(rel_a_np).to(device=device, dtype=torch.float32)
    s = torch.from_numpy(stats).to(device=device, dtype=torch.float32)
    return GraphTensor(x=x, a=a, rel_a=rel_a, stats=s)


class SimpleGraphEncoder(nn.Module):
    """Lightweight message-passing encoder without heavy graph dependencies."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        edge_types: Tuple[str, ...] | None = None,
        gate_enable: bool = False,
        gate_hidden_dim: int = 32,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        input_dim = len(NODE_TYPES) + 3 + len(ROLE_TYPES)
        self.edge_types = edge_types if edge_types is not None else EDGE_TYPES
        self.num_rel = len(self.edge_types)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.self_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.nei_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.rel_layers = nn.ModuleList(
            [
                nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(self.num_rel)])
                for _ in range(num_layers)
            ]
        )
        self.rel_alpha = nn.Parameter(torch.zeros(num_layers, self.num_rel))
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * 2
        self.gate_enable = bool(gate_enable)
        self.gate_temperature = float(max(1e-3, gate_temperature))
        self.gate_hidden_dim = int(max(8, gate_hidden_dim))
        self.node_gate = (
            nn.Sequential(
                nn.Linear(input_dim, self.gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.gate_hidden_dim, 1),
            )
            if self.gate_enable
            else None
        )

    @staticmethod
    def _row_normalize_torch(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        den = torch.clamp(a.sum(dim=-1, keepdim=True), min=eps)
        return a / den

    def forward(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        rel_a: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]:
        gate: Optional[torch.Tensor] = None
        adj = a
        rel_adj = rel_a
        if self.node_gate is not None:
            gate_logits = self.node_gate(x).squeeze(-1)
            gate = torch.sigmoid(gate_logits / self.gate_temperature)
            pair_gate = gate.unsqueeze(1) * gate.unsqueeze(0)
            eye = torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
            adj = self._row_normalize_torch(a * pair_gate + 1e-3 * eye)
            rel_adj = self._row_normalize_torch(rel_a * pair_gate.unsqueeze(0))

        h = F.relu(self.in_proj(x))
        if gate is not None:
            h = h * gate.unsqueeze(-1)
        for layer_idx, (self_layer, nei_layer, rel_layer_list) in enumerate(
            zip(self.self_layers, self.nei_layers, self.rel_layers)
        ):
            m = torch.matmul(adj, h)
            rel_msg = torch.zeros_like(m)
            rel_gate = torch.sigmoid(self.rel_alpha[layer_idx])
            for rel_idx, rel_layer in enumerate(rel_layer_list):
                mr = torch.matmul(rel_adj[rel_idx], h)
                rel_msg = rel_msg + rel_gate[rel_idx] * rel_layer(mr)
            h = F.relu(self_layer(h) + nei_layer(m) + rel_msg)
            h = self.dropout(h)
            if gate is not None:
                h = h * gate.unsqueeze(-1)
        if gate is not None:
            h_pool = h * gate.unsqueeze(-1)
            gate_sum = torch.clamp(gate.sum(), min=1e-6)
            mean_pool = h_pool.sum(dim=0) / gate_sum
            max_pool = h_pool.max(dim=0).values
        else:
            mean_pool = h.mean(dim=0)
            max_pool = h.max(dim=0).values
        z = torch.cat([mean_pool, max_pool], dim=-1)
        if return_gate:
            return z, gate
        return z


class SingleViewModel(nn.Module):
    """Single-view unsupervised model: graph encoder + projection head."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        edge_types: Tuple[str, ...] | None = None,
        edge_weight_config: Optional[Dict[str, Any]] = None,
        gate_enable: bool = False,
        gate_hidden_dim: int = 32,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        self.edge_types = edge_types if edge_types is not None else EDGE_TYPES
        self.edge_weight_config = dict(edge_weight_config or {})
        self.gate_enable = bool(gate_enable)
        self.encoder = SimpleGraphEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            edge_types=self.edge_types,
            gate_enable=self.gate_enable,
            gate_hidden_dim=int(gate_hidden_dim),
            gate_temperature=float(gate_temperature),
        )
        stats_dim = len(NODE_TYPES) + len(self.edge_types) + 4
        self.stats_proj = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proj = nn.Sequential(
            nn.Linear(self.encoder.out_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.embed_dim = hidden_dim

    def forward(
        self,
        graph_tensor: GraphTensor,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        gate_vec: Optional[torch.Tensor] = None
        if self.gate_enable:
            z_enc, gate_vec = self.encoder(
                graph_tensor.x,
                graph_tensor.a,
                graph_tensor.rel_a,
                return_gate=bool(return_aux),
            )
            if isinstance(z_enc, tuple):
                z = z_enc[0]
            else:
                z = z_enc
        else:
            z = self.encoder(graph_tensor.x, graph_tensor.a, graph_tensor.rel_a)  # type: ignore[assignment]
        z_stats = self.stats_proj(graph_tensor.stats.unsqueeze(0)).squeeze(0)
        out = self.proj(torch.cat([z, z_stats], dim=-1))
        if not return_aux:
            return out

        role_means = torch.zeros((len(ROLE_TYPES),), dtype=out.dtype, device=out.device)
        role_exists = torch.zeros((len(ROLE_TYPES),), dtype=out.dtype, device=out.device)
        gate_mean = torch.tensor(0.0, dtype=out.dtype, device=out.device)
        if gate_vec is not None and gate_vec.numel() > 0:
            gate_mean = gate_vec.mean()
            role_start = len(NODE_TYPES) + 3
            role_feat = graph_tensor.x[:, role_start : role_start + len(ROLE_TYPES)]
            for ridx in range(len(ROLE_TYPES)):
                mask = role_feat[:, ridx] > 0.5
                if bool(mask.any().item()):
                    role_exists[ridx] = 1.0
                    role_means[ridx] = gate_vec[mask].mean()
        aux = {
            "gate_mean": gate_mean,
            "gate_role_means": role_means,
            "gate_role_exists": role_exists,
        }
        return out, aux


def batch_encode(
    model: SingleViewModel,
    graphs: List[Dict[str, Any]],
    device: torch.device | str = "cpu",
) -> np.ndarray:
    model.eval()
    rows: List[np.ndarray] = []
    with torch.no_grad():
        for g in graphs:
            gt = graph_to_tensor(
                g,
                device=device,
                edge_types=model.edge_types,
                edge_weight_config=model.edge_weight_config,
            )
            z = model(gt)
            rows.append(z.detach().cpu().numpy())
    if not rows:
        return np.zeros((0, model.embed_dim), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)
