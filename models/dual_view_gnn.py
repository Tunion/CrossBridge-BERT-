from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.mechanism_text_view_utils import tokenize_mechanism_text
from models.single_view_gnn import GraphTensor, SingleViewModel, graph_to_tensor


@dataclass
class DualViewOutput:
    z_full: torch.Tensor
    z_skeleton: torch.Tensor
    z_joint: torch.Tensor
    z_graph_joint: Optional[torch.Tensor] = None
    z_text: Optional[torch.Tensor] = None
    z_global: Optional[torch.Tensor] = None
    z_fused: Optional[torch.Tensor] = None
    gate: Optional[torch.Tensor] = None
    cls_logit: Optional[torch.Tensor] = None
    slice_gate_sparsity: Optional[torch.Tensor] = None
    slice_gate_closure: Optional[torch.Tensor] = None


class MechanismTextEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, vocab_size: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = int(max(8, vocab_size))
        self.embedding = nn.EmbeddingBag(self.vocab_size, hidden_dim, mode="mean")
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() != 1:
            token_ids = token_ids.view(-1)
        if token_ids.numel() <= 0:
            token_ids = torch.zeros((1,), dtype=torch.long, device=token_ids.device)
        offsets = torch.zeros((1,), dtype=torch.long, device=token_ids.device)
        z = self.embedding(token_ids, offsets)
        return self.proj(z).squeeze(0)


class DualViewModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        edge_types: Tuple[str, ...] | None = None,
        edge_weight_config: Optional[Dict[str, Any]] = None,
        text_view_enable: bool = False,
        text_fuse_enable: bool = True,
        text_vocab_size: int = 8192,
        text_max_tokens: int = 96,
        text_mix_weight: float = 0.20,
        text_dropout: float = 0.10,
        gate_enable: bool = False,
        gate_hidden_dim: int = 32,
        gate_temperature: float = 1.0,
        gate_role_target: float = 0.35,
    ):
        super().__init__()
        self.full_model = SingleViewModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            edge_types=edge_types,
            edge_weight_config=edge_weight_config,
            gate_enable=bool(gate_enable),
            gate_hidden_dim=int(gate_hidden_dim),
            gate_temperature=float(gate_temperature),
        )
        self.skeleton_model = SingleViewModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            edge_types=edge_types,
            edge_weight_config=edge_weight_config,
            gate_enable=bool(gate_enable),
            gate_hidden_dim=int(gate_hidden_dim),
            gate_temperature=float(gate_temperature),
        )
        self.edge_types = self.full_model.edge_types
        self.edge_weight_config = dict(edge_weight_config or {})
        self.relation_count = len(self.edge_types)
        self.embed_dim = hidden_dim
        self.text_view_enable = bool(text_view_enable)
        self.text_fuse_enable = bool(text_fuse_enable)
        self.text_vocab_size = int(max(8, text_vocab_size))
        self.text_max_tokens = int(max(8, text_max_tokens))
        self.text_mix_weight = float(min(0.95, max(0.0, text_mix_weight)))
        self.gate_enable = bool(gate_enable)
        self.gate_role_target = float(min(1.0, max(0.0, gate_role_target)))
        self.text_model = (
            MechanismTextEncoder(hidden_dim=hidden_dim, vocab_size=self.text_vocab_size, dropout=float(text_dropout))
            if self.text_view_enable
            else None
        )

    def _gate_role_deficit(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        role_means = aux.get("gate_role_means")
        role_exists = aux.get("gate_role_exists")
        if role_means is None or role_exists is None:
            dev = next(self.parameters()).device
            return torch.tensor(0.0, dtype=torch.float32, device=dev)
        target = torch.tensor(float(self.gate_role_target), dtype=role_means.dtype, device=role_means.device)
        deficits = F.relu(target - role_means) * role_exists
        den = torch.clamp(role_exists.sum(), min=1.0)
        return deficits.sum() / den

    def encode_text(self, mechanism_text: str, device: torch.device | str) -> Optional[torch.Tensor]:
        if self.text_model is None:
            return None
        token_ids = tokenize_mechanism_text(
            mechanism_text,
            vocab_size=self.text_vocab_size,
            max_tokens=self.text_max_tokens,
        )
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        return self.text_model(token_tensor)

    def forward(
        self,
        full_graph: GraphTensor,
        skeleton_graph: GraphTensor,
        mechanism_text: Optional[str] = None,
    ) -> DualViewOutput:
        full_aux: Optional[Dict[str, torch.Tensor]] = None
        skel_aux: Optional[Dict[str, torch.Tensor]] = None
        if self.gate_enable:
            zf, full_aux = self.full_model(full_graph, return_aux=True)
            zs, skel_aux = self.skeleton_model(skeleton_graph, return_aux=True)
        else:
            zf = self.full_model(full_graph)
            zs = self.skeleton_model(skeleton_graph)
        z_graph_joint = 0.5 * (zf + zs)
        z_text = None
        z_joint = z_graph_joint
        gate_sparsity = None
        gate_closure = None
        if self.gate_enable and full_aux is not None and skel_aux is not None:
            gate_sparsity = 0.5 * (full_aux["gate_mean"] + skel_aux["gate_mean"])
            gate_closure = 0.5 * (self._gate_role_deficit(full_aux) + self._gate_role_deficit(skel_aux))
        if self.text_model is not None and mechanism_text is not None:
            z_text = self.encode_text(mechanism_text, z_graph_joint.device)
            if self.text_fuse_enable:
                mix = float(self.text_mix_weight)
                z_joint = (1.0 - mix) * z_graph_joint + mix * z_text
        return DualViewOutput(
            z_full=zf,
            z_skeleton=zs,
            z_joint=z_joint,
            z_graph_joint=z_graph_joint,
            z_text=z_text,
            slice_gate_sparsity=gate_sparsity,
            slice_gate_closure=gate_closure,
        )


class GlobalLocalDualViewModel(nn.Module):
    """Dual-view local encoder plus per-contract global context branch."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        classifier_enable: bool = False,
        classifier_hidden_dim: int = 64,
        edge_types: Tuple[str, ...] | None = None,
        edge_weight_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.full_model = SingleViewModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            edge_types=edge_types,
            edge_weight_config=edge_weight_config,
        )
        self.skeleton_model = SingleViewModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            edge_types=edge_types,
            edge_weight_config=edge_weight_config,
        )
        self.global_model = SingleViewModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            edge_types=edge_types,
            edge_weight_config=edge_weight_config,
        )
        self.edge_types = self.full_model.edge_types
        self.edge_weight_config = dict(edge_weight_config or {})
        self.relation_count = len(self.edge_types)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier_enable = bool(classifier_enable)
        self.classifier_hidden_dim = int(classifier_hidden_dim)
        if self.classifier_enable:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, self.classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.classifier_hidden_dim, 1),
            )
        else:
            self.classifier = None
        self.embed_dim = hidden_dim

    def fuse(self, z_joint: torch.Tensor, z_global: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_in = torch.cat([z_joint, z_global, torch.abs(z_joint - z_global)], dim=-1)
        gate = torch.sigmoid(self.gate(gate_in)).squeeze(-1)
        z_fused = gate.unsqueeze(-1) * z_joint + (1.0 - gate).unsqueeze(-1) * z_global
        return z_fused, gate

    def classify(
        self,
        z_fused: torch.Tensor,
        z_joint: torch.Tensor,
        z_global: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.classifier is None:
            return None
        feat = torch.cat([z_fused, torch.abs(z_joint - z_global)], dim=-1)
        return self.classifier(feat).squeeze(-1)

    def forward(
        self,
        full_graph: GraphTensor,
        skeleton_graph: GraphTensor,
        global_graph: GraphTensor,
    ) -> DualViewOutput:
        zf = self.full_model(full_graph)
        zs = self.skeleton_model(skeleton_graph)
        z_joint = 0.5 * (zf + zs)
        zg = self.global_model(global_graph)
        z_fused, gate = self.fuse(z_joint, zg)
        cls_logit = self.classify(z_fused, z_joint, zg)
        return DualViewOutput(
            z_full=zf,
            z_skeleton=zs,
            z_joint=z_joint,
            z_global=zg,
            z_fused=z_fused,
            gate=gate,
            cls_logit=cls_logit,
        )


def compute_dual_loss(
    out: DualViewOutput,
    center: torch.Tensor | None = None,
    lambda_align: float = 1.0,
    lambda_compact: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    align = F.mse_loss(out.z_full, out.z_skeleton)
    if center is None:
        center = torch.zeros_like(out.z_joint)
    compact = torch.mean((out.z_joint - center) ** 2)
    loss = lambda_align * align + lambda_compact * compact
    return loss, {"align": float(align.item()), "compact": float(compact.item())}


def dual_view_to_tensors(
    full_graph: Dict[str, Any],
    skeleton_graph: Dict[str, Any],
    device: torch.device | str = "cpu",
    edge_types: Tuple[str, ...] | None = None,
    edge_weight_config: Optional[Dict[str, Any]] = None,
) -> Tuple[GraphTensor, GraphTensor]:
    return (
        graph_to_tensor(full_graph, device=device, edge_types=edge_types, edge_weight_config=edge_weight_config),
        graph_to_tensor(skeleton_graph, device=device, edge_types=edge_types, edge_weight_config=edge_weight_config),
    )
