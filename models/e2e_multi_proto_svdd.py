from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


@dataclass
class E2EMultiProtoOutput:
    dist2: torch.Tensor
    assign_prob: torch.Tensor
    nearest_idx: torch.Tensor
    nearest_dist2: torch.Tensor
    sample_margin: torch.Tensor


class E2EMultiProtoSVDDHead(nn.Module):
    """End-to-end multi-prototype one-class head with learnable centers and radii."""

    def __init__(
        self,
        embed_dim: int,
        n_prototypes: int = 8,
        init_radius: float = 1.0,
        tau: float = 0.2,
        min_radius: float = 1e-3,
        max_radius: float = 10.0,
        learnable_tau: bool = False,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.n_prototypes = int(max(1, n_prototypes))
        self.min_radius = float(max(1e-6, min_radius))
        self.max_radius = float(max(self.min_radius + 1e-6, max_radius))

        self.centers = nn.Parameter(torch.zeros(self.n_prototypes, self.embed_dim))
        nn.init.normal_(self.centers, mean=0.0, std=0.02)

        init_r = float(max(self.min_radius, min(init_radius, self.max_radius)))
        self.log_radius = nn.Parameter(torch.full((self.n_prototypes,), np.log(init_r), dtype=torch.float32))

        if learnable_tau:
            self.log_tau = nn.Parameter(torch.tensor(np.log(max(tau, 1e-4)), dtype=torch.float32))
        else:
            self.register_buffer("log_tau", torch.tensor(np.log(max(tau, 1e-4)), dtype=torch.float32))
        self.learnable_tau = bool(learnable_tau)

    def radius(self) -> torch.Tensor:
        r = torch.exp(self.log_radius)
        return torch.clamp(r, min=self.min_radius, max=self.max_radius)

    def temperature(self) -> torch.Tensor:
        t = torch.exp(self.log_tau)
        return torch.clamp(t, min=1e-4, max=10.0)

    def forward(self, z: torch.Tensor) -> E2EMultiProtoOutput:
        # z: [B, D]
        if z.ndim != 2:
            raise ValueError(f"Expected 2D embedding tensor, got shape={tuple(z.shape)}")
        centers = self.centers
        dist2 = torch.sum((z[:, None, :] - centers[None, :, :]) ** 2, dim=-1)  # [B, K]
        tau = self.temperature()
        assign_prob = F.softmax(-dist2 / tau, dim=1)
        nearest_dist2, nearest_idx = torch.min(dist2, dim=1)

        r2 = self.radius() ** 2  # [K]
        per_proto_margin = F.relu(dist2 - r2.unsqueeze(0))  # [B, K]
        sample_margin = torch.sum(assign_prob * per_proto_margin, dim=1)  # [B]
        return E2EMultiProtoOutput(
            dist2=dist2,
            assign_prob=assign_prob,
            nearest_idx=nearest_idx,
            nearest_dist2=nearest_dist2,
            sample_margin=sample_margin,
        )

    def separation_loss(self, sep_scale: float = 1.0) -> torch.Tensor:
        if self.n_prototypes <= 1:
            return self.centers.new_tensor(0.0)
        cdist2 = torch.cdist(self.centers, self.centers, p=2) ** 2
        mask = ~torch.eye(self.n_prototypes, dtype=torch.bool, device=cdist2.device)
        if not torch.any(mask):
            return self.centers.new_tensor(0.0)
        s = float(max(1e-6, sep_scale))
        sep = torch.exp(-cdist2 / s)
        return torch.mean(sep[mask])

    def balance_loss(self, assign_prob: torch.Tensor) -> torch.Tensor:
        if assign_prob.numel() == 0:
            return assign_prob.new_tensor(0.0)
        q = torch.mean(assign_prob, dim=0)
        u = torch.full_like(q, 1.0 / float(self.n_prototypes))
        return torch.mean((q - u) ** 2)

    @torch.no_grad()
    def initialize_from_embeddings(
        self,
        embeddings: np.ndarray,
        boundary_quantile: float = 0.9,
        random_state: int = 42,
    ) -> Dict[str, float]:
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] <= 0:
            return {"used_k": 0.0, "radius_mean": float(self.radius().mean().item())}

        uniq = np.unique(np.round(arr, decimals=6), axis=0)
        uniq_count = max(1, int(uniq.shape[0]))
        k = min(self.n_prototypes, arr.shape[0], uniq_count)
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(arr)
        centers = km.cluster_centers_.astype(np.float32)

        device = self.centers.device
        dtype = self.centers.dtype
        self.centers.data[:k] = torch.from_numpy(centers).to(device=device, dtype=dtype)
        if k < self.n_prototypes:
            base = torch.mean(self.centers.data[:k], dim=0, keepdim=True)
            noise = 0.01 * torch.randn(self.n_prototypes - k, self.embed_dim, device=device, dtype=dtype)
            self.centers.data[k:] = base + noise

        q = float(min(0.999, max(0.5, boundary_quantile)))
        radii = np.zeros((k,), dtype=np.float32)
        for i in range(k):
            di = np.linalg.norm(arr[labels == i] - centers[i], axis=1)
            if len(di) == 0:
                radii[i] = float(np.exp(self.log_radius.data[i].cpu().item()))
            else:
                radii[i] = float(np.quantile(di, q))
        if np.any(np.isfinite(radii)):
            rr = np.clip(radii, self.min_radius, self.max_radius)
            self.log_radius.data[:k] = torch.log(torch.from_numpy(rr).to(device=device, dtype=dtype))
        if k < self.n_prototypes:
            fill = float(np.median(radii)) if len(radii) else float(np.exp(self.log_radius.data[0].item()))
            fill = float(min(self.max_radius, max(self.min_radius, fill)))
            self.log_radius.data[k:] = torch.log(torch.full((self.n_prototypes - k,), fill, device=device, dtype=dtype))

        r_mean = float(torch.mean(self.radius()).item())
        return {"used_k": float(k), "radius_mean": r_mean}

