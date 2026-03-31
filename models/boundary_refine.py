from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.min(x)) if len(x) else 0.0
    hi = float(np.max(x)) if len(x) else 1.0
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo + 1e-8)


def local_density_score(embeddings: np.ndarray, k: int = 10) -> np.ndarray:
    if len(embeddings) == 0:
        return np.zeros((0,), dtype=np.float32)
    k = max(2, min(k, len(embeddings)))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(embeddings)
    dists, _ = nn.kneighbors(embeddings)
    # dists[:,0] is self (0)
    mean_dist = np.mean(dists[:, 1:], axis=1)
    density = 1.0 / (mean_dist + 1e-6)
    return _normalize(density)


def perturbation_instability(
    embeddings: np.ndarray,
    centers: np.ndarray,
    n_perturb: int = 5,
    noise_std: float = 0.02,
    random_state: int = 42,
) -> np.ndarray:
    if len(embeddings) == 0:
        return np.zeros((0,), dtype=np.float32)
    rng = np.random.default_rng(random_state)
    snapshots = []
    for _ in range(n_perturb):
        noise = rng.normal(0.0, noise_std, size=embeddings.shape).astype(np.float32)
        pert = embeddings + noise
        dmat = np.linalg.norm(pert[:, None, :] - centers[None, :, :], axis=2)
        nearest = dmat.min(axis=1)
        snapshots.append(nearest)
    arr = np.stack(snapshots, axis=0)
    std = np.std(arr, axis=0)
    return _normalize(std.astype(np.float32))


@dataclass
class RefineWeights:
    w_proto: float = 0.45
    w_view: float = 0.25
    w_density: float = 0.15
    w_stability: float = 0.15


class BoundaryRefiner:
    def __init__(self, weights: RefineWeights | None = None):
        self.weights = weights or RefineWeights()

    def refine(
        self,
        proto_margin: np.ndarray,
        view_gap: np.ndarray,
        embeddings: np.ndarray,
        centers: np.ndarray,
        density_k: int = 10,
        n_perturb: int = 5,
        noise_std: float = 0.02,
        random_state: int = 42,
        adaptive_local: bool = False,
        adaptive_focus_quantile: float = 0.65,
        adaptive_min_scale: float = 0.30,
        adaptive_max_scale: float = 1.00,
        adaptive_outside_scale: float = 0.00,
    ) -> Dict[str, np.ndarray]:
        proto_score = _normalize(np.maximum(proto_margin, 0.0))
        view_score = _normalize(view_gap)
        density = local_density_score(embeddings, k=density_k)
        density_risk = 1.0 - density
        instability = perturbation_instability(
            embeddings=embeddings,
            centers=centers,
            n_perturb=n_perturb,
            noise_std=noise_std,
            random_state=random_state,
        )
        legacy_final = (
            self.weights.w_proto * proto_score
            + self.weights.w_view * view_score
            + self.weights.w_density * density_risk
            + self.weights.w_stability * instability
        ).astype(np.float32)
        boundary_proximity = (1.0 - _normalize(np.abs(np.asarray(proto_margin, dtype=np.float32)))).astype(np.float32)
        correction = (
            self.weights.w_view * view_score
            + self.weights.w_density * density_risk
            + self.weights.w_stability * instability
        ).astype(np.float32)

        if adaptive_local and len(proto_score):
            q = float(np.clip(adaptive_focus_quantile, 0.0, 1.0))
            if q <= 0.0:
                focus_mask = np.ones_like(proto_score, dtype=bool)
            elif q >= 1.0:
                focus_mask = np.zeros_like(proto_score, dtype=bool)
            else:
                thr = float(np.quantile(boundary_proximity, q))
                focus_mask = boundary_proximity >= thr

            min_s = float(max(0.0, adaptive_min_scale))
            max_s = float(max(min_s, adaptive_max_scale))
            outside_s = float(np.clip(adaptive_outside_scale, 0.0, max_s))
            uncertainty = _normalize(0.60 * view_score + 0.20 * density_risk + 0.20 * instability)
            driver = _normalize(0.70 * boundary_proximity + 0.30 * uncertainty)
            local_scale = (min_s + (max_s - min_s) * driver).astype(np.float32)
            local_scale = np.where(focus_mask, local_scale, np.full_like(local_scale, outside_s)).astype(np.float32)
            final = (self.weights.w_proto * proto_score + local_scale * correction).astype(np.float32)
        else:
            focus_mask = np.ones_like(proto_score, dtype=bool)
            local_scale = np.ones_like(proto_score, dtype=np.float32)
            final = legacy_final

        final = np.clip(final, 0.0, 1.0).astype(np.float32)
        q90 = float(np.quantile(final, 0.9)) if len(final) else 0.0
        q75 = float(np.quantile(final, 0.75)) if len(final) else 0.0
        status = np.array(
            [
                "high_risk" if x >= q90 else ("boundary" if x >= q75 else "normal")
                for x in final
            ]
        )
        return {
            "final_risk": final,
            "legacy_final_risk": legacy_final.astype(np.float32),
            "proto_score": proto_score,
            "view_score": view_score,
            "density_risk": density_risk.astype(np.float32),
            "instability": instability.astype(np.float32),
            "boundary_proximity": boundary_proximity.astype(np.float32),
            "adaptive_local_scale": local_scale.astype(np.float32),
            "adaptive_focus_mask": focus_mask.astype(np.bool_),
            "status": status,
        }
