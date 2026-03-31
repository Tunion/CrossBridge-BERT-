from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class PrototypeFitResult:
    centers: np.ndarray
    labels: np.ndarray
    radii: np.ndarray


class MultiPrototypeHead:
    """Unsupervised normal pattern modeling with multiple prototypes."""

    def __init__(self, n_prototypes: int = 8, boundary_quantile: float = 0.9, random_state: int = 42):
        self.n_prototypes = int(max(1, n_prototypes))
        self.boundary_quantile = float(boundary_quantile)
        self.random_state = random_state
        self.kmeans: KMeans | None = None
        self.centers: np.ndarray | None = None
        self.radii: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray) -> PrototypeFitResult:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        # Prevent degenerate KMeans settings when embeddings collapse.
        uniq = np.unique(np.round(embeddings, decimals=6), axis=0)
        uniq_count = max(1, int(uniq.shape[0]))
        k = min(self.n_prototypes, max(1, embeddings.shape[0]), uniq_count)
        self.kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = self.kmeans.fit_predict(embeddings)
        centers = self.kmeans.cluster_centers_.astype(np.float32)
        dists = np.linalg.norm(embeddings - centers[labels], axis=1)
        radii = np.zeros(k, dtype=np.float32)
        for i in range(k):
            di = dists[labels == i]
            if len(di) == 0:
                radii[i] = 0.0
            else:
                radii[i] = float(np.quantile(di, self.boundary_quantile))
        self.centers = centers
        self.radii = radii
        return PrototypeFitResult(centers=centers, labels=labels, radii=radii)

    def _check_ready(self) -> None:
        if self.centers is None or self.radii is None:
            raise RuntimeError("Prototype head is not fitted yet.")

    def nearest(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._check_ready()
        assert self.centers is not None
        dmat = np.linalg.norm(embeddings[:, None, :] - self.centers[None, :, :], axis=2)
        assign = np.argmin(dmat, axis=1)
        nearest = dmat[np.arange(len(embeddings)), assign]
        return assign.astype(np.int64), nearest.astype(np.float32)

    def boundary_margin(self, embeddings: np.ndarray) -> np.ndarray:
        self._check_ready()
        assign, nearest = self.nearest(embeddings)
        assert self.radii is not None
        radii = self.radii[assign]
        # >0 means outside boundary.
        return (nearest - radii).astype(np.float32)

    def score(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        assign, nearest = self.nearest(embeddings)
        margin = self.boundary_margin(embeddings)
        return {"assign": assign, "nearest_distance": nearest, "boundary_margin": margin}
