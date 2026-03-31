from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from models.prototype_head import MultiPrototypeHead


def coarse_family_key_from_row(row: Dict[str, Any]) -> str:
    file_key = str(
        row.get("relative_source_path")
        or row.get("file")
        or row.get("source_path")
        or row.get("contract_id")
        or "<unknown_family>"
    ).replace("\\", "/")
    toks = [t.strip().lower() for t in file_key.split("/") if t.strip()]
    if not toks:
        return "<unknown_family>"
    generic = {
        "dataset",
        "contract",
        "contracts",
        "manually-labeled dataset",
        "real_attack_dataset_format",
        "src",
        "source",
        "sources",
        "sol",
        "bridge",
        "bridges",
        "project",
        "projects",
        "mainnet",
        "testnet",
    }
    for tok in toks[:-1]:
        if tok in generic:
            continue
        if tok.startswith("0x") and len(tok) >= 8:
            continue
        stem = os.path.splitext(tok)[0]
        if stem and stem not in generic:
            return stem
    stem = os.path.splitext(toks[-1])[0]
    return stem or "<unknown_family>"


def _local_mix_weight(count: int, min_samples: int, shrinkage_tau: float, global_mix_floor: float) -> float:
    if count < int(max(4, min_samples)):
        return 0.0
    raw = float(count) / float(count + max(1.0, shrinkage_tau))
    max_local = float(max(0.0, 1.0 - min(1.0, max(0.0, global_mix_floor))))
    return float(max(0.0, min(max_local, raw * max_local)))


def _centroid_match_confidence(distances: np.ndarray, selected_idx: int, conf_floor: float) -> float:
    d = np.asarray(distances, dtype=np.float32).reshape(-1)
    if len(d) <= 1 or selected_idx < 0 or selected_idx >= len(d):
        return 1.0
    d_sel = float(d[selected_idx])
    alt = np.delete(d, int(selected_idx))
    if alt.size == 0:
        return 1.0
    d_alt = float(np.min(alt))
    sep = max(0.0, d_alt - d_sel) / max(1e-6, max(d_alt, d_sel))
    floor = float(min(1.0, max(0.0, conf_floor)))
    return float(floor + (1.0 - floor) * min(1.0, max(0.0, sep)))


def fit_hierarchical_local_prototypes(
    rows: Sequence[Dict[str, Any]],
    embeddings: np.ndarray,
    boundary_quantile: float,
    local_k: int,
    min_samples: int,
    random_state: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    emb = np.asarray(embeddings, dtype=np.float32)
    fam_to_idx: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        fam_to_idx.setdefault(coarse_family_key_from_row(row), []).append(i)

    bank: Dict[str, Dict[str, Any]] = {}
    local_family_count = 0
    fallback_rows = 0
    family_sizes: Dict[str, int] = {}
    for fam, idx_list in fam_to_idx.items():
        family_sizes[fam] = int(len(idx_list))
        if len(idx_list) < int(max(4, min_samples)):
            fallback_rows += int(len(idx_list))
            continue
        sub = emb[np.asarray(idx_list, dtype=np.int64)]
        proto = MultiPrototypeHead(
            n_prototypes=int(max(1, min(local_k, len(idx_list)))),
            boundary_quantile=float(boundary_quantile),
            random_state=int(random_state),
        )
        fit = proto.fit(sub)
        bank[fam] = {
            "centers": fit.centers.astype(np.float32),
            "radii": fit.radii.astype(np.float32),
            "count": int(len(idx_list)),
            "family_centroid": np.mean(sub, axis=0).astype(np.float32),
        }
        local_family_count += 1
    diag = {
        "enabled": True,
        "local_k": int(local_k),
        "min_samples": int(max(4, min_samples)),
        "num_families": int(len(fam_to_idx)),
        "local_family_count": int(local_family_count),
        "fallback_rows": int(fallback_rows),
        "family_sizes": {k: int(v) for k, v in family_sizes.items()},
    }
    return bank, diag


def score_hierarchical_prototypes(
    rows: Sequence[Dict[str, Any]],
    embeddings: np.ndarray,
    global_assign: np.ndarray,
    global_nearest: np.ndarray,
    global_margin: np.ndarray,
    local_bank: Dict[str, Dict[str, Any]],
    min_samples: int,
    shrinkage_tau: float,
    global_mix_floor: float,
    match_conf_floor: float,
) -> Dict[str, np.ndarray]:
    n = len(rows)
    local_assign = np.full((n,), -1, dtype=np.int64)
    local_nearest = np.asarray(global_nearest, dtype=np.float32).copy()
    local_margin = np.asarray(global_margin, dtype=np.float32).copy()
    local_weight = np.zeros((n,), dtype=np.float32)
    combined_nearest = np.asarray(global_nearest, dtype=np.float32).copy()
    combined_margin = np.asarray(global_margin, dtype=np.float32).copy()
    family_available = np.zeros((n,), dtype=np.float32)
    local_route_exact = np.zeros((n,), dtype=np.float32)
    local_route_fallback_centroid = np.zeros((n,), dtype=np.float32)
    local_match_conf = np.ones((n,), dtype=np.float32)

    emb = np.asarray(embeddings, dtype=np.float32)
    bank_keys = list(local_bank.keys())
    bank_centroids = None
    if bank_keys:
        bank_centroids = np.stack(
            [
                np.asarray(local_bank[k].get("family_centroid", np.mean(local_bank[k]["centers"], axis=0)), dtype=np.float32)
                for k in bank_keys
            ],
            axis=0,
        )
    for i, row in enumerate(rows):
        fam = coarse_family_key_from_row(row)
        bank = local_bank.get(fam)
        selected_bank_idx = -1
        if bank is not None:
            local_route_exact[i] = 1.0
            selected_bank_idx = bank_keys.index(fam) if fam in bank_keys else -1
        elif bank_centroids is not None and len(bank_keys) > 0:
            d_fam = np.linalg.norm(bank_centroids - emb[i][None, :], axis=1)
            selected_bank_idx = int(np.argmin(d_fam))
            bank = local_bank[bank_keys[selected_bank_idx]]
            local_route_fallback_centroid[i] = 1.0
            local_match_conf[i] = _centroid_match_confidence(d_fam, selected_bank_idx, match_conf_floor)
        if not bank:
            continue
        if bank_centroids is not None and selected_bank_idx >= 0 and local_route_exact[i] > 0:
            d_fam = np.linalg.norm(bank_centroids - emb[i][None, :], axis=1)
            local_match_conf[i] = _centroid_match_confidence(d_fam, selected_bank_idx, match_conf_floor)
        centers = np.asarray(bank["centers"], dtype=np.float32)
        radii = np.asarray(bank["radii"], dtype=np.float32)
        d = np.linalg.norm(emb[i][None, :] - centers, axis=1)
        lid = int(np.argmin(d))
        lnear = float(d[lid])
        lmargin = float(lnear - radii[lid])
        w = _local_mix_weight(
            count=int(bank.get("count", 0)),
            min_samples=int(min_samples),
            shrinkage_tau=float(shrinkage_tau),
            global_mix_floor=float(global_mix_floor),
        )
        local_assign[i] = lid
        local_nearest[i] = lnear
        local_margin[i] = lmargin
        local_weight[i] = float(w * local_match_conf[i])
        family_available[i] = 1.0
        combined_nearest[i] = float((1.0 - local_weight[i]) * float(global_nearest[i]) + local_weight[i] * lnear)
        combined_margin[i] = float((1.0 - local_weight[i]) * float(global_margin[i]) + local_weight[i] * lmargin)

    return {
        "global_assign": np.asarray(global_assign, dtype=np.int64),
        "global_nearest": np.asarray(global_nearest, dtype=np.float32),
        "global_margin": np.asarray(global_margin, dtype=np.float32),
        "local_assign": local_assign,
        "local_nearest": local_nearest,
        "local_margin": local_margin,
        "local_weight": local_weight,
        "local_match_conf": local_match_conf,
        "family_available": family_available,
        "local_route_exact": local_route_exact,
        "local_route_fallback_centroid": local_route_fallback_centroid,
        "combined_assign": np.asarray(global_assign, dtype=np.int64),
        "combined_nearest": combined_nearest,
        "combined_margin": combined_margin,
    }


def pack_local_bank(local_bank: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
    families = sorted(local_bank.keys())
    offsets = [0]
    centers_list = []
    radii_list = []
    counts = []
    for fam in families:
        item = local_bank[fam]
        centers = np.asarray(item["centers"], dtype=np.float32)
        radii = np.asarray(item["radii"], dtype=np.float32)
        centers_list.append(centers)
        radii_list.append(radii)
        offsets.append(offsets[-1] + int(len(radii)))
        counts.append(int(item.get("count", 0)))
    family_centroids = (
        np.stack(
            [np.asarray(local_bank[fam].get("family_centroid", np.mean(local_bank[fam]["centers"], axis=0)), dtype=np.float32) for fam in families],
            axis=0,
        ).astype(np.float32)
        if families
        else np.zeros((0, 0), dtype=np.float32)
    )
    all_centers = np.concatenate(centers_list, axis=0).astype(np.float32) if centers_list else np.zeros((0, 0), dtype=np.float32)
    all_radii = np.concatenate(radii_list, axis=0).astype(np.float32) if radii_list else np.zeros((0,), dtype=np.float32)
    return {
        "hier_local_families": np.asarray(families),
        "hier_local_offsets": np.asarray(offsets, dtype=np.int32),
        "hier_local_centers": all_centers,
        "hier_local_radii": all_radii,
        "hier_local_counts": np.asarray(counts, dtype=np.int32),
        "hier_local_family_centroids": family_centroids,
    }


def unpack_local_bank(proto_arr: Any) -> Dict[str, Dict[str, Any]]:
    if "hier_local_families" not in proto_arr or "hier_local_offsets" not in proto_arr:
        return {}
    families = [str(x) for x in np.asarray(proto_arr["hier_local_families"]).tolist()]
    offsets = np.asarray(proto_arr["hier_local_offsets"], dtype=np.int32).reshape(-1)
    centers = np.asarray(proto_arr["hier_local_centers"], dtype=np.float32)
    radii = np.asarray(proto_arr["hier_local_radii"], dtype=np.float32).reshape(-1)
    counts = np.asarray(proto_arr["hier_local_counts"], dtype=np.int32).reshape(-1) if "hier_local_counts" in proto_arr else np.zeros((len(families),), dtype=np.int32)
    family_centroids = (
        np.asarray(proto_arr["hier_local_family_centroids"], dtype=np.float32)
        if "hier_local_family_centroids" in proto_arr
        else None
    )
    bank: Dict[str, Dict[str, Any]] = {}
    for i, fam in enumerate(families):
        lo = int(offsets[i])
        hi = int(offsets[i + 1]) if i + 1 < len(offsets) else int(len(radii))
        bank[fam] = {
            "centers": centers[lo:hi].astype(np.float32),
            "radii": radii[lo:hi].astype(np.float32),
            "count": int(counts[i]) if i < len(counts) else 0,
            "family_centroid": (
                family_centroids[i].astype(np.float32)
                if family_centroids is not None and i < len(family_centroids)
                else np.mean(centers[lo:hi], axis=0).astype(np.float32)
            ),
        }
    return bank
