from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _safe_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def _row_file_key(row: Mapping[str, Any]) -> str:
    return str(
        row.get("relative_source_path")
        or row.get("file")
        or row.get("source_path")
        or row.get("contract_id")
        or "<unknown_file>"
    )


def _family_key_from_file(file_key: str) -> str:
    p = str(file_key).replace("\\", "/").strip().lower()
    toks = [t for t in p.split("/") if t and t not in {".", ".."}]
    if not toks:
        return "<unknown_family>"
    generic = {
        "data",
        "dataset",
        "datasets",
        "manually-labeled dataset",
        "real_attack_dataset_format",
        "manually-labeled",
        "manual",
        "manual_eval_artifacts",
        "contracts",
        "contract",
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
    core = [t for t in toks[:-1] if not t.endswith(".sol") and t not in generic]
    if len(core) >= 2:
        return f"{core[0]}/{core[1]}"
    if len(core) == 1:
        return core[0]
    stem = toks[-1]
    if stem.endswith(".sol"):
        stem = stem[:-4]
    return stem or "<unknown_family>"


def group_key_from_row(row: Mapping[str, Any], group_by: str = "family") -> str:
    mode = str(group_by or "family").strip().lower()
    if mode == "prototype":
        proto = row.get("prototype_id")
        if proto is None or str(proto).strip() == "":
            return "<unknown_prototype>"
        return f"proto:{proto}"
    return _family_key_from_file(_row_file_key(row))


def _strictly_increasing(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).copy()
    if arr.size <= 1:
        return arr
    for i in range(1, arr.size):
        if arr[i] <= arr[i - 1]:
            arr[i] = np.nextafter(arr[i - 1], np.inf)
    return arr


def _quantile_grid(num_bins: int) -> np.ndarray:
    bins = int(max(17, num_bins))
    return np.linspace(0.0, 1.0, num=bins, dtype=np.float32)


def _quantile_values(scores: np.ndarray, grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if arr.size <= 0:
        return np.zeros_like(grid, dtype=np.float32)
    return np.quantile(arr, grid.astype(np.float64), method="linear").astype(np.float32)


def _score_to_quantile(scores: np.ndarray, value_grid: np.ndarray, quant_grid: np.ndarray) -> np.ndarray:
    src = np.asarray(scores, dtype=np.float32).reshape(-1)
    vgrid = _strictly_increasing(np.asarray(value_grid, dtype=np.float32).reshape(-1))
    qgrid = np.asarray(quant_grid, dtype=np.float32).reshape(-1)
    if src.size <= 0 or vgrid.size <= 0 or qgrid.size != vgrid.size:
        return np.zeros_like(src, dtype=np.float32)
    return np.interp(src.astype(np.float64), vgrid, qgrid, left=0.0, right=1.0).astype(np.float32)


def _quantile_to_score(quantiles: np.ndarray, value_grid: np.ndarray, quant_grid: np.ndarray) -> np.ndarray:
    q = np.asarray(quantiles, dtype=np.float32).reshape(-1)
    vgrid = np.asarray(value_grid, dtype=np.float32).reshape(-1)
    qgrid = _strictly_increasing(np.asarray(quant_grid, dtype=np.float32).reshape(-1))
    if q.size <= 0 or vgrid.size <= 0 or qgrid.size != vgrid.size:
        return np.zeros_like(q, dtype=np.float32)
    return np.interp(np.clip(q.astype(np.float64), 0.0, 1.0), qgrid, vgrid).astype(np.float32)


def _fit_scope_accept(row: Mapping[str, Any], fit_scope: str) -> bool:
    scope = str(fit_scope or "normal").strip().lower()
    if scope == "all":
        return True
    status = str(row.get("status", "")).strip().lower()
    if scope in {"normal", "low_risk"}:
        return status == "normal"
    if scope == "boundary":
        return status == "boundary"
    if scope == "high_risk":
        return status == "high_risk"
    return True


def fit_group_score_calibration(
    rows: Sequence[Mapping[str, Any]],
    score_column: str = "energy_head_score",
    score_values: Optional[np.ndarray] = None,
    fit_scope: str = "normal",
    group_by: str = "family",
    min_samples: int = 32,
    shrinkage_tau: float = 16.0,
    global_mix_floor: float = 0.20,
    quantile_bins: int = 257,
) -> Dict[str, Any]:
    used_rows = list(rows)
    n = int(len(used_rows))
    if score_values is not None:
        score_arr = np.asarray(score_values, dtype=np.float32).reshape(-1)
        if score_arr.size != n:
            n = min(n, int(score_arr.size))
            used_rows = used_rows[:n]
            score_arr = score_arr[:n]
    else:
        score_arr = np.asarray([_safe_float(row, score_column, 0.0) for row in used_rows], dtype=np.float32)
    if n <= 0 or score_arr.size <= 0:
        return {
            "enabled": False,
            "missing_reason": "empty_rows",
            "diag": {"enabled": False, "num_rows": 0},
        }

    fit_idx = [i for i, row in enumerate(used_rows[: score_arr.size]) if _fit_scope_accept(row, fit_scope)]
    if not fit_idx:
        return {
            "enabled": False,
            "missing_reason": f"empty_fit_scope:{fit_scope}",
            "diag": {"enabled": False, "num_rows": 0, "fit_scope": str(fit_scope)},
        }
    fit_idx_arr = np.asarray(fit_idx, dtype=np.int64)
    fit_scores = np.asarray(score_arr[fit_idx_arr], dtype=np.float32)
    qgrid = _quantile_grid(quantile_bins)
    global_q = _quantile_values(fit_scores, qgrid)
    group_scores: Dict[str, List[float]] = {}
    for src_idx in fit_idx_arr.tolist():
        gkey = group_key_from_row(used_rows[src_idx], group_by=group_by)
        group_scores.setdefault(gkey, []).append(float(score_arr[src_idx]))
    group_quantiles = {
        gkey: _quantile_values(np.asarray(vals, dtype=np.float32), qgrid)
        for gkey, vals in group_scores.items()
        if vals
    }
    group_counts = {gkey: int(len(vals)) for gkey, vals in group_scores.items()}
    diag = {
        "enabled": True,
        "score_column": str(score_column),
        "fit_scope": str(fit_scope),
        "group_by": str(group_by),
        "num_rows": int(n),
        "fit_rows": int(fit_scores.size),
        "global_count": int(fit_scores.size),
        "group_count": int(len(group_quantiles)),
        "min_samples": int(max(1, min_samples)),
        "shrinkage_tau": float(max(1e-6, shrinkage_tau)),
        "global_mix_floor": float(min(0.95, max(0.0, global_mix_floor))),
        "score_mean": float(np.mean(fit_scores)) if fit_scores.size else 0.0,
        "score_std": float(np.std(fit_scores)) if fit_scores.size else 0.0,
        "small_group_ratio": float(
            np.mean([count < int(max(1, min_samples)) for count in group_counts.values()]) if group_counts else 0.0
        ),
    }
    return {
        "enabled": True,
        "score_column": str(score_column),
        "fit_scope": str(fit_scope),
        "group_by": str(group_by),
        "min_samples": int(max(1, min_samples)),
        "shrinkage_tau": float(max(1e-6, shrinkage_tau)),
        "global_mix_floor": float(min(0.95, max(0.0, global_mix_floor))),
        "quantile_bins": int(max(17, quantile_bins)),
        "quantile_grid": qgrid.astype(np.float32),
        "global_quantiles": global_q.astype(np.float32),
        "global_count": int(fit_scores.size),
        "group_counts": group_counts,
        "group_quantiles": group_quantiles,
        "diag": diag,
    }


def fit_group_score_calibration_from_csv(
    csv_path: Path,
    score_column: str = "energy_head_score",
    fit_scope: str = "normal",
    group_by: str = "family",
    min_samples: int = 32,
    shrinkage_tau: float = 16.0,
    global_mix_floor: float = 0.20,
    quantile_bins: int = 257,
) -> Dict[str, Any]:
    path = Path(csv_path)
    if not path.exists():
        return {
            "enabled": False,
            "missing_reason": f"missing_source:{path}",
            "diag": {"enabled": False, "source": str(path), "missing_reason": f"missing_source:{path}"},
        }
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if score_column not in (reader.fieldnames or []):
            return {
                "enabled": False,
                "missing_reason": f"missing_column:{score_column}",
                "diag": {
                    "enabled": False,
                    "source": str(path),
                    "missing_reason": f"missing_column:{score_column}",
                },
            }
        for row in reader:
            rows.append(dict(row))
    out = fit_group_score_calibration(
        rows,
        score_column=score_column,
        fit_scope=fit_scope,
        group_by=group_by,
        min_samples=min_samples,
        shrinkage_tau=shrinkage_tau,
        global_mix_floor=global_mix_floor,
        quantile_bins=quantile_bins,
    )
    out.setdefault("diag", {})["source"] = str(path)
    return out


def apply_group_score_calibration(
    rows: Sequence[Mapping[str, Any]],
    base_scores: np.ndarray,
    calibration: Optional[Mapping[str, Any]],
    alpha: float = 1.0,
    min_samples: Optional[int] = None,
    shrinkage_tau: Optional[float] = None,
    global_mix_floor: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    base = np.asarray(base_scores, dtype=np.float32).reshape(-1)
    n = int(base.size)
    if calibration is None or not bool(calibration.get("enabled", False)) or n <= 0:
        return base.astype(np.float32), {"enabled": False}

    qgrid = np.asarray(calibration.get("quantile_grid", []), dtype=np.float32).reshape(-1)
    global_qvals = np.asarray(calibration.get("global_quantiles", []), dtype=np.float32).reshape(-1)
    if qgrid.size <= 1 or global_qvals.size != qgrid.size:
        return base.astype(np.float32), {"enabled": False, "missing_reason": "invalid_quantile_grid"}

    group_by = str(calibration.get("group_by", "family"))
    group_quantiles = dict(calibration.get("group_quantiles", {}) or {})
    group_counts = dict(calibration.get("group_counts", {}) or {})
    min_n = int(max(1, min_samples if min_samples is not None else calibration.get("min_samples", 32)))
    tau = float(max(1e-6, shrinkage_tau if shrinkage_tau is not None else calibration.get("shrinkage_tau", 16.0)))
    g_floor = float(
        min(0.95, max(0.0, global_mix_floor if global_mix_floor is not None else calibration.get("global_mix_floor", 0.20)))
    )
    blend_alpha = float(min(1.0, max(0.0, alpha)))

    q_global = _score_to_quantile(base, global_qvals, qgrid)
    q_mix = q_global.copy()
    group_to_idx: Dict[str, List[int]] = {}
    for i, row in enumerate(list(rows)[:n]):
        group_to_idx.setdefault(group_key_from_row(row, group_by=group_by), []).append(i)

    local_weight_arr = np.zeros((n,), dtype=np.float32)
    adjusted_rows = 0
    fallback_rows = 0
    for gkey, idx_list in group_to_idx.items():
        idx = np.asarray(idx_list, dtype=np.int64)
        qvals = np.asarray(group_quantiles.get(gkey, []), dtype=np.float32).reshape(-1)
        count = int(group_counts.get(gkey, 0))
        if idx.size <= 0 or count < min_n or qvals.size != qgrid.size:
            fallback_rows += int(idx.size)
            continue
        q_local = _score_to_quantile(base[idx], qvals, qgrid)
        local_weight = float(min(1.0 - g_floor, count / (count + tau)))
        q_mix[idx] = (1.0 - local_weight) * q_global[idx] + local_weight * q_local
        local_weight_arr[idx] = local_weight
        adjusted_rows += int(idx.size)

    cal_score = _quantile_to_score(q_mix, global_qvals, qgrid)
    out = ((1.0 - blend_alpha) * base + blend_alpha * cal_score).astype(np.float32)
    diag = {
        "enabled": True,
        "group_by": group_by,
        "alpha": float(blend_alpha),
        "min_samples": int(min_n),
        "shrinkage_tau": float(tau),
        "global_mix_floor": float(g_floor),
        "group_count": int(len(group_quantiles)),
        "adjusted_rows": int(adjusted_rows),
        "fallback_rows": int(fallback_rows),
        "local_weight_mean": float(np.mean(local_weight_arr)) if local_weight_arr.size else 0.0,
        "local_weight_std": float(np.std(local_weight_arr)) if local_weight_arr.size else 0.0,
        "score_mean_before": float(np.mean(base)) if base.size else 0.0,
        "score_mean_after": float(np.mean(out)) if out.size else 0.0,
        "score_std_before": float(np.std(base)) if base.size else 0.0,
        "score_std_after": float(np.std(out)) if out.size else 0.0,
    }
    return out.astype(np.float32), diag
