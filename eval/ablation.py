from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation helper: compare full/skeleton/dual risk rankings.")
    p.add_argument("--full", type=str, required=True, help="CSV from full-only mode.")
    p.add_argument("--skeleton", type=str, required=True, help="CSV from skeleton-only mode.")
    p.add_argument("--dual", type=str, required=True, help="CSV from dual-view mode.")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--out", type=str, default="outputs/results/ablation_summary.json")
    return p.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    score_col = "risk_score" if "risk_score" in rows[0] else ("final_risk" if "final_risk" in rows[0] else None)
    if score_col is None:
        raise ValueError(f"Cannot find score column in {path}")
    rows = sorted(rows, key=lambda x: float(x[score_col]), reverse=True)
    for r in rows:
        r["_score_col"] = score_col
    return rows


def top_ids(rows: List[Dict[str, str]], k: int) -> List[str]:
    return [r.get("slice_id", "") for r in rows[: min(k, len(rows))]]


def main() -> None:
    args = parse_args()
    full_rows = load_rows(Path(args.full).resolve())
    skel_rows = load_rows(Path(args.skeleton).resolve())
    dual_rows = load_rows(Path(args.dual).resolve())

    top_full = top_ids(full_rows, args.top_k)
    top_skel = top_ids(skel_rows, args.top_k)
    top_dual = top_ids(dual_rows, args.top_k)

    set_full, set_skel, set_dual = set(top_full), set(top_skel), set(top_dual)
    summary = {
        "top_k": args.top_k,
        "overlap": {
            "full_vs_skeleton": len(set_full & set_skel),
            "full_vs_dual": len(set_full & set_dual),
            "skeleton_vs_dual": len(set_skel & set_dual),
            "all_three": len(set_full & set_skel & set_dual),
        },
        "mean_scores": {
            "full": float(sum(float(r[r["_score_col"]]) for r in full_rows[: args.top_k]) / max(1, min(args.top_k, len(full_rows)))),
            "skeleton": float(sum(float(r[r["_score_col"]]) for r in skel_rows[: args.top_k]) / max(1, min(args.top_k, len(skel_rows)))),
            "dual": float(sum(float(r[r["_score_col"]]) for r in dual_rows[: args.top_k]) / max(1, min(args.top_k, len(dual_rows)))),
        },
    }
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ablation] summary={out_path}")


if __name__ == "__main__":
    main()

