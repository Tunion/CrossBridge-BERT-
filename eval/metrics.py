from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluation metrics for unsupervised cross-chain risk ranking.")
    p.add_argument("--pred", type=str, required=True, help="Prediction csv (risk ranked).")
    p.add_argument("--label-file", type=str, default="processed/label_standard_local.csv")
    p.add_argument("--label-mode", type=str, choices=["row88", "file_function"], default="row88")
    p.add_argument("--score-col", type=str, default="final_risk")
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--out", type=str, default="outputs/results/metrics.json")
    return p.parse_args()


def normalize_path_token(x: str) -> str:
    return x.replace("\\", "/").lower()


def build_label_set(df: pd.DataFrame, mode: str) -> Set[Tuple[str, str, int]]:
    keys: Set[Tuple[str, str, int]] = set()
    for _, r in df.iterrows():
        file_ = str(r["file"])
        fn = "" if pd.isna(r["function"]) else str(r["function"])
        ln = int(r["line"])
        if mode == "row88":
            keys.add((normalize_path_token(file_), fn.lower(), ln))
        else:
            keys.add((normalize_path_token(file_), fn.lower(), -1))
    return keys


def pred_to_candidate_keys(pred_row: Dict[str, str], mode: str) -> Set[Tuple[str, str, int]]:
    out: Set[Tuple[str, str, int]] = set()
    if "file" in pred_row and "function" in pred_row:
        file_ = normalize_path_token(pred_row["file"])
        fn = pred_row.get("function", "").strip().lower()
        ln = int(pred_row.get("line", -1)) if mode == "row88" else -1
        out.add((file_, fn, ln))
        return out

    rel = normalize_path_token(pred_row.get("relative_source_path", ""))
    basename = rel.split("/")[-1] if rel else ""
    fn_names = [x.strip().lower() for x in pred_row.get("function_names", "").split("|") if x.strip()]
    if not fn_names:
        fn_names = [""]
    for fn in fn_names:
        key_path = basename if basename else rel
        out.add((key_path, fn, -1 if mode == "file_function" else -1))
    return out


def match_labels(
    labels: Set[Tuple[str, str, int]],
    preds: List[Dict[str, str]],
    mode: str,
) -> Tuple[int, List[Tuple[str, str, int]]]:
    hits: Set[Tuple[str, str, int]] = set()
    # Build file-name fallback map for weak matching.
    labels_by_basename: Dict[str, Set[Tuple[str, str, int]]] = {}
    for k in labels:
        base = k[0].split("/")[-1]
        labels_by_basename.setdefault(base, set()).add(k)

    for row in preds:
        cands = pred_to_candidate_keys(row, mode=mode)
        for c in cands:
            if c in labels:
                hits.add(c)
                continue
            # Fallback: if prediction only has basename, match by basename+function.
            base = c[0].split("/")[-1]
            for target in labels_by_basename.get(base, set()):
                if mode == "row88":
                    if c[1] and c[1] == target[1]:
                        hits.add(target)
                else:
                    if c[1] and c[1] == target[1]:
                        hits.add(target)
    return len(hits), sorted(hits)


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred).resolve()
    label_path = Path(args.label_file).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_df = pd.read_csv(label_path)
    label_set = build_label_set(label_df, mode=args.label_mode)

    with pred_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.score_col not in rows[0]:
        if "risk_score" in rows[0]:
            args.score_col = "risk_score"
        else:
            raise ValueError(f"score column {args.score_col} not found in prediction csv")

    rows = sorted(rows, key=lambda x: float(x[args.score_col]), reverse=True)
    topk = rows[: min(args.top_k, len(rows))]
    hit_count, hit_items = match_labels(label_set, topk, mode=args.label_mode)
    denom = len(label_set)
    recall = hit_count / denom if denom else 0.0
    precision = hit_count / len(topk) if topk else 0.0

    result = {
        "label_mode": args.label_mode,
        "num_labels": denom,
        "num_predictions": len(rows),
        "top_k": len(topk),
        "hit_count": hit_count,
        "recall_at_k": recall,
        "precision_at_k": precision,
        "hit_items": [
            {"file": f, "function": fn, "line": ln if ln >= 0 else None}
            for f, fn, ln in hit_items
        ],
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[metrics] result={out_path}")
    print(f"[metrics] recall@{len(topk)}={recall:.4f} precision@{len(topk)}={precision:.4f}")


if __name__ == "__main__":
    main()
