from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate seen-family vs unseen-family recall from manual-eval risk csv."
    )
    p.add_argument("--risk-csv", type=str, default="outputs/results/manual_eval_risk_scores.csv")
    p.add_argument("--label-csv", type=str, default="processed/label_standard_local.csv")
    p.add_argument(
        "--train-list",
        type=str,
        default="processed/dataset_non_overlap_local.txt",
        help="Training list used to define seen families.",
    )
    p.add_argument("--topk", type=str, default="20,50,100,200")
    p.add_argument("--line-window", type=int, default=2, help="Line tolerance for row-level match.")
    p.add_argument("--out", type=str, default="outputs/results/generalization_split_summary.json")
    return p.parse_args()


def parse_topk(s: str) -> List[int]:
    vals: List[int] = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    vals = sorted(set(v for v in vals if v > 0))
    return vals or [200]


def _canonical_fn(x: str) -> str:
    y = str(x or "").strip().lower()
    return y if y else "<unknown_function>"


def _function_key_set(fn_pipe: str) -> Set[str]:
    out: Set[str] = set()
    for x in str(fn_pipe or "").split("|"):
        z = _canonical_fn(x)
        if z:
            out.add(z)
    return out or {"<unknown_function>"}


def _family_key_from_path(path_text: str) -> str:
    s = str(path_text or "").replace("\\", "/").strip()
    if not s:
        return "<unknown_family>"
    toks = [t for t in s.split("/") if t]
    if not toks:
        return "<unknown_family>"
    generic = {
        "users",
        "desktop",
        "myidea",
        "fse24-smartaxe-main",
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
        "sol",
        "bridge",
        "bridges",
        "project",
        "projects",
        "mainnet",
        "testnet",
    }
    core = [t.lower() for t in toks[:-1] if not t.lower().endswith(".sol") and t.lower() not in generic]
    if len(core) >= 2:
        return f"{core[0]}/{core[1]}"
    if len(core) == 1:
        return core[0]
    stem = toks[-1].lower()
    if stem.endswith(".sol"):
        stem = stem[:-4]
    return stem or "<unknown_family>"


def load_seen_families(train_list_path: Path) -> Set[str]:
    if not train_list_path.exists():
        return set()
    fams: Set[str] = set()
    for line in train_list_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        x = line.strip()
        if not x:
            continue
        fams.add(_family_key_from_path(x))
    return fams


def load_label_sets(label_csv: Path) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, int]]]:
    df = pd.read_csv(label_csv)
    file_fn_pos: Set[Tuple[str, str]] = set()
    file_line_pos: Set[Tuple[str, int]] = set()
    for _, r in df.iterrows():
        f = str(r.get("file", "")).strip()
        if not f:
            continue
        f = f.replace("\\", "/")
        fn = _canonical_fn(str(r.get("function", "")))
        file_fn_pos.add((f, fn))
        try:
            ln = int(r.get("line"))
            file_line_pos.add((f, ln))
        except Exception:
            continue
    return file_fn_pos, file_line_pos


def _line_candidates(line_pipe: str) -> Set[int]:
    out: Set[int] = set()
    for x in str(line_pipe or "").split("|"):
        x = x.strip()
        if not x:
            continue
        try:
            out.add(int(x))
        except Exception:
            continue
    return out


def filefn_recall_at_k(
    rows: List[Dict[str, Any]],
    pos: Set[Tuple[str, str]],
    seen_families: Set[str],
    k: int,
) -> Dict[str, Any]:
    head = rows[: max(0, min(k, len(rows)))]
    pred: Set[Tuple[str, str]] = set()
    for r in head:
        f = str(r.get("file", "")).replace("\\", "/").strip()
        if not f:
            continue
        for fn in _function_key_set(str(r.get("function_names", ""))):
            pred.add((f, fn))

    hit = pos.intersection(pred)
    seen_pos = {(f, fn) for (f, fn) in pos if _family_key_from_path(f) in seen_families}
    unseen_pos = pos.difference(seen_pos)
    seen_hit = hit.intersection(seen_pos)
    unseen_hit = hit.intersection(unseen_pos)

    def rec(a: int, b: int) -> float:
        return float(a) / float(max(1, b))

    return {
        "top_k": int(k),
        "strict_file_function_recall": rec(len(hit), len(pos)),
        "strict_file_function_seen_recall": rec(len(seen_hit), len(seen_pos)),
        "strict_file_function_unseen_recall": rec(len(unseen_hit), len(unseen_pos)),
        "strict_file_function_hit": int(len(hit)),
        "strict_file_function_seen_hit": int(len(seen_hit)),
        "strict_file_function_unseen_hit": int(len(unseen_hit)),
        "strict_file_function_pos": int(len(pos)),
        "strict_file_function_seen_pos": int(len(seen_pos)),
        "strict_file_function_unseen_pos": int(len(unseen_pos)),
    }


def row_recall_at_k(
    rows: List[Dict[str, Any]],
    pos: Set[Tuple[str, int]],
    seen_families: Set[str],
    k: int,
    line_window: int = 2,
) -> Dict[str, Any]:
    file_to_pos: Dict[str, List[int]] = {}
    for f, ln in pos:
        file_to_pos.setdefault(f, []).append(int(ln))
    for f in list(file_to_pos.keys()):
        file_to_pos[f] = sorted(set(file_to_pos[f]))

    matched: Set[Tuple[str, int]] = set()
    head = rows[: max(0, min(k, len(rows)))]
    w = max(0, int(line_window))
    for r in head:
        f = str(r.get("file", "")).replace("\\", "/").strip()
        if not f:
            continue
        cand = _line_candidates(str(r.get("line_candidates", "")))
        pos_lines = file_to_pos.get(f, [])
        if not cand or not pos_lines:
            continue
        for c in cand:
            for p in pos_lines:
                if abs(int(c) - int(p)) <= w:
                    matched.add((f, int(p)))

    seen_pos = {(f, ln) for (f, ln) in pos if _family_key_from_path(f) in seen_families}
    unseen_pos = pos.difference(seen_pos)
    seen_hit = matched.intersection(seen_pos)
    unseen_hit = matched.intersection(unseen_pos)

    def rec(a: int, b: int) -> float:
        return float(a) / float(max(1, b))

    return {
        "top_k": int(k),
        "window_row_recall": rec(len(matched), len(pos)),
        "window_row_seen_recall": rec(len(seen_hit), len(seen_pos)),
        "window_row_unseen_recall": rec(len(unseen_hit), len(unseen_pos)),
        "window_row_hit": int(len(matched)),
        "window_row_seen_hit": int(len(seen_hit)),
        "window_row_unseen_hit": int(len(unseen_hit)),
        "window_row_pos": int(len(pos)),
        "window_row_seen_pos": int(len(seen_pos)),
        "window_row_unseen_pos": int(len(unseen_pos)),
    }


def main() -> None:
    args = parse_args()
    risk_csv = (ROOT / args.risk_csv).resolve()
    label_csv = (ROOT / args.label_csv).resolve()
    train_list = (ROOT / args.train_list).resolve()
    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    risk_df = pd.read_csv(risk_csv)
    risk_df = risk_df.sort_values(by="final_risk", ascending=False)
    rows = risk_df.to_dict("records")
    topk = parse_topk(args.topk)
    seen_fams = load_seen_families(train_list)
    pos_file_fn, pos_row = load_label_sets(label_csv)

    filefn_metrics = [filefn_recall_at_k(rows, pos_file_fn, seen_fams, k=x) for x in topk]
    row_metrics = [row_recall_at_k(rows, pos_row, seen_fams, k=x, line_window=args.line_window) for x in topk]
    summary = {
        "risk_csv": str(risk_csv),
        "label_csv": str(label_csv),
        "train_list": str(train_list),
        "seen_family_count": int(len(seen_fams)),
        "line_window": int(args.line_window),
        "topk": topk,
        "file_function_metrics": filefn_metrics,
        "row_metrics": row_metrics,
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[generalization_split] out={out_path}")


if __name__ == "__main__":
    main()
