from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.io_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a compact case-study report for top risky slices.")
    p.add_argument("--risk-csv", type=str, default="outputs/results/unsup_final_risk_scores.csv")
    p.add_argument("--dual-view-input", type=str, default="data/slices/dual_views.jsonl")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--out", type=str, default="outputs/results/case_study_topk.md")
    return p.parse_args()


def load_risk(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    score_col = "final_risk" if "final_risk" in rows[0] else "risk_score"
    rows = sorted(rows, key=lambda x: float(x[score_col]), reverse=True)
    for r in rows:
        r["_score_col"] = score_col
    return rows


def main() -> None:
    args = parse_args()
    risk_rows = load_risk(Path(args.risk_csv).resolve())
    view_rows = load_jsonl(Path(args.dual_view_input).resolve())
    view_map = {r["slice_id"]: r for r in view_rows}

    top = risk_rows[: min(args.top_k, len(risk_rows))]
    md: List[str] = []
    md.append("# Case Study: Top Risky Slices")
    md.append("")
    md.append(f"- Top K: {len(top)}")
    md.append(f"- Risk source: `{args.risk_csv}`")
    md.append("")

    for i, r in enumerate(top, start=1):
        sid = r["slice_id"]
        view = view_map.get(sid, {})
        full_meta = view.get("full_graph", {}).get("meta", {})
        sk_meta = view.get("skeleton_graph", {}).get("meta", {})
        role_counter = Counter()
        for n in view.get("full_graph", {}).get("nodes", []):
            for role in n.get("roles", []) or []:
                role_counter[role] += 1
        md.append(f"## {i}. slice_id={sid}")
        md.append(f"- contract_id: `{r.get('contract_id', '')}`")
        md.append(f"- relative_source_path: `{r.get('relative_source_path', '')}`")
        md.append(f"- function_names: `{r.get('function_names', '')}`")
        md.append(f"- risk_score: `{float(r[r['_score_col']]):.6f}`")
        if "status" in r:
            md.append(f"- status: `{r['status']}`")
        md.append(
            f"- full_view nodes/edges: `{full_meta.get('node_count', 0)}/{full_meta.get('edge_count', 0)}`"
        )
        md.append(
            f"- skeleton_view nodes/edges: `{sk_meta.get('node_count', 0)}/{sk_meta.get('edge_count', 0)}`"
        )
        md.append(f"- role_counts: `{dict(role_counter)}`")
        md.append("")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[case_study] report={out_path}")


if __name__ == "__main__":
    main()
