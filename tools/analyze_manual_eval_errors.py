from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd


def normalize_path(path: str) -> str:
    text = str(path or "").replace("\\", "/").strip()
    prefix = "manually-labeled dataset/Real_attack_dataset_format/"
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def parse_fn_set(text: str) -> Set[str]:
    return {tok.strip() for tok in str(text or "").split("|") if tok.strip()}


def primary_fn(text: str) -> str:
    toks = [tok.strip() for tok in str(text or "").split("|") if tok.strip()]
    return toks[0] if toks else "<unknown_fn>"


def family_key(file_path: str) -> str:
    toks = [tok for tok in normalize_path(file_path).split("/") if tok]
    return toks[0] if toks else "<unknown_family>"


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def confusion_dict(df: pd.DataFrame) -> Dict[str, float]:
    tp = int(((df["binary_true"] == 1) & (df["binary_pred"] == 1)).sum())
    fp = int(((df["binary_true"] == 0) & (df["binary_pred"] == 1)).sum())
    fn = int(((df["binary_true"] == 1) & (df["binary_pred"] == 0)).sum())
    tn = int(((df["binary_true"] == 0) & (df["binary_pred"] == 0)).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)
    return {
        "rows": int(len(df)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positive_rows": int(df["binary_true"].sum()),
        "pred_positive_rows": int(df["binary_pred"].sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "mean_binary_score": float(df["binary_score"].mean()) if len(df) else 0.0,
        "max_binary_score": float(df["binary_score"].max()) if len(df) else 0.0,
        "mean_final_risk": float(df["final_risk"].mean()) if len(df) else 0.0,
        "max_final_risk": float(df["final_risk"].max()) if len(df) else 0.0,
    }


def group_confusion(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for key, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: key[i] for i, col in enumerate(group_cols)}
        row.update(confusion_dict(sub))
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["fn", "fp", "tp", "rows"], ascending=[False, False, False, False]).reset_index(drop=True)


def build_attack_function_coverage(
    pred_df: pd.DataFrame,
    label_df: pd.DataFrame,
    line_df: pd.DataFrame,
) -> pd.DataFrame:
    label_use = label_df.copy()
    label_use["category"] = label_use["category"].astype(str)
    label_use["file_norm"] = label_use["file"].map(normalize_path)
    label_use["family"] = label_use["file_norm"].map(family_key)
    label_use = label_use[label_use["category"].str.upper() != "OTHER"].reset_index(drop=True)

    line_lookup = line_df.copy()
    line_lookup["file_norm"] = line_lookup["file"].map(normalize_path)
    line_lookup["line"] = pd.to_numeric(line_lookup["line"], errors="coerce").fillna(-1).astype(int)
    line_score_map = {
        (row.file_norm, int(row.line)): float(row.line_risk)
        for row in line_lookup.itertuples(index=False)
    }

    pred_by_file: Dict[str, pd.DataFrame] = {k: v.copy() for k, v in pred_df.groupby("file_norm", dropna=False)}
    attack_rows: List[Dict[str, object]] = []
    for (file_norm, function), sub in label_use.groupby(["file_norm", "function"], dropna=False):
        sub = sub.copy()
        file_pred = pred_by_file.get(file_norm, pred_df.iloc[0:0].copy())
        if len(file_pred):
            match_mask = file_pred["function_set"].map(lambda s, fn=function: fn in s)
            matched = file_pred.loc[match_mask].copy()
        else:
            matched = file_pred
        label_lines = sorted({int(x) for x in sub["line"].tolist()})
        line_scores = [line_score_map.get((file_norm, ln), np.nan) for ln in label_lines]
        valid_line_scores = [float(x) for x in line_scores if not pd.isna(x)]
        tp = int(((matched["binary_true"] == 1) & (matched["binary_pred"] == 1)).sum()) if len(matched) else 0
        fn = int(((matched["binary_true"] == 1) & (matched["binary_pred"] == 0)).sum()) if len(matched) else 0
        true_rows = int(matched["binary_true"].sum()) if len(matched) else 0
        pred_pos = int(matched["binary_pred"].sum()) if len(matched) else 0
        recall_slices = safe_div(tp, tp + fn)
        if true_rows <= 0:
            status = "no_positive_slices"
        elif tp <= 0:
            status = "zero_hit"
        elif recall_slices < 0.20:
            status = "very_low"
        elif recall_slices < 0.50:
            status = "low"
        else:
            status = "ok"
        attack_rows.append(
            {
                "family": family_key(file_norm),
                "file": file_norm,
                "function": function,
                "categories": "|".join(sorted({str(x) for x in sub["category"].tolist()})),
                "label_line_count": int(len(label_lines)),
                "label_lines": "|".join(str(x) for x in label_lines),
                "matched_slice_rows": int(len(matched)),
                "true_slice_rows": true_rows,
                "pred_positive_rows": pred_pos,
                "tp": tp,
                "fn": fn,
                "slice_recall": recall_slices,
                "pred_positive_ratio": safe_div(pred_pos, max(true_rows, 1)),
                "mean_binary_score": float(matched["binary_score"].mean()) if len(matched) else 0.0,
                "max_binary_score": float(matched["binary_score"].max()) if len(matched) else 0.0,
                "mean_final_risk": float(matched["final_risk"].mean()) if len(matched) else 0.0,
                "max_final_risk": float(matched["final_risk"].max()) if len(matched) else 0.0,
                "line_risk_mean": float(np.mean(valid_line_scores)) if valid_line_scores else np.nan,
                "line_risk_max": float(np.max(valid_line_scores)) if valid_line_scores else np.nan,
                "line_risk_min": float(np.min(valid_line_scores)) if valid_line_scores else np.nan,
                "line_scores_found": int(len(valid_line_scores)),
                "status": status,
            }
        )
    out = pd.DataFrame(attack_rows)
    if out.empty:
        return out
    return out.sort_values(
        ["status", "fn", "slice_recall", "label_line_count", "max_binary_score"],
        ascending=[True, False, True, False, True],
    ).reset_index(drop=True)


def build_fp_primary_function_table(pred_df: pd.DataFrame, attack_fn_keys: Set[Tuple[str, str]]) -> pd.DataFrame:
    norm_df = pred_df.copy()
    norm_df["attack_primary_fn"] = norm_df.apply(
        lambda row: (row["file_norm"], row["primary_fn"]) in attack_fn_keys,
        axis=1,
    )
    norm_df = norm_df[~norm_df["attack_primary_fn"]].copy()
    if norm_df.empty:
        return norm_df
    rows: List[Dict[str, object]] = []
    for (family, file_norm, fn_name), sub in norm_df.groupby(["family", "file_norm", "primary_fn"], dropna=False):
        fp = int(((sub["binary_true"] == 0) & (sub["binary_pred"] == 1)).sum())
        pred_pos = int(sub["binary_pred"].sum())
        if fp <= 0 and pred_pos <= 0:
            continue
        rows.append(
            {
                "family": family,
                "file": file_norm,
                "primary_fn": fn_name,
                "rows": int(len(sub)),
                "fp": fp,
                "pred_positive_rows": pred_pos,
                "fp_rate": safe_div(fp, max(int((sub["binary_true"] == 0).sum()), 1)),
                "mean_binary_score": float(sub["binary_score"].mean()),
                "max_binary_score": float(sub["binary_score"].max()),
                "mean_final_risk": float(sub["final_risk"].mean()),
                "max_final_risk": float(sub["final_risk"].max()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["fp", "pred_positive_rows", "max_binary_score"], ascending=[False, False, False]).reset_index(drop=True)


def build_attack_family_summary(attack_cov: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for family, sub in attack_cov.groupby("family", dropna=False):
        rows.append(
            {
                "family": family,
                "attack_functions": int(len(sub)),
                "zero_hit_functions": int((sub["status"] == "zero_hit").sum()),
                "very_low_functions": int((sub["status"] == "very_low").sum()),
                "low_functions": int((sub["status"] == "low").sum()),
                "no_positive_slice_functions": int((sub["status"] == "no_positive_slices").sum()),
                "label_line_count": int(sub["label_line_count"].sum()),
                "true_slice_rows": int(sub["true_slice_rows"].sum()),
                "tp": int(sub["tp"].sum()),
                "fn": int(sub["fn"].sum()),
                "mean_slice_recall": float(sub["slice_recall"].mean()),
                "mean_line_risk_max": float(sub["line_risk_max"].mean(skipna=True)),
                "mean_binary_score": float(sub["mean_binary_score"].mean()),
                "max_binary_score": float(sub["max_binary_score"].max()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["zero_hit_functions", "very_low_functions", "fn", "mean_slice_recall"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def write_markdown(
    out_path: Path,
    pred_df: pd.DataFrame,
    family_df: pd.DataFrame,
    attack_family_df: pd.DataFrame,
    attack_fn_df: pd.DataFrame,
    fp_fn_df: pd.DataFrame,
) -> None:
    overall = confusion_dict(pred_df)
    top_fn = attack_fn_df[attack_fn_df["status"].isin(["zero_hit", "very_low", "low"])].head(10)
    top_fp = fp_fn_df.head(10)
    lines: List[str] = []
    lines.append("# Manual Eval Error Analysis")
    lines.append("")
    lines.append("## Overall")
    lines.append(
        f"- rows={overall['rows']} tp={overall['tp']} fp={overall['fp']} fn={overall['fn']} tn={overall['tn']}"
    )
    lines.append(
        f"- precision={overall['precision']:.4f} recall={overall['recall']:.4f} f1={overall['f1']:.4f} fpr={overall['fpr']:.4f}"
    )
    lines.append("")
    lines.append("## Top Families By Missed Positive Slices")
    for row in family_df.head(10).itertuples(index=False):
        lines.append(
            f"- {row.family}: fn={row.fn} fp={row.fp} tp={row.tp} recall={row.recall:.4f} f1={row.f1:.4f}"
        )
    lines.append("")
    lines.append("## Top Attack Families Needing Slice Coverage")
    for row in attack_family_df.head(10).itertuples(index=False):
        lines.append(
            f"- {row.family}: zero_hit={row.zero_hit_functions} very_low={row.very_low_functions} "
            f"low={row.low_functions} fn={row.fn} mean_slice_recall={row.mean_slice_recall:.4f}"
        )
    lines.append("")
    lines.append("## Top Attack Functions Needing Coverage")
    for row in top_fn.itertuples(index=False):
        lines.append(
            f"- {row.family} / {row.file} / {row.function}: status={row.status} "
            f"true_rows={row.true_slice_rows} tp={row.tp} fn={row.fn} line_risk_max={row.line_risk_max}"
        )
    lines.append("")
    lines.append("## Top Normal Functions Causing False Positives")
    for row in top_fp.itertuples(index=False):
        lines.append(
            f"- {row.family} / {row.file} / {row.primary_fn}: fp={row.fp} pred_pos={row.pred_positive_rows} "
            f"max_binary_score={row.max_binary_score:.4f}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate family/function TP-FP-FN diagnostics from manual-eval outputs.")
    p.add_argument("--binary-csv", type=str, required=True)
    p.add_argument("--label-csv", type=str, required=True)
    p.add_argument("--line-risk-csv", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--tag", type=str, default="manual_eval")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    binary_csv = Path(args.binary_csv).resolve()
    label_csv = Path(args.label_csv).resolve()
    line_risk_csv = Path(args.line_risk_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(binary_csv)
    pred_df["file_norm"] = pred_df["file"].map(normalize_path)
    pred_df["family"] = pred_df["file_norm"].map(family_key)
    pred_df["function_set"] = pred_df["function_names"].map(parse_fn_set)
    pred_df["primary_fn"] = pred_df["function_names"].map(primary_fn)
    pred_df["binary_true"] = pd.to_numeric(pred_df["binary_true"], errors="coerce").fillna(0).astype(int)
    pred_df["binary_pred"] = pd.to_numeric(pred_df["binary_pred"], errors="coerce").fillna(0).astype(int)
    pred_df["binary_score"] = pd.to_numeric(pred_df["binary_score"], errors="coerce").fillna(0.0).astype(float)
    pred_df["final_risk"] = pd.to_numeric(pred_df["final_risk"], errors="coerce").fillna(0.0).astype(float)

    label_df = pd.read_csv(label_csv)
    line_df = pd.read_csv(line_risk_csv)

    family_df = group_confusion(pred_df, ["family"])
    file_df = group_confusion(pred_df, ["family", "file_norm"])
    attack_fn_df = build_attack_function_coverage(pred_df, label_df, line_df)
    attack_fn_keys = {(str(row.file), str(row.function)) for row in attack_fn_df.itertuples(index=False)}
    fp_fn_df = build_fp_primary_function_table(pred_df, attack_fn_keys)
    attack_family_df = build_attack_family_summary(attack_fn_df)

    family_csv = out_dir / f"{args.tag}_family_metrics.csv"
    file_csv = out_dir / f"{args.tag}_file_metrics.csv"
    attack_fn_csv = out_dir / f"{args.tag}_attack_function_coverage.csv"
    attack_family_csv = out_dir / f"{args.tag}_attack_family_summary.csv"
    fp_fn_csv = out_dir / f"{args.tag}_normal_function_fp.csv"
    md_out = out_dir / f"{args.tag}_report.md"
    json_out = out_dir / f"{args.tag}_summary.json"

    family_df.to_csv(family_csv, index=False)
    file_df.to_csv(file_csv, index=False)
    attack_fn_df.to_csv(attack_fn_csv, index=False)
    attack_family_df.to_csv(attack_family_csv, index=False)
    fp_fn_df.to_csv(fp_fn_csv, index=False)
    write_markdown(md_out, pred_df, family_df, attack_family_df, attack_fn_df, fp_fn_df)

    summary = {
        "binary_csv": str(binary_csv),
        "label_csv": str(label_csv),
        "line_risk_csv": str(line_risk_csv),
        "overall": confusion_dict(pred_df),
        "top_family_fn": family_df.head(10).to_dict(orient="records"),
        "top_attack_family": attack_family_df.head(10).to_dict(orient="records"),
        "top_attack_function": attack_fn_df.head(20).to_dict(orient="records"),
        "top_normal_fp_function": fp_fn_df.head(20).to_dict(orient="records"),
        "outputs": {
            "family_csv": str(family_csv),
            "file_csv": str(file_csv),
            "attack_fn_csv": str(attack_fn_csv),
            "attack_family_csv": str(attack_family_csv),
            "fp_fn_csv": str(fp_fn_csv),
            "report_md": str(md_out),
        },
    }
    json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[analyze_manual_eval_errors] summary={json_out}")
    print(f"[analyze_manual_eval_errors] report={md_out}")


if __name__ == "__main__":
    main()
