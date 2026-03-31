from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate manual-label dataset protocol and export a compact report.")
    p.add_argument("--label-csv", type=str, default="processed/label_standard_local.csv")
    p.add_argument("--manual-root", type=str, default="manually-labeled dataset/Real_attack_dataset_format")
    p.add_argument("--out", type=str, default="outputs/results/dataset_protocol_report.json")
    return p.parse_args()


def normalize_file_key(s: str) -> str:
    return str(s).replace("\\", "/").strip()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    label_csv = (root / args.label_csv).resolve()
    manual_root = (root / args.manual_root).resolve()
    out = (root / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(label_csv)
    if {"file", "function", "line"}.difference(df.columns):
        raise RuntimeError(f"label csv missing required columns: {label_csv}")

    df["file_key"] = df["file"].map(normalize_file_key)
    df["function_key"] = df["function"].fillna("").astype(str).str.strip().str.lower()

    unique_file_function = (
        df.loc[df["function_key"] != "", ["file_key", "function_key"]]
        .drop_duplicates()
        .shape[0]
    )
    manual_files = sorted(str(p.relative_to(manual_root)).replace("\\", "/") for p in manual_root.rglob("*.sol"))
    manual_file_set = set(manual_files)
    labeled_file_set = set(df["file_key"].unique().tolist())

    report = {
        "label_csv": str(label_csv),
        "manual_root": str(manual_root),
        "rows_total": int(len(df)),
        "category_counts": {
            str(k): int(v) for k, v in df["category"].value_counts(dropna=False).to_dict().items()
        }
        if "category" in df.columns
        else {},
        "unique_files_in_label": int(df["file_key"].nunique()),
        "unique_file_function_in_label": int(unique_file_function),
        "manual_sol_files_total": int(len(manual_files)),
        "labeled_files_missing_on_disk": sorted(x for x in labeled_file_set if x not in manual_file_set),
        "manual_files_without_labels_count": int(sum(1 for x in manual_file_set if x not in labeled_file_set)),
        "expected_protocol": {
            "gold_row88": 88,
            "gold_unique_file_function": 75,
            "gold_unique_files": 39,
        },
        "protocol_match": {
            "gold_row88_ok": bool(len(df) == 88),
            "gold_unique_file_function_ok": bool(unique_file_function == 75),
            "gold_unique_files_ok": bool(df["file_key"].nunique() == 39),
        },
    }

    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
