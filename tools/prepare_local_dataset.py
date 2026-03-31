from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


OLD_PREFIX = "/mnt/3.6TB-DATA/zhy/zhy_dir/data/FSE24-SmartAxe-main/FSE24-SmartAxe-main/"


@dataclass
class Config:
    root: Path
    dataset_dir: Path
    manual_dir: Path
    label_csv: Path
    provided_overlap_csv: Path
    provided_non_overlap_txt: Path
    output_dir: Path


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_sol_files(base: Path) -> Iterable[Path]:
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".sol":
            yield p


def to_posix_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def map_old_path_to_local(root: Path, old_path: str) -> Optional[Path]:
    if not isinstance(old_path, str) or not old_path.startswith(OLD_PREFIX):
        return None
    rel = old_path[len(OLD_PREFIX) :]
    return (root / rel.replace("/", "\\")).resolve()


def load_md5_index(paths: List[Path]) -> Dict[str, List[Path]]:
    md5_map: Dict[str, List[Path]] = defaultdict(list)
    for p in paths:
        md5_map[md5_file(p)].append(p)
    return md5_map


def write_overlap_csv(
    path: Path,
    root: Path,
    rows: List[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_rel_path",
                "manual_rel_path",
                "md5",
                "dataset_abs_path",
                "manual_abs_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset_rel_path": to_posix_relative(root, row["dataset_path"]),
                    "manual_rel_path": to_posix_relative(root, row["manual_path"]),
                    "md5": row["md5"],
                    "dataset_abs_path": str(row["dataset_path"].resolve()),
                    "manual_abs_path": str(row["manual_path"].resolve()),
                }
            )


def write_non_overlap_txt(path: Path, root: Path, non_overlap_paths: List[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [to_posix_relative(root, p) for p in non_overlap_paths]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    cfg = Config(
        root=root,
        dataset_dir=root / "DataSet",
        manual_dir=root / "manually-labeled dataset" / "Real_attack_dataset_format",
        label_csv=root / "label_standard.csv",
        provided_overlap_csv=root / "dataset_manual_overlap.csv",
        provided_non_overlap_txt=root / "dataset_non_overlap.txt",
        output_dir=root / "processed",
    )

    dataset_files = sorted(iter_sol_files(cfg.dataset_dir))
    manual_files = sorted(iter_sol_files(cfg.manual_dir))

    manual_md5_map = load_md5_index(manual_files)

    overlap_rows: List[dict] = []
    non_overlap_paths: List[Path] = []
    for p in dataset_files:
        p_md5 = md5_file(p)
        if p_md5 in manual_md5_map:
            manual_match = sorted(manual_md5_map[p_md5], key=lambda x: str(x))[0]
            overlap_rows.append({"dataset_path": p, "manual_path": manual_match, "md5": p_md5})
        else:
            non_overlap_paths.append(p)

    overlap_rows = sorted(overlap_rows, key=lambda x: str(x["dataset_path"]))
    non_overlap_paths = sorted(non_overlap_paths, key=lambda x: str(x))

    overlap_out = cfg.output_dir / "dataset_manual_overlap_local.csv"
    non_overlap_out = cfg.output_dir / "dataset_non_overlap_local.txt"
    write_overlap_csv(overlap_out, cfg.root, overlap_rows)
    write_non_overlap_txt(non_overlap_out, cfg.root, non_overlap_paths)

    # Localized label table.
    label = pd.read_csv(cfg.label_csv)
    local_abs_paths: List[Optional[str]] = []
    local_rel_paths: List[Optional[str]] = []
    local_exists: List[bool] = []
    for p in label["full_path"]:
        q = map_old_path_to_local(cfg.root, p)
        if q is None:
            local_abs_paths.append(None)
            local_rel_paths.append(None)
            local_exists.append(False)
        else:
            local_abs_paths.append(str(q))
            local_rel_paths.append(to_posix_relative(cfg.root, q) if q.exists() else None)
            local_exists.append(q.exists())
    label["local_rel_path"] = local_rel_paths
    label["local_abs_path"] = local_abs_paths
    label["local_exists"] = local_exists
    label_out = cfg.output_dir / "label_standard_local.csv"
    label.to_csv(label_out, index=False, encoding="utf-8")

    # Recall-key table using README's "file+function" alignment rule.
    recall_keys = (
        label[["file", "function", "local_rel_path", "local_abs_path", "local_exists"]]
        .drop_duplicates(subset=["file", "function"])
        .sort_values(["file", "function"])
    )
    recall_key_out = cfg.output_dir / "label_recall_keys_file_function.csv"
    recall_keys.to_csv(recall_key_out, index=False, encoding="utf-8")

    # Duplicate groups on (file, function), useful for multi-label/overload analysis.
    dup_groups = (
        label.groupby(["file", "function"], dropna=False)
        .agg(
            rows=("line", "size"),
            unique_lines=("line", "nunique"),
            categories=("category", lambda x: "|".join(sorted({str(i) for i in x}))),
            lines=("line", lambda x: "|".join(str(i) for i in x)),
        )
        .reset_index()
    )
    dup_groups = dup_groups[dup_groups["rows"] > 1].sort_values(
        ["rows", "file", "function"], ascending=[False, True, True]
    )
    dup_group_out = cfg.output_dir / "label_duplicate_groups.csv"
    dup_groups.to_csv(dup_group_out, index=False, encoding="utf-8")

    # Compare with provided overlap/non-overlap lists.
    provided_overlap = pd.read_csv(cfg.provided_overlap_csv)
    provided_overlap_mapped = set()
    for p in provided_overlap["dataset_path"]:
        q = map_old_path_to_local(cfg.root, p)
        if q is not None and q.exists():
            provided_overlap_mapped.add(str(q))

    provided_non_overlap = [
        x.strip()
        for x in cfg.provided_non_overlap_txt.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    provided_non_overlap_mapped_existing = set()
    for p in provided_non_overlap:
        q = map_old_path_to_local(cfg.root, p)
        if q is not None and q.exists():
            provided_non_overlap_mapped_existing.add(str(q))

    local_overlap_set = {str(x["dataset_path"].resolve()) for x in overlap_rows}
    local_non_overlap_set = {str(p.resolve()) for p in non_overlap_paths}
    non_overlap_local_only = sorted(local_non_overlap_set - provided_non_overlap_mapped_existing)

    bridge_counter = Counter(
        to_posix_relative(cfg.root / "DataSet", p).split("/")[0] for p in dataset_files
    )
    label_category_counter = Counter(label["category"].tolist())
    label_attack_counter = Counter(str(x).split("/")[0] for x in label["file"].tolist())

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(cfg.root),
        "counts": {
            "dataset_bridge_dirs": len([d for d in cfg.dataset_dir.iterdir() if d.is_dir()]),
            "dataset_sol_files": len(dataset_files),
            "manual_attack_dirs_top_level": len(
                [
                    d
                    for d in cfg.manual_dir.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ]
            ),
            "manual_sol_files": len(manual_files),
            "label_rows": int(len(label)),
            "label_unique_file_function_keys": int(recall_keys.shape[0]),
            "label_local_exists_true": int(sum(local_exists)),
            "overlap_local_rows": len(overlap_rows),
            "non_overlap_local_rows": len(non_overlap_paths),
            "provided_overlap_rows": int(len(provided_overlap)),
            "provided_overlap_mapped_existing": len(provided_overlap_mapped),
            "provided_non_overlap_rows": len(provided_non_overlap),
            "provided_non_overlap_mapped_existing": len(provided_non_overlap_mapped_existing),
        },
        "consistency_checks": {
            "overlap_exact_match_with_provided_after_mapping": (
                local_overlap_set == provided_overlap_mapped
            ),
            "provided_non_overlap_is_subset_of_local_non_overlap_after_mapping": (
                provided_non_overlap_mapped_existing.issubset(local_non_overlap_set)
            ),
            "non_overlap_local_only_count_vs_provided_mapped_existing": len(non_overlap_local_only),
        },
        "label_distribution": {
            "category_counts": dict(label_category_counter),
            "top_attack_prefix_counts": dict(label_attack_counter),
        },
        "dataset_distribution": {
            "top_20_bridges_by_sol_count": bridge_counter.most_common(20),
        },
        "differences": {
            "non_overlap_local_only_files": [
                Path(p).resolve().relative_to(cfg.root.resolve()).as_posix()
                for p in non_overlap_local_only
            ]
        },
        "outputs": {
            "dataset_manual_overlap_local_csv": str(overlap_out),
            "dataset_non_overlap_local_txt": str(non_overlap_out),
            "label_standard_local_csv": str(label_out),
            "label_recall_keys_file_function_csv": str(recall_key_out),
            "label_duplicate_groups_csv": str(dup_group_out),
        },
    }

    report_out = cfg.output_dir / "dataset_profile.json"
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Short markdown summary for quick reading.
    summary = []
    summary.append("# Dataset Processing Summary (Local)")
    summary.append("")
    summary.append(f"- Generated at (UTC): `{report['generated_at_utc']}`")
    summary.append(f"- DataSet `.sol` files: **{report['counts']['dataset_sol_files']}**")
    summary.append(f"- Manual set `.sol` files: **{report['counts']['manual_sol_files']}**")
    summary.append(f"- Local overlap (MD5): **{report['counts']['overlap_local_rows']}**")
    summary.append(f"- Local non-overlap (MD5): **{report['counts']['non_overlap_local_rows']}**")
    summary.append(
        "- Provided `dataset_manual_overlap.csv` matches local recomputation after path mapping: "
        f"**{report['consistency_checks']['overlap_exact_match_with_provided_after_mapping']}**"
    )
    summary.append(
        "- Provided `dataset_non_overlap.txt` mapped-existing rows: "
        f"**{report['counts']['provided_non_overlap_mapped_existing']}** "
        f"(local recomputed non-overlap has **{report['counts']['non_overlap_local_rows']}** rows)"
    )
    summary.append("")
    summary.append("## Produced Files")
    summary.append("- `processed/dataset_manual_overlap_local.csv`")
    summary.append("- `processed/dataset_non_overlap_local.txt`")
    summary.append("- `processed/label_standard_local.csv`")
    summary.append("- `processed/label_recall_keys_file_function.csv`")
    summary.append("- `processed/label_duplicate_groups.csv`")
    summary.append("- `processed/dataset_profile.json`")
    summary.append("")
    summary.append("## Note")
    summary.append("- For local training/evaluation, prefer `processed/dataset_non_overlap_local.txt`.")
    summary.append("- Existing `dataset_non_overlap.txt` contains historical absolute paths from another machine.")
    (cfg.output_dir / "PROCESSING_SUMMARY.md").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print("Generated files:")
    print(overlap_out)
    print(non_overlap_out)
    print(label_out)
    print(recall_key_out)
    print(dup_group_out)
    print(report_out)
    print(cfg.output_dir / "PROCESSING_SUMMARY.md")


if __name__ == "__main__":
    main()
