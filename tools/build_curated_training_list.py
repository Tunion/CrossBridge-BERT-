from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


WELL_KNOWN_BRIDGES = [
    "Arbitrum Bridge",
    "Optimism",
    "Polygon Bridge",
    "Wormhole",
    "Hop.Exchange",
    "Synapse Protocol",
    "Multichain",
    "deBridge",
    "Router Protocol",
    "cBridge",
    "Celo Optics Bridge",
    "RenBridge",
    "Ronin Bridge",
    "zkSync Portal Bridge",
    "Boba Gateway bridge",
    "Nomad",
    "LI.FI",
]

DEFAULT_EXCLUDE_BRIDGES = [
    "Ronin Bridge",
    "Nomad",
    "Multichain",
]


LIB_FILENAME_BLACKLIST = {
    "safemath.sol",
    "address.sol",
    "context.sol",
    "ierc20.sol",
    "ownable.sol",
    "safeerc20.sol",
    "erc20.sol",
    "initializable.sol",
}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a curated, de-duplicated, balanced training file list from local non-overlap contracts."
    )
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--input-list", type=str, default="processed/dataset_non_overlap_local.txt")
    p.add_argument("--output-list", type=str, default="processed/dataset_non_overlap_curated_standard.txt")
    p.add_argument("--output-report", type=str, default="processed/dataset_non_overlap_curated_standard_report.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-lines", type=int, default=20)
    p.add_argument("--max-lines", type=int, default=6000)
    p.add_argument("--min-functions", type=int, default=1)
    p.add_argument("--max-per-bridge", type=int, default=180)
    p.add_argument("--target-size", type=int, default=3200)
    p.add_argument(
        "--bridge-whitelist",
        type=str,
        default="|".join(WELL_KNOWN_BRIDGES),
        help="Pipe-separated bridge names. Empty means no bridge filtering.",
    )
    p.add_argument(
        "--exclude-bridge-keywords",
        type=str,
        default="|".join(DEFAULT_EXCLUDE_BRIDGES),
        help="Pipe-separated keywords to drop potentially contaminated bridge groups.",
    )
    p.add_argument("--drop-common-lib-filenames", action="store_true")
    return p.parse_args()


def count_functions(text: str) -> int:
    return text.count("function ")


def line_count(text: str) -> int:
    return len(text.splitlines())


def parse_bridge(rel_path: str) -> str:
    parts = rel_path.replace("\\", "/").split("/")
    # expected: DataSet/<bridge>/...
    if len(parts) >= 2 and parts[0] == "DataSet":
        return parts[1]
    return "UNKNOWN"


def is_whitelisted_bridge(bridge: str, whitelist: List[str]) -> bool:
    if not whitelist:
        return True
    low = bridge.lower()
    return any(w.lower() in low for w in whitelist)


def is_excluded_bridge(bridge: str, excluded: List[str]) -> bool:
    if not excluded:
        return False
    low = bridge.lower()
    return any(x.lower() in low for x in excluded)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    root = Path(args.project_root).resolve()
    in_list = (root / args.input_list).resolve()
    out_list = (root / args.output_list).resolve()
    out_report = (root / args.output_report).resolve()
    out_list.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    whitelist = [x.strip() for x in args.bridge_whitelist.split("|") if x.strip()]
    excluded = [x.strip() for x in args.exclude_bridge_keywords.split("|") if x.strip()]
    rel_paths = [x.strip() for x in in_list.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]

    records: List[Dict[str, object]] = []
    reject_counter = Counter()
    for rel in rel_paths:
        p = (root / rel.replace("/", "\\")).resolve()
        if not p.exists():
            reject_counter["missing_file"] += 1
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            reject_counter["read_error"] += 1
            continue
        lines = line_count(text)
        nfunc = count_functions(text)
        bridge = parse_bridge(rel)
        name_low = p.name.lower()
        if lines < args.min_lines or lines > args.max_lines:
            reject_counter["line_count_filter"] += 1
            continue
        if nfunc < args.min_functions:
            reject_counter["function_count_filter"] += 1
            continue
        if not is_whitelisted_bridge(bridge, whitelist):
            reject_counter["bridge_not_whitelisted"] += 1
            continue
        if is_excluded_bridge(bridge, excluded):
            reject_counter["bridge_excluded"] += 1
            continue
        if args.drop_common_lib_filenames and name_low in LIB_FILENAME_BLACKLIST:
            reject_counter["common_lib_filename"] += 1
            continue
        records.append(
            {
                "rel": rel.replace("\\", "/"),
                "abs": str(p),
                "bridge": bridge,
                "line_count": lines,
                "function_count": nfunc,
                "filename": p.name,
            }
        )

    # MD5 de-dup at content level.
    md5_seen = set()
    unique_records: List[Dict[str, object]] = []
    for rec in records:
        m = md5_file(Path(rec["abs"]))  # type: ignore[arg-type]
        if m in md5_seen:
            continue
        md5_seen.add(m)
        rec["md5"] = m
        unique_records.append(rec)

    # Balanced sampling by bridge.
    by_bridge: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for rec in unique_records:
        by_bridge[str(rec["bridge"])].append(rec)
    for b in by_bridge:
        # Prefer files with richer semantics.
        by_bridge[b].sort(
            key=lambda x: (
                int(x["function_count"]),
                int(x["line_count"]),
            ),
            reverse=True,
        )

    selected: List[Dict[str, object]] = []
    for bridge, rows in by_bridge.items():
        selected.extend(rows[: args.max_per_bridge])
    random.shuffle(selected)
    if args.target_size and args.target_size > 0:
        selected = selected[: args.target_size]

    selected = sorted(selected, key=lambda x: str(x["rel"]))
    out_lines = [str(x["rel"]) for x in selected]
    out_list.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    bridge_counter = Counter(str(x["bridge"]) for x in selected)
    report = {
        "input_total": len(rel_paths),
        "after_rule_filter": len(records),
        "after_md5_dedup": len(unique_records),
        "selected_total": len(selected),
        "reject_counts": dict(reject_counter),
        "params": vars(args),
        "top_bridges": bridge_counter.most_common(30),
        "output_list": str(out_list),
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build_curated_training_list] selected={len(selected)} output={out_list}")
    print(f"[build_curated_training_list] report={out_report}")


if __name__ == "__main__":
    main()
