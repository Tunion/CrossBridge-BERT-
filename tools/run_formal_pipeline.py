from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run formal project pipeline from YAML config (role-aware directional slicing + formal skeleton)."
    )
    p.add_argument("--config", type=str, default="configs/formal_v1.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summary-out", type=str, default="")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=-1, help="Override seed; -1 means config value.")
    return p.parse_args()


def flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


NEGATIVE_BOOL_FLAGS = {
    "build_curated_standard": "--no-build-curated-standard",
    "curated_drop_common_libs": "--no-curated-drop-common-libs",
    "run_manual_eval": "--no-run-manual-eval",
    "slice_formal_propagation": "--no-slice-formal-propagation",
    "manual_slice_formal_propagation": "--no-manual-slice-formal-propagation",
    "unsup_adaptive_boundary_refine": "--no-unsup-adaptive-boundary-refine",
    "manual_adaptive_boundary_refine": "--no-manual-adaptive-boundary-refine",
}


def build_cmd(run_args: Dict[str, Any]) -> list[str]:
    cmd = [sys.executable, "run_pipeline.py"]
    for k, v in run_args.items():
        if v is None:
            continue
        f = flag_name(k)
        if isinstance(v, bool):
            if v:
                cmd.append(f)
            elif k in NEGATIVE_BOOL_FLAGS:
                cmd.append(NEGATIVE_BOOL_FLAGS[k])
            continue
        cmd.extend([f, str(v)])
    return cmd


def main() -> None:
    args = parse_args()
    cfg_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    run_args = dict(cfg.get("run_pipeline_args", {}))

    if args.summary_out:
        run_args["summary_out"] = args.summary_out
    if args.quick:
        run_args["quick"] = True
    if args.seed >= 0:
        run_args["seed"] = int(args.seed)

    cmd = build_cmd(run_args)
    print("[run_formal_pipeline] RUN:", " ".join(cmd))
    if args.dry_run:
        return

    t0 = time.time()
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    elapsed = float(time.time() - t0)

    meta = {
        "config_path": str(cfg_path),
        "resolved_run_args": run_args,
        "elapsed_sec": elapsed,
    }
    meta_out = ROOT / "outputs/results/formal_pipeline_invocation.json"
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[run_formal_pipeline] invocation={meta_out}")


if __name__ == "__main__":
    main()
