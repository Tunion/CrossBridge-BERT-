from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the score-guided prototype-energy experiment with isolated outputs."
    )
    p.add_argument("--config", type=str, default="configs/scoreguided_seed42.yaml")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_cmd(cmd: List[str], cwd: Path, stage: str, logs: List[Dict[str, Any]]) -> None:
    print(f"[run_scoreguided_pipeline] [{stage}] RUN: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, cwd=str(cwd), check=True)
    logs.append({"stage": stage, "command": " ".join(cmd), "elapsed_sec": time.time() - t0})


def main() -> None:
    args = parse_args()
    cfg_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    run_args = dict(cfg.get("run_scoreguided_args", {}))

    py = sys.executable
    out_tag = str(run_args.get("out_tag", "scoreguided_seed42"))
    logs: List[Dict[str, Any]] = []

    proto_out = f"outputs/models/prototypes_{out_tag}.npz"
    energy_out = f"outputs/models/energy_head_{out_tag}.pt"
    unsup_risk = f"outputs/results/unsup_final_risk_scores_{out_tag}.csv"
    unsup_summary = f"outputs/results/unsup_final_summary_{out_tag}.json"

    manual_risk = f"outputs/results/manual_eval_risk_scores_{out_tag}.csv"
    manual_binary = f"outputs/results/manual_eval_binary_predictions_{out_tag}.csv"
    manual_summary = f"outputs/results/manual_eval_summary_{out_tag}.json"
    manual_line = f"outputs/results/manual_eval_line_risk_scores_{out_tag}.csv"
    manual_stats = f"outputs/results/manual_state_slice_stats_{out_tag}.json"
    manual_work = f"outputs/manual_eval_artifacts_{out_tag}"
    invocation_out = ROOT / f"outputs/results/run_scoreguided_pipeline_{out_tag}.json"

    unsup_cmd = [
        py,
        "trainers/train_unsup.py",
        "--dual-view-input",
        str(run_args.get("dual_input", "data/slices/dual_views.jsonl")),
        "--embedding-npz",
        str(run_args.get("embedding_npz", "outputs/results/dual_view_embeddings.npz")),
        "--embedding-key",
        str(run_args.get("embedding_key", "z_joint")),
        "--risk-out",
        unsup_risk,
        "--summary-out",
        unsup_summary,
        "--prototype-out",
        proto_out,
        "--energy-head-enable",
        "--energy-head-mode",
        str(run_args.get("energy_mode", "proto_hybrid")),
        "--energy-head-loss-mode",
        str(run_args.get("energy_loss_mode", "proto_margin")),
        "--energy-head-out",
        energy_out,
        "--energy-head-score-teacher-enable",
        "--energy-head-score-teacher-weight",
        str(run_args.get("score_teacher_weight", 0.35)),
        "--energy-head-score-fit-scope",
        str(run_args.get("score_fit_scope", "low_risk")),
        "--energy-head-score-random-state",
        str(run_args.get("score_random_state", 42)),
        "--seed",
        str(run_args.get("seed", 42)),
    ]
    if bool(run_args.get("energy_stable_train", True)):
        unsup_cmd.append("--energy-head-stable-train")
    else:
        unsup_cmd.append("--no-energy-head-stable-train")
    if bool(run_args.get("energy_family_aware_anchor", False)):
        unsup_cmd.append("--energy-head-family-aware-anchor")
        unsup_cmd.extend(
            [
                "--energy-head-anchor-family-min-samples",
                str(run_args.get("energy_anchor_family_min_samples", 96)),
            ]
        )
    else:
        unsup_cmd.append("--no-energy-head-family-aware-anchor")

    manual_cmd = [
        py,
        "eval/evaluate_manual_set.py",
        "--project-root",
        ".",
        "--manual-root",
        str(run_args.get("manual_root", "manually-labeled dataset/Real_attack_dataset_format")),
        "--label-csv",
        str(run_args.get("label_csv", "processed/label_standard_local.csv")),
        "--scope",
        str(run_args.get("scope", "label-files")),
        "--max-files",
        "0",
        "--work-dir",
        manual_work,
        "--dual-model",
        str(run_args.get("dual_model", "outputs/models/dual_view.pt")),
        "--prototype-npz",
        proto_out,
        "--score-head-method",
        "iforest",
        "--score-head-fit-scope",
        str(run_args.get("score_fit_scope", "low_risk")),
        "--score-head-random-state",
        str(run_args.get("score_random_state", 42)),
        "--binary-head-method",
        "energy_head",
        "--energy-head-ckpt",
        energy_out,
        "--triage-gray-threshold",
        str(run_args.get("triage_gray_threshold", 0.28)),
        "--summary-out",
        manual_summary,
        "--risk-out",
        manual_risk,
        "--line-risk-out",
        manual_line,
        "--binary-out",
        manual_binary,
        "--slice-stats-out",
        manual_stats,
    ]

    if args.dry_run:
        print("[run_scoreguided_pipeline] DRY RUN ONLY")
        for cmd in [unsup_cmd, manual_cmd]:
            print(" ".join(cmd))
        return

    run_cmd(unsup_cmd, ROOT, "train_unsup_scoreguided", logs)
    run_cmd(manual_cmd, ROOT, "manual_eval_scoreguided", logs)

    meta = {
        "config_path": str(cfg_path),
        "resolved_run_args": run_args,
        "out_tag": out_tag,
        "outputs": {
            "prototype_out": proto_out,
            "energy_out": energy_out,
            "manual_summary": manual_summary,
        },
        "logs": logs,
    }
    invocation_out.parent.mkdir(parents=True, exist_ok=True)
    invocation_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[run_scoreguided_pipeline] invocation={invocation_out}")


if __name__ == "__main__":
    main()
