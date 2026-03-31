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
        description="Run the dynamic-skeleton experiment with isolated outputs."
    )
    p.add_argument("--config", type=str, default="configs/dynamic_skeleton_seed42.yaml")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_cmd(cmd: List[str], cwd: Path, stage: str, logs: List[Dict[str, Any]]) -> None:
    print(f"[run_dynamic_skeleton_pipeline] [{stage}] RUN: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, cwd=str(cwd), check=True)
    logs.append({"stage": stage, "command": " ".join(cmd), "elapsed_sec": time.time() - t0})


def main() -> None:
    args = parse_args()
    cfg_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    run_args = dict(cfg.get("run_dynamic_skeleton_args", {}))

    py = sys.executable
    out_tag = str(run_args.get("out_tag", "dynamicskel_seed42"))
    logs: List[Dict[str, Any]] = []

    work_root = ROOT / f"outputs/{out_tag}"
    slices = work_root / "slices.jsonl"
    dual_views = work_root / "dual_views.jsonl"
    work_root.mkdir(parents=True, exist_ok=True)

    slice_stats = ROOT / f"outputs/results/state_slice_stats_{out_tag}.json"
    dual_model = ROOT / f"outputs/models/dual_view_{out_tag}.pt"
    dual_risk = ROOT / f"outputs/results/dual_view_risk_scores_{out_tag}.csv"
    dual_summary = ROOT / f"outputs/results/dual_view_summary_{out_tag}.json"
    dual_emb = ROOT / f"outputs/results/dual_view_embeddings_{out_tag}.npz"

    proto_out = ROOT / f"outputs/models/prototypes_{out_tag}.npz"
    energy_out = ROOT / f"outputs/models/energy_head_{out_tag}.pt"
    unsup_risk = ROOT / f"outputs/results/unsup_final_risk_scores_{out_tag}.csv"
    unsup_summary = ROOT / f"outputs/results/unsup_final_summary_{out_tag}.json"

    manual_work = ROOT / f"outputs/manual_eval_artifacts_{out_tag}"
    manual_work.mkdir(parents=True, exist_ok=True)
    manual_summary = ROOT / f"outputs/results/manual_eval_summary_{out_tag}.json"
    manual_risk = ROOT / f"outputs/results/manual_eval_risk_scores_{out_tag}.csv"
    manual_line = ROOT / f"outputs/results/manual_eval_line_risk_scores_{out_tag}.csv"
    manual_binary = ROOT / f"outputs/results/manual_eval_binary_predictions_{out_tag}.csv"
    manual_stats = ROOT / f"outputs/results/manual_state_slice_stats_{out_tag}.json"
    invocation_out = ROOT / f"outputs/results/run_dynamic_skeleton_pipeline_{out_tag}.json"

    tagged_graph_dir = str(run_args.get("graph_dir", "data/graphs/tagged"))
    slice_cmd = [
        py,
        "slicing/state_influence_slice.py",
        "--graph-dir",
        tagged_graph_dir,
        "--output",
        str(slices.relative_to(ROOT)),
        "--hops",
        str(run_args.get("slice_hops", 2)),
        "--forward-hops",
        str(run_args.get("slice_forward_hops", 2)),
        "--backward-hops",
        str(run_args.get("slice_backward_hops", 2)),
        "--min-slice-nodes",
        str(run_args.get("slice_min_nodes", 6)),
        "--mechanism-path-max-len",
        str(run_args.get("slice_mechanism_path_max_len", 4)),
        "--function-fallback-mode",
        str(run_args.get("slice_function_fallback_mode", "none")),
        "--function-fallback-min-nodes",
        str(run_args.get("slice_function_fallback_min_nodes", 3)),
        "--dedup-line-jaccard",
        str(run_args.get("slice_dedup_line_jaccard", 0.95)),
        "--max-slices-per-function",
        str(run_args.get("slice_max_slices_per_function", 120)),
        "--stats-out",
        str(slice_stats.relative_to(ROOT)),
    ]
    if bool(run_args.get("slice_include_contains", True)):
        slice_cmd.append("--include-contains")
    if bool(run_args.get("slice_formal_propagation", True)):
        slice_cmd.append("--formal-propagation")

    dual_build_cmd = [
        py,
        "slicing/dual_view_builder.py",
        "--input",
        str(slices.relative_to(ROOT)),
        "--output",
        str(dual_views.relative_to(ROOT)),
        "--max-slices",
        str(run_args.get("max_slices", 20000)),
        "--skeleton-mode",
        str(run_args.get("dual_skeleton_mode", "dynamic")),
        "--mechanism-path-max-len",
        str(run_args.get("dual_mechanism_path_max_len", 4)),
        "--dynamic-aux-keep-prob",
        str(run_args.get("dual_dynamic_aux_keep_prob", 0.35)),
        "--dynamic-seed",
        str(run_args.get("dual_dynamic_seed", 42)),
    ]

    dual_train_cmd = [
        py,
        "trainers/train_dual_view.py",
        "--input",
        str(dual_views.relative_to(ROOT)),
        "--mode",
        "dual-view",
        "--max-slices",
        str(run_args.get("max_slices", 20000)),
        "--epochs",
        str(run_args.get("epochs_dual", 4)),
        "--seed",
        str(run_args.get("seed", 42)),
        "--hidden-dim",
        str(run_args.get("hidden_dim", 64)),
        "--num-layers",
        str(run_args.get("num_layers", 2)),
        "--device",
        str(run_args.get("device", "cpu")),
        "--model-out",
        str(dual_model.relative_to(ROOT)),
        "--risk-out",
        str(dual_risk.relative_to(ROOT)),
        "--summary-out",
        str(dual_summary.relative_to(ROOT)),
        "--embedding-out",
        str(dual_emb.relative_to(ROOT)),
    ]

    unsup_cmd = [
        py,
        "trainers/train_unsup.py",
        "--dual-view-input",
        str(dual_views.relative_to(ROOT)),
        "--embedding-npz",
        str(dual_emb.relative_to(ROOT)),
        "--embedding-key",
        str(run_args.get("embedding_key", "z_joint")),
        "--prototype-k",
        str(run_args.get("prototype_k", 8)),
        "--family-aware-weight",
        str(run_args.get("family_aware_weight", 0.30)),
        "--adaptive-boundary-refine",
        "--risk-rerank-mode",
        str(run_args.get("risk_rerank_mode", "function_coverage")),
        "--prototype-out",
        str(proto_out.relative_to(ROOT)),
        "--risk-out",
        str(unsup_risk.relative_to(ROOT)),
        "--summary-out",
        str(unsup_summary.relative_to(ROOT)),
        "--energy-head-enable",
        "--energy-head-mode",
        str(run_args.get("energy_head_mode", "proto_hybrid")),
        "--energy-head-loss-mode",
        str(run_args.get("energy_head_loss_mode", "proto_margin")),
        "--energy-head-out",
        str(energy_out.relative_to(ROOT)),
        "--seed",
        str(run_args.get("seed", 42)),
    ]

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
        str(run_args.get("manual_scope", "label-files")),
        "--max-files",
        str(run_args.get("manual_max_files", 0)),
        "--work-dir",
        str(manual_work.relative_to(ROOT)),
        "--dual-model",
        str(dual_model.relative_to(ROOT)),
        "--prototype-npz",
        str(proto_out.relative_to(ROOT)),
        "--device",
        str(run_args.get("device", "cpu")),
        "--hops",
        str(run_args.get("manual_slice_hops", 2)),
        "--slice-forward-hops",
        str(run_args.get("manual_slice_forward_hops", 2)),
        "--slice-backward-hops",
        str(run_args.get("manual_slice_backward_hops", 2)),
        "--min-slice-nodes",
        str(run_args.get("manual_min_slice_nodes", 6)),
        "--slice-mechanism-path-max-len",
        str(run_args.get("manual_slice_mechanism_path_max_len", 4)),
        "--slice-function-fallback-mode",
        str(run_args.get("manual_slice_function_fallback_mode", "none")),
        "--slice-function-fallback-min-nodes",
        str(run_args.get("manual_slice_function_fallback_min_nodes", 3)),
        "--slice-dedup-line-jaccard",
        str(run_args.get("manual_slice_dedup_line_jaccard", 0.95)),
        "--slice-max-slices-per-function",
        str(run_args.get("manual_slice_max_slices_per_function", 120)),
        "--dual-skeleton-mode",
        str(run_args.get("manual_dual_skeleton_mode", "dynamic")),
        "--dual-mechanism-path-max-len",
        str(run_args.get("manual_dual_mechanism_path_max_len", 4)),
        "--dual-dynamic-aux-keep-prob",
        str(run_args.get("manual_dual_dynamic_aux_keep_prob", 0.35)),
        "--dual-dynamic-seed",
        str(run_args.get("manual_dual_dynamic_seed", 42)),
        "--binary-head-method",
        "energy_head",
        "--energy-head-ckpt",
        str(energy_out.relative_to(ROOT)),
        "--triage-gray-threshold",
        str(run_args.get("manual_triage_gray_threshold", 0.28)),
        "--risk-out",
        str(manual_risk.relative_to(ROOT)),
        "--line-risk-out",
        str(manual_line.relative_to(ROOT)),
        "--binary-out",
        str(manual_binary.relative_to(ROOT)),
        "--summary-out",
        str(manual_summary.relative_to(ROOT)),
        "--slice-stats-out",
        str(manual_stats.relative_to(ROOT)),
        "--line-risk-closure-boost",
        str(run_args.get("manual_line_risk_closure_boost", 0.30)),
        "--line-risk-stage2-weight",
        str(run_args.get("manual_line_risk_stage2_weight", 0.15)),
        "--line-risk-stage2-top-functions",
        str(run_args.get("manual_line_risk_stage2_top_functions", 640)),
        "--line-risk-stage2-support-weight",
        str(run_args.get("manual_line_risk_stage2_support_weight", 0.60)),
    ]
    if bool(run_args.get("manual_slice_formal_propagation", True)):
        manual_cmd.append("--slice-formal-propagation")
    if bool(run_args.get("manual_slice_include_contains", False)):
        manual_cmd.append("--slice-include-contains")

    if args.dry_run:
        print("[run_dynamic_skeleton_pipeline] DRY RUN ONLY")
        for cmd in [slice_cmd, dual_build_cmd, dual_train_cmd, unsup_cmd, manual_cmd]:
            print(" ".join(cmd))
        return

    run_cmd(slice_cmd, ROOT, "state_slice", logs)
    run_cmd(dual_build_cmd, ROOT, "dual_view_build_dynamic", logs)
    run_cmd(dual_train_cmd, ROOT, "train_dual_dynamic", logs)
    run_cmd(unsup_cmd, ROOT, "train_unsup_dynamic", logs)
    run_cmd(manual_cmd, ROOT, "manual_eval_dynamic", logs)

    meta = {
        "config_path": str(cfg_path),
        "resolved_run_args": run_args,
        "out_tag": out_tag,
        "outputs": {
            "dual_model": str(dual_model),
            "prototype_out": str(proto_out),
            "energy_out": str(energy_out),
            "manual_summary": str(manual_summary),
        },
        "logs": logs,
    }
    invocation_out.parent.mkdir(parents=True, exist_ok=True)
    invocation_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[run_dynamic_skeleton_pipeline] invocation={invocation_out}")


if __name__ == "__main__":
    main()
