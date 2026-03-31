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
        description="Run the implicit-mechanism-edge experiment in isolated output paths."
    )
    p.add_argument("--config", type=str, default="configs/implicit_mechanism_seed42.yaml")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_cmd(cmd: List[str], cwd: Path, stage: str, logs: List[Dict[str, Any]]) -> None:
    print(f"[run_implicit_mechanism_pipeline] [{stage}] RUN: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, cwd=str(cwd), check=True)
    logs.append({"stage": stage, "command": " ".join(cmd), "elapsed_sec": time.time() - t0})


def main() -> None:
    args = parse_args()
    cfg_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    run_args = dict(cfg.get("run_implicit_mechanism_args", {}))

    py = sys.executable
    out_tag = str(run_args.get("out_tag", "implicitmech"))
    logs: List[Dict[str, Any]] = []

    work_root = ROOT / f"outputs/{out_tag}"
    graph_full = work_root / "graphs_full"
    graph_tagged = work_root / "graphs_tagged"
    normalized = work_root / "contracts_normalized.jsonl"
    graph_index = work_root / "graph_index.jsonl"
    graph_report = ROOT / f"outputs/results/graph_semantic_coverage_{out_tag}.json"
    slice_stats = ROOT / f"outputs/results/state_slice_stats_{out_tag}.json"
    slices = work_root / "slices.jsonl"
    dual_views = work_root / "dual_views.jsonl"

    dual_model = ROOT / f"outputs/models/dual_view_{out_tag}.pt"
    dual_risk = ROOT / f"outputs/results/dual_view_risk_scores_{out_tag}.csv"
    dual_summary = ROOT / f"outputs/results/dual_view_summary_{out_tag}.json"
    dual_emb = ROOT / f"outputs/results/dual_view_embeddings_{out_tag}.npz"

    proto_out = ROOT / f"outputs/models/prototypes_{out_tag}.npz"
    energy_out = ROOT / f"outputs/models/energy_head_{out_tag}.pt"
    unsup_risk = ROOT / f"outputs/results/unsup_final_risk_scores_{out_tag}.csv"
    unsup_summary = ROOT / f"outputs/results/unsup_final_summary_{out_tag}.json"

    manual_work = ROOT / f"outputs/manual_eval_artifacts_{out_tag}"
    manual_summary = ROOT / f"outputs/results/manual_eval_summary_{out_tag}.json"
    manual_risk = ROOT / f"outputs/results/manual_eval_risk_scores_{out_tag}.csv"
    manual_line = ROOT / f"outputs/results/manual_eval_line_risk_scores_{out_tag}.csv"
    manual_binary = ROOT / f"outputs/results/manual_eval_binary_predictions_{out_tag}.csv"
    manual_stats = ROOT / f"outputs/results/manual_state_slice_stats_{out_tag}.json"
    invocation_out = ROOT / f"outputs/results/run_implicit_mechanism_pipeline_{out_tag}.json"

    for p in [work_root, graph_full, graph_tagged, manual_work]:
        p.mkdir(parents=True, exist_ok=True)

    normalize_cmd = [
        py,
        "parser/normalize_or_expand.py",
        "--project-root",
        ".",
        "--dataset-root",
        "DataSet",
        "--input-list",
        str(run_args.get("train_list", "processed/dataset_non_overlap_curated_standard.txt")),
        "--output",
        str(normalized.relative_to(ROOT)),
        "--max-files",
        str(run_args.get("max_files", 0)),
        "--expand-modifiers",
    ]

    graph_cmd = [
        py,
        "parser/build_hetero_graph.py",
        "--input",
        str(normalized.relative_to(ROOT)),
        "--graph-dir",
        str(graph_full.relative_to(ROOT)),
        "--index-out",
        str(graph_index.relative_to(ROOT)),
        "--report-out",
        str(graph_report.relative_to(ROOT)),
        "--implicit-mechanism-edges",
    ]

    tag_cmd = [
        py,
        "slicing/role_tagging.py",
        "--graph-dir",
        str(graph_full.relative_to(ROOT)),
        "--output-dir",
        str(graph_tagged.relative_to(ROOT)),
    ]

    slice_cmd = [
        py,
        "slicing/state_influence_slice.py",
        "--graph-dir",
        str(graph_tagged.relative_to(ROOT)),
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
        str(run_args.get("dual_skeleton_mode", "formal")),
        "--mechanism-path-max-len",
        str(run_args.get("dual_mechanism_path_max_len", 4)),
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
        "z_joint",
        "--prototype-k",
        str(run_args.get("prototype_k", 8)),
        "--family-aware-weight",
        str(run_args.get("family_aware_weight", 0.30)),
        "--family-top-quantile",
        str(run_args.get("family_top_quantile", 1.0)),
        "--family-min-samples",
        str(run_args.get("family_min_samples", 8)),
        "--family-boundary-quantile",
        str(run_args.get("family_boundary_quantile", 0.9)),
        "--family-shrinkage-tau",
        str(run_args.get("family_shrinkage_tau", 16.0)),
        "--family-ramp-width",
        str(run_args.get("family_ramp_width", 6)),
        "--family-global-mix-floor",
        str(run_args.get("family_global_mix_floor", 0.20)),
        "--family-radius-min-ratio",
        str(run_args.get("family_radius_min_ratio", 0.60)),
        "--family-radius-max-ratio",
        str(run_args.get("family_radius_max_ratio", 1.80)),
        "--adaptive-boundary-focus-quantile",
        str(run_args.get("adaptive_boundary_focus_quantile", 0.65)),
        "--adaptive-boundary-min-scale",
        str(run_args.get("adaptive_boundary_min_scale", 0.85)),
        "--adaptive-boundary-max-scale",
        str(run_args.get("adaptive_boundary_max_scale", 1.20)),
        "--adaptive-boundary-outside-scale",
        str(run_args.get("adaptive_boundary_outside_scale", 1.00)),
        "--risk-rerank-mode",
        str(run_args.get("risk_rerank_mode", "function_coverage")),
        "--risk-rerank-topn",
        str(run_args.get("risk_rerank_topn", 2000)),
        "--risk-rerank-file-penalty",
        str(run_args.get("risk_rerank_file_penalty", 0.06)),
        "--risk-rerank-filefn-penalty",
        str(run_args.get("risk_rerank_filefn_penalty", 0.12)),
        "--risk-rerank-fn-novelty-bonus",
        str(run_args.get("risk_rerank_fn_novelty_bonus", 0.04)),
        "--risk-rerank-fn-overlap-penalty",
        str(run_args.get("risk_rerank_fn_overlap_penalty", 0.01)),
        "--risk-rerank-max-per-filefn",
        str(run_args.get("risk_rerank_max_per_filefn", 0)),
        "--risk-rerank-max-per-file",
        str(run_args.get("risk_rerank_max_per_file", 0)),
        "--risk-rerank-file-repeat-power",
        str(run_args.get("risk_rerank_file_repeat_power", 1.0)),
        "--risk-rerank-family-penalty",
        str(run_args.get("risk_rerank_family_penalty", 0.05)),
        "--risk-rerank-family-novelty-bonus",
        str(run_args.get("risk_rerank_family_novelty_bonus", 0.03)),
        "--risk-rerank-aux-weight",
        str(run_args.get("risk_rerank_aux_weight", 0.20)),
        "--risk-rerank-max-per-family",
        str(run_args.get("risk_rerank_max_per_family", 0)),
        "--risk-rerank-warmup-topk",
        str(run_args.get("risk_rerank_warmup_topk", 0)),
        "--risk-rerank-warmup-file-cap",
        str(run_args.get("risk_rerank_warmup_file_cap", 3)),
        "--risk-rerank-warmup-filefn-cap",
        str(run_args.get("risk_rerank_warmup_filefn_cap", 1)),
        "--seed",
        str(run_args.get("seed", 42)),
        "--boundary-quantile",
        "0.9",
        "--risk-out",
        str(unsup_risk.relative_to(ROOT)),
        "--summary-out",
        str(unsup_summary.relative_to(ROOT)),
        "--prototype-out",
        str(proto_out.relative_to(ROOT)),
        "--energy-head-mode",
        str(run_args.get("energy_head_mode", "proto_hybrid")),
        "--energy-head-hidden-dim",
        str(run_args.get("energy_head_hidden_dim", 64)),
        "--energy-head-epochs",
        str(run_args.get("energy_head_epochs", 40)),
        "--energy-head-batch-size",
        str(run_args.get("energy_head_batch_size", 256)),
        "--energy-head-lr",
        str(run_args.get("energy_head_lr", 1e-3)),
        "--energy-head-weight-decay",
        str(run_args.get("energy_head_weight_decay", 1e-4)),
        "--energy-head-noise-std",
        str(run_args.get("energy_head_noise_std", 0.05)),
        "--energy-head-dropout",
        str(run_args.get("energy_head_dropout", 0.10)),
        "--energy-head-out",
        str(energy_out.relative_to(ROOT)),
        "--energy-head-loss-mode",
        str(run_args.get("energy_head_loss_mode", "proto_margin")),
        "--energy-head-anchor-family-min-samples",
        str(run_args.get("energy_head_anchor_family_min_samples", 64)),
    ]
    if bool(run_args.get("adaptive_boundary_refine", True)):
        unsup_cmd.append("--adaptive-boundary-refine")
    else:
        unsup_cmd.append("--no-adaptive-boundary-refine")
    if bool(run_args.get("energy_head_enable", True)):
        unsup_cmd.append("--energy-head-enable")
    else:
        unsup_cmd.append("--no-energy-head-enable")
    if bool(run_args.get("energy_head_stable_train", True)):
        unsup_cmd.append("--energy-head-stable-train")
    else:
        unsup_cmd.append("--no-energy-head-stable-train")
    if bool(run_args.get("energy_head_family_aware_anchor", False)):
        unsup_cmd.append("--energy-head-family-aware-anchor")
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
        str(run_args.get("manual_scope", "label-files")),
        "--max-files",
        str(run_args.get("manual_max_files", 0)),
        "--work-dir",
        str(manual_work.relative_to(ROOT)),
        "--dual-model",
        str(dual_model.relative_to(ROOT)),
        "--prototype-npz",
        str(proto_out.relative_to(ROOT)),
        "--binary-head-method",
        str(run_args.get("manual_binary_head_method", "energy_head")),
        "--energy-head-ckpt",
        str(energy_out.relative_to(ROOT)),
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
        str(run_args.get("manual_dual_skeleton_mode", "legacy")),
        "--dual-mechanism-path-max-len",
        str(run_args.get("manual_dual_mechanism_path_max_len", 4)),
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
        "--line-risk-stage2-max-file-share",
        str(run_args.get("manual_line_risk_stage2_max_file_share", 1.0)),
        "--implicit-mechanism-edges",
    ]
    if bool(run_args.get("manual_slice_formal_propagation", True)):
        manual_cmd.append("--slice-formal-propagation")
    if bool(run_args.get("manual_slice_include_contains", False)):
        manual_cmd.append("--slice-include-contains")
    if bool(run_args.get("adaptive_boundary_refine", True)):
        manual_cmd.append("--adaptive-boundary-refine")

    if args.dry_run:
        print("[run_implicit_mechanism_pipeline] DRY RUN ONLY")
        for cmd in [normalize_cmd, graph_cmd, tag_cmd, slice_cmd, dual_build_cmd, dual_train_cmd, unsup_cmd, manual_cmd]:
            print(" ".join(cmd))
        return

    run_cmd(normalize_cmd, ROOT, "normalize", logs)
    run_cmd(graph_cmd, ROOT, "build_graph_implicit", logs)
    run_cmd(tag_cmd, ROOT, "role_tagging", logs)
    run_cmd(slice_cmd, ROOT, "state_slice", logs)
    run_cmd(dual_build_cmd, ROOT, "dual_view_build", logs)
    run_cmd(dual_train_cmd, ROOT, "train_dual", logs)
    run_cmd(unsup_cmd, ROOT, "train_unsup", logs)
    run_cmd(manual_cmd, ROOT, "manual_eval", logs)

    meta = {
        "config_path": str(cfg_path),
        "resolved_run_args": run_args,
        "out_tag": out_tag,
        "outputs": {
            "graph_report": str(graph_report),
            "dual_summary": str(dual_summary),
            "unsup_summary": str(unsup_summary),
            "manual_summary": str(manual_summary),
        },
        "logs": logs,
    }
    invocation_out.parent.mkdir(parents=True, exist_ok=True)
    invocation_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[run_implicit_mechanism_pipeline] invocation={invocation_out}")


if __name__ == "__main__":
    main()
