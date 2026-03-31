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
    p = argparse.ArgumentParser(description="Run the isolated global-local hybrid branch with dedicated outputs.")
    p.add_argument("--config", type=str, default="configs/glocal_hybrid_015.yaml")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_cmd(cmd: List[str], cwd: Path, stage: str, logs: List[Dict[str, Any]]) -> None:
    print(f"[run_glocal_hybrid_pipeline] [{stage}] RUN: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, cwd=str(cwd), check=True)
    logs.append({"stage": stage, "command": " ".join(cmd), "elapsed_sec": time.time() - t0})


def main() -> None:
    args = parse_args()
    cfg_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    run_args = dict(cfg.get("run_glocal_hybrid_args", {}))

    out_tag = str(run_args.get("out_tag", "glocalhyb"))
    py = sys.executable
    logs: List[Dict[str, Any]] = []

    dual_model = f"outputs/models/dual_view_{out_tag}.pt"
    dual_risk = f"outputs/results/dual_view_risk_scores_{out_tag}.csv"
    dual_summary = f"outputs/results/dual_view_summary_{out_tag}.json"
    dual_emb = f"outputs/results/dual_view_embeddings_{out_tag}.npz"
    proto_out = f"outputs/models/prototypes_{out_tag}.npz"
    unsup_risk = f"outputs/results/unsup_final_risk_scores_{out_tag}.csv"
    unsup_summary = f"outputs/results/unsup_final_summary_{out_tag}.json"
    manual_risk = f"outputs/results/manual_eval_risk_scores_{out_tag}.csv"
    manual_binary = f"outputs/results/manual_eval_binary_predictions_{out_tag}.csv"
    manual_summary = f"outputs/results/manual_eval_summary_{out_tag}.json"
    manual_line = f"outputs/results/manual_eval_line_risk_scores_{out_tag}.csv"
    manual_stats = f"outputs/results/manual_state_slice_stats_{out_tag}.json"
    manual_work = f"outputs/manual_eval_artifacts_{out_tag}"
    invocation_out = ROOT / f"outputs/results/run_glocal_hybrid_pipeline_{out_tag}.json"

    dual_cmd = [
        py,
        "trainers/train_dual_view.py",
        "--input",
        str(run_args.get("dual_input", "data/slices/dual_views.jsonl")),
        "--mode",
        "dual-view",
        "--max-slices",
        str(run_args.get("max_slices", 5000)),
        "--train-ratio",
        str(run_args.get("train_ratio", 0.8)),
        "--epochs",
        str(run_args.get("epochs_dual", 4)),
        "--seed",
        str(run_args.get("seed", 42)),
        "--hidden-dim",
        str(run_args.get("hidden_dim", 64)),
        "--num-layers",
        str(run_args.get("num_layers", 2)),
        "--batch-size",
        str(run_args.get("batch_size", 32)),
        "--device",
        str(run_args.get("device", "cuda")),
        "--global-local-enable",
        "--global-graph-dir",
        str(run_args.get("dual_global_graph_dir", "data/graphs/tagged")),
        "--lambda-global-align",
        "0.20",
        "--lambda-global-risk",
        "0.20",
        "--model-out",
        dual_model,
        "--risk-out",
        dual_risk,
        "--summary-out",
        dual_summary,
        "--embedding-out",
        dual_emb,
    ]

    unsup_cmd = [
        py,
        "trainers/train_unsup.py",
        "--dual-view-input",
        str(run_args.get("dual_input", "data/slices/dual_views.jsonl")),
        "--embedding-npz",
        dual_emb,
        "--embedding-key",
        "z_hybrid",
        "--hybrid-alpha",
        str(run_args.get("hybrid_alpha", 0.15)),
        "--prototype-k",
        "8",
        "--boundary-quantile",
        "0.9",
        "--mechanism-risk-weight",
        "0.0",
        "--mechanism-top-quantile",
        "1.0",
        "--family-aware-weight",
        "0.3",
        "--family-top-quantile",
        "1.0",
        "--family-min-samples",
        "8",
        "--family-boundary-quantile",
        "0.9",
        "--family-shrinkage-tau",
        "16.0",
        "--family-ramp-width",
        "6",
        "--family-global-mix-floor",
        "0.2",
        "--family-radius-min-ratio",
        "0.6",
        "--family-radius-max-ratio",
        "1.8",
        "--density-k",
        "10",
        "--n-perturb",
        "5",
        "--noise-std",
        "0.02",
        "--adaptive-boundary-refine",
        "--adaptive-boundary-focus-quantile",
        "0.65",
        "--adaptive-boundary-min-scale",
        "0.85",
        "--adaptive-boundary-max-scale",
        "1.2",
        "--adaptive-boundary-outside-scale",
        "1.0",
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
        str(run_args.get("risk_rerank_aux_weight", 0.2)),
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
        "--risk-out",
        unsup_risk,
        "--summary-out",
        unsup_summary,
        "--prototype-out",
        proto_out,
    ]

    manual_cmd = [
        py,
        "eval/evaluate_manual_set.py",
        "--project-root",
        ".",
        "--manual-root",
        "manually-labeled dataset/Real_attack_dataset_format",
        "--label-csv",
        "processed/label_standard_local.csv",
        "--scope",
        "label-files",
        "--max-files",
        "0",
        "--work-dir",
        manual_work,
        "--dual-model",
        dual_model,
        "--prototype-npz",
        proto_out,
        "--embedding-key",
        "z_hybrid",
        "--device",
        "cpu",
        "--hops",
        "2",
        "--slice-forward-hops",
        "0",
        "--slice-backward-hops",
        "0",
        "--min-slice-nodes",
        "6",
        "--slice-mechanism-path-max-len",
        "4",
        "--slice-function-fallback-mode",
        "none",
        "--slice-function-fallback-min-nodes",
        "3",
        "--slice-dedup-line-jaccard",
        "0.95",
        "--slice-max-slices-per-function",
        "120",
        "--dual-skeleton-mode",
        "legacy",
        "--dual-mechanism-path-max-len",
        "4",
        "--risk-out",
        manual_risk,
        "--binary-out",
        manual_binary,
        "--summary-out",
        manual_summary,
        "--line-risk-out",
        manual_line,
        "--line-risk-closure-boost",
        "0.3",
        "--line-risk-stage2-weight",
        "0.15",
        "--line-risk-stage2-top-functions",
        "640",
        "--line-risk-stage2-support-weight",
        "0.6",
        "--mechanism-risk-weight",
        "0.0",
        "--mechanism-top-quantile",
        "1.0",
        "--mechanism-semantic-min-pairs",
        "2",
        "--mechanism-semantic-max-hops",
        "6",
        "--family-aware-weight",
        "0.3",
        "--family-top-quantile",
        "1.0",
        "--family-min-samples",
        "8",
        "--family-boundary-quantile",
        "0.9",
        "--family-shrinkage-tau",
        "16.0",
        "--family-ramp-width",
        "6",
        "--family-global-mix-floor",
        "0.2",
        "--family-radius-min-ratio",
        "0.6",
        "--family-radius-max-ratio",
        "1.8",
        "--adaptive-boundary-focus-quantile",
        "0.65",
        "--adaptive-boundary-min-scale",
        "0.85",
        "--adaptive-boundary-max-scale",
        "1.2",
        "--adaptive-boundary-outside-scale",
        "1.0",
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
        str(run_args.get("risk_rerank_aux_weight", 0.2)),
        "--risk-rerank-max-per-family",
        str(run_args.get("risk_rerank_max_per_family", 0)),
        "--risk-rerank-warmup-topk",
        str(run_args.get("risk_rerank_warmup_topk", 0)),
        "--risk-rerank-warmup-file-cap",
        str(run_args.get("risk_rerank_warmup_file_cap", 3)),
        "--risk-rerank-warmup-filefn-cap",
        str(run_args.get("risk_rerank_warmup_filefn_cap", 1)),
        "--score-head-method",
        str(run_args.get("score_head_method", "iforest")),
        "--score-head-fit-scope",
        str(run_args.get("score_head_fit_scope", "low_risk")),
        "--binary-head-method",
        str(run_args.get("binary_head_method", "ocsvm")),
        "--binary-head-fit-scope",
        str(run_args.get("binary_head_fit_scope", "all")),
        "--binary-head-contamination",
        str(run_args.get("binary_head_contamination", 0.08)),
        "--triage-low-quantile",
        str(run_args.get("triage_low_quantile", 0.0)),
        "--triage-high-quantile",
        str(run_args.get("triage_high_quantile", 0.9)),
        "--triage-gray-threshold",
        str(run_args.get("triage_gray_threshold", 0.5130835772)),
        "--binary-target-mode",
        "file_function",
        "--binary-target-window",
        "2",
        "--binary-eval-topn",
        "0",
        "--slice-stats-out",
        manual_stats,
        "--adaptive-boundary-refine",
        "--slice-formal-propagation",
    ]

    if args.dry_run:
        print("[run_glocal_hybrid_pipeline] DRY RUN ONLY")
        for cmd in [dual_cmd, unsup_cmd, manual_cmd]:
            print(" ".join(cmd))
        return

    run_cmd(dual_cmd, ROOT, "train_dual_glocal", logs)
    run_cmd(unsup_cmd, ROOT, "train_unsup_hybrid", logs)
    run_cmd(manual_cmd, ROOT, "manual_eval_hybrid", logs)

    meta = {
        "config_path": str(cfg_path),
        "resolved_run_args": run_args,
        "out_tag": out_tag,
        "outputs": {
            "dual_model": dual_model,
            "prototype_out": proto_out,
            "manual_summary": manual_summary,
        },
        "logs": logs,
    }
    invocation_out.parent.mkdir(parents=True, exist_ok=True)
    invocation_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[run_glocal_hybrid_pipeline] invocation={invocation_out}")


if __name__ == "__main__":
    main()
