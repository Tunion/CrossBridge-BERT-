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
    p = argparse.ArgumentParser(description="Run the LLM-role-hint experiment with isolated outputs.")
    p.add_argument("--config", type=str, default="configs/llm_role_hint_seed42.yaml")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_cmd(cmd: List[str], cwd: Path, stage: str, logs: List[Dict[str, Any]]) -> None:
    print(f"[run_llm_role_hint_pipeline] [{stage}] RUN: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, cwd=str(cwd), check=True)
    logs.append({"stage": stage, "command": " ".join(cmd), "elapsed_sec": time.time() - t0})


def main() -> None:
    args = parse_args()
    cfg_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    run_args = dict(cfg.get("run_llm_role_hint_args", {}))

    py = sys.executable
    out_tag = str(run_args.get("out_tag", "llmrole_seed42"))
    logs: List[Dict[str, Any]] = []

    work_root = ROOT / f"outputs/{out_tag}"
    graph_full = work_root / "graphs_full"
    graph_tagged = work_root / "graphs_tagged"
    normalized = work_root / "contracts_normalized.jsonl"
    graph_index = work_root / "graph_index.jsonl"
    slices = work_root / "slices.jsonl"
    dual_views = work_root / "dual_views.jsonl"
    graph_report = ROOT / f"outputs/results/graph_semantic_coverage_{out_tag}.json"
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
    manual_summary = ROOT / f"outputs/results/manual_eval_summary_{out_tag}.json"
    manual_risk = ROOT / f"outputs/results/manual_eval_risk_scores_{out_tag}.csv"
    manual_line = ROOT / f"outputs/results/manual_eval_line_risk_scores_{out_tag}.csv"
    manual_binary = ROOT / f"outputs/results/manual_eval_binary_predictions_{out_tag}.csv"
    manual_stats = ROOT / f"outputs/results/manual_state_slice_stats_{out_tag}.json"
    invocation_out = ROOT / f"outputs/results/run_llm_role_hint_pipeline_{out_tag}.json"

    for p in [work_root, graph_full, graph_tagged, manual_work]:
        p.mkdir(parents=True, exist_ok=True)

    hint_path = str(run_args.get("llm_role_hints", "")).strip()
    if not hint_path:
        raise RuntimeError("llm_role_hints must be provided in config for this experiment.")

    normalize_cmd = [
        py, "parser/normalize_or_expand.py",
        "--project-root", ".",
        "--dataset-root", "DataSet",
        "--input-list", str(run_args.get("train_list", "processed/dataset_non_overlap_curated_standard.txt")),
        "--output", str(normalized.relative_to(ROOT)),
        "--max-files", str(run_args.get("max_files", 0)),
        "--expand-modifiers",
    ]
    graph_cmd = [
        py, "parser/build_hetero_graph.py",
        "--input", str(normalized.relative_to(ROOT)),
        "--graph-dir", str(graph_full.relative_to(ROOT)),
        "--index-out", str(graph_index.relative_to(ROOT)),
        "--report-out", str(graph_report.relative_to(ROOT)),
    ]
    tag_cmd = [
        py, "slicing/role_tagging.py",
        "--graph-dir", str(graph_full.relative_to(ROOT)),
        "--output-dir", str(graph_tagged.relative_to(ROOT)),
        "--llm-role-hints", hint_path,
        "--llm-role-low-confidence-max", str(run_args.get("llm_role_low_confidence_max", 0.60)),
        "--llm-role-min-prob", str(run_args.get("llm_role_min_prob", 0.72)),
        "--llm-role-weight", str(run_args.get("llm_role_weight", 0.45)),
    ]
    slice_cmd = [
        py, "slicing/state_influence_slice.py",
        "--graph-dir", str(graph_tagged.relative_to(ROOT)),
        "--output", str(slices.relative_to(ROOT)),
        "--hops", "2",
        "--forward-hops", "2",
        "--backward-hops", "2",
        "--min-slice-nodes", "6",
        "--mechanism-path-max-len", "4",
        "--function-fallback-mode", "none",
        "--function-fallback-min-nodes", "3",
        "--dedup-line-jaccard", "0.95",
        "--max-slices-per-function", "120",
        "--stats-out", str(slice_stats.relative_to(ROOT)),
        "--include-contains",
        "--formal-propagation",
    ]
    dual_build_cmd = [
        py, "slicing/dual_view_builder.py",
        "--input", str(slices.relative_to(ROOT)),
        "--output", str(dual_views.relative_to(ROOT)),
        "--max-slices", str(run_args.get("max_slices", 20000)),
        "--skeleton-mode", "formal",
        "--mechanism-path-max-len", "4",
    ]
    dual_train_cmd = [
        py, "trainers/train_dual_view.py",
        "--input", str(dual_views.relative_to(ROOT)),
        "--mode", "dual-view",
        "--max-slices", str(run_args.get("max_slices", 20000)),
        "--epochs", str(run_args.get("epochs_dual", 4)),
        "--seed", str(run_args.get("seed", 42)),
        "--hidden-dim", "64",
        "--num-layers", "2",
        "--device", str(run_args.get("device", "cpu")),
        "--model-out", str(dual_model.relative_to(ROOT)),
        "--risk-out", str(dual_risk.relative_to(ROOT)),
        "--summary-out", str(dual_summary.relative_to(ROOT)),
        "--embedding-out", str(dual_emb.relative_to(ROOT)),
    ]
    unsup_cmd = [
        py, "trainers/train_unsup.py",
        "--dual-view-input", str(dual_views.relative_to(ROOT)),
        "--embedding-npz", str(dual_emb.relative_to(ROOT)),
        "--embedding-key", "z_joint",
        "--prototype-k", "8",
        "--family-aware-weight", "0.30",
        "--adaptive-boundary-refine",
        "--risk-rerank-mode", "function_coverage",
        "--prototype-out", str(proto_out.relative_to(ROOT)),
        "--risk-out", str(unsup_risk.relative_to(ROOT)),
        "--summary-out", str(unsup_summary.relative_to(ROOT)),
        "--energy-head-enable",
        "--energy-head-mode", "proto_hybrid",
        "--energy-head-loss-mode", "proto_margin",
        "--energy-head-out", str(energy_out.relative_to(ROOT)),
        "--seed", str(run_args.get("seed", 42)),
    ]
    manual_cmd = [
        py, "eval/evaluate_manual_set.py",
        "--project-root", ".",
        "--manual-root", str(run_args.get("manual_root", "manually-labeled dataset/Real_attack_dataset_format")),
        "--label-csv", str(run_args.get("label_csv", "processed/label_standard_local.csv")),
        "--scope", str(run_args.get("manual_scope", "label-files")),
        "--max-files", "0",
        "--work-dir", str(manual_work.relative_to(ROOT)),
        "--dual-model", str(dual_model.relative_to(ROOT)),
        "--prototype-npz", str(proto_out.relative_to(ROOT)),
        "--binary-head-method", "energy_head",
        "--energy-head-ckpt", str(energy_out.relative_to(ROOT)),
        "--triage-gray-threshold", "0.28",
        "--summary-out", str(manual_summary.relative_to(ROOT)),
        "--risk-out", str(manual_risk.relative_to(ROOT)),
        "--line-risk-out", str(manual_line.relative_to(ROOT)),
        "--binary-out", str(manual_binary.relative_to(ROOT)),
        "--slice-stats-out", str(manual_stats.relative_to(ROOT)),
    ]

    if args.dry_run:
        print("[run_llm_role_hint_pipeline] DRY RUN ONLY")
        for cmd in [normalize_cmd, graph_cmd, tag_cmd, slice_cmd, dual_build_cmd, dual_train_cmd, unsup_cmd, manual_cmd]:
            print(" ".join(cmd))
        return

    for stage, cmd in [
        ("normalize", normalize_cmd),
        ("build_graph", graph_cmd),
        ("role_tagging", tag_cmd),
        ("slice", slice_cmd),
        ("dual_build", dual_build_cmd),
        ("dual_train", dual_train_cmd),
        ("unsup", unsup_cmd),
        ("manual_eval", manual_cmd),
    ]:
        run_cmd(cmd, ROOT, stage, logs)

    meta = {"config_path": str(cfg_path), "resolved_run_args": run_args, "out_tag": out_tag, "logs": logs}
    invocation_out.parent.mkdir(parents=True, exist_ok=True)
    invocation_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[run_llm_role_hint_pipeline] invocation={invocation_out}")


if __name__ == "__main__":
    main()
