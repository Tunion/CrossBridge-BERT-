# FSE24-SmartAxe Unsupervised Cross-Chain Vulnerability Framework

本仓库已实现一个“先最小可运行、再逐步增强”的无监督实验框架，对应以下目标链路：

1. 异构程序图构建  
2. 角色感知 + 状态影响切片  
3. 同切片双视图表示（完整语义 / 安全机制骨架）  
4. 多原型正常模式边界建模  
5. 边界样本局部修正（视图差异 + 局部密度 + 扰动稳定性）  
6. 风险评分与高风险样本输出

---

## 1) 当前目录结构

```text
data/
  raw/
  processed/
  graphs/
  slices/
parser/
  normalize_or_expand.py
  build_hetero_graph.py
  joern_or_slither_adapter.py
slicing/
  role_tagging.py
  state_influence_slice.py
  dual_view_builder.py
models/
  single_view_gnn.py
  dual_view_gnn.py
  prototype_head.py
  boundary_refine.py
trainers/
  train_baseline.py
  train_dual_view.py
  train_unsup.py
eval/
  metrics.py
  ablation.py
  case_study.py
  evaluate_manual_set.py
configs/
  default.yaml
outputs/
results/
run_pipeline.py
```

已有数据与本地化处理说明：
- 原始数据：`DataSet/`, `manually-labeled dataset/`
- 本地去重与口径核验产物：`processed/`
- 详细注意事项：`注意事项_本地数据处理与评测口径.md`

---

## 2) 方法与模块映射

### 阶段1：最小可运行底座
- `parser/normalize_or_expand.py`  
  源码读取 + 结构展开（modifier 语义并回）+ 语义归并（约束检查/外部交互/状态变更）。
- `parser/build_hetero_graph.py`  
  统一异构图格式（节点：function/statement/state_var/call/condition/data_object；边：contains/control/data/call/constraint）。
- `trainers/train_baseline.py` + `models/single_view_gnn.py`  
  单视图无监督训练（中心紧致）与风险分数输出。

### 阶段2：角色感知与状态影响切片
- `slicing/role_tagging.py`  
  在已有图上打三类安全角色标签：`state_change/external_interaction/auth_constraint`。
- `slicing/state_influence_slice.py`  
  以角色节点为种子，沿控制/数据/调用/约束关系前后向扩展，输出局部风险传播子图。

### 阶段3：双视图表示
- `slicing/dual_view_builder.py`  
  从同一切片生成：
  - 完整语义视图（full semantic）
  - 安全机制骨架视图（security skeleton）
- `trainers/train_dual_view.py` + `models/dual_view_gnn.py`  
  支持 `full-only / skeleton-only / dual-view` 三种模式训练。

### 阶段4：多原型正常模式建模
- `models/prototype_head.py`  
  多原型中心 + 局部半径边界（分位数半径）。

### 阶段5：边界样本局部修正
- `models/boundary_refine.py`  
  融合三类信号：视图差异、局部密度、扰动稳定性。
- `trainers/train_unsup.py`  
  输出最终风险分数与 `high_risk/boundary/normal` 状态。

---

## 3) 最小可运行命令链（推荐先跑）

### 一键执行（推荐）

快速 sanity check（小规模）：
```powershell
python run_pipeline.py --quick --run-baseline --run-ablation --run-manual-eval --manual-scope label-files --manual-max-files 30
```

完整实验（按机器资源调整）：
```powershell
python run_pipeline.py --run-baseline --run-ablation --run-manual-eval --manual-scope label-files
```

---

## 4) 分阶段命令链（可单独调试）

### Step 0：准备本地 non-overlap 清单（已做过可跳过）
```powershell
python tools\prepare_local_dataset.py
```

### Step 1：结构展开与语义归并
```powershell
python parser\normalize_or_expand.py `
  --project-root . `
  --dataset-root DataSet `
  --input-list processed/dataset_non_overlap_local.txt `
  --output data/processed/contracts_normalized.jsonl `
  --max-files 200 `
  --keep-comments `
  --expand-modifiers
```

### Step 2：异构图构建
```powershell
python parser\build_hetero_graph.py `
  --input data/processed/contracts_normalized.jsonl `
  --graph-dir data/graphs/full `
  --index-out data/graphs/graph_index.jsonl
```

### Step 3：角色标注与状态影响切片
```powershell
python slicing\role_tagging.py `
  --graph-dir data/graphs/full `
  --output-dir data/graphs/tagged

python slicing\state_influence_slice.py `
  --graph-dir data/graphs/tagged `
  --output data/slices/slices.jsonl `
  --hops 2 `
  --min-slice-nodes 6
```

### Step 4：双视图构建
```powershell
python slicing\dual_view_builder.py `
  --input data/slices/slices.jsonl `
  --output data/slices/dual_views.jsonl
```

### Step 5：阶段1基线训练（单视图）
```powershell
python trainers\train_baseline.py `
  --graph-dir data/graphs/full `
  --max-graphs 500 `
  --epochs 15 `
  --model-out outputs/models/baseline_single_view.pt `
  --risk-out outputs/results/baseline_risk_scores.csv `
  --summary-out outputs/results/baseline_summary.json
```

### Step 6：阶段3双视图训练（建议至少跑 dual-view）
```powershell
python trainers\train_dual_view.py `
  --input data/slices/dual_views.jsonl `
  --mode dual-view `
  --max-slices 2000 `
  --epochs 15 `
  --model-out outputs/models/dual_view.pt `
  --risk-out outputs/results/dual_view_risk_scores.csv `
  --summary-out outputs/results/dual_view_summary.json `
  --embedding-out outputs/results/dual_view_embeddings.npz
```

### Step 7：阶段4+5（多原型 + 边界修正）
```powershell
python trainers\train_unsup.py `
  --dual-view-input data/slices/dual_views.jsonl `
  --embedding-npz outputs/results/dual_view_embeddings.npz `
  --prototype-k 8 `
  --boundary-quantile 0.9 `
  --risk-out outputs/results/unsup_final_risk_scores.csv `
  --summary-out outputs/results/unsup_final_summary.json `
  --prototype-out outputs/models/prototypes.npz
```

---

## 5) 评估与分析脚本

### 4.1 召回/精度（Top-K）
```powershell
python eval\metrics.py `
  --pred outputs/results/unsup_final_risk_scores.csv `
  --label-file processed/label_standard_local.csv `
  --label-mode row88 `
  --score-col final_risk `
  --top-k 200 `
  --out outputs/results/metrics_row88.json
```

可选函数级口径：
```powershell
python eval\metrics.py `
  --pred outputs/results/unsup_final_risk_scores.csv `
  --label-file processed/label_standard_local.csv `
  --label-mode file_function `
  --score-col final_risk `
  --top-k 200 `
  --out outputs/results/metrics_file_function.json
```

### 4.2 消融比较
```powershell
python trainers\train_dual_view.py --input data/slices/dual_views.jsonl --mode full-only --risk-out outputs/results/full_only_risk.csv
python trainers\train_dual_view.py --input data/slices/dual_views.jsonl --mode skeleton-only --risk-out outputs/results/skeleton_only_risk.csv
python trainers\train_dual_view.py --input data/slices/dual_views.jsonl --mode dual-view --risk-out outputs/results/dual_view_risk_scores.csv

python eval\ablation.py `
  --full outputs/results/full_only_risk.csv `
  --skeleton outputs/results/skeleton_only_risk.csv `
  --dual outputs/results/dual_view_risk_scores.csv `
  --top-k 100 `
  --out outputs/results/ablation_summary.json
```

### 4.3 案例分析
```powershell
python eval\case_study.py `
  --risk-csv outputs/results/unsup_final_risk_scores.csv `
  --dual-view-input data/slices/dual_views.jsonl `
  --top-k 20 `
  --out outputs/results/case_study_top20.md
```

### 4.4 手工标注集正式评测（双口径召回）
```powershell
python eval\evaluate_manual_set.py `
  --project-root . `
  --manual-root "manually-labeled dataset/Real_attack_dataset_format" `
  --label-csv processed/label_standard_local.csv `
  --scope label-files `
  --dual-model outputs/models/dual_view.pt `
  --prototype-npz outputs/models/prototypes.npz `
  --risk-out outputs/results/manual_eval_risk_scores.csv `
  --summary-out outputs/results/manual_eval_summary.json
```

---

## 6) 重要口径与注意事项

1. 本地训练请使用 `processed/dataset_non_overlap_local.txt`，不要直接用旧版 `dataset_non_overlap.txt`。  
2. `label_standard.csv` 的“88”是行级口径；函数级口径（`file+function`）不是同一个分母。  
3. 训练集是近似正常集合，可能混入未标注异常；请保留“二次清洗 + 边界修正”流程。  
4. 数据内部重复率较高，建议在正式实验中补充分层采样/去重降权策略。  
5. 标签质量问题与建议修复见 `processed/label_quality_issues.csv`。  

详见：`注意事项_本地数据处理与评测口径.md`

---

## 7) 依赖

已在本环境验证可用：
- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `pyyaml`

---

## 8) 已完成的最小运行验证

在本地已用小样本执行完以下链路并产出结果：
- `normalize_or_expand -> build_hetero_graph -> role_tagging -> state_influence_slice -> dual_view_builder`
- `train_baseline`
- `train_dual_view (full-only/skeleton-only/dual-view)`
- `train_unsup`
- `eval/ablation.py`, `eval/case_study.py`, `eval/metrics.py`

结果文件位于 `outputs/results/`。

---

## 9) Formal Pipeline (New)

Formal mode adds:
- lightweight AST/CFG/DFG semantic edges in hetero graph construction;
- role-aware directional slicing (`forward` + `backward`) with mechanism-closure paths;
- formal security-skeleton view construction (role-triplet closure).

Quick sanity run:

```powershell
python tools/run_formal_pipeline.py --quick --summary-out outputs/results/formal_pipeline_summary_quick.json
```

Full run from YAML defaults:

```powershell
python tools/run_formal_pipeline.py --config configs/formal_v1.yaml
```

Preview resolved command without execution:

```powershell
python tools/run_formal_pipeline.py --config configs/formal_v1.yaml --dry-run
```

## 10) AST/CFG/DFG Preprocessing Upgrade (2026-03-19)

`parser/normalize_or_expand.py` now uses `solidity-parser` as the primary backend:

- Parses Solidity source into AST (with graceful fallback to regex heuristic mode).
- Extracts function statements from AST with:
  - statement nesting (`parent_stmt_index`),
  - structured CFG successors (`cfg_succ_indices`),
  - def/use hints (`defs`, `uses`, `state_defs`),
  - call targets and key data objects (`call_targets`, `data_objects`).
- Preserves compatibility with the old output schema.

`parser/build_hetero_graph.py` now consumes these structured fields first:

- Builds CFG edges from `cfg_succ_indices` when available.
- Builds nested AST parent/control edges from `parent_stmt_index`.
- Builds DFG from explicit `defs/uses` before regex fallback.

New normalized record metadata:

- `parser_backend`: `solidity_parser_ast` or `heuristic_regex`
- `ast_parse_ok`: `true/false`
- `ast_parse_error`: parse failure reason when fallback is used

Pipeline default now prefers prebuilt AST-v2 graphs:

- If `data/graphs/full_ast_full_v2` exists, `run_pipeline.py` will reuse it automatically and skip `normalize/build_graph`.
- You can force a specific graph set via:
  - `--prebuilt-graph-dir data/graphs/full_ast_full_v2`

## 11) End-to-End Multi-Prototype SVDD Head (2026-03-20)

New files:

- `models/e2e_multi_proto_svdd.py`
- `trainers/train_e2e_unsup.py`

This stage keeps the existing parser/slicing/dual-view input, and upgrades Stage-4/5 from
"offline KMeans + post-hoc boundary scoring" to a trainable end-to-end multi-prototype head.

Pipeline switch is also available:

```powershell
python run_pipeline.py --unsup-mode e2e --epochs-e2e 8 --warmup-epochs-e2e 2
```

### Quick smoke run

```powershell
python trainers/train_e2e_unsup.py `
  --input data/slices/dual_views.jsonl `
  --max-slices 120 `
  --epochs 2 `
  --warmup-epochs 1 `
  --batch-size 16 `
  --device cpu `
  --risk-rerank-mode none `
  --mechanism-risk-weight 0.0 `
  --model-out outputs/models/e2e_mp_svdd_smoke.pt `
  --prototype-out outputs/models/e2e_prototypes_smoke.npz `
  --embedding-out outputs/results/e2e_dual_embeddings_smoke.npz `
  --risk-out outputs/results/e2e_risk_smoke.csv `
  --summary-out outputs/results/e2e_summary_smoke.json
```

### Full run template

```powershell
python trainers/train_e2e_unsup.py `
  --input data/slices/dual_views.jsonl `
  --max-slices 12000 `
  --epochs 8 `
  --warmup-epochs 2 `
  --batch-size 32 `
  --hidden-dim 64 `
  --num-layers 2 `
  --prototype-k 8 `
  --boundary-quantile 0.9 `
  --risk-rerank-mode function_coverage `
  --risk-rerank-topn 2000 `
  --risk-rerank-file-penalty 0.08 `
  --risk-rerank-filefn-penalty 0.12 `
  --risk-rerank-fn-novelty-bonus 0.02 `
  --risk-rerank-fn-overlap-penalty 0.01 `
  --device cpu `
  --risk-out outputs/results/e2e_unsup_final_risk_scores.csv `
  --summary-out outputs/results/e2e_unsup_final_summary.json `
  --prototype-out outputs/models/e2e_prototypes.npz `
  --model-out outputs/models/e2e_mp_svdd.pt
```

---

## Slice dedup stats (Goal-2 minimal)

`slicing/state_influence_slice.py` now supports near-duplicate filtering observability:

- `--dedup-line-jaccard` (default `0.95`): drop near-duplicate slices in the same function.
- `--max-slices-per-function` (default `120`): hard cap per function.
- `--stats-out`: emit slicing statistics json.

Quick check example:

```powershell
python slicing/state_influence_slice.py `
  --graph-dir data/graphs/tagged `
  --output outputs/tmp/slices_test_goal2.jsonl `
  --max-graphs 5 `
  --dedup-line-jaccard 0.95 `
  --max-slices-per-function 120 `
  --stats-out outputs/results/state_slice_stats_goal2_test.json
```

Pipeline-level pass-through is available in:

- `run_pipeline.py`
- `tools/run_multiseed_pipeline.py`
- `eval/evaluate_manual_set.py` (manual-set slicing path)

## Family-aware stabilization (Goal-3 minimal)

Family-aware scoring now uses smooth local/global shrinkage instead of hard switching:

- `--family-shrinkage-tau`: controls how fast local family boundary gains trust.
- `--family-ramp-width`: smooth transition width near `family_min_samples`.
- `--family-global-mix-floor`: keep minimum global component for stability.
- `--family-radius-min-ratio` / `--family-radius-max-ratio`: clip local radius against global radius.

Available in:

- `trainers/train_unsup.py`
- `eval/evaluate_manual_set.py`
- propagated by `run_pipeline.py` and `tools/run_multiseed_pipeline.py`.

## Graph semantic coverage report (Goal-4 minimal)

`parser/build_hetero_graph.py` supports `--report-out` to emit semantic coverage stats:

- AST/CFG/DFG edge coverage ratios
- AST parse-backed ratio (`ast_parse_ok`)
- structured CFG availability ratio (`cfg_structured_edge_count > 0`)
- aggregate node/edge type counts

Example:

```powershell
python parser/build_hetero_graph.py `
  --input data/processed/contracts_normalized.jsonl `
  --graph-dir data/graphs/full `
  --index-out data/graphs/graph_index.jsonl `
  --report-out outputs/results/graph_semantic_coverage.json
```

## Adaptive boundary refine (Goal-5 minimal)

`models/boundary_refine.py` now supports local adaptive correction around boundary-uncertain samples.

- Enable with `--adaptive-boundary-refine`
- Key knobs:
  - `--adaptive-boundary-focus-quantile` (default `0.65`)
  - `--adaptive-boundary-min-scale` (default `0.85`)
  - `--adaptive-boundary-max-scale` (default `1.20`)
  - `--adaptive-boundary-outside-scale` (default `1.00`)

The same options are wired through:

- `trainers/train_unsup.py`
- `eval/evaluate_manual_set.py`
- `run_pipeline.py` (`--unsup-*` / `--manual-*` prefixed)
- `tools/run_multiseed_pipeline.py`

## Generalization split eval (Goal-6 minimal)

New script:

- `eval/generalization_split.py`

It reports Top-K recall split by:

- seen families (appearing in training list)
- unseen families (not appearing in training list)

Example:

```powershell
python eval/generalization_split.py `
  --risk-csv outputs/results/manual_eval_risk_scores.csv `
  --label-csv processed/label_standard_local.csv `
  --train-list processed/dataset_non_overlap_local.txt `
  --topk 20,50,100,200 `
  --line-window 2 `
  --out outputs/results/generalization_split_summary.json
```

Pipeline integration:

- `run_pipeline.py --run-generalization-split`
- `tools/run_multiseed_pipeline.py --run-generalization-split`
