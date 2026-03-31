# Curated Training + Anti-Collapse Notes (2026-03-19)

This note records the new pipeline changes requested in this chat:
- reduce training-set contamination by curated selection
- avoid representation collapse in dual-view unsupervised training
- compare strict vs relaxed curation to pick a practical default

## 1) New code changes

### 1.1 Curated training-list builder
File: `tools/build_curated_training_list.py`

Added/updated:
- bridge whitelist sampling (`--bridge-whitelist`)
- contamination exclusion keywords (`--exclude-bridge-keywords`)
- common library filename filter (`--drop-common-lib-filenames`)
- line/function quality filters (`--min-lines`, `--max-lines`, `--min-functions`)
- content dedup by MD5
- per-bridge cap and global target size control
- rejection statistics (`reject_counts`) in report JSON

Default exclusion keywords now include:
- `Ronin Bridge|Nomad|Multichain`

### 1.2 Pipeline integration
File: `run_pipeline.py`

Added new args:
- `--train-list`
- `--build-curated-standard`
- `--curated-output-list`
- `--curated-output-report`
- `--curated-max-per-bridge`
- `--curated-target-size`
- `--curated-exclude-bridges`
- `--curated-drop-common-libs`

Behavior:
- when `--build-curated-standard` is enabled, pipeline first builds curated list and uses it in normalization stage.
- dual-view build stage now also accepts `--max-slices` to avoid huge intermediate artifacts.

### 1.3 Anti-collapse dual-view training
File: `trainers/train_dual_view.py`

Reworked objective from simple compactness to multi-term regularized self-supervised training:
- contrastive alignment (InfoNCE): `--lambda-contrast`, `--temperature`
- view invariance (MSE): `--lambda-invariance`
- variance regularization: `--lambda-variance`, `--var-target`
- covariance decorrelation: `--lambda-covariance`
- weak compactness: `--lambda-compact`
- risk fusion with view-gap: `--lambda-gap-risk`

Also outputs embedding diagnostics:
- `mean_std`, `min_std`, `max_std`, `avg_l2`

These diagnostics are used to detect collapse quickly.

### 1.4 Prototype robustness in low-diversity embeddings
File: `models/prototype_head.py`

KMeans cluster count is bounded by the number of unique rounded embeddings.
This avoids degenerate `k > effective-unique-points` settings.

## 2) Runs performed in this chat

## Run A: strict curated
Summary:
- `outputs/results/pipeline_run_summary_curated_anticollapse.json`

Data report:
- `processed/dataset_non_overlap_curated_standard_report_strict.json`

Key settings:
- exclude bridges: `Ronin Bridge|Nomad|Multichain`
- selected training files: 758
- generated slices: 34304
- dual-view training slices used: 20000
- dual-view epochs: 4

Key outputs:
- `outputs/results/dual_view_summary_strict_curated.json`
- `outputs/results/unsup_final_summary_strict_curated.json`
- `outputs/results/manual_eval_summary_strict_curated.json`

Manual eval (file_function recall):
- top20: 0.0133
- top50: 0.1067
- top100: 0.2133
- top200: 0.4000

Row88 recall remained 0.0.

## Run B: relaxed curated
Summary:
- `outputs/results/pipeline_run_summary_curated_relaxed_anticollapse.json`

Data report:
- `processed/dataset_non_overlap_curated_standard_report_relaxed.json`

Key settings:
- exclude bridges: `__NONE__` (effectively disabled)
- selected training files: 813
- generated slices: 36639
- dual-view training slices used: 20000
- dual-view epochs: 4

Key outputs:
- `outputs/results/dual_view_summary_relaxed_curated.json`
- `outputs/results/unsup_final_summary_relaxed_curated.json`
- `outputs/results/manual_eval_summary_relaxed_curated.json`

Manual eval (file_function recall):
- top20: 0.1200
- top50: 0.1867
- top100: 0.2933
- top200: 0.5200

Row88 recall remained 0.0.

## 3) A/B comparison artifact

Generated file:
- `outputs/results/curated_experiment_comparison.json`

Contains side-by-side metrics for strict vs relaxed:
- selected file counts
- embedding diagnostics
- manual top-k file_function recall
- unsup status distribution

## 4) Practical conclusion for current repo

1. Anti-collapse objective works structurally:
- embedding std is non-trivial (around 0.48~0.49), unlike previous collapse symptoms.
- strict run ablation overlap is no longer 100/100/100 (now much lower), indicating representations are not all identical rankings.

2. Over-cleaning can hurt practical recall:
- strict curation (758 files) reduced manual file_function recall@200 to 0.40.
- relaxed curation (813 files) improved it to 0.52 under same training setup.

3. Recommended default for now:
- use relaxed curation as default training list generation (do not hard-exclude bridges yet).
- keep anti-collapse training objective enabled.
- keep dual-view `max-slices=20000` for fast iteration; scale up later.

## 5) Repro commands (recommended default)

```powershell
python run_pipeline.py \
  --max-files 0 \
  --max-slices 20000 \
  --epochs-dual 4 \
  --hidden-dim 64 \
  --num-layers 2 \
  --device cpu \
  --prototype-k 8 \
  --build-curated-standard \
  --train-list processed/dataset_non_overlap_local.txt \
  --curated-output-list processed/dataset_non_overlap_curated_standard.txt \
  --curated-output-report processed/dataset_non_overlap_curated_standard_report.json \
  --curated-max-per-bridge 120 \
  --curated-target-size 2500 \
  --curated-drop-common-libs \
  --curated-exclude-bridges __NONE__ \
  --run-manual-eval \
  --manual-scope label-files \
  --summary-out outputs/results/pipeline_run_summary_curated_relaxed_anticollapse.json
```

## 6) Important caveat

Current row-level recall (`row88`) is still 0.0 in this framework.
Reason: ranking is currently slice-level risk, while row88 matching is strict on exact line-level hits.
This likely needs an explicit line-localization calibration module if row88 is a primary KPI.

## 7) Bug fixes applied after initial runs (2026-03-19, later)

Three concrete parser/slicing bugs were fixed:

1. `--keep-comments` behavior bug
- file: `parser/normalize_or_expand.py`
- before: comment lines were always dropped from function statements.
- after: comments are kept when `--keep-comments` is enabled.

2. One-line function body parsing bug
- file: `parser/normalize_or_expand.py`
- before: `function f() { ... }` patterns often became zero-statement functions.
- after: added `parse_inline_statements` to extract simple inline body statements.

3. Slice seed metadata bug
- file: `slicing/state_influence_slice.py`
- before: each slice stored `seed_node_ids` as the full graph seed list.
- after: each slice now stores only the true seed IDs that generated that specific slice.

## 8) Retrain after bug fixes (relaxed curated)

Pipeline run summary:
- `outputs/results/pipeline_run_summary_bugfix_retrain_relaxed.json`

Main outputs:
- `outputs/results/dual_view_summary_bugfix_retrain_relaxed.json`
- `outputs/results/unsup_final_summary_bugfix_retrain_relaxed.json`
- `outputs/results/manual_eval_summary_bugfix_retrain_relaxed.json`
- `outputs/results/manual_eval_risk_scores_bugfix_retrain_relaxed.csv`

Comparison artifact:
- `outputs/results/bugfix_retrain_comparison.json`
- `outputs/results/bugfix_retrain_upper_bounds.json`

Observed metrics (manual eval):
- before bugfix (old parser): row88@200 = 0/88, file_function@200 = 39/75
- bugfix parser + old model (no retrain): row88@200 = 5/88, file_function@200 = 36/75
- bugfix parser + retrain: row88@200 = 4/88, file_function@200 = 33/75

Interpretation:
- Bug fixes removed the hard lock on row-level recall (no longer always zero).
- However, single-run retrain still has ranking variance; top-k function recall dropped in this run.
- Post-bugfix coverage upper bound confirms ceiling remains constrained:
  - row88 max hit across all slices: 14/88
  - file+function match (ignoring line) max: 57/75
  - file+function+line max: 11/75

Practical note:
- Parser bugs are fixed and should remain.
- For stable reporting, use multi-seed runs and report mean/std instead of one run only.

