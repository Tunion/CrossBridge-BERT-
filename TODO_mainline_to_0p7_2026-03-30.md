# Mainline 0.7 TODO

Current fixed reference:
- `protofree_sqrt`
- historical manual-set best `F1=0.6343`

This round has already landed the three agreed code paths. They are integrated but remain controllable.

## 1. Closure Completeness Features

Goal:
- Lift false negatives whose slices only keep local `require/deposit/verify` fragments and lose cross-function mechanism closure.

What is implemented:
- Add closure-completeness scalar features into the learned `energy head`.
- Features include role coverage, role-pair connectivity, skeleton retain ratio, mechanism link coverage, and mechanism-complete flag.

Main switch:
- `trainers/train_unsup.py --energy-head-include-closure-features`

## 2. Soft Edge-Type Weighting

Goal:
- Keep cross-function context, but reduce helper/math/summary-edge pollution during message passing instead of hard-pruning slices.

What is implemented:
- Weighted adjacency support in graph tensorization.
- Model/train/eval checkpoint loading can all reuse the same soft edge config.

Main switches:
- `trainers/train_dual_view.py --edge-soft-weight-enable`
- Optional weights:
  - `--edge-weight-local-call-summary`
  - `--edge-weight-shared-object-summary`
  - `--edge-weight-cross-function-state-flow`
  - `--edge-weight-local-call`
  - `--edge-weight-state-summary`

## 3. Manual-Set Line Alignment

Goal:
- Fix misleading line-level diagnosis where exact gold-line matching was effectively unusable.

What is implemented:
- Build nearest statement-line alignment under the same file/function.
- Preserve raw metrics and add aligned recall diagnostics in parallel.
- Export aligned labels CSV during manual evaluation.

Main switches:
- `eval/evaluate_manual_set.py --line-align-enable`
- `--line-align-max-shift`

## Recommended Experiment Order

1. `soft edge weighting` only
2. `closure features` only
3. `soft edge weighting + closure features`
4. Re-check aligned manual-set localization before deciding whether to continue on this branch
