# Experiment Starter Kit

This folder starts execution for the plan in `REPLAN_FROM_SCRATCH.md`.

## Layout
- `manifests/`: run definitions for each experiment condition.
- `results/`: output metrics and run summaries.

## Initial sequence (recommended)
1. `e2_scaffold_baseline.yaml`
2. `e1_pure_emergence.yaml`
3. `e3_hybrid.yaml`

## Run
```bash
python experiments/run_experiments.py
```

## MNIST gate (>98%)
```bash
python experiments/run_mnist_mlp.py
```
Result file: `experiments/results/mnist_results.json`

## Analyze
```bash
python experiments/analyze_results.py
```
Result file: `experiments/results/analysis_report.md`

## Minimal run contract
Each run should emit:
- `pressure_mean`
- `pressure_p50`
- `pressure_p90`
- `time_to_silence`
- `identity_entropy`
- `nodes_final`
- `splits_total`
- `prunes_total`
- `seed`
- `status`

Store one CSV row per run in `results/runs.csv`.
