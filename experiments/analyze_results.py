#!/usr/bin/env python3
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def mean(vals):
    return sum(vals) / len(vals)


def stdev(vals):
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0


def ci95(vals):
    m = mean(vals)
    s = stdev(vals)
    half = 1.96 * s / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return m - half, m + half


def load_runs(path):
    rows = list(csv.DictReader(open(path)))
    metrics = defaultdict(lambda: defaultdict(list))
    for r in rows:
        e = r['experiment_id']
        for k in ['pressure_mean', 'pressure_p50', 'pressure_p90', 'time_to_silence', 'identity_entropy']:
            metrics[e][k].append(float(r[k]))
    return rows, metrics


def main():
    rows, metrics = load_runs('experiments/results/runs.csv')
    mnist = json.loads(Path('experiments/results/mnist_results.json').read_text())

    lines = []
    lines.append('# Results Analysis\n')
    lines.append(f'- Total synthetic runs: **{len(rows)}**')
    lines.append(f"- MNIST best single accuracy: **{mnist['best_single_model']['test_accuracy']:.4f}**")
    lines.append(f"- MNIST top-3 ensemble accuracy: **{mnist['ensemble_top3_test_accuracy']:.4f}**\n")

    lines.append('## Synthetic experiment summary (mean ± 95% CI)\n')
    for exp in sorted(metrics):
        pm = metrics[exp]['pressure_mean']
        tts = metrics[exp]['time_to_silence']
        ent = metrics[exp]['identity_entropy']
        pm_ci = ci95(pm)
        tts_ci = ci95(tts)
        ent_ci = ci95(ent)
        lines.append(f"### {exp}")
        lines.append(f"- pressure_mean: {mean(pm):.4f} (95% CI: {pm_ci[0]:.4f} to {pm_ci[1]:.4f})")
        lines.append(f"- time_to_silence: {mean(tts):.1f} (95% CI: {tts_ci[0]:.1f} to {tts_ci[1]:.1f})")
        lines.append(f"- identity_entropy: {mean(ent):.4f} (95% CI: {ent_ci[0]:.4f} to {ent_ci[1]:.4f})\n")

    if 'E2' in metrics and 'E3' in metrics:
        e2 = mean(metrics['E2']['pressure_mean'])
        e3 = mean(metrics['E3']['pressure_mean'])
        rel = (e2 - e3) / e2 * 100.0
        lines.append(f"- E3 vs E2 pressure improvement: **{rel:.2f}%**")

    out = Path('experiments/results/analysis_report.md')
    out.write_text('\n'.join(lines))
    print(out)


if __name__ == '__main__':
    main()
