#!/usr/bin/env python3
import csv
import glob
import math
import os
import random
from dataclasses import dataclass
from statistics import mean


def clip(x, lo=-10.0, hi=10.0):
    return max(lo, min(hi, x))


def safe_mean(values, default=0.0):
    vals = [v for v in values if math.isfinite(v)]
    return mean(vals) if vals else default


def parse_simple_yaml(path: str):
    root = {}
    stack = [(0, root)]
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        raw = lines[i].rstrip('\n')
        i += 1
        if not raw.strip() or raw.strip().startswith('#'):
            continue
        indent = len(raw) - len(raw.lstrip(' '))
        line = raw.strip()

        while stack and indent < stack[-1][0]:
            stack.pop()
        container = stack[-1][1]

        if line.startswith('- '):
            val = line[2:].strip()
            if isinstance(container, list):
                container.append(cast_value(val))
            continue

        if ':' not in line:
            continue
        key, rest = line.split(':', 1)
        key = key.strip()
        rest = rest.strip()

        if rest == '':
            # Detect next non-empty line for list vs dict
            j = i
            next_nonempty = None
            while j < len(lines):
                s = lines[j].strip()
                if s:
                    next_nonempty = s
                    break
                j += 1
            new_obj = [] if (next_nonempty and next_nonempty.startswith('- ')) else {}
            if isinstance(container, dict):
                container[key] = new_obj
            stack.append((indent + 2, new_obj))
        else:
            if isinstance(container, dict):
                container[key] = cast_value(rest)

    return root


def cast_value(v: str):
    v = v.strip().strip('"')
    if v.lower() in ('true', 'false'):
        return v.lower() == 'true'
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


@dataclass
class RunResult:
    experiment_id: str
    condition: str
    seed: int
    pressure_mean: float
    pressure_p50: float
    pressure_p90: float
    time_to_silence: int
    identity_entropy: float
    nodes_final: int
    splits_total: int
    prunes_total: int
    status: str
    notes: str


def percentile(vals, p):
    if not vals:
        return float('nan')
    s = sorted(vals)
    idx = int((len(s)-1) * p)
    return s[idx]


def entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        q = c / total
        h -= q * math.log(q + 1e-12, 2)
    return h


def simulate_particle_series(seed: int, steps: int, n=20, dt=0.01):
    rng = random.Random(seed)
    pos = [[rng.random(), rng.random()] for _ in range(n)]
    vel = [[(rng.random()-0.5)*0.1, (rng.random()-0.5)*0.1] for _ in range(n)]
    series = []
    for _ in range(steps+1):
        sig = []
        for i in range(n):
            sig.extend([pos[i][0], pos[i][1], (vel[i][0]+1)/2, (vel[i][1]+1)/2])
        series.append(sig)

        forces = [[0.0, 0.0] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[j][0]-pos[i][0]
                dy = pos[j][1]-pos[i][1]
                d2 = dx*dx + dy*dy + 1e-4
                d = math.sqrt(d2)
                # attraction at distance, repulsion nearby
                f = 0.0008/d2 - 0.00012/(d2*d)
                fx = f*dx/d
                fy = f*dy/d
                forces[i][0] += fx
                forces[i][1] += fy
                forces[j][0] -= fx
                forces[j][1] -= fy

        for i in range(n):
            vel[i][0] = 0.997*vel[i][0] + forces[i][0]
            vel[i][1] = 0.997*vel[i][1] + forces[i][1]
            pos[i][0] += vel[i][0]*dt
            pos[i][1] += vel[i][1]*dt
            for k in (0, 1):
                if pos[i][k] < 0:
                    pos[i][k] = -pos[i][k]
                    vel[i][k] = -vel[i][k]
                if pos[i][k] > 1:
                    pos[i][k] = 2-pos[i][k]
                    vel[i][k] = -vel[i][k]
    return series


def run_online_linear(series, lr=0.001):
    d = len(series[0])
    # Fast diagonal predictor: y_i ~= w_i * x_i + b_i
    w = [0.0 for _ in range(d)]
    b = [0.0 for _ in range(d)]
    pressures = []
    for t in range(len(series)-1):
        x = series[t]
        y = series[t+1]
        yhat = [clip(b[i] + w[i] * x[i], -2.0, 2.0) for i in range(d)]
        err = [y[i]-yhat[i] for i in range(d)]
        p = safe_mean(abs(e) for e in err)
        pressures.append(p)
        for i in range(d):
            ei = clip(err[i], -1.0, 1.0)
            b[i] += lr * ei
            w[i] = clip(w[i] + lr * ei * x[i], -5.0, 5.0)
    return pressures, 0.0, 1, 0, 0


def run_hybrid(series, lr=0.001):
    d = len(series[0])
    n_particles = d // 4
    # One 4x4 specialist per particle to model local position-velocity coupling.
    models = []
    for _ in range(n_particles):
        W = [[0.0 for _ in range(4)] for _ in range(4)]
        b = [0.0 for _ in range(4)]
        models.append((W, b))
    pressures = []
    assign_counts = [0 for _ in range(n_particles)]
    for t in range(len(series)-1):
        x_all = series[t]
        y_all = series[t+1]
        perr = []
        for p in range(n_particles):
            W, b = models[p]
            base = p * 4
            x = x_all[base: base + 4]
            y = y_all[base: base + 4]
            yhat = []
            for i in range(4):
                pred = b[i] + sum(W[i][j] * x[j] for j in range(4))
                yhat.append(clip(pred, -2.0, 2.0))
            err = [y[i] - yhat[i] for i in range(4)]
            for i in range(4):
                ei = clip(err[i], -1.0, 1.0)
                b[i] += lr * ei
                for j in range(4):
                    W[i][j] = clip(W[i][j] + lr * ei * x[j], -3.0, 3.0)
            perr.extend(abs(e) for e in err)
            assign_counts[p] += 1
        pressures.append(safe_mean(perr))

    h = entropy(assign_counts)
    return pressures, h, n_particles, 0, 0


def run_pure_emergence(series, lr=0.001, obs_window=200):
    d = len(series[0])
    rng = random.Random(1234)
    genes = [
        "velocity", "position", "collision", "gradient",
        "periodic", "chaos", "momentum", "boundary"
    ]
    # Each gene sees 25% of dimensions (as in the report idea).
    dims = list(range(d))
    models = []
    for g in genes:
        rng.shuffle(dims)
        gdims = sorted(dims[: max(1, d // 4)])
        w = [0.0 for _ in gdims]
        b = [0.0 for _ in gdims]
        models.append({
            "gene": g,
            "dims": gdims,
            "w": w,
            "b": b,
            "res": 0.0,
            "wins": 0,
        })

    pressures = []
    active = []  # expressed genes
    split_count = 0
    prune_count = 0
    prev = series[0]
    for t in range(len(series)-1):
        x_all = series[t]
        y_all = series[t+1]
        node_errs = []

        # environment signatures for resonance updates
        deltas = [x_all[i] - prev[i] for i in range(d)]
        abs_d = [abs(v) for v in deltas]
        mean_abs_d = safe_mean(abs_d, default=0.0)
        sign_flips = safe_mean(
            1.0 if (x_all[i] - prev[i]) * (prev[i] - (series[t-1][i] if t > 0 else prev[i])) < 0 else 0.0
            for i in range(d)
        )
        boundary_ratio = safe_mean(1.0 if (x_all[i] < 0.05 or x_all[i] > 0.95) else 0.0 for i in range(d))

        for model in models:
            gdims = model["dims"]
            w = model["w"]
            b = model["b"]
            x = [x_all[k] for k in gdims]
            y = [y_all[k] for k in gdims]
            yhat = [clip(b[i] + w[i] * x[i], -2.0, 2.0) for i in range(len(gdims))]
            err = [y[i]-yhat[i] for i in range(len(gdims))]
            mae = safe_mean((abs(e) for e in err), default=1.0)
            node_errs.append((mae, model, x, err))

            # Resonance: gene-specific signal affinity + prediction quality.
            affinity = {
                "velocity": mean_abs_d,
                "position": 1.0 - mean_abs_d,
                "collision": sign_flips,
                "gradient": 1.0 - sign_flips,
                "periodic": max(0.0, 1.0 - abs(mean_abs_d - 0.02) * 20.0),
                "chaos": min(1.0, mean_abs_d * 10.0),
                "momentum": max(0.0, 1.0 - sign_flips * 2.0),
                "boundary": boundary_ratio,
            }[model["gene"]]
            model["res"] = 0.99 * model["res"] + affinity + 1.0 / (mae + 1e-6)

        if t < obs_window:
            winners = node_errs
        else:
            if not active:
                # Express top-2 genes once the observation window completes.
                active = sorted(models, key=lambda m: m["res"], reverse=True)[:2]
            active_set = {id(m) for m in active}
            candidate = [z for z in node_errs if id(z[1]) in active_set]
            winners = [min(candidate, key=lambda z: z[0])]

        combined_err = []
        for mae, model, x, err in winners:
            gdims = model["dims"]
            w = model["w"]
            b = model["b"]
            combined_err.extend(abs(e) for e in err)
            for i in range(len(gdims)):
                ei = clip(err[i], -1.0, 1.0)
                b[i] += lr * ei
                w[i] = clip(w[i] + lr * ei * x[i], -5.0, 5.0)
            model["wins"] += 1

        pressures.append(safe_mean(combined_err, default=0.0))

        # split/prune over active genes
        if t > obs_window and active and len(active) < 4 and pressures[-1] > 0.20 and t % 300 == 0:
            best = max(active, key=lambda m: m["res"])
            new_dims = best["dims"][::2] if len(best["dims"]) > 2 else best["dims"][:]
            m = {
                "gene": best["gene"] + "_child",
                "dims": new_dims,
                "w": [0.0 for _ in new_dims],
                "b": [0.0 for _ in new_dims],
                "res": best["res"] * 0.5,
                "wins": 0,
            }
            active.append(m)
            split_count += 1
        if t > obs_window and active and len(active) > 2 and pressures[-1] < 0.08 and t % 300 == 0:
            active.sort(key=lambda m: m["wins"])
            active.pop(0)
            prune_count += 1
        prev = x_all

    if not active:
        active = sorted(models, key=lambda m: m["res"], reverse=True)[:2]
    identity_scores = [m["wins"] + 1 for m in active]
    h = entropy(identity_scores)
    return pressures, h, len(active), split_count, prune_count


def time_to_silence(pressures, threshold=0.03):
    for i, p in enumerate(pressures):
        if p <= threshold:
            return i + 1
    return len(pressures)


def run_manifest(manifest_path):
    cfg = parse_simple_yaml(manifest_path)
    exp_id = cfg['experiment_id']
    name = cfg['name']
    seeds = int(cfg.get('seed_count', 5))
    steps = int(cfg['environment'].get('steps', 2000))
    model_cfg = cfg['model']

    out = []
    for seed in range(1, seeds + 1):
        series = simulate_particle_series(seed=seed, steps=steps)
        if exp_id == 'E2' and not model_cfg.get('differentiation_enabled', False):
            pressures, h, nodes, splits, prunes = run_online_linear(series)
        elif exp_id == 'E3' and model_cfg.get('differentiation_enabled', False):
            pressures, h, nodes, splits, prunes = run_hybrid(series)
        else:
            pressures, h, nodes, splits, prunes = run_pure_emergence(series)

        out.append(RunResult(
            experiment_id=exp_id,
            condition=name,
            seed=seed,
            pressure_mean=mean(pressures),
            pressure_p50=percentile(pressures, 0.50),
            pressure_p90=percentile(pressures, 0.90),
            time_to_silence=time_to_silence(pressures),
            identity_entropy=h,
            nodes_final=nodes,
            splits_total=splits,
            prunes_total=prunes,
            status='ok',
            notes=cfg.get('notes', ''),
        ))
    return out


def write_results(path, results):
    fields = [
        'experiment_id', 'condition', 'seed', 'pressure_mean', 'pressure_p50', 'pressure_p90',
        'time_to_silence', 'identity_entropy', 'nodes_final', 'splits_total', 'prunes_total',
        'status', 'notes'
    ]
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)


def summarize(results):
    by_exp = {}
    for r in results:
        by_exp.setdefault(r.experiment_id, []).append(r)
    lines = []
    for exp in sorted(by_exp):
        rs = by_exp[exp]
        lines.append(
            f"{exp}: pressure_mean={mean(r.pressure_mean for r in rs):.4f}, "
            f"p50={mean(r.pressure_p50 for r in rs):.4f}, p90={mean(r.pressure_p90 for r in rs):.4f}, "
            f"tts={mean(r.time_to_silence for r in rs):.1f}, entropy={mean(r.identity_entropy for r in rs):.3f}"
        )
    return '\n'.join(lines)


def main():
    manifests = sorted(glob.glob('experiments/manifests/*.yaml'))
    all_results = []
    for m in manifests:
        res = run_manifest(m)
        all_results.extend(res)
    os.makedirs('experiments/results', exist_ok=True)
    write_results('experiments/results/runs.csv', all_results)
    print(summarize(all_results))


if __name__ == '__main__':
    main()
