# Pressure-Driven Self-Organizing Neural Architecture — Re-Plan (From Zero)

## 0) Reframed Goal
Build a **falsifiable, staged research program** that can answer one core question:

> Can architecture and specialization emerge from environmental pressure alone, without backpropagation, labels, or hand-designed role assignments?

This re-plan shifts from a single prototype report to a full evidence pipeline: baseline controls, ablations, reproducibility, and scale-up criteria.

---

## 1) Research Thesis, Claims, and Falsification Criteria

### 1.1 Thesis
A homogeneous population of genome-carrying nodes can self-differentiate, self-restructure, and reduce prediction pressure through local rules driven by environment dynamics.

### 1.2 Primary claims to test
1. **Differentiation claim**: Distinct node identities emerge reliably under structured environments.
2. **Utility claim**: Differentiation improves pressure reduction versus non-differentiating controls.
3. **Nativity claim**: Nodes specialized in environment A underperform in environment B relative to native B nodes.
4. **Scalability claim**: Emergence persists as dimensionality and node count increase.
5. **Efficiency claim**: A GPU-native implementation yields better pressure-per-joule than baseline alternatives.

### 1.3 Falsification thresholds (pre-registered)
- If identity distributions are statistically indistinguishable from random in >= 3 structured environments, differentiation claim fails.
- If pressure reduction is not better than fixed-architecture controls by a pre-set margin (e.g., >= 15%), utility claim fails.
- If cross-environment nativity gap is not significant across seeds, nativity claim fails.
- If gains collapse with scale (larger node counts or higher-dimensional signals), scalability claim fails.

---

## 2) Experimental Program Architecture

### 2.1 Track structure
Run four parallel tracks:

- **Track A: Mechanism validation (small scale, high observability)**
- **Track B: Robustness and ablation (multi-seed statistical testing)**
- **Track C: Systems implementation (GPU-native kernel and profiling)**
- **Track D: External validity (new environments beyond 2D physics)**

### 2.2 Stage gates
Proceed only when prior gate passes:

- **Gate 1**: deterministic reproducibility + logging completeness
- **Gate 2**: differentiation reproduced over seeds
- **Gate 3**: ablation evidence supports mechanism (observation window, sparse gene sensitivity, splitting/pruning)
- **Gate 4**: nativity validated on at least 3 environment families
- **Gate 5**: GPU version matches CPU semantics and beats baseline efficiency

---

## 3) Environment Suite (From Simple to Complex)

### 3.1 Tiered environment ladder
1. **Tier 0 (sanity)**: synthetic waveforms (periodic, trend, burst, chaotic transitions)
2. **Tier 1 (current domain)**: 2D particle physics (bounded, collisions, variable timesteps)
3. **Tier 2 (hybrid dynamics)**: switched-rule physics (phase transitions mid-run)
4. **Tier 3 (partially observed)**: occluded/missing channels and delayed signals
5. **Tier 4 (cross-modal toy world)**: paired streams (e.g., dynamics + symbolic events)

### 3.2 Why this ladder
- isolates failure modes early,
- prevents overfitting conclusions to one environment,
- enables clear scaling diagnostics.

---

## 4) Metrics and Statistical Protocol

### 4.1 Core metrics
- Mean and percentile **pressure** over time
- **Time-to-silence band** (steps to reach target pressure interval)
- **Identity entropy** and diversity balance
- **Split/prune efficiency** (net structural cost vs pressure gain)
- **Cross-environment transfer matrix** (nativity map)
- **Energy metrics** (joules per 1% pressure reduction, on GPU track)

### 4.2 Statistical design
- Minimum seeds per condition: **30** (unless compute-limited; then report power shortfall)
- Report confidence intervals + effect sizes, not only p-values
- Use pre-declared primary endpoint per stage to prevent metric shopping

### 4.3 Baselines
- Fixed-architecture predictor with comparable parameter budget
- Same node field but no differentiation (genes disabled)
- Same node field but no structural evolution (split/prune disabled)
- Random identity assignment control

---

## 5) Ablation Matrix (Mandatory)

Evaluate impact of each mechanism by removing one at a time:

1. Observation window removed
2. Sparse gene sensitivity (25%) replaced by full-dimension sensitivity
3. Resonance decay removed
4. Split policy removed
5. Prune policy removed
6. Gene thresholds homogenized vs adaptive
7. Local-only pressure vs any global pressure sharing

Expected output: ranked contribution chart with interaction notes.

---

## 6) Systems Plan (GPU-Native First-Class Workstream)

### 6.1 Implementation phases
- **P0**: CPU reference implementation (authoritative semantics)
- **P1**: CUDA kernels for pressure, resonance accumulation, and differentiation checks
- **P2**: CUDA kernels for split/prune bookkeeping
- **P3**: fused kernels + memory layout optimization

### 6.2 Parity and performance checks
- CPU/GPU numerical parity suite on fixed seeds
- Throughput benchmarks: nodes/sec, steps/sec
- Efficiency benchmarks: pressure reduction per joule

### 6.3 Tooling
- deterministic seed management
- structured experiment manifests
- artifact logging (metrics, traces, checkpoints)

---

## 7) Timeline and Deliverables

### Phase 1 (Weeks 1–2): Foundation
- finalize pre-registered hypotheses, metrics, falsification rules
- build reproducible experiment harness and logging schema
- deliverable: **Protocol v1 + Reproducibility report**

### Phase 2 (Weeks 3–5): Mechanism Validation
- run Tier 0/1 environments across seeds
- execute mandatory ablations + controls
- deliverable: **Mechanism Evidence Pack**

### Phase 2.5 (Week 6): MNIST Competency Gate
- run real MNIST supervised benchmark as a competency checkpoint
- requirement: >98% test accuracy before advancing scaling claims
- deliverable: **MNIST Gate Report**

### Phase 3 (Weeks 6–8): Nativity and Robustness
- run cross-environment transfer matrix (Tier 1/2)
- test phase transitions and recovery behavior
- deliverable: **Nativity and Stability report**

### Phase 4 (Weeks 9–11): GPU Realization
- implement and validate CUDA path against CPU semantics
- profile efficiency against baseline models
- deliverable: **GPU parity + efficiency report**

### Phase 5 (Week 12): Publication Draft
- integrated results, limitations, and failure analysis
- deliverable: **Research Report v2 (submission-ready)**

---

## 8) Risk Register and Mitigations

1. **False emergence** (metric artifacts)
   - mitigation: stronger controls, hidden test streams, and holdout environments.
2. **Gene collapse** (single identity dominates)
   - mitigation: adaptive thresholds, diversity regularizers, constrained expression quotas for diagnostics only.
3. **Unstable scaling**
   - mitigation: curriculum environments, resource-aware split/prune limits.
4. **GPU semantic drift from CPU**
   - mitigation: lockstep parity tests per kernel rollout.
5. **Compute bottlenecks**
   - mitigation: prioritize ablations by expected information gain.

---

## 9) Decision Criteria for “Go / No-Go”

Advance to larger-scale investment only if all are true:

- differentiation replicated in >= 3 environment families,
- utility over fixed controls exceeds target margin,
- nativity matrix shows consistent diagonal dominance,
- GPU path reaches parity and demonstrates efficiency advantage.

If any criterion fails, pause scaling and iterate on mechanism design rather than expanding scope.

---

## 10) Immediate Next Actions (This Week)

1. Write pre-registration document (claims, thresholds, endpoints).
2. Implement experiment manifest format (yaml/json) with seed locking.
3. Build baseline/control runners.
4. Run a pilot (5 seeds, Tier 0 and Tier 1) to validate instrumentation.
5. Freeze protocol and launch full-seed runs.

This gives a clean restart: hypothesis-first, statistically grounded, reproducible, and ready for credible external review.

---

## 10.5) Publication-Readiness Checklist

- Freeze code + manifests + seeds used for headline claims.
- Include confidence intervals and effect sizes for all primary endpoints.
- Report negative findings and failure cases (no cherry-picking).
- Provide exact hardware/runtime settings and reproducibility instructions.
- Include competency benchmarks (MNIST gate >98%) separate from core emergence claims.

---

## 11) Concrete Experiment Set (Execute First)

The following experiment set translates the plan into runnable studies.

### E1 — Pure Emergence Baseline (No Hand-Crafted Macro Roles)
- Setup: homogeneous node field, genome enabled, split/prune enabled.
- Environments: Tier 0 and Tier 1.
- Seeds: 30.
- Success criteria:
  - non-random identity distribution in structured streams,
  - >= 15% pressure improvement over random identity control.

### E2 — Guided Scaffold Baseline (Fixed Good Architecture)
- Setup: fixed macro architecture (input -> latent field -> output), but no differentiation.
- Environments: Tier 0 and Tier 1.
- Seeds: 30.
- Success criteria:
  - stable training dynamics,
  - defines reference pressure and compute cost.

### E3 — Hybrid Model (Scaffold + Differentiation)
- Setup: same macro scaffold as E2, differentiation enabled in latent field.
- Environments: Tier 0 and Tier 1.
- Seeds: 30.
- Success criteria:
  - outperforms E2 on pressure and/or time-to-silence,
  - maintains identity diversity (no single-gene collapse).

### E4 — Hybrid + Structural Evolution
- Setup: same as E3, add split/prune in latent field only.
- Environments: Tier 1 and Tier 2.
- Seeds: 30.
- Success criteria:
  - better pressure-per-node than E3,
  - bounded structural growth (no runaway expansion).

### E5 — Nativity Matrix Test
- Setup: train specialized populations in A/B/C environments, test cross-transfer.
- Environments: three families minimum (e.g., smooth physics, high collision, switched-rule).
- Seeds: 30 per family.
- Success criteria:
  - transfer matrix diagonal dominance,
  - statistically significant native vs foreign performance gaps.

### E6 — Phase Transition Recovery
- Setup: switch rules mid-run (Tier 2) and track recovery.
- Conditions: with and without de-differentiation mechanism.
- Seeds: 30.
- Success criteria:
  - recovery to pre-switch pressure band within fixed step budget,
  - lower collapse rate in de-differentiation condition.

### E7 — Ablation Priority Pass
- Setup: run top 4 high-value ablations first (observation window, sparse sensitivity, decay, split).
- Environments: Tier 0 and Tier 1.
- Seeds: 30.
- Success criteria:
  - ranked mechanism contribution with effect sizes,
  - clear keep/remove recommendations.

### E8 — CPU vs GPU Parity + Efficiency
- Setup: run identical manifests on CPU reference and CUDA implementation.
- Environments: Tier 1.
- Seeds: 10 for parity, 30 for throughput/energy.
- Success criteria:
  - parity within tolerance envelope,
  - measurable pressure-per-joule benefit on GPU.

### Execution order
1. E2 (scaffold reference)
2. E1 (pure emergence)
3. E3 (hybrid)
4. E4 (hybrid + restructuring)
5. E7 (ablation priority)
6. E5 (nativity)
7. E6 (phase transitions)
8. E8 (GPU parity + efficiency)

### Minimum decision package after first cycle
- Winner model class (pure, scaffold, or hybrid)
- Mechanisms kept for v2
- Go/No-Go recommendation with evidence table
