# Diag — capacity diagnostics for trained QNN policies

`qnn.diag` is the unified diagnostic package for trained policies. Use it after training to interpret what an ablation actually showed, decide what to ablate next, or check whether an 8-epoch comparison is reliable.

For procedural guidance on *when* and *how* to run the suite, see the `diag` skill (`agents/skills/diag/`). This doc is the technical reference.

## Quick start

```bash
python -m qnn.diag \
    --checkpoint runs/bc/<run>/checkpoints/bc_best_model.pth \
    --data-dir artifacts/collect/qwd \
    --report-out runs/bc/<run>/diagnostics.md
```

Default output covers history, convergence, rank, ablation, gradients, participation, attention. Slow tools (`pruning`, `linear_probe`) are opt-in:

```bash
python -m qnn.diag ... --include pruning,linear_probe
```

## Submodules

Each module is independently importable. Modules without `torch` work outside the trainer container.

| Module | Question answered | Cost | Needs torch |
|--------|-------------------|------|-------------|
| `qnn.diag.history` | What did training look like across epochs? | instant | no |
| `qnn.diag.convergence` | Is the run converged? Is comparison X vs Y reliable? | instant | no |
| `qnn.diag.rank` | Are Linear weights low-rank / compressible? | seconds | yes |
| `qnn.diag.gradients` | Where is gradient flowing? Which params are dead? | seconds | yes |
| `qnn.diag.ablation` | Which submodules are essential? | seconds–minutes | yes |
| `qnn.diag.participation` | Are head bottlenecks utilized? Dead units? | tens of seconds | yes |
| `qnn.diag.attention` | Are attention heads redundant? Asymmetric? | tens of seconds | yes |
| `qnn.diag.pruning` | Which neurons are essential vs redundant? | minutes | yes |
| `qnn.diag.linear_probe` | Are encoder features linearly separable per task? | minutes | yes |

## Convergence — the critical pre-check

Before trusting any other comparison, run the convergence check on both runs being compared. The empirical thresholds (calibrated on this BC setup):

- Late-epoch slope < 0.002 / epoch → **converged**, comparisons reliable
- Late-epoch slope < 0.005 / epoch → **near**, comparisons OK with caveats
- Late-epoch slope ≥ 0.005 / epoch → **descending**, undertrained; comparisons biased toward whichever config converges faster

Two runs are comparable if both are at least `near` AND their slopes differ by less than 0.002. Otherwise:

- `compare_runs` returns `reliable=False` and a warning
- The asymptote-projected delta (from the exponential decay fit) is more trustworthy than the last-epoch delta
- The comparison may need a calibration run at higher epochs to settle

## Reading the report

A typical markdown report has these sections:

### Training history
Per-epoch best metrics, train/val gap progression, "still improving" flags. Free signal — no model load needed.

### Convergence reliability
Late-epoch slope, classification, asymptote projection (if fittable). Use this to flag biased comparisons before drawing conclusions.

### Effective rank
SVD effective rank for every Linear. Sorted by `frac` ascending — most-overparameterized first. Layers with frac < 0.7 are candidates for low-rank factorization or width reduction.

### Layer ablation
Each submodule's val-loss delta when zeroed. Larger delta = more essential. Standard ranking on a healthy v22 model: encoder.blocks.0 > target_pointer > encoder.blocks.1 > weapon_head > gru > move_head > attack_head > look_head.

### Gradient norms
Per-parameter gradient norm during one supervised batch. Aggregated by module. The summary identifies the fraction of layers with near-zero gradient (cruft).

### Head bottleneck activations
Participation ratio + dead-unit count + activation fire-rate distribution per head. Useful supporting signal but not a primary cut decision (see Pitfalls).

### Attention head patterns
Per-head entropy + cross-head similarity. Combined with per-attention-head ablation tells you whether heads are doing distinct work or redundant.

### Per-attention-head ablation
Zeros each head one at a time. **Always interpret across multiple seeds** — single-seed asymmetry is initialization-driven, not architectural.

### Pruning sensitivity (slow, opt-in)
Per-neuron val-loss delta. Sort by impact, find K such that top-K explain 90% of impact. Direct measurement of what's essential.

### Linear probe (slow, opt-in)
Train a logistic regression on frozen head-input features. Compare to trained-head F1. Gap = how much the head's nonlinearity is doing.

## Pitfalls

### PR is not a cut decision
A head with PR=5 still benefited from B=192 in our sweep. PR measures activation correlation, not necessity. For "what can I cut," use `pruning` or `ablation`, not `participation`.

### Single-seed attention asymmetry is illusory
With 2 heads and 1 seed you'll always see one head "winning" — but which one is seed-dependent. Run across 3+ seeds before concluding architectural redundancy. The `attention` diag at 1 seed is descriptive, not prescriptive.

### Low ablation impact ≠ unused
A head can have small ablation impact (its forward output mostly redundant with a strong prior) yet contribute via training-time gradient pressure on the encoder. The `look` head exemplifies this: cos_sim_look saturates at B=16 but the look gradient still shapes how the encoder encodes engagement geometry. To distinguish, run the head with `loss_weight=0` and see if metrics on *other* heads change. Use `compare_runs` to flag if the comparison is convergence-biased.

### 8-epoch ablations on an undertrained baseline
This BC setup typically isn't converged at 8 epochs (slope ≈ −0.005 at ep7). For decisions near the noise floor, either run a calibration at 16 epochs or trust the asymptote projection from `convergence.compare_runs`. Most ablation rankings hold across this bias when both configs are similarly undertrained, but the absolute gap may overstate the asymptotic delta.

## Files

- `qnn/diag/__init__.py` — package docstring + submodule index
- `qnn/diag/cli.py`, `qnn/diag/__main__.py` — CLI
- `qnn/diag/report.py` — aggregator
- `qnn/diag/{module}.py` — per-tool implementations
- `agents/skills/diag/SKILL.md` — when-to-use procedural guidance (outside `src/`, full path)
