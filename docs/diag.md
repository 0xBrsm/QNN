# Diag — capacity diagnostics for trained QNN policies

`qnn.diag` is the unified diagnostic package for trained policies. Use it after
training to interpret what an ablation actually showed, decide what to ablate
next, or check whether an 8-epoch comparison is reliable.

For procedural guidance on *when* and *how* to run the suite, see the `diag`
skill (`agents/skills/diag/`). This doc is the technical reference.

## CLI

The package has two subcommands:

### `analyze` — unified per-head analysis

```bash
python -m qnn.diag analyze \
    --run-dir  runs/bc/<run> \
    --cache-dir artifacts/collect/qwd \
    --heads    attack,look,move \
    --segment  engaged \
    --out      runs/bc/<run>/diag_report.json
```

`--run-dir` must contain `config/probe.json` and a checkpoint under
`checkpoints/`.  `--cache-dir` is the root cache directory; the resident
source is built from `<cache-dir>/precomputed_val`.

`--heads` defaults to `attack,look,move`; `weapon` is included automatically
when `qnn.diag.weapon.analyze` is importable. `--segment` is `engaged`
(frames where `act.target != 0`) or `all` (no filter).

The output JSON follows the Phase-2 schema:

```json
{
  "run":     "<run_id>",
  "cache":   "<cache_dir>",
  "segment": "engaged",
  "heads":   { "attack": {...}, "look": {...}, "move": {...} },
  "meta":    { "n_episodes": N, "n_frames": M }
}
```

### `diagnostic` — checkpoint-based diagnostic suite (legacy)

```bash
python -m qnn.diag diagnostic \
    --checkpoint runs/bc/<run>/checkpoints/best_<run_id>.pth \
    --data-dir   artifacts/collect/qwd \
    --report-out runs/bc/<run>/diagnostics.md
```

Default output covers history, convergence, rank, ablation, gradients,
participation, attention. Slow tools (`pruning`, `linear_probe`) are opt-in:

```bash
python -m qnn.diag diagnostic ... --include pruning,linear_probe
```

The legacy flat-arg form (`python -m qnn.diag --checkpoint …`) is still
accepted for backward compatibility and routes to the same `run_report` path.

## Per-head `analyze()` dispatch

`qnn.diag.analyze.run_analyze` dispatches to each head module's `analyze()`
function. Each head module consolidates the canonical per-head analysis
scripts:

| Head module | Functions consolidated from |
|-------------|----------------------------|
| `qnn.diag.attack` | `fire_target_conditional`, `attack_offset_distribution`, `attack_empirical_range`, `attack_input_ablation` |
| `qnn.diag.look` | `look_prior_fit`, `look_prior_explore4`, `look_history_attention`, `look_horizon_ceiling`, `look_metric_references`, `look_target_intersection`, `aim_point_z_offset`, `look_ground_spin`, `look_aim_prior_decode` |
| `qnn.diag.move` | `momentum_baseline`, `stream_dynamics`, `jump_discrim`, `jump_onset_probe`, `rate_fidelity` |
| `qnn.diag.weapon` | `corpus_stats`, `intent_decompose`, `intent_psth`, `gate_sweep`, `switch_gated`, `anticip_roc`, `decode_sweep`, `switch_window_roc`, `switch_decompose`, `switch_timing_detail`, `switch_leadtime`, `switch_leak_test`, `switch_vs_token`, `switchframe_decomp`, `when_switch_detect` |

`analyze()` in each module runs the subset that is compatible with a loaded
policy + resident source. Functions requiring additional inputs (NPZ caches,
GBT fits, two-corpus rate comparisons, etc.) remain standalone named
functions with their own signatures.

## Shared kernels

`qnn.diag.loader.load_policy` is the canonical checkpoint loader used by
every analysis script, bench probe, and decode-fit sweep. It finds the best
checkpoint, applies compat shims, loads via `QNNPolicy.load`, sets
`policy.model.eval()` and `policy.input_mask = True`, and installs the run's
polar look-grid from `config/look_grid.json` if present.

`qnn.diag.look_data` and `qnn.diag.move_metrics` are shared kernels used
across multiple analysis functions within their respective head modules.

## Operative filters (correctness requirement)

Each head applies its own operative-frame filter. **Do not remove or relax
these at call sites.**

| Head | Filter |
|------|--------|
| attack | `op = input_mask & 1` (bit 0 of `act.input_mask`) |
| move (jump) | `jump_feas = ((im >> 7) \| (im >> 6)) & 1` (ground-jump or swim-up feasible) |
| look | no filter (look is not `input_mask`-gated) |
| weapon | `label != 0` (weapon-present frames); NOT `input_mask & 1` — bit 0 is the attack bit |

## `diagnostic` subcommand sections

A typical markdown report produced by `diagnostic` has these sections:

### Training history
Per-epoch best metrics, train/val gap progression, "still improving" flags.
Free signal — no model load needed.

### Convergence reliability
Late-epoch slope, classification, asymptote projection (if fittable). Use
this to flag biased comparisons before drawing conclusions. Empirical thresholds:

- Late-epoch slope < 0.002 / epoch → **converged**, comparisons reliable
- Late-epoch slope < 0.005 / epoch → **near**, comparisons OK with caveats
- Late-epoch slope ≥ 0.005 / epoch → **descending**, undertrained; comparisons biased toward whichever config converges faster

Two runs are comparable if both are at least `near` AND their slopes differ
by less than 0.002.

### Effective rank
SVD effective rank for every Linear. Sorted by `frac` ascending —
most-overparameterized first. Layers with frac < 0.7 are candidates for
low-rank factorization or width reduction.

### Layer ablation
Each submodule's val-loss delta when zeroed. Larger delta = more essential.

### Gradient norms
Per-parameter gradient norm during one supervised batch. Aggregated by
module. The summary identifies the fraction of layers with near-zero gradient
(cruft).

### Head bottleneck activations
Participation ratio + dead-unit count + activation fire-rate distribution per
head. Useful supporting signal but not a primary cut decision (see Pitfalls).

### Attention head patterns
Per-head entropy + cross-head similarity. Combined with per-attention-head
ablation tells you whether heads are doing distinct work or redundant.

### Per-attention-head ablation
Zeros each head one at a time. **Always interpret across multiple seeds** —
single-seed asymmetry is initialization-driven, not architectural.

### Pruning sensitivity (slow, opt-in)
Per-neuron val-loss delta. Sort by impact, find K such that top-K explain 90%
of impact. Direct measurement of what's essential.

### Linear probe (slow, opt-in)
Train a logistic regression on frozen head-input features. Compare to
trained-head F1. Gap = how much the head's nonlinearity is doing.

## Pitfalls

### PR is not a cut decision
A head with PR=5 still benefited from B=192 in our sweep. PR measures
activation correlation, not necessity. For "what can I cut," use `pruning`
or `ablation`, not `participation`.

### Single-seed attention asymmetry is illusory
With 2 heads and 1 seed you'll always see one head "winning" — but which one
is seed-dependent. Run across 3+ seeds before concluding architectural
redundancy. The `attention` diag at 1 seed is descriptive, not prescriptive.

### Low ablation impact ≠ unused
A head can have small ablation impact (its forward output mostly redundant
with a strong prior) yet contribute via training-time gradient pressure on
the encoder. The `look` head exemplifies this. To distinguish, run the head
with `loss_weight=0` and see if metrics on *other* heads change. Use
`compare_runs` to flag if the comparison is convergence-biased.

### 8-epoch ablations on an undertrained baseline
This BC setup typically isn't converged at 8 epochs (slope ≈ −0.005 at ep7).
For decisions near the noise floor, either run a calibration at 16 epochs or
trust the asymptote projection from `convergence.compare_runs`. Most ablation
rankings hold across this bias when both configs are similarly undertrained,
but the absolute gap may overstate the asymptotic delta.

## Files

- `qnn/diag/__main__.py`, `qnn/diag/cli.py` — CLI entry point and subcommand parsers
- `qnn/diag/analyze.py` — `run_analyze` dispatcher (Phase-2 `analyze` subcommand)
- `qnn/diag/report.py` — `run_report` aggregator (`diagnostic` subcommand)
- `qnn/diag/loader.py` — `load_policy`, shared by all analysis scripts
- `qnn/diag/attack.py`, `look.py`, `move.py`, `weapon.py` — per-head analysis modules
- `qnn/diag/look_data.py`, `move_metrics.py` — shared computation kernels
- `qnn/diag/{history,convergence,rank,ablation,gradients,participation,attention,pruning,linear_probe}.py` — suite tools
- `agents/skills/diag/SKILL.md` — when-to-use procedural guidance (outside `src/`, full path)
