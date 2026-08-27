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

`--heads` defaults to `attack,look,move`. The categorical attack-with output is
analyzed by `qnn.diag.attack`; there is no separate action-weapon diagnostic.
`--segment` is `engaged`
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

### `spatial_reconstruction` — pre-training geometry fidelity

Spatial-v2 can be evaluated before training by comparing its quantized token
payload with independent dense hull-1 traces from real-map demos:

```bash
python -m qnn.diag.spatial_reconstruction run \
  --worker assets/bin/qw_demo_worker --assets assets \
  --demo path/to/demo.qwd --sidecar spatial.jsonl

python -m qnn.diag.spatial_reconstruction score \
  povdmm4.jsonl dm2.jsonl dm6.jsonl \
  --max-false-block-rate-all 0.01 \
  --max-blocked-early-gt-32-rate 0.03
```

The scorer reconstructs directional supporting-plane depth from token bytes
alone. It reports error, obstacle misses, per-elevation results, and
near-obstacle classification at 32–256 units. False blocks are reported both
conditional on a truly open direction (`false_block_rate_given_open`) and as
a fraction of every sampled direction (`false_block_rate_all`), with raw
confusion counts. This tests preservation of local collision geometry; it does
not replace a trained policy ablation or prove full map-topology recovery. See
[`wire.12`](contracts/wire/wire.12.md#reconstruction-diagnostic) for the
payload, optional acceptance gates, and current three-map reference results.

### `static_map_memory` — immutable map-load geometry memory

This diagnostic tests whether exact static hull-1 faces can be encoded once at
map load and queried from moving poses without rebuilding map tokens. First
build the diagnostic worker and export one face table per map:

```bash
src/engine/build/build_qw_demo_worker.sh

python -m qnn.diag.static_map_memory dump-faces \
  --worker assets/bin/qw_demo_worker \
  --demo-dir artifacts/corpus/qwd \
  --manifest artifacts/corpus/qwd_probe_manifest.ndjson \
  --asset-root assets --maps dm2 dm4 dm6 \
  --out-dir artifacts/diag/static_map_memory/faces
```

Schema-8 `spatial_reconstruction run` sidecars include
`static_atlas_code` and `static_atlas_distance`. These fields use the exact
static world and exclude translated mover instances, so movers cannot make an
immutable map table appear wrong. Fit either learned query arm against those
targets:

```bash
python -m qnn.diag.static_map_memory train \
  --faces-dir artifacts/diag/static_map_memory/faces \
  --sidecars artifacts/diag/static_map_memory/sidecars/*.jsonl \
  --position-mode relative_bias --device cuda \
  --output-dir runs/eval/_static_map_memory_relative_v1
```

`absolute` uses only static and query Fourier positions through dot-product
attention. `relative_bias` adds pose-dependent center distance, ray alignment,
and face-orientation bias while retaining the same cached map K/V. Reports hash
the cache before and after every validation query and include map-encode time,
cache bytes, and query latency.

Use the deterministic ceiling to distinguish missing geometry from failed
learned routing:

```bash
python -m qnn.diag.static_map_memory oracle-routing \
  --faces-dir artifacts/diag/static_map_memory/faces \
  --sidecars artifacts/diag/static_map_memory/sidecars/*.jsonl \
  --face-limits 16,32,64,128,256,512 \
  --output runs/eval/_static_map_memory_oracle_v1/report.json
```

The oracle intersects rays exactly against either the full immutable face table
or the nearest-N faces selected once per pose. It uses the same reconstruction
acceptance thresholds as the shipped spatial atlas. This command measures the
information and routing ceiling; it does not measure policy utility.

### `static_probe_memory` — cached directional field

This diagnostic tests a query-conditioned static field after flat hull faces
fail. Build a dense 3D navmesh probe table at map load:

```bash
python -m qnn.bc.probe_atlas \
  --worker assets/bin/qw_demo_worker \
  --demo-dir artifacts/corpus/qwd_probe \
  --manifest artifacts/corpus/qwd_probe_manifest.ndjson \
  --maps dm2 dm4 dm6 --spacing 32 --z-spacing 32 \
  --out-dir artifacts/diag/static_probe_memory/navatlas_s32z32
```

Each probe stores all eleven world-anchored panorama bands as one directional
function. The query route selects K probe indices, evaluates one band at the
ray's world yaw, and fuses K values. It never rolls or projects a panorama at
tick time. Score the target-informed information ceiling and deterministic
fusion arms before training:

```bash
python -m qnn.diag.static_probe_memory \
  --probe-dir artifacts/diag/static_probe_memory/navatlas_s32z32 \
  --sidecars artifacts/diag/static_map_memory/sidecars/*.jsonl \
  --k 9 \
  --oracle-output runs/eval/_static_probe_memory_k9_oracle/report.json
```

`best_k` may choose the target-closest raw or parallax-corrected donor and is
therefore an information ceiling, not a runtime rule. The same report includes
nearest, fixed-quantile, hit-point-reprojection, and hybrid rules.

Train the learned selective-fusion arm only when its matching ceiling passes:

```bash
python -m qnn.diag.static_probe_memory \
  --probe-dir artifacts/diag/static_probe_memory/navatlas_s32z32 \
  --sidecars artifacts/diag/static_map_memory/sidecars/*.jsonl \
  --k 9 --harmonics 12 --device cuda \
  --output-dir runs/eval/_static_probe_memory_h12k9
```

Reports include nearest-probe coverage, immutable cache hashes, cached field
bytes, query latency, probe-function count, and the common spatial gate.

### `static_cell_memory` — convex cells and portals

This diagnostic retains the non-solid hull-1 convex leaves that the engine's
map-load carve already constructs. It labels each face fragment as a solid
boundary or a portal to another leaf, then tests exact analytic portal
traversal and bounded-hop approximations against the static atlas teacher.

Export the map complexes:

```bash
python -m qnn.diag.static_cell_memory dump-cells \
  --worker assets/bin/qw_demo_worker \
  --demo-dir artifacts/corpus/qwd \
  --manifest artifacts/corpus/qwd_probe_manifest.ndjson \
  --maps dm2 dm4 dm6 \
  --out-dir artifacts/diag/static_cell_memory_v1/cells
```

Run the retained information and gather-budget sweep:

```bash
python -m qnn.diag.static_cell_memory analyze \
  --cells-dir artifacts/diag/static_cell_memory_v1/cells \
  --sidecars artifacts/diag/static_map_memory_v1/sidecars/*.jsonl \
  --output runs/eval/_static_cell_memory_v1/report.json
```

The report includes point-location coverage, portal reciprocity, compact map
bytes, portal-hop percentiles, local neighborhood sizes, exact reconstruction,
and optimistic/conservative hop truncations. Origins outside every non-solid
cell and polygon-edge degeneracies use the immutable global solid-face table;
the report measures that fallback rather than silently snapping the origin.

### `static_cell_plane_cache` — first-hit plane field

A probe's cached distance is tied to the probe origin. This diagnostic instead
caches the first-hit solid-face index on a world-yaw grid inside each convex
cell, then evaluates that face's plane at the actual pose. It measures whether
plane identity removes parallax and prices the spatial spacing required when a
collision leaf is not a visibility cell.

```bash
python -m qnn.diag.static_cell_plane_cache \
  --cells-dir artifacts/diag/static_cell_memory_v1/cells \
  --sidecars artifacts/diag/static_map_memory_v1/sidecars/*.jsonl \
  --grid-spacings 24 16 \
  --output runs/eval/_static_cell_plane_cache_v1/report.json
```

The cell-center depth, cell-center plane, same-direction oracle, fixed-grid
plane, and full portal-control arms share the common spatial gate. Map cost is
reported as sample count, compact bytes, and load-time ray-query count. Python
timings are diagnostic implementation timings, not an engine runtime claim.

When two spacings are supplied from coarse to fine, the report also prices a
sparse first-hit override proxy. It compares face identities at retained poses
and estimates a per-fine-sample change mask with packed 12-bit face values.
This is a feasibility estimate until a full-map face-change census reproduces
the byte count.

### `spatial_atlas_bench` — C-side atlas query cost

Use the worker's timed `atlas_bench` query to compare true 72-, 36-, and
24-yaw emission loops without demo reset or JSON time:

```bash
python -m qnn.diag.spatial_atlas_bench \
  --worker assets/bin/qw_demo_worker \
  --demo-dir artifacts/corpus/qwd \
  --manifest artifacts/corpus/qwd_probe_manifest.ndjson \
  --sidecars artifacts/diag/static_map_memory_v1/sidecars/*.jsonl \
  --yaw-counts 72 36 24 --iterations 2000 --repeats 5 \
  --output runs/eval/_spatial_atlas_bench_v1/report.json
```

The timed region includes direction construction, static carved-face queries,
and quantization. It excludes IPC, reset, JSON, and dynamic movers. Reported
single-core fractions assume 20 Hz.

## Per-head `analyze()` dispatch

`qnn.diag.analyze.run_analyze` dispatches to each head module's `analyze()`
function. Each head module consolidates the canonical per-head analysis
scripts:

| Head module | Functions consolidated from |
|-------------|----------------------------|
| `qnn.diag.attack` | `fire_target_conditional`, `attack_offset_distribution`, `attack_empirical_range`, `attack_input_ablation` |
| `qnn.diag.look` | `look_prior_fit`, `look_prior_explore4`, `look_history_attention`, `look_horizon_ceiling`, `look_metric_references`, `look_target_intersection`, `aim_point_z_offset`, `look_ground_spin`, `look_aim_prior_decode` |
| `qnn.diag.move` | `momentum_baseline`, `stream_dynamics`, `jump_discrim`, `jump_onset_probe`, `rate_fidelity` |

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
- `qnn/diag/attack.py`, `look.py`, `move.py` — per-head analysis modules
- `qnn/diag/look_data.py`, `move_metrics.py` — shared computation kernels
- `qnn/diag/spatial_reconstruction.py` — map/token geometry reconstruction
- `qnn/diag/static_map_memory.py` — immutable hull-face memory reconstruction
- `qnn/diag/static_probe_memory.py` — cached directional-field routing
- `qnn/diag/static_cell_memory.py` — convex-cell and portal reconstruction
- `qnn/diag/static_cell_plane_cache.py` — cell-clipped first-hit plane fields
- `qnn/diag/spatial_atlas_bench.py` — in-engine atlas cost frontier
- `qnn/diag/{history,convergence,rank,ablation,gradients,participation,attention,pruning,linear_probe}.py` — suite tools
- `agents/skills/diag/SKILL.md` — when-to-use procedural guidance (outside `src/`, full path)
