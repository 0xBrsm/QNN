# Setup Guide

How to go from a fresh clone to a trained model.

## Prerequisites

- Docker with GPU passthrough (ROCm for AMD, CUDA for NVIDIA), or a native
  Linux environment with PyTorch and a C compiler
- Quake shareware or full PAK files (`PAK0.PAK`, `PAK1.PAK`)
- A demo corpus (see [Demo Corpus](#demo-corpus) below)

## Container Setup

The trainer container is defined in `docker/`:

```bash
docker compose -f docker/compose.yaml build trainer
docker compose -f docker/compose.yaml run --rm trainer bash
```

The container provides `fteqcc` (QuakeC compiler), `build-essential`, PyTorch,
and all Python dependencies. Key environment variables are set
in `compose.yaml`:

| Variable | Purpose |
|----------|---------|
| `QUAKE_BASEDIR` | Asset root — PAK files, demos, compiled binaries |
| `QNN_DEVICE` | `gpu` or `cpu` for torch device |
| `QNN_ARTIFACT_ROOT` | Asset root inside the container (`/workspace/assets`) |
| `QNN_AUTOCAST_DTYPE` | `fp32` / `bf16` / `fp16` (defaults to fp32; BC sets it from `train.json.dtype`) |
| `QNN_ROCM_INFERENCE_PAD_BATCH` | ROCm-only: pad batch size for inference (default 32) |
| `PYTHONPATH` | Must include the repo root |

For devcontainer users, `.devcontainer/devcontainer.json` provides an
equivalent setup.

## Asset Layout

Place game files and demos under the asset root (default `assets/`):

```text
assets/
  id1/
    PAK0.PAK
    PAK1.PAK
  corpus/
    dem/                     (NetQuake .dem demos)
    dem_manifest.ndjson      (NQ corpus metadata — optional, enables match trimming)
    qwd/                     (QuakeWorld .mvd/.qwd demos)
    qwd_manifest.ndjson      (QW corpus metadata)
  collect/
    prod/                    (sharded BC training data — output of qnn.bc.collect)
    tmp/                     (scratch collection output)
  bin/                       (built by build scripts)
  frikbotnex_train/          (built by build-progs.sh)
```

## Building

Five binaries and one QuakeC progs.dat:

```bash
# All at once (inside container):
scripts/build-mod.sh

# Or individually:
bash engine/build/build_ppo_worker.sh assets/bin/ppo_worker
bash engine/build/build_nq_demo_worker.sh assets/bin/nq_demo_worker
bash engine/build/build_nq_client.sh assets/bin/nq_client
bash engine/build/build_qw_demo_worker.sh assets/bin/qw_demo_worker
bash engine/build/build_qw_classifier.sh assets/bin/qw_classifier
```

| Binary | Purpose |
|--------|---------|
| `ppo_worker` | Live training worker (PPO, interactive) — runs the full engine with bots |
| `nq_demo_worker` | NetQuake demo collection — replays `.dem` files, emits obs/action pairs |
| `nq_client` | NetQuake client wired to `qnn.eval.live` — drives a trained policy against a live NQ server |
| `qw_demo_worker` | QuakeWorld demo collection — replays `.mvd`/`.qwd` files, same QOBS output format |
| `qw_classifier` | Standalone QWD parser used by `qnn.demo.classify` — replaces the old Python parser end-to-end |

The build scripts fetch upstream id Software Quake source, apply headless
patches, and compile with the QNN C modules from `engine/`.

### QuakeC and Bot Integration

The live training worker needs compiled QuakeC (`progs.dat`) with reward hooks.
`scripts/build-progs.sh` builds this from vendored sources. If you use a
different bot (not FrikBotNex), you need four QuakeC-side hooks that the
engine calls as builtins #79-#82:

```c
// Builtin #79 — called on every weapon discharge
void(entity shooter, float weapon_id) qnn_training_note_shot;

// Builtin #80 — called inside T_Damage()
void(entity attacker, entity target, float weapon_id,
     float health_before, float armor_before,
     float armor_type_before, float is_splash) qnn_training_note_damage;

// Builtin #81 — called on player death
void(entity victim, entity attacker, float weapon_id,
     float flags) qnn_training_note_death;

// Builtin #82 — called on item pickup and item respawn
void(entity actor, entity item, float event_kind,
     float category, float amount, float weapon_id) qnn_training_note_item;
```

`weapon_id` maps 1-8 (axe through thunderbolt). `event_kind` is 1 (pickup)
or 2 (respawn). `category` is 1-6 (health, armor, ammo, weapon, powerup,
backpack). `flags` bit 1 = gib. See `frikbotnex/qnn_training.qc` for the
reference implementation.

These hooks feed the QTRN reward sidecar. Without them the observation
pipeline still works, but reward computation has no data.

Demo workers do **not** need `progs.dat` — they replay the network stream
directly without executing QuakeC.

## Demo Corpus

The demo workers support two protocols:

### NetQuake `.dem` (NQ demo worker)

Standard NetQuake demo format. All demos in the current corpus are NQ
spectator recordings — the camera tracks a player, and the worker
reconstructs movement labels from observed position deltas via BSP-clipped
physics simulation.

**Characteristics:**
- Native tick rates vary widely (10-72 Hz depending on the player's client);
  the worker resamples to a fixed rate (default 20 Hz)
- Spectator demos: `cl.velocity` is always zero (spectator camera), so
  velocity is derived from position deltas
- Move labels are inferred, not ground truth — the 9-candidate physics
  simulation recovers the most likely input, but keyboard-only players and
  unusual physics interactions (movers, push triggers) add noise
- Match trimming available when `manifest.ndjson` provides match start/end
  text markers

### QuakeWorld `.mvd` / `.qwd` (QW demo worker)

QuakeWorld multi-view demo format. The QW worker produces the same QOBS
output format as the NQ worker, so `bc_collect.py` consumes both
identically.

**Characteristics:**
- MVD format carries multiple player perspectives in one file
- Server-authoritative playerstate includes velocity (no reconstruction
  needed for some fields)
- QW physics differ from NQ (bunny hopping, air control) — the QW worker
  has its own physics simulation module

### First-Person vs Spectator

First-person demos (player records their own game) provide direct button
inputs — move labels are exact. Spectator demos (external camera tracking a
player) require physics-based input reconstruction.

The current corpus is entirely spectator. First-person NQ demos would
produce cleaner move labels but are rarer for competitive play.

### Collection

Collection and training are **separate steps**. Collection extracts
obs/action pairs from demos into sharded `.npy` caches. Training reads those
caches — it never touches demos directly.

```bash
python -m qnn.bc.collect \
    --demo-dir artifacts/corpus/qwd \
    --workers 30
```

The collector starts a persistent demo worker process per worker, feeds it
demos sequentially, and writes sharded `.npy` files plus a train/val split
manifest into `artifacts/collect/<demo-type>/`. Resume is supported via an append-only
done log. God-mode frames, dead-time, and frozen-alive periods are filtered
at the C emission layer before reaching Python.

You only need to re-collect when the C worker's label or observation
code changes. Multiple BC training runs can share the same cached data.

## Training

### Create a BC Run

```bash
python -m qnn.run.init \
    --name bc_v1 \
    --mode bc \
    --resume true
```

This copies templates from `qnn/bc/templates/` into a frozen
`runs/bc/bc_v1/config/` directory. Edit the config files before training if
needed — see [run.md](run.md) for the full config reference.

Key settings to check:
- `machine.json`: `bc_data_dir` (path to precomputed caches), `device`, `batch_size`
- `train.json`: `lr`, `epochs`, `sequence_length`, `seed`
- `model.json`: architecture (defaults are production values)

### BC Training

```bash
docker compose -f src/docker/compose.yaml run -d --rm trainer \
    agents/skills/train/scripts/train.sh runs/bc/bc_v1
```

BC mode trains from the existing `precomputed_train/` and `precomputed_val/`
shard caches under `bc_data_dir`. Collect or recollect separately with
`qnn.bc.collect` when the C worker's label or observation code changes.
Checkpoints, metrics, and history are written to the run directory.

### PPO Fine-Tuning

```bash
python -m qnn.run.init \
    --name ppo_v1 \
    --mode ppo \
    --checkpoint-path runs/bc/bc_v1/checkpoints/bc_best.pth \
    --resume true

docker compose -f src/docker/compose.yaml run -d --rm trainer \
    agents/skills/train/scripts/train.sh runs/ppo/ppo_v1
```

PPO runs the live engine with bots, using the BC checkpoint as the initial
policy. This requires built `ppo_worker` and `progs.dat`.

### PPO From Random Weights

If you want to exercise the PPO pipeline without a trained BC seed —
e.g., smoke-testing the trainer — generate a random-init policy and
point `--checkpoint-path` at it:

```bash
python scripts/make_random_checkpoint.py \
    --model qnn/ppo/templates/model.json \
    --seed 29 \
    --output assets/seeds/rand_seed29/rand_seed29.pth
```

The checkpoint has the same architecture/sidecar layout as a BC best model (`best_<run_id>.pth`), so `qnn.run.init --mode ppo --checkpoint-path ...` accepts it directly.

### Live Play

Drive a trained checkpoint against a real NQ server via the `nq_client`
binary:

```bash
python -m qnn.eval.live \
    --checkpoint runs/bc/<run>/checkpoints/best_<run_id>.pth \
    --server <host>:<port>
```

`qnn.eval.live` co-locates the policy with `nq_client`; NAT breaks NetQuake's
port-switching handshake, so run the client on the same network segment as
the server.

### Microbenchmark

Per-process `ppo_worker` throughput ceiling (bare pipe IPC, no policy or
learner) — useful when isolating engine and transport overhead:

```bash
python scripts/bench_ppo_worker.py --steps 5000 --warmup 500
```

### GPU Check

```bash
python -m qnn.utils.check_accelerator --device gpu
```
