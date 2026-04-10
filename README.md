# Quake AI

Transformer policy trained via behavioral cloning and PPO for PvP Quake.
Packaged as `quake-ai-v0`.

## Packages

| Package | Purpose |
|---------|---------|
| `quake_ai` | Python model, training pipeline, and observation contract |
| `quake_ai.model` | Transformer tokenizer, trunk, and GRU actor-critic policy |
| `quake_ai.ppo` | Sample Factory APPO integration (env, encoder, worker) |
| `quake_ai.rl` | Run management, training router, runners, reward, evaluation |
| `quake_ai.utils` | Device detection, reproducibility, IO helpers |
| `demo` | Quake `.dem` parser and action-label extraction |
| `engine` | C worker source, build scripts, and engine patches |
| `mapgen` | Procedural `.map` generation and compilation |

## Observation Contract

v9 token-dict observation. Wire format defined in `engine/worker/qnn_io.h`,
Python adapter in `quake_ai/obs_format.py`, model shapes in
`quake_ai/model/observation.py`.

| Token type | Count | Scalars | Notes |
|------------|-------|---------|-------|
| Self | 1 | 14 + 4 embed IDs | health, armor, weapons, ammo, velocity |
| Entity | up to 16 | 8/19/15/14 by type | projectile/actor/item/mover, type-tagged |
| Spatial | 9 | 13 | view-relative directional sectors |
| Action history | up to 8 | 8 | recent action frames |

Action space: 5 play heads (move, look, fire, jump, switch) + 4 recall heads.

## Run Modes

All training runs are driven by a frozen run directory. The training router
dispatches to the correct runner based on `run.json.mode`:

| Mode | Runner | Entry point |
|------|--------|-------------|
| `bc` | `quake_ai.rl.runners.bc` | Behavioral cloning from demo data |
| `ppo` | `quake_ai.rl.runners.ppo` | PPO via Sample Factory APPO |
| `pbt` | `quake_ai.rl.runners.pbt` | Population-based training |
| `optuna` | `quake_ai.rl.runners.optuna` | Hyperparameter search |
| `eval` | `quake_ai.rl.runners.eval` | Evaluation only |

## Commands

```bash
# Install (editable)
python -m pip install -e .
python -m pip install -e .[dev]

# Create a run
python -m quake_ai.rl.init_run \
    --name <run_name> \
    --checkpoint-path <ckpt> \
    --resume true

# Launch training
python -m quake_ai.rl.training --run-dir runs/<run_name>

# GPU check
python -m quake_ai.utils.check_accelerator --device gpu
```

## Run Directory Layout

```text
runs/<name>/
  run.json          # manifest (mode, resume, checkpoint_path, output paths)
  run.md            # notes
  config/
    trainer.json
    scenario.json
    reward.json
    machine.json
    model.json
  checkpoints/
  metrics/
  logs/
```

Run templates live in `quake_ai/rl/run_templates/`. `init_run` freezes
copies into the run directory at creation time.

## Config Templates

Training config is split by concern under `quake_ai/rl/run_templates/`:

| Template | Scope |
|----------|-------|
| `trainer.json` | Learning rate, batch size, PPO hyperparameters |
| `scenario.json` | Bot ladder, map pool, episode settings |
| `model.json` | Architecture (d_model, heads, layers, GRU) |
| `reward.json` | Reward shaping weights |
| `machine.json` | Workers, devices, memory limits |
| `eval.json` | Evaluation-specific overrides |
| `run.json` | Mode, checkpoint, resume, output paths |

## Docker

The trainer container is defined in `docker/`:

```bash
docker compose -f docker/compose.yaml build trainer
docker compose -f docker/compose.yaml run --rm trainer bash
```

## Engine Worker

C worker source lives in `engine/worker/`. Build scripts and engine patches
are in `engine/build/` and `engine/patches/`. The worker implements the
observation packing (`qnn_io.h`), reward computation (`qnn_reward.c`),
and game-state oracle (`qnn_oracle.c`, `qnn_store.c`).
