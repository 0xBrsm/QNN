# Quake AI

This repo is centered on one training path: competitive BC warm start plus Sample Factory APPO fine-tuning against FrikBotNex on Quake deathmatch maps.

## Current Path

| Item | Current state |
|------|---------------|
| Observation contract | token dict with `self`, `object`, `event`, and `spatial` tensors |
| Policy | transformer trunk + GRU actor-critic |
| PPO backend | Sample Factory 2.1.1 APPO |
| Run entry point | `runs/<name>/run.json` |
| Wire format | v6 (see `token-spec.md`) |
| Resume control | explicit `run.json.resume` |
| Warm start | explicit `run.json.checkpoint_path` when PPO has no retained checkpoint to resume |
| Scenario source | `runs/<name>/config/scenario.json` |
| PPO artifact root | `run.json.output.checkpoints` directly, with only SF `checkpoint_p*` subdirs underneath |

The old campaign and E1M1 lanes are gone from the supported surface.

## What Matters

- `overview.md` is the source of truth for architecture and observation layout.
- `quake_ai.rl.training` is the promoted orchestration entry point.
- `Training containers use `docker compose -f src/docker/compose.yaml`.
- Every training and eval action is driven by a frozen run directory under `runs/`.

## Install

From `src/`:

```bash
python -m pip install -e .
python -m pip install -e .[dev]
```

For Sample Factory:

```bash
pip install numpy==1.26.4
pip install sample-factory==2.1.1 --no-deps
pip install gymnasium==0.29.1
pip install psutil tensorboard tensorboardX signal-slot-mp wandb colorlog opencv-python-headless pyglet threadpoolctl huggingface-hub
```

For AMD GPUs, use the training container instead of modifying the editor environment.

## Promoted Commands

From the repo root:

```bash
docker compose -f src/docker/compose.yaml run --build trainer python -m quake_ai.utils.check_accelerator --device gpu
docker compose -f src/docker/compose.yaml run --build trainer bash
python scripts/init_run.py --name <run_name> --mode ppo --runtime-scale live --resume false --checkpoint-path <ckpt> --trainer src/quake_ai/rl/configs/trainer.json --scenario src/quake_ai/rl/configs/scenario.json --reward src/quake_ai/rl/configs/reward.json --machine src/quake_ai/rl/configs/machine.json --model src/quake_ai/rl/configs/model.json
scripts/container.sh run scripts/train.sh runs/<run_name>
```

## Scenario Ladder

The ladder is defined in `quake_ai/rl/configs/scenario.json`.

| Scenario | Map | Purpose |
|----------|-----|---------|
| `duel-dm2` | `dm2` | close-range duel timing and quick re-engagements |
| `open-dm4` | `dm4` | long sightlines and open pursuit |
| `vertical-dm3` | `dm3` | vertical pickup timing and route choice |
| `pressure-dm6` | `dm6` | spawn pressure and repeated explosive trades |

The frozen `scenario.json` in each run defines the full multi-scenario ladder and emits per-scenario evaluation metrics.

## Config Surface

Training config is split by concern under `quake_ai/rl/configs/`:

- `machine.json`: flat run-mode machine config
- `scenario.json`: flat run-mode scenario surface
- `model.json`: flat architecture config
- `trainer.json`: flat run-mode trainer config
- `reward.json`: reward weights
- `run.json`: run manifest template copied into each run directory, including explicit `resume`

See [`docs/training-config-matrix.md`](docs/training-config-matrix.md) for the required key matrix and the list of removed hidden defaults.

## Outputs That Matter

| Path | Purpose |
|------|---------|
| `runs/<name>/checkpoints/` | BC outputs, PPO checkpoints, retained best checkpoint |
| `runs/<name>/metrics/` | eval summaries and metrics |
| `runs/<name>/logs/` | watcher and runtime logs |

## Related Docs

- [`overview.md`](../overview.md)
- [`token-spec.md`](../token-spec.md)
- [`agents/training-quality-plan.md`](../agents/training-quality-plan.md)
- [`agents/environment.md`](../agents/environment.md)
