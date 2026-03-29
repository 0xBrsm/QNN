# Quake AI

This repo has one promoted training workflow: create a frozen run directory,
then execute it through the training router.

## Current Path

| Item | Current state |
|------|---------------|
| Observation contract | token dict with `self`, `object`, `event`, and `spatial` tensors |
| Policy | transformer trunk + GRU actor-critic |
| PPO backend | Sample Factory APPO |
| Public run modes | `bc`, `ppo`, `pbt`, `eval`, `optuna` |
| Run entry point | `runs/<name>/run.json` |
| Templates | `src/quake_ai/rl/run_templates/` |
| Resume control | explicit `run.json.resume` |
| Warm start | explicit `run.json.checkpoint_path` |
| PPO artifact root | `run.json.output.checkpoints` |

Campaign-era and profile-era training surfaces are not the promoted path
anymore. Use frozen run directories for everything.

## What Matters

- [`docs/overview.md`](../docs/overview.md) is the source of truth for architecture and observation layout.
- `python -m quake_ai.rl.training --run-dir runs/<name>` is the runtime router.
- `scripts/init_run.py` is the only promoted way to create a run.
- `scripts/train.sh runs/<name>` is the promoted launch wrapper inside the trainer container.
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

For AMD GPUs, use the training container instead of modifying the editor
environment.

## Promoted Commands

From the repo root:

```bash
docker compose -f src/docker/compose.yaml build trainer
docker compose -f src/docker/compose.yaml run --rm trainer python -m quake_ai.utils.check_accelerator --device gpu
docker compose -f src/docker/compose.yaml run --rm trainer bash
python scripts/init_run.py --name <run_name> --checkpoint-path <ckpt> --trainer src/quake_ai/rl/run_templates/trainer.json --scenario src/quake_ai/rl/run_templates/scenario.json --reward src/quake_ai/rl/run_templates/reward.json --machine src/quake_ai/rl/run_templates/machine.json --model src/quake_ai/rl/run_templates/model.json
docker compose -f src/docker/compose.yaml run --rm trainer scripts/train.sh runs/<run_name>
```

Use the same path for every mode:

- `ppo` is the default comparison lane
- `pbt` is still a run dir launched through the same wrapper
- `optuna` is still a run dir launched through the same wrapper
- `eval` is a run dir with `mode: "eval"`

## Run Layout

```text
runs/<name>/
  run.json
  run.md
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

Generated Optuna trial wrappers live under `runs/<name>/trials/`. Each wrapper
stores trial metadata plus a child PPO run under `ppo/`. Treat them as runtime
output, not curated run manifests.

## Scenario Ladder

The default ladder template is
`src/quake_ai/rl/run_templates/scenario.json`. Each run freezes its own copy in
`runs/<name>/config/scenario.json`.

## Config Surface

Training config is split by concern under `src/quake_ai/rl/run_templates/`:

- `machine.json`
- `scenario.json`
- `model.json`
- `trainer.json`
- `reward.json`
- `run.json`

See [`docs/training-config-matrix.md`](../docs/training-config-matrix.md) for
the required key matrix.

## Related Docs

- [`docs/overview.md`](../docs/overview.md)
- [`docs/token-spec.md`](../docs/token-spec.md)
- [`agents/training-quality-plan.md`](../agents/training-quality-plan.md)
- [`agents/training-status.md`](../agents/training-status.md)
- [`agents/environment.md`](../agents/environment.md)
