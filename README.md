# Quake AI

This repo is centered on one training path: competitive BC warm start plus Sample Factory APPO fine-tuning against FrikBotNex on Quake deathmatch maps.

## Current Path

| Item | Current state |
|------|---------------|
| Observation contract | token dict with `self`, `object`, `event`, and `spatial` tensors |
| Policy | transformer trunk + GRU actor-critic |
| PPO backend | Sample Factory 2.1.1 APPO |
| Default profile | `combat-bot-multi` |
| Warm start | `artifacts/runs/competitive_materialized_competitive_all51/` |
| Scenario source | `quake_ai/rl/configs/combat_bot_scenarios.json` |

The old campaign and E1M1 lanes are gone from the supported surface.

## What Matters

- `overview.md` is the source of truth for architecture and observation layout.
- `quake_ai.rl.training` is the promoted orchestration entry point.
- `docker/train-container.sh` is the promoted runtime wrapper for engine-backed work.
- The public profile surface is only `combat-bot-multi[-verify]` plus `combat-bot-<scenario>[-verify]`.

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
src/docker/train-container.sh check
src/docker/train-container.sh install-frikbotnex
src/docker/train-container.sh live-check
src/docker/train-container.sh combat-bot-multi-verify plan
src/docker/train-container.sh combat-bot-multi-verify check
src/docker/train-container.sh combat-bot-multi-verify all
src/docker/train-container.sh combat-bot-multi all
src/docker/train-container.sh combat-bot-open-dm4-verify all
```

Direct invocation remains available:

```bash
python -m quake_ai.rl.training --profile combat-bot-multi --action plan --device gpu
python -m quake_ai.rl.training --profile combat-bot-multi --action ppo --device gpu
python -m quake_ai.rl.training --profile combat-bot-multi --action eval --device gpu
```

## Scenario Ladder

The ladder is defined in `quake_ai/rl/configs/combat_bot_scenarios.json`.

| Scenario | Map | Purpose |
|----------|-----|---------|
| `duel-dm2` | `dm2` | close-range duel timing and quick re-engagements |
| `open-dm4` | `dm4` | long sightlines and open pursuit |
| `vertical-dm3` | `dm3` | vertical pickup timing and route choice |
| `pressure-dm6` | `dm6` | spawn pressure and repeated explosive trades |

`combat-bot-multi[-verify]` trains across all four scenarios in one run and emits per-scenario evaluation metrics.

## Config Surface

BC, PPO, and eval hyperparameters are inlined as module-level dicts in `quake_ai.rl.training` and overridable via `LiveProfile` override dicts. The only external config file is the scenario ladder:

- `quake_ai/rl/configs/combat_bot_scenarios.json`

Sample Factory settings are built from the PPO dicts inside `quake_ai.rl.training`; there are no separate YAML entry points.

## Outputs That Matter

| Path | Purpose |
|------|---------|
| `artifacts/runs/competitive_materialized_competitive_all51/` | current BC warm start |
| `artifacts/runs/competitive_bot_multi_verify/` | retained multi-scenario verify root |
| `artifacts/runs/competitive_bot_multi_live/` | retained multi-scenario live root |
| `artifacts/runs/competitive_bot_open_dm4_verify/` | retained single-scenario verify comparison |
| `artifacts/runs/competitive_bot_open_dm4_live/` | retained single-scenario live comparison |
| `artifacts/runs/competitive_bot_pressure_dm6_verify/` | retained secondary verify comparison |

## Related Docs

- [`overview.md`](/workspaces/dev-qnn/overview.md)
- [`agents/training-quality-plan.md`](/workspaces/dev-qnn/agents/training-quality-plan.md)
- [`agents/environment.md`](/workspaces/dev-qnn/agents/environment.md)
