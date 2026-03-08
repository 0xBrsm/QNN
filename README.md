# Quake AI

This repo targets an engine-backed Quake policy trained on shared `MapStateV2` and `WorldTickV2` contracts. The current live direction is combat-survival training on `world_v2_competitive`, warm-started from competitive BC. The old symbolic and E1M1-only paths remain in-tree for regression and worker-health checks.

## Current Status

| Item | Status |
|------|--------|
| Active live target | `world_v2_competitive` plus `combat_survival` reward on the real Quake worker |
| Competitive BC warm start | `artifacts/runs/competitive_materialized_competitive_all51/`: `50` replayable demos, `1,300,465` ticks, `best_val_accuracy=0.8306`, `test_accuracy=0.8040` |
| Retained worker regression baseline | `artifacts/runs/e1m1_corpus_world/`: BC sampled `stuck_rate=0.9099`; PPO sampled `stuck_rate=0.0906`; completion still `0.0` |
| Live combat profiles | `combat-verify` and `combat` are wired through `quake_ai.live_training` |
| Bot-backed live profiles | Named `combat-bot-<scenario>[-verify]` ladder profiles use `FrikBotNex` through `-game frikbotnex` |
| Retained ladder evidence | `open-dm4` verify/live, `pressure-dm6` verify, and an `open-dm4` feed-forward verification comparison are now retained in `artifacts/runs/competitive_bot_*` |
| Worker control boundary | `native_args` and reset options support `-game` mods, match settings, and pre/post-map commands |
| Current blockers | No second retained live ladder scenario yet, and the campaign comparison lane still relies on the older retained artifact surface |

If you are doing training work, start with [Training Quality Plan](../agents/training-quality-plan.md).

## What Matters Now

- `world_v2_competitive` is the active live training path.
- Compare worker stability and contract health against `artifacts/runs/e1m1_corpus_world/`, but do not treat E1M1 completion as the main optimization target.
- Use competitive BC as the practical warm start, not as the final product.
- `quake_ai.live_training` now falls back to the profile-local BC checkpoint when the configured PPO warm start metadata does not match the current `world_v2_competitive` observation shape.
- Keep opponent and mod setup at the worker boundary rather than baking Quake-specific startup rules into Python.
- Use the named bot ladder profiles for verify/live promotion work instead of the legacy single-profile aliases.

## Action Contract

| Head | Live meaning |
|------|--------------|
| `move` | `0` idle, `1` forward, `2` back |
| `strafe` | `0` idle, `1` left, `2` right |
| `look_yaw` | 25 discrete sensitivity-3 mouse-count bins for left and right look |
| `look_pitch` | 25 discrete sensitivity-3 mouse-count bins for up and down look |
| `fire` | `0` off, `1` on |
| `jump` | `0` off, `1` on |
| `weapon` | `0` no switch, `1..8` direct weapon-slot switch |

`move` and `strafe` stay as separate 3-way heads, so diagonal movement comes from combining them rather than from a single analog movement head.

## Install

From `src/`:

```bash
python -m pip install -e .
python -m pip install -e .[dev]
```

For AMD GPUs, use the isolated ROCm trainer with `src/scripts/train-container.sh` rather than trying to layer ROCm into the editor environment.

## Current Commands

From the repo root:

```bash
src/scripts/train-container.sh check
src/scripts/train-container.sh install-frikbotnex
src/scripts/train-container.sh live-check
src/scripts/train-container.sh e1m1-world report
src/scripts/train-container.sh run -- python -m quake_ai.live_training --profile combat-verify --action check --device gpu
src/scripts/train-container.sh run -- python -m quake_ai.live_training --profile combat-verify --action ppo --device gpu
src/scripts/train-container.sh run -- python -m quake_ai.live_training --profile combat-verify --action eval --device gpu
src/scripts/train-container.sh combat-bot-open-dm4-verify check
src/scripts/train-container.sh combat-bot-open-dm4-verify all
src/scripts/train-container.sh combat-bot-pressure-dm6-verify all
src/scripts/train-container.sh combat-bot-open-dm4 all
```

The shell wrapper exposes `combat-verify`, `combat`, the named `combat-bot-<scenario>[-verify]` ladder shortcuts, the legacy `combat-bot-verify` and `combat-bot` aliases, and `install-frikbotnex`. `run -- python -m quake_ai.live_training --profile ...` remains available for ad hoc profile work.

## Bot Ladder

The current FrikBotNex ladder lives in `configs/combat_bot_scenarios.json`.

| Scenario | Map | Purpose | Verify profile | Live profile |
|----------|-----|---------|----------------|--------------|
| `duel-dm2` | `dm2` | Short-horizon duel aim and fast re-engagements | `combat-bot-duel-dm2-verify` | `combat-bot-duel-dm2` |
| `open-dm4` | `dm4` | Open pursuit, long sightlines, and visibility changes | `combat-bot-open-dm4-verify` | `combat-bot-open-dm4` |
| `vertical-dm3` | `dm3` | Vertical pickup pressure and route choice | `combat-bot-vertical-dm3-verify` | `combat-bot-vertical-dm3` |
| `pressure-dm6` | `dm6` | Opening pressure, spawn variance, and repeated weapon trades | `combat-bot-pressure-dm6-verify` | `combat-bot-pressure-dm6` |

Recent retained evidence:

- `artifacts/runs/competitive_bot_open_dm4_verify/`: recurrent retained verify run with non-zero outgoing combat metrics.
- `artifacts/runs/competitive_bot_pressure_dm6_verify/`: retained verify run that trades combat output for higher survival/return than `open-dm4`.
- `artifacts/runs/competitive_bot_open_dm4_verify_ff/`: feed-forward retained verify comparison against the recurrent `open-dm4` run.
- `artifacts/runs/competitive_bot_open_dm4_live/`: first retained live bot-backed promotion, with sampled eval return improving from `-5.7744` in `eval_bc` to `13.9230`.

## Manual Combat Loop

From `src/`:

```bash
export QUAKE_BASEDIR=/assets
engine/build/build_quake_worker.sh ../artifacts/bin/quake_worker
python -m quake_ai.install_frikbotnex --asset-root "$QUAKE_BASEDIR"
python -m quake_ai.train_bc --config configs/bc_combat_bootstrap_verify.yaml
python -m quake_ai.live_training --profile combat-verify --action check --device cpu
python -m quake_ai.live_training --profile combat-verify --action ppo --device cpu
python -m quake_ai.live_training --profile combat-verify --action eval --device cpu
```

If a bot-capable mod is available, pass it through the config-driven worker boundary with `native_args` and `native_options` instead of custom Python glue.
To use the bundled FrikBotNex path, install it once under the asset root and then run one of the named ladder profiles such as `combat-bot-open-dm4-verify` or `combat-bot-open-dm4`.

## Outputs That Matter

| Path | Purpose |
|------|---------|
| `artifacts/runs/competitive_materialized_competitive_all51/` | Current mixed-mode competitive BC warm start |
| `artifacts/runs/campaign_combat_verify/` | Bounded combat live profile outputs |
| `artifacts/runs/campaign_combat_live/` | Retained combat live profile outputs |
| `artifacts/runs/competitive_bot_open_dm4_verify/` | Example bounded bot-backed ladder verify output root |
| `artifacts/runs/competitive_bot_pressure_dm6_verify/` | Retained bot-backed ladder verify output root for the secondary pressure scenario |
| `artifacts/runs/competitive_bot_open_dm4_verify_ff/` | Retained feed-forward verify comparison output root for `open-dm4` |
| `artifacts/runs/competitive_bot_open_dm4_live/` | Example retained bot-backed ladder live output root |
| `artifacts/runs/e1m1_corpus_world/` | Retained worker-regression baseline |
| `artifacts/runs/e1m1_corpus_world_baseline_20260307_fail/` | Archived collapse kept for historical comparison |

## Regression-Only Paths

- The symbolic configs under `configs/bc_e1m1.yaml`, `configs/ppo_e1m1.yaml`, and `configs/eval_e1m1.yaml` remain for fast regression checks only.
- The old `e1m1-world` profiles remain useful for worker-health and deterministic-behavior checks, not as the main training target.

## Related Docs

- [Training Quality Plan](../agents/training-quality-plan.md)
- [Status](../agents/status.md)
- [Competitive Path Plan](../COMPETITIVE_BC_REENGINEERING_PLAN.md)
- [Architecture](docs/architecture.md)
