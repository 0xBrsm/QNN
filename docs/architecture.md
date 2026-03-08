# Architecture

This document describes the current engine-backed training system. Historical symbolic pieces remain in-tree, but only as regression scaffolding.

## Current System

| Layer | Current role |
|------|--------------|
| Static world model | `MapStateV2` keeps persistent BSP-derived map context, stable region ids, and static object indices |
| Dynamic tick stream | `WorldTickV2` carries player state, sparse entity updates, semantic events, short action history, and lightweight debug counters |
| Navigation encoder path | `WorldObservationEncoder` supports the retained `world_v2` regression lane |
| Competitive encoder path | `CompetitiveObservationEncoder` adds compact combat-oriented entity and event semantics for `world_v2_competitive` live combat work |
| Offline training | BC learns from replayed `world_v2` artifacts, including competitive subsets |
| Live training | PPO fine-tunes on the real Quake worker using either navigation or combat-survival reward shaping with feed-forward or recurrent actor-critic policies |
| Runtime boundary | Native worker emits JSON-over-stdio observations; Python owns orchestration, reward shaping, evaluation, and reporting |

## Active Training Lanes

| Lane | Purpose | Current status |
|------|---------|----------------|
| `world_v2` | Retained worker-health and regression baseline on E1M1 | `artifacts/runs/e1m1_corpus_world/` remains the retained comparison point |
| `world_v2_competitive` | Competitive BC warm start plus combat-survival live PPO | Competitive BC is retained; campaign comparison artifacts remain in-tree; retained ladder evidence now exists for `open-dm4` verify/live, `pressure-dm6` verify, and the `open-dm4` feed-forward-versus-recurrent verify comparison |

## Active Training Loop

1. `collect` replays demos and writes `world_map.json`, `world_ticks.ndjson`, telemetry, packets, metadata, and summaries.
2. `train_bc` trains either the navigation or competitive observation path from replay artifacts.
3. `eval_bc` measures the held-out live BC baseline on the native worker.
4. `train_rl` fine-tunes that checkpoint with PPO on the native worker.
5. `eval` measures the PPO checkpoint, and reporting compares `eval` against `eval_bc` plus any retained comparison profiles.

The retained wrapper around this loop lives in `quake_ai.live_training` and is exposed through `src/scripts/train-container.sh` plus direct `python -m quake_ai.live_training --profile ...` invocations. The shell wrapper now supports named `combat-bot-<scenario>[-verify]` shortcuts in addition to the legacy campaign and compatibility aliases. When the configured PPO warm start metadata does not match the current live observation shape, `live_training` falls back to the profile-local BC checkpoint so retained combat runs can continue on the updated competitive encoder.

## Current Artifacts

| Run | What it means |
|-----|----------------|
| `artifacts/runs/competitive_materialized_competitive_all51/` | Current competitive BC warm start. `50` replayable demos, `1,300,465` ticks, `best_val_accuracy=0.8306`, `test_accuracy=0.8040`. |
| `artifacts/runs/campaign_combat_verify/` | Bounded campaign-combat comparison lane for `world_v2_competitive`. |
| `artifacts/runs/campaign_combat_live/` | Retained campaign-combat comparison lane. |
| `artifacts/runs/competitive_bot_open_dm4_verify/` | Example bounded FrikBotNex ladder output root for the promoted `open-dm4` verify scenario. Other ladder scenarios follow the same naming pattern. |
| `artifacts/runs/competitive_bot_pressure_dm6_verify/` | Retained verify output root for the promoted `pressure-dm6` secondary scenario. |
| `artifacts/runs/competitive_bot_open_dm4_verify_ff/` | Retained feed-forward verification comparison against the recurrent `open-dm4` verify run. |
| `artifacts/runs/competitive_bot_open_dm4_live/` | Example retained FrikBotNex ladder output root for the promoted `open-dm4` live scenario. |
| `artifacts/runs/e1m1_corpus_world/` | Current retained worker-regression baseline for the navigation path. |
| `artifacts/runs/e1m1_corpus_world_baseline_20260307_fail/` | Archived retained collapse. Keep it only as a regression reference. |

## Action Contract

The live worker exposes a player-like discrete control surface instead of the earlier `turn` and `use` abstraction.

| Head | Meaning |
|------|---------|
| `move` | `0` idle, `1` forward, `2` back |
| `strafe` | `0` idle, `1` left, `2` right |
| `look_yaw` | 25 discrete mouse-count bins mapped through Quake sensitivity `3` for left and right look |
| `look_pitch` | 25 discrete mouse-count bins mapped through Quake sensitivity `3` for up and down look |
| `fire` | Binary attack input |
| `jump` | Binary jump input |
| `weapon` | `0` no switch, `1..8` direct weapon-slot switch |

`move` and `strafe` remain separate heads, so diagonal movement is represented by combining them. `look_yaw` and `look_pitch` are constrained to sensitivity-3 mouse-count bins, which keeps turning human-like and prevents instant turns.

## Contracts

### `MapStateV2`

- Persistent per-map world model loaded once
- Stable ids for regions or another map partition
- Full-map planning structure rather than a local-only crop
- Static object index for items, triggers, movers, doors, and spawn points

### `WorldTickV2`

- Per-tick player state
- Sparse entity-state updates for client-relevant dynamic objects
- Sparse event list for meaningful semantic transitions
- Region ids keyed into `MapStateV2`
- Short action history in the canonical player-like action space
- Lightweight debug counters such as `frags`, `monster_kills`, `monster_total`, outgoing combat totals, and per-weapon efficiency totals

### Worker Reset Options

- `maxplayers`
- `skill`
- `deathmatch`
- `coop`
- `teamplay`
- `fraglimit`
- `timelimit`
- `samelevel`
- `pre_map_commands`
- `post_map_commands`

These options let live runs swap between campaign-style and mod-backed combat setups without new Python-side special cases.

## Model Direction

- Keep the player-like action contract stable while retained evidence, opponent setup, and live metric interpretation are the main blockers.
- Use `CompetitiveObservationEncoder`, combat-survival reward shaping, and the recurrent-capable actor-critic path for the current live lane.
- Treat the retained E1M1 navigation path as a worker-regression baseline, not as the primary optimization target.
- The next meaningful questions are retained scenario selection and signal quality, not whether recurrence should exist in-tree.

## Historical Notes

- `TelemetryTickV1`, `MapFeaturesV1`, and the symbolic environment remain for regression and fast debugging only.
- `engine/build/build_stub_worker.sh` remains the fast protocol-regression worker for tests that do not need Quake assets.
- The main open question is no longer whether the repo can run engine-backed PPO, expose outgoing combat signals, or carry recurrent state. It can. The remaining question is which retained ladder scenarios and comparisons best justify promotion.
