# QNN Overview

Competitive Quake PvP agent trained end-to-end: native C worker emits semantic
tokens, a transformer encoder attends over them, GRU maintains temporal state,
and Sample Factory APPO optimizes a factored policy against FrikBotNex opponents.

Related docs: [token-spec.md](token-spec.md) (wire format),
[vocab.md](vocab.md) (IDs and event mapping), [run.md](run.md) (config schema),
[vendor.md](vendor.md) (dependencies).

## Policy Architecture

```text
TransformerTrunk (x_t: 64, self-token readout at position 0)
        │
        ├──► scout BC baseline: heads read x_t directly
        │        ├──► move:       Linear(64→3)    [forward, strafe, jump-Z]
        │        ├──► look:       Linear(64→3)    [yaw_delta, pitch_delta, roll]
        │        ├──► fire:       Linear(64→2)    [off, on]
        │        ├──► switch:     Linear(64→6)    [no-switch, 5 weapon classes]
        │        ├──► recall_0…3: Linear(64→65)   [no-op, 64 object handles] ×4
        │        └──► value:      Linear(64→1)
        │
        └──► recurrent confirmation path
                 ▼
          GRU (64 → 64, 1 layer) → h_t
                 │
                 └──► fuse z_t = concat(x_t, h_t) = 128
                          │
                          └──► same head families on fused 128-dim features
```

8 factored heads total: 2 continuous play heads (`move[3]`, `look[3]`), 2
discrete play heads (`fire`, `switch`), and 4 discrete recall heads.

| Trunk parameter | Value |
|-----------------|-------|
| d_model | 64 |
| n_heads | 1 |
| n_layers | 2 (pre-norm) |
| ffn_dim | 256 |

Token sequence: `self + 9 spatial + [action history] + 16 entities = 26+ tokens`.
Invalid entity rows masked via `key_padding_mask`.

### BC Loss: Sparse Binary Masking

Fire and switch are sparse actions (~10-15% positive rate in demos). Standard
cross-entropy rewards "always predict 0" on the ~85% of idle ticks, collapsing
the head before PPO can use it.

`_SPARSE_BINARY_HEADS` in `policy.py` masks out true-negative ticks (target=0
AND pred=0) from the loss. Only mismatches and correct positives contribute
gradient.

## Reward System

PvP-only. 4 signals:

| Signal | Formula | Weight | Type |
|--------|---------|--------|------|
| frag_bonus | `+3.0 × frag_gain` | sparse | offensive capstone |
| death_penalty | `-2.0 × player_died` | sparse | defensive capstone |
| ehp_delta | `0.5 × log(ehp_now / ehp_prev)` | dense | self-preservation (skipped on death tick) |
| edp_delta | `0.6 × edp_raw` | dense | outgoing damage (from QC T_Damage hook) |

The 0.6 > 0.5 asymmetry makes equal trades net-positive, biasing toward
aggression.

`effective_hp(health, armor, armor_type) = min(health + armor, health / (1 - armor_type))`, floored at 1.0.

## Training Surface

The promoted training surface is the FrikBotNex ladder frozen into each run's
`config/scenario.json`.

| Scenario | Map | Focus |
|----------|-----|-------|
| duel-dm2 | dm2 | close-range re-engagement |
| open-dm4 | dm4 | long sightlines, pursuit |
| vertical-dm3 | dm3 | vertical pickup timing |
| pressure-dm6 | dm6 | spawn pressure, explosive trades |

## Source Layout

| Path | Purpose |
|------|---------|
| `qnn/vocab.py` | shared semantic IDs (source of truth) |
| `qnn/actions.py` | canonical mixed action schema and look decoding |
| `qnn/wire.py` | binary wire format parser, action struct layout |
| `qnn/schema.py` | OBS_SCHEMA, tokenizer input shapes |
| `qnn/model/transformer.py` | tokenizer + transformer trunk |
| `qnn/model/policy.py` | actor-critic with GRU + mixed play/recall heads |
| `qnn/run/router.py` | run-dir router |
| `qnn/run/config.py` | strict run-dir config loader and builders |
| `qnn/{bc,ppo,eval}/` | mode runners (`bc`, `ppo`, `pbt`, `eval`, `optuna`) |
| `qnn/bc/train.py` | BC training loop |
| `qnn/bc/collect.py` | demo collection pipeline |
| `qnn/ppo/env.py` | Sample Factory gymnasium wrapper |
| `qnn/ppo/encoder.py` | PPO encoder wrapper |
| `qnn/ppo/train.py` | PPO registration + config builder |
| `qnn/env/reward.py` | PvP reward shaping |
| `engine/common/` | shared C worker: store, oracle, entity, event, io, sound, spatial |
| `engine/nq/` | NetQuake collect/trainer main loops, physics, input |
| `engine/qw/` | QuakeWorld collect main loop, physics, input |
| `engine/bridge.py` | worker subprocess management |
| `engine/training_protocol.py` | training extras parser |
| `mapgen/` | procedural Quake .map generator for training variety |
