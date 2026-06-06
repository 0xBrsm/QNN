# QNN Overview

Competitive Quake PvP agent trained end-to-end: native C worker emits semantic
tokens, a transformer encoder attends over them, a GRU maintains temporal
state, supervised pointer attention picks an engagement target, and factored
action heads (move, look, fire, weapon) drive the engine. The pipeline is
seeded by behavioral cloning on Quake demos and fine-tuned via Sample Factory
APPO against FrikBotNex opponents.

Related docs: [token-spec.md](token-spec.md) (wire format, head shapes),
[vocab.md](vocab.md) (IDs and event mapping), [run.md](run.md) (config schema),
[vendor.md](vendor.md) (dependencies),
[target_labeler_engine_alignment.md](target_labeler_engine_alignment.md)
(target labeler + engine sticky design),
[diag.md](diag.md) (capacity diagnostics on trained policies —
`python -m qnn.diag`).

## Policy Architecture

```text
TransformerEncoder
  self_readout x_t           (slot 0 = CLS; pools self subtokens / spatial / entity)
  actor tokens e_i           (entity slots where type == ACTOR)
        │
        ├──► TargetPointer:  query = linear(x_t)        (or GRU output if gru_target_query)
        │      scores  = (e_i * query).sum(-1), non-actor slots masked
        │      target_logits   (B, 16) — CE label vs labeler
        │      target_feat = softmax(scores) @ e_i  (soft pool)
        │
        └──► GRU input:      x_t -> h_t                  (sequence of cls_readouts)

Action features (per tick):
  motor heads (move, look, fire):
    features = cat(gru_out, target_feat, weapon_context)
  weapon head:
    features = cat(gru_out, cls_readout, target_feat)   (gru_out optional)

  weapon_context comes from the held weapon in obs (default) or from a
  softmax over the weapon head's own logits.

  ├──► move    (3 categorical axes × 3 classes {neg, none, pos})
  ├──► look    target-anchored base direction + learned residual
  ├──► fire    binary BCE
  └──► weapon  8-way categorical → Quake impulse byte 1..8

  Target is supervised internally; it conditions action heads but is
  never sampled. The 8-class weapon head emits a direct Quake impulse;
  the engine-facing "switch" slot is gone end-to-end.
```

Primary supervised heads: `target` (pointer over actor slots), `move`,
`look`, `fire`, `weapon`. The weapon class index is converted to a Quake
impulse byte (1..8 = axe..lightning) by the engine bridge; no separate
switch controller.

| Encoder parameter | Value |
|-----------------|-------|
| d_model | 64 |
| n_heads | 2 |
| n_layers | 2 (pre-norm) |
| ffn_dim | 256 |
| gru_hidden | 64 |

Token sequence: `self(1) + spatial(9) + entities(up to 16) = up to 26 tokens`.
Invalid entity rows masked via `key_padding_mask`. Action-history tokens are
parked (templates set `action_history_tokens: 0`; no wire region in v11).

### BC Loss Notes

Move is trained as three independent categorical axes; the up/down axis
carries jump and can be reweighted via `jump_pos_weight` with linear decay.

Fire is trained as binary BCE with corpus-derived positive weighting.

Weapon is trained as 8-class CE on the demonstrator's held-weapon impulse.
No-weapon frames (pre-spawn, dead, transitional) carry 0 on disk and are
masked from the CE loss via `ignore_index=-100`.

Target is trained as 16-way CE on the labeler output described in
[target_labeler_engine_alignment.md](target_labeler_engine_alignment.md);
unlabeled frames carry `-100` and are skipped.

Per-head loss weighting is configured via `head_loss_weights` in
`train.json`; a head with weight 0 still emits logits but contributes no
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
| `qnn/actions.py` | canonical mixed action schema, move pack/unpack, look decoding |
| `qnn/wire.py` | binary wire format parser, action struct layout, LOBS labeler frame |
| `qnn/schema.py` | OBS_SCHEMA, observation-embedding input shapes |
| `qnn/filter_dsl.py` | shared filter mini-language (collect + train) |
| `qnn/collection_fingerprint.py` | collect-identity hash recorded at train time |
| `qnn/model/transformer.py` | observation embedding + transformer encoder |
| `qnn/model/target.py` | TargetPointer attention module |
| `qnn/model/policy.py` | actor-critic with GRU + play heads |
| `qnn/run/router.py` | run-dir router |
| `qnn/run/config.py` | strict run-dir config loader and builders |
| `qnn/{bc,ppo,eval}/` | mode runners (`bc`, `ppo`, `pbt`, `eval`, `optuna`) |
| `qnn/bc/supervised_loop.py` | chunked supervised loop (extracted from `loop.py`) |
| `qnn/bc/class_weights.py` | per-head class-weight derivation |
| `qnn/bc/token_filter.py` | train-time token mask compiler |
| `qnn/bc/target_labeler.py` | offline target label generator |
| `qnn/bc/collect.py` | demo collection pipeline (dispatches to engine workers) |
| `qnn/ppo/env.py` | Sample Factory gymnasium wrapper |
| `qnn/ppo/encoder.py` | PPO encoder wrapper |
| `qnn/ppo/train.py` | PPO registration + config builder |
| `qnn/env/reward.py` | PvP reward shaping |
| `qnn/diag/` | trained-policy diagnostics package |
| `qnn/labeler/probes/` | standalone target-head probes (causal TCN, GBT) |
| `qnn/eval/live.py` | live-play entry point (NQ servers) |
| `engine/common/` | shared C worker: store, oracle, entity, event, io, sound, spatial, fault, watchdog, tick |
| `engine/common/qnn_mvd_collect.{c,h}` | shared MVD/QWD demo collect runtime |
| `engine/common/qnn_qwd_collect.{c,h}` | QWD usercmd-path collect helpers |
| `engine/common/qnn_labeler_collect.{c,h}` | native-rate labeler LOBS emit |
| `engine/nq/` | NetQuake collect/trainer/client main loops, physics, input |
| `engine/qw/` | QuakeWorld collect main loop, physics, input |
| `engine/build/` | build scripts for `ppo_worker`, `nq_demo_worker`, `nq_client`, `qw_demo_worker`, `qw_classifier` |
| `mapgen/` | procedural Quake .map generator for training variety |
