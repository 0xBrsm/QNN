# QNN Overview

Competitive Quake PvP agent trained end-to-end: native C worker emits semantic
tokens, a transformer encoder attends over them, a GRU maintains temporal
state, supervised pointer attention picks an engagement target, and factored
action heads (move, look, attack) drive the engine. The pipeline is
seeded by behavioral cloning on Quake demos and fine-tuned by native bounded
PPO against FrikBotNex opponents. PPO overlaps one immutable recurrent rollout
with the preceding learner update and caps policy lag at exactly one update.

Related docs: [RESEARCH.md](../../docs/RESEARCH.md) (the research spine — hypothesis,
endpoints, claims register, roadmap),
[head-metrics.md](../../docs/head-metrics.md) (the canonical metric contract —
how every action head is judged), [contracts/](contracts/README.md) (wire / semantics /
arch contract versions + I/O signatures), [vocab.md](vocab.md) (IDs and event mapping), [run.md](run.md) (config schema),
[release-flow.md](release-flow.md) (per-head ablation → assembled RC line → PPO rc1),
[vendor.md](vendor.md) (dependencies),
[diag.md](diag.md) (capacity diagnostics on trained policies —
`python -m qnn.diag analyze` for unified per-head analysis,
`python -m qnn.diag diagnostic` for the checkpoint-based suite).

## Policy Architecture

```text
TransformerEncoder
  self_readout x_t           (slot 0 = CLS; pools self subtokens / spatial / entity)
  actor tokens e_i           (entity slots where type == ACTOR)
        │
        ├──► TargetPointer:  scores_i = w2 · gelu(W1 · e_i + b1) + b2  (per-entity MLP, hidden = d_target)
        │      non-enemy / padding slots masked to -1e9
        │      target_logits   (B, 16) — soft-CE label vs labeler
        │      target_feat = softmax(scores) @ e_i  (soft pool)
        │
        └──► GRU input:      x_t -> h_t                  (sequence of cls_readouts)

Action heads (per tick):
  ├──► move_seg  semi-Markov movement commitments
  ├──► jump      vertical action classifier
  ├──► look      target-anchored polar direction + learned residual
  └──► attack    9-way attack choice
                  class 0 = no attack
                  classes 1..8 = attack with Quake impulse 1..8

  The current attack-with choice also selects per-weapon aim physics and
  decode gains. Target is supervised internally; it conditions action heads
  but is never sampled.
```

Primary A27 supervised heads are `target` (pointer over actor slots),
`move_seg`, `jump`, `look`, and `attack` (implemented by the `attack_with`
head type). There is no
separate binary attack head, equipped-weapon input, intent-history buffer, or
switch controller.

| Encoder parameter | Value |
|-----------------|-------|
| d_model | 64 |
| n_heads | 2 |
| n_layers | 2 (pre-norm) |
| ffn_dim | 256 |
| gru_hidden | 64 |

### Model Assembly

The model is assembled from a declarative `GraphSpec` (see `src/docs/model-graph.md`). Every pipeline — BC, bench probes, eval, PPO, and ONNX export — calls `qnn.model.graph.build_network(obs_dim, spec)` as the single factory. Node builders self-register into `qnn.model.node_registry` via `@register_head` / `@register_encoder` / `@register_pointer` / `@register_temporal` decorators declared beside each node class; `build_network` dispatches by the spec's type discriminators. Named base-graph compositions are registered by each model generation's `graphs` module via `register_base_graph`; probes are expressed as override dicts merged onto a base via `qnn.model.graph.merge_overrides`.

The A27 base graph uses four self rows (`cls`, `state`, `arsenal`, `motion`),
eleven spatial rows, and up to sixteen combat-entity rows: at most 31 tokens.
Invalid entity rows masked via `key_padding_mask`. Action-history tokens are
parked (templates set `action_history_tokens: 0`; no wire region in v11).

On A27, entity attention is a pure combat stream: actors and projectiles only.
Both may arrive as current-frame `SIGHT` or `PROXIMITY` observations;
projectiles participate in encoder attention, while the target pointer remains
actor-only. Item/mover tokens, recency, SOUND, and MEMORY are outside the fast
action substrate. The POC PROXIMITY producer is engine PVS ground truth and is
an explicit substitution point for a future higher-layer belief model.

On spatial-v2 (`wire.12`), the eleven spatial attention rows are a
center-ray depth atlas: elevation bands from −75° to +75° in 15° steps,
each carrying 24 fifteen-degree yaw cells of 4-bit log-quantized depth
against the carved hull-1 face set (world plus live-translated movers).
A row is nibble-packed into 12 bytes on the wire and expanded to 24 depth
plus 24 hit scalars model-side.
A learned band-ID embedding identifies each fixed row; row order alone
is not visible because this transformer has no positional encoding. The
self token includes view pitch so the model can relate yaw-frame
geometry to camera aim.

### BC Loss Notes

Move is trained as three independent categorical axes; the up/down axis
carries jump and can be reweighted via `jump_pos_weight` with linear decay.

Attack is trained as 9-class CE. Class 0 means no effective attack;
classes 1 through 8 mean attack with that Quake impulse. The QWD/MVD collector
stamps a nonzero `attack` label only on an effective attack frame, so the
label is action truth rather than equipped state. Per-class feasibility uses
ownership/ammo readiness and cooldown state without exposing the equipped
weapon ID.

Target is trained as 16-way CE on the adaptive-cone target-labeler output;
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
| `qnn/wire.py` | binary wire format parser, action struct layout (QOBS) |
| `qnn/schema.py` | OBS_SCHEMA, observation-embedding input shapes |
| `qnn/filter_dsl.py` | shared filter mini-language (collect + train) |
| `qnn/collection_fingerprint.py` | collect-identity hash recorded at train time |
| `qnn/model/graph/` | declarative model assembly: `GraphSpec`, `build_network`, `merge_overrides`, `GraphObsEmbedding` |
| `qnn/model/node_registry.py` | self-registering node-builder registry (`@register_head`, `@register_encoder`, etc.) |
| `qnn/model/transformer.py` | transformer encoder block (registers via `@register_encoder`) |
| `qnn/model/target.py` | TargetPointer attention module (registers via `@register_pointer`) |
| `qnn/model/policy.py` | actor-critic with GRU + play heads |
| `qnn/model/decode.py` | generation-stable decode primitives (attack sigmoid+thresh, move axis decode, sampling prims) |
| `qnn/model/decode_config.py` | run-pinned JSON decode configuration schema |
| `qnn/run/router.py` | run-dir router |
| `qnn/run/config.py` | strict run-dir config loader and builders |
| `qnn/{bc,ppo,eval}/` | mode runners (`bc`, `ppo`, `pbt`, `eval`, `optuna`) |
| `qnn/bc/supervised_loop.py` | chunked supervised loop (extracted from `loop.py`) |
| `qnn/bc/class_weights.py` | per-head class-weight derivation |
| `qnn/bc/token_filter.py` | train-time token mask compiler |
| `qnn/bc/target_labeler.py` | offline target label generator |
| `qnn/bc/collect.py` | demo collection pipeline (dispatches to engine workers) |
| `qnn/ppo/vec_env.py` | vectorized native-worker lanes, batched drain, and async resets |
| `qnn/ppo/collector.py` | fixed-window recurrent rollout collection |
| `qnn/ppo/learner.py` | native recurrent PPO update and trainer-owned value head |
| `qnn/ppo/train.py` | native PPO orchestration, checkpointing, and telemetry |
| `qnn/env/reward.py` | PvP reward shaping |
| `qnn/diag/` | trained-policy diagnostics plus pre-training spatial-token reconstruction against real-map hull traces |
| `qnn/labeler/probes/` | standalone target-head probes (causal TCN, GBT) |
| `qnn/eval/live.py` | live-play entry point (NQ servers) |
| `engine/common/` | shared C worker: store, oracle, entity, event, io, sound, spatial, fault, watchdog, tick |
| `engine/common/qnn_mvd_collect.{c,h}` | shared MVD/QWD demo collect runtime |
| `engine/common/qnn_qwd_collect.{c,h}` | QWD usercmd-path collect helpers |
| `engine/nq/` | NetQuake collect/trainer/client main loops, physics, input |
| `engine/qw/` | QuakeWorld collect main loop, physics, input |
| `engine/build/` | build scripts for `ppo_worker`, `nq_demo_worker`, `nq_client`, `qw_demo_worker`, `qw_classifier` |
| `mapgen/` | procedural Quake .map generator for training variety |
