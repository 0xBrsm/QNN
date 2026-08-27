# Semantics contract `semantics.2` — A27 pure combat observations

`semantics.2` inherits the scalar scales, item-bit layout, entity/action
vocabulary IDs, spatial depth-code meaning, and temporal-alignment clause from
[`semantics.1`](semantics.1.md). It changes the meaning and active vocabulary
of the combat entity stream used by [`wire.13.1` and
`wire.13.2`](../wire/wire.13.md) (the A27 pure-combat wire line; a26's
[`wire.12.1`/`wire.12.2`](../wire/wire.12.md) keep `semantics.1`).

Stamped as `semantics_contract=semantics.2`. The current
`semantics_sig` is **`d1057da0f5d4ead7`**; recompute it from
`qnn.contracts.semantics_sig()` rather than treating the value as frozen.

## Combat observation semantics

Only actors and projectiles are combat tokens. Both types use the two-entry
combat modality vocabulary:

| ID | name | POC meaning |
|---:|------|-------------|
| 0 | `SIGHT` | Current view-cone qualification plus unobstructed world trace |
| 1 | `PROXIMITY` | Current engine PVS ground truth when SIGHT did not win |

An entity that qualifies for neither channel in the current frame is absent.
Sound and memory timestamps never create combat tokens, and a previous
observation never decays into the current stream. The engine-wide modality
table retains `SOUND=2` and `MEMORY=3` for store/event compatibility, but the
A27 model has a two-row modality embedding and those IDs are invalid inputs.

The engine-PVS implementation is the proof-of-concept producer for PROXIMITY.
The intended interface permits a higher-level belief model to replace it with
predicted actor/projectile state while preserving these IDs and the A27 input
shape.

## Active entity layouts

| token | model scalar layout | width |
|-------|---------------------|------:|
| projectile | `rel[3]`, derived `dist`, `vel[3]` | 7 |
| actor | `half_extents[3]`, `rel[3]`, derived `dist`, `vel[3]`, `path[3]`, `path_dist`, `eta`, `facing`, `team`, `score` | 18 |

Item and mover rows are not part of the combat stream. Recency is absent from
both active layouts. This semantic epoch also removes the item/mover-only
`amount`, `regen`, and `state` inputs from the **model boundary** — but item
and mover tokens remain live engine entities (geometry, events, rewards, and
the wire.11/wire.12 codecs that share the bin), so their value-semantics
(`ITEM_AMOUNT_MULT/CONST`, `TOKEN_ITEM/TOKEN_MOVER`, `ITEM/MOVER_SCALAR_DIM`)
stay in the `semantics_sig` fingerprint. They are omitted from wire.13 obs, not
deleted from the contract.

## Temporal alignment

The `semantics.1` timeliness clause remains normative: `obs(t)` reflects the
agent's own commands through `t-1`, with server-time-aligned self state. The
new combat qualification adds a second current-frame requirement: every actor
or projectile token must be justified by SIGHT or PROXIMITY at that same tick.
This producer behavior is audited in collection; it is not fully encoded by
the tensor fingerprint.

## Attack semantics

The self observation has no equipped-weapon ID. Ownership/ammo readiness and
`attack_finished` remain because they describe which attacks the engine can
execute, not which weapon is currently displayed.

The action-side `attack` label is one categorical byte. It is `0` when no
effective engine attack occurs and `1..8` only when that frame attacks with the
corresponding Quake impulse. There is no parallel binary fire label and no
action-side `weapon` label.

The packed `move` byte may still retain raw usercmd button-0 evidence in bit 0.
That bit answers "did the demonstrator press attack in this collection window?"
and is not the supervised attack target. A press rejected by cooldown, death,
ownership, or ammo therefore has `move & 1 != 0` but `attack == 0`.

The A27 `attack` head learns one 9-way distribution: no effective attack plus
attack with each impulse. Runtime decode emits the same category. A nonzero
category drives both button 0 and that impulse in one tick; category 0 drives
neither. Decode does not consult equipped state, repair labels from equipped
state, apply switch hysteresis, or carry a weapon-selection action across
frames.
