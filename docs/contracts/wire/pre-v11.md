# Wire contracts `wire.1`–`wire.6` — pre-v11 (below the faithful-load floor)

Everything below the [`wire.7`](wire.7.md) floor. These are **reserved,
numbered slots, not built** — each requires fields the current engine no longer
emits (see the [registry](../README.md) dead-field list), so it is at best a
zero-fill approximation (**Band B**) or impossible (**Band C**), and **no model
artifact survives** for any of them. Full historical detail lives in the
archived bundled docs linked below; this page exists so a found checkpoint is
immediately diagnosable and a future codec has a named slot.

| Wire | Release | Bundled doc | Defining wire delta | Dead fields | Band |
|------|---------|-------------|---------------------|-------------|------|
| `wire.1` | 0.1–0.2 | [obs-spec-v1](../../../../docs/archive/obs-spec-v1.md) / [v2](../../../../docs/archive/obs-spec-v2.md) | Flat `obs_dim` tensor (~290–347), world-frame absolute origin, CLS+player+entity+event+sound+item tokens, action_history concatenated post-transformer | world-origin frame, action_history (+ wholly different tensor architecture) | **C** |
| `wire.2` | ~0.3 | [obs-spec-v3](../../../../docs/archive/obs-spec-v3.md) | Dict token obs replaces flat; identity vocab; CLS pooling | action_history; different arch lineage | **C** |
| `wire.3` | 0.4–0.9 | [token-spec-v5](../../../../docs/archive/token-spec-v5.md)/[v6](../../../../docs/archive/token-spec-v6.md)/[v7](../../../../docs/archive/token-spec-v7.md) | QTOK packed: 23 self scalars, `cluster_id`+`route_embed`, MENTAL recall heads, 5/6-bin weapon, object scalars 8→13 | cluster_id, route_embed, recall/MENTAL, action_history | **C** |
| `wire.4` | 0.10 | [token-spec-v8](../../../../docs/archive/token-spec-v8.md) | Three-store oracle; self **14 scalars** (paired 0/0.5/1.0 weapon floats); object 17 scalars; action_history 2×7 | cluster_id, route_embed, recall/MENTAL, action_history | **B** |
| `wire.5` | 0.11 | [token-spec-v9](../../../../docs/archive/token-spec-v9.md) | Per-type variable-length entity stream; `entity_embed=42`; cluster/qualifier removed; Cartesian look[3]; action_history 8×8; recall heads still present | recall/MENTAL, action_history | **B** |
| `wire.6` | 0.14–0.15 | [token-spec-v10](../../../../docs/archive/token-spec-v10.md) | move[2]+jump **collapsed → move[3]**, jump head removed; modality vocab 5→4 | recall/MENTAL, action_history | **B** |

Semantics for this range: `wire.1–2` → `semantics.0a`; `wire.3` → `semantics.0b`;
`wire.4` → `semantics.0c`; `wire.5–6` → `semantics.0d` (see
[`semantics/pre-v11.md`](../semantics/pre-v11.md)).

**Why not built:** Bands B/C carry a dead field *and* have no surviving model —
building these codecs is engineering against zero artifacts. If a checkpoint
ever resurfaces, promote its row to a full `wire.N.md`, decide whether zero-fill
is acceptable, and add the codec. The `wire.1`/`.2` flat era additionally has no
migration path (different tensor architecture).
