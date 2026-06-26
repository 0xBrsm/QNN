# Model Graph — Declarative Model Assembly

Design and reference for the declarative model-assembly config ("model graph"). One JSON document describes the whole model — tokens as dicts of scalar and vocab fields, encoder/temporal/pointer nodes, and heads with their parameters and named input edges. Every pipeline (BC, bench head probes, eval, PPO, ONNX export) builds the model from this one spec via one builder.

Status: implemented on `refactor/model-graph` — spec/builder in `qnn/model/graph/`, self-describing checkpoints, bench deltas, PPO encoder alignment, legacy prune.

## Why

The pre-graph assembly had three problems:

| Problem | Where it lived | Consequence |
|---------|----------------|-------------|
| Checkpoints not self-describing | `utils/checkpoint_converter.py` `_head_probe_model_factory` | Loading a bench checkpoint re-runs `HEADS[head].build(probe)` from a `config/probe.json` found by relative path; breaks if the run dir moves or the registry changes. |
| Per-probe imperative factories | `qnn/model/bench/*.py` (26 registered heads) | Every ablation hand-wires `Network` slots and re-derives `slot_dims`; most differ from canonical by one or two nodes. |
| Architecture knobs spread across flat flags | `ModelConfig` (`use_gru`, `use_weapon_head`, `weapon_sources`, `look_bypass_gru`, ...) | Slot presence, input edges, and node widths are entangled in booleans plus special cases; adding an edge means a new flag plus `slot_dims` surgery. |

The graph spec fixes all three: the checkpoint stores the graph; probes are deltas on a base graph; edges are data.

## The Spec

A model graph is a JSON object with five sections: `tokens`, `encoder`, `temporal`, `pointers`, `heads`. Node parameters are scalars; node wiring is named edges. The canonical full_5head model:

```json
{
  "graph_version": 1,
  "tokens": {
    "cls":     {"kind": "cls"},
    "state":   {"scalars": ["health_armor"], "vocab": ["armor_type"], "vocab_sum": ["powerup_state"]},
    "arsenal": {"readiness": true, "vocab_sum": ["powerup_arsenal"]},
    "motion":  {"scalars": ["velocity", "view_pitch", "look_delta"], "vocab": ["movement_id"], "vocab_sum": ["powerup_motion"]},
    "held_weapon": {"scalars": ["weapon_static", "attack_finished"], "vocab": ["weapon_id"]},
    "spatial":  {"kind": "spatial"},
    "entities": {"kind": "entities"}
  },
  "encoder": {"type": "transformer", "d_model": 64, "n_heads": 2, "n_layers": 2, "d_ffn": 256, "attn_dropout": 0.0},
  "temporal": {"type": "gru", "d_gru": 64},
  "pointers": {
    "target": {"type": "mlp", "d_target": 16}
  },
  "heads": {
    "move":   {"type": "cls",   "inputs": ["readout", "target.feat"], "d_hidden": 64, "activation": "gelu"},
    "look":   {"type": "polar", "inputs": ["readout", "target.feat"], "d_hidden": 64, "activation": "gelu"},
    "attack": {"type": "cls",   "inputs": ["readout", "target.feat"], "d_hidden": 64, "activation": "gelu"},
    "weapon": {"type": "cls",   "inputs": ["gru"], "d_hidden": 64, "activation": "gelu",
               "context_from_obs": false}
  }
}
```

### Tokens

Each entry under `tokens` declares one self-block token built by `TokenBuilder` from the field catalog in `qnn/model/tokens/obs_fields.py`:

- `scalars` — names from `SCALAR_FIELDS`, concatenated into one `ScalarGroup` (one `Linear`). Order in the list fixes the Linear layout, so token definitions live in committed configs, never in ad-hoc dicts.
- `vocab` — names from `VOCAB_FIELDS` with `reduce="none"`, one masked `VocabEmbed` each.
- `vocab_sum` — names with `reduce="sum"` (`VocabSum`).
- `readiness: true` — appends the `WeaponReadiness` einsum source.
- `kind_tag: true` — adds the encoder's self `KindTag` row (the monolithic-self layout uses it; the split-self winners do not).
- `kind` — reserved token kinds the builder constructs outside `TokenBuilder`: `cls`, `spatial`, `entities`. Entity tokens keep the per-type projections (`actor`, `projectile`, `item`, `mover`) inside `ObsEmbedding`; they are not field-declarative.

Field order inside a token is fixed (scalars → kind_tag → vocab → readiness → vocab_sum) so the parameter layout is deterministic. `GraphObsEmbedding` replaces the `ObsEmbedding` subclass zoo (`SplitSelfObsEmbedding`, `HeldWeaponSplitObsEmbedding`, monolithic-self): layouts are token dicts, not classes.

### Encoder, Temporal, Pointers

- `encoder.type` ∈ {`transformer`, `passthrough`}. `passthrough` replaces the bench `PreAttnEncoder`.
- `temporal` is `null` or `{"type": "gru", "d_gru": N}`. Absent temporal means heads see `self_readout` directly (the old `temporal=Off`).
- `pointers` is a dict so future pointer nodes (or none) are first-class. `target.type` ∈ {`mlp`, `gt`} — `gt` is the oracle pointer, kept for ceiling probes.

### Heads and Edges

Each head names its `type` (resolved in a head registry) and its `inputs` — an ordered list of edge names whose widths are summed to size the head's input Linear:

| Edge | Width | Meaning |
|------|-------|---------|
| `self_readout` | `d_model` | Encoder CLS readout. |
| `gru` | `d_gru` | Temporal hidden. Error if no temporal node. |
| `readout` | `d_gru` if temporal else `d_model` | The canonical motor readout alias. |
| `target.feat` | `d_model` | Pointer-blended target feature. Error if no `target` pointer. |
| `weapon.context` | `d_model` | Weapon context embed (implied automatically for motor heads when a weapon head is present, to match current `motor_in`). |

`inputs` replaces both `weapon_sources` and the implicit `motor_in` composition. Dangling edges are a build-time error — there is no "silently dropped" source and no zero-padding for absent nodes (the `slot_dims` "if target is off, it's off" rule becomes a property of edge resolution).

`decode` blocks carry the head's decode contract (the weapon sticky-gate `sticky_confidence` / `sticky_margin`, mapped onto `ModelConfig.weapon_switch_*` and baked into ONNX export) — the decode regime is part of the model, never decided by the engine.

The PPO critic is not a graph node: Sample Factory owns the value head inside its actor-critic. PPO consumes the graph's token/encoder/pointer nodes through its encoder wrapper (`qnn/ppo/encoder.py`, built from `--quake_model_graph`); `Network`'s `values` output remains a vestigial zeros tensor kept only for forward-contract stability.

## The Builder

`qnn.model.graph` owns the spec and the builder:

- `GraphSpec.from_dict / to_dict` — strict JSON round-trip, unknown keys raise.
- `build_network(obs_dim, spec) -> Network` — the single model factory. Resolves nodes from small registries (`HEAD_TYPES`, encoder/pointer types), sizes heads via `slot_dims`, then wires `Network` slots.
- `model_config_from_graph(spec)` — the flat `ModelConfig` bridge for the policy-layer flags `QNNPolicy`/`Network` still read.
- `graph_from_model_config(cfg)` — translation of a flat v20+ canonical `ModelConfig` into the equivalent graph (the v17 `look_bypass_gru` layout is not expressible and stays on the legacy path).

Checkpoints save `meta["model_graph"] = spec.to_dict()`. The loader branches: `model_graph` present → `build_network` with a strict load; legacy checkpoints (flat `meta["model"]`, or pre-graph head probes rehydrated from `config/probe.json` via the shrunken `HEADS` registry) keep loading exactly as before — no state-dict key migration.

## Bench Probes as Graph Deltas

A probe.json names a base graph and overrides:

```json
{
  "base": "full_5head",
  "overrides": {
    "heads": {"weapon": {"inputs": ["gru", "target.feat"]}}
  }
}
```

Base graphs are committed JSON files under `qnn/model/graph/bases/` (`full_4head.json`, `full_5head.json`). Overrides deep-merge by section/key; an explicit `null` deletes a node. Loss weights stay in `train.json` (`head_loss_weights`) like every other training knob. A probe that needs a genuinely new module registers one head/token/encoder type in the `qnn.model.graph.build` registries — not a whole network factory. The bench `HEADS` registry survives only to reload legacy (pre-graph) head-probe checkpoints.

## Legacy Ablation Paths

Concluded experiments are removed from the registry and preserved here for reference; findings live in their result docs:

| Removed path | Verdict | Findings |
|--------------|---------|----------|
| `weapon_aim/` joint look+attack | Resolved; do not re-propose | `weapon-head.md` |
| `attack_prior/` prior-mode sweeps (`geomfix`, `geomperw`, `geomscalar`, `hittest`) | Geometry cannot cross f1@0.5 | `fire-discrimination.md` |
| Attack probes (`attack_flat`, `attack` preattn, `attack_preattn_oracle`, `attack_bundle`, `attack_geom_bundle`, `attack_look_style`) | Cooldown dominates; geometry capped | `fire-discrimination.md` |
| Target query pointer variants (`target`, `target_constant_query`, `target_mlp_query`, `target_mlp_query_full_stack`, `target_self_query`, `target_weapon_query`) | MLP scorer won on `val_target_kl` | `target-head.md` |
| Weapon probes (`weapon` preattn, `weapon_arsenal`, `weapon_switch` WHEN/WHAT) | Dense CLS+GRU classifier won; do not re-propose switch head | `weapon-head.md` |
| Look probes (`look_cls`, `look_head_move_token`) | POLAR head won by held-out Δloglik | `look-head.md` |
| Move probes (`move_motion_token`), `temporal` probe | Momentum-capped; superseded by full_Nhead deltas | `move-head.md` |
| Single-head `*_cls_transformer` specs (`move_cls_transformer`, `attack_cls_transformer`, `weapon_cls_transformer`) | Winners; classes promoted to `qnn.model.cls_heads` (specs removed) | per-head docs |
| Flat-feature probe machinery (`features.py`, `flat.py`) | Superseded by full_Nhead deltas | per-head docs |
| `look_bypass_gru` ModelConfig flag | v17-only fidelity shim | — |

The winning modules were promoted out of `bench`: `CLSMoveHead` / `CLSAttackHead` / `CLSWeaponHead` to `qnn.model.cls_heads`, `PurePolarLookHead` to `qnn.model.look_head_polar`.

Old run dirs under `runs/head_probe/` are immutable history; their `probe.json` files are not loadable by the new code and are not migrated.

## Pipeline Alignment

- **BC** — `run_behavior_cloning(graph=...)` builds via `build_network` and stamps every checkpoint with the graph; bench probes are graph deltas (above).
- **Eval / export** — both load through `load_checkpoint`, which prefers the embedded graph over legacy probe.json rehydration; nothing else changed (eval was already meta-driven).
- **PPO** — the pipeline reads the warm-start seed's `meta.model_graph` (sidecar JSON), fail-louds if the run's `model.json` disagrees, and threads it to the SF encoder as `--quake_model_graph`; `QuakeTransformerEncoder` then builds the seed's exact token/encoder/pointer layout so BC weights map 1:1 (test-pinned). `sf_to_qnn` carries the graph back out, so converted PPO checkpoints stay self-describing. The SF-owned critic and action parameterization are unchanged.

## Training Speedup — Audit Verdicts (June 2026)

Cache format is frozen (disk budget). A code audit proposed in-process levers; each was checked against the live full_5head run's measured epoch profile (`bc_history.json`: train 291.6 s @ 30.3k rows/s, val 21.6 s @ 95.6k rows/s, train_eval 10.8 s). The forward-only val pass uses the same data path as training and runs 3.1× faster, so data loading / host-to-device transfer is **not** the training bottleneck — backward+optimizer is, at the standard ~2× forward cost. Verdicts:

| Proposed lever | Verdict | Why |
|----------------|---------|-----|
| Cache `segment_mask` filtering "per epoch" | Refuted | Filtering runs once at source construction (`streaming_source.from_cache_dir` → `_shard_segments`), not per epoch. |
| Wire up `pin_memory` (dead `BCConfig` field) | Refuted | H2D is not the bottleneck (val sustains 95k rows/s on the same path). Field kept — removing it breaks the strict machine.json schema of existing run-dirs. |
| Resident-source prefetch | Refuted | Resident tensors live on-device; there is no transfer to overlap. |
| Cache training-view fingerprint | Negligible | Manifest-level hashing, startup milliseconds; the resident daemon amortizes startup anyway. |
| Per-step grad-norm `.item()` sync | Already done | Grad norms accumulate as tensors, flushed at epoch end. |
| bf16 autocast | Already done | `dtype: "bf16"` is the template default (`QNN_AUTOCAST_DTYPE`). |
| `torch.compile` | Ruled out (measured) | Net-negative at this model size (~189K params); documented in `bc/train.py`. |
| PPO `worker_inference` | Already done | Template default `true`; recent runs all use it. |

The real, already-built levers remain the resident ablation daemon (~2.5× vs streaming) and parallel ablation containers (4× sweet spot, 5× practical limit). Further single-job BC throughput means changing the training itself (lane count / model size), not the harness.
