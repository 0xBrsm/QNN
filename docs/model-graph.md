# Model Graph — Declarative Model Assembly

Design and reference for the declarative model-assembly config ("model graph"). One JSON document describes the whole model — tokens as dicts of scalar and vocab fields, encoder/temporal/pointer nodes, and heads with their parameters and named input edges. Every pipeline (BC, bench head probes, eval, PPO, ONNX export) builds the model from this one spec via one builder.

## Why

The pre-graph assembly had three problems:

| Problem | Where it lived | Consequence |
|---------|----------------|-------------|
| Checkpoints not self-describing | `utils/checkpoint_converter.py` `_head_probe_model_factory` | Loading a bench checkpoint re-ran a reload table looked up from a relative `config/probe.json`; breaks if the run dir moves or the registry changes. |
| Per-probe imperative factories | One bespoke `Network`-wiring function per ablation | Every ablation hand-wired `Network` slots and re-derived `slot_dims`; most differed from canonical by one or two nodes. |
| Architecture knobs spread across flat flags | `ModelConfig` (`use_gru`, `use_weapon_head`, `weapon_sources`, `look_bypass_gru`, ...) | Slot presence, input edges, and node widths were entangled in booleans plus special cases; adding an edge meant a new flag plus `slot_dims` surgery. |

The graph spec fixes all three: the checkpoint stores the graph; probes are deltas on a base graph; edges are data.

## The Spec

A model graph is a JSON object with five sections: `tokens`, `encoder`, `temporal`, `pointers`, `heads`. Node parameters are scalars; node wiring is named edges. Example structure:

```json
{
  "graph_version": 1,
  "tokens": {
    "cls":     {"kind": "cls"},
    "state":   {"scalars": ["health_armor"], "vocab": ["armor_type"], "vocab_sum": ["powerup_state"]},
    "arsenal": {"scalars": ["attack_finished"], "readiness": true, "vocab_sum": ["powerup_arsenal"]},
    "motion":  {"scalars": ["velocity", "view_pitch", "look_delta"], "vocab": ["movement_id"], "vocab_sum": ["powerup_motion"]},
    "spatial":  {"kind": "spatial"},
    "entities": {"kind": "entities"}
  },
  "encoder": {"type": "transformer", "d_model": 64, "n_heads": 2, "n_layers": 2, "d_ffn": 256, "attn_dropout": 0.0},
  "temporal": {"type": "gru", "d_gru": 64},
  "pointers": {
    "target": {"type": "mlp", "d_target": 16}
  },
  "heads": {
    "move_seg": {"type": "canonical", "inputs": ["readout", "target.feat"], "d_hidden": 64, "activation": "gelu"},
    "jump":   {"type": "canonical", "inputs": ["readout", "target.feat"], "d_hidden": 64, "activation": "gelu"},
    "look":   {"type": "polar", "inputs": ["readout", "target.feat"], "d_hidden": 64, "activation": "gelu"},
    "attack": {"type": "attack_with", "inputs": ["gru", "target.feat"], "d_hidden": 64, "activation": "gelu", "feasibility_mask": true}
  }
}
```

### Tokens

Each entry under `tokens` declares one token built by `TokenBuilder` from the field catalog in `qnn/model/tokens/obs_fields.py`:

- `scalars` — names from `SCALAR_FIELDS`, concatenated into one `ScalarGroup` (one `Linear`). Order in the list fixes the Linear layout, so token definitions live in committed configs, never in ad-hoc dicts.
- `vocab` — names from `VOCAB_FIELDS` with `reduce="none"`, one masked `VocabEmbed` each.
- `vocab_sum` — names with `reduce="sum"` (`VocabSum`).
- `readiness: true` — appends the `WeaponReadiness` einsum source.
- `kind_tag: true` — adds the encoder's self `KindTag` row (the monolithic-self layout uses it; split-self tokens do not).
- `kind` — reserved token kinds the builder constructs outside `TokenBuilder`: `cls`, `spatial`, `entities`. A27 entity tokens keep the actor/projectile projections inside `GraphObsEmbedding`; they are not field-declarative.

Field order inside a token is fixed (scalars → kind_tag → vocab → readiness → vocab_sum) so the parameter layout is deterministic. `GraphObsEmbedding` replaces the `ObsEmbedding` subclass zoo: layouts are token dicts, not classes.

Token ordering constraint: `[cls?, fields.., spatial?, entities]` — the reserved tail is `spatial` then `entities`; `cls` must be first when present. `GraphSpec.validate()` enforces this at load time.

### Encoder, Temporal, Pointers

- `encoder.type` ∈ {`transformer`, `passthrough`}. `passthrough` replaces the old `PreAttnEncoder`.
- `temporal` is `null` or `{"type": "gru", "d_gru": N}`. Absent temporal means heads see `self_readout` directly (the old `temporal=Off`).
- `pointers` is a dict so future pointer nodes (or none) are first-class. `target.type` ∈ {`mlp`, `gt`} — `gt` is the oracle pointer, kept for ceiling probes.

### Heads and Edges

Each head names its `type` (resolved via the self-registered node builders in `node_registry`) and its `inputs` — an ordered list of edge names whose widths are summed to size the head's input Linear.

Current public head names are `move`, `look`, and `attack`, plus the
generation-specific `move_seg`, `jump`, and retired compatibility slots.

Motor heads (`move`, `look`, `attack`, `move_hazard`) must share identical `inputs` — `Network` builds one shared motor feature vector and feeds it to all of them. When a target pointer is declared, motor heads must include `target.feat` (there is no per-head opt-out).

| Edge | Width | Meaning |
|------|-------|---------|
| `readout` | `d_gru` if temporal else `d_model` | The canonical motor readout alias. Required for all motor heads. |
| `self_readout` | `d_model` | Encoder CLS readout (weapon head only). |
| `gru` | `d_gru` | Temporal hidden. Error if no temporal node. |
| `target.feat` | `d_model` | Pointer-blended target feature. Error if no `target` pointer. |
| `token.<name>` | `d_model` | Per-self-token encoder output at the named token index (weapon head only). |

`inputs` replaces both `weapon_sources` and the implicit `motor_in` composition. Dangling edges are a build-time error — there is no "silently dropped" source and no zero-padding for absent nodes.

The A27 `attack` slot contains the 9-class attack-with head. Class 0 is no
attack and classes 1 through 8 are attack with that Quake impulse. Its optional
`feasibility_mask` gates impossible attack classes from ownership/ammo readiness
and cooldown state before softmax, loss, and decode.

The PPO critic is not a graph node. The native trainer owns a small MLP value
head over the graph's motor features; it is saved in RL resume checkpoints but
is never part of deploy checkpoints or ONNX exports. `Network`'s `values`
output remains a vestigial zeros tensor kept only for forward-contract
stability.

## Self-Registering Node Builders

There is no central type table in `build.py`. Instead, each node module owns its builder beside the class it constructs and registers it into `qnn.model.node_registry` via one of the decorator functions:

```python
# In a node module (e.g. qnn.model.move_head):
@register_head("move", "cls")
def _build(head_spec, dims, d_model) -> nn.Module:
    ...

# In an encoder module:
@register_encoder("transformer")
def _build(encoder_spec) -> nn.Module:
    ...
```

The registry functions are `register_head(head_name, type_name)`, `register_encoder(type_name)`, `register_pointer(type_name)`, `register_temporal(type_name)`. Lookup misses return `None`; `build.py` raises `GraphSpecError` with the list of registered types so the message is informative.

`build.py` imports the canonical node modules (`qnn.model.move_head`, `look_head`, `attack_head`, `weapon_head`, `temporal`, `target`, `transformer`) solely for their registration side effects. A generation-specific module can self-register additional head/encoder/pointer/temporal types from its own file — no edit to `build.py` required.

### Base Graphs

Named base-graph compositions are registered the same way, via `register_base_graph(name, graph_dict)` in `node_registry`. The arch composition **lives with its generation module**: each generation's `graphs` module calls `register_base_graph` for the base graphs it provides, and `build.py` imports those generation modules for their registration side effects. The generation-agnostic graph package contains no base graphs of its own.

`base_graph_dict(name)` and `load_base_graph(name)` in `qnn.model.graph` resolve names from the registry and raise `GraphSpecError` (listing the registered names) on a miss — so until a generation module is imported, no base graph resolves.

## The Builder

`qnn.model.graph` owns the spec and the builder:

- `GraphSpec.from_dict / to_dict` — strict JSON round-trip; unknown keys raise, missing required keys raise, mismatched `graph_version` raises.
- `build_network(obs_dim, spec) -> Network` — the single model factory. Dispatches each node kind through the self-registered builders, sizes heads via `slot_dims`, then wires `Network` slots.
- `model_config_from_graph(spec)` — the flat `ModelConfig` bridge for policy-layer flags `QNNPolicy`/`Network` still read.
- `graph_from_model_config(cfg)` — translation of a flat v20+ canonical `ModelConfig` into the equivalent graph (the v17 `look_bypass_gru` layout is not expressible and stays on the legacy path).

`HEAD_TYPES` is a materialized `{head_name: {type_name: builder}}` view re-exported by `qnn.model.graph` for introspection and back-compat; the registry is the source of truth.

Checkpoints save `meta["model_graph"] = spec.to_dict()`. The loader branches: `model_graph` present → `build_network` with a strict load; flat-canonical checkpoints (`meta["model"]`, no graph) load via `ModelConfig`. The pre-graph head-probe reload path (the reload table + relative `config/probe.json` rehydration) was retired — the rc lineage was re-stamped self-describing and concluded ablations archived.

## Graph Overrides (Probe Deltas)

`merge_overrides` lets a spec be expressed as a base graph plus a delta — a `probe.json` names a base graph and overrides:

```json
{
  "base": "full_5head",
  "overrides": {
    "heads": {"attack": {"inputs": ["gru", "target.feat"]}}
  }
}
```

Base graphs are registered into `qnn.model.node_registry` by each generation's `graphs.py`; `base_graph_dict(name)` resolves them. Overrides deep-merge by section/key via `merge_overrides`:

- Dicts merge recursively.
- An explicit `null` deletes the key — drop a token, head, pointer, or the temporal node.
- A `null` for a key the base does not have raises (`GraphSpecError`) — a typo'd delete would otherwise no-op silently, training the un-ablated model and recording a false null result.
- Scalars and lists replace wholesale.

Loss weights stay in `train.json` (`head_loss_weights`) like every other training knob. A probe that needs a genuinely new module self-registers one head/token/encoder/temporal type from its own module via the `qnn.model.node_registry` decorators — not a whole network factory.

## Decode Base and Config

**`qnn.model.decode`** holds the cross-generation-stable decode primitives:

- Sampling primitives: `categorical_sample`, `bernoulli_sample`, `gumbel_argmax` (trace-safe, used by ONNX export).
- `decode_attack_bit` — sigmoid + threshold (greedy) or temperature-Bernoulli (sampled).
- `decode_move_axes` — per-axis categorical readout (argmax greedy / categorical-sample sampled).

What does NOT live here: generation-specific geometry (polar-look hybrid and
aim-prior blend) and stateful move layers (hysteresis, semi-Markov hazard,
watermarking) — those belong to the generation's decode module and are
referenced by the decode config.

**`qnn.model.decode_config`** resolves a run-pinned decode-config JSON (`decode_version` 1) into modules and parameters for export and eval. Schema fields: `decode_module` (dotted import — gen decode geometry), `guard_module` (dotted import or `"none"`), `version` (provenance string), `look_grid`, `move_hazard`, `params` (flat str→scalar/list map). The exporter stamps the resolved config's sha256 and the repo git sha into ONNX `metadata_props`. Regime names (`a24rc1`, `a24rc2`, etc.) are convenience aliases resolved to bundled template JSONs; A/B comparison is done by pointing at a different config file, not by changing code.

## Pipeline Alignment

- **BC** — `run_behavior_cloning(graph=...)` builds via `build_network` and stamps every checkpoint with the graph; bench probes are graph deltas (above).
- **Eval / export** — both load through `load_checkpoint`, which prefers the embedded graph over legacy flat-meta rehydration; nothing else changed (eval was already meta-driven).
- **PPO** — the native trainer loads the warm-start seed through `QNNPolicy`,
  prefers its embedded `meta.model_graph`, and fails loud if the frozen
  `model.json` disagrees. PPO updates that exact network directly; deployable
  best checkpoints remain self-describing QNN checkpoints with no intermediate
  encoder or checkpoint conversion.

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
| Flat-feature probe machinery (`features.py`, `flat.py`) | Superseded by full_Nhead deltas | per-head docs |
| `look_bypass_gru` ModelConfig flag | v17-only fidelity shim | — |

Old run dirs under `runs/bc/bench/` are immutable history; their `probe.json` files are not loadable by the new code and are not migrated.

## Training Speedup — Audit Verdicts (June 2026)

Cache format is frozen (disk budget). A code audit proposed in-process levers; each was checked against the live run's measured epoch profile (`bc_history.json`: train 291.6 s @ 30.3k rows/s, val 21.6 s @ 95.6k rows/s, train_eval 10.8 s). The forward-only val pass uses the same data path as training and runs 3.1× faster, so data loading / host-to-device transfer is **not** the training bottleneck — backward+optimizer is, at the standard ~2× forward cost. Verdicts:

| Proposed lever | Verdict | Why |
|----------------|---------|-----|
| Cache `segment_mask` filtering "per epoch" | Refuted | Filtering runs once at source construction (`streaming_source.from_cache_dir` → `_shard_segments`), not per epoch. |
| Wire up `pin_memory` (dead `BCConfig` field) | Refuted | H2D is not the bottleneck (val sustains 95k rows/s on the same path). Field kept — removing it breaks the strict machine.json schema of existing run-dirs. |
| Resident-source prefetch | Refuted | Resident tensors live on-device; there is no transfer to overlap. |
| Cache training-view fingerprint | Negligible | Manifest-level hashing, startup milliseconds; the resident daemon amortizes startup anyway. |
| Per-step grad-norm `.item()` sync | Already done | Grad norms accumulate as tensors, flushed at epoch end. |
| bf16 autocast | Already done | `dtype: "bf16"` is the template default (`QNN_AUTOCAST_DTYPE`). |
| `torch.compile` | Promoted, narrow scope | Compiling `obs_embedding` + `encoder` after the sync-free rewrite raised the retained real-corpus path from 50.6k to 76.0k RPS. Whole-model/loss compilation and compiled autograd remain net-negative; see `bc/train.py`. |
| Native PPO collection | Separate campaign | See `research/performance-tuning.md`; the bounded BF16 host-staged pipeline and confirmed 768-lane knee are the retained defaults. |

The real, already-built levers remain the resident ablation daemon (~2.5× vs streaming) and parallel ablation containers (4× sweet spot, 5× practical limit). Further single-job BC throughput means changing the training itself (lane count / model size), not the harness.

## Defensibility (arch-defensibility-sweep, 2026-08-04, N=6 seeds)

Every open ablation of the `core` base graph, measured at the seed-
DISTRIBUTION level against fixed rulers (FrikBot ladder + the a26rc1b
fitted bar + pairwise h2h). Instrument validation: ladder frags/min and
h2h win rate rank checkpoints identically (r=0.83, N=6) — closed-loop
seed variance is a stable checkpoint property, so distribution-level
acceptance is mandatory (any 2-seed verdict samples tails). Data:
runs/eval/ladder_sweep{,_rc1b}/, runs/eval/cellB_*/, runs/eval/cellC_*/;
plan agents/plans/arch-defensibility-sweep.md.

| Component / question | Cell | N | Effect (mean ± sd) | Verdict |
|---|---|---|---|---|
| weapon_ctx removal (the a28 cut) | A | 6 vs bar | frag-diff −4.48±0.57 vs bar −4.48; pace 5.54±0.46 vs 6.32 | **CUT STANDS** — differential parity; pace/return gap confounded with decode fit (bar fitted, arms on comparability template). Residual: decode-fit one median seed + re-ladder. |
| aux look_seg loss ("trunk effect") | B | 6+6 | h2h win 0.403±0.207 (common) / 0.368±0.136 (base); ladder sd 3× core's | **REJECTED** — the 8/3 "67/61% trunk lead" was the top two seeds of six; mean below 50% at both operating points; triples closed-loop seed variance. |
| 10 Hz decision rate (vs 20 Hz twin) | C | 6 pairs | win 0.320±0.128, CI on mean [0.22, 0.43]; own-rate ladder pace 3.91±0.87 vs 5.54±0.46 | **REAL PENALTY (~−18 pts)** — 20 Hz retained for the deploy line; 10 Hz stays a goal with known levers (fire-at-aim-completion coordination objective, pinning fine-tune). |
| move_seg, attack_with, polar look, softmax-feat inputs | — | — | settled pre-sweep (multi-way or outside-CI both seeds + rate-robust) | retained |

Seed-variance calibration (the number to hold against every pre-sweep
2-seed verdict): ladder frags/min sd — core 0.46, core@10 0.87,
lookseg-aux 1.38; h2h win-rate sd across seeds 0.13–0.21.
