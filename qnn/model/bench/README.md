# bench — component-ablation library

Uses the canonical BC training pipeline. Per-component ablations are **graph
deltas**: `probe.json` names a `base` graph plus `overrides` (see
`qnn.model.graph`), and the runner builds the model with `build_network`. Every
checkpoint embeds its resolved `model_graph` (self-describing) — there is no
factory/registry reload path. (The legacy `HEADS` registry + `full_*.py`
assemblies were retired once the rc lineage migrated to graph checkpoints;
concluded ablations are archived under `runs/_archive/bc/a24/`.)

The runner translates a bench run-dir into a `BCConfig` + the resolved
`GraphSpec` + `head_loss_weights` and hands it to `run_behavior_cloning`. Shard
loading, segment_mask filtering, fingerprint verification, episode shuffling,
BPTT batching, eval cadence, checkpointing, and `bc_history.json` are all the
canonical BC implementations.

## Layout

| path | role |
|---|---|
| `inputs/` | slot components (encoders/pointers/embeddings) used by graph nodes |
| `a24/` | a24-generation node overrides of the `qnn.model` base (heads/decode/rc1), registered into the graph node registry |
| `future/` | parked primitives not wired into any probe |
| `side_channels.py` | label-derived forward-scoped contexts |
| `runner.py` | run-dir / `qnn.run.router` entry point |
| `templates/` | run.json + train.json + machine.json + probe.json + run.md |

Concluded single-head experiments were removed; their findings live in
`src/docs` (see model-graph.md "Legacy Ablation Paths").
