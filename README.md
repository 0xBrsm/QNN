# Quake Neural Network

Transformer policy for competitive Quake PvP, trained via behavioral cloning
from demos and PPO fine-tuning against bots. A native C worker observes game
state and emits semantic tokens; a transformer encoder with a GRU temporal
core attends over them, a supervised TargetPointer picks the engagement
target, and factored heads (move, look, attack, weapon) drive the engine.

## Quick Start

See [Setup Guide](docs/setup.md) for the full path from clone to trained
model: container setup, building the engine, preparing demos, and running
training.

```bash
# Build (inside container)
scripts/build-mod.sh

# Create and run a BC training run
python -m qnn.run.init --name bc_v1 --mode bc --resume true
python -m qnn.run.router --run-dir runs/bc/bc_v1
```

## Source Layout

| Path | Purpose |
|------|---------|
| `qnn/` | Python model, training pipeline, observation contract |
| `qnn/model/` | Declarative model graph (`graph/`, `node_registry`, `tokens/`): observation embedding, transformer encoder, GRU temporal core, TargetPointer, factored heads, and the decode layer |
| `qnn/run/` | Run directory management, config, router |
| `qnn/bc/` | Behavioral cloning — demo collection, target labeler, supervised loop |
| `qnn/ppo/` | Native bounded PPO — vectorized collection, host-staged rollout pipeline, and recurrent learner |
| `qnn/eval/` | Evaluation — run checkpoints against bots and live NQ servers |
| `qnn/env/` | Live engine interface — NativeWorldEnv, reward, planning |
| `qnn/diag/` | Per-head analysis (`qnn.diag analyze`) and capacity diagnostics |
| `qnn/labeler/probes/` | Standalone target-head probes (causal TCN, GBT) |
| `engine/` | C worker source — `common/` (shared), `nq/` (NetQuake), `qw/` (QuakeWorld) |
| `engine/build/` | Build scripts for worker binaries (`ppo_worker`, `nq_demo_worker`, `nq_client`, `qw_demo_worker`, `qw_classifier`) |
| `demo/` | Quake `.dem` parser and label extraction |
| `mapgen/` | Procedural `.map` generation |
| `docker/` | Trainer container (Dockerfile, compose, entrypoint) |

## Documentation

| Doc | What it covers |
|-----|----------------|
| [Setup Guide](docs/setup.md) | Prerequisites, building, demo corpus, training walkthrough |
| [Overview](docs/overview.md) | Architecture, heads, reward system, training surface, source file map |
| [Training Config Matrix](docs/run.md) | Run directory schema and config reference |
| [Contract Registry](docs/contracts/README.md) | Wire / semantics / arch contract versions, ONNX I/O signatures, the load set |
| [Semantic Vocabulary](docs/vocab.md) | Entity, action, modality IDs and event mapping |
| [Vendored Dependencies](docs/vendor.md) | Third-party dependencies and upstream sources |
| [Diagnostics](docs/diag.md) | `qnn.diag` per-head analysis + capacity diagnostics |
