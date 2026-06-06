# Quake Neural Network

Transformer policy for competitive Quake PvP, trained via behavioral cloning
from demos and PPO fine-tuning against bots. A native C worker observes game
state and emits semantic tokens; a transformer encoder with a GRU temporal
core attends over them, a supervised TargetPointer picks the engagement
target, and factored heads (move, look, fire, weapon) drive the engine.

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
| `qnn/model/` | Transformer tokenizer, trunk, TargetPointer, GRU actor-critic policy |
| `qnn/run/` | Run directory management, config, router |
| `qnn/bc/` | Behavioral cloning — demo collection, target labeler, supervised loop |
| `qnn/ppo/` | PPO — Sample Factory APPO integration and RL training |
| `qnn/eval/` | Evaluation — run checkpoints against bots and live NQ servers |
| `qnn/env/` | Live engine interface — NativeWorldEnv, reward, planning |
| `qnn/diag/` | Capacity diagnostics for trained policies |
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
| [Token Specification](docs/token-spec.md) | Wire format, obs buffer layout, head shapes, action struct |
| [Semantic Vocabulary](docs/vocab.md) | Entity, action, modality IDs and event mapping |
| [Vendored Dependencies](docs/vendor.md) | Third-party dependencies and upstream sources |
| [Input Inference](docs/input-inference.md) | Recovering player-intent labels from server signals (fire/jump) |
| [Target Labeler and Engine Sticky](docs/target_labeler_engine_alignment.md) | Adaptive-cone Schmitt-trigger labeler + engine alignment |
| [Diagnostics](docs/diag.md) | `qnn.diag` capacity diagnostics for trained policies |
