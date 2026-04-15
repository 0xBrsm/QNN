# Quake AI

Transformer policy for competitive Quake PvP, trained via behavioral cloning
from demos and PPO fine-tuning against bots. A native C worker observes game
state and emits semantic tokens; a transformer encoder with GRU temporal core
produces actions through factored continuous and discrete heads.

## Quick Start

See [docs/setup.md](docs/setup.md) for the full path from clone to trained
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
| `qnn/model/` | Transformer tokenizer, trunk, GRU actor-critic policy |
| `qnn/run/` | Run directory management, config, router |
| `qnn/bc/` | Behavioral cloning — demo collection and supervised training |
| `qnn/ppo/` | PPO — Sample Factory APPO integration and RL training |
| `qnn/eval/` | Evaluation — run checkpoints against bots, record demos |
| `qnn/env/` | Live engine interface — NativeWorldEnv, reward, planning |
| `engine/` | C worker source — `common/` (shared), `nq/` (NetQuake), `qw/` (QuakeWorld) |
| `engine/build/` | Build scripts for worker binaries |
| `demo/` | Quake `.dem` parser and label extraction |
| `mapgen/` | Procedural `.map` generation |
| `docker/` | Trainer container (Dockerfile, compose, entrypoint) |

## Documentation

| Doc | What it covers |
|-----|----------------|
| [setup.md](docs/setup.md) | Prerequisites, building, demo corpus, training walkthrough |
| [overview.md](docs/overview.md) | Architecture, reward system, training surface, source file map |
| [run.md](docs/run.md) | Run directory schema and config reference |
| [token-spec.md](docs/token-spec.md) | Wire format, obs buffer layout, action schema |
| [vocab.md](docs/vocab.md) | Entity, action, modality IDs and event mapping |
| [vendor.md](docs/vendor.md) | Third-party dependencies and upstream sources |
