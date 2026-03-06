# Quake AI V0 (E1M1 Imitation -> RL)

This repository implements a runnable V0 baseline to train a small discrete-control policy for original Quake-style E1M1 navigation.

## What is implemented
- Deterministic demo replay harness for JSON fixture demos and real NetQuake binary `.dem` files.
- Telemetry/packet collection to NDJSON:
  - `TelemetryTickV1`
  - `PacketEventV1`
  - `EpisodeSummaryV1`
- Map feature extraction (`MapFeaturesV1`) from BSP entity lump, JSON map metadata, or observed telemetry transitions.
- Corpus materialization from `download_manifest.ndjson` for map-specific training subsets.
- Behavior cloning with a PyTorch-backed 2-layer MLP policy trunk.
- PPO fine-tuning initialized from the BC checkpoint.
- Held-out greedy/sampled evaluation and model card/manifest outputs.
- Packet/telemetry alignment validation report.
- Native engine process boundary scaffolding with a JSON-over-stdio bridge and C stub server for contract testing.
- Explicit torch device selection plus GPU-friendly BC/PPO/distill batching when an accelerator is visible.

## Install
```bash
python -m pip install -e .
python -m pip install -e .[dev]
```

For AMD GPUs, use the isolated ROCm trainer with `scripts/train-container.sh` rather than trying to layer ROCm into your editor environment.

## End-to-end quickstart
From `src/`:

```bash
python -m quake_ai.collect --map E1M1 --demo_dir tests/demo_data --out ../artifacts/runs/collect --map_path tests/fixtures/e1m1_map.json
python -m quake_ai.validate_packets --telemetry ../artifacts/runs/collect/telemetry.ndjson --packets ../artifacts/runs/collect/packets.ndjson --out ../artifacts/runs/collect/packet_report.json
python -m quake_ai.train_bc --config configs/bc_e1m1.yaml
python -m quake_ai.train_rl --config configs/ppo_e1m1.yaml --init_ckpt ../artifacts/runs/bc/bc_best_model.npz
python -m quake_ai.eval --config configs/eval_e1m1.yaml --ckpt ../artifacts/runs/ppo/ppo_model.npz
python -m quake_ai.check_accelerator
```

## Isolated training container
From the repo root:

```bash
scripts/train-container.sh build
scripts/train-container.sh target
scripts/train-container.sh check
scripts/train-container.sh e1m1-corpus all
```

The trainer:
- auto-selects `docker/training/compose.amd-wsl.yaml` on Windows 11 + WSL hosts that expose `/dev/dxg`
- otherwise falls back to `docker/training/compose.amd-rocm.yaml` for native Linux ROCm hosts that expose `/dev/kfd` and `/dev/dri`
- uses AMD's ROCm PyTorch base image
- mounts the repo read-only at `/workspace`
- mounts a writable sibling host artifact directory at `/artifacts`
- keeps training outputs and corpus state outside the code repo while preserving the existing config paths

## Corpus quickstart
From the `src/` repo root, with an optional `dzip` binary available for classic SDA `.dz` files:

```bash
DZIP_BIN=/path/to/dzip python scripts/materialize_corpus_subset.py --map E1M1 --manifest ../artifacts/corpus/netquake/meta/download_manifest.ndjson --out ../artifacts/runs/e1m1_corpus/demos
python -m quake_ai.collect --map E1M1 --demo_dir ../artifacts/runs/e1m1_corpus/demos --out ../artifacts/runs/e1m1_corpus/collect
python -m quake_ai.train_bc --config configs/bc_e1m1_corpus.yaml
python -m quake_ai.train_rl --config configs/ppo_e1m1_corpus.yaml --init_ckpt ../artifacts/runs/e1m1_corpus/bc/bc_best_model.npz
python -m quake_ai.eval --config configs/eval_e1m1_corpus.yaml --ckpt ../artifacts/runs/e1m1_corpus/ppo/ppo_model.npz
```

## Outputs
- `../artifacts/runs/collect/`: telemetry, packets, summaries, map features.
- `../artifacts/runs/bc/`: BC model checkpoint, split manifest, history, summary, experiment manifest.
- `../artifacts/runs/ppo/`: PPO checkpoint/history/summary/manifest.
- `../artifacts/runs/eval/`: eval summary, model card, eval manifest.
- `../artifacts/runs/e1m1_corpus/`: materialized demos plus corpus-scale collect/BC/PPO/eval artifacts.
- `../artifacts/corpus/`: crawler manifests, raw payloads, extracted demos, and corpus worker logs.

## Notes
- Config files in `configs/*.yaml` are JSON-compatible YAML to avoid runtime parser dependencies.
- Packet traces are used for validation/alignment in V0, not as primary model inputs.
- The symbolic environment now uses heading-aware movement and requires `use` on the exit trigger to complete an episode.
- `materialize_corpus_subset.py` reads the crawl manifest and can extract classic `.dz` payloads when `dzip` is available via `PATH` or `DZIP_BIN`.
- The policy/training core now runs on PyTorch while preserving the existing CLI and artifact layout.
- Set `--device gpu` or `--device cuda`/`--device rocm` on training and evaluation CLIs once the container can actually see an accelerator.
- `python -m quake_ai.check_accelerator` is the fastest way to verify whether the current container has usable CUDA/ROCm/MPS access.
- `scripts/train-container.sh` defaults the host artifact root to a sibling `../artifacts` directory so heavy training outputs stay outside the repo checkout.
- `scripts/train-container.sh target` prints the selected AMD runtime path: `wsl` for `/dev/dxg` hosts, `rocm` for native Linux ROCm hosts, or use `TRAINING_AMD_TARGET=wsl|rocm` to override auto-detection.
- On AMD WSL hosts, the trainer bind-mounts `libdxcore.so` and `libhsa-runtime64.so.1` into the container so the ROCm runtime can bridge into the Windows driver stack.
- `scripts/train-container.sh migrate-legacy` will move old in-repo `runs/` and `corpus/` trees into the external artifact root if you are migrating an older checkout.
- `engine.native_bridge` is the start of the engine-backed architecture: Python remains the control plane, while a native worker process owns real-time simulation.
