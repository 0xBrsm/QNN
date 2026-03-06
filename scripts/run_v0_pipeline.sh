#!/usr/bin/env bash
set -euo pipefail

python -m quake_ai.collect --map E1M1 --demo_dir tests/demo_data --out ../artifacts/runs/collect --map_path tests/fixtures/e1m1_map.json
python -m quake_ai.validate_packets --telemetry ../artifacts/runs/collect/telemetry.ndjson --packets ../artifacts/runs/collect/packets.ndjson --out ../artifacts/runs/collect/packet_report.json
python -m quake_ai.train_bc --config configs/bc_e1m1.yaml
python -m quake_ai.train_rl --config configs/ppo_e1m1.yaml --init_ckpt ../artifacts/runs/bc/bc_best_model.npz
python -m quake_ai.eval --config configs/eval_e1m1.yaml --ckpt ../artifacts/runs/ppo/ppo_model.npz
