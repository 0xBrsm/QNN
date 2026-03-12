## 0.3.0

### Added
- Transformer encoder (`entity_transformer.py`): tokenized self/object/spatial trunk with
  event-trace attachment and CLS pooling
- `identity.py`: vocabulary of 64 category/subtype/event indices across all entity types
- `token_observation.py`: shared live/replay token observation packer
- `reward.py`: 4-signal PvP reward (frag_bonus, death_penalty, ehp_delta, edp_delta)
- Sample Factory APPO integration: `sf/train.py`, `sf/quake_env.py`, `sf/quake_encoder.py`,
  `sf/checkpoint_converter.py`
- `autotune/train.py`: git-ratchet hyperparameter tuning harness with TSV logging
- Sound capture in native worker (`qnn_worker_sound.c`)
- Token bridge and protocol (`token_bridge.py`, `token_protocol.py`, `qnn_worker_token.c`)
- Demo worker for token playback (`qnn_demo_worker_main.c`)
- `semantic_vocab.py`: semantic vocabulary with fusion rules and object/event mapping
- `training_protocol.py` and `qnn_worker_training.c`: training-side engine protocol
- Root documentation: `overview.md`, `event_vocab.md`, `outstanding.md`

### Removed
- Python collection pipeline (`collector.py`, `netquake_demo.py`, `world_stream.py`,
  `packet_validation.py`, `validate_packets.py`)
- Legacy encoders (`competitive_encoder.py`, `world_encoder.py`)
- E1M1/world_v2/campaign combat configs and training paths
- `training_rl.py`, `training_distill.py`, `train_distill.py`, `train_rl.py`
- Sample Factory temperature shim (`sf/quake_action_dist.py`)
- Legacy shell pipeline wrapper (`scripts/run_v0_pipeline.sh`)
- Navigation module (`navigation.py`)
- Engine adapter (`engine/adapter.py`)

### Changed
- Pipeline is now PvP-only: competitive BC warm start plus SF APPO fine-tuning
- Observation format is dict-only with self/object/event/spatial tokens
- `NativeWorldEnv` returns token dict observations via `token_step_v2` interface
- Demo classifier slimmed and renamed NQ→QNN
- Worker renamed from `nq_worker` to `qnn_worker`
- Wire protocol magic bytes renamed to Q-prefix: `QWLD`, `QTOK`, `QTRN`
- Engine bridge consolidated (`native_bridge.py` + `token_bridge.py` → `bridge.py`)
- `data/` renamed to `demos/`, `demo_classifier.py` → `classifier.py`
- Schema classes drop version suffixes (`WorldTickV2` → `WorldTick`, etc.);
  backward-compatible aliases retained
- Removed dead V1 schema classes (`TelemetryTickV1`, `PacketEventV1`, `EpisodeSummaryV1`)
  and `schema_version` tags from `to_dict()` output
- Removed dead code: `demo.py`, `models/identity.py`, `engine/adapter.py`

## 0.2.0

### Added
- Demo corpus classifier (`demo_classifier.py`) with serverinfo/stufftext cvar parsing
- Competitive demo metadata extraction and BC bootstrap configs
- FrikBotNex bot training integration with combat ladder and recurrent live training
- Combat-survival live training path with progress reward
- Player-like live action space (`actions.py`)
- Live-run reporting and parity coverage

### Changed
- Demo parser tolerant to unknown opcodes with competitive heuristics
- Stabilized ROCm trainer GEMM path and runtime checks
- Adopted player-like controls for live action space

## 0.1.0

### Added
- E1M1 imitation-to-RL v0 baseline (`train_bc.py`, `training_bc.py`, `training_rl.py`)
- NetQuake demo corpus crawler and source ingestion pipeline
- Symbolic navigation training
- E1M1 corpus training pipeline
- Native Quake worker with asset-backed world model (`nq_worker_main.c`, `nq_world_model.c`)
- Engine world model and structured data path
- PPO evaluation loop
- ROCm trainer support with WSL bridge (`train-container.sh`)
- Retained ROCm live training pipeline
