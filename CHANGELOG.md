# Changelog

## 0.5.0

### Added
- Autonomous PPO training loops with gated status snapshots, retained run metrics, and hyperparameter history for live PvP runs
- Multi-seed PBT warm starts, fresh output roots, and explicit PPO launch overrides for retained foundation runs

### Changed
- Promoted the runtime surface from `sf` to `ppo`, including the APPO env/train entry points and `ppo/` run layout
- Behavior cloning now trains full sequential episodes with GRU state carry-forward, focal look losses, label smoothing, and auxiliary aim supervision
- Evaluation and reporting now track richer episode metrics, effective game time, and parallel verify policy modes

### Fixed
- Best-checkpoint archival and warm-start selection no longer collide or drop retained policies
- CHTV spectator demo parsing now bounds server string copies to avoid crashes on long names and lightstyles

## 0.4.0

### Added
- Navmesh system: Recast/Detour query, traversal graph, routing cache, macro clustering
- Procedural map generator (C++ and Python) with WAD generation and navmesh validation
- Token spec v5: weapon classes, cluster embeddings, spatial tokens
- FFA multi-bot training with procgen maps, PBT, and demo recording
- BC episode precomputation and batch prefetching for GPU utilization

### Fixed
- Auto-respawn on death, free ehp_delta reward on respawn
- SF checkpoint loading under PyTorch 2.6+, recurrence/report interval patches
- Procgen maps: WAD race condition, surface extents, stair geometry, entity placement
- Demo archiving collisions, warm-start checkpoint selection

### Changed
- Training pipeline decoupled from profiles, flattened run directory
- Policy renamed `MLPGRUPolicy` → `QNNPolicy`, procgen maps generated inline
- Priority-sorted object emission capped at 64 tokens

## 0.3.0

### Added
- Transformer encoder with tokenized self/object/spatial trunk and CLS pooling
- Identity vocabulary, token observation packer, 4-signal PvP reward
- Sample Factory APPO integration, autotune harness, sound capture
- Token bridge/protocol, demo worker, semantic vocabulary

### Removed
- Python collection pipeline, legacy encoders, E1M1/campaign configs
- Legacy training scripts, navigation module, engine adapter

### Changed
- PvP-only pipeline: BC warm start → SF APPO fine-tuning
- Dict-only token observations, NQ→QNN rename, schema version suffixes dropped

## 0.2.0

### Added
- Demo corpus classifier, FrikBotNex bot training, player-like action space

### Changed
- Demo parser tolerant to unknown opcodes, stabilized ROCm GEMM path

## 0.1.0

### Added
- E1M1 imitation-to-RL v0 baseline with demo corpus pipeline
- Native Quake worker, PPO evaluation loop, ROCm trainer with WSL bridge
