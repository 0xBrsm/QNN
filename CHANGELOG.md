# Changelog

## 0.7.0

### Changed
- Training pipeline standardized on `run.json` single entry point, flat `reward.json` per run, unified `checkpoint_path`
- Container simplified to single compose file; vendored FrikBotNex QC at repo root

### Added
- Arena mode with flat box map, all-weapons spawn, Red Armor, LG default, and `self_damage_penalty` reward
- 4-way parallel PPO learning-rate sweep watchdog and host metrics collector

### Fixed
- Worker segfault on spatial raytrace; FrikBots invisible to model via server-side actor detection

## 0.6.0

### Changed
- Final BC model form: 2-layer transformer (d_model 64, 1 head, FFN 256), self-readout, GRU temporal core, 2 action history tokens
- Full sequential episode training with GRU carry-forward, per-chunk gradient accumulation, and TBPTT
- Ablation phases 1–6: 2 action tokens win (+72%), GRU redundant with action tokens, trunk sizing and focal loss evaluated
- Cosine LR decay, regression-based stopping on MAE sum, resumable checkpoints with NAS archival

### Added
- Step-level BC dashboard, Prometheus metrics exporter, memory-mapped episode cache with parallel precompute
- 6-class weapon embeddings with axe class; configurable readout, look smoothing, and tick resampling

### Fixed
- Action labels computed at emission points; look deltas accumulated across resampling window
- Val MAE sum for early stopping; zeroed heads excluded from eval loss mean

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
