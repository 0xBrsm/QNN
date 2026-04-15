# Changelog

## 0.15.0

### Added
- QuakeWorld demo worker (`qw_demo_worker`) — headless MVD/QWD playback with full feature parity to the NQ worker: QW physics, PVS culling, MVD playerstate view-angle recovery, sound precache
- BC collect rewritten for 10K+ demo scale — parallel with resume, sharded `.npy` output, persistent demo worker subprocess per worker, throughput + ETA progress reporting
- Look head cosine loss on the unit sphere, toggled via `look_cosine`; selection metric replaced with angular look + move + fire F1
- Copy-previous-action BC baseline (`qnn/bc/baseline.py`) for corpus difficulty benchmarking

### Changed
- Python package renamed `quake_ai` → `qnn`; reorganized into `bc/`, `ppo/`, `eval/`, `env/`, `run/`; C worker split into `engine/common/` + `engine/nq/` + `engine/qw/`
- Run templates split per pipeline (`qnn/bc/templates/`, `qnn/ppo/templates/`); BC runs drop `scenario.json`, `reward.json`, `eval.json`. Run directories partitioned by mode: `runs/<mode>/<name>/`
- Corpus layout moved to `assets/corpus/{dem,qwd}` with sharded output under `assets/collect/{prod,tmp}`

### Fixed
- Action history patched in the obs buffer at emit time (removes 2-frame lag)
- MVD playerstate circular-buffer aliasing corrupting view angles; move-label reversal via simvel anchor + pmove substeps
- QW `MAX_MSGLEN` raised 1450 → 65536 for modern QWD demos; `-game` dir derived from `--demo-dir`; match-text detection handles colored text via `svc_print` hook

## 0.14.0

### Changed
- Unified 3D wishdir move action: `move` float[3] replaces `move` float[2] + separate `jump` head
- 9-candidate BSP-clipped move inference with entity interaction, continuity bias, jitter filter, and outlier rejection
- Frame filtering (god-mode, dead-time, frozen-alive) moved from Python to C emission layer
- Span-based GRU forward batches contiguous sequences between resets; vectorized obs/action unpacking
- NONE sentinel removed from modality vocab (5 → 4)

### Added
- Binary snap labels for keyboard demos with sound-based jump Z detection
- Sharded precomputed caches, batch prefetch, pinned host memory for BC training throughput
- Train eval schedule with proxy gap and val regression early triggers
- Persistent demo worker subprocess reused across collections
- Demo corpus manifest and inventory; BC v6/v7 ablation run configs

### Fixed
- Demo worker state isolation between sequential collects
- Continuous move pass-through for PPO (was truncated to 2D)
- Mover over-push bug and dm4 hull guard crash in physics sim
- Match trimming for CHTV demos without match text markers

## 0.13.0

### Changed
- Demo movement labels reconstructed via BSP-clipped emission-window physics instead of position deltas
- Velocity derived from position deltas — spectator `cl.velocity` is always zero
- `evaluate.py`, `init_run.py`, `bc_collect.py`, `quickeval.py`, `observe.py` moved into `qnn/rl/`
- Demo package consolidated under `demo/`; `mapgen_cpp` removed

### Fixed
- Movement key inference: three bugs in button state, strafe direction, and ground detection
- Z-delta ground detection and BSP ground trace for spectator demos

## 0.12.0

### Changed
- Obs/action pairing aligned so actions are taken FROM the observed state
- Training loss averages only active heads; step log includes discrete metrics
- `torch.compile` evaluated and reverted — net negative for 189K param model

### Added
- `warmup_epochs` config for linear LR warmup before cosine decay
- `look_deadzone` config to zero jitter in look labels
- `sparse_discrete` config toggle for BC discrete head loss
- Per-epoch timing (`train_secs`, `val_secs`), wall clock timestamps, `epoch_done` sentinel

### Fixed
- BC resume starts fresh when no checkpoint exists instead of crashing
- Skip physics inference on done tick to prevent BSP crash
- `lr_override.json` used wrong json import alias

## 0.11.0

### Changed
- Per-type entity tokens (projectile/actor/item/mover) with type-specific scalar dims and projections
- Variable-length wire format with type tags replaces fixed entity struct
- Event vocab v2: subject/action/source triples — qualifier and magnitude dropped
- Cartesian look: 3D direction vector replaces yaw/pitch axes
- View-relative spatial directions; `dist` and `path_dist` added as explicit entity scalars
- `qualifier_embed` and `cluster_embed` removed from token pipeline

### Added
- Python v9 obs parser, tokenizer, and vocab (`obs_format.py`, `vocab.py`)
- Per-modality recency thresholds (SIGHT 2s, PROXIMITY/SOUND 0.1s, MEMORY 1s)
- DIMLIGHT sight-derived ACTIVE/POWERUP events on actors

### Fixed
- Armor and powerup embed masking (0-as-none convention)
- `entity_types` bounds, env/checkpoint schema tests
- `map_id` bridge bug; obs schema unified across env and checkpoint

## 0.10.0

### Changed
- C worker split into `qnn_entity.c`, `qnn_event.c`, `qnn_oracle.c`, `qnn_store.c` with typed structs
- Three-store world state (actor/object/projectile) wired into oracle, old structs removed
- Unified tick IO in `qnn_io.c`; action history folded in, headers merged
- v8 token spec implemented in C worker (Phases 1–3)
- Entity store populated from server baselines instead of BSP parse order
- Modality system: SIGHT requires FOV, PROXIMITY for PVS-only, SOUND omnidirectional
- Legacy region system and server edict dependency removed

### Added
- Sparse binary loss masking for fire and jump BC heads
- Hot-reload LR from `lr_override.json` during BC training
- Mover state tracking (doors, platforms, trains, buttons)
- Per-store token emitters with normalized scalars

### Fixed
- Health normalization /100 (mega=2.5), effective armor /160
- Nav oracle walk speed 300 → 320 (`sv_maxspeed` default)
- Pickup/respawn/teleport sound classification via `QNN_EmitRecord`
- `QNN_MAX_SOUNDS` 16 → 128 to match engine `MAX_CHANNELS`
- Corpse filtering, powerup lifecycle, edict reuse, velocity spikes

## 0.9.0

### Changed
- C worker split by concern: monolithic `qnn_worker_main.c` replaced by `qnn_trainer_main.c`, `qnn_demo_main.c`, `qnn_obs.c`, `qnn_reward.c`, `qnn_metrics.c`, `qnn_world.c`, `qnn_sys.c`
- Unified header `qnn.h`, PascalCase all worker functions, Quake-style naming
- BC collect pipeline rewritten: parallel `ProcessPoolExecutor`, resume support, look smoothing (window=3)
- Renamed`training_bc.py` to `bc_train.py` with dead token/precompute code stripped

### Added
- Demo worker emits binary obs buffers for BC collection (`PackObsBuffer`)
- Separated metrics API from reward computation (`qnn_metrics.c` / `qnn_metrics.h`)
- O(1) event lookup via per-owner linked list with inline `WeaponIndex`
- `obs_format.py` canonical obs format shared between C worker and Python
- `bc_status.py` for checking BC training progress from `bc_history.json`

### Fixed
- Critical `memset` bug zeroing wrong size in worker state init
- `R_AddEfrags` NULL guard and `MAX_OSPATH` increased to 512 for long demo paths
- Non-fatal model precache for headless demo playback
- Entity `half_extents` derived from model bounds during demo playback

## 0.8.0

### Changed
- Object tokens widened from 8 to 13 scalars: bbox half-extents (3) and look-hint axes (2)
- Reward computation moved to C worker (QTRN v2)
- Binary action protocol for step hot path
- BC data pipeline decoupled into standalone collect/precompute scripts
- `.npz` eliminated; all checkpoints use `.pth`

### Added
- Optuna sweep mode with parallel containers, orphan recovery, live config reload
- Configurable inventory system in `scenario.json`
- `eval.json` with fixed seed pool and standard eval surface
- BC fine-tuning from seed checkpoint with per-column learning rate tracking

### Fixed
- Modality priority overwrite (VISUAL > AUDITORY > MENTAL)
- CheatCommand no-op in deathmatch; bots spawning unarmed
- Bbox extent normalization scale mismatch with relative positions

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
