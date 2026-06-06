# Changelog

## 0.21.0

### Added

- Dynamic BC ablation daemon + reusable ablation-sequence runner (BC-source bundle plumbing) on top of the `head_probe` run-dir mode
- Attack-head probe family: `engagement_ema` (op-frame EMA prior, α configurable via `probe.json`), `geom` (radial_vel/tang_speed/dist), `weapon_embed`, and the stacked `engaged_geom_weapon_embed`; `attack_op_only` loss/metric gating
- PPO BC warm-start that survives modern arch drift: SF obs-space matched to the native worker, v23 meta normalization, overnight smoke script
- `diag` attack-head input-column ablation + saliency report (also ablates encoder `self_scalars`); temporal look-head probe (prev_look + aim_vec prior, K-sweep)

### Changed

- Hyperparameters standardize on a `d_*` prefix (mirrors `n_*`); per-head MLP dims split into scalar fields (`bottleneck_dim` → `d_hidden`); the MLP target-head variant promoted to canonical (legacy → bench)
- Target metrics locked to `val_target_kl` + `val_target_kl_multi`; the slot-confounded `acc_target*` family dropped
- `input_mask` persisted in checkpoint meta (pre-fix reloads silently defaulted to `False`); `best_val_loss` renamed

### Removed

- `src/docker/runs/` leak

### Fixed

- BC obs capture made MDP-correct: obs snapshotted pre-`Host_Frame` so `(obs[t], action[t])` align, with pre-loop `attack_finished`
- PPO encoder/core arity mismatch (3-way → 2-way split); worker stdout kept protocol-clean for trainer/demo workers

## 0.20.0

### Added

- `qnn.model` package: `Network`, `MoveHead`/`LookHead`/`AttackHead`/`WeaponHead`, `Temporal`, `TransformerEncoder`, and `TargetPointer` extracted from `policy.py` as composable components, with dataclass I/O and a reshape-once `Network.forward`
- Slot-configurable `Network` with an `Off` sentinel, plus `PreAttnEncoder` and `GTTargetPointer` for component-slot ablations; `qnn.model.testing` harness
- Self-token redesign: self split into state/arsenal/motion subtokens with a CLS readout
- `input_mask` per-axis engine-act byte (jump vs upmove split); residual-on-geometric-prior `AttackHead` (mirrors the look head); `hit_test` C primitive + ctypes/torch port as an attack-prior mode
- One BC training platform behind `train_on_batches` with two data pipelines — GPU-resident and chunked streaming — plus `parallel_prefetch_iter`

### Changed

- `fire` → `attack` end-to-end (action axis, heads, locals) to match QW `button0` / `BUTTON_ATTACK`
- `Tokenizer` → `ObsEmbedding`, `TokenizedFeatureEncoder` → `PreAttnEncoder`, `trunk` → `encoder`, `*_token` → `*_preattn`; `target_dist` → `target_probs`, `slot` → `idx`
- `op_input` → `input_mask` (1 = operative), now pure feasibility with no demo-press AND-gate; `microbatch_size` → `batch_size` (fails loudly on the old name)
- BC heads/probes refactored to `Network` slot overrides; `qnn.bc.components` → `qnn.model.bench`, alternate slot inputs to `qnn.model.bench.inputs`

### Removed

- `ModelConfig.from_dict` back-compat defaults; attack-prior back-compat from the live load path; prior modes stripped from the canonical `AttackHead` (moved to `bench/attack_prior/`); dead labeler-v2 hard-label code

### Fixed

- Checkpoint converter loads pre-rename v23 BC checkpoints; `@dataclass(slots=True)` restored after the over-broad `slot` → `idx` rename
- `input_mask` attack-feasibility off-by-one; `qnn_action_t` shrunk to 16 bytes (single press byte mirrors `input_mask`)

## 0.19.0

### Added

- v3 confidence-distributed target labeler (`qnn.bc.target_labeler`): weapon-aware analytic-cone lead correction, noisy-OR aggregation, logistic engagement confidence; emits a soft `act_target_dist.npy` and TargetPointer trains by soft-CE. Predicates evaluated against a real QuakeC VM (`qwprogs.dat`); shared `weapon_physics.py` projectile/lead model
- Native-obs wire format (`engine_norm`): canonical native-width field tables for the self/spatial/entity/action blocks with Self/Spatial/Entity/Action dequantizers and a `wire.py` native buffer parser
- In-process ONNX inference (`qnn_onnx.c` / libqnn) for `nq_client`, dropping the Python pipe; console-driven `qnn_client` (Quake-style argv, `-model` flag, idle-without-model startup, `.onnx` auto-append), shipped via `docker/Dockerfile.live` and the `engine/build` client scripts
- `head_probe` run-dir mode: per-head MLP ablation pipeline where `probe.json` is the only entry point (no Python-level defaults); flat-feature fire/target probes migrated onto the canonical BC pipeline
- `op_input` operative-feasibility column + QC-VM `noop_input_mask` per-tick no-op signal; `op_input_mask` loss toggle that drops held-but-ignored frames per axis
- Legacy QNN checkpoint-meta converter (`migrate_legacy_flat_meta`) for pre-`engine_norm` weights

### Changed

- BC collect writes native-width obs on disk with a sparse `target_dist`; parallel per-shard loader + numpy densify (17 min → ~3 min preload), GPU-resident epoch loop with pre-dequantize, chunked per-batch dequant (~75k rows/s) and `MADV` page-drop
- `pmove` seeded from `cl.simorg`/`simvel` and prev-tick state (closes a 91% recall gap); `op_jump` derived from pmove substeps; `self.weapon` persisted across QC `ImpulseCommands`; fire/jump predicates driven per-cmd
- `self_items` u32 → i32, `half_extents` u8, `act_fire` packed into `act_move` bit 6; item-amount normalization moved C → model lookup; entity table drops the redundant `dist` scalar
- self token gains an `attack_finished` cooldown scalar (`SELF_SCALAR_DIM` 16→17, `QNN_TIME_SCALE`-normalized) and a current-weapon embedding
- `ModelConfig` centralizes the architecture (defaults/ReLU stripped); oracle entity order is pool-then-edict (sort stripped)

### Removed

- Wire back-compat passthrough (native obs is the only contract), the legacy f32 wire parser, and `act_target_dist` from the wire (recomputed at training start)
- Legacy precomputed BC loader; `demo/sanitize.py` (superseded by QC-driven `noop_input_mask`); v3 labeler hard-label compat shims and the hard per-weapon range gate

### Fixed

- Case-sensitive PAK glob on Linux; shards written in submit-order without blocking workers; completed futures popped so parent RSS doesn't grow with worker count
- Checkpoint converter defaults fire/jump `distance_sigma` to 0.0 in `migrate_legacy_flat_meta`

## 0.18.0

### Added

- `qnn.diag` package for trained-policy analysis: ablation, attention, convergence, gradients, linear_probe, participation, pruning, rank, report — driven by the `/diag` skill
- `qnn.labeler.probes.target_head_probe` standalone causal TCN slot probe + GBT variants for offline target-head analysis
- `qw_classifier` C binary replaces the Python QWD classifier end-to-end
- `nq_client` binary + `qnn.eval.live` — canonical live-play entry point against real NQ servers
- Per-head F1 headline metrics with per-class weapon/move visibility; `head_loss_weights` for per-head loss gating; `jump_pos_weight` with linear decay
- Collection-identity fingerprint sidecar verified at train time; sub-episode splitting on filter-mask drops; `segment_mask` / `token_mask` config knobs
- Native-rate labeler LOBS stream with per-tick `target_valid_mask`; engine-side sticky engagement gated on PVS modality

### Changed

- BC training loop split: chunked `supervised_loop.py` extracted from `loop.py`; class-weight derivation, token-mask filter, filter DSL each in their own modules
- Engine collect modularized: monolithic `qnn_collect_main` split into `qnn_mvd_collect`, `qnn_qwd_collect`, `qnn_labeler_collect`; common runtime via `qnn_collect_helpers`, `qnn_fault`, `qnn_watchdog`, `qnn_tick`
- MVD fire/jump labels: ping-driven walk-back with per-record demotime back-shift; sound + velocity-sign jump detection at native tick rate
- QW worker hardening: per-demo SIGALRM, bounds-checked baselines, protocols 24-27 accepted, graceful post-signon `svc_disconnect`, 4× statics cap, per-demo worker restart
- Checkpoint converter trimmed to v17 + v20 era with `drop_action_history` and `drop_fire_align_scalar` migrations
- BC templates: 8 epochs, flat LR, bf16, `head_loss_weights` default; PPO scenario skill 0 → 3

### Removed

- `microbench`, `attn_dump`, tactics head + labeler, `length_bucket_window`, regression-stop, loss-shape ablation knobs, `MODEL_VERSION` tag, legacy precomputed manifest path

### Fixed

- QWD other-player entities now register for actor-token emission; spectator demos filtered via `svc_serverdata` bit
- Five mid-demo crash classes in the QW worker (cross-demo `tick_emit` jitter reset, unknown `svc` codes, missing actor-store creation, etc.)
- BC sub-episodes split on filter drops so the target labeler never reads across dropped intervals

## 0.17.0

### Added

- TargetPointer head: pointer distribution over actor slots; `target_feat = probs @ actor_tokens` conditions move/look/fire/weapon. Target is supervised internally, not a sampled action
- Supervised target labels via `qnn.bc.target_labeler`: cone-anchored + Schmitt-trigger release + sticky-by-PID engagement; on-disk `actions["target"]` is (T,) int64 with -100 ignore
- 8-class weapon head emits a direct Quake impulse byte (1-8); engine action-byte 28 renamed `switch` → `weapon`
- Per-weapon one-hot ownership in `self_scalars` — SG/SSG/NG/SNG each get their own bit instead of the paired 0/0.5/1.0 floats

### Changed

- Token spec v9 → v11; entity vocab 42 → 44 with SHOTGUN/SUPER_SHOTGUN and NAILGUN/SUPER_NAILGUN as distinct rows, weapons renumbered into Quake impulse order (ids 3-10) so SG/SSG and NG/SNG embed rows sit adjacent
- `self_scalars` 14 → 16; `entity_stream` offset shifts back to byte 564 (action_history wire region removed); max obs ~2389B
- `move` action: continuous float[3] → 3 categorical axes (fb/lr/ud) × 3 classes {neg, none, pos}. Engine still receives float[3] in {-1, 0, +1}; on-disk corpus stores 6-bit packed uint8
- Encoder returns a tuple `(self_readout, target_feat, target_logits, target_query)` instead of a single tensor; downstream heads consume `target_feat` directly. GRU input remains `self_readout` (the in-branch `linear(cat(self_readout, mean(actors)))` experiment lost to `gru_input_self_only=True` and was baked out)
- Corpus on-disk action format: packed `move` uint8, fp16 `look`, raw engine `weapon` byte (no-weapon frames carry 0 and are masked from CE)

### Removed

- `recall_0..3` heads and recall wire bytes; `qnn_store[].mem` kept dormant as a revival hook
- `action_history` ripped from wire, schema, and ObsEmbedding end-to-end; checkpoint converter keeps `migrate_drop_action_history` to strip pre-rip-out weights
- 5-slot `switch` head end-to-end (replaced by direct-impulse `weapon`)

## 0.16.0

### Added

- PPO rollout workers run inference on CPU in parallel; 2.4× over SF's central GPU inference
- BC training ~2× faster: bf16 autocast via `QNN_AUTOCAST_DTYPE`, AOTriton + hipBLASLt
- Deterministic mid-epoch resume via wall-clock `snapshot_interval`; `MADV_SEQUENTIAL`/`MADV_DONTNEED` on mmap shards bounds page-cache usage
- Threat- and team-aware entity token slot ordering
- Engine `qnn_train` cvar (`"arena"` / `"target"` / off) replaces boolean `qnn_arena_mode`; `"target"` freezes a frikbot for aim-practice runs
- Per-head loss weights for PPO (`--head_loss_weights`): weight 0 zeros that head's log-prob, entropy, and KL gradient end-to-end through the SF `TupleActionDistribution`
- `look_cosine` L2-normalization, look-head bias-init to `[1, 0, 0]`, and look sample normalization in `env.step`; `initial_stddev` plumbed from `train.json`
- `qnn.env.sim`: offline trainer for static token scenes (oracle / sl / sl-episodic / rl curriculum)

### Changed

- SF `RunningMeanStd` obs normalizer disabled (encoder already pre-scales); Adam uses fused kernel on CUDA/ROCm
- Demo analyzer split: `demo.analyze` → `demo.classify` with mode classification
- `worker_inference_device` moved from env var into `machine.json`
- PPO runs accept empty `checkpoint_path` as random-init; `eval` still requires one
- Tracking reward + entity metrics migrated from orphaned `snapshot->known[]` to `qnn_store[]` — the latent zero-reward bug from the 0.10.0 entity-pipeline refactor is fixed

### Fixed

- PPO second-reset hang: `QNN_PrepareMap` aliased the buffer `QNN_FreeMapState` zeroed mid-call. Latent since 0.10.x
- PPO warm-start silently ignored for any seed filename without `best_` in it; SF's scan for `checkpoint_<step>_<envsteps>.pth` missed it and orthogonal-init'd. Now always copies to `checkpoint_000000000_0.pth`
- QW demo worker stability at scale: per-worker config isolation, BSP presence filter, fail-fast on stuck signon, per-demo SIGALRM timeout, baseline-seeded entities
- SF `worker_inference` crashes on empty valid-ratios minibatches and on 0-D scatter outputs at `num_envs_per_worker=1`

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
- Ablation phases 1–6: 2 action tokens win (+72%), GRU redundant with action tokens, encoder sizing and focal loss evaluated
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
- Transformer encoder with tokenized self/object/spatial encoder and CLS pooling
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
