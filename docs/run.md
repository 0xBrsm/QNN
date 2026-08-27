# Training Config Matrix

This is the source-of-truth reference for the run-dir training schema.

Templates live in each pipeline's package (`qnn/bc/templates/` or
`qnn/ppo/templates/`). `qnn.run.init` copies them into
`runs/<mode>/<name>/config/`, and training reads only the frozen files
inside that run directory.

## Rules

- `run.json` is the single entry point.
- Every config path in `run.json.config` is required — and no more.
- BC runs only declare `train`, `machine`, and `model`. PPO / Optuna add
  `scenario`, `reward`, and `eval`.
- `run.json.resume` is the only resume switch.
- No code-level defaults fill in missing training keys.
- Public run modes are `bc`, `ppo`, `eval`, and `optuna`; `pbt` is retired.
- `run.json.output.checkpoints` is the PPO checkpoint root. Native PPO keeps
  one atomic `ppo_state_<run_id>.pt` resume state plus deployable best models.
- Generated Optuna trial wrappers live under `runs/optuna/<name>/trials/`.
  Each wrapper keeps `params.json`, `reward.json`, and `diagnostics.json`,
  and the child PPO run lives under `ppo/`. They are runtime output, not
  curated run manifests.

## Run Directory

BC:

```text
runs/bc/<name>/
  run.json
  run.md
  config/
    train.json
    machine.json
    model.json
  checkpoints/
  metrics/
  logs/
```

PPO / Optuna:

```text
runs/<mode>/<name>/
  run.json
  run.md
  config/
    train.json
    scenario.json
    reward.json
    machine.json
    model.json
  checkpoints/
  metrics/
  logs/
```

## Checkpoint artifacts

Naming and retention are defined in `qnn.utils.artifacts`; identity comes
from `run.json.run_id`, never from directory names.

| File | Purpose |
|------|---------|
| `checkpoints/ckpt_e<epoch>_<run_id>.pt` | rolling resume checkpoint (model + optimizer). Exactly one per run: each epoch's atomic write (tmp + fsync + rename) supersedes and deletes the previous epoch's file |
| `checkpoints/best_<run_id>.pth` (+ `.json` sidecar) | best model by selection score; the deploy/eval artifact. Native PPO writes its deployable best under `checkpoints/best/` |
| `checkpoints/snapshot.pt` | mid-epoch resume state, removed at each clean epoch boundary |

There is no per-epoch checkpoint archive. At end of run, BC pushes
`best_<run_id>.pth` + the final resume checkpoint + `bc_summary.json` to the
NAS under `bc_checkpoints/<run_id>/` (connection from `QNN_NAS_*` env vars,
defaults in `corpus/nas.py`; failure is non-fatal). Pre-rename run dirs keep
their legacy names (`bc_training_checkpoint.pt` / `bc_best_model.pth`) —
loaders discover both.

## Ownership

| File | Owns |
|------|------|
| `machine.json` | device, asset root, binaries, worker geometry, batch sizing |
| `scenario.json` | map surface, native args, match options, scenario ladder (PPO/Optuna) |
| `model.json` | policy architecture |
| `train.json` | trainer and eval knobs for the run mode |
| `reward.json` | reward weights (PPO/Optuna) |
| `run.json` | run metadata, mode, resume behavior, checkpoint path, output roots |

## run.json

| Path | Purpose |
|------|---------|
| `name` | run name (human label — never an identity key) |
| `run_id` | immutable run identity, `YYYYMMDD-xxxxxx` (qnn.utils.artifacts); stamped into checkpoint names/meta, bc_summary, eval summaries, the trainer ledger, and ONNX metadata |
| `parent_run_id` | lineage edge for derived runs (empty otherwise) |
| `mode` | `bc`, `ppo`, `eval`, or `optuna` |
| `runtime_scale` | metadata label, usually `live` or `verify` |
| `resume` | `true` resumes this run’s own outputs when they exist; `false` archives them and starts fresh |
| `description` | human-readable run note |
| `checkpoint_path` | BC seed for `ppo` / `optuna`, or explicit checkpoint for `eval` |
| `trial_count` | Optuna trial count template/default for `mode: "optuna"` |
| `trial_budget_steps` | env-step budget per Optuna trial |
| `study_name` | Optuna study name |
| `storage` | Optuna storage URL; required for resumable studies |
| `config.*` | frozen config file paths under `config/` |
| `output.*` | output roots relative to the run dir |

## machine.json

Common:

| Path | Purpose |
|------|---------|
| `device` | requested device, usually `gpu` or `cpu`; `gpu` is the portable accelerator alias and may resolve to PyTorch `cuda` on CUDA/ROCm hosts |
| `asset_root` | Quake asset root |
| `worker_binary` | native worker binary |

BC:

| Path | Purpose |
|------|---------|
| `bc_data_dir` | precomputed BC data root (`artifacts/collect/qwd` by default) |
| `batch_size` | BC batch size — per-step frame count (frame-shuffled / non-recurrent) or parallel-lane count (lane-packed / recurrent) |
| `pin_memory` | pin host tensors for GPU transfer |
| `prefetch` | batch prefetch toggle |
| `snapshot_interval` | mid-epoch snapshot save interval (`snapshot.pt`, deterministic in-epoch resume) |

PPO / Optuna trials:

| Path | Purpose |
|------|---------|
| `num_lanes` | native engine subprocess count (`768` throughput default) |
| `minibatch_lanes` | recurrent PPO minibatch width (`768` default); rows/update = `minibatch_lanes × rollout_steps` |
| `collect_device` | action-inference device (`cpu` for the current small policy) |
| `collector_num_threads` | CPU intra-op threads used by the collector (`16` for the promoted process topology; use an explicitly measured value for narrower batches) |
| `env_backend` | `process` (promoted default) or optional grouped `arena_grid` |
| `matches_per_server` | grouped-arena 1v1 matches per world (`1..8`) |
| `seat_mode` | grouped-arena opponents: `bot` or shared-current-policy `self_play` |
| `arena_server_binary` / `arena_client_binary` | grouped-arena engine executables |
| `arena_map_id` / `arena_base_port` / `arena_bot_skill` | grouped-arena map, first server port, and FrikBot skill |
| `eval_num_envs` | post-train eval env count |
| `eval_num_episodes` | post-train eval episode count |

`arena_grid` is an optional dense-match backend; `process` remains the promoted
throughput default. Build its reproducible artifacts inside the trainer
container before launch:

```bash
python scripts/install_training_gamedir.py --asset-root assets
src/engine/build/build_ppo_arena_server.sh
src/engine/build/build_ppo_arena_client.sh
python scripts/build_arena_grid.py
```

Bot mode exposes one learner trajectory per match. `self_play` exposes both
seats to the same current policy; frozen-opponent routing is not implemented.

Eval:

| Path | Purpose |
|------|---------|
| `num_envs` or `eval_num_envs` | eval env count |
| `num_episodes` or `eval_num_episodes` | eval episode count |

## scenario.json

Present for PPO / Optuna runs only. BC runs do not carry a scenario.

| Path | Purpose |
|------|---------|
| `map_id` | base map id |
| `note` | optional run-plan note |
| `native_args` | Quake CLI args |
| `options` | base server options |
| `procgen` | procgen config when used |
| `scenarios` | scenario ladder; when present, the whole file is the scenario source |

## model.json

Common:

| Path | Purpose |
|------|---------|
| `encoder_hidden` | encoder width |
| `gru_hidden` | GRU hidden width |
| `n_heads` | transformer head count |
| `n_layers` | transformer depth |
| `ffn_dim` | transformer FFN width |
| `d_model` | transformer token width |
| `attn_dropout` | transformer attention dropout |
| `use_gru` | recurrent toggle |

BC-only (PPO model templates carry only the common keys):

| Path | Purpose |
|------|---------|
| `use_weapon_head` | enable the 8-class weapon head |
| `weapon_use_gru` | feed GRU output into weapon-head features |
| `d_target` | TargetPointer MLP hidden width (per-entity scoring head) |
| `head_bottleneck_dim` | per-head bottleneck width (0 = no bottleneck) |
| `head_use_relu` | apply ReLU inside the head bottleneck |

## train.json

BC:

| Path | Purpose |
|------|---------|
| `fixed_tick_hz` | expected cache tick rate (matches collection) |
| `sequence_length` | BC chunk length (0 = full episode) |
| `lr`, `lr_min`, `epochs`, `seed` | BC optimization schedule (flat LR by default) |
| `dtype` | training precision (`bf16` is the production default) |
| `head_loss_weights` | per-head loss weighting; weight 0 zeroes that head's gradient |
| `jump_pos_weight`, `jump_pos_weight_end` | per-axis positive weighting for the move-ud (jump) class, optionally linearly decayed |
| `warmup_epochs` | linear LR warmup before flat or cosine schedule |
| `max_grad_norm`, `tbptt_limit` | stability controls |
| `regression_threshold`, `regression_patience` | regression-style early stop |
| `train_eval_interval`, `train_eval_gap_threshold`, `train_eval_val_regression_threshold`, `train_eval_train_improve_threshold` | train/val proxy gap triggers |
| `segment_mask`, `token_mask` | filter-DSL expressions (see `qnn/filter_dsl.py`) for per-frame segment filtering and per-token entity filtering |
| `step_report_interval_seconds` | per-step metrics log cadence |
| `prometheus_pushgateway_url` | optional metrics export target |

PPO / Optuna trials:

| Path | Purpose |
|------|---------|
| `mode` | env mode |
| `native_workdir` | worker working directory |
| `fixed_tick_hz`, `max_steps_per_episode`, `seed` | run timing and seed |
| `rollout_steps`, `total_env_steps` | PPO horizon and total training budget |
| `policy_lr`, `ppo_epochs`, `clip_ratio` | optimizer controls |
| `trainable` | `full` (default) trains the transformer, GRU, pointer, and enabled heads; `heads` freezes the trunk |
| `learner_dtype`, `collector_dtype` | independent numeric precision; both default to `bf16` on the current APU |
| `rl_head_weights` | per-head PPO loss weighting; weight 0 removes that head's log-prob, entropy, KL, and gradient |
| `entropy_coef`, `rl_temperatures`, `sample_temperatures` | per-head exploration and distribution temperatures |
| `gamma`, `gae_lambda`, `max_grad_norm`, `value_coef` | PPO objective weights |
| `compile_learner`, `compile_collector` | independent compile switches; the measured default is eager learner plus compiled fixed-width collector |
| `pipeline_depth` | `1` for synchronous parity mode or `2` for the bounded one-update-lag pipeline |
| `pipeline_host_staging` | collect into CPU buffers and bulk-copy complete unrolls to the reusable GPU learner buffer |
| `eval_seed` | post-train eval seed |
| `eval_policy_modes`, `eval_start_mode` | post-train eval scheduling |
| `eval_holdout_seed_offset`, `eval_sample_seed_offset` | eval RNG offsets |
| `eval_map_features_path` | eval map features path |
| `eval_parallel_policy_modes` | eval execution controls |

Eval:

| Path | Purpose |
|------|---------|
| `mode` | eval env mode |
| `native_workdir` | worker working directory |
| `fixed_tick_hz`, `max_steps_per_episode`, `seed` | eval timing and seed |
| `policy_modes`, `start_mode` or `eval_policy_modes`, `eval_start_mode` | policy scheduling |
| `holdout_seed_offset`, `sample_seed_offset` or `eval_*` variants | RNG offsets |
| `map_features_path` or `eval_map_features_path` | eval map features path |
| `parallel_policy_modes` or `eval_parallel_policy_modes` | eval execution controls |

## PPO Topology

One native PPO trainer owns all engine subprocesses, one CPU collection replica,
and one GPU learner. The retained depth-two scheduler collects immutable
behavior generation `k` while it learns generation `k-1`. It permits exactly
one update of policy lag; there is no free-running actor queue or mid-unroll
weight refresh. `pipeline_depth=1` remains the synchronous parity path.

| Setting | Current Value | Rationale |
|---|---:|---|
| `num_lanes` | `768` | confirmed pipeline knee; 832, 896, and 1,024 lanes regress |
| `minibatch_lanes` | `768` | one full recurrent lane batch per PPO epoch |
| `collect_device` | `cpu` | avoids ROCm launch overhead for the small action model |
| `learner_dtype` / `collector_dtype` | `bf16` / `bf16` | measured win on both GPU learning and CPU collection |
| `pipeline_depth` | `2` | overlaps complete immutable collection and learner windows |
| `pipeline_host_staging` | `true` | avoids concurrent GPU rollout writes during learning |

The fixed-width CPU action model compiles as one `dynamic=False` graph.
Variable-width truncation bootstraps bypass that graph through eager
`model.forward`; compiling those widths causes late graph storms. Trainer
containers persist Inductor and MIOpen caches across runs. The full learner is
deliberately eager: its compile probe produced no first update after more than
eleven minutes.

Pipeline metrics distinguish `fill`, `steady`, and `drain` phases. Compare
throughput with aggregate steady frames divided by total `pipeline_cycle_s`;
do not include fill/drain FPS or rank candidates by a single-window peak.

## Derived Values

- PPO-family `num_envs = num_lanes`
- Runtime plans are copied from explicit frozen values; there is no auto-sizing

## Removed Hidden Defaults

- No nested `bc`, `ppo`, `eval`, or `scale` sections
- No standalone `check` mode
- No hardware auto-sizing
- No implicit PPO resume outside `run.json.resume`
- No alternate asset-root or demo-dir search order
- No hidden reward or procgen defaults in the promoted run-dir flow
