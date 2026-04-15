# Training Config Matrix

This is the source-of-truth reference for the run-dir training schema.

Templates live in each pipeline's package (`qnn/bc/templates/` or
`qnn/ppo/templates/`). `qnn.run.init` copies them into
`runs/<mode>/<name>/config/`, and training reads only the frozen files
inside that run directory.

## Rules

- `run.json` is the single entry point.
- Every config path in `run.json.config` is required — and no more.
- BC runs only declare `train`, `machine`, and `model`. PPO / PBT / Optuna add
  `scenario`, `reward`, and `eval`.
- `run.json.resume` is the only resume switch.
- No code-level defaults fill in missing training keys.
- Public run modes are `bc`, `ppo`, `pbt`, `eval`, and `optuna`.
- `run.json.output.checkpoints` is the real PPO checkpoint root. Sample Factory
  writes only its internal `checkpoint_p*` subdirs under it.
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

PPO / PBT / Optuna:

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
    eval.json
  checkpoints/
  metrics/
  logs/
```

## Ownership

| File | Owns |
|------|------|
| `machine.json` | device, asset root, binaries, worker geometry, batch sizing |
| `scenario.json` | map surface, native args, match options, scenario ladder (PPO/PBT/Optuna) |
| `model.json` | policy architecture |
| `train.json` | trainer and eval knobs for the run mode |
| `reward.json` | reward weights (PPO/PBT/Optuna) |
| `eval.json` | post-train eval pool, seeds, policy modes (PPO/PBT/Optuna) |
| `run.json` | run metadata, mode, resume behavior, checkpoint path, output roots |

## run.json

| Path | Purpose |
|------|---------|
| `name` | run name |
| `mode` | `bc`, `ppo`, `pbt`, `eval`, or `optuna` |
| `runtime_scale` | metadata label, usually `live` or `verify` |
| `resume` | `true` resumes this run’s own outputs when they exist; `false` archives them and starts fresh |
| `description` | human-readable run note |
| `checkpoint_path` | BC seed for `ppo` / `pbt` / `optuna`, or explicit checkpoint for `eval` |
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
| `device` | requested device, usually `gpu` or `cpu` |
| `asset_root` | Quake asset root |
| `worker_binary` | native worker binary |

BC:

| Path | Purpose |
|------|---------|
| `bc_data_dir` | precomputed BC data root (`assets/collect/prod` by default) |
| `batch_size` | BC batch size |
| `pin_memory` | pin host tensors for GPU transfer |
| `prefetch` | batch prefetch toggle |

PPO / PBT / Optuna trials:

| Path | Purpose |
|------|---------|
| `num_workers` | PPO rollout worker count |
| `num_envs_per_worker` | env count per worker |
| `worker_num_splits` | Sample Factory worker splits |
| `policy_workers_per_policy` | centralized inference workers per policy |
| `batched_sampling` | Sample Factory batched sampling toggle |
| `worker_inference` | per-worker inference toggle |
| `minibatch_size` | PPO minibatch size |
| `eval_num_envs` | post-train eval env count |
| `eval_num_episodes` | post-train eval episode count |

Eval:

| Path | Purpose |
|------|---------|
| `num_envs` or `eval_num_envs` | eval env count |
| `num_episodes` or `eval_num_episodes` | eval episode count |

## scenario.json

Present for PPO / PBT / Optuna runs only. BC runs do not carry a scenario.

| Path | Purpose |
|------|---------|
| `map_id` | base map id |
| `note` | optional run-plan note |
| `native_args` | Quake CLI args |
| `options` | base server options |
| `procgen` | procgen config when used |
| `scenarios` | scenario ladder; when present, the whole file is the scenario source |

## model.json

| Path | Purpose |
|------|---------|
| `trunk_hidden` | trunk width |
| `gru_hidden` | GRU hidden width |
| `n_heads` | transformer head count |
| `n_layers` | transformer depth |
| `ffn_dim` | transformer FFN width |
| `d_model` | transformer token width |
| `readout` | transformer readout mode |
| `action_history_tokens` | action-history token count |
| `attn_dropout` | transformer attention dropout |
| `use_gru` | recurrent toggle |

## train.json

BC:

| Path | Purpose |
|------|---------|
| `fixed_tick_hz` | expected cache tick rate (matches collection) |
| `sequence_length` | BC chunk length (0 = full episode) |
| `lr`, `lr_min`, `epochs`, `patience`, `seed` | BC optimization schedule |
| `head_loss_weights`, `focal_gamma`, `sparse_discrete` | BC loss shaping |
| `look_deadzone`, `look_turn_alpha`, `look_cosine` | look head loss and label controls |
| `warmup_epochs` | linear LR warmup before cosine decay |
| `class_weight_power`, `class_weight_min`, `class_weight_max` | BC class weighting |
| `max_grad_norm`, `tbptt_limit` | stability controls |
| `regression_stop`, `regression_threshold`, `regression_patience` | regression-based early stop |
| `train_eval_interval`, `train_eval_gap_threshold`, `train_eval_val_regression_threshold`, `train_eval_train_improve_threshold` | train/val proxy gap triggers |
| `prometheus_pushgateway_url` | optional metrics export target |

PPO / PBT / Optuna trials:

| Path | Purpose |
|------|---------|
| `mode` | env mode |
| `native_workdir` | worker working directory |
| `fixed_tick_hz`, `max_steps_per_episode`, `seed` | run timing and seed |
| `rollout_steps`, `total_steps` | PPO horizon |
| `policy_lr`, `ppo_epochs`, `clip_ratio` | optimizer controls |
| `entropy_coef`, `bc_kl_coef` | exploration and KL shaping |
| `gamma`, `gae_lambda`, `max_grad_norm`, `value_coef` | PPO objective weights |
| `with_pbt`, `num_policies`, `pbt_*` | PBT controls |
| `eval_seed` | post-train eval seed |
| `eval_policy_modes`, `eval_start_mode` | post-train eval scheduling |
| `eval_holdout_seed_offset`, `eval_sample_seed_offset` | eval RNG offsets |
| `eval_map_features_path` | eval map features path |
| `eval_record_demos`, `eval_parallel_policy_modes` | eval execution controls |
| `demo_policy_mode` | single-episode observation/demo mode |

Eval:

| Path | Purpose |
|------|---------|
| `mode` | eval env mode |
| `native_workdir` | worker working directory |
| `fixed_tick_hz`, `max_steps_per_episode`, `seed` | eval timing and seed |
| `policy_modes`, `start_mode` or `eval_policy_modes`, `eval_start_mode` | policy scheduling |
| `holdout_seed_offset`, `sample_seed_offset` or `eval_*` variants | RNG offsets |
| `map_features_path` or `eval_map_features_path` | eval map features path |
| `record_demos`, `parallel_policy_modes` or `eval_*` variants | eval execution controls |

## eval.json

PPO / PBT / Optuna post-train eval pool.

| Path | Purpose |
|------|---------|
| `seed_pool` | seeds available to eval |
| `num_seeds` | seeds drawn per run |
| `episodes_per_seed` | episodes per drawn seed |
| `policy_mode` | `greedy` or `sample` |
| `start_mode` | `holdout` or `randomized` |
| `record_demos` | write demo files during eval |
| `metric` | selection statistic across episodes (`median`, etc.) |

## Derived Values

- PPO-family `num_envs = num_workers * num_envs_per_worker`
- Runtime plans are copied from explicit frozen values; there is no auto-sizing

## Removed Hidden Defaults

- No nested `bc`, `ppo`, `eval`, or `scale` sections
- No standalone `check` mode
- No hardware auto-sizing
- No implicit PPO resume outside `run.json.resume`
- No alternate asset-root or demo-dir search order
- No hidden reward or procgen defaults in the promoted run-dir flow
