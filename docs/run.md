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
| `bc_data_dir` | precomputed BC data root (`artifacts/collect/qwd` by default) |
| `batch_size` | BC batch size |
| `microbatch_size` | gradient-accumulation microbatch size |
| `pin_memory` | pin host tensors for GPU transfer |
| `prefetch` | batch prefetch toggle |
| `snapshot_interval` | epochs between archived checkpoints |

PPO / PBT / Optuna trials:

| Path | Purpose |
|------|---------|
| `num_workers` | PPO rollout worker count |
| `num_envs_per_worker` | env count per worker |
| `worker_num_splits` | Sample Factory worker splits |
| `policy_workers_per_policy` | centralized inference workers per policy |
| `batched_sampling` | Sample Factory batched sampling toggle |
| `worker_inference` | per-worker inference toggle (see [PPO worker inference](#ppo-worker-inference)) |
| `worker_inference_device` | `cpu` (default) or `gpu` — device each rollout worker runs inference on when `worker_inference=true` |
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

Common:

| Path | Purpose |
|------|---------|
| `trunk_hidden` | trunk width |
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
| `weapon_context_from_obs` | use observed (vs predicted) held weapon when building motor-head context |
| `weapon_switch_confidence` | minimum weapon-head softmax to emit a switch at inference |
| `weapon_switch_margin` | minimum margin over currently-held weapon to emit a switch |
| `gru_target_query` | route GRU output into the TargetPointer query (otherwise self_readout) |
| `hard_target_feat` | hard-argmax target pooling instead of softmax |
| `weapon_in_target_query` | add a current-weapon embedding to the target query |
| `linear_slot_prior` | additive linear slot-index prior on target logits |
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
| `segment_mask`, `token_mask` | filter-DSL expressions (see `qnn/filter_dsl.py`) for episode and per-token filtering |
| `step_report_interval_seconds` | per-step metrics log cadence |
| `prometheus_pushgateway_url` | optional metrics export target |

PPO / PBT / Optuna trials:

| Path | Purpose |
|------|---------|
| `mode` | env mode |
| `native_workdir` | worker working directory |
| `fixed_tick_hz`, `max_steps_per_episode`, `seed` | run timing and seed |
| `rollout_steps`, `total_steps` | PPO horizon |
| `policy_lr`, `ppo_epochs`, `clip_ratio` | optimizer controls |
| `entropy_coef`, `bc_kl_coef`, `initial_stddev` | exploration, KL shaping, continuous action sampling width |
| `head_loss_weights` | per-head PPO loss weighting (same JSON schema as BC); weight 0 zeros that head's log-prob/entropy/KL so no gradient flows to its parameters |
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

## PPO worker inference

Sample Factory's default rollout architecture sends every env's observation
to a central inference worker that batches and forwards on GPU. For this
project's small (~50k-param) policy on ROCm/WSL, the coordination tax of
gathering 128 envs into one GPU batch dominates the GPU compute, and the
`worker_inference=true` path (local inference inside each rollout worker)
is substantially faster:

| Config | Aggregate FPS | vs baseline |
|---|---|---|
| Central GPU inference (default) | ~1,900 | 1.0× |
| `worker_inference=true`, GPU | ~2,800 | 1.5× |
| `worker_inference=true`, `worker_inference_device=cpu` | ~4,570 | 2.4× |

The CPU path wins because 32 rollout-worker processes get real per-core
parallelism, while 32 processes all hitting one ROCm device serialize
their kernel launches. The win flips back toward GPU once the policy
grows (rebenchmark above a few hundred thousand params).

With `worker_inference=true`, per-worker policy lag variance increases;
`_record_summaries` in SF's learner is monkey-patched to guard against
an empty minibatch.

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
