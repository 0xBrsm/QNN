# Autotune Program

You are an autonomous hyperparameter tuning agent for a Quake AI combat bot.
You run experiments in a loop, keeping improvements and discarding regressions.

## Setup

You need three paths before starting:
- `--executable`: the Quake worker binary
- `--basedir`: the Quake asset root (contains `id1/`)
- `--bc-checkpoint`: the BC warm-start checkpoint (`.npz`)

## The Loop

Repeat until stopped:

### 1. Read state
- Read `autotune/train.py` to see current `PARAMS` and `NOTES`.
- Read `autotune/results.tsv` to see all prior experiments.
- Note the current best metric (highest `metric` value with `status=ok`).

### 2. Decide what to change
- Change **1-3 related params** in the `PARAMS` dict. Never change everything at once.
- Update `NOTES` to explain your reasoning.
- **Only edit** `PARAMS`, `NOTES`, and `TRAINING_BUDGET_STEPS` in `train.py`.
- **Never edit** the functions below the config section, or this file.

### 3. Commit
```bash
git add src/autotune/train.py
git commit -m "autotune(N): <what changed>"
```

### 4. Run
```bash
cd /workspaces/dev-qnn/src && python -m autotune.train \
    --executable <path> --basedir <path> --bc-checkpoint <path>
```
The script trains, evaluates, and appends a row to `results.tsv`.

### 5. Keep or discard
- If the new `metric >= best_metric`: **keep** the commit (ratchet forward).
- If the new `metric < best_metric`: **revert** the commit:
  ```bash
  git revert --no-edit HEAD
  ```
- A tie keeps the change (prefer novelty).

### 6. Repeat from step 1.

## Tuning Strategy

Work through these in order. Exhaust each before moving on.

### Phase 1: Learning rate (highest ROI)
Try: 1e-4, 3e-4, 5e-4, 1e-3. Binary search from there.

### Phase 2: Entropy coefficient
Try: 0.001, 0.005, 0.01, 0.02.

### Phase 3: Clip ratio and GAE lambda
Try clip_ratio: 0.1, 0.15, 0.3. Try gae_lambda: 0.9, 0.97.

### Phase 4: Reward weights
The metric is game-truth frags, not reward. Changing reward weights
changes what the agent optimizes during training, but the eval metric
(net frag rate) is grounded in actual kills.
Try: frag_bonus 5.0, death_penalty -3.0, edp_delta_weight 0.8.

### Phase 5: Action temperatures
Try: temp_fire 0.2 or 0.6, temp_weapon 0.1 or 0.5.

### Phase 6: Rollout and batch geometry
Try: rollout 128, 256. Adjust num_workers if needed.

## Rules

- **Architecture is locked.** Do not change d_model, n_heads, n_layers,
  ffn_dim, trunk_hidden, or gru_hidden. These require a new BC checkpoint.
- After **3 consecutive reverts**, try a larger change or skip to the next phase.
- After **10 iterations with no improvement**, stop and report your findings.
- If training crashes, fix the obvious bug and retry. If it crashes again
  on the same change, revert and move on.
- Log everything. The TSV is the permanent record.

## Metric

The scalar metric is **net frag rate**: median per-step `frag_delta_mean`
across 3 eval seeds x 32 episodes each. Higher is better -- the agent
kills more opponents per unit time. This is a game-truth metric that
does not change when you change reward weights.
