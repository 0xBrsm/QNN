# Corpus Encounter Statistics

Measured encounter-duration and re-engagement structure of the `qwd` NPY
corpus, and what they imply for the temporal component's effective horizon
(GRU hidden state / TCN receptive field). Regenerate with
`scripts/corpus_encounter_analysis.py` whenever the corpus is recollected —
these numbers drift with the collect.

Measured on `artifacts/collect/qwd` (train: 50,679 episodes / 34.7M rows /
482 h @ 20 Hz; val: 5,842 episodes / 3.9M rows / 54 h). Definitions:

- The corpus stores the **full stream** — ~74% of frames are no-target
  (`argmax(target_probs) == 0`); only ~26% are engaged.
- An **encounter** (sub-encounter) is a contiguous same-pid engaged run,
  split at recency boundaries (label persisting through a full ~2.0 s
  occlusion starts a fresh encounter on re-sight).
- Targets are stored as `target_probs` `(T, 17)`: slot 0 is no-target, slot
  `j` maps to entity index `j-1`. `entity_recency` is in seconds (0 =
  visible this frame, ~2.0 = SIGHT ceiling).

## Encounter Duration

Per-encounter duration, in 20 Hz ticks (seconds in parentheses). Train and
val agree closely.

| Split | median | p90 | p95 | p99 | max |
|-------|--------|-----|-----|-----|-----|
| train | 42 (2.1 s) | 96 (4.8 s) | 120 (6.0 s) | 181 (9.0 s) | 3754 (187.7 s) |
| val | 43 (2.2 s) | 98 (4.9 s) | 124 (6.2 s) | 189 (9.4 s) | 2630 (131.5 s) |

Mean ≈ 51 ticks (2.5 s), ~3.3 encounters per episode, and 33.8% of episodes
involve more than one enemy.

Key shape: the **bulk is short** (median ~42 ticks) but the **tail is heavy
and effectively unbounded** — p99 is 181 ticks and the max runs to 3754
ticks (188 s of continuous single-enemy engagement).

## Re-engagement Structure

Two re-engagement patterns, measured as the gap between consecutive same-pid
runs:

| Pattern | Meaning | Count (train) | Gap median | Gap p95 |
|---------|---------|---------------|------------|---------|
| A→A | Same enemy re-sighted after a brief occlusion (label persisted) | 288 | — (recency ≈ ceiling) | — |
| A→B→A | Same enemy after fighting a different one in between | 103,288 | 147 ticks (7.4 s) | 1310 ticks (65.5 s) |

A→A is **rare**; A→B→A **dominates**. Enemies recur constantly but across
long gaps (median 7.4 s, p95 65 s), so cross-encounter identity memory would
require horizons far beyond any plausible (or human-relevant) temporal
window.

## Implications for Temporal Horizon

Window size and encounter coverage are **independent**. Bounding the
prediction horizon does not drop long encounters from training: a 188 s
encounter still contributes ~3754 labeled frames regardless of window: each
frame is predicted from its own recent context. The window only bounds how
far back a single prediction looks.

- **Size to the dominant mass, not the tail.** A receptive field of ~120–128
  ticks (p95) covers 95% of encounters end-to-end. The rare long encounters
  harmlessly predict from their most recent ~6 s — which is all the signal
  supports, since the useful temporal signal is short-horizon (move momentum,
  weapon cooldown, look momentum).
- **Do not size to `max`.** The tail is unbounded-ish (p99 181, max 3754), so
  "never truncate" is infeasible for a fixed-window model and undesirable for
  a recurrent one — 188 s of conditioning is not wanted.
- **GRU vs TCN converge here.** At RF ≈ 128 a separable TCN
  (`qnn/model/temporal_tcn.py`) covers p95 fully with non-decaying access;
  the GRU reaches a similar effective horizon via gate decay. The heavy tail
  is not a reason to prefer either — choose on parallel-training vs.
  zero-plumbing-change, not on coverage.
- **Do not bridge A→B→A gaps.** Reset recurrent state at the
  encounter/episode boundary. No architecture should carry "I fought this
  enemy before" across a 7–65 s gap; humans do not either.

## Temporal Architecture Probe (GRU vs TCN)

The sizing conclusions above are testable via a bench probe that holds the
entire canonical network fixed and swaps **only** the temporal slot. See the
[Bench Skill](../../agents/skills/bench/SKILL.md) for the run-dir flow.

| Component | Role |
|-----------|------|
| `qnn/model/temporal_tcn.py` `SeparableTCN` | Causal dilated depthwise-separable stack; same `TemporalInput`/`TemporalOutput` contract as the GRU `Temporal`. Default RF 95 (~96 frames), output width `d_gru`. |
| `qnn/model/bench/temporal_probe.py` (`head="temporal"`) | Full canonical network; `variant: gru\|tcn` selects the slot. All heads, pointer, and MLP widths identical across variants. |
| `temporal_probe_gru.json` / `temporal_probe_tcn.json` | Paired probe configs. |

At `C=64` the separable TCN is **26,496 params** — parity with the `d_gru=64`
GRU (~24,960), so the comparison is not confounded by capacity. A dense (non-
separable) stack at the same RF would be ~74k and is intentionally not the
parity config.

Launch the pair (then set `head_loss_weights` with all heads active and
`tbptt_limit` >= typical episode length in each `config/train.json`):

```bash
python -m qnn.run.init --name head_probe_temporal_gru_seed17 --mode head_probe \
  --probe src/qnn/model/bench/temporal_probe_gru.json \
  --description "temporal parity: canonical GRU baseline" --resume false
python -m qnn.run.init --name head_probe_temporal_tcn_seed17 --mode head_probe \
  --probe src/qnn/model/bench/temporal_probe_tcn.json \
  --description "temporal parity: separable-TCN RF95" --resume false
```

Compare on `look_dll` (the most temporal-sensitive head) plus the composite
selection score.

**Cold-start caveat.** The BC loop carries cross-chunk state in a fixed
`(n_lanes, d_gru)` buffer, too small for a TCN's `RF-1` raw-frame tail. So the
TCN does not carry state across TBPTT chunk boundaries — each chunk is
zero-left-padded (a cold start); it is exact within a chunk. Setting
`tbptt_limit` at least the typical episode length keeps chunk boundaries from
falling mid-episode, making the cold start negligible. True stateful-TCN carry
would require a per-temporal `carry_dim` in canonical BC, deliberately out of
bench scope.

## Regenerating

```bash
# Both splits; default --data is artifacts/collect/qwd_new
python scripts/corpus_encounter_analysis.py --data artifacts/collect/qwd --split both
```

The script reports encounter-duration percentiles, A→A re-engagement recency
at the boundary frame, and A→B→A gap-length percentiles with a cumulative
threshold table. It reads via `mmap` and needs only NumPy (no GPU).
