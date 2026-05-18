# Input Inference

How we recover player-intent labels (button presses, movement, aim,
weapon switches) from demo recordings for BC training.

## Two pipelines, completely separate code paths

**QWD pipeline (direct).** Each frame, read the player's actual
usercmd byte. The intent is observable — no inference, no ping math,
no back-shift. Labels = ground truth. QWD is our validation reference
and the source of truth for what real labels look like.

**MVD pipeline (inference).** No usercmds in the recording — only
server-side state changes. Inference recovers press time by
back-shifting from each observed state change using observable
signals (per-player ping from `svc_updateping`, server tick rate
from inter-record msec deltas). Labels match QWD ground truth in
expectation; validated on paired demos.

Both pipelines emit the same label format. The model trains on the
union of both. See [agents/plans/mvd-reconstruction.md](../../agents/plans/mvd-reconstruction.md)
for the broader plan and current implementation status.

This document covers the per-action signal recovery. Section 1 (fire)
is mostly complete; sections on switch / move / look will fill in as
the MVD pipeline is built out.

## 1. Fire (button 0)

### Problem

BC training labels for "did the player press fire at tick T?" come from
two sources:

- **QWD path**: read `cmd->buttons & 1` directly from the recorded
  `usercmd_t` (ground truth — the player's actual button bit).
- **MVD path**: no usercmd available; the C-rule in
  `QNN_DetectFireEvent` infers a shot from server signals (weapon sound
  + ammo decrement). The label lands on the frame those signals arrive
  at the recording client, **not** when the player pressed fire.

Since MVD covers the majority of the corpus (no usercmd ever recorded
for those formats), the MVD labels must approximate button-press timing
or BC will systematically learn to fire late by one network RTT.

### Empirical baseline

Paired collect at 20 Hz on 133 duel demos (`fire_lat2`, May 2026) — same
QWD demos run through both paths to produce truth and candidate labels:

- 31,895 truth rising edges (`cmd->buttons & 1`)
- 59,641 candidate rising edges (C-rule on server signals)
- Greedy symmetric ±W matching at the rising-edge level

Of matched pairs (truth, candidate):

| offset (frames) | count | pct |
|-----------------|-------|-----|
| −3              |    31 | 0.10% |
| −2              |   197 | 0.62% |
| −1              |    73 | 0.23% |
| 0               | 11858 | 37.43% |
| +1              | 14726 | 46.48% |
| +2              | 3166  | 9.99% |
| +3              | 994   | 3.14% |
| +4              | 77    | 0.24% |

Mean offset = +0.836 frames, std 1.164. 61.5% of candidates land
**after** the truth press (correct lag direction); only 1.1% land
before (20Hz quantization noise). At W=4 forward (±200 ms) the
event-recall hits 99.08% on cooldown-filtered truth.

### Cooldown filtering

Truth presses during the previous shot's `attack_finished` window are
ignored by the engine — the press creates no shot, sound, or ammo
change, and so produces no candidate event. Counting these as FN is
unfair: there's no possible match.

Cooldown table, source-of-truth `vendor/quake/QW/progs/weapons.qc`
`attack_finished` values; native ticks at QW server rate 77 Hz, emit
frames at 20 Hz collect rate:

| weapon | QC `attack_finished` | native ticks @ 77Hz | 20 Hz frames |
|--------|----------------------|---------------------|--------------|
| Axe (3) | `time + 0.5` | 38 | 10 |
| SG  (4) | `time + 0.5` | 38 | 10 |
| SSG (5) | `time + 0.7` | 54 | 14 |
| NG  (6) | `time + 0.2` | 15 |  4 |
| SNG (7) | `time + 0.2` | 15 |  4 |
| GL  (8) | `time + 0.6` | 46 | 12 |
| RL  (9) | `time + 0.8` | 62 | 16 |
| LG  (10)| `time + 0.1` |  8 |  2 |

Truth edges within the prior candidate's cooldown window for the active
weapon are dropped. On `fire_lat2`, this removes 7,689 of 31,895 truth
edges (24%). The recall numbers above all use the filtered set.

### Latency components

The offset is a sum of:

- **Network RTT**: client → server → client, the dominant component.
  Recorded by the server as `svc_updateping` (opcode 36, 1-byte slot +
  2-byte ping ms). For the recorder slot, the classifier accumulates
  these into `avg_ping_ms` (excluding the 999 'unknown' sentinel and 0).
- **Server-frame quantization**: server runs at ~77 Hz on KTX, so the
  cmd → shot detection rounds to the next server tick (≤ 13 ms).
- **Sound mix / event dispatch jitter**: small, ≤ 5 ms.
- **20 Hz emit quantization**: ±25 ms uniform bin selection at the
  emit-rate boundary, std ≈ 14 ms.

Corpus-wide `avg_ping_ms` distribution (4628 of 4685 demos with data):

| stat | value |
|------|-------|
| p25 | 14 ms |
| median | 18 ms |
| p75 | 44 ms |
| p95 | 96 ms |
| max | 615 ms |

### Correction model

For each MVD fire event, shift the emit time backward by

```
shift_frames = round((avg_ping_ms + ε) / 50)
ε ~ N(0, 30 ms)
```

where `avg_ping_ms` is the per-demo scalar from the corpus manifest, ε
is an independent per-event Gaussian draw, and 50 ms is the 20 Hz emit
bin width.

The Gaussian noise injects realistic per-event jitter so the model
doesn't see a deterministic constant offset (which would not generalize
to live deployment where every shot has slightly different network
conditions). The 30 ms scale was selected by L1 fit against the
post-shift residual PMF; the L1 error sweep has a clean minimum at
σ=30:

| σ_ms | L1 fit error |
|------|--------------|
| 10   | 0.65 |
| 20   | 0.35 |
| 25   | 0.26 |
| **30** | **0.18** |
| 35   | 0.21 |
| 50   | 0.41 |
| 100  | 0.79 |

The 30 ms is also consistent with a physical-additive-noise bound: the
empirical residual variance is 0.74² ≈ 0.55 frames² after applying the
per-demo median shift; subtracting the 20 Hz quantization noise floor
(variance 1/12 ≈ 0.083 frames²) leaves 0.47 frames² ≈ (34 ms)² for "real"
jitter, in agreement with the L1 minimum.

### Validation

On `fire_lat2` (31,681 matched edges, symmetric ±16):

| method | mean offset | std | P(0) | P(\|≤1\|) |
|--------|-------------|-----|------|----------|
| ORIGINAL (no shift) | +0.836 | 1.164 | 0.374 | 0.841 |
| per-demo median shift | +0.290 | 1.179 | 0.565 | 0.916 |
| per-event most-recent ping shift | +0.282 | 1.195 | 0.563 | 0.915 |

Per-event tracking (looking up `svc_updateping` history at each fire
frame) gains <0.01 frames over the per-demo median, because ping is
stable within a demo. The manifest's `avg_ping_ms` scalar is sufficient
— no per-frame engine-side tracking is required.

### Implementation status

- `avg_ping_ms` is emitted by `qw_classifier` and lives in
  `artifacts/corpus/qwd_manifest.ndjson` (added May 2026).
- Engine-side shift in `engine/qw/qnn_collect_main.c` is **deferred**.
  When implemented, it should:
  - Read `avg_ping_ms` from the manifest at demo start (already passed
    to the worker via `--manifest`).
  - At each MVD fire detection in `QNN_InferNativeAction_MVD`, compute
    the shift formula above and write `fire=1` to the appropriate past
    emit slot (or shift the emit buffer at flush time — TBD design).
- Diagnostic scripts (`scripts/fire_latency_window_sweep.py`,
  `scripts/test_ping_methods.py`) validate the model against QWD ground
  truth on paired collects.

### What this does not address

- **Engine-skipped presses**: rapid taps during cooldown produce truth
  edges with no possible candidate. These are dropped via the cooldown
  filter when computing recall, but they remain real button presses
  that the BC model could in principle predict. We treat them as
  unlabeled noise rather than negative examples.
- **Frame-level (vs edge-level) precision**: the C-rule emits 1-frame
  pulses; QWD-truth emits sustained held-down windows. Frame-level
  precision is 99% (when candidate says fire, truth almost always
  agrees) but frame-level recall is 38% (candidate misses most of the
  hold). Edge-level matching avoids this asymmetry; that's the metric
  the correction is tuned for.

### Hold duration (per-weapon)

QWD-truth encodes a held trigger as `fire = 1` on every emit frame for
the duration of the hold. The MVD C-rule emits `fire = 1` only on the
single emit frame containing a detected shot event. To make MVD labels
match QWD-truth in shape, each MVD shot must be extended to a hold of
realistic per-weapon duration.

The fire head is a single binary sigmoid logit per frame (see
`FIRE_HEAD_SIZE = 1` in `qnn/model/policy.py`); the model decides
frame-by-frame whether to hold. We chose binary over a {press,
release, none} representation because the engine ultimately only needs
button state, dense supervision is easier to learn than sparse events,
and class imbalance gets worse with rare-event classes.

#### Per-weapon hold profile (production corpus, 2,536 episodes)

From `artifacts/collect/qwd` QWD-truth labels:

| weapon | active frames | fire=1 frames | fire% | n_holds | median hold |
|--------|---------------|---------------|-------|---------|-------------|
| LG  | 1,888,972 | 642,618 | 34.0% | 49,807 |  9 |
| RL  | 13,168,657 | 1,614,751 | 12.3% | 397,994 | 3 |
| SG  | 22,546,106 | 2,572,731 | 11.4% | 419,970 | 3 |
| GL  | 1,325,985 | 278,218 | 21.0% | 54,439 | 3 |
| SSG | 1,490,056 | 226,610 | 15.2% | 41,120 | 4 |
| SNG | 1,063,782 | 303,926 | 28.6% | 17,991 | 10 |
| Axe | 2,026,574 | 101,016 | 5.0%  | 34,270 | 2 |
| NG  | 447,126 | 111,998 | 25.1% | 5,959  | 8 |

Cooldown-respect: per shot, did the trigger stay held continuously
through the full engine cooldown window?

| weapon | held_full | released_in_cd | hold_ratio |
|--------|-----------|----------------|------------|
| LG  | 43,655 |     0 | 1.000 |
| NG  |  3,906 |    36 | 0.991 |
| SNG | 12,905 |    44 | 0.997 |
| Axe |  1,333 |   431 | 0.756 |
| SG  | 46,643 | 48,972 | 0.488 |
| SSG |  3,270 |  8,566 | 0.276 |
| GL  |  5,089 | 11,411 | 0.308 |
| RL  | 16,099 | 82,527 | 0.163 |

Two distinct classes emerge:

- **Continuous-fire weapons** (LG/NG/SNG, hold-ratio ≥ 0.99): cooldown
  is ≤ 4 emit frames, players keep the trigger held during cooldown.
- **Tap-fire weapons** (SG/SSG/GL/RL, hold-ratio ≤ 0.49): cooldown is
  ≥ 10 frames, players release between shots.
- **Axe** sits in between (0.756) — small sample of follow-up cases,
  but mechanically a discrete-pulse weapon (cooldown 10 ≫ typical hold
  of 2 frames).

#### MVD label-extension policy

Continuous weapons (LG/NG/SNG): hold-ratio is 0.993–1.000 across the
full 49k-demo QWD corpus — players nearly always hold the trigger
through the full cooldown window.  The simple approach works: on each
detected shot at frame T, extend fire=1 through T+cd-1.  When the next
shot arrives at T+cd the runs are adjacent, producing seamless held
fire across the burst.  No distribution fitting is required.

```
for each detected shot at frame T, weapon W in {LG, NG, SNG}:
    fire[T : T + cd[W]] = 1   # always fill the full cooldown window
```

The cd+ fraction (67–88%) and the near-perfect hold_ratio confirm this
is the right model.  The minority sub-cd holds (12–32%) are dominated
by end-of-engagement releases and tap-during-cooldown attempts that the
engine ignores; fitting a distribution to that noise would add
complexity for no gain.

Tap weapons (SG/SSG/GL/RL/Axe): MVD sees only the shot event, not the
button hold duration. The hold length is not meaningfully predictable
from observable state — we have no decomposition of variance into
player-identity, game-state, or motor-noise components — so a learned
labeler would recover at best the marginal distribution. A random draw
from the empirical marginal is equivalent and far simpler.

#### Full per-frame hold profile (sub-cooldown only, 49k-demo QWD corpus)

Measured by `scripts/fire_weapon_behavior_profile.py` on
`artifacts/collect/qwd`. Only sub-cooldown holds (hold < cd) are
shown per-frame; `cd+` bucket covers holds ≥ cd (player held through
the full engine cooldown — these become multi-shot runs and are
handled by shot-spacing in the MVD stream).

| weapon | cd |  n sub-cd | mean | f=1 | f=2 | f=3 | f=4 | f=5 | f=6 | f=7 | f=8 | f=9 | f=10 | f=11–cd-1 | cd+ |
|--------|----|-----------|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|-----------|-----|
| Axe | 10 | 27,879 | 2.0 | 30% | 46% | 16% | 3% | 1% | 0.4% | 0.3% | 0.2% | 0.1% | — | — | 4% |
| SG  | 10 | 304,753 | 3.1 | 19% | 29% | 15% | 16% | 10% | 5% | 2% | 1% | 0.7% | — | — | 11% |
| SSG | 14 | 34,301 | 3.8 | 15% | 19% | 18% | 16% | 13% | 7% | 5% | 3% | 2% | 1% | 0.6–0.3% | 7% |
| GL  | 12 | 36,005 | 3.2 | 19% | 21% | 21% | 19% | 10% | 4% | 2% | 1% | 0.7% | 0.6% | 0.4% | 10% |
| RL  | 16 | 289,738 | 3.4 | 16% | 20% | 22% | 21% | 11% | 4% | 2% | 1% | 0.7% | 0.5% | 0.4–0.1% | 3% |

#### Distribution fitting

Three families were fit to the sub-cooldown per-frame empirical PMF
via maximum likelihood (`scripts/fire_hold_fit.py`), using L1 error
as the acceptance criterion:

- **Geometric** (`p*(1-p)^(k-1)`, mode always at k=1): consistently
  the worst fit — underestimates the mode-at-2 shape for all weapons.
- **Poisson+1** (`Poisson(λ)+1`): decent bell shape but underestimates
  the frame-1 spike; best for RL but still 0.19 L1.
- **Truncated log-normal** (`round(exp(N(μ,σ)))` clamped to [1, cd-1]):
  best fit for all weapons except RL, where it trails Poisson+1 by 0.04.

Log-normal is adopted for all tap weapons — one family, five parameter
pairs, no per-weapon code branching.

| weapon | μ | σ | L1 (lognorm) | L1 (pois1) | L1 (geom) |
|--------|---|---|--------------|------------|-----------|
| Axe | 0.5992 | 0.4241 | **0.0605** | 0.2253 | 0.5157 |
| SG  | 0.9943 | 0.6484 | **0.1468** | 0.2870 | 0.3386 |
| SSG | 1.1702 | 0.6640 | **0.1626** | 0.3079 | 0.3233 |
| GL  | 1.0066 | 0.6201 | **0.1960** | 0.2085 | 0.3575 |
| RL  | 1.0557 | 0.5880 | 0.2236 | **0.1878** | 0.4299 |

RL is the one weapon where Poisson+1 wins (0.1878 vs 0.2236). The
gap is small and driven by the frame-1 spike (16% observed, 13%
lognorm, 9% pois1) — neither family handles it well. Log-normal is
kept for RL to avoid branching on a marginal improvement.

#### Final algorithm

```
for each MVD-detected shot at frame T, weapon W:
    if W in {LG, NG, SNG}:
        # Shot-spacing carries the hold signal
        link with prior same-weapon shot if within cooldown+slack
        emit fire=1 for the resulting linked run
    else:
        # SG, SSG, GL, RL, Axe — sample hold from truncated log-normal
        ext = clip(round(exp(randn() * σ[W] + μ[W])), 1, cd[W] - 1)
        fire[T : T + ext] = 1
```

Sampling is `numpy.random.default_rng(seed).lognormal(μ, σ)` rounded
and clamped to `[1, cd-1]`. The seed is fixed per collect run for
reproducibility. Applied as an offline relabeler
(`scripts/relabel_mvd_fire_hold.py`) that rewrites the `actions/fire`
NPY files in-place, following the same pattern as
`scripts/relabel_mvd_move.py`.

### Implementation status

- `avg_ping_ms` in manifest: **live**.
- Cooldown table validated against QC: **live in scripts**.
- Per-weapon hold profile: **measured, recorded above**.
- Engine-side ping shift: **live** (commit `c0d3266f`).
- Engine-side hold extension: **live** (commit `91caf04c`) —
  tap weapons sample from truncated log-normal; continuous weapons
  extend for full cooldown window. `scripts/relabel_mvd_fire_hold.py`
  provides an offline equivalent for existing NPY collects.

## 2. Jump (button 1)

### Problem

QWD-truth records `move_ud=2` (jump=up) on every emit frame the player
holds the jump key, exactly as for fire. The MVD path detects a jump
from the `player/plyrjmp8.wav` sound event (same per-event scheme as
fire) and back-shifts the press time by one network RTT via
`QNN_BackShiftWriteJumpEvents`, emitting a single-frame pulse per event.

Validation on the 133-demo fire_lat2 set (backshift + hold binary):

| metric | value |
|--------|-------|
| frame recall | 0.284 |
| frame precision | 0.559 |
| event recall ±2 | 0.767 |
| event precision ±2 | 0.372 |
| truth events | 21,106 |
| candidate events | 43,567 |

Two distinct issues:
- **Press-frame timing**: candidate emits 2× more events than truth.
  Precision of 0.372 means most candidate jump events have no matching
  truth event nearby — the detection is over-firing.
- **Hold duration**: once a true event is detected, it needs to be
  extended forward (same pattern as fire).

Press-frame accuracy must be fixed before hold extension is useful.

### Hold duration profile (49k-demo QWD corpus)

Measured by `scripts/jump_hold_profile.py` on `artifacts/collect/qwd`.

| stat | frames | ms @ 20Hz |
|------|--------|-----------|
| p25 | 3 | 150 ms |
| median | 4 | 200 ms |
| p75 | 6 | 300 ms |
| p90 | 9 | 450 ms |
| p99 | 18 | 900 ms |

Per-frame distribution (473,979 jump runs):

| f | % | f | % |
|---|---|---|---|
| 1 | 1.8% | 6 | 6.7% |
| 2 | 8.2% | 7 | 4.9% |
| 3 | **26.4%** | 8 | 4.0% |
| 4 | **24.7%** | 9 | 3.1% |
| 5 | 12.3% | 10+ | 8.4% |

Strong peak at 3–4 frames (51%) reflects bunny-hop tap behaviour:
players release the jump key quickly after pressing so they can
re-press on landing. The max=7910 outlier is an anomaly
(held-jump for an entire demo).

Distribution fit (`scripts/jump_hold_profile.py`):

| family | L1 |
|--------|----|
| geometric | 0.6492 |
| Poisson+1 | 0.4347 |
| **log-normal** | **0.2717** |

Log-normal best: μ=1.4501, σ=0.5012, cap=20 frames.
The 3–4 frame peak is not perfectly captured (log-normal underestimates
by ~6pp there, overestimates frame 2 by ~4pp), but it is the best
available smooth family. Practical cap: 20 frames (covers p99=18,
excludes multi-second held-jump anomalies).

### Implementation status

- Jump hold profile: **measured, recorded above**.
- Press-frame accuracy: **under investigation** — 2× overcounting in
  candidate events must be resolved before hold extension is useful.
- Engine-side hold extension: **deferred** pending press-frame fix.
