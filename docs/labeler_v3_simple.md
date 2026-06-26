# Labeler V3 Simple

This is the refined v3 target-label design for BC after reading the full
analysis set. It keeps the prior design's core pivot: replace hard per-frame
`argmax` target labels with a deterministic confidence distribution over
`NO_TARGET, slot_0, ..., slot_15`.

The main correction is that physics-hit evidence is no longer a small bump.
The hit-stream analysis says that when physics streams overlap, the cone
labeler still chooses one of the physics pids 98.4% of the time. That makes
physics a co-primary candidate source, not a tie-breaker and not a secondary
confidence nudge. The hybrid experiment also says exactly what not to do:
inject physics into a hard sticky argmax state machine. V3 uses physics to add
parallel probability mass and lets soft CE consume uncertainty instead of
forcing a one-hot switch.

## Evidence Readback

| Finding | Design consequence |
|---------|--------------------|
| `orig`/opt3: 844,723 labels, 3,109 switches, quality ratio 0.285 | Hard opt3 is the current baseline to beat. |
| v2 lead/range: 849,185 labels, 3,214 switches, quality ratio 0.282 | Keep lead correction; do not widen cones with sigma/K. The hard per-weapon range gate at admit time was dropped — see the Magic Constants table. |
| hit-anchored: 733,233 labels, 2,454 switches, quality ratio 0.169 | Physics-only loses too much cone-only engagement signal and is unfairly punished by the cone-biased angvel classifier. |
| hybrid: 844,693 labels, 3,788 switches, quality ratio 0.238 | Physics as hard tie-break/override adds flipping noise. Do not use hard physics tie-breaks. |
| Multi-pid physics overlap is only 2% of frames; cone matches one active physics pid 98.4% of overlap frames | Physics-hit streams are highly compatible with cone labels and should be co-primary candidates. |
| Cone-only/no-physics is 106k frames; half are recency > 0, and recency-0 runs include 22% with 2+ fires | Cone-only is not all noise. Keep sustained recency-0 cone-only engagements with lower confidence. |
| 16% of cone-only recency-0 runs end in demonstrator death; 35% for 31-100 frame runs | Demonstrator death is useful as a confidence penalty, not a hard rejection. |

## Output Contract

V3 emits:

```text
target_probs: float32[T, 17]
index 0: NO_TARGET
index 1..16: slot_0..slot_15
row sum: 1.0
```

Start dense. Add sparse top-k storage only if collection size becomes a real
problem.

For model training, `NO_TARGET` is a loss gate, not a model logit. The target
pointer remains a 16-slot head:

```text
present = 1 - target_probs[:, NO_TARGET]
slot_target = target_probs[:, 1:] / max(present, eps)
target_loss = present * CE_soft(slot_logits, slot_target)
```

Skip target loss when `present < 0.05`.

## Algorithm

Use the same three-pass shape as `qnn/bc/target_labeler.py`, but carry
probability mass instead of one pid.

1. **Fire evidence:** on every fire frame, score every plausible enemy pid.
   A pid is plausible if cone evidence, recency-0 physics-hit evidence, or
   optional frag evidence supports it. Do not choose a winner.
2. **Per-pid stream grouping:** for each pid independently, merge its fire
   anchors while that pid remains continuously present in the token stream.
   Multiple pid streams may overlap.
3. **Frame expansion:** extend each pid stream backward and forward through
   continuous stream presence, accumulate scores in the pid's current slot,
   normalize to 17 classes, and put remaining mass on `NO_TARGET`.

### Candidate Geometry

For frame `t` and actor slot `s`:

```text
rel_s       = actor relative position in Quake units
vel_s       = actor velocity in Quake units / second
dist_s      = ||rel_s||
look_t      = normalize(actions["look"][t])
weapon_t    = actions["weapon"][t]
```

No hard range gate at admit time: the physics-hit path is already
range-aware (`_hitscan_test` requires `dist <= max_range`, projectiles
expire at `max_t * speed`), so a hard cone-side gate only rejected
chase frames (e.g. axe selected, closing on an enemy at 200u) where the
demonstrator's intent is unambiguous. Cone evidence admits at any
distance; long-range cone-only candidates collapse on their own through
low `cone`, recency-decayed `vis`, and missing fire support.

For projectile weapons:

```text
t_flight_s = dist_s / projectile_speed[weapon_t]
aim_s      = rel_s + vel_s * t_flight_s
```

For hitscan weapons, `aim_s = rel_s`.

Use the v2 sticky-robust cosine:

```text
cos_s   = max(dot(normalize(aim_s), look_t),
              dot(normalize(rel_s), look_t))
theta_s = arccos(clamp(cos_s, -1, 1))
```

Use opt3 acquire width as a soft scale:

```text
theta_acq_s = clamp(atan(208 / dist_s), 5 deg, 30 deg)
cone_s      = exp(-0.5 * (theta_s / theta_acq_s)^2)
```

Reject only obviously unrelated geometry:

```text
theta_s > 45 deg and no physics hit and no frag support
```

This preserves co-angular uncertainty and blocks enemies that are nowhere near
the demonstrated aim.

### Physics Evidence

Use the hit logic from `scripts/analysis/hit_labeler.py` as the reference:

- Hitscan: ray/AABB plus weapon range and spread.
- Projectiles: closest approach against constant-velocity target.
- RL/GL: splash radius 120u.
- GL: straight-line/gravity approximation from the analysis script.
- Recency gate: only `recency == 0` can be a physics hit.

For each fire frame, compute all pids that physics would hit, not just the
first hit. `scripts/analysis/hit_streams.py` already has the right shape in
`all_hits_at_fire()`.

### Fire Evidence

Candidate set:

```text
C_t = {s | enemy, cone_s >= cone_admit}
      union {s | recency_s == 0 and pid_s in physics_hit_pids_t}
```

Per-candidate evidence (**noisy-OR aggregation**):

```text
cone_e_s = cone_s if cone_s >= cone_admit else 0
hit_e_s  = physics_hit_base if physics hit else 0
base_s   = 1 - (1 - cone_e_s) * (1 - hit_e_s)
vis_s    = exp(-recency_s / recency_tau)
e_s      = base_s * vis_s
```

The noisy-OR form captures the agreement boost between cone and physics that
`max(cone_e, hit_e)` discards. With cone=0.8 and physics=0.95: max→0.95,
noisy-OR→0.99. With cone=0.8 alone: both→0.8. Empirically validated against
v2-and-physics consensus (see "Empirical Validation" below):
on the agreement subset (~5.6% of fires), noisy-OR gains +3pp accuracy and
modest NLL improvement vs max; a fitted logistic with an interaction term
gains another ~65% NLL reduction on the same set but requires per-collection
fitting and is currently deferred — noisy-OR is the principled zero-parameter
upgrade.

Frag support is design-only and not currently wired into
`label_enemy_target_probs`. The "Frag And Death Plumbing" section
below describes the extraction path if frag support is ever added; today
only cone + physics admit candidates.

Important points:

- `physics_hit_base` is high by default (0.95) so a recency-0 physics hit
  enters the candidate set as a co-primary stream rather than a small bump.
- `vis_s` depends on recency only. Do not penalize non-SIGHT modalities
  separately. The earlier `0.45 for non-SIGHT` rule is dropped because modality
  is only a fire-attribution gate for physics. The token stream already carries
  recency and stream continuity.
- `recency_tau` is in seconds (recency is stored in seconds, capped at the
  SIGHT max of 2.0s), not frames. Default 1.0s.
- Cone-only recency-0 runs with 2+ fires are retained through `cone_s`, lower
  `base_s`, and fire-count confidence.

Anchor mass:

```text
present_t  = min(sum_s e_s, present_cap)
anchor_s_t = present_t * e_s / sum_j e_j
```

If no candidate has positive evidence, no anchor is created.

### Engagement Confidence

Per-stream features:

```text
n_fire_e         = number of fire anchors
mean_anchor_e    = mean(anchor_s_t over fire anchors)
max_anchor_e     = max(anchor_s_t over fire anchors)
duration_e       = fwd - back + 1   (extension window length in frames)
bad_end_e        = demonstrator dies during the stream or within
                   death_window frames after it
fire_count_conf_e = 1 - exp(-n_fire_e / fire_count_tau)
death_penalty_e   = death_penalty if bad_end_e else 1.0
```

`eng_conf_e` is a **fitted logistic regression** on those features
(replaces the prior hand-tuned `clip(eng_bias + anchor_weight*mean_anchor +
fire_count_weight*fire_count_conf, eng_min, eng_max)` form):

```text
logit_e   = eng_logistic_intercept
          + w_mean_anchor     * mean_anchor_e
          + w_fire_count_conf * fire_count_conf_e
          + w_max_anchor      * max_anchor_e
          + w_log_duration    * log1p(duration_e)
          + w_log_n_fires     * log1p(n_fire_e)
eng_conf_e = death_penalty_e * sigmoid(logit_e)
```

Defaults are fitted on QWD val shards 0..5 (7,345 physics-confirmed streams)
against a v2-and-physics consensus support target — see "Empirical Validation"
below. Re-fit per collection if labelers, weapon set, or demo sources change
via `scripts/analysis/labeler_v3_eng_conf_audit.py --emit-config`.

`mean_recency` was tested as a feature and dropped: its coefficient is
essentially zero across shards (recency-based decay is already in `vis_s`
during Pass 1, so it carries no additional stream-level signal).

Frag support is not in the default engagement formula. It is optional
calibration/evaluation evidence only unless a cheap, reliable victim-pid path
is implemented. The labeler should not grow a second parser just to get small
frag bonuses.

### Extension Confidence

Extend each stream backward from its first fire and forward from its last fire
while the pid remains present in the token stream. Streams for different pids
may overlap.

For frame `t` in stream `e`:

```text
dt_fire_e_t   = min(abs(t - f) for f in fire_times_e)
time_conf_e_t = time_floor + (1 - time_floor) * exp(-dt_fire_e_t / extension_tau)
score_e_t     = eng_conf_e * time_conf_e_t * vis_e_t
```

Accumulate `score_e_t` into the pid's current slot. If the pid is absent at
`t`, it contributes nothing.

Final distribution:

```text
S_t = sum_s slot_score[t, s]

if S_t == 0:
    p[t, NO_TARGET] = 1.0
    p[t, slot_s] = 0.0
else:
    target_present_t = min(S_t, present_cap)
    p[t, NO_TARGET] = 1.0 - target_present_t
    p[t, slot_s] = target_present_t * slot_score[t, s] / S_t
```

## Magic Constants And Calibration

These are the current `LabelerConfig` defaults in
[`qnn/bc/target_labeler.py`](../qnn/bc/target_labeler.py). Treat them as
pre-calibration starting points; the recipe column describes how to
re-tune per collection.

| Constant | Default | Status | Calibration recipe |
|----------|---------|--------|--------------------|
| `present_cap` | 0.98 | Load-bearing | Grid `{0.95, 0.98, 0.995}`. Pick the smallest cap with no worse held-out consensus NLL and no target-present saturation above 5% of labeled frames. Purpose: avoid fake certainty. |
| `recency_tau` | 1.00s | Load-bearing | Recency is in seconds (SIGHT max 2.0s), not frames. Grid `{0.25, 0.5, 1.0, 1.5, 2.0}` on held-out demos. Minimize consensus NLL on easy frames where opt3, v2, and physics agree; constrain cone-only recency>0 coverage within 10% of opt3 extension coverage. |
| `extension_tau` | 40f | Load-bearing, retuned | Frames. Was 35f; retuned upward based on the boundary audit (dt>=21 frames are 73-67% confirmed, but the old curve predicted ~60%). Grid `{20, 30, 40, 50, 70}`. Minimize NLL on last pre-fire/first post-fire windows for consensus streams. |
| `time_floor` | 0.65 | Load-bearing, retuned | Was 0.50; retuned based on the dt=51+ confirmation rate (~67%). Floor on `time_conf` so far-from-fire frames carry non-trivial mass for streams that the engagement-confidence regression has already judged solid. |
| `death_penalty` | 0.65 | Load-bearing, weak | Grid `{0.4, 0.55, 0.65, 0.8, 1.0}`. Optimize Brier/NLL on cone-only recency-0 runs using demonstrator death as negative outcome. Keep only if it improves held-out calibration; otherwise set to 1.0. |
| Non-SIGHT modality penalty | — | Dropped | Not present in `LabelerConfig`; superseded by recency + stream continuity. |
| `cone_admit` | 0.25 | Load-bearing | Grid `{0.10, 0.20, 0.25, 0.35, 0.50}`. Select by BC target CE and cone-only sustained-run recall. Require the 2+ fire cone-only recency-0 bucket to retain at least 80% of opt3 frames. |
| `physics_hit_base` | 0.95 | Load-bearing | Default lands at the high end of the co-primary range. Grid `{0.80, 0.90, 0.95, 0.98}`. Pick by held-out consensus NLL and overlap-frame calibration. With noisy-OR aggregation, a cone+physics agreement at this default yields `base = 1 - (1-cone)(1-0.95)`, which compresses to ≥0.95 — the agreement-boost target. |
| Evidence aggregator | noisy-OR | Load-bearing, new | `base = 1 - (1-cone_e)(1-phys_e)` replaces `max(cone_e, phys_e)`. Empirically: ~3pp accuracy gain on the agreement subset (~5.6% of fires); the agreement region is where max() loses information. A fitted logistic with cone·physics interaction is a further upgrade (~65% NLL reduction on the consensus subset) but requires per-collection fitting; deferred. |
| `frag_base` | — | Not implemented | Frag support is not wired into `label_enemy_target_probs`. If added, grid `{0.85, 0.90, 0.95, 0.98}` on unambiguous frag windows; metric is NLL for the inferred victim pid. |
| `theta_reject_deg` | 45.0 | Load-bearing | Hard reject for cone-only candidates whose angle exceeds this; physics hits bypass the reject. |
| `fire_count_tau` | 2 | Load-bearing | Grid `{1, 2, 3, 5}`. Optimize calibration for cone-only recency-0 runs bucketed by fire count. The 2+ fire bucket should get higher present mass than 0-fire extension-only runs. |
| `eng_logistic_*` (intercept + 5 weights) | see code | Load-bearing as a family, **fitted** | σ(intercept + w·x) over `(mean_anchor, fire_count_conf, max_anchor, log1p(duration), log1p(n_fires))`. Defaults fitted on QWD val shards 0..5 (7,345 physics-confirmed streams) against v2-and-physics consensus support. Brier 0.061 (vs 0.080 for the prior clip-linear; ~24% reduction); calibration near-perfect across quintiles. Refit via `labeler_v3_eng_conf_audit.py --shards ... --emit-config --drop-recency`. |
| (Removed) `eng_bias`, `anchor_weight`, `fire_count_weight`, `eng_min`, `eng_max` | — | Dropped | Hand-tuned clip-linear form. Replaced by the fitted logistic above. The audit found the prior values systematically under-confident on the top quintile (predicted 0.85 when consensus support was 0.96). |
| (Removed) `mean_recency` feature | — | Dropped | Coefficient was -0.03 across all shards. Recency is already in `vis_s` during Pass 1; stream-level recency carries no additional signal. |
| `death_window` | 10f | Load-bearing | Frames. Matches `cone_only_deaths.py` `POST_K = 10`. Test `{10, 35, 70}` and keep larger values only if they improve held-out calibration without over-penalizing long normal fights. |
| `dead_health_threshold` | 0.05 | Load-bearing | Normalized health (`obs_self_scalars[:, 0]`) below this counts as dead for the death-penalty window. |
| HMM/CRF over stream states | — | Not justified | Boundary audit showed that at fixed `dt_fire`, boundary frames are not systematically worse than interior frames (often slightly better). The dominant error pattern is dt-decay, which `time_conf` already models. Deferred indefinitely. |
| Hard per-weapon range gate | — | Dropped | Cone admits at any distance; physics hit-test is already range-aware (`_hitscan_test` rejects past `max_range`, projectiles expire at `max_t * speed`). |

Calibration split:

1. Hold out 10% of demos by demo hash, not by frame.
2. Build consensus labels on trivial cases: opt3, v2, and physics all select
   the same pid at a fire or stream frame.
3. Build disagreement buckets: cone-only recency>0, cone-only recency=0 with
   0/1/2+ fires, physics-only, and cone+physics multi-pid overlap.
4. Minimize a weighted objective:

   ```text
   objective =
       consensus_nll
     + 0.5 * consensus_brier
     + 0.5 * cone_only_bucket_brier
     + 0.25 * entropy_penalty_on_easy_consensus
   ```

5. Reject any setting that increases hard-argmax jitter against opt3 by more
   than 10% when converting `target_probs` to argmax labels for diagnostics.

This keeps calibration empirical without pretending any single hard labeler is
ground truth.

## Empirical Validation

The three hand-tuned pieces of v3 — evidence aggregation, engagement
confidence, and stream extension — have each been audited against QWD val
data. Scripts live under `scripts/analysis/labeler_v3_*_audit.py`.

### Aggregation: max → noisy-OR (logistic deferred)

A one-shot audit (since removed) compared three per-candidate aggregators on
the v2-and-physics consensus subset (fires where v2's hard pick is also in
the recency-0 physics-hit set; unbiased between the two LFs):

| aggregator | shard 0 NLL | shard 0 acc@1 | shard 1 NLL | shard 1 acc@1 |
|---|---|---|---|---|
| max (prior production) | 0.786 | 91.5% | 0.957 | 94.7% |
| **noisy-OR** (current default) | **0.778** | **94.8%** | **0.952** | **97.9%** |
| logistic with cone·physics interaction | 0.279 | 95.5% | 0.649 | 96.8% |

Findings:

- **`max` discards the agreement signal.** When one candidate has both cone
  and physics evidence and another candidate has only cone, the cone-only
  candidate gets near-equal mass under `max` after sum-normalization. Noisy-OR
  pushes the agreement candidate to ≥0.95 and the cone-only candidate stays
  at its raw cone value.
- **Logistic coefficients are sign-stable across shards**: cone strongly
  negative (alongside a physics-hit candidate, cone-only is *evidence
  against* that candidate), physics strongly positive, interaction positive
  (real agreement boost). The logistic captures information noisy-OR can't
  but requires per-collection fitting.
- **Aggregator choice only matters on the agreement subset** (~5.6% of fires
  on QWD). The other ~94% have either single candidates or are cone-only
  everywhere, where all three aggregators are equivalent.

Decision: **noisy-OR as the production default** (zero parameters, principled
aggregation, ships the agreement boost). Logistic-with-interaction is the
future upgrade if downstream BC validation shows residual agreement-frame
error.

### Engagement confidence: clip-linear → fitted logistic

`scripts/analysis/labeler_v3_eng_conf_audit.py` extracts per-stream features
and a v2-and-physics consensus support target (fraction of frames in
`[back, fwd]` where v2's hard label maps to the stream's pid AND the pid is
in the physics-hit set at at least one fire). Fitted on 6 shards combined
(7,345 physics-confirmed streams):

| predictor | Brier | MAE | corr |
|---|---|---|---|
| clip_linear (prior production) | 0.080 | 0.244 | 0.38 |
| **logistic (current default)** | **0.061** | **0.138** | **0.45** |

Calibration table (5 quintiles, predicted → actual support):

```text
                clip_linear        logistic
quintile 0:     0.60 → 0.70       0.67 → 0.67
quintile 1:     0.72 → 0.84       0.92 → 0.88
quintile 2:     0.79 → 0.95       0.97 → 0.93
quintile 3:     0.83 → 0.97       0.98 → 0.97
quintile 4:     0.87 → 0.97       0.99 → 0.98
```

Findings:

- **Prior clip-linear was systematically under-confident.** It predicted
  0.83–0.87 for streams whose actual consensus support was 0.95–0.97.
  Downstream BC training would learn to discount the labeler's own confidence
  signal.
- **Logistic is well-calibrated across the full range.** Predicted ≈ actual at
  every quintile.
- **Two new features matter**: `max_anchor` (+1.02 logit weight) and
  `log1p(duration)` (-0.70; longer streams are slightly less likely to be
  fully clean, capturing extension dilution).
- **`mean_recency` dropped**: coefficient ≈ -0.03 across all fits. Recency
  decay is already encoded in `vis_s` during Pass 1; stream-level recency
  carries no additional signal.

Production coefficients (frozen in `LabelerConfig`):

```text
intercept              = -4.98
w_mean_anchor          = +5.31
w_fire_count_conf      = +4.54
w_max_anchor           = +1.02
w_log_duration         = -0.70
w_log_n_fires          = +0.60
```

Refit recipe:

```bash
PYTHONPATH=src python3 scripts/analysis/labeler_v3_eng_conf_audit.py \
    --shards 0,1,2,3,4,5 --emit-config --drop-recency
```

### Extension boundaries: HMM/CRF not justified

`scripts/analysis/labeler_v3_boundary_audit.py` measures per-frame
consensus-confirmation rate inside `[back, fwd]`, bucketed by distance from
the nearest fire and by interior-vs-boundary position.

By `dt_fire` (smooth monotonic decay):

```text
dt=0       0.905
dt=1-2     0.876
dt=3-5     0.879
dt=6-10    0.863
dt=11-20   0.813
dt=21-50   0.737
dt=51+     0.666
```

By position (boundary = first/last 5 frames of `[back, fwd]`), holding `dt`
fixed:

```text
                interior   boundary    delta
dt=0            91.1%      84.8%      -6.3pp
dt=1-2          88.6%      80.8%      -7.8pp
dt=3-5          88.3%      84.9%      -3.4pp
dt=6-10         85.6%      88.6%      +3.0pp
dt=11-20        80.1%      85.1%      +5.0pp
dt=21-50        73.4%      74.5%      +1.1pp
dt=51+          66.2%      67.9%      +1.7pp
```

Findings:

- **The dominant error pattern is dt-decay, not structural boundary error.**
  At fixed dt, boundary frames are not systematically worse than interior
  frames — in mid-dt buckets they're actually slightly *better* (the
  termination rule "extend while pid is in stream" cuts streams roughly where
  attribution actually gets weak).
- **An HMM/CRF over per-pid stream states would mostly relearn `dt_fire`.**
  The structural extension rule is doing its job; replacing it with a learned
  transition model is high-cost low-yield. Deferred indefinitely.
- **The real fix was retuning `time_conf` decay shape.** Prior defaults
  (`time_floor = 0.50`, `extension_tau = 35`) under-predicted the dt>=20 tail.
  Current defaults (`time_floor = 0.65`, `extension_tau = 40`) match the
  observed confirmation curve closely.

### End-to-end effect on emitted labels

On shard 0 (262,241 frames, 79,927 labeled), the upgraded labeler vs the
prior production version:

| | prior | current |
|---|---|---|
| labeled frames (present > 0.05) | 79,770 | 79,927 |
| mean present-mass on labeled | 0.745 | **0.852** |
| structural correlation (present-mass) | — | 0.988 |

Same streams (high correlation), substantially better-calibrated confidence.
Downstream BC loss will see less under-confidence on solid streams and the
new `NO_TARGET` mass is closer to the true uncertainty.

## Frag And Death Plumbing

### Death Detection

Use death detection by default because it is already present in collected
arrays and costs almost nothing.

Source:

- `obs_self_scalars[:, 0]` is normalized health (`health / MAX_HEALTH`) per
  `src/docs/contracts/semantics/semantics.1.md`.
- The analysis scripts use `_HEALTH = 0` and `DEAD_THR = 0.05`.

Rules:

```text
dead[t] = self_scalars[t, 0] < dead_health_threshold
death_edge[t] = dead[t] and not dead[t - 1]
bad_end(stream) =
    any(dead[stream.start:stream.end + 1]) or
    any(death_edge[stream.end + 1:stream.end + 1 + death_window])
```

Use `dead_health_threshold = 0.05` initially. Calibrate only against obvious
`labels.dead` intervals from the demo classifier if needed. Never read outside
the current sub-episode.

### Frag Events

Frag events are optional. The default v3-simple design does not need them.

The cheap extraction path is metadata-level and useful for diagnostics:

- `demo/parser.py:parse_demo_metadata()` already captures `SVC_UPDATEFRAGS`
  into `DemoMetadata.frag_updates`.
- Each row has `player_slot`, cumulative `frags`, `message_index`, and
  `time_s`.
- `demo/qw_classifier.c` already reads `QW_SVC_UPDATEFRAGS` and sets
  `active_state.frag_up` when `slot == self_slot` and the cumulative frag count
  increases.

That gives "the recorder got a frag near this time." It does not give the
victim pid.

If frag windows are used for calibration:

1. In `qnn/bc/collect.py`, pass source demo metadata into `_unpack_episode()`
   only as an optional sidecar, not through the model obs.
2. Map a frag update to emitted frames by `time_s` when available; otherwise
   use `message_index * n_emitted / total_frames` as an approximation.
3. Create `frag_up[t] = 1` for the first emitted frame at or after the update.
4. Infer a candidate victim only when exactly one enemy actor has a `DEATH`
   event (`ACTION_IDS["DEATH"]`) within `[t - 5, t + 20]` and that pid is in
   the recent stream/candidate set.
5. Drop the frag event for label calibration if there are zero or multiple
   death-event candidates, if the candidate was not in stream, or if the frame
   crosses a sub-episode boundary.

Do not parse `svc_damage` for QWD. `svc_damage` is a broadcast view kick and
damage-direction message; the current Python parser and classifier both call
`skip_damage()`, and it does not provide outgoing victim pid. Training-sidecar
damage records are useful for PPO/training telemetry, but historical QWD BC
collections should not depend on them.

If this frag path becomes more code than the labeler itself, delete frag
support from v3-simple and keep only death penalties plus physics/cone
calibration. The expected gain is too small to justify brittle demo plumbing.

## Why Hybrid Failed

`scripts/analysis/hybrid_labeler.py` used opt3 sticky cone attribution, then
changed hard decisions in two places:

- On acquire, if multiple cone-admitted enemies existed and exactly one was a
  physics hit, it chose the physics pid instead of the cone argmax.
- On sticky keep, if the current pid was still inside the release cone but was
  not a physics hit, and exactly one other admitted pid was a physics hit, it
  overrode sticky and switched.

That design made switches worse: 3,788 vs. 3,109 for opt3, with worse quality
ratio. The failure is expected. A hard sticky state machine has one current
pid; any per-fire tie-break that sometimes prefers physics turns close
co-angular frames into A/B choices again. The sticky override is especially
dangerous because it breaks the opt3 invariant that a current pid inside the
release cone stays held.

V3 avoids this in three ways:

1. Physics creates or strengthens a candidate stream; it does not override
   another stream.
2. Per-pid streams are grouped independently, so overlapping evidence remains
   overlapping probability mass instead of becoming a forced switch.
3. Soft CE sees the ambiguous distribution. It is not asked to learn that the
   correct class changed from A to B for one noisy fire.

Soft CE is not a magic eraser. If the final distribution is collapsed by bad
calibration, it will reproduce hybrid's error as a high-confidence wrong label.
That is why the validation plan includes a co-primary physics ablation, entropy
checks on disagreement frames, and hard-argmax switch diagnostics as a guardrail.

## What To Keep And Drop

Keep:

- Enemy actor and teammate filtering from `qnn/bc/target_labeler.py`.
- Projectile lead correction and `max(cos_lead, cos_current)`.
- Opt3 adaptive acquire geometry as a soft cone scale.
- Three-pass fire/group/extend structure.
- Stream-loss boundaries within sub-episodes.

Drop:

- Sigma/K(p) as hard cone width machinery.
- `p_accept` and `p_release` user-facing knobs.
- Exclusive sticky `current_pid` as the output decision.
- Margin transfer.
- Non-SIGHT confidence penalty.
- Required frag support.
- Hard per-weapon range gate at admit time (chase frames matter; physics
  hit-test is already range-aware).
- `max(cone, physics)` per-candidate aggregation (loses the agreement
  boost). Replaced by noisy-OR.
- Hand-tuned `clip(eng_bias + anchor_weight*mean_anchor +
  fire_count_weight*fire_count_conf, eng_min, eng_max)` engagement
  confidence (systematically under-confident on solid streams). Replaced
  by a fitted logistic regression.
- `mean_recency` as a stream-level feature (coefficient ≈ 0 across
  shards; recency is already in `vis_s`).
- HMM/CRF over per-pid stream states (boundary error is not the dominant
  disagreement mode; `time_conf` dt-decay already captures the right
  shape after retuning).

## Validation Plan

### Required Variants

Run all validation on full QWD val and at least one 10% train calibration split.

| Variant | Purpose |
|---------|---------|
| opt3/orig hard labels | Baseline hard labeler. |
| v2 lead/range hard labels | Confirms lead/range changes do not regress. |
| hit-anchored hard labels | Physics-only lower-coverage reference. |
| hybrid hard labels | Failure-mode reference. |
| v3 cone+physics-bump | Prior simple behavior: cone primary, physics high base but no co-primary stream admission beyond old thresholding. |
| v3 cone-OR-physics co-primary | Required ablation; expected preferred default. |
| v3 no-death-penalty | Tests whether death penalty is useful or cargo-culted. |

The deciding metric between `cone+physics-bump` and `cone-OR-physics` is
held-out calibration, not raw hard switch count:

```text
primary:   consensus NLL + Brier
secondary: entropy separation on disagreement frames
guardrail: hard-argmax jitter <= opt3 * 1.10
guardrail: sustained cone-only recency-0 2+fire recall >= 80% of opt3
```

### Cross-Labeler Calibration

Expected:

| Bucket | Requirement |
|--------|-------------|
| Consensus frames | `p(consensus_pid) >= 0.80` on at least 90% of frames after calibration. |
| Two-pid disagreement | `p(pid_a) + p(pid_b) >= 0.80` on at least 90% of frames. |
| Physics/cone overlap | Co-primary v3 has lower NLL than cone+physics-bump. |
| Cone-only recency=0 with 2+ fires | Present mass exceeds 0-fire extension-only bucket by at least 0.15. |
| Cone-only death bucket | If death penalty is enabled, Brier improves over no-death-penalty. |

### Hard-Argmax Diagnostics

Convert `target_probs` to hard labels only for diagnostics:

```text
hard[t] = TARGET_IGNORE if p(NO_TARGET) >= 0.5 else argmax_slot(p_slots)
```

Report the same table as the analysis scripts:

```text
variant    labeled    switches    jitter    midband    legit    legit/jitter
```

This diagnostic should catch hybrid-like flipping, but it should not be the
only promotion criterion because the classifier is cone-biased against physics
streams.

### BC Ablation

Train same-seed BC runs:

| Run | Labels |
|-----|--------|
| baseline | current hard target labels |
| v3-bump | distributions with cone primary and physics bump |
| v3-co-primary | distributions with cone OR physics candidate streams |
| v3-co-primary-no-death | same, death penalty disabled |

Compare target soft CE, hard compatibility accuracy, downstream eval damage,
live-play target stability, and whether move/look/fire/weapon validation
metrics regress.

## Implementation Plan

#### PR 1 - Distribution labeler

File: `qnn/bc/target_labeler.py`

1. Add `label_enemy_target_probs(...) -> np.ndarray` returning `(T, 17)`.
2. Port `all_hits_at_fire()` and weapon physics helpers from
   `scripts/analysis/hit_labeler.py` / `hit_streams.py`.
3. Remove sigma/K hard cone width code from the new distribution path.
4. Unit-test row sums, empty episodes, axe/LG gates, co-angular split,
   physics-only admission, cone-only sustained admission, and stream loss.

#### PR 2 - Collector emit format

File: `qnn/bc/collect.py`

1. Emit `act_target_probs.npy`.
2. Continue emitting `act_target.npy` during transition.
3. Change `combat_only` to keep frames where `1 - p(NO_TARGET) >= 0.25`.
4. Add manifest metadata:
   `target_labeler_version = "v3-simple-co-primary"` and
   `target_probs_classes = ["NO_TARGET", "slot_0", ..., "slot_15"]`.

#### PR 3 - Soft CE

Files: `qnn/bc/train.py`, loader plumbing, and policy loss code.

1. Load `act_target_probs` when present.
2. Train target pointer with present-weighted soft CE.
3. Keep hard-label CE fallback for old collections.
4. Report `loss_target_soft`, `target_present_mean`, target entropy, and hard
   argmax compatibility.

#### PR 4 - Analysis

Files: `scripts/analysis/`.

1. Add a v3 distribution report.
2. Add the required co-primary vs bump ablation.
3. Reuse `switch_quality.py` only for hard-argmax diagnostics.
4. Save calibration tables by bucket.

## Pseudocode

```python
def label_enemy_target_probs(obs, actions, *, config=DEFAULT,
                                    sight_only=False):
    T, N = obs["entity_types"].shape
    slot_scores = zeros((T, N), float32)

    enemy = actor_mask(obs) & ~teammate_mask(obs)
    if sight_only:
        enemy &= modality(obs) == SIGHT

    rel, vel, dist = actor_rel_vel_dist(obs)
    look = normalize(actions["look"])
    weapon = actions.get("weapon", full(T, ROCKET))

    aim = lead_corrected_aim(rel, vel, dist, weapon)
    cos_lead = dot(normalize(aim), look)
    cos_cur = dot(normalize(rel), look)
    theta = arccos(clip(max(cos_lead, cos_cur), -1, 1))
    theta_acq = clamp(atan(208.0 / max(dist, 1e-3)), deg(5), deg(30))
    cone = exp(-0.5 * (theta / theta_acq) ** 2)
    theta_reject = deg(config.theta_reject_deg)

    hit_pids_by_t = all_physics_hit_pids(obs, actions, enemy)
    anchors_by_pid = defaultdict(list)

    for t in where(actions["fire"] == 1):
        cand = []
        evid = []

        for s in range(N):
            if not enemy[t, s]:
                continue
            pid = pid_at(obs, t, s)
            hit = pid in hit_pids_by_t[t] and recency(obs, t, s) == 0
            cone_ok = cone[t, s] >= config.cone_admit
            if not (cone_ok or hit):
                continue
            if theta[t, s] > theta_reject and not hit:
                continue

            # Noisy-OR aggregation captures cone+physics agreement boost.
            cone_e = cone[t, s] if cone_ok else 0.0
            phys_e = config.physics_hit_base if hit else 0.0
            base = 1.0 - (1.0 - cone_e) * (1.0 - phys_e)
            vis = exp(-recency(obs, t, s) / config.recency_tau)
            cand.append((pid, s))
            evid.append(base * vis)

        total = sum(evid)
        if total <= 0:
            continue
        present = min(total, config.present_cap)
        for (pid, _), e in zip(cand, evid):
            anchors_by_pid[pid].append((t, present * e / total))

    streams = []
    dead = obs["self_scalars"][:, 0] < config.dead_health_threshold
    death_edge = dead & ~shift_right(dead)

    for pid, anchors in anchors_by_pid.items():
        for group in split_on_stream_gaps(pid, anchors, enemy, obs):
            n_fire = len(group)
            anchor_vals = [a for _, a in group]
            mean_anchor = mean(anchor_vals)
            max_anchor = max(anchor_vals)
            fire_conf = 1.0 - exp(-n_fire / config.fire_count_tau)
            window = extend_while_pid_in_stream(pid, group, enemy, obs)
            duration = window.end - window.start + 1
            bad_end = death_in_or_after(window, dead, death_edge,
                                        config.death_window)
            death_pen = config.death_penalty if bad_end else 1.0
            # Fitted logistic on (mean_anchor, fire_count_conf, max_anchor,
            # log1p(duration), log1p(n_fires)); defaults frozen on QWD val
            # shards 0..5.  See "Empirical Validation" for the calibration.
            logit = (
                config.eng_logistic_intercept
                + config.eng_logistic_w_mean_anchor     * mean_anchor
                + config.eng_logistic_w_fire_count_conf * fire_conf
                + config.eng_logistic_w_max_anchor      * max_anchor
                + config.eng_logistic_w_log_duration    * log1p(duration)
                + config.eng_logistic_w_log_n_fires     * log1p(n_fire)
            )
            eng_conf = death_pen * sigmoid(logit)
            streams.append((pid, group, window, eng_conf))

    for pid, group, (start, end), eng_conf in streams:
        fire_ts = [t for t, _ in group]
        for t in range(start, end + 1):
            s = slot_for_pid(obs, t, pid, enemy)
            if s < 0:
                continue
            dt = min(abs(t - f) for f in fire_ts)
            time_conf = (
                config.time_floor
                + (1 - config.time_floor) * exp(-dt / config.extension_tau)
            )
            vis = exp(-recency(obs, t, s) / config.recency_tau)
            slot_scores[t, s] += eng_conf * time_conf * vis

    dist17 = zeros((T, 17), float32)
    for t in range(T):
        S = slot_scores[t].sum()
        if S <= 0:
            dist17[t, 0] = 1.0
            continue
        present = min(S, config.present_cap)
        dist17[t, 0] = 1.0 - present
        dist17[t, 1:] = present * slot_scores[t] / S
    return dist17
```

The design should stay this small. If implementation needs many more special
cases, prefer deleting optional frag support before complicating the core
cone-plus-physics distribution labeler.
