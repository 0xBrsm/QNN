# Target labeler and engine sticky: design, validation, and remaining gaps

> **Historical.** This doc describes the v2 hard-label
> `label_enemy_target` design and its engine alignment. v2 has been
> removed from `qnn/bc/target_labeler.py`; the live path is the v3
> distribution labeler `label_enemy_target_probs`
> (see [labeler_v3_simple.md](labeler_v3_simple.md)). The engine sticky
> machine in `qnn_oracle.c` is unchanged and still mirrors the v2 cone
> geometry; the agreement numbers below remain a useful reference for
> the engine state machine.
>
> **Slot ordering is also stale.** The "bucket A = target == slot 0 /
> bucket B = target ≠ slot 0" framing, the
> `acc_target_slot0_baseline` metric, and the slot-0-prior class
> weighting all assume a deterministic per-frame entity ordering
> (pool → recency → team → threat). That ordering was dropped; entity
> slots are now arbitrary (effectively engine edict order). Treat
> every slot-0-keyed claim below as a historical artifact, not a live
> baseline. For current target-head comparisons use `val_target_kl`
> and restrict to multi-candidate frames.

## TL;DR

We redesigned the BC target labeler (`qnn/bc/target_labeler.py`) and the
engine sticky state machine (`engine/common/qnn_oracle.c`) around a shared
**adaptive-cone Schmitt-trigger** rule:

```
acquire cone(d) = clamp(atan(208/d), 5°, 30°)     # tighter at long range
release cone(d) = clamp(atan(416/d), 5°, 45°)     # K=2 transverse offset, capped
sticky:           keep current_pid across fires
                  as long as release cone holds
```

On the QWD training data (1,672 demos, 30M ticks):

- **within-segment pid switches: 40,726 → 18,567 (−54.4%)** vs the original
  pure cone-argmax labeler.
- **target == slot 0 alignment**: 97.56% on train, 97.43% on val.
- **labeler↔engine pid agreement (overlap frames)**: 98.24%.
- Remaining 2.5% disagreement decomposes into two legitimate learned-target
  signals — modality-driven (~35%) and lookahead/cascade (~65%) — neither
  fixable causally.

The design was selected from a sweep of >20 candidate configurations
(symmetric vs asymmetric cones, fixed vs adaptive, with/without margin
transfer, with/without occlusion locks, and with/without modality filtering),
each evaluated against frame-level metrics on the full corpus. Two empirical
findings were instrumental in landing the final design:

1. **The cone-jitter is a labeling artifact, not real target switching.**
   Removed-switches happen at baseline-rate aim motion (mean angvel 4.81° vs
   4.76° for non-switch frames) while preserved switches happen at flick-rate
   (mean 13.51°, p99 156°). Statistically, the removed switches are
   indistinguishable from random non-switch combat motion.

2. **The engine's modality filter is doing real work, not a quirk to
   eliminate.** A controlled test where we relaxed the engine's modality
   restriction produced 60× more disagreements with labels than it fixed
   (1.62M strict-wins vs 27k relaxed-wins). The filter implicitly forces
   engagement boundaries that line up with labeler's lookahead-driven
   boundaries.

Everything below documents the evidence behind each choice.

---

## 1. Background

### 1.1 Target labels in BC training

QNN BC training uses per-frame `target` labels indicating which obs slot
(0–15) contains the demonstrator's current engagement target. Labels of −100
indicate "no target / skip loss". The target head predicts this slot index
from observation context; the predicted pid is recovered via
`entity_ids[t, target[t], 2]`.

For training to be informative, the labels must reliably reflect the
demonstrator's intent — not just the cone-argmax of their crosshair on each
fire frame, which is what the pre-existing labeler used.

### 1.2 The previous labeler

`label_enemy_target` (pre-opt3) used pure cone-argmax fire-anchored attribution:

```
pass 1: for each fire tick t, if any enemy within 30° cone, assign valid_shots[t] = argmax(cos)
pass 2: group consecutive same-pid valid_shots into engagements (split on pid change or stream gap)
pass 3: extend each engagement's labels backward (to previous engagement's end) and forward (until pid leaves obs)
```

This produced **40,726 within-segment pid switches** on the train set —
sequences where the labeled pid alternates within a single contiguous labeled
run (e.g., A→B→A→B→A). Most are bursts of 3–5 alternations.

### 1.3 The original engine sticky (commit `1d539aba`)

The engine's `qnn_oracle.c` already had a sticky-by-pid state machine that
mirrored the labeler heuristic at engine time: at fire press, acquire/transfer
to the cone-argmax pid (cos ≥ cos(30°)) and persist it across frames as
long as the pid remained in the candidate pool. Slot 0 was promoted to the
sticky pid.

The two were not perfectly aligned: cone-argmax can pick a different pid
frame-to-frame when two enemies are co-angular, producing label oscillation
that the engine inherits.

### 1.4 The question

Why does the same `(look, fire, entity)` data produce these spurious-looking
target switches, and is there a labeler design that captures real engagement
intent more reliably while remaining compatible with a causal engine that the
trained model will see at inference?

---

## 2. Diagnosis: the cone-jitter artifact

### 2.1 Structural finding

Of the 40,726 within-segment switches, **20,364 segments (11.5%) contain at
least one pid switch; 4,276 segments contain ≥2 switches (oscillations)**.
Among multi-switch segments, A→B→A patterns dominate (49% of multi-switch
segments are exactly A→B→A; another 24% are A→B→A→B or longer).

This is not a sparse failure mode; it's a structural property of fire-anchored
cone-argmax labels when two enemies are visible simultaneously.

### 2.2 Cosine geometry at switch frames

At every within-segment switch frame `c` we compared the two pids' cosines
against the demonstrator's aim:

```
median cos(look, B) at switch    = 0.956   (≈ 17° off aim)
median cos(look, A) at switch    = 0.953   (≈ 18° off aim)
median |cos_B − cos_A| at switch = 0.005
both pids inside the 30° cone    = 84.2% of switches
```

The two enemies are nearly co-angular at the moment of attribution flip.
The cone-argmax pick is essentially noise — whichever enemy was slightly
better aligned at that specific fire frame.

### 2.3 The 3D separation signature

At the same switch frames, the median 3D world-space separation between the
two pids was **229 Quake units** (≈ 7 player-widths). For control "real"
switches (defined later), the separation was **384 units**. The 67% gap is a
clean geometric signature: artifact switches happen when two enemies are
physically close *and* angularly stacked from the demonstrator's POV.

### 2.4 The angular-velocity test

A clean discriminator emerged from the demonstrator's aim motion (frame-to-frame
look-vector angle):

| population                             | mean angvel | p99 angvel    |
|----------------------------------------|-------------|---------------|
| switches removed by sticky labeler     |     4.81°   |    31°/frame  |
| switches preserved by sticky labeler   |    13.51°   |   156°/frame  |
| random non-switch labeled frames       |     4.76°   |    40°/frame  |

Removed switches occur at **baseline-rate aim motion** — statistically
indistinguishable from non-switch combat motion. Preserved switches occur at
flick-rate motion — clearly signaling demonstrator's intentional target
change. This is the strongest direct evidence that the cone-jitter
switches are noise (no demonstrator-intent change associated).

---

## 3. Design space exploration

We swept the labeler design space along five axes, each independently
validated on the full 118-shard train corpus.

### 3.1 Sticky vs. no sticky (1 axis)

Adding a sticky `current_pid` that persists across fires (only released when
pid leaves the cone or stream) reduced switches by **42.91%** (40,726 → 23,252)
with **0% rule violations** in 31M frames, indicating the implementation is
correct (every preserved switch happens at a legitimate cone-release or
stream-loss boundary).

The frame coverage Δ vs. current labels was **−0.03%** — essentially
unchanged.

### 3.2 Adaptive vs. fixed acquire cone

A fixed 30° cone admits enemies up to `d·tan(30°) ≈ 0.577·d` units perpendicular
from aim, which at long range corresponds to enemies meaningfully apart in 3D.
We tested an **adaptive acquire cone** of the form
`cone(d) = clamp(atan(X/d), 5°, 30°)`, parametrized by transverse offset X:

| X (units) | switch Δ vs current | labeled-frame Δ |
|-----------|---------------------|------------------|
|  80       |     −62.19%         |    −8.28%        |
| 150       |     −51.23%         |    −2.42%        |
| 180       |     −48.77%         |    −1.55%        |
| 208       |     −47.42%         |    −1.18%        |
| 220       |     −46.40%         |    −0.92%        |
| 300       |     −43.78%         |    −0.36%        |

The marginal-cost ratio (frame loss per switch removed) climbs monotonically
as X tightens. There is no sharp knee; X=180–220 sits on the most efficient
part of the curve. We selected **X = 208** as a defensible mid-curve
multiple of 16, and verified that switch reduction is essentially flat across
176–224 when paired with the asymmetric release rule selected below
(differences within 1% of switch counts).

The "lost frames" are predominantly long-range speculative fires where no
enemy was precisely aimed at; these were arguably noise in the previous
labeler too.

### 3.3 Release cone (asymmetric Schmitt trigger)

Once an engagement is established, releasing it on the same tight cone used
for acquire causes too much within-engagement transfer. We tested asymmetric
release with the same `trans_X` form but a wider angular cap:

| release config              | switch Δ vs current | labeled Δ | pid disagree vs current |
|-----------------------------|---------------------|-----------|--------------------------|
| trans_416, cap 30° (= acquire) | −51.23%          |  −2.42%  |   1.81%                  |
| trans_416, cap 45°            | −54.02%          |  −2.42%  |   2.25%                  |
| trans_416, cap 60°            | −59.41%          |  −2.42%  |   2.98%                  |
| trans_416, no cap (=89°)      | −57.51%          |  −0.79%  |   2.56%                  |

The no-cap variant was identified through example dumps to **over-hold**:
e.g., a demonstrator with sticky on far-side enemy A who clearly transferred
to near-side enemy B was kept on A by the labeler because A remained inside
the (wide at close range) release cone. The 45° cap eliminated these failure
modes without introducing other distortions.

### 3.4 K=2 transverse multiplier

For the release cone's transverse offset, we tested K ∈ {1, 1.5, 2, 2.5, 3}
where K = release_X / acquire_X:

| K   | release_X | switch Δ | over-hold examples remaining |
|-----|-----------|----------|-------------------------------|
| 1.0 | 208       | −46.31%  | many                          |
| 1.5 | 312       | −55.34%  | few                           |
| 2.0 | 416       | −57.54%  | none in dumps                 |
| 2.5 | 520       | −58.93%  | introduces new over-holds     |
| 3.0 | 624       | −59.67%  | introduces more over-holds    |

K = 2.0 was selected as the canonical Schmitt-trigger 2:1 ratio. Beyond K=2,
the over-hold failure mode reappears in dumped examples without commensurate
metric improvement.

### 3.5 Margin transfer rule (rejected)

We tested adding a **cos margin** rule: at each fire frame where sticky is
still in release cone, also check if any other admitted enemy has
`cos(other) > cos(sticky) + δ`, and transfer if so. We swept δ ∈
{0.025, 0.05, 0.075, 0.10, 0.15, 0.20}.

**The margin rule introduced its own jitter.** Tracking dwell time on the
new pid after each margin-triggered transfer at δ=0.075:

```
total margin transfers across train shards : 6,365
median dwell on new pid                    : 13 frames (~650 ms)
revert to prev_pid                         : 32.54%
dwell ≤4 frames (sub-200ms blips)          : 12.49%
```

The 32.54% revert rate compares to 8.29% for sticky-cone-driven transfers —
nearly 4× higher. Each margin transfer that reverts produces an A→B→A pattern,
defeating the entire purpose of the sticky design.

**Rejected.** Pure release-cone Schmitt-trigger preferred.

### 3.6 Occlusion rule (rejected)

Following an intuition that two enemies could form an A↔B pattern when one
geometrically occludes the other, we tested an occlusion lock: refuse to
transfer to a candidate that is in occlusion relationship with current_pid.

```
total locks activated across 31M train frames : 25
total frame labels affected                   : 81
switch Δ vs no occlusion rule                 : −8 (noise)
```

The occlusion rule fires too rarely under the existing release-cone semantics
(which already handles the same cases) to justify the implementation surface.
Under tighter release cones (e.g., cap_30°) it activates more frequently but
those configs are themselves Pareto-dominated by cap_45°.

**Rejected.** Effect within noise; complexity not warranted.

### 3.7 Engine modality filter

The engine's original sticky logic restricts retention to SIGHT or PROXIMITY
modality (filtering out SOUND modality, which actors can take when last
observed via sound rather than sight). We tested **relaxing** this filter to
match the labeler's "any actor in obs" semantics. The result was striking
and counterintuitive — see Section 5.4 below.

**Rejected.** Modality filter retained as is.

---

## 4. Implementation

### 4.1 Python labeler

Pass 1 was rewritten to use the sticky state machine; pass 2 and pass 3 (which
handle engagement grouping and label extension) were preserved unchanged.

The complete pass-1 logic, post-bugfix (Section 6.2 below), is in
`qnn/bc/target_labeler.py` (commit `256a8391`).

### 4.2 Engine sticky state machine

Mirror changes were applied in `engine/common/qnn_oracle.c` (commit
`683e4852`) to keep the engine's slot-0 promotion logic aligned with the
labeler at inference time:

- Added `dist` field to the candidate struct (acquire cone is per-candidate
  distance-dependent).
- Added `QNN_OracleAcquireConeCos(dist)` and `QNN_OracleReleaseConeCos(dist)`
  inline helpers using the same formulas as the labeler.
- Restructured the per-fire-frame block to do sticky-keep release-cone
  check, then conditional acquire — matching the labeler's pass-1 control
  flow.

Both binaries (`ppo_worker`, `nq_demo_worker`, `qw_demo_worker`) were
rebuilt and verified to compile clean with `-Werror=implicit-function-declaration`.

---

## 5. Validation

### 5.1 Switch reduction (the headline metric)

```
                          before          after          Δ
within-segment switches   40,726         18,567         −54.41%
multi-pid segments        20,364         14,840         −27.13%
≥4-switch oscillations     2,538            470         −81.5%
total engagements        217,969        201,033          −7.8%
frames per engagement (median)  24            27          +12.5%
fires per engagement (median)    7             8          +14.3%
```

### 5.2 Label-frame coverage

```
total frames in train          : 30,991,688
labeled frames before          : 6,409,586 (20.70%)
labeled frames after           : 6,326,242 (20.42%)
                                 ─────────
                                 −0.45 pp
```

The lost frames are concentrated at:
1. Long-range speculative fires that the adaptive acquire cone now rejects
   (no enemy was sufficiently in line at that range)
2. Stream-loss windows where the new labeler correctly releases sticky
   (per the Section 6.2 bug fix)

### 5.3 Engine alignment

After landing opt3 in both the labeler and the engine and recollecting:

```
target == slot 0 (train): 97.56%
target == slot 0 (val):   97.43%
```

The 2.5% remaining disagreement is explained in Section 6.

### 5.4 The relaxed-engine head-to-head test

The most important validation result is from a controlled experiment on the
engine's modality filter. We implemented a "relaxed engine" in Python (causal
state machine that keeps sticky through SIGHT→SOUND transitions, mirroring
labeler semantics) and compared per-frame pid choices against labels:

```
                                       train labeled frames: 6,354,798

  strict engine matches label,
     relaxed engine matches label too : 4,563,983 (71.82%)
  only relaxed engine matches label   :    27,276 ( 0.43%)   ← relaxed wins
  only strict engine matches label    : 1,621,151 (25.51%)   ← STRICT WINS BIG
  neither matches label               :   142,388 ( 2.24%)

  Net advantage of relaxing engine    : −1,593,875 frames
```

**Relaxing the engine produces 60× more disagreements than it fixes.** The
mechanism: when the strict engine releases sticky on a SIGHT→SOUND modality
transition, the next fire re-acquires via cone-argmax — which usually picks
the demonstrator's next real target. The labeler's pass-3 lookahead also
back-walks the new target into the same window. Both end up on the same
new pid; the strict engine and the labeler stay aligned via this indirect
mechanism. The relaxed engine kept the old pid, persisting a stale target
that the demonstrator had already moved on from.

This is *why* the modality filter looks like a legacy quirk but is actually
load-bearing: it forces re-engagement at the same boundaries the labeler's
lookahead places engagements.

### 5.5 Train/val parity

```
                          train       val           delta
switch reduction         −54.4%      −53.5%        within 1pp
labeled coverage          20.42%      20.07%       within 0.4pp
target == slot 0          97.56%      97.43%       within 0.2pp
```

Generalization holds. The labeler design is not overfit to train.

---

## 6. Open questions and remaining gaps

### 6.1 The 2.5% labeler–engine disagreement

After all alignment work, ~155k train frames have target ≠ slot 0. The
decomposition:

```
Cause                          fraction      explanation
─────────────────────────────────────────────────────────
modality-driven (recency > 0)  ~35%   engine releases sticky on SIGHT→SOUND;
                                       labeler extends through MEMORY/SOUND
                                       window (≤ 0.05–0.1 s) via lookahead
lookahead back-walk             ~65%   labeler labels pre-fire frames with
+ cascade                              upcoming engagement's pid; engine
                                       cannot causally anticipate
```

Both categories represent **legitimate learned-target signals**, not bugs:

- **Modality cases**: the demonstrator's target left immediate vis briefly
  (e.g., enemy ran behind cover) but the demonstrator is still engaged. The
  label says "maintain target through brief vis loss" — a non-trivial signal
  that helps the model handle occlusion-tolerant target tracking.

- **Lookahead cases**: the demonstrator is about to commit to a new target.
  The label says "switch your attention now, before the fire" — teaching the
  model to anticipate target transitions from pre-fire cues (turning, weapon
  prep) rather than waiting for the fire to follow the engine's sticky update.

Neither can be fixed without giving the engine future knowledge.

### 6.2 Labeler bug discovered during validation

While building the engine↔labeler alignment story, we discovered that the
labeler's pass 1 was **not releasing sticky** when its current_pid transiently
left the obs entity pool between two fires. The labeler only iterates fire
ticks; if pid X dropped from obs at frame F (between fire t1 and t2) and
returned by t2, the labeler's per-fire check at t2 saw X back in obs and
kept the sticky.

This was inconsistent with the engine, which releases on every-frame check.
The fix (commit `256a8391`) adds an explicit between-fires stream-loss check
to pass 1. Effect: −0.45% labeled frames, +0.23 pp slot-0 alignment.

Frames now correctly labeled −100 during the gap; the post-gap fire goes
through the acquire path (matching engine behavior).

### 6.3 Open: cascade attribution

About 55% of fire-frame disagreements at first-divergence events have no
immediate-window cause (no stream gap, no recency drop). These are
**cascaded** from earlier divergences that propagate. Decomposing the cascade
source distribution to within-episode events would tighten the explanation
but isn't a design lever.

### 6.4 Open: float precision near cone boundaries

Labeler computes cos in Python float32 over float16 obs; engine computes in
C float32 from world coordinates. A small fraction of disagreements at cone
boundaries are likely numerical-precision flips. Not isolated precisely, but
visible as a low-percent residual in the modality-vs-lookahead breakdown.

---

## 7. Methodology notes (for future analysis)

### 7.1 Empirical-first design discipline

Every parameter (X, K, release cap, margin threshold) was selected via
direct measurement on the full train corpus (118 shards, 31M frames), not
via intuition. Selection criteria:

- Switch count Δ (lower = better, all else equal)
- Pid agreement with current labels (closer to 100% = closer to "no labels
  changed"; closer to baseline-sticky 1.57% = "match conservative existing
  behavior")
- Frame coverage (don't lose training signal unnecessarily)
- Manual inspection of worst-disagreement segments (qualitative correctness)

### 7.2 Failure modes encountered during analysis

Two errors of analysis were caught mid-design and revised:

1. **Initial "K=2 with no upper cap" recommendation.** Was reversed after
   example dumps showed over-hold failures the aggregate metrics didn't
   capture. Lesson: always inspect worst-case segments before trusting
   aggregate Pareto curves.

2. **Buggy Python "engine replica" used in early analysis.** The first
   replica checked release cone every frame instead of only at fire frames.
   This produced an incorrect numerical answer (98.81% replica-vs-slot0
   agreement under wrong premise) that was reversed when the bug was found
   and the analysis re-run. Lesson: when implementing a Python mirror of C
   code, verify by running both on identical input and confirming bit-level
   output.

3. **Misframing the engine as "fire-based" vs. "cone-based".** The engine's
   slot 0 is determined by sticky + cone-aware sort; fire is only the trigger
   for sticky updates, not the source of slot-0 ordering between fires.
   Lesson: be precise about which sub-mechanism is invoked by which signal.

### 7.3 Tests that did NOT work

- **Frag-anchored attribution as ground truth.** Frag updates contain only
  the killer's slot, not the victim. Cross-referencing with entity-death
  events to derive victim pid is fiddly and incomplete (splash kills,
  bystander kills, delayed projectiles confound the attribution).

- **Damage-event extraction.** Quake demo's `svc_damage` packet was
  considered as a hit-attribution signal but requires meaningful additional
  parser code (currently `skip_damage`). Deferred as future work.

- **Test 3 ("did demonstrator return to target after vis loss?").**
  Tautologically true — that's how the labeler's pass-2 continuity check
  already works. Discarded as uninformative.

### 7.4 Tests that DID work

- **Angular-velocity discrimination** (Section 2.4): cleanly separates
  artifact switches from real switches without requiring ground truth.

- **3D separation analysis** (Section 2.3): showed the geometric mechanism
  driving artifacts.

- **Engine relaxation head-to-head test** (Section 5.4): conclusively
  demonstrated that the modality filter is load-bearing.

- **Example-segment dumps**: surfaced over-hold failures that aggregate
  metrics missed (Section 3.3, K=2 cap=45° choice).

---

## 8. Post-deployment analysis (May 2026)

After the opt3 labeler + engine merged, two follow-up investigations
sharpened the picture: a learned-model probe of the residual 2.5% gap,
and a re-collect with all four token pools (not just PVS actors).

### 8.1 Learned probe — can a causal model close the gap?

Built a standalone causal target-prediction probe to isolate the 16-way
slot classification problem from the full BC trunk
(`qnn/labeler/probes/target_head_probe.py`). Architecture: per-slot scalar
MLP → flat per-frame vector → 7-layer dilated causal TCN (kernel 3,
channels 128, RF=127 frames ≈ 1.8 s @ 70 Hz), 369k params total.
Trained on the full opt3-relabeled QWD corpus with slot-0 down-weighted
to 0.15 to escape the 97.5% majority-class prior. 6 epochs on AMD
Radeon 8060S (~12 min/epoch, bf16).

**Argmax results (val, 683,504 labeled frames):**

| operating point | bucket A | bucket B | overall |
|---|---|---|---|
| strict-engine baseline (always slot 0) | 100.0% | 0.00% | 97.43% |
| probe argmax | 94.12% | 84.21% | 93.87% |
| probe + τ=0.9 threshold-override | 99.80% | 15.73% | 97.63% |

The threshold-policy ("predict argmax when softmax≥0.9, else default to
slot 0") gives +0.20 pp over baseline at τ=0.9, correctly recovering
~650 of the ~17.5k disagreement frames without touching bucket A.

**Per-feature ablation** (run with the trained model, zeroing one feature
group at a time, measured at the τ=0.9 operating point):

| ablation | override accuracy | overrides |
|---|---|---|
| baseline | 67.11% | 4,120 |
| ablate recency | 66.07% | 4,267 |
| ablate self_scalars | 66.78% | 4,061 |
| ablate fire | 62.78% | 4,659 |
| ablate look | 61.38% | 5,683 |
| **ablate rel** | **29.92%** | 5,447 |
| **ablate slot_enemy** | **77.78%** | **234** (1.04% B) |
| ablate slot_scalars (all) | 30.99% | 1,536 |

Two features carry confident overrides: the per-slot **rel** vector and
the **enemy-mask** flag. Recency, look, fire, self-scalars are
essentially noise at the high-confidence operating point. The bucket B
split is also informative — model gets 89.4% on "lookahead" frames
(label_pid in SIGHT *now*, recency=0) and only 74.2% on "modality"
frames (label_pid in non-SIGHT modality). The TCN's 1.8 s receptive
field exceeds the SOUND-modality window only when the gap was short;
longer occlusions fall outside.

The conclusion from the probe: the residual 2.5% gap is fundamentally
information-limited, not capacity-limited. Every architecture (MLP,
transformer, TCN) saturates the same ceiling because the per-frame
features genuinely don't distinguish the labeler's choice in those
cases. The labeler is using temporal information (past fires for
sticky, future fires for lookahead) that simply isn't in the
observation.

### 8.2 4-pool re-collect — the 97.5% baseline was filter-inflated

The opt3 collect used `--entity-filter pvs_actors` at collect time,
which dropped all projectile/item/mover tokens and all non-SIGHT actor
emissions before saving shards. Re-collecting the same 1,672 demos
with the filter removed (now `--entity-filter all`, since superseded by
train-time `token_mask`) produces identical per-demo statistics (same
1,672 demos, 1 skipped, 19 errors, 30,991,688 train ticks, 3,406,112
val ticks — to the row) but a **very different label distribution**.

| corpus | val labels | slot 0 share |
|---|---|---|
| original (pvs_actors filter, opt3 labeler) | 683,504 | 97.43% |
| new (4-pool, default labeler) | 844,723 | 89.10% |
| new + train-time mask `type=ACTOR, modality=SIGHT` | 766,857 | 90.62% |
| new + `--sight` labeler + train-time mask + compaction | 683,419 | 97.43% |
| new + sight + recency=0 filter + compact | 515,036 | 98.39% |

What changed:

1. **The labeler can now track pids through SOUND modality.** In the old
   collect, the pvs_actors filter stripped SOUND actors from the obs
   before the labeler ran; the labeler's stream-loss check broke at
   modality transitions and engagements terminated. In the new collect
   the labeler sees the full obs, so engagements extend through
   transient SIGHT loss. That adds ~161k labels to the val set.

2. **Those new labels are predominantly bucket B.** When the labeler
   tracked a pid through SOUND modality, the engine's strict sticky
   often dropped to a different pid. The new label points to a slot
   that's not slot 0 → bucket B. ~75k of the ~161k extra labels are
   slot ≠ 0, which is why the apparent "engine baseline" dropped 8 pp.

3. **The original 97.43% baseline relied on the collect-time filter
   hiding these frames.** The filter dropped the very frames where the
   engine and labeler would disagree most. Reproducing the old number
   requires both `--sight` on the labeler (drops SOUND-modality
   engagement extensions) AND train-time compaction (compacts kept
   slots so the post-mask "slot 0" matches what the old filter
   produced). Without compaction, the SIGHT-only baseline is 93.93%
   because the engine sticky persists at a SOUND-modality actor's slot
   even when masking removes it from the visible set.

This doesn't invalidate the opt3 design — the labeler↔engine sticky
state machines are still aligned. It does say the published "97.5%"
should be read as "97.5% on the strict-SIGHT-only subset", not "97.5%
on all engaged frames". The honest 4-pool number is **89.10%**.

### 8.3 GBT probe — how much of the gap is per-frame learnable?

Same per-frame features as the TCN probe, but a LightGBM multiclass
classifier (615 features, 16 classes, ~500 boosting rounds with early
stopping) over 500k–2M sampled labeled frames. Configured as either
**default** (per-frame natural slot order preserved) or **randomize**
(per-frame uniform permutation of all 16 slots, target slot remapped).

#### 8.3.1 4-pool data, strong-default GBT (slot0_weight=1.0)

| metric | value | vs baseline |
|---|---|---|
| overall | 95.14% | +6.04 pp over 89.10% |
| bucket A | 99.19% | −0.81 pp (model trusts slot 0 less in noisy frames) |
| bucket B | 62.03% | +62.03 pp (recovers many disagreement cases) |
| τ=0.9 acc / coverage | 98.71% @ 82.9% | practical operating point |

A class-weight sweep (slot0_w ∈ {0.15, 0.3, 0.5, 0.7, 1.0, 2.0}) found
the overall-accuracy peak at w=0.7 (95.30%), but the curve is flat
across 0.5–1.0 (within 0.2 pp). Up-weighting slot 0 (w=2.0) hurts;
strongly down-weighting (w=0.15) trades 4 pp of overall for higher
bucket B coverage. **Net: a 132-tree LightGBM trained in 97 seconds
matches or beats a 369k-param TCN trained for ~70 minutes on this
task.**

Top features by gain (default mode): per-slot `rel` (x, y, z
components), `eta`, `facing`, `enemy_mask`, `vel`. Recency contributes
but is not in the top 5. Same picture as the TCN ablation — the
problem is dominated by spatial geometry on actor candidates.

#### 8.3.2 Randomize mode — how much is slot-position prior worth?

| metric | default (4-pool) | randomize (4-pool) | Δ |
|---|---|---|---|
| overall | 95.14% | 93.75% | −1.4 pp |
| bucket A | 99.19% | 96.13% | −3.1 pp |
| bucket B | 62.03% | 74.25% | +12.2 pp |

The slot-position prior is worth ~1.4 pp overall. Most of that is
bucket A where the prior "predict slot 0" is a powerful shortcut.
Randomize mode actually *improves* bucket B because the model can't
shortcut and has to use real per-frame discrimination.

**Per-original-target-slot breakdown in randomize mode** (revealing
what the per-frame features can and can't distinguish):

| original target slot | val frames | randomize-mode accuracy |
|---|---|---|
| 0 (engine + labeler agreed) | 665,922 | 97.49% |
| 1 (engine + labeler disagreed) | 16,685 | **47.65%** |
| 2 | 875 | 17.26% |
| 3 | 21 | 38.10% |

When the labeler picked the unambiguous primary engagement target
(slot 0 cases), GBT identifies it from features alone 97.5% of the
time. When the labeler picked a "secondary" enemy that the engine
sticky missed (slot 1 cases), GBT can only identify it 48% of the
time — confidence on these is also flat (correct mean 0.74 vs incorrect
mean 0.72, gap 0.02). The per-frame features genuinely do not encode
which of two similarly-positioned enemies the labeler will pick.

#### 8.3.3 Implication — the engine + labeler agreement is structural

The randomize result confirms the framing in §6.1: the labeler-engine
2.5% (or 10.9% under 4-pool) disagreement isn't fixable from
per-frame causal features. Even an oracle-trained classifier given
only the current frame's data can't reliably distinguish the labeler's
"correct" pid from a co-positioned distractor — because the
distinguishing signal lives in past engagement history (which only the
labeler-side sticky has access to via fire times) and future fires
(lookahead).

This validates the architecture: the engine's sticky-with-cone
algorithm is making the best causal-state-only call. Three independent
algorithms — the engine's C state machine, the Python labeler running
forward, and a from-scratch LightGBM — converge on the same answer in
~95% of frames. The remaining 5% reflect a real information limit, not
a design defect.

### 8.4 Implications for production training

- **The 97.43% baseline in §5 was a measurement artifact** of the
  collect-time pvs_actors filter. The honest engine baseline on the
  full 4-pool stream is 89.10% on the post-opt3-labeler corpus. The
  "missing" 8 pp are real disagreements the filter hid.
- **The collect-time entity_filter flag has been removed.** Replaced
  with `token_mask` (parallel to `segment_mask`) in the BC train.json,
  so a single 4-pool collect serves all ablations. Equivalent of the
  old pvs_actors filter:
  `{"type": 1, "pid": {"$gt": 0}, "modality": 0}` (SIGHT only;
  actors never get PROXIMITY).
- **A `--sight` flag was added to the labeler** for callers who need
  the old engagement-break-on-SOUND-loss behavior. Combined with the
  token_mask above and slot compaction at load time, this reproduces
  the original 97.43% baseline byte-for-byte.
- **Slot-position ordering is worth ~1.4 pp** — small enough that
  dropping forced ordering at emit (letting actors appear in
  entity-number order with no pool/sticky-aware sort) is worth
  considering. The trade is stable per-pid slot identity (good for
  GRU/TCN per-slot state) at the cost of the slot-0-as-target prior.

## 9. Conclusion

The pre-existing labeler attributed targets via pure cone-argmax at each
fire, producing 40,726 within-segment pid switches dominated by A↔B
oscillation patterns on co-angular enemy pairs. Statistical analysis
(angular velocity, 3D separation) confirms these switches are noise, not
demonstrator-intent target changes.

The opt3 design — sticky fire-anchored attribution with adaptive trans_208
acquire cone and trans_416 cap_45° release cone — reduces switch count by
54.4% while preserving label frame coverage (within 0.5%). The same logic
was mirrored into the engine's `qnn_oracle.c` sticky state machine, bringing
labeler↔engine pid agreement to 97.56% on labeled overlap frames.

The remaining 2.5% gap is structurally lookahead-driven (~65%) and
modality-driven (~35%), both of which represent legitimate learned-target
signals: maintain through brief vis loss, and anticipate transitions before
the fire happens. Closing either further would require either lookahead in
the engine (impossible causally) or degrading label accuracy.

A controlled head-to-head test confirmed the engine's modality filter is
load-bearing despite appearing to be a legacy restriction: relaxing it
produces 60× more disagreements with labels than it fixes, because the
filter forces re-engagement at boundaries that align with the labeler's
lookahead-driven boundaries.

The design space was swept comprehensively (>20 configurations across 5
axes) and every parameter is empirically justified. Two failed
alternatives (cos-margin transfer, occlusion lock) were rejected with
concrete metrics showing their failure modes.

The labeler design and engine binary are in place. The next observable
test is a BC ablation comparing target-head accuracy on opt3 labels vs the
prior labels.

---

## Appendix A: Final configuration

```python
# qnn/bc/target_labeler.py
ACQUIRE_TRANSVERSE_U = 208.0
RELEASE_TRANSVERSE_U = 416.0
_ACQUIRE_CAP_COS   = cos(30°)
_ACQUIRE_FLOOR_COS = cos(5°)
_RELEASE_CAP_COS   = cos(45°)
_RELEASE_FLOOR_COS = cos(5°)

# pass 1 (causal mirror of engine):
current_pid = 0
for t in fire_ticks:
    # Stream-loss release on between-fire transient vis loss
    if current_pid > 0 and pid was absent in any frame [prev_fire+1, t-1]:
        current_pid = 0
    if no enemy at frame t:
        current_pid = 0; continue
    if current_pid is in obs and cos >= release_cone(d):
        attribute fire to current_pid    # sticky-keep
        continue
    # Acquire (per-enemy adaptive cone)
    admit = enemies with cos >= acquire_cone(d_each)
    if admit empty: current_pid = 0; continue
    best_slot = argmax cos over admitted
    current_pid = pids[best_slot]
    attribute fire to current_pid
```

```c
// engine/common/qnn_oracle.c
#define QNN_ACQUIRE_TRANSVERSE_U  208.0f
#define QNN_RELEASE_TRANSVERSE_U  416.0f
#define QNN_ACQUIRE_CAP_COS       0.8660254f   // cos(30°)
#define QNN_RELEASE_CAP_COS       0.7071068f   // cos(45°)
#define QNN_CONE_FLOOR_COS        0.9961947f   // cos(5°)

// at each frame:
//   - candidate loop applies modality filter (SIGHT/PROXIMITY only) for
//     sticky retention; releases if sticky pid not found in filtered set
//   - on fire press:
//       - if sticky still in release cone: sticky-keep
//       - else: release; if any candidate in acquire cone, acquire it
```

## Appendix B: Commits

```
683e4852  target labeler + engine sticky: adaptive cone + Schmitt-trigger release (opt3)
87b7b535  scripts: in-place target re-labeling with backup sidecar
256a8391  target labeler: release sticky on between-fire stream loss (bugfix)
740f6da6  bc: move entity_filter from collect-time flag to train-time token_mask
73350bc4  bc: doc fix — actors never use PROXIMITY modality
<pending>  bc: --sight flag on target labeler for SOUND-modality-break behavior
```

## Appendix C: Data artifacts

- The opt3 collect (post-engine-change, pre-4-pool) was at
  `artifacts/collect/qwd/`, fingerprint
  `d041710d839afca0eb664a9d3975654a4a21467040e67a0875219e3e4cb6809f`.
  This collect baked `entity_filter=pvs_actors` into the shards and is
  the source of the §5 97.43% number.
- The current 4-pool collect (same 1,672 demos, no entity filter at
  collect time) is at `artifacts/collect/qwd/`, fingerprint
  `54a09eb143fd1bd141819e84e75396412dd7a0a26ad664a171e5955884d54684`.
  Filter config: `artifacts/collect/archive/qwd/filter.json` (drops
  CTF/TF/spectator demos).
- Original (pre-opt3) labels from the d041710d collect are preserved as
  `*_act_target_orig.npy` sidecars; the slot-pid map from the
  pre-engine-change archive is at
  `artifacts/collect/archive_qwd_slot_pid_map/`.

## Appendix D: Validation scripts

The original analysis scripts (§2-6 metrics) live in `/tmp/`:

- `test4_relaxed_vs_strict.py` — §5.4 head-to-head engine relaxation test
- `diag_modality_share.py` — §6.1 modality vs lookahead split
- `diag_fire_disagreement.py` and `diag_cascade.py` — fire-frame
  disagreement causes
- `analyze_switch_angvel.py` — §2.4 angular-velocity discrimination

The §8 post-deployment scripts are checked in:

- `qnn/labeler/probes/target_head_probe.py` — causal TCN target-prediction
  probe (§8.1)
- `qnn/labeler/probes/target_head_diag.py` — feature-ablation diagnostics on
  trained TCN
- `qnn/labeler/probes/target_head_gbt.py` — LightGBM target-prediction probe
  (§8.3); supports `default` and `randomize` modes plus `--no-masks` and
  `--token-mask` knobs
- `qnn/labeler/probes/target_head_gbt_eval.py` — per-original-slot eval of
  randomize-mode GBT (§8.3.2 table)

The 4-pool re-collect baseline tables in §8.2 were produced by ad-hoc
scripts that loaded the precomputed shards, applied the relevant
filter/compaction, and re-ran the v2 hard-label `label_enemy_target`
that previously lived in `qnn/bc/target_labeler.py` (since removed in
favor of `label_enemy_target_probs`). The `--sight` flag on the v2
labeler was the in-code knob that reproduced the SOUND-break behavior.

Re-running any of these requires the existing precomputed shards on
disk at `artifacts/collect/qwd/precomputed_{train,val}/`.
