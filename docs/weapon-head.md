# Weapon head: findings & design

What drives the held weapon in the `qwd` corpus, why per-frame weapon classifiers
plateau, why the full-encoder probe overfits, and the switch-process head the data
actually calls for. Numbers measured on `artifacts/collect/qwd` (val split unless
noted). Regenerate with the `scripts/analysis/_weapon_*.py` diagnostics referenced
per section.

## TL;DR — the reframe

Weapon selection is **not** a per-frame "right weapon for the job" choice. It is a
**hazard-driven cycling process**: in combat you keep your weapon ~85% of frames,
switch ~4% of frames, and *when* you switch is driven by **how long you've held it
+ how many weapons you own** (not geometry), while *what* you switch to is driven by
**inventory + the weapon you're leaving** (not geometry). The long-standing
per-frame objective threw away the persistence/duration structure that *is* the
process — which is why every model plateaus at macro-F1 ~0.49 and why a high-capacity
encoder overfits trying to reconstruct persistence from an episode fingerprint.

## 1. Bench form sweep (idealized scaffold)

Scaffold: `PreAttnEncoder` passthrough + `GTTargetPointer` (oracle) + no temporal;
head reads named obs tokens directly. Segment `act.target!=0`, input_mask=true,
n=1,014,088. **This is not a deployable head** — oracle target, no real encoder, no
temporal — it only measures what the named features carry under ideal conditions.

| form (3-seed mean where noted) | macro_f1 | micro_acc | nll |
|---|---|---|---|
| arsenal only | 0.4718 | 0.5789 | 0.9064 |
| arsenal + state | 0.4774 | 0.5817 | 0.9117 |
| arsenal + motion (3 seeds) | **0.4925** ±0.004 | 0.598 | 0.893 |
| arsenal + motion + target (3 seeds) | 0.4825 | — | — |
| dense (full self readout) | 0.4808 | 0.6262 | 0.8709 |

Reads: motion helps (+0.024 over arsenal); state ≈ dead weight; **target *hurts*
macro by ~0.010 paired across seeds** (it's an "LG-viability gate" — reshapes RL↔LG
but nets lower). dense's micro/NLL edge is partly the `attack_finished` soft
incumbent leak. (`_weapon_switch_threshold_eval.py` grades; sweep in `runs/head_probe/`.)

## 2. Full-encoder probes + the overfit mechanism

`weapon_cls_transformer` puts the weapon head on the **real** CLS attention encoder
(token stream `[CLS, state, arsenal, motion, spatial×9, entity×16]`, weapon token
dropped). Trained at lr 0.003 / 50 ep (move_cls recipe).

- `cls_xfmr` (no temporal): **overfits** — val NLL rises monotonically from epoch 1
  (train 0.87↓, val 0.90→1.24↑), best epoch 2. Graded best: macro 0.496 / nll 1.01.
- `cls_gru` (+GRU temporal): healthy (val→0.87), macro 0.4925, micro 0.626, nll 0.873
  — matches dense's bulk *legitimately* via temporal, but only ties the bench form on
  macro. Plateaus by ~epoch 11.

**Not a bug** (move_cls on the identical recipe is healthy). The overfit needs
*memorizable surface × episode-persistent label × no generalizing shortcut* — the
unique `cls_xfmr` cell. Readout dissection (`_weapon_readout_probe.py`, hooks
`weapon_head.mlp` input):

| readout | dim | weapon F1 (within-val) | held-weapon acc | episode-id acc |
|---|---|---|---|---|
| arsenal+motion | 128 | 0.31 | 0.51 | 0.58 |
| cls_xfmr | 64 | 0.75 | 0.66 | **0.97** |
| cls_gru | 64 | 0.74 | 0.71 | **0.99** |

The CLS readout encodes a near-complete **episode fingerprint** (epID 0.97-0.99) and
reconstructs the *dropped* held weapon. Token attribution (`_weapon_token_attribution.py`,
zero a group, re-measure epID): the fingerprint is carried **mostly by the self
tokens** (Δepid −0.25; continuous health/armor/velocity/readiness), not entities
(−0.05) or spatial (−0.01); it's redundant (no_self floor 0.72 vs arsenal 0.58). For
`cls_gru` the fingerprint is temporal (robust to per-frame token ablation).

## 3. Corpus structure — why

`_weapon_corpus_stats.py`, `_weapon_combat_vs_full.py` (full val stream vs combat =
`argmax(target_probs)!=0`):

| stat | FULL (3.9M) | COMBAT (972k) |
|---|---|---|
| switch rate (P switch\|frame) | 1.89% | **4.03%** |
| persistence H(w\|prev) explains | 91.4% | **84.9%** |

Instantaneous determinants of the held weapon (combat, ΔH = bits explained of
H(weapon)=1.95): **inventory (owned_set) 34.1%**, n_owned 8.9%, **target_distance
0.9%**, health 1.1%. Per-weapon target distance is only weakly ordered (nailguns/SSG
closer, rocket/grenade farther) with massive overlap — geometry is ~noise. Dwell
medians 3-21 frames (rocket/SSG/grenade short bursts; shotgun/nailgun/lightning long).

### Switch events (`_weapon_switch_events.py`, combat)

- **96.4% voluntary** (had the weapon, old one still had ammo); only 2.6%
  forced-ammo + 1.0% pickup. (NB: this undercounts forced vs the
  [weapon-switch-source backlog](../../agents/plans/weapon-switch-source-classification.md),
  which flags ammo auto-switch / pickup as a large share of *transitions* — our
  forced detector requires from-weapon ammo == 0 at the exact switch frame and likely
  misses low-ammo / timing-offset auto-switches. Worth reconciling.)
- Switch **target** is predictable: H(new | from+owned) = 0.48 bits (76% of the
  2.04-bit target entropy). Targets dominated by shotgun (39%) + rocket (36%).

### WHEN triggers (`_weapon_when_triggers.py`, combat, 3.96% switch-next)

Logistic probe on observables predicting switch-next: **AUC 0.730** — switches are
predictable, not irreducible noise. Per-feature univariate AUC:

| feature | AUC |
|---|---|
| dwell_age | 0.694 |
| n_owned | 0.634 |
| held_weapon (which gun) | 0.577 |
| held_readiness / distance / dist_delta / health | 0.50–0.53 (≈chance) |

Switch *timing* is a **duration/hazard + inventory** process. Geometry, ammo, and
health do not trigger switches. The predictive features (dwell-age, inventory,
held-weapon) are **generalizing** — not the episode fingerprint — so a head built on
them should not fingerprint-overfit.

## 4. Design — switch-process weapon head

Model weapon as the process it is, not per-frame classification:

- **WHEN head** — a dwell-age **hazard** modulated by `n_owned` + held-weapon →
  P(switch this frame). Requires tracking dwell-age (temporal/stateful). Handle the
  ~4% positive rate with focal / pos_weight (as the attack head does). Viability
  ceiling ~AUC 0.73 (linear; nonlinear/temporal may exceed).
- **WHAT head** — `inventory + from-weapon` → P(new weapon | switch), trained on the
  ~39k combat switch events. Distance is noise; do not wire it in.
- **Inputs that matter are the ones we'd been excluding as "leaks":** held weapon +
  temporal state are the *signal*, not contamination.
- **Eval on distributions**, not per-frame macro_f1: dwell-time / switch-rate
  fidelity (EMD/MMD) + switch-target accuracy. Per-frame macro_f1 ≈ 0.49 is the
  irreducible situational ceiling and is the wrong objective.

See [persistence-and-changepoints.md](persistence-and-changepoints.md) — this is the
same WHEN-hazard + WHAT-residual structure found for look/move.

### Offline validation (milestone 1, `_weapon_switch_head.py` / `_weapon_when_hazard_eval.py`)

Tiny heads on frozen generalizing features (combat):
- **WHAT** (new weapon | switch): 79.6% acc / 0.558 macro-F1 (vs 40.6% baseline) — learnable.
- **WHEN** (MLP hazard): AUC 0.783; only 17.5% per-frame precision/recall at base-rate
  threshold (timing is stochastic — rank, don't pinpoint). But a *calibrated* hazard
  (plain BCE) reproduces the distribution: overall switch rate 3.98% vs 3.96% actual,
  and the empirical hazard h(age) — negative duration dependence, 8% at age 0-2 → 0.3%
  at 81+ — is tracked closely (reliability deciles on the diagonal). Since the dwell
  distribution is determined by h(age), the form is validated. **Eval distributionally,
  never on per-frame switch precision.**

### In-pipeline integration (BC trainer, `weapon_switch` bench head)

The switch-process head now trains end-to-end through the BC pipeline (bench head
`weapon_switch`, PreAttn encoder, arsenal token + from-weapon embed for WHAT,
dwell-age bucket-embed for WHEN). Two bugs fixed to get there:

1. **OOB cross_entropy target (was a GPU "hang").** A switch *to weapon 0 (none)*
   gave WHAT target `0-1 = -1`, not `ignore_index=-100` — an out-of-bounds class.
   On ROCm the OOB gather wedges the GPU command queue asynchronously, so the CPU
   spins in the HIP queue-full busy-wait (300% CPU, ~40 steps in — switch-to-none
   is rare). Fix: WHAT target valid only on a switch into a real weapon 1..8
   (`derive_weapon_switch_labels`). The corpus contained these frames (guard fired
   ~102×/epoch).
2. **Dropped metrics.** `supervised_loop` only reduces sum-prefixed or
   averaged-prefixed metric keys into epoch history; `what_acc`/`when_*_rate` matched
   neither and vanished. Renamed to `acc_weapon_what`, `pred_rate_weapon_when`,
   `pos_rate_weapon_when`.

#### VERDICT: VALIDATED in-pipeline (with precomputed labels)

With precomputed per-frame labels the head reproduces the offline result in the real
BC pipeline (run `head_probe_weapon_switch_precomp_seed17`, 6 ep, detached docker,
QNN_ALLOW_FINGERPRINT_MISMATCH=1):

| metric | in-pipeline (precomputed) | offline ceiling | garbage-label (batch-derived) |
|---|---|---|---|
| WHAT acc | 0.787 train / 0.71–0.72 val | 0.794 | 0.619 (stuck) |
| WHEN hazard mean | 0.0395 ≈ 0.0396 actual (calibrated) | ~0.04 | 0.48–0.65 (diverged) |
| WHEN BCE | 0.151 stable | — | 0.66→1.88 rising |

WHAT recovers the full from-weapon+inventory signal (78.7% train, ~72% held-out val vs
40.6% baseline); WHEN calibrates to the 4% base rate with stable low BCE. **The
switch-process head approach works in-pipeline.** Every earlier in-pipeline failure was
purely the label-derivation bug below — not the head, not pos_weight, not optimization.

#### How it was blocked (root cause)

First in-pipeline runs looked broken — WHAT plateaued at 61.9% (vs 79.4% offline),
WHEN would not calibrate (mean hazard ~0.5, rising). Tracing it to ground truth (step
prints) revealed the actual cause, which supersedes the earlier "inventory-pathway" /
"WHEN-optimization" guesses:

**The bench scaffold trains non-temporally (`sequence_length` irrelevant — the model is
`temporal=Off` → `use_gru=False`), so `supervised_loop` delivers data via
`frame_shuffled_batches`: individual frames, SHUFFLED, with NO time axis and NO
`reset_mask`** (the lane-packed `(T,B)` + `reset_mask` path is GRU-only). At the
side-channel the weapon arrives as a flat `(N,)` with `masks=None`. So
`derive_weapon_switch_labels`, which needs temporal `(t, t+1)` neighbors, was computing
"switches" between *unrelated shuffled frames* — a **64.5% switch rate** (vs the true
4%) and dwell-ages capped at ~10. The heads were training on garbage:
- WHEN can't fit a 65%-positive noise label → stuck near init / drifts up (NOT a
  pos_weight or optimization bug — the exact head structure calibrates perfectly in
  isolation, `_when_head_isolation.py`: bf16 lr=0.0045 → mean pred 0.018 ≈ base 0.0184).
- WHAT only recovers the from-weapon marginal (61.9%) from scrambled targets.

`derive` was rewritten to compute along the time axis of a `(T,B)` batch, but that only
helps a *temporal* delivery path. **The correct fix is to PRECOMPUTE the per-frame switch
labels (`dwell_age`, `switch_next`, `new_weapon_target`) per-episode at data-prep time
(via `episode_offsets`, exactly as the offline probe does) and carry them as action
columns** — then they survive frame-shuffling, like `target_probs`. Batch-time derivation
is the wrong layer for a non-temporal, shuffled pipeline.

**Bottom line:** the switch-process head HAS value — offline it cleanly beats baselines
(WHAT 79.4% vs 40.6%; WHEN AUC 0.78, calibrated to 4%) and the head structure trains
correctly given correct labels. The only thing standing between that and an in-pipeline
result is precomputed per-frame labels. Until those exist, `weapon_switch` bench numbers
are not meaningful.

Two real bugs were also fixed en route (both still valid): the OOB `cross_entropy` target
(above) and the dropped distributional metrics (above).

## 5. Apples-to-apples: per-frame held-weapon agreement (dense vs switch)

The switch fork was validated on its *own* objective (WHAT acc, hazard calibration).
But the demonstrator-fidelity question is narrower: *does the agent hold the weapon the
demonstrator held, frame for frame?* That is a per-frame held-weapon classification
metric, and it is the dense classifier's home turf. To compare fairly, both heads are
reduced to a per-frame held-weapon trajectory on the **same** val combat segment
(`act.target!=0`, n=1,014,088) and scored identically. The dense head emits it directly
(`argmax P(weapon|obs)`); the switch head is a transition kernel, so it is **free-rolled**
(start from the true initial weapon; each frame `switch ~ Bernoulli(σ(WHEN(dwell)))`, on
switch `wb := argmax WHAT(arsenal, from=wb)`). Teacher-forcing the from-weapon every frame
would read the answer off `t-1` and is used only as a plumbing check (reproduces WHAT
≈0.79 / hazard ≈0.045). Grader: `_weapon_dense_vs_switch.py`.

| approach | macro-F1 | micro-acc | weighted-F1 | commit rate |
|---|---|---|---|---|
| persistence ceiling (copy prev true weapon) | 0.958 | 0.961 | 0.961 | — |
| **CLS encoder + GRU + weapon token** — gated (C=0.59,M=0) | **0.798** | **0.855** | — | **92%** |
| CLS encoder + GRU + weapon token — raw | 0.795 | 0.852 | 0.852 | 100% |
| CLS encoder + weapon token, **no GRU** — gated (overfits, best ep0) | 0.736 | 0.799 | — | 66% |
| **held-weapon-input dense, self-embed** — gated (C=0.74,M=0.6) | 0.756 | 0.816 | — | 51% |
| **held-weapon-input dense, split tokens** — gated (C=0.30,M=0.57) | 0.739 | **0.825** | — | 51% |
| held-weapon-input dense, self-embed — raw | 0.728 | 0.747 | 0.746 | 100% |
| held-weapon-input dense, split (weapon+arsenal+motion) — raw | 0.726 | 0.742 | 0.743 | 100% |
| no-held-weapon DENSE + conf-gated hold (C=0.95, M=0) | 0.694 | 0.811 | — | **2.1%** |
| no-held-weapon DENSE raw per-frame argmax | 0.493 | 0.626 | 0.606 | 100% |
| no-held-weapon DENSE + prior-adjust (τ=0.4) | 0.532 | 0.612 | — | 100% |
| SWITCH free rollout (5 seeds) | 0.439 ±0.004 | 0.509 ±0.003 | 0.517 ±0.003 | ~5% |
| always-shotgun | 0.086 | 0.525 | 0.362 | — |

The **conf-gated hold** row is the prior "extensive threshold/confidence testing"
(`_weapon_threshold_opt_*.json`, 182-cell coarse + 60-cell fine `(C,M)` grid per variant):
the runtime commits `argmax` only when `conf≥C ∧ margin≥M`, else holds the last committed
weapon. Across all 12 weapon variants raw macro-F1 clusters at 0.47–0.50 but the gated
hold stream lifts every one to ~0.69 — at a 0.1–2.2% commit rate. The gain is almost
entirely *holding*: trust argmax only on the rare near-certain frame, otherwise keep what
you've got. That gate is itself a hand-built hold/switch process layered on the dense
head, and it is the strongest *no-held-weapon* per-frame tracker on record.

### Feeding the held weapon as input wins outright (the "both" result)

Retrained in-distribution on the current corpus (`weapon_token_d64_h16_repro` =
held weapon **embedded in the self readout**; `weapon_arsenal_motion_weapon_token` =
held weapon as its **own split token** alongside arsenal + motion). Both reproduce the
May-20 numbers and **beat the no-held-weapon dense by ~0.23 macro-F1 raw**:

- **Raw** (always commit): macro 0.726–0.728, micro 0.742–0.747 — already above the
  no-held-weapon dense's *gated* 0.694, with no hold crutch.
- **Gated**: macro 0.74–0.76, acc 0.82, at a **51% commit rate** — vs the no-held-weapon
  dense's 0.69 at **2.1%**. The held-weapon model trusts and commits its own argmax on
  half the frames; the no-held-weapon model only dares 2% and rides persistence for the
  rest. Gating barely helps it (0.728→0.756) because the raw output is already strong.
- **Marginal**: shotgun *under*-predicted (self-embed 47.2% / split 40.0% vs 52.5% true —
  the +6.3pp over-prediction is gone) and the starved rare guns recover: axe F1 0.53
  (vs 0.034), grenade 0.66 (vs 0.328). Per-frame agreement **and** a faithful distribution,
  from the single change of feeding `w_{t-1}`.
- **Delivery is a wash**: self-embed vs split-token (weapon+arsenal+motion) match within
  noise on every per-class F1 (split runs a touch cooler on shotgun). The held-weapon
  *signal* is what matters, not whether it's fused into the self readout or its own token.
  Sweeps: `_weapon_threshold_opt_head_probe_weapon_token_d64_h16_repro_seed17.json`,
  `_weapon_threshold_opt_head_probe_weapon_arsenal_motion_weapon_token_seed17.json`.

This is the §6 transition-head thesis confirmed empirically: the held-weapon input is the
single lever that buys both objectives at once.

#### Temporal fidelity — switch rate + dwell-time (the *dynamic* distribution)

Per-frame F1 and the static marginal say nothing about chatter. The committed-stream
switch rate + dwell distribution vs the demonstrator (combat: 3.96% switch, dwell median
10, mean 18; `_weapon_temporal_fidelity.py`):

| self-embed operating point | switch% | dwell median | dwell EMD | macro-F1 |
|---|---|---|---|---|
| raw argmax | 4.82 | 9 | 2.49 | 0.728 |
| **light gate (C=0.30, M=0.05)** | **3.79** | **11** | **1.42** | 0.728 |
| macro-F1-optimal gate (C=0.70, M=0.60) | 1.10 | 27 | 18.96 | 0.756 |

At the raw / light-gate point the self-embed switch *rate* matches (3.79–4.82% vs 3.96%,
dwell median 11 vs 10, EMD 1.42), and the *macro-F1-optimal* gate **over-holds** (switch
1.1%, dwell median 27, EMD ~19) — maximizing per-frame F1 rewards "when unsure, hold." So
matching the switch-rate / dwell distribution is an operating-point (commit-gate) choice,
not a new head. (All teacher-forced on true `w_{t-1}`, which *is* the deployment condition —
the engine supplies the real held weapon each frame, so no compounding drift.)

**But matching the switch *rate* is not modelling switch *timing*** — decomposing the raw
argmax stream (`_weapon_switch_decompose.py`) shows neither head predicts *when* to switch:

| | self-embed | split (wpn+ars+mot) |
|---|---|---|
| mean P(incumbent) | 0.647 | 0.639 |
| pred switch rate | 4.82% | 2.87% |
| switch **recall** (of true switches) | 11.4% | 11.2% |
| hold-frame **chatter** (false switches) | 4.55% | 2.52% |
| switch **precision** | 9.4% | 15.5% |

Both catch the same ~11% of real switches; the rate difference is **pure chatter**
(self-embed flips spuriously on 4.55% of hold frames, split only 2.52%). So the self-embed's
"matching" 4.82% rate is mostly false switches that coincidentally sum near 4% — its dwell-EMD
win is partly luck, its switches land on real switch frames only 9.4% of the time. The split
is stickier because it chatters *less* (its switches are higher-precision, 15.5%), **not**
because it has a sharper incumbent signal (peakedness is near-identical). This is the §3
finding resurfacing: switch *timing* is near-irreducible (logistic AUC 0.73), and a per-frame
classifier can't pin it. The held-weapon heads excel at per-frame held-weapon agreement
(~0.75, riding the incumbent's autocorrelation); they do **not** predict when/what to switch.

**Prod implication:** prefer the lower-chatter head. The split's cleaner holds (2.5% vs 4.6%
spurious-flip rate) are more desirable than the self-embed's chatter-inflated switch rate;
match the human switch rate via the commit gate, not by tolerating chatter. Graders:
`_weapon_temporal_fidelity.py`, `_weapon_switch_decompose.py`.

**Tolerance-windowed switch timing — and does the WHEN head do better?** Exact-frame
recall/precision understates timing (it double-penalizes ±1-frame jitter). With a ±k window
(`_weapon_switch_decompose.py`, `_weapon_when_switch_detect.py`, all at the 3.96% human
alarm budget):

| ±k | self-embed dense | split dense | WHEN (dwell-only, AUC 0.715) |
|---|---|---|---|
| 0 | 11.4 / 9.4 | 11.2 / 15.5 | 12.7 / 12.7 |
| ±1 | 38.3 / 32.7 | 33.6 / 45.3 | 21.7 / 21.5 |
| ±2 | 56.1 / 43.5 | 54.2 / 60.9 | 26.6 / 26.0 |
| ±5 | 70.5 / 58.3 | 66.5 / 72.8 | 76.7 / 73.7 |

(recall / precision %.) Two reads: (1) the dense heads are **not** timing-blind — they
localize switches to ±2–3 frames (±2 recall ~55%); exact-frame 11% was harshness. (2) The
**deployed WHEN head does NOT beat them on timing** — tied at exact frame, *worse* at ±1–2
(dense localizes tighter), only catching up at ±5. Reason: the deployed WHEN is **dwell-age
only** — it *ranks* switch-prone frames (AUC 0.715) but can't pinpoint which frame inside a
dwell window switches, so alarms smear across just-switched frames; the dense heads see full
obs and flip closer to the real switch. The WHEN head's value is the calibrated hazard
*score* (reproduces switch-rate / dwell distribution), not exact timing. A richer WHEN MLP
(dwell + inventory + held-weapon, §4 AUC 0.783) would localize better than dwell-only, but
switch *timing* stays largely irreducible (§3). The split dense's higher precision at every
tolerance restates the chatter finding. (WHEN ±3 row jumps oddly — a dwell-clustering
artifact; don't lean on it.)

**Among the no-held-weapon heads, the gated-dense approach wins** — the switch
rollout (0.509 micro) is barely above always-shotgun (0.525). Cause: the switch head
switches at the right *rate* (0.0497 realized vs 0.0396) but the wrong *times* (stochastic
hazard, AUC 0.73); once its weapon diverges it stays diverged until the next switch, while
the dense head re-grounds in obs every frame. The 0.961 persistence ceiling (copy the
previous true weapon) shows how much of this task is pure "keep holding" — neither learned
head approaches it because both drop the held-weapon token.

What the fork *does* fix — but it doesn't move F1: the dense head over-predicts shotgun
(**58.8% pred vs 52.5% true, +6.3pp**) and starves rare guns (axe recall 1.8%; dumps
axe→sg 63%, gl→sg 55%). The switch head's from-conditioning balances the marginal (shotgun
49.6%, axe recall 30.6%, grenade F1 0.433 vs 0.328) — but the timing desync costs more on
the common guns (shotgun F1 0.586 vs 0.711) than the rare-class coverage gains back, so
overall F1 still trails.

**Verdict.** For "hold the right weapon most of the time" (per-frame F1), the heavy switch
fork does **not** beat the existing dense classifier; the gated-dense head is the better
tool and its only real flaw (the +6.3pp shotgun bias) is mild. The switch fork earns its
keep only on the *distributional* objective — switch-rate / dwell-time / weapon-marginal
fidelity and rare-gun coverage — which a per-frame dense classifier structurally cannot
satisfy (sampled per-frame it would chatter and over-shotgun). Per-frame agreement and
distributional human-likeness are different objectives; the fork optimizes the latter.

### Full encoder + held-weapon token — the overall winner

Putting the held-weapon token on the real CLS attention encoder (`weapon_cls_transformer`
`include_weapon_token=true` = HeldWeaponSplitObsEmbedding's 5th held-weapon subtoken),
with vs without the GRU temporal stream. Both retrained in-distribution; the no-GRU run was
plateau-killed at epoch 6 (best epoch 0).

| head | raw macro | gated macro / acc | commit | shotgun pred/true | dwell EMD | health |
|---|---|---|---|---|---|---|
| **CLS + GRU + weapon token** | **0.795** | **0.798 / 0.855** | **92%** | **52.7 / 52.5** | **0.55** | healthy (best ep22) |
| CLS + weapon token, no GRU | 0.713 | 0.736 / 0.799 | 66% | 49.2 / 52.5 | — | **overfits** (best ep0, val NLL rises) |
| dense held-weapon (self-embed) | 0.728 | 0.756 / 0.816 | 51% | 47.2 / 52.5 | 1.42 | healthy |
| no-token CLS+GRU (§2) | 0.493 | 0.694 / 0.811 | 2% | 58.8 / 52.5 | — | rides persistence |

**CLS+GRU+weapon token wins on every axis** — best per-frame (0.855 acc), commits **92%** of
frames (a confident predictor, not a hold crutch), a **near-perfect marginal** (52.7 vs 52.5,
bias gone), AND the **best temporal fidelity** (dwell EMD 0.55 vs the dense head's 1.42; switch
4.0% vs 3.96% at the matching gate; dwell median 10 = true) — the GRU sees history so it
chatters less. It supersedes the earlier "dense self-embed for deployment" call.

Two levers, both real and additive:
1. **Held-weapon token: +~0.23 macro** (no-token 0.49 → any token form 0.71–0.73). The dominant lever.
2. **GRU temporal: +~0.08 macro on top** (CLS+token no-GRU 0.713 → +GRU 0.795), and it's what makes the encoder *healthy* + temporally faithful.

The **attention encoder without GRU adds nothing** over the flat dense head (CLS+token no-GRU
0.713 ≤ dense 0.727) and still **overfits** even with the held-weapon token (val NLL rises
monotonically from epoch 0 — the §2 `cls_xfmr` signature; the token raises the epoch-0 peak
from 0.496 to 0.713 but doesn't cure the overfit). Depth without temporal isn't worth it.

#### Switch vs hold — the leak-free metric (pick ≠ equipped token = switch)

A *switch* is the head outputting a weapon it is **not** currently holding (`pred ≠
self_weapon_id`); outputting the equipped weapon is a **hold**. This is the leak-free framing:
on frames where the target differs from the equipped token (`act.weapon ≠ self_weapon_id`,
25.7% of combat), the token is the *wrong* answer, so the head **cannot echo it** — any
correctness there is genuine switch-decision skill (`_weapon_switch_vs_token.py`):

| frame type | GRU+token | dense self-embed |
|---|---|---|
| HOLD (target == token, 74.3%) | 93.8% | 95.2% |
| **SWITCH (target ≠ token, 25.7%) — leak-free** | **60.4%** | **15.4%** |
| SWITCH @ attack | **34.7%** | 8.8% |
| overall | 85.2% | 74.7% |

On genuine switch frames the GRU+token correctly commands the non-held weapon **60.4%** of the
time vs the memoryless dense head's **15.4%** — a clean **4× advantage with no leak** (token = 0%
there). The dense head, on switch frames, outputs the *stale held weapon* 80.5% of the time —
it barely switches; it holds/echoes. The GRU genuinely learns to call switches.

(Earlier drafts framed this as a frame-buffer "echo leak" using an argmax-change switch
definition — `pred[t]≠pred[t-1]`. That conflated *settling the hold onto a just-equipped weapon*
with *commanding a switch*; once a switch is defined as pick≠equipped, the metric is leak-free
and the GRU's advantage is real. There is no leak.)

The 60.4% is **per-frame (no buffer)**. Decomposing the switch frames by whether the engine
later equips the target (`_weapon_switch_leadtime.py`):

| switch-frame type | share | head acc | lead over engine |
|---|---|---|---|
| intent leads engine (target equipped within 40f) | 62.8% | 66.0% | median **7f**, mean 10, p90 25 |
| stale / auto-switch (target never equipped) | 37.2% | 51.0% | — |

So on the genuine in-flight switches (63% of switch frames) the head commits to the new weapon a
**median ~7 frames before the engine equips it** (≈0.35 s @20 Hz; only 12% within 1f — genuinely
ahead, not coincident), at 66% accuracy, while fed the *old* equipped weapon — real anticipation.
That lead ≈ the demonstrator's own key-press→equip latency: the head predicts `act.weapon`
(intent), which leads the engine by that much. The other 37% are stale-label / auto-switch frames
(rl→sg ammo-out etc.) where 51% "acc" is reproducing the lagging intent, not anticipation. So
~**40% of switch frames** (0.628×0.66) are genuine anticipated switches called ~7f ahead.

**Attack frames are the correctness break points** (the weapon only counts when you fire,
`_weapon_switch_timing_detail.py`): attack frames are **mostly holds** (you fire what you hold),
so HOLD@attack is 90–96% for both heads and dominates the attack-frame average (GRU 79.8 ≈ dense
79.6). The consequential subset is **switch-at-attack** — firing a weapon you're mid-switching
to — where the genuine gap shows: **GRU 34.7% vs dense 8.8%**. A hard, low-baseline task (the
demonstrator's own switch-to-fire is fast), but the GRU is 4× the echo-only head there too.

## 6. Getting both — the unified transition head

> **Status: CONFIRMED (§5), incl. temporal fidelity.** Feeding the held weapon `w_{t-1}`
> as input — by either delivery — gets both objectives at once (macro 0.73 raw, balanced
> marginal) AND, at the raw / light-gate operating point, matches the human switch rate and
> dwell distribution (3.8% vs 3.96%, dwell EMD 1.4). The temporal hold/switch head below is
> **not needed** — the dynamic distribution is an operating-point (commit-gate) choice, not
> a missing architecture. The reasoning that follows is what predicted the held-weapon
> lever; keep it as the rationale, but the question is settled empirically.

The two no-held-weapon winners are the same architecture seen twice: *hold unless a trigger
fires, then pick a new weapon.* They differ in two axes only:

| | trigger (WHEN) | new weapon (WHAT) | per-frame | marginal |
|---|---|---|---|---|
| gated-dense | conf-gate on dense argmax (re-grounds in obs) | dense argmax (shotgun-biased) | **0.81** | biased |
| switch fork | sampled dwell-hazard (desyncs) | from+inventory (balanced) | 0.51 | **balanced** |

Each owns one column. The gate's *timing* is good because the dense confidence spikes in
obs exactly when the weapon changes; the fork's *target* is good because WHAT conditions on
the from-weapon + inventory. The synthesis is to take the good column from each.

**The unit of both is the transition kernel `P(w_t | w_{t-1}, obs_t)`.** The dense head
*drops* the held-weapon token (so it reconstructs a noisy fingerprint → 0.49 raw); the
switch fork *uses* `w_{t-1}` but only as a generator (so it desyncs). The fix is to model
the transition densely, with `w_{t-1}` as an **explicit input** (the agent always observes
its own current weapon — deployable, not a leak), factorized hold-vs-switch:

> `P(w_t | w_{t-1}, obs_t) = P(hold)·δ(w_{t-1}) + P(switch)·P(w_t | from=w_{t-1}, inventory)`

Trained per-frame (dense CE, so it re-grounds and approaches the 0.96 persistence ceiling
on hold frames) but with the switch factor carrying the from-conditioned WHAT structure
(so the ~4% switch frames get balanced targets, fixing the shotgun bias and rare-gun
coverage). This is a per-frame WHEN×WHAT — same factorization as the fork, but evaluated
densely against the held weapon instead of generated, so timing locks to obs.

Two ways to get there, cheap → principled:

1. **Offline graft (no training, validates the hypothesis):** keep the dense conf-gated
   hold for *timing*, but when it commits a switch, route the new weapon through the WHAT
   head (or the prior-adjust debias) instead of raw argmax. Commits are ~2% of frames, so
   WHAT only fires on the switch-target frames — exactly where the marginal is set. Expect
   to retain ~0.81 acc while pulling shotgun share toward 52.5% and lifting rare-gun recall.
   Buildable now by recombining the two checkpoints we already have
   (`_weapon_dense_vs_switch.py` already has both forwards wired).
2. **Unified head (the real answer):** one head taking `w_{t-1}` as an explicit input, with
   a temporal stream for dwell, a hold/switch gate, and a from-conditioned switch target;
   dense per-frame loss + an auxiliary hazard/marginal term. Eval on BOTH per-frame F1 and
   the distributional metrics so neither objective is silently sacrificed.

## Label vs engine state (act.weapon ≠ self_weapon_id)

The BC target `act.weapon` is a **sticky weapon-select impulse** label (player intent,
ownership-gated, built in `QwdBuildActionLabel`), not the equipped state `self_weapon_id`.
On the qwd val combat segment they agree only **74.3%** (best lag 77%, no clean offset, so
the gap is semantic not pipeline-misalignment). Where they diverge:

- **At/near switches** — expected action-lead latency (the key is pressed before the engine
  equips). Recovers with a few frames of slack.
- **Persistent deep-hold divergence** — the label misses **engine-forced auto-switches**
  (ammo-out `W_BestWeapon`, pickups, respawn) that change the equipped weapon with no impulse.

Quantified via label-recall of engine-state transitions (`_weapon_label_state_recall.py`):

| ±k | label tracks state transition |
|---|---|
| 0 (exact) | 17.3% |
| ±1 | 50.0% |
| ±2 | 67.0% |
| ±3 | 71.7% |

**33% of state transitions are never matched** (±2), dominated by **rl→sg (42%)**. Two
engine-forced sources feed rl→sg (full-stream check): mostly **ammo-out `W_BestWeapon`**
(~88%, at normal combat health) plus a real minority of **respawn-after-death** (~12% follow
health→0 within 10 frames; transitions *into* shotgun run 12–15% death-enriched vs ~5%
baseline, since respawn defaults to SG). Then lg→sg, rl→axe (also ammo-out). This reproduces
the May-2026 `agents/plans/weapon-switch-source-classification.md` backlog (recall ~31%, same
RL→SG signature) on the current corpus. All are engine-forced non-decisions the intent label
correctly omits.

**Implications:** (1) `act.weapon` (intent) is the *correct* BC target — auto-switches happen
for free in the engine, so the head shouldn't have to predict them; `self_weapon_id` is
legitimate *input* (the from-weapon), never a leak. Do **not** retarget to `self_weapon_id`.
(2) The ~85% per-frame weapon-head ceiling is partly switch-latency label noise + auto-switches
the head is asked to predict but the demonstrator never chose — not a model failure. (3) If
switch fidelity is ever needed, the fix is the backlog (classify press vs auto/pickup/respawn),
not relabeling. The impulse/`op_weapon` byte is **not** in the precomputed cache, so confirming
press-vs-auto per frame needs that instrumentation (re-collect or demo parse).

## 7. Deployed decode — in-graph sticky gate (Pattern A)

The sticky-weapon controller (top-2 softmax + `conf≥C ∧ margin≥M` gate, else hold)
used to run **engine-side** with the thresholds hardcoded as `#define`s
(`QNN_ONNX_WEAPON_SWITCH_CONFIDENCE 0.65 / MARGIN 0.15`, "v23 ModelConfig values").
That split the decode from the model and let the engine's constants silently drift
from what the head was tuned against. The gate now runs **in-graph** (Pattern A):
the exported graph emits a DECIDED `weapon` impulse (int64), and the engine passes
it through ([`qnn_onnx.c`](../../src/engine/common/qnn_onnx.c) — controller and
`#define`s removed). The held weapon is read from the `self_weapon_id` input via the
same `weapon_index_from_id` mapping training uses, so the deployed decision matches
[`policy.py`](../../src/qnn/model/policy.py) `emit_actions` exactly. The `(C,M)`
thresholds are baked as graph constants AND stamped into the ONNX `decode.*`
metadata so they travel with the weights. See [`look-head.md`](look-head.md#4-design-principle--the-decode-regime-lives-in-the-model)
for the design principle.

### v24 operating point — swept on the actual model

The gate `(C,M)` is the remaining lever (not architecture — §1). Swept on the
deployed model itself (`head_probe_full_4head_seed17`, `_weapon_switch_threshold_eval.py
--segment-mask target`, 1.01M frames →
[`_weapon_threshold_opt_head_probe_full_4head_seed17.json`](../../runs/head_probe/_weapon_threshold_opt_head_probe_full_4head_seed17.json)):

| operating point | committed macro-F1 | acc | commit |
|---|--:|--:|--:|
| raw (always commit) | 0.8125 | 0.8721 | 100% |
| **historical engine 0.65 / 0.15** | 0.8122 | 0.8723 | 89.3% |
| **swept best — 0.56 / 0.0** | **0.8139** | **0.8732** | 94.6% |

The principled value is **C=0.56, M=0.0** — margin **0** (flat 0.00–0.05), not the
hardcoded 0.15; the historical 0.65/0.15 is measurably worse. Export the deploy
ONNX with `--weapon-switch-confidence 0.56 --weapon-switch-margin 0.0` (the bench
checkpoint carries 0.0/0.0, so the exporter WARNS and the override is required until
the value is stamped onto the checkpoint). Re-run the sweep per checkpoint — the
optimum is a property of the head's calibration, not a constant.

### Migration note

Pattern A makes the `weapon` output a decided int, so every weapon-bearing model
must be **re-exported** through the current pipeline (checkpoints are on NAS). v17
has no weapon head (emits no weapon output); v22 has one and re-exports to a decided
`weapon`. Once v17/v22 are re-exported to wire.9, the legacy wire.7 codec has no
remaining ONNX and can be retired from the engine.

## Regenerate

`scripts/analysis/_weapon_corpus_stats.py`, `_weapon_combat_vs_full.py`,
`_weapon_switch_events.py`, `_weapon_when_triggers.py` (model-free, diag/resident
loaders); `_weapon_readout_probe.py`, `_weapon_token_attribution.py` (need a model —
run at the checkpoint's training commit); `_weapon_switch_threshold_eval.py` (dense
per-frame grade); `_weapon_dense_vs_switch.py` (dense-vs-switch held-weapon F1; dense
side reads the threshold-opt JSON since the cls_gru checkpoint predates the shared-embed
refactor and won't load on current code).

Held-weapon-input dense (§5 "both" result): retrain in-distribution via the bench daemon —
`head_probe_weapon_token_d64_h16_repro_seed17` (self-embed, `self_weapon_embed_in_self=true`)
and `head_probe_weapon_arsenal_motion_weapon_token_seed17` (split token, `weapon_arsenal`
with `use_weapon_token=true`), then `_weapon_switch_threshold_eval.py --segment-mask target`
for the carry-forward sweep. The original May-20 `weapon_token_*` checkpoints are restored
from NAS backup but are pre-token-unification + OOD (won't load on current code) — retrain,
don't reload. Temporal fidelity (switch rate + dwell-time of the committed stream vs the
demonstrator): `_weapon_temporal_fidelity.py --run <weapon_run>`.
