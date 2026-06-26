# Fire discrimination — "fires into the void"

Why a target-less model that scores fine on per-frame fire metrics still
*looks* like it fires at nothing in live play, and the metrics that actually
catch it.

## 1. The complaint

The v24 `full_4head` head-probe runs (no target head; trained with
`segment_mask {"act.target": {"$ne": 0}}`, i.e. target-present frames only)
play well in many respects but "arbitrarily look off and fire into the void."
The standing belief was that target tracking "didn't help" because it never
moved the offline metrics.

## 2. Offline finding — the attack head is fine

`scripts/analysis/fire_target_conditional.py` forwards the model over the
**unfiltered** val set (segment_mask=None, so the ~74% no-target frames the
model never trained on are kept) and stratifies its fire decision two ways.
Per-episode padded-lane forward, GRU hidden carried within each episode, so
predictions match how `val_f1_fire` is produced.

### Fire rate by target presence (op-masked, thr=0.5)

| target presence | n (op) | model fire | model prob | human fire |
|---|--:|--:|--:|--:|
| none (p<.05) | 2,538,124 | 1.00% | 0.040 | 1.27% |
| [.05,.2) | 3,931 | 5.55% | 0.119 | 3.10% |
| [.2,.5) | 22,017 | 6.04% | 0.108 | 2.35% |
| [.5,.8) | 136,522 | 12.29% | 0.161 | 9.59% |
| locked (p≥.8) | 285,742 | 17.95% | 0.218 | 14.68% |

void ratio (no-target ÷ locked): model 0.056, human 0.087.

### Fire rate by crosshair→target angle (target present p≥.5, op-masked)

| crosshair→target | n | model fire | model prob | human fire |
|---|--:|--:|--:|--:|
| [0,2°) | 4,576 | 58.0% | 0.568 | 46.3% |
| [2,5°) | 19,670 | 51.4% | 0.511 | 39.3% |
| [5,10°) | 50,721 | 35.3% | 0.385 | 26.8% |
| [10,20°) | 100,999 | 20.5% | 0.254 | 16.1% |
| [20,45°) | 129,887 | 9.7% | 0.144 | 8.5% |
| [45,180°] | 116,411 | 3.6% | 0.066 | 3.7% |

Bearing is view-frame `arccos(entity_rel_x / |entity_rel|)` to the
argmax-`target_probs` token (slot 0 = no-target; entity token j ↔ slot j+1).

**Conclusion:** the attack head gates fire on BOTH presence and aim alignment
at least as tightly as a human. Both "ignores presence" and "ignores aim"
hypotheses are refuted.

## 3. Why it still fires into the void live

Every offline metric feeds the model **human-trajectory** observations, where
"target present" and "crosshair on target" are well-defined because the human
only ever lined up on legitimate enemies. The model learned the easy job —
*fire when an entity token is present and aligned* — which on human data is
indistinguishable from *fire at the right enemy*.

In live play the bot drives its own trajectory and look, and aligns its
crosshair with whatever actor token lands in front of it (teammate, enemy
through a wall, far/stale/mis-tracked actor). The attack head's condition is
satisfied, so it fires. Target's real value was never fire-*gating* — it's
**discrimination**: which entity is the legitimate target. That only matters
off the human manifold, which is precisely why it's invisible to
in-distribution offline metrics and why "target didn't help" offline while it
visibly matters live.

## 4. Live eval metrics (`qnn.eval.run` → `eval_summary.json`)

Obs-side fire discrimination, computed per fire tick from the model's own obs
tokens (mirrors §2's geometry so live numbers compare to the human curve):

- `obs_blind_fire_rate` — fire ticks with no actor within the aim cone
  (`obs_fire_cone_deg`, default 10°). The live "fires into the void" number.
- `obs_blind_fire_no_actor_rate` / `obs_blind_fire_offcone_rate` — split into
  "no actor token present" vs "actor present but off-cone".
- `obs_fire_aim_cos_mean` — mean crosshair→best-actor cosine at fire ticks.
- `obs_fire_angle_hist` — fire distribution over §2's angle bins.
- `obs_fire_ticks`, `obs_fire_rate` — context.

## 5. Live result (full_4head vs frikbot, arena 1v1, 24 ep, 43,200 ticks)

Run: `runs/eval/full_4head_fire_los_corr`. Engine-side P(fire | LOS-aim-angle
to nearest visible enemy):

| LOS aim→enemy | n_ticks | P(fire) |
|---|--:|--:|
| [0,2°) | 1,430 | 34.1% |
| [2,5°) | 2,800 | 34.0% |
| [5,10°) | 4,590 | 31.9% |
| [10,20°) | 7,345 | 25.3% |
| [20,45°) | 9,873 | 17.2% |
| [45,180°] | 17,162 | 9.2% |

- corr(fire, tracking_cos) = +0.21 (weak). Mean tracking_cos at fire = 0.79
  (~37°); at no-fire = 0.57 (~56°). [45,180°] lumps off-axis with no-LOS-enemy.
- Obs-side: `obs_blind_fire_rate` = 0.63, of which `offcone` = 0.60 and
  `no_actor` = 0.03. accuracy 0.21, frag_delta −7.8/ep.

**Revision of §3.** Offline the attack head fired tightly (58% at <2°). Live it
fires with its crosshair off every perceived actor 60% of the time, while
almost always having *some* enemy in perception (only 3% fire at nothing). So
the dominant live failure is **aim/look quality in closed loop**, not entity
discrimination: the attack head fires on features that merely correlate with
alignment on the human manifold (engagement/cooldown rhythm); under the bot's
own drifting look those decouple and it fires off-target. The
offline-can't-see-it thesis holds; the mechanism resolves to aim, not
discrimination. Lever: look-head closed-loop aim + make the attack head gate on
true crosshair alignment.

### Engine-side ground-truth (implemented)
`engine_fire_tracking_cos_corr`, `engine_fire_tracking_cos_mean` /
`engine_nofire_tracking_cos_mean`, and `engine_fire_by_los_angle` are emitted
by `qnn.eval.run`. `tracking_cos` is PVS + traceline gated (nearest in-LOS
enemy; cos=0 default when none visible).

### Planned (follow-up)

The engine already computes per-entity `tracking_cos`
(`qnn_metrics_t.entities[]`, `QNN_ComputeTracking`). An engine-side blind-fire
metric uses true world positions + LOS instead of the bot's obs. The
**divergence** between obs-side (model's perception says aligned) and
engine-side (nothing real to shoot) blind-fire IS the discrimination gap — and
the case for putting the target pointer back in as a head feeding the
attack/look path, evaluated on this OOD metric rather than on per-frame F1.
`mistarget_rate` (fired at a non-valid actor: teammate / no LOS) is the
sharpest version and needs the engine's actor classification.

## 6. Look-aim is the attack lever (2026-06-09)

§5 fingered "aim/look quality in closed loop" as the dominant live failure — the
bot has *some* enemy in perception 97% of fire ticks but fires with the crosshair
off it 60% of the time. The look-head investigation
([`look-head.md`](look-head.md)) supplies the mechanism. The look head's
*deterministic* heading is human-persistent (consec-dir cos 0.908 vs human 0.916),
but the **current decode samples the direction bin per frame**, tripling the
heading-reversal rate (2.1%→6.2%). Under that jittering crosshair the attack
head's "aligned" belief is satisfied transiently, mid-flip, on whatever actor
token sweeps past — so it fires off-target. The attack head needs **no fix of its
own**; it inherits the look decode fix (hybrid: sampled magnitude × continuous
`look_predict` direction — [`look-head.md`](look-head.md#3-the-decode-regime--hybrid-sampled-mag--continuous-dir)),
which stabilizes the aim it gates on.

**Falsifiable prediction:** deploying the look hybrid decode (+ move sticky τ=0.6)
should drop `obs_blind_fire_rate` / `obs_blind_fire_offcone_rate` materially with
no change to the attack head. If void-firing persists with a stabilized crosshair,
the residual is genuine entity discrimination (the §1–§3 target-pointer case);
if it drops, the live void-fire was the look-aim artifact. This is the cheaper
test to run before re-introducing a target head.

*Outcome:* the hybrid decode helped live play (a24b: less spin, better feel)
but lead-less, immature aim persisted — both the look-decode artifact AND the
target case were real. The target-pointer A/B is §7.

## 7. Target pointer A/B — full_5head live result (2026-06-10)

`full_5head` = full_4head + the canonical `TargetPointer` (d_target=16),
`target_feat` concatenated into the motor feature base
([target-head.md](target-head.md); run
`runs/head_probe/head_probe_full_5head_seed17`, best ep 52). Offline BC
metrics: motor parity with full_4head, `target_kl` 0.036 converged, weapon
macro +0.006 — flat, as predicted by §3's redundancy argument. Live A/B,
identical scenario/decode/seed protocol (24 ep arena 1v1 vs frikbot, sampled;
`runs/eval/full_5head_fire_los_corr` vs `runs/eval/full_4head_fire_los_corr`):

| metric | full_4head | full_5head | Δ |
|---|--:|--:|--:|
| `obs_blind_fire_rate` | 0.629 | **0.563** | −6.6pp |
| `obs_blind_fire_offcone_rate` | 0.599 | **0.547** | −5.2pp |
| `obs_blind_fire_no_actor_rate` | 0.030 | **0.016** | halved |
| `obs_fire_aim_cos_mean` | 0.825 | **0.867** | +0.042 |
| `engine_fire_tracking_cos_mean` | 0.794 | **0.845** | ~37°→32° |
| `engine_nofire_tracking_cos_mean` | 0.566 | **0.687** | +0.121 |
| ticks within 10° LOS of an enemy | 8,820 | **12,285** | +39% |
| accuracy | 0.214 | **0.242** | +13% rel |
| frags / deaths per ep | 4.3 / 12.1 | **5.1 / 10.3** | −7.8→−5.2 |

**CORRECTION (same day) — the table above is CONFOUNDED.** The full_4head
baseline was evaluated 2026-06-09 03:29, BEFORE a day of Python-side decode
changes landed (engine sticky fb/lr move decode `9bf8d5db`, w7/w9 decode
decoupling `d0fd7559`, eval diagnostics `b09e4b35`); the full_5head run used
today's code. A same-checkpoint/same-seed re-eval of full_4head on today's
code (`runs/eval/full_4head_fire_los_corr_rep2`) moved its numbers by more
than the apparent pointer effect (blind-fire 0.629→0.538) — most of the §7
table's Δ was the decode fixes, not the pointer. The eval harness is
fully deterministic given (code, seeds, checkpoint): an identical-config
re-eval of full_5head reproduced r1 bit-exactly, which is what isolates the
code drift as the cause.

### 7.1 Matched-code A/B (both arms on today's code, same episode seeds)

| metric | full_4head (rep2) | full_5head | Δ (pointer) |
|---|--:|--:|--:|
| `obs_blind_fire_rate` | **0.538** | 0.563 | +2.6pp (worse) |
| `engine_fire_tracking_cos_mean` | 0.849 | 0.845 | ≈parity |
| `engine_nofire_tracking_cos_mean` | 0.635 | **0.687** | +0.052 |
| accuracy | **0.255** | 0.242 | −0.013 |
| frags / deaths per ep | 5.6 / 12.0 | 5.1 / **10.3** | deficit −6.4 vs −5.2 |

Revised reads:

- **The decode fixes were the big lever** on blind-fire/tracking (full_4head
  alone: 0.629→0.538 blind-fire from code changes) — §6's prediction
  validated, but credited to the wrong cause in the first version of this
  section.
- **The pointer's surviving live effects are modest:** between-shot tracking
  +0.052 cos and fewer deaths (12.0→10.3, frag deficit −6.4→−5.2);
  blind-fire and accuracy are flat-to-slightly-worse. The dramatic
  "+39% time near enemy / −6.6pp blind-fire" claims do NOT survive the
  matched-code comparison.
- Episode-sampling noise bound (true-seed replications `*_rep3`,
  eval_holdout_seed_offset 9000; spread = |seedsA − seedsB| per arch, delta =
  cross-arch mean):

  | metric | seed spread (4h / 5h) | pointer Δ | verdict |
  |---|--:|--:|---|
  | `engine_nofire_tracking_cos_mean` | 0.006 / 0.003 | **+0.051** | real (~10× spread) |
  | deaths/ep | 0.79 / 0.13 | **−1.21** | real |
  | `obs_blind_fire_rate` | 0.007 / 0.005 | +0.020 | real, small, **worse** |
  | `engine_fire_tracking_cos_mean` | 0.004 / 0.003 | −0.001 | parity |
  | accuracy | 0.021 / 0.014 | +0.004 | noise |
  | frags/ep | 0.38 / 0.54 | −0.04 | noise |

  Settled verdict: the pointer's live effect is **+0.05 between-shot
  tracking and −1.2 deaths/ep, at the cost of +2pp blind-fire**, with
  fire-time aim and accuracy at parity. Modest, coherent (target_feat
  conditions look/move toward sustained enemy awareness), and an order of
  magnitude smaller than the §7-table claims. For scale: the decode-drift
  confound (0.09) was ~15× the per-run episode noise (~0.006) — both
  baselines and noise bounds are cheap, run them.
- Process lesson: live A/B baselines are only valid at the same code commit —
  re-evaluate the baseline whenever decode/eval code moves (cheap, ~1 h CPU,
  and the deterministic harness makes drift detection trivial).
- **Still open:** absolute blind-fire is high (~0.55 — humans ~0.09 void
  ratio), and projectile lead is still absent. The next ladder is the
  sticky-pointer hysteresis (the pointer over-switches 3.6× vs human on
  multi-enemy frames — `runs/head_probe/_target_switch_dynamics.json`) and an
  `aim_vec`-prior residual look head (procedural track-with-lead default, NN
  learns the human deviation; judged live with a too-good-to-be-human ceiling
  check against §2's human curves).
