# Attack head: findings & design

What drives the binary **attack** (fire) decision in the `qwd` corpus, why aim
geometry alone can't predict it, why weapon-readiness (cooldown) dominates, why the
weapon-aware lead-corrected `aim_vec` did not pan out, and where the real ceiling
lives. Numbers measured on `artifacts/collect/qwd` (val split, combat segment
`act.target != 0` unless noted). Regenerate with the bench head probes + the
`_attack_*` / `_fire_*` analysis sidecars referenced per section.

Terminology: `attack` (not "fire"), `op`/operative (not "legal"). Many older run
names and a few metric keys still spell it `fire` — those are the same head.

## TL;DR — the reframe

Attack is **not** an aim-geometry decision ("crosshair is on the enemy → fire").
It is a **weapon-readiness / cooldown gating process**: you fire when the held
weapon is *ready* (`attack_finished` cooldown elapsed) and you are *engaged* (a
recently-active combat bout), modulated weakly by alignment and range. Target
geometry alone is near-useless — a from-scratch head fed only soft-pooled
`(rel_dir, dist)` collapses to the base rate (**f1 = 0.0**); add the cooldown
scalar and it jumps to **0.34**, add velocity **0.36**. Two independent
input-ablations agree the single dominant feature is `attack_finished` (cooldown):
zeroing it costs **+96%** attack loss on the full stack and **+58%** inside the
weapon token, while damage/radius/`weapon_embed`/ammo are near-dead. Temporal
recurrence (GRU) is the next-biggest lever (+~0.08 f1 over a flat CLS encoder),
consistent with cooldown being an inherently temporal "time-since-last-shot"
quantity. The classifier plateaus at per-frame **f1 ≈ 0.51–0.59** because attack
*timing* is the bottleneck: at the exact frame f1 is ~0.32–0.51, but with a ±2–3
frame tolerance window it rises to ~0.54–0.64 (and ~0.81 at ±10) — the head knows
*that* you fire and roughly *when*, just not the exact 20 Hz tick.

## A note on the metric (masked vs unmasked)

Two metric regimes appear across the history and they are **not comparable**:

- **Older `fire_*` runs** trained through the *canonical fire-head* path, which
  emits both `val_f1_fire` (global, includes easy no-target / out-of-combat
  negatives) and `val_f1_fire_masked` (only frames the engine op input-mask keeps).
  When `op_input_mask=true` the right number is `val_f1_fire_masked`; the bare
  `val_f1_fire` / `final_val_f1_fire` is the **inflated unmasked** version. Example:
  `head_probe_fire_masked_baseline_seed17_h128_lr6e3` reports `f1_fire_masked` **0.51**
  but `f1_fire` **0.30** at the same epoch — the unmasked number is *lower* here
  because masking removes negatives the global metric scored as easy true-negatives
  inside the f1's pos-class accounting; the point is they measure different
  populations. The pre-mask `fire_token` sweeps report only `val_f1_fire ≈ 0.77`,
  which is **not** the in-combat metric and not comparable to anything below.
- **Newer bench `attack_*` runs** (cls / preattn / geom / bundle) use the bench
  [`attack_metrics`](../qnn/model/bench/attack.py#L43-L68), which emits a *single*
  `f1_attack` computed over the already-filtered frames (`input_mask=true` **and**
  `segment_mask = act.target!=0`). There is **no** separate masked/unmasked split —
  `val_f1_attack` *is* the in-distribution metric. So the §1–§4 numbers below are
  all "masked-equivalent" (in-combat) by construction.

**OPEN:** the headline **0.588** (CLS+GRU) was logged as `val_f1_attack` under
`input_mask=true`, so it is in-distribution by the §A construction — but it was
*never* re-graded through the `*_masked` fire-head metric, and the CLS runs carry
**no** `*_masked` keys at all. Treat 0.588 as the in-combat f1; do not assume it
equals a canonical `f1_fire_masked` without a re-grade.

## 1. Geometry alone is dead; cooldown carries it

`attack_geom_bundle` ([`attack_geom_bundle.py`](../qnn/model/bench/attack_geom_bundle.py))
is a no-CLS / no-GRU head fed the **explicit** GT-soft-pooled target geometry
`[rel_dir(3), dist(1)]`, with optional cooldown and velocity blocks. 12 ep,
`input_mask=true`, segment `act.target!=0`, lr 0.003, seed 17. Best epoch by
`val_f1_attack`:

| run | inputs | best f1_attack | precision | recall |
|---|---|---|---|---|
| `head_probe_attack_geom_reldist_seed17` | rel_dir + dist | **0.000** | 0.00 | 0.00 |
| `head_probe_attack_geom_reldist_cd_seed17` | + `attack_finished` | **0.336** | 0.343 | 0.329 |
| `head_probe_attack_geom_reldist_cd_vel_seed17` | + cooldown + vel | **0.355** | 0.365 | 0.346 |

Geometry-only collapses to the all-negative base rate (tp=fp=0). The head comment
explains why: the attack signal lives in the *normalized* forward-alignment cosine
`rel_dir[...,0]` (attack rate 4%→24% across its quartiles, corr≈0.15), not in raw
`soft_rel[0]` (corr≈0.03) — and even with normalization, alignment alone is too weak
to cross f1@0.5. **Cooldown is the carry feature**: +cd alone buys +0.34, velocity
adds a further +0.02. (Verified — matches the prior "reldist≈0, +cd 0.34, +cd+vel
0.36" memory.)

The `attack_preattn` prior sweep ([`attack_preattn.py`](../qnn/model/bench/attack_preattn.py),
canonical `AttackHead` over PreAttn + GT pointer, 4 ep × seeds {17,23,42}) tells the
same story from the other side — bolting a *geometric alignment prior* onto the head
does **not** help and slightly hurts:

| prior_mode | f1_attack (seed17 / 23 / 42) |
|---|---|
| `none` | 0.461 / 0.465 / 0.461 |
| `geometric_fixed` | 0.442 / 0.431 / 0.433 |
| `geometric_scalar` | 0.449 / 0.446 / 0.469 |
| `geometric_perweapon` | 0.461 (seed17 only) |
| `hit_test` | (config staged, **never run** — only seed17 dir, no history) |

(`geomperw` seeds 23/42 and all `hittest` seeds were staged but never trained — only
`config/` exists.) Every geometric prior ≤ the no-prior baseline. Geometry as a
*prior bias* is dead weight, mirroring the geometry-as-input result.

## 2. The cooldown smoking gun — two input ablations

Both ablations were run on trained checkpoints at their training commit (grad +
zero-block on the val cache, segment `act.target!=0`).

**(a) Full-stack head input ablation**
([`_attack_input_ablation_*_full_stack_noprior_a50_seed17.md`](../../runs/head_probe/_attack_input_ablation_head_probe_weapon_aim_canonical_parity_full_stack_noprior_a50_seed17.md)),
zeroing one block of the 140-dim attack-MLP input, and one scalar group of the 17-dim
self bundle:

| input block | Δ attack loss | % Δ |
|---|---:|---:|
| `attack_finished` (cooldown, self bundle) | +0.0814 | **+96.2%** |
| `self_readout` (CLS-ish self token) | +0.1162 | +137% |
| `dist_norm` (target range) | +0.0271 | +32% |
| `target_feat` (pooled target token) | +0.0248 | +29% |
| `health` | +0.0095 | +11% |
| `weapon_embed` (8-d per-weapon) | −0.0003 | ~dead |
| `engagement_ema` | −0.0015 | ~dead |
| ammo (shells/nails/rockets/cells) | ≈ 0 | dead |

**(b) Weapon-token column ablation**
([`_attack_bundle_weapon_proj_ablation_*.md`](../../runs/head_probe/_attack_bundle_weapon_proj_ablation_head_probe_attack_bundle_all_seed17.md)),
zeroing one column of the 3-d `weapon_proj` input `[damage, radius, attack_finished]`:

| column | Δ attack loss | % Δ |
|---|---:|---:|
| `attack_finished` (cooldown) | +0.1220 | **+57.8%** |
| `damage` | +0.0138 | +6.5% |
| `radius` | +0.0048 | +2.3% |

**Reads:** the weapon token's value is almost entirely the cooldown scalar, *not*
the damage/radius identity. `weapon_embed`, `engagement_ema`-as-feature, and ammo are
near-dead. The dominant signals are (1) the cooldown gate, (2) the self/CLS readout,
(3) target range/`target_feat` as a weak secondary. This is the cooldown-dominance
thesis, confirmed two independent ways. (SETTLED.)

## 3. Bench form sweep — what flat features carry

`attack_bundle` ([`attack_bundle.py`](../qnn/model/bench/attack_bundle.py)): PreAttn
passthrough + GT oracle pointer + `AttackBundleHead`, no CLS / no GRU. Idealized
scaffold (oracle target, no real encoder, no temporal). 12 ep, `input_mask=true`,
segment `act.target!=0`, seed 17, best epoch by `val_f1_attack`:

| bundle (target_feat +) | best f1_attack | precision | recall |
|---|---:|---:|---:|
| `none` (target_feat only) | **0.000** | 0.077 | 0.000 |
| `motion` | 0.101 | 0.302 | 0.061 |
| `engaged_ema` | 0.009 | 0.296 | 0.004 |
| `weapon` (incl. cooldown) | 0.457 | 0.550 | 0.391 |
| `weapon + motion` | 0.526 | 0.540 | 0.512 |
| `weapon_embed_only` (v2) | 0.422 | 0.298 | 0.729 |
| `engaged + weapon_embed` (v2) | 0.487 | 0.378 | 0.682 |
| `all` (weapon+motion+engaged+geom) | **0.539** | 0.546 | 0.534 |
| `all` (postfix re-grade) | 0.541 | 0.417 | 0.772 |

Reads: target geometry only (`none`) → base rate; the **weapon token (carrying the
cooldown) is the single block that unlocks the head** (0.0 → 0.457); motion adds
+0.07 on top of weapon; the full bundle tops out at **~0.54** — the best
*non-temporal* attack head on record. (`engaged_ema` as a standalone feature is dead
here, but as a *conditioner* it's the top MI driver — see §5; the bundle form just
can't use it without temporal state.)

The `weapon_aim` "full-stack" attack head (canonical AttackHead, cooldown + weapon
embed + engagement + geometry, no prior, 30 ep) reaches the same neighborhood —
`head_probe_weapon_aim_canonical_parity_full_stack_noprior_a50_seed17` **f1_attack
0.513** (best ep 9). Single-conditioner variants of that probe isolate the levers:

| variant | f1_attack |
|---|---:|
| `weapon_embed` | 0.501 |
| `geom` | 0.500 |
| `engaged_noprior` | 0.499 |
| `engaged` | 0.488 |
| `canonical_parity` (baseline) | 0.471 |
| `weapon_cooldown` (cooldown only) | 0.416 |
| `weapon_ammo` | **0.007** (ammo is dead) |
| `target_only` (geometry only) | **0.003** (geometry is dead) |

`target_only` 0.003 and `weapon_ammo` 0.007 re-confirm §1–§2: geometry and ammo
carry nothing; cooldown alone gets you to 0.42, and the extras (engagement, weapon
embed, geometry-as-feature) each nudge it toward ~0.50.

## 4. Encoder / temporal — the GRU lever

`attack_cls_transformer` ([`attack_cls_transformer.py`](../qnn/model/bench/attack_cls_transformer.py))
puts the attack head on the **real** CLS attention encoder (a 1:1 clone of the
`move_cls_transformer` regime move/look were tested in), with/without the GRU temporal
stream, and with/without a real target pointer. d_model 64, n_heads 2, n_layers 2,
50 ep, lr 0.003, `input_mask=true`, segment `act.target!=0`, seed 17. Best epoch by
`val_f1_attack`:

| run | encoder | best ep | f1_attack | precision | recall |
|---|---|---:|---:|---:|---:|
| `head_probe_attack_cls_xfmr_d64_h2_seed17` | CLS, no GRU | 30 | **0.509** | 0.481 | 0.541 |
| `head_probe_attack_cls_gru_xfmr_d64_h2_seed17` | CLS + GRU | 6 | **0.588** | 0.515 | 0.685 |
| `head_probe_attack_cls_target_xfmr_d64_h2_seed17` | CLS + target pointer (no GRU) | 23 | **0.504** | 0.499 | 0.509 |

Reads (all verified):

- **CLS no-GRU ≈ 0.51** — barely above the flat bundle's 0.54 (within noise), and it
  takes 30 epochs to get there.
- **+GRU ≈ 0.59 (+~0.08)** — temporal recurrence is the biggest single architectural
  lever, and it peaks early (best ep 6). Consistent with cooldown being a temporal
  "frames-since-last-shot" quantity the GRU can integrate directly. **This 0.588 is
  the best attack-head number on record.**
- **CLS + target pointer ≈ 0.50** — an explicit target pointer adds **nothing** (it's
  ≤ the plain CLS). Localizing the target does not help attack; this restates the §1
  geometry-is-dead finding at the encoder level.

The no-GRU CLS health was not separately audited here for the overfit signature its
`move`/`weapon` siblings showed (best epoch 30 of 50 suggests it did *not* overfit
hard, unlike `weapon_cls_xfmr`), but that wasn't explicitly checked. (Minor OPEN.)

## 5. Why it plateaus — timing is the bottleneck

Two diagnostics show the ~0.5 per-frame ceiling is a **timing** problem, not a
"can't tell combat from idle" problem.

**(a) Methodical conditioner analysis** (`_attack_prior_methodical/report.md`,
model-free greedy NLL on fire-op frames, base rate 2.80%, marginal NLL 0.1278
nats/frame). Greedy feature selection and marginal mutual information:

| feature | marginal MI (nats) | greedy ΔNLL |
|---|---:|---:|
| `engaged` (active combat bout) | +0.0305 | +0.0305 (step 1) |
| `frames_since_engaged` | +0.0251 | +0.0020 |
| `aim_align` | +0.0248 | +0.0135 (step 2) |
| `pred_look_on_target` | +0.0247 | — |
| `attack_finished` (cooldown) | +0.0228 | +0.0006 |
| `dist_target` | +0.0087 | +0.0005 |
| `weapon` | +0.0023 | +0.0027 (step 3) |
| `has_ammo` | +0.0002 | ~0 |

Greedy stops at NLL 0.0769 (from 0.1278). `engaged` + `aim_align` + `weapon` +
`frames_since_engaged` capture nearly all of it; ammo and self-speed are noise.
Note `engaged`/`frames_since_engaged` (temporal combat-bout state) is the top
conditioner — which is exactly what the GRU buys in §4 and what the *flat* bundle
can't use (§3, where `engaged_ema`-as-feature was dead).

**(b) Event-tolerance sweeps.** Grading the trained heads with a ±k-frame tolerance
window (a fired-frame counts as correct if a true fire is within ±k) shows the heads
localize fire to within a few frames — the exact-frame f1 is pessimistic:

| ±k tolerance | full-stack (`_attack_eval_full_stack_noprior_a50.json`, thr 0.1) | look-style (`_attack_look_style_threshold_scan_ep7.json`, thr 0.1) | fire masked baseline (`_fire_baseline_masked_eval.json`, thr 0.1) |
|---:|---:|---:|---:|
| 0 (exact) | 0.38 | 0.38 | 0.34 |
| ±1 | 0.50 | 0.50 | 0.46 |
| ±2 | ~0.56 | 0.58 | 0.54 |
| ±3 | 0.61 | 0.64 | 0.60 |
| ±5 | 0.68 | — | — |
| ±10 | 0.81 | — | — |

The head knows you're firing; the residual error is mostly ±1–3 frame jitter in
*exactly which* 20 Hz tick the shot lands — the same bimodal emit-frame timing issue
the labeler plan flags for the C-side fire label (raw rising-edge F1 0.79, first-shot
recall 0.97; see
[`labeler-jump-fire-refinement.md`](../../agents/plans/labeler-jump-fire-refinement.md)).
So per-frame f1 ≈ 0.5 understates real fire-decision quality.

## 6. The aim_vec dead-end (weapon-aware lead aiming)

`weapon_aim` ([`weapon_aim/`](../qnn/model/bench/weapon_aim/)) trained look + attack
**jointly**, sharing a closed-form, weapon-aware, lead-corrected aim point
([`lead_aim.py`](../qnn/model/bench/weapon_aim/lead_aim.py): per-entity intercept
quadratic, hitscan / linear-projectile / ballistic-GL parameterized, gravity-corrected,
no trainable params). The hypothesis: a weapon-aware `aim_vec` ("where must I aim to
hit, given my gun's projectile") is a better attack/look signal than a plain look
direction.

**It did not pan out as real attack signal.** The apparent gains traced to a
`prev_look` persistence **leak**, not weapon-aware aiming — the genuine `aim_vec`
contributes only ~+0.01 nats over a plain look-prior, while the +0.09 was `prev_look`
(this is the look-head `aim_vec`/`prev_look` decomposition in project memory; see
[persistence-and-changepoints.md](persistence-and-changepoints.md) for the shared
copycat/leak framing). For the **attack** head specifically: the `aim_vec`-prior /
`prev_look` weapon_aim variants logged look metrics only (attack head effectively idle
in those run dirs), and the attack head's own ablation (§2) shows `aim_align`/geometry
as a *weak secondary* and `weapon_embed` as *dead* — there is no weapon-aware-aiming
attack lever hiding in the data. Weapon-aware aiming is settled as a non-result for
attack. (SETTLED — though note the *attack-specific* aim_vec ablation was inferred
from §2 + the look-head decomposition, not from a dedicated attack-only aim_vec run.)

Two real bugs were fixed building `weapon_aim` (both still valid, per the bench skill):
a weapon-id encoding mismatch (`ENTITY_IDS` 3..10 vs impulse 0..8, async ROCm OOB
gather — commit `cccdcb0`) and a forward-hook hang on the BC training loop (replaced by
per-head `aim_vec` recompute — commit `5a91d3d6`).

## 7. The oracle ceiling — how high can geometry+timing go?

`attack_preattn_oracle` ([`attack_preattn_oracle.py`](../qnn/model/bench/attack_preattn_oracle.py),
`OracleAttackHead`) and the `_attack_oracle_*` sidecars bound what a *privileged*
geometry+range+timing model can do. The oracle uses GT target, per-weapon range
LUTs, and a temporal alignment kernel:

- `_attack_oracle_validate.json`: best oracle f1 across its threshold sweep is
  **0.341** (thr 4.9, precision 0.30 / recall 0.39); at low thresholds it is
  high-recall / low-precision (f1 ~0.24). Note this oracle was graded on a
  *different* frame population (`n_kept 392367`, `n_dropped_pos 255296`) — it drops a
  large share of positives, so it is **not** directly comparable to the §1–§4
  in-combat f1.
- `_attack_oracle_per_weapon_calibration.json`: per-weapon best f1 raw 0.339,
  calibrated 0.335 — calibration does **not** help; SG 0.35, SSG 0.36, NG higher
  (small n), AXE 0.23.
- `_attack_oracle_range_fit.json`: the best (cutoff, width) range gate tops out around
  **f1 0.35** — i.e. an *idealized* per-weapon range model still can't beat the
  cooldown-driven flat bundle (0.54) or the GRU (0.59).
- `_attack_empirical_range.json`: demonstrators fire **far beyond engine weapon
  range** — axe fires median 297 qu (engine range 64 qu; **94%** of axe fires are
  "out of range"), SG median 277 qu. Fire is **not** gated on being in lethal range;
  it's gated on readiness + engagement. This is *why* a geometric range oracle caps so
  low.

**Read:** a pure geometry/range/timing oracle ceilings at **f1 ~0.34**, *below* the
learned cooldown+temporal heads. The learnable signal is readiness + combat-state, not
where the crosshair points relative to lethal range. (SETTLED — geometry is not the
ceiling; it's a distraction.)

## 8. Burst initiation vs continuation — the consequential decision

The per-frame f1 lumps two very different decisions. A fire on an op (feasible, off-cooldown)
frame is a **burst CONTINUATION** if the previous op frame also fired (sustained — trigger held,
autocorrelated, easy) or a **burst INITIATION** if it did not (the genuine "decide to engage").
Decomposing the retrained GRU head (`_attack_burst_decompose.py`, op-frame f1 0.612 @ thr 0.55):

| fire type | share | recall |
|---|---|---|
| burst CONTINUATION (sustained) | 45.4% | **81.4%** |
| burst INITIATION (engage decision) | 54.6% | **51.4%** |
| all op-fires | — | 65.0% |

**The headline f1 is inflated by sustained-fire autocorrelation** — continuation recall is 81%,
but the genuine engage decision (when to *start* firing) is only **51.4%**. This is the attack
analog of the weapon hold-vs-switch split, and it lands right next to weapon's 54% switch-to-fire:
the consequential decision is much weaker than the aggregate metric implies.

**Lead-side vs lag-side timing buffer for burst initiations** (model fire within the window of the
true init frame):

| ±k | lead `[−k,0]` | lag `[0,+k]` |
|---|---|---|
| 0 | 51.4 | 51.4 |
| 2 | 58.6 | 51.4 |
| 5 | 61.8 | 53.8 |

The **lead side rises** (51→62% by ±5 — the model fires a few frames *before* the demonstrator's
burst start = genuine anticipation), while the **lag side is flat** (~51%). That flatness is the
cooldown structure: once the demonstrator fires, `attack_finished ≠ 0` locks out further firing,
so there is no post-fire window to "echo" a late detection into — the lag side carries no credit
and no leak. So the burst-init detection that exists is genuinely anticipatory, not lagged echo.
(SETTLED — the ~0.59 headline is continuation-dominated; burst initiation, the decision that
matters, is ~51% and anticipatory; no lag-side inflation.)

## What's settled vs open

**SETTLED:**
- Burst f1 is continuation-inflated: continuation recall 81% vs burst-initiation 51% (the
  consequential engage decision); init detection is anticipatory (lead-side, not lag echo —
  cooldown blocks the lag side). The ~0.59 headline overstates the engage decision.
- Aim geometry alone cannot predict attack (f1 0.0 from-scratch; `target_only` 0.003;
  geometric priors ≤ no-prior baseline; oracle range gate ceilings at ~0.34).
- Cooldown (`attack_finished`) is the dominant feature (+96% loss when ablated on the
  full stack, +58% inside the weapon token; cooldown-only head 0.42).
- Temporal recurrence (GRU) is the biggest architectural lever: +~0.08 f1 over a flat
  CLS encoder (0.51 → 0.59), consistent with cooldown + `engaged`/`frames_since_engaged`
  being temporal quantities.
- An explicit target pointer adds ~nothing (CLS+target 0.50 ≈ CLS 0.51).
- `weapon_embed`, ammo, damage/radius identity are near-dead inputs.
- Weapon-aware lead `aim_vec` is a non-result for attack; its look-side "gains" were a
  `prev_look` leak.
- The per-frame f1 ≈ 0.5 plateau is a timing-jitter artifact: ±2–3 frame tolerance
  lifts it to ~0.55–0.64 (~0.81 at ±10).

**OPEN / UNVERIFIED:**
- Whether **0.588** equals a canonical `f1_fire_masked`. It is the in-combat
  `val_f1_attack` under `input_mask=true` + `segment_mask=target` (so
  masked-equivalent by §A construction), but the CLS runs carry **no** `*_masked`
  keys and were never re-graded through the fire-head metric. Don't conflate the two
  metric families without a re-grade.
- **No leak-free / attack-frame-style audit** of the GRU head exists (the kind done
  for the weapon head's switch-vs-hold leak-free metric). The 0.588 could include
  trivially-easy negatives within the combat segment; its true switch-on/switch-off
  decision skill is unmeasured.
- **The real ceiling is unknown.** The oracle (privileged geometry+range+timing) caps
  at ~0.34 but on a different frame population; the GRU hits 0.59 on the in-combat
  segment. No clean, comparable upper bound on the temporal task exists. The labeler
  plan's C-side fire label (raw F1 0.79, first-shot recall 0.97) suggests the *label*
  has ±1-frame timing noise that bounds any per-frame head — quantifying that noise
  floor is the missing piece.
- The no-GRU CLS overfit signature (best ep 30/50) was not explicitly health-checked
  the way `weapon_cls_xfmr` was.
- `hittest` prior and `geomperw` seeds 23/42 were staged but never trained — the
  prior sweep is single-seed for those modes.

## Design implication

Model attack as a **temporal readiness gate**, not a geometry classifier: the head
needs the cooldown scalar (`attack_finished`), a temporal stream (GRU) to integrate
`frames_since_engaged` / time-since-last-shot, and an engagement-state signal —
target geometry is a weak secondary at best. Evaluate with an **event-tolerance**
metric (±2–3 frames), not exact-frame f1, since the label's own timing jitter caps
per-frame agreement. Don't wire in per-weapon range gates, ammo, or weapon-aware
aim correction — all measured as dead or near-dead for the attack decision.

## Regenerate

Bench heads (retrain in-distribution via `qnn.run.init --mode head_probe` + the bench
daemon; configs are frozen under each run's `config/`):
- Geometry: `head_probe_attack_geom_reldist{,_cd,_cd_vel}_seed17`
  ([`attack_geom_bundle.py`](../qnn/model/bench/attack_geom_bundle.py)).
- Flat bundle: `head_probe_attack_bundle_{none,motion,weapon,weapon_motion,all,...}_seed17`
  ([`attack_bundle.py`](../qnn/model/bench/attack_bundle.py)).
- Encoder/temporal: `head_probe_attack_cls_{xfmr,gru_xfmr,target_xfmr}_d64_h2_seed17`
  ([`attack_cls_transformer.py`](../qnn/model/bench/attack_cls_transformer.py)).
- Prior sweep: `head_probe_attack_preattn_prior_{none,geomfix,geomscalar,geomperw,hittest}_seed{17,23,42}`
  ([`attack_preattn.py`](../qnn/model/bench/attack_preattn.py),
  [`attack_prior/`](../qnn/model/bench/attack_prior/)).
- Joint look+attack: `head_probe_weapon_aim_canonical_parity_*_seed17`
  ([`weapon_aim/`](../qnn/model/bench/weapon_aim/)).
- Oracle ceiling: `head_probe_fire_oracle_*` / `attack_preattn_oracle`
  ([`attack_preattn_oracle.py`](../qnn/model/bench/attack_preattn_oracle.py)).

Analysis sidecars at the `runs/head_probe/` root (need a trained checkpoint at its
training commit unless noted model-free):
- Cooldown / input ablation: `_attack_input_ablation_*_full_stack_noprior_a50_seed17.md`,
  `_attack_bundle_weapon_proj_ablation_*.md`, `_attack_cooldown_*.json`.
- Conditioner analysis (model-free): `_attack_prior_methodical/report.md`.
- Event-tolerance grades: `_attack_eval_full_stack_noprior_a50.json`,
  `_attack_look_style_threshold_scan_ep7{,_tol5}.json`, `_fire_baseline_masked_eval.json`,
  `_fire_threshold_scan.json`.
- Oracle ceiling: `_attack_oracle_{ceiling,validate,range_fit,per_weapon_calibration,
  cluster_timing,nofire_stratify}.json`, `_attack_empirical_range.json`.
