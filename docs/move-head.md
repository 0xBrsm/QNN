# Move head: findings & design

What drives the move axes (forward/back, left/right, up/down=jump) in the `qwd`
corpus, why per-frame accuracy is the wrong objective, and why — unlike look /
weapon / attack — recurrence does **not** clearly help the move head per-frame.
Numbers measured on `artifacts/collect/qwd` (val split, in-distribution
`act.target!=0`, n=1,014,088 unless noted). Regenerate with the `scripts/analysis/move_*.py`
diagnostics referenced per section.

## TL;DR — the reframe

Move is **momentum-dominated**. Per-frame movement is overwhelmingly predictable
from *recent motion* (velocity = the physical integral of recent inputs) plus the
previous action; a trivial first-order Markov chain (per-axis dll **0.407**)
matches or *beats* the trained Transformer+GRU (**0.378**). The fb/lr/ud axes are
**nearly independent** (total correlation 0.0077 nats; real joint KL 0.0027) — a
joint/coupling move head buys nothing. The jump axis only *looks* collapsed
because **argmax** of a calibrated head emits ~0 jumps while the head's expected
jump mass matches the human rate; **sample, don't argmax**. The per-frame ceiling
(~0.41 dll) is a momentum cap — pushing past it means re-introducing a prev-action
copy leak (the move analogue of the `prev_look` leak). So the real human-likeness
lever for move is **temporal-stream fidelity** (dwell-times, switch-rates, the
~88% autocorrelation), evaluated *distributionally*, not per-frame argmax accuracy.

This makes move the **contrast head**: look / weapon / attack want the GRU (it adds
real per-frame signal); move is Markov-1-capped, so CLS-or-GRU is roughly a wash on
per-frame predictability and the GRU's only honest contribution is temporal-stream
coherence, not point accuracy.

## 1. Bench head forms

Two bench head families, both move-only (`head_loss_weights move=1`, rest 0),
`input_mask=true`, `segment_mask act.target!=0`, seed 17. Output is the 3 axes ×
3 classes (neg/none/pos) = 9 logits ([`move_cls_transformer.py:36`](../../src/qnn/model/bench/move_cls_transformer.py#L36)).

- **`move_motion_token`** ([`move_motion_token.py`](../../src/qnn/model/bench/move_motion_token.py)) —
  hand-built tokens through a passthrough `PreAttnEncoder`, no temporal, no attention.
  MLP over the motion token (vel_xyz, view_pitch, rot_vel, movement_embed,
  powerup_motion), optionally concatenated with a second token (target / state /
  state-no-weapon / weapon / spatial / spatial+state). 12 ep, lr 0.0045.
- **`move_cls_transformer`** ([`move_cls_transformer.py`](../../src/qnn/model/bench/move_cls_transformer.py)) —
  move head on the CLS readout of a real Transformer encoder over the full token
  stream `[CLS, state, arsenal, motion, weapon(dmg+rad), spatial×9, entity×N]`,
  with (`use_gru`) or without a GRU temporal stream. d64/h2/l2/d_ffn128, 50 ep, lr 0.003.

The declarative `selection_metric` on both `HeadSpec`s is `move_dll` (higher is
better), matching the live `_selection_score` composite which prefers
`move_dll → loss_move → f1_move` ([`train.py:197`](../../src/qnn/bc/train.py#L197)).
**Caveat:** the per-epoch `bc_history.json` records only `f1/acc/loss_move`, not
`move_dll`, for these runs — the distributional dll numbers below come from the
post-hoc `_move_dist_eval.py` / momentum-baseline sidecars, not from per-epoch history.

### Per-frame macro-F1 (val, best `final_val` from `live_run_report.json`)

These are argmax point-accuracy numbers — the **wrong objective** for move (see §3),
listed here only to characterise the form sweep. Macro-F1 = mean over the 3 axes.

| head / form | ep | best_sel_score | val f1_move | f1_fb | f1_lr | f1_ud | val acc_move |
|---|---|---|---|---|---|---|---|
| **CLS + GRU** | 13 | 0.904 | **0.699** | 0.799 | 0.792 | 0.505 | 0.864 |
| CLS xfmr (no GRU) | 44 | 1.103 | 0.632 | 0.726 | 0.716 | 0.454 | 0.819 |
| motion + spatial+state | 7 | 1.212 | 0.596 | 0.679 | 0.680 | 0.430 | 0.792 |
| motion + spatial | 11 | 1.250 | 0.583 | 0.665 | 0.662 | 0.423 | 0.783 |
| motion + state | 8 | 1.285 | 0.572 | 0.667 | 0.669 | 0.379 | 0.784 |
| motion + state-no-weapon | 11 | 1.295 | 0.568 | 0.666 | 0.665 | 0.374 | 0.783 |
| motion + weapon | 7 | 1.319 | 0.560 | 0.657 | 0.649 | 0.375 | 0.775 |
| motion only | 11 | 1.326 | 0.558 | 0.651 | 0.647 | 0.377 | 0.773 |
| motion + target | 10 | 1.330 | 0.557 | 0.652 | 0.644 | 0.374 | 0.772 |

Reads: (1) **temporal recurrence is the single biggest per-frame lever** in this
sweep — CLS+GRU 0.699 vs memoryless CLS 0.632 (+0.065) vs best token-concat 0.596
(+0.10). (2) Among the no-temporal token forms, **spatial geometry helps a little**
(+0.038 over motion-only), **state helps a little** (+0.014), **weapon/target/
state-no-weapon are ≈ dead weight** (within ±0.01 of motion-only). (3) The **ud/jump
axis F1 is low everywhere** (0.37–0.51) — but that is an argmax artifact, not a
modelling failure (§2). (`best_selection_score` is the composite, lower=better;
it tracks loss not F1 and is not directly comparable across the two head families
because the CLS runs ran 50 ep / lr 0.003 and the token runs 12 ep / lr 0.0045.)

## 2. Jump (ud) is an argmax artifact — SETTLED

The ud axis is heavily imbalanced: human `pos_rate_move_ud_pos = 0.0135`,
`ud_none = 0.986`, `ud_neg ≈ 0.0002` (val). Argmax of any calibrated head almost
never picks the 1.35%-prior jump class, so its per-frame **F1 collapses** (ud_pos
F1 0.10–0.42) while its **expected jump mass is calibrated**. From
[`_move_joint_ablation.json`](../../runs/head_probe/_move_joint_ablation.json) on
the 0.699 CLS+GRU checkpoint:

| quantity | value |
|---|---|
| human P(jump) (`move_ud_pos_human`) | 0.0135 |
| model **expected** P(jump) (`move_ud_pos_model_exp`) | 0.0158 |
| model **argmax** P(jump) (`move_ud_pos_argmax`) | **0.000079** |

The head's expected jump mass (0.0158) ≈ the human rate (0.0135); only the argmax
**rate** is ~0. **The head is calibrated — re-weighting or threshold-tuning the
jump class would distort it.** The fix is to **sample**, not argmax. This is the
exact analogue of the look-head finding that argmax point metrics hide a calibrated
distribution. (Verified twice: the commit that introduced the metric trio and the
joint-ablation rerun both report it; it is **unaffected** by the joint-KL ordering
bug in §4.)

## 3. Momentum dominance — SETTLED

`move_momentum_baseline.py` builds pure-numpy baselines on the in-distribution
frames with episode boundaries respected
([`_move_momentum_baseline.json`](../../runs/head_probe/_move_momentum_baseline.json)).
The momentum-vs-state ladder (mean per-axis dll, nats; H_marg ≈ 0.74/axis):

| model | mean per-axis dll | "skill" (dll / H_marg) | what it uses |
|---|---|---|---|
| marginal | 0.000 | 0.00 | reference (class priors) |
| memoryless CLS xfmr | 0.304 | 0.41 | single-frame state (velocity) |
| GRU (the 0.699 model) | 0.378 | 0.51 | state + temporal memory |
| **markov1 (pure momentum)** | **0.407** | **0.55** | previous action only |

**A trivial first-order Markov chain (0.407) matches/beats the trained
Transformer+GRU (0.378).** Per-axis Markov-1 dll: fb 0.610, lr 0.600, **ud 0.010**
(jump is near-unpredictable from prev-jump alone). Copy-rates (P next action ==
prev action): **fb 88%, lr 86%, ud 98%**.

Reads: (1) single-frame velocity already recovers ~75% of the momentum signal
(velocity *is* the physical integral of recent inputs); (2) the GRU recovers ~93%
of the prev-action ceiling **without** a prev-action input (no leak); (3) per-frame
dll is **momentum-capped at ~0.41** — beating it would require feeding the previous
action, i.e. re-introducing a `prev_look`-style copy leak. So **pushing per-frame
dll is the wrong goal**; the move human-likeness lever is temporal-stream fidelity
(§5), and naive per-frame sampling would shred the 88% autocorrelation.

This is the key contrast with the other heads: for look / weapon / attack the GRU
adds genuine per-frame signal, so they want recurrence; for **move the GRU does not
beat the Markov-1 momentum baseline per-frame**, so move ≈ CLS-or-GRU on point
predictability and the GRU earns its keep only on temporal-stream coherence (§5).

## 4. Axes are independent — no joint head — SETTLED (with a bug correction)

### The "3.26" joint-gap was a diagnostic ordering bug

When the distributional metrics were first added (`cc1d0031`), the 27-bin fb/lr/ud
**joint combo KL** read **3.26 nats** on the 0.699 checkpoint — seemingly saying the
independent-axis factorization badly misses human axis co-occurrence. **This was a
binning bug, not a real gap.** In `QNNPolicy` the human joint histogram was binned
`fb + 3*lr + 9*ud`, but the model's expected joint mass came out of an
`einsum().reshape()` in C-order (`fb*9 + 3*lr + ud`) — the KL was comparing
**scrambled combo bins**. Fix (`12e01bc2`): `permute(2,1,0)` before reshape so model
mass lands on the same combo index as the human histogram.

### Corrected numbers — axes are ~independent

On the same 0.699 CLS+GRU checkpoint, in-distribution
([`_move_joint_ablation.json`](../../runs/head_probe/_move_joint_ablation.json)):

| quantity | value | meaning |
|---|---|---|
| `move_tc_joint` (total correlation) | **0.0077** | Σ H(marg) − H(joint): the *whole* joint gap an independent model could ever recover |
| real `move_kl_joint` (corrected) | **0.0027** | model joint vs human joint — **below** the TC ceiling |
| `move_kl_marg` | 0.00035 | per-axis marginal KL (already tiny) |

The real joint KL (0.0027) is **below** the total correlation (0.0077), and the TC
itself is near-noise — the fb/lr/ud axes are nearly independent in aggregate, so the
per-axis factorization already reproduces the human joint and **a joint/coupling head
buys nothing**. The offline ablation cross-validates: an exploratory `MoveJointHead`
with `{none, pairwise, full}` coupling on cheap features recovers joint KL
0.0065 / 0.0021 / 0.0011 — all noise-level, all below TC 0.0077 — confirming coupling
is unnecessary. `move_tc_joint` was added to the live combine as the diagnostic
bracket; the exploratory `src/qnn/model/move_joint.py` was kept but **NOT promoted**.

(Note: the `_move_joint_ablation.json` `move_dll` values here are ~0.20, lower than
the §3 ladder's ~0.38 — that sidecar is an offline no-encoder ablation on 400k train
frames, a different scaffold from the in-loop checkpoint eval. Use it for the TC /
joint-KL geometry, not for the dll ladder; for dll use §3's `_move_momentum_baseline.json`.)

## 5. Distributional human-likeness — the right objective

Because move is momentum-capped (§3) and jump is argmax-fragile (§2), the project
selects on `move_dll` (a proper scoring rule) and the **human-likeness** target is
the *temporal-stream distribution*: per-axis dwell-times, switch-rates, and the ~88%
autocorrelation — NOT per-frame argmax F1. `humanlikeness_move_cls_gru.py` runs the
0.699 CLS+GRU checkpoint over val episodes **preserving temporal order** (carrying
the GRU hidden state, not shuffled BPTT batches) and compares three streams —
**human**, **argmax**, **per-frame independent sample** — on dwell-time EMD/KS and
switch-rate ([`_humanlikeness_move_cls_gru.json`](../../runs/head_probe/_humanlikeness_move_cls_gru.json),
1,014,059 in-dist frames, 16,292 episodes):

| axis | stream | switch rate | rate ÷ human | dwell median | dwell mean | dwell EMD vs human |
|---|---|---|---|---|---|---|
| fb | human | 0.1217 | — | 5 | 7.36 | — |
| fb | argmax | 0.1784 | 1.47× | 3 | 5.22 | 2.15 |
| fb | sampled | 0.3309 | **2.72×** | 1 | 2.93 | 4.44 |
| lr | human | 0.1385 | — | 5 | 6.56 | — |
| lr | argmax | 0.1935 | 1.40× | 3 | 4.84 | 1.72 |
| lr | sampled | 0.3587 | **2.59×** | 1 | 2.71 | 3.85 |
| ud | human | 0.0191 | — | 18 | 28.69 | — |
| ud | argmax | 0.0046 | 0.24× | 38 | 48.66 | 19.97 |
| ud | sampled | 0.0216 | 1.13× | 15 | 26.83 | **1.87** |

Reads — **neither pure stream is human-like, and the right choice is axis-dependent**:

- **fb/lr (the high-switch axes):** per-frame **independent sampling over-switches
  ~2.6–2.7×** and shreds the dwell distribution (median 1 vs human 5; EMD ~4) —
  exactly the autocorrelation-destruction hypothesis. **Argmax is closer** (1.4–1.5×
  switch, EMD ~1.7–2.2) but still over-switches and under-dwells (it has no hold
  inertia of its own beyond what the GRU state carries).
- **ud/jump (the rare-switch axis):** the reverse — **argmax catastrophically
  under-switches** (0.24× rate, dwell median 38 vs 18, EMD ~20: it basically never
  jumps, the §2 collapse), while **sampling nails it** (1.13× rate, EMD 1.87,
  KS 0.055 — the only near-human cell in the table).

So the human-likeness operating point is **not** "always argmax" or "always sample":
fb/lr want something *stickier* than per-frame sampling (the GRU's temporal hold,
or an explicit hold/switch process), and jump wants sampling. This is the move
analogue of the persistence / change-point structure documented for look and weapon
(a WHEN-to-switch process layered on a WHAT distribution), and it is **OPEN** —
no hold/switch move head has been built or measured yet.

## 6. What's settled vs open

**SETTLED:**
- **Axes ~independent → no joint head** (§4). TC 0.0077, corrected joint KL 0.0027.
  The "3.26" was a combo-ordering bug; documented and fixed.
- **Jump collapse is an argmax artifact → sample** (§2). Expected jump mass 0.0158 ≈
  human 0.0135; argmax rate 0.00008. Head is calibrated; do not re-weight.
- **Momentum / Markov-1 dominance** (§3). Markov-1 per-axis dll 0.407 ≥ GRU 0.378;
  per-frame dll is momentum-capped at ~0.41; pushing past = prev-action leak.
- **Distributional objective** (§5). Select on `move_dll`; judge human-likeness on
  dwell/switch-rate/autocorrelation, not per-frame F1. Move is the head where
  recurrence does **not** clearly help per-frame (contrast with look/weapon/attack).
- **Feature reads** (§1): temporal is the dominant per-frame lever; spatial + state
  add a little; weapon / target / state-no-weapon ≈ dead weight for move.

**OPEN / UNVERIFIED:**
- **The hold/switch move head** (§5). The data calls for a stickier fb/lr generator
  (temporal hold or explicit WHEN-switch process) + sampled jump, but no such head
  is built or measured. The argmax-vs-sampled streams bracket it; the synthesis is
  untested.
- **Canonical move head.** There is no dedicated `src/qnn/model/move_head.py`
  findings doc beyond the shared `MoveHeadInput/Output` interface and the
  unpromoted `move_joint.py` exploratory; all numbers here are from **bench** probes.
- **Cross-head composite rebalancing.** `move_dll` is in nats to match `look_dll`
  in the composite, but the multi-head balance with weapon/attack/look at deployment
  is not characterised here.
- **dll provenance mismatch.** The §3 ladder (~0.38–0.41) and the §4 ablation
  sidecar (~0.20) use different scaffolds/sample sizes; the in-loop per-epoch dll was
  never persisted to `bc_history`, so the *per-run, per-epoch* dll trajectory cannot
  be reconstructed from artifacts — only the post-hoc checkpoint evals.

## Regenerate

- **Momentum / Markov-1 ladder** (§3): `scripts/analysis/move_momentum_baseline.py`
  — pure-numpy Markov-1 / marginal baselines on in-distribution frames, episode
  boundaries respected. Output → [`_move_momentum_baseline.json`](../../runs/head_probe/_move_momentum_baseline.json).
- **Joint / TC ablation** (§4): `scripts/analysis/move_joint_ablation.py` — offline
  no-encoder ablation; cross-validates TC and `{none,pairwise,full}` coupling. Output
  → [`_move_joint_ablation.json`](../../runs/head_probe/_move_joint_ablation.json).
- **In-distribution distributional eval** (§2, §4 in-loop numbers):
  `scripts/analysis/_move_dist_eval.py` — injects the bench probe factory, builds
  sources via config so `segment_mask` + `input_mask` match training (the built-in
  `--eval-only` can't load a bench head and skips `segment_mask`, giving OOD numbers).
  Run in the trainer container at the checkpoint's training commit.
- **Temporal human-likeness** (§5): `scripts/analysis/humanlikeness_move_cls_gru.py`
  — sequential per-episode forward (GRU state preserved) → dwell/switch-rate/EMD/KS
  for human/argmax/sampled. Output → [`_humanlikeness_move_cls_gru.json`](../../runs/head_probe/_humanlikeness_move_cls_gru.json).
  Run in the trainer GPU container.
- **Feature ranking** (§1 reads): `scripts/analysis/_move_feature_ranking.py` —
  per-group Pearson r + linear-probe accuracy, ranks what's worth a neural probe.

Bench runs (retrain in-distribution via the bench daemon, seed 17): the CLS family
`head_probe_move_cls_gru_xfmr_d64_h2_seed17` (with GRU) and
`head_probe_move_cls_xfmr_d64_h2_seed17` (no GRU); the token family
`head_probe_move_cls_motion_token_d64_seed17` (+ `_target`, `_state`,
`_state_no_weapon`, `_weapon`, `_spatial`, `_spatial_state` variants via the
mutually-exclusive second-token knobs in `probe.json`).

## Live-play decode finding — v24 lateral jitter (2026-06-09)

v24 (`head_probe_full_4head_seed17`) in live play "jitters" laterally — taps
left ~100× and right ~30× across 130 frames instead of holding the key. Root
cause is **per-frame independent sampling breaking the human hold
autocorrelation**, NOT a training deficit on fb/lr. Measured by
[`scripts/analysis/move_temporal.py`](../../scripts/analysis/move_temporal.py)
over the full contiguous val set (→ [`_move_temporal.json`](../../runs/head_probe/_move_temporal.json)):

| axis | top-class prob | human switch | sampled switch | sticky τ0.6 switch |
|------|---------------|-------------|----------------|--------------------|
| fb | 0.85 | 9.3% | 27.0% | 10.3% |
| lr | 0.81 | 13.1% | 34.8% | 13.7% |

The fb/lr per-frame distributions are **peaked** (model knows the key); sampling
re-rolls them and inflates switching ~3×. Plain argmax also over-switches
(14–19%, the mode itself flickers). A **sticky/hysteresis decode (hold unless the
argmax-class confidence ≥ τ≈0.6)** lands on the human switch rate and preserves
directional balance.

**Locked decode (no retrain):** sticky τ=0.6 on fb/lr, **sampling on ud**
(sticky/argmax would worsen jump — see below).

IMPLEMENTED (2026-06-10): the sticky decode is **stateful** (needs the previous
emitted class), so it runs engine-side (Pattern B) — `qnn_onnx.c
qnn_onnx_decode_core` holds `prev_move` per axis and switches only when the
argmax-class softmax prob ≥ τ; τ travels with the model as the stamped
`decode.move_sticky_tau` (default 0.6). The export emits raw fb/lr logits and
gumbel-perturbs only the jump row so the engine's argmax on ud is a sample. See
[`look-head.md`](look-head.md#4-design-principle--the-decode-regime-lives-in-the-model)
for the Pattern A (in-graph) vs Pattern B (engine reads stamped param) split. The
jump training gap below is unchanged.

### Jump (ud) is a training/perception gap, not decode-fixable

ud is near-deterministic "none" (top-class 0.98). Both argmax (1.2%) and sampling
(2.6%) under-fire jump vs human (7.2%), and
[`scripts/analysis/jump_discrim.py`](../../scripts/analysis/jump_discrim.py)
(→ [`_jump_discrim.json`](../../runs/head_probe/_jump_discrim.json)) shows the
model has **no jump-initiation policy**: AUC(P(jump) vs human jump) = **0.47**
(≈ random). The conditional-mean gap (0.134 vs 0.018) is a heavy-tail artifact,
not rank discrimination. Only weak *persistence* is learned (AUC 0.69 in the 1–8
ticks after a jump; human jumps are 82% clustered within 3 frames = hold/bunny-
hop continuation). No decode recovers an absent policy — jump needs a
training/representation fix (likely an obs-perception question: does the model
see gap/ledge geometry?), parked separately from the four-axis decode work.
