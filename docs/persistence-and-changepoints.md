# Persistence and Change-Points — the Shared Structure of QNN's Control Heads

Every behavior-cloning head in QNN — look, move, weapon, attack, target — predicts a stream that is overwhelmingly *persistent*: frame to frame, the human almost always repeats the previous decision. The information lives in the rare **change-points** (start/stop strafing, flick the aim, switch weapon, re-target, begin/end an engagement). This document records the evidence that all heads share this structure, why per-frame metrics mislead under it, the current per-head status, the literature framework for modeling it, the cautionary precedent of the removed tactics head, and the resulting plan.

The objective throughout is **human-likeness**: the generated action stream should be distributionally indistinguishable from a human's, not maximally accurate per frame and not maximally winning. See [Human-Likeness Objective](#) in the project memory and the per-head findings below.

## The Core Finding: Heads Are Persistence-Dominated

Measured in-distribution (target-present frames, `input_mask` applied, `artifacts/collect/qwd` val):

| Head | Temporal input? | Label persistence (frame-to-frame repeat) | Per-frame metric |
|------|-----------------|-------------------------------------------|------------------|
| Move (fb / lr / ud) | yes (GRU) | 88% / 86% / 98% | `move_dll` 0.378 nats |
| Look | yes (GRU) | copy-previous turn ~0.99 cosine | `look_dll` (turn buckets) |
| Target | no (within-frame attention) | 99% (switch rate 0.88%, dwell ~2 s) | `val_target_kl` 0.034, `acc_target` 0.986 |
| Weapon | no (dense, no GRU) | high (engagement dwell) | `f1_weapon_global` 0.72 |
| Attack | no (cooldown state input only) | engagement-level (sustained fire) | `f1_attack` 0.54 |

The shared shape is a low-entropy, autocorrelated stream where "hold" dominates and the decisions worth modeling are sparse transitions. But two distinct things must not be conflated:

- **Label-stream autocorrelation** — the decision repeats frame to frame because the *situation* does. This is a property of the data, present for every head.
- **A model exploiting that autocorrelation as a shortcut** — the copycat / momentum pathology. This can only happen with a temporal or history input.

The two are independent. **Move and look are temporal**, so for them the predictable signal is mostly momentum and the per-frame ceiling is set by autocorrelation (the next section). **Target, weapon, and attack are memoryless** — no GRU, no previous-decision input (attack sees only current cooldown state). A memoryless head *cannot* exploit persistence; it has no access to its previous output. Its per-frame success is genuine current-state grounding, not a copy shortcut, and its persistent *output* is a downstream consequence of the situation being persistent. The one real question for memoryless heads is therefore not "is the metric inflated" but "does the chained output reproduce human switch/dwell dynamics" — addressed under Per-Head Status and the plan.

### The Momentum Cap

For move, a decomposition makes the trap explicit (mean per-axis Δloglik vs the human marginal):

| Model | `move_dll` | `move_skill` | Knows |
|-------|-----------|-------------|-------|
| marginal (base rate) | 0.000 | 0.00 | nothing |
| memoryless CLS transformer | 0.304 | 0.41 | single-frame state (velocity) |
| GRU (the 0.698 probe) | 0.378 | 0.51 | state + temporal memory |
| Markov-1 (previous action only) | 0.407 | 0.55 | the previous action |

A trivial first-order Markov chain matches or beats the trained transformer+GRU. Single-frame velocity already recovers ~75% of the achievable signal, because velocity is the physical integral of recent inputs. The per-frame predictive ceiling (~0.41 nats) is essentially the autocorrelation of the stream. Note this is a property of move's *temporal* setting — the Markov-1 baseline uses the previous action as input. It does not transfer to the memoryless heads: target's `acc_target` 0.986 is achieved *without* any previous-target input, so it is genuine current-state grounding, not a Markov-style ride on the 99% persistence. (A copy-previous-target baseline would score ~99%, i.e. higher — yet the memoryless head reaches near that from current state alone.)

## Why Per-Frame Metrics Mislead

Three consequences follow, and they are the reason this document exists.

1. **Per-frame accuracy / log-likelihood is momentum-capped.** Pushing `move_dll` past the Markov ceiling requires feeding the previous action, which is a leak (next section). The trained heads are already near the honest ceiling.

2. **Conditioning on the previous action is the copycat trap.** Feeding the previous action (or anything that recovers it, such as `prev_look`) lets the model "cheat by predicting the previous action rather than the next." This is causal confusion: more history can yield *worse* control. QNN hit this directly with the `prev_look` input. References: Wen et al., *Fighting Copycat Agents* (NeurIPS 2020, https://arxiv.org/abs/2010.14876); de Haan et al., *Causal Confusion in Imitation Learning* (NeurIPS 2019, https://arxiv.org/abs/1905.11979); Chuang et al., RAP (ECCV 2022, https://arxiv.org/abs/2207.09705).

3. **The objective is distributional, not per-frame.** Human-likeness is whether the generated stream's *temporal statistics* match a human's — dwell-time and switch-rate distributions, not per-frame hit rate. Naive per-frame sampling (even from a perfectly calibrated per-frame distribution) destroys the 88–99% autocorrelation and produces jittery, switch-happy, non-human behavior.

## Per-Head Status

| Head | Best result | Status | Real open problem |
|------|-------------|--------|-------------------|
| Move | `f1_move` 0.70, `move_kl_marg` 0.0005, `move_kl_joint` 0.003 | solved as a dense head | temporal-stream fidelity; sampling coherence |
| Look | distributional `look_dll`/POLAR head | largely solved | same (momentum stream) |
| Weapon | `f1_weapon_global` 0.72, plateaued, linear ≈ MLP | solved as a dense head | no measured usage-distribution match |
| Attack | `f1_attack` 0.54 (beats 0.34 geometric oracle) | solved to label-noise ceiling | none at per-frame level; it is target-gated |
| Target | `val_target_kl` 0.034, `acc_target` 0.986 | solved as a memoryless head | **output switch-dynamics unmeasured (over-switch at boundaries?)** |

#### Move — solved as a dense head

Marginals match the human distribution (`move_kl_marg` 0.0005); the fb/lr/ud axes are nearly independent (total correlation `move_tc_joint` 0.0077) so the corrected joint gap `move_kl_joint` is 0.003 — there is no joint-combo problem (an earlier "3.26" was a combo-ordering bug in `qnn/model/policy.py`, since fixed). Jump "collapse" is an argmax artifact of a calibrated head: mean softmax `P(jump)` 0.0137 ≈ human 0.0135 while argmax-jump ≈ 0. Selection now uses `move_dll`, not `f1_move`.

#### Weapon — solved, but the shotgun question is unsettled

`f1_weapon_global` is flat at ~0.72 across seeds and epochs, and a linear probe matches the MLP, so the weapon-token features are near-linearly separable and the head is at its ceiling. Shotgun is the majority action (57.5% of target-present frames), but the solved token head *under*-recalls it (precision 0.898, recall 0.727) — the opposite of a blind default. A shotgun collapse appears only in the under-featured `weapon_dense_noembed` probe. If a shotgun default shows up in rollout, suspect the inference-side sticky-weapon controller (`qnn/model/policy.py`) or sampling temperature, not the head. The genuine gap: no logged divergence between the predicted and human weapon-usage histograms — "matches the human distribution" is currently inferred from per-class precision/recall, not measured.

#### Attack — solved to the irreducible ceiling

`f1_attack` ≈ 0.54 already beats the best hand-tuned geometric/hit-test oracle (0.34); the residual is label stochasticity (humans fire at moments not deducible from geometry), not a model bottleneck. Attack is gated by `segment_mask act.target != 0` and dominated by actor `dist`/`rel` (target geometry) plus the `attack_finished` cooldown gate. It is downstream of target by construction.

#### Target — solved as a memoryless head; only switch-dynamics untested

Target identity is the most persistent of all: 99% frame-to-frame, switch rate 0.88%, dwell mean ~39 frames (~2 s). But the head is **memoryless** — within-frame attention over entity tokens, no GRU, no previous-target input — so it does not exploit that persistence. `acc_target` 0.986 is achieved purely from current-frame state, which is the strongest possible position: it re-derives the right target every frame without a temporal crutch. The per-frame metric is honest, not inflated.

The single untested question is the **chained output**: re-deciding independently each frame with no commitment can over-switch at ambiguous boundaries (two similarly-attractive enemies → frame-to-frame flip-flop, where a human commits for ~2 s). Whether the memoryless head's output switch-rate matches the human 0.88% is unmeasured. This is empirical, not assumed — and the fix, if needed, is a minimal commitment layer, not a temporal head.

## Target Gates the Engagement Subsystem

Fire is target-gated and target-geometry-driven; weapon is conditioned on the target. Target's change-points (acquire / switch / lose) *are* engagement onset and offset, which gate the entire fire subsystem. So target is the upstream signal for the engagement, and it is the natural place to look first if any human-likeness gap traces to engagement timing. But this is an architectural observation, not a claim that target needs a temporal/WHEN head: as a memoryless head it may already produce human-like switching. The when/what machinery below is the answer for the *temporal* heads (move, look) and for any memoryless head that the evaluation shows over-switches — it is not a premise for target.

## The WHEN / WHAT Framework

The literature converges on one answer for densely-persistent control streams: **model duration explicitly and decouple "when to change" from "what to change to,"** with guards against the copycat leak. See the persistence/change-point reference in project memory for the full citation set.

#### Decoupling when from what

- **Explicit-duration / hidden semi-Markov models** (R-HSMM, ICLR 2017, https://openreview.net/forum?id=HJGODLqgx; DNN-HSMM for TTS, https://arxiv.org/pdf/2108.13985): a duration counter forces a "hold" until it expires, then re-decides — separating how long a state persists from its content. TTS adopted this because plain sequence models lack explicit duration handling.
- **Options / semi-MDP** (Sutton, Precup & Singh, 1999, https://www.sciencedirect.com/science/article/pii/S0004370299000521): an option pairs a policy (what) with a termination condition β (when to stop). The β-termination head is the natural change-point indicator.
- **Action repetition** (TempoRL, ICML 2021): a skip-policy predicts how long to commit to the chosen action, conditioned on that action.

#### Predicting the rare change-point

- **Focal loss** (https://arxiv.org/abs/1708.02002) and **class-balanced loss** (https://arxiv.org/abs/1901.05555): keep the rare "change" frames from being swamped by the "hold" majority on a binary change-vs-hold head.
- **WTTE-RNN / discrete-time hazard** (Weibull time-to-event): a "frames-until-next-change" head emitting a duration distribution with right-censoring, trained without conditioning on the held action — so it carries no copycat leak and is samplable.
- **Bayesian Online Change-Point Detection** (https://arxiv.org/abs/0710.3742): an offline segmenter to auto-label change-points and dwell durations from the human streams.

#### Copycat guards (mandatory)

Any head that re-injects the previous action — including a naive duration counter that leaks the held value — reinstates the copycat degeneracy. Remedies: RAP's residual-action architecture (predict the *change*, not the absolute value) and Wen et al.'s adversarial removal of previous-action information from the features. de Haan's intervention fix needs online/expert access and does not apply to offline BC.

## Evaluating Human-Likeness

Per-frame accuracy is explicitly rejected in this literature. Score the *distributions* of temporal statistics instead.

| Statistic (per channel) | Why it matters |
|-------------------------|----------------|
| Dwell-time / hold-duration | most diagnostic for a repeat-dominated stream |
| Switch-rate / inter-event interval | catches over/under-switching (the BC jitter failure) |
| Turn-magnitude + bout duration | look stream (consistent with the turn≥15° buckets) |
| Spatial occupancy | top movement-realism metric in shooter work |
| Action co-occurrence | unnatural simultaneity |

Distances and tests: 1-Wasserstein / EMD as the headline scalar (native units) with a Kolmogorov-Smirnov or permutation p-value for 1-D statistics; MMD with a bootstrap permutation test for the overall "indistinguishable from human?" verdict (framework: https://arxiv.org/abs/2203.05965). Validate against a pairwise human Turing test (HNTT, CHI 2023, https://arxiv.org/abs/2303.02160; BotPrize). Shooter precedents using EMD/JS on these exact statistics: MLMove (https://arxiv.org/abs/2408.13934) and tactical-shooter human-like bots (https://arxiv.org/abs/2501.00078). BotPrize's lesson is instructive: bots that were *too* precise read as non-human.

## Prior Art — the Tactics Head (and Why It Failed)

QNN already attempted one level above per-frame control: the **tactics head**, a dense per-frame 3-way combat-posture classifier (approach / hold / retreat) in the v11–v15 BC era. It peaked at ~67% val accuracy and was removed (CHANGELOG 0.18.0), replaced by deriving the same labels post-hoc from the move head. The lessons constrain any renewed effort:

1. **Do not re-derive what a lower head already emits.** Approach/hold/retreat was a deterministic function of the move vector and target geometry, so a separate head was pure redundancy. A new temporal-abstraction head must predict something *not* recoverable post-hoc from the per-frame heads.
2. **Hand-tuned threshold labels hit a noise ceiling.** Savitzky-Golay smoothing plus a Schmitt-trigger cut concentrated label noise at the decision boundary, capping learnable accuracy. Prefer event-anchored labels over thresholded continuous signals.
3. **Watch for majority-class collapse.** Soft labels "improved" accuracy only by collapsing toward the majority (hold) class while retreat recall fell. Use per-class recall / macro-F1, not aggregate accuracy.
4. **Match training to deployment.** The head trained on smoothed labels, but there is no smoothing in the loop at inference — a systematic train/deploy mismatch.
5. **Prove downstream benefit.** Tactics never reached PPO and was BC-diagnostic only. The abstraction that survived was **target** (who am I engaging), which conditions move/look/fire/weapon via `target_feat`. The team's own conclusion was that target, not posture, is the useful above-per-frame structure — which is exactly where the persistence analysis points.

## Plan

Sequenced so each step yields a usable result and respects the tactics lessons and the copycat guard.

1. **Build the human-likeness evaluation suite first.** It is the objective and it is cheap: human reference dwell-time / inter-event-interval / turn distributions come straight from the demo labels. Sharpest first experiment: sample the model's cached per-frame softmaxes *sequentially* and measure the resulting dwell/switch distributions against human, scored by EMD/KS — this directly quantifies the per-frame-sampling jitter prediction with no environment rollout.

2. **Add a weapon-usage distribution metric.** Log the divergence between the predicted and human weapon histograms (the way `move_kl_marg` was added). This confirms weapon human-likeness and settles whether any shotgun skew is real or inference-side.

3. **Measure the memoryless heads' switch-dynamics (target, weapon, attack).** From step 1's machinery, compare each head's output switch-rate / dwell-time distribution against human. If they match, the pure per-frame design wins and these heads are done. This is the test of the memoryless-head hypothesis, not a commitment to change anything.

4. **Add a minimal commitment layer only where step 3 shows over-switching.** Prefer a small hysteresis (like the existing inference-side sticky-weapon controller) over a temporal head. Do not bolt WHEN/WHAT machinery onto a memoryless head that already produces human-like switching — that is the redundancy the tactics post-mortem warns against.

5. **Apply the WHEN/WHAT framework to the temporal heads (move, look) and to any head step 3 flags.** Design constraints: event-anchored labels (not thresholded continuous signals); per-class / macro metrics (not aggregate accuracy); train at the deployment frame rate (no smoothing-in-loop mismatch); no previous-action conditioning (RAP residual or adversarial info-removal); a defined path by which any abstraction conditions the per-frame heads, with downstream benefit demonstrated rather than assumed.

6. **Fix inference sampling coherence.** Per-frame independent sampling shreds the autocorrelation; sampling must preserve hold-runs, and the jump finding's "sample, don't argmax" needs this temporal-coherence qualifier.
