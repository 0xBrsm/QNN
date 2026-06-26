# Look head: findings & design

What drives the look (view-turn) axis, why v24's live "spins" are a **decode**
artifact rather than a head/training defect, and the decode regime that fixes
them. This doc focuses on the live-play spin investigation; the polar look-head
distribution work (POLAR mag×dir + explicit hold bin won the held-out Δloglik
sweep) and the `look_r2` / momentum framing are covered in
[`persistence-and-changepoints.md`](persistence-and-changepoints.md) and the
project look memories. Numbers measured on `artifacts/collect/qwd` (val split,
unfiltered `segment_mask=None`, 500 episodes, 333,163 turning frames unless
noted). Regenerate with
[`scripts/analysis/look_ground_spin.py`](../../scripts/analysis/look_ground_spin.py)
→ [`_look_ground_spin.json`](../../runs/head_probe/_look_ground_spin.json).

## TL;DR

v24 (`head_probe_full_4head_seed17`, polar look head) "looks off and whips the
view around" in live play. The head is **fine** — its per-frame turn-magnitude
distribution is human, it is target-grounded as well as a human, and it is not
hurt by the missing target head. The spin is a **decode artifact**: per-frame
independent sampling of the direction bin (plus 16-bin quantization) shreds the
frame-to-frame heading-hold the head actually learned, tripling the directional
reversal rate vs human. The fix is a **decode regime change, no retrain**: keep
sampling the magnitude bin (preserves the human turn-size distribution) but take
the heading from the **continuous circular mean of the direction softmax**
(unquantized, defined every frame — preserves the human heading-hold). This is the
look analogue of the move lateral-jitter finding
([`move-head.md`](move-head.md#live-play-decode-finding--v24-lateral-jitter-2026-06-09)):
the per-frame distribution is right, but independent sampling breaks the temporal
hold the head learned.

## The complaint

v24 plays well but "arbitrarily looks off and fires into the void," and "spins"
— sometimes whips the view around. Two standing hypotheses: (1) the spin is the
sampler drawing oversized turns (a turn-magnitude / tail-mass defect); (2) look
is **ungrounded** because there is no target head to anchor *which* entity to
look at (suspected because it is "less of a problem when `qnn_fov` is high" =
more entities in sight). A third arose from the fire investigation: v24 trained
on target-present frames only (`act.target != 0`), so the spins might be the
**OOD no-target regime** (~74% of live frames), the same root as void-firing.

**All three are refuted.** The first two by the in-distribution cuts below; the
third because the no-target split is no worse than target-present.

## 1. The head is fine — three refuted hypotheses

### Turn magnitude is human (refutes "over-sampling")

Sampled turn-magnitude (draw a mag bin → `MAG_CENTERS` magnitude) vs human, over
all turning frames:

| flick | human | sampled |
|---|--:|--:|
| ≥15° | 8.87% | 8.05% |
| ≥45° | 0.57% | 0.69% |
| ≥90° | 0.098% | **0.013%** |
| EMD vs human | — | **0.91°** |

The model produces *fewer* ≥90° whips than humans and the whole distribution is
<1° EMD from human. The spin is **not** oversized turns.

### Look is target-grounded (refutes "ungrounded / needs a target head")

On target-present turning frames (n=28,012), cosine alignment (tangent space)
between `look_predict` and the direction to the argmax-`target_probs` token:

| | mean cos | frac cos>0.5 |
|---|--:|--:|
| **human** (ceiling) | 0.316 | 58.2% |
| **model** | **0.335** | 59.9% |

The model aims at the target *as well as the human does*, and does not degrade as
actors multiply (1→0.336, 2→0.342, 3+→0.313). The look head **implicitly** learned
target-directed turning via entity conditioning; the missing target head does not
leave it ungrounded in-distribution.

### No-target frames are no worse (refutes "OOD no-target")

Presence-stratified turn magnitude:

| split | n | sampled EMD vs human |
|---|--:|--:|
| target-present | 86,665 | 0.83° |
| no-target | 246,498 | 0.93° |

The no-target EMD (0.93°) is barely above target-present (0.83°), and the model
again under-produces big turns in both. Look does not go haywire on no-target
frames offline — unlike fire, the OOD-no-target story does not hold for look.

## 2. The real cause — directional hold-breaking (SETTLED)

Every magnitude cut is human, so the spin is **temporal**, not per-frame. Humans
hold a heading; the metric is the cosine between *consecutive* turn directions
(both frames real turns, ≥3°). Three streams — human, the head's deterministic
`look_predict` (mean), and per-frame independent (mag,dir) sampling:

| stream | mean consec-cos | % persist (>0.5) | % reversal (<−0.5) |
|---|--:|--:|--:|
| **human** (heading-hold ceiling) | 0.916 | 95.6% | 2.1% |
| **mean** (`look_predict` deterministic) | 0.908 | 95.7% | 2.7% |
| **sampled** (per-frame mag+dir draws) | **0.759** | 85.6% | **6.2%** |

Two conclusions:

1. **The head learned heading-hold.** Its deterministic stream is essentially
   human (0.908 vs 0.916). Nothing to fix in the architecture or training.
2. **Per-frame sampling destroys it.** Independent (mag,dir) draws drop
   consecutive-cosine 0.916→0.759 and **triple the reversal rate (2.1%→6.2%)**.
   Those reversals — heading flips of >90° between adjacent frames — integrating
   at frame rate **are the live spins.**

A secondary cause compounds it: the direction head has only **16 bins (22.5°
apart)**, so *any* discrete-bin decode (sampled or argmax) snaps heading to a bin
center and hops between adjacent bins — capping persistence below ~0.86 even at
low temperature. The continuous `look_predict` (a softmax-weighted heading) is not
quantized, which is why the mean stream reaches 0.908.

## 3. The decode regime — hybrid (sampled mag × continuous dir)

Candidate decodes, on persistence **and** turn-magnitude (both must be human).
Magnitude is sampled in every hybrid (preserving the human turn-size spread —
flicks and EMD exactly the current sampler's); the variants differ only in how
direction is taken:

| decode | consec-cos | reversal | flick≥15° | mag EMD | verdict |
|---|--:|--:|--:|--:|---|
| **human** (target) | 0.916 | 0.021 | 0.089 | — | — |
| sample mag+dir, τ=1 (**old live**) | 0.759 | 0.062 | 0.081 | 0.908 | spins |
| `look_predict` only (greedy) | 0.908 | 0.027 | **0.058** | **1.186** | persistent but **under-turns** |
| decoupled (mag τ=1, dir τ↓) | ≤0.855 | 0.044 | 0.081 | 0.908 | quantization-capped |
| hybrid: mag × `look_predict` dir | 0.887 | 0.033 | 0.081 | 0.908 | **not deployable** (see below) |
| hybrid: mag × **argmax** dir | 0.863 | 0.044 | 0.081 | 0.908 | deployable; quantized |
| **hybrid: mag × circular-mean dir** | **0.872** | **0.032** | **0.081** | **0.908** | **IMPLEMENTED** |

A global sampling temperature cannot win: persistence wants low τ, magnitude
fidelity wants τ=1, and they trade off (lowering τ sharpens the magnitude that was
already human, inflating its EMD to 1.11 and suppressing big flicks). Mag and dir
come from **separate** logit heads, so the fix treats them separately:

- **Magnitude:** sample the mag bin (the human turn-size spread).
- **Direction:** the **continuous circular mean** of the direction softmax,
  `φ = atan2(Σ pᵢ·sin φᵢ, Σ pᵢ·cos φᵢ)` over the 16 dir-bin centers — unquantized
  and defined on every frame.

`look = expmap( θ_sampled · (cos φ, sin φ) )`. The circular-mean direction cuts
the reversal rate in half (0.062→0.032, nearest human 0.021) with the magnitude
distribution untouched (EMD 0.908). It beats argmax-dir (0.863 / 0.044 — the 22.5°
quantization injects heading hops) and is *deployable*, unlike the `look_predict`-dir
proxy: `look_predict` is the **argmax reconstruction of both bins**
([`look_head_polar.py:55`](../../src/qnn/model/bench/look_head_polar.py#L55)), so its
direction is undefined (zeroed) whenever argmax-magnitude is *hold* — on a frame
where the **sampled** magnitude turns but argmax-magnitude holds, that decode emits
a zero direction and silently drops the turn. Its 0.887 number is inflated by those
dropped frames; the circular mean has no such coupling.

## 4. Design principle — the decode regime lives in the model

**The head output regime is part of the model's contract, not an engine-side
decision.** The exported graph must emit the *final* look vector under its own
decode protocol so the weights carry their own decode; the engine is a dumb
consumer of the single `LOOK` output ([`qnn_onnx.c:92`](../../src/engine/common/qnn_onnx.c#L92)
`QNN_ONNX_OUT_LOOK`). A given checkpoint's decode regime travels with its
weights — no engine rebuild or per-model engine branch when the decode changes.

Implemented in two mirrored sites (the offline path must match the deployed
graph): the deployed ONNX decode is
[`tools/export_onnx.py`](../../tools/export_onnx.py) `ExportWrapper._polar_to_lookvec`
(Gumbel-sampled magnitude bin × circular-mean direction, in-graph), and the
offline/eval path is [`policy.py:544`](../../src/qnn/model/policy.py#L544)
(`emit_actions` sampled branch, same formula). The exporter's PyTorch↔ORT parity
check passes (logits within 1e-5), and an ORT smoke on the deployed graph confirms
the `look` output is unit-norm, direction is deterministic across samples (the
circular mean), and magnitude varies (the Gumbel sampler) — i.e. the regime lives
entirely in the model/export and the C client is untouched.

## What's settled vs open

**SETTLED:**
- **The head is fine** (§1): turn magnitude human (EMD 0.91°), target-grounded at
  the human level (cos 0.335 vs 0.316), no-target no worse. "Over-sampling,"
  "ungrounded," and "OOD no-target" all refuted.
- **The spin is directional hold-breaking** (§2): the head's deterministic heading
  is human-persistent (0.908); per-frame dir sampling + 16-bin quantization triple
  the reversal rate (2.1%→6.2%). It is the move-jitter mechanism on the look axis.
- **The fix is the hybrid decode** (§3): sampled magnitude × continuous
  circular-mean direction — near-human on persistence (consec-cos 0.872, reversal
  0.032) with the magnitude distribution untouched.
- **Decode regime lives in the model** (§4): IMPLEMENTED in the exported graph
  (`export_onnx.ExportWrapper`) and the mirrored offline path (`policy.py`);
  parity passes (1e-5), ORT smoke confirms the deployed behavior; engine consumes
  one LOOK vector.

**OPEN:**
- **Live validation**: deploy move (sticky τ=0.6) + look (hybrid) together and
  re-measure jitter/spins and `obs_blind_fire_rate` (the attack head inherits the
  look fix — see [`fire-discrimination.md`](fire-discrimination.md#6-look-aim-is-the-attack-lever-2026-06-09)).
- **Residual gap**: hybrid persistence 0.872 vs human 0.916 — within the human
  band, but the closed-loop test is the real arbiter.
- **Move sticky-τ in the graph**: move's hysteresis decode (τ=0.6) is stateful
  (needs the previous action); unlike look it is not yet baked into the export.

## Regenerate

[`scripts/analysis/look_ground_spin.py`](../../scripts/analysis/look_ground_spin.py)
— per-episode sequential forward (GRU hidden carried), polar head captured via a
forward hook. Emits turn-magnitude flicks/EMD (overall + presence-stratified),
target grounding by actor count, directional persistence (human/mean/sampled),
and the decode sweeps (coupled τ, decoupled dir-τ, hybrid). Output →
[`_look_ground_spin.json`](../../runs/head_probe/_look_ground_spin.json). Run in
the trainer GPU container (`agents/skills/gpu/scripts/run.sh`).
