# Target head: findings & design

Which enemy the player is engaging, how the labeler supervises it, and why the
canonical metric for it is **KL — never `acc_target`**. Numbers measured on
`artifacts/collect/qwd` (val split, segment `act.target != 0`, input_mask=true,
n ≈ 893k labeled frames). Bench code:
[`src/qnn/model/bench/target.py`](../qnn/model/bench/target.py); labeler:
[`src/qnn/bc/target_labeler.py`](../qnn/bc/target_labeler.py); canonical pointer:
[`src/qnn/model/target.py`](../qnn/model/target.py).

## TL;DR — the reframe

The target head is a **soft pointer over entity tokens**, not a classifier. It
is supervised by a **distribution labeler** (`label_enemy_target_probs`, v3) that
emits a 17-class confidence distribution `(NO_TARGET, idx_0..idx_15)` per frame —
never a one-hot argmax. Two facts dominate everything else:

1. **Entity-token order is arbitrary** (engine emit order ≈ edict number), so
   `acc_target*` / per-slot F1 are confounded by a slot-0 majority effect
   (~46–97% of frames point at slot 0 depending on segment). The settled
   selection metric is **`target_kl`** (present-weighted NLL − label entropy),
   and the discrimination metric is **`target_kl_multi`** (KL restricted to
   frames with >1 live enemy). The slot-keyed metric family was deliberately
   **dropped** from the bench head ([`target.py:50-81`](../qnn/model/bench/target.py#L50-L81),
   locked in by commit `ab79ae2a`).
2. **`target_probs[:,0]` is NO_TARGET; entity token `j` ↔ `target_probs[:,j+1]`.**
   An off-by-one here silently wrecks all downstream target geometry
   (soft-pooled rel/dist would point one slot off). This is consistent across
   the live labeler, the loss path, and the analysis scripts (verified below).

The labeler design (weapon-aware projectile-lead cone, per-weapon speeds,
noisy-OR cone+physics aggregation, fitted-logistic engagement confidence) is the
real substance of the target head's supervision — the head itself is a thin
scoring layer. **Settled**: KL-not-acc evaluation, off-by-one alignment, the v3
lead-cone labeler design. **Open/unverified**: most bench KL numbers are short
10-epoch / 3-epoch probes that select at epoch 0 (overfit onset); the
canonical-vs-bench parity is a wash within that noise.

## 1. The label — v3 distribution labeler

`label_enemy_target_probs` ([`target_labeler.py:188-408`](../qnn/bc/target_labeler.py#L188-L408))
emits a `(T, 17)` float32 row-stochastic distribution. `p(NO_TARGET) = 1 −
Σ p_indices`; frames with no candidate evidence collapse to `p(NO_TARGET)=1`.
Design docs: [`labeler_v3_simple.md`](labeler_v3_simple.md) (live spec),
[`labeler_weapon_lead_spec.md`](labeler_weapon_lead_spec.md) (the v2 lead-cone
geometry it inherits, marked historical),
[`labeler_v3_proposal.md`](labeler_v3_proposal.md) (original proposal).

Three passes:

1. **Per-fire anchor evidence** — on each attack frame, every enemy pid with
   cone evidence *or* recency-0 physics-hit evidence becomes a candidate; mass
   is distributed proportionally (**never argmaxed**).
2. **Per-pid stream grouping** — anchors merge into streams while the pid stays
   continuously present in the token stream; multiple pid streams may overlap.
3. **Frame extension** — each stream extends backward/forward through continuous
   presence; per-idx score = `eng_conf × time_conf × vis`; normalize to 17
   classes.

### 1.1 Weapon-aware lead cone

The aim vector is lead-corrected by projectile flight time
([`target_labeler.py:249-270`](../qnn/bc/target_labeler.py#L249-L270)):
`aim = rel + vel · t_flight`, `t_flight = dist / speed` (0 for hitscan). Per-weapon
projectile speeds (Quake u/s, from `vendor/quake/QW/progs/weapons.qc`,
[`target_labeler.py:86-96`](../qnn/bc/target_labeler.py#L86-L96)):

| weapon (impulse) | speed | treatment |
|---|---|---|
| Axe (1) | ∞ | melee, range-gated 64u |
| Shotgun (2) / Super Shotgun (3) | ∞ | hitscan, pellet spread |
| Nailgun (4) / Super Nailgun (5) | 1000 | projectile lead |
| Grenade Launcher (6) | 600 | straight-line approx (horizontal 600 after v_up term) |
| Rocket Launcher (7) | 1000 | projectile lead, no self-vel inheritance |
| Lightning Gun (8) | ∞ | hitscan, range-gated 600u |

The cone is **sticky-robust**: `cos = max(cos_lead, cos_current)` so a target
that hasn't started moving still admits ([`:262-265`](../qnn/bc/target_labeler.py#L262-L265)).
The adaptive half-width is `theta_acq = clamp(atan(208/dist), 5°, 30°)` and the
cone is a Gaussian `exp(−0.5·(theta/theta_acq)²)` ([`:267-270`](../qnn/bc/target_labeler.py#L267-L270)).
A hard reject fires only at `theta > 45°` with no physics hit
([`:295`](../qnn/bc/target_labeler.py#L295)). The **release cone** (416u / 45° cap)
is documented in the lead spec as a hysteresis/Schmitt-trigger acquire-vs-release
pair, but the **live v3 path uses only the acquire cone** — the wider release cone
is analysis/hybrid-only ([`:38-42`](../qnn/bc/target_labeler.py#L38-L42)). The
lead spec's recommended `V_perp_max = 500 u/s` transverse cap is a labeler-design
constant; it is **not** applied as an explicit clamp in the current
`label_enemy_target_probs` (the lead term uses raw obs velocity). *(Unverified
whether a V_perp clamp was ever wired; the live code does not clamp.)*

### 1.2 Evidence aggregation & confidence

- **Noisy-OR** cone+physics: `base = 1 − (1−cone_e)(1−phys_e)`
  ([`:301`](../qnn/bc/target_labeler.py#L301)) replaces `max()`; ≈+3pp accuracy
  on the agreement subset (~5.6% of fires) per
  [`labeler_v3_simple.md`](labeler_v3_simple.md#L283).
- **Engagement confidence** is a **fitted logistic** on per-stream features
  `(mean_anchor, fire_count_conf, max_anchor, log1p(duration), log1p(n_fires))`
  ([`:372-380`](../qnn/bc/target_labeler.py#L372-L380); coefficients
  [`:144-149`](../qnn/bc/target_labeler.py#L144-L149)), fitted on QWD val shards
  0–5 (7,345 physics-confirmed streams) vs v2-and-physics consensus. Reported
  Brier 0.061 vs 0.080 for the prior clip-linear (~24% reduction).
- **Death penalty** ×0.65 if the demonstrator dies inside/just after the window
  ([`:366-368`](../qnn/bc/target_labeler.py#L366-L368)); `present_cap` 0.98 caps
  fake certainty. These are `LabelerConfig` knobs
  ([`:119-161`](../qnn/bc/target_labeler.py#L119-L161)) frozen per collection.

### 1.3 Label-quality verdict (settled)

From [`labeler_v3_simple.md`](labeler_v3_simple.md) evidence readback (label
counts / switches / quality ratio on QWD): hard opt3 = 844,723 labels / 3,109
switches / quality 0.285; v2 lead/range = 849,185 / 3,214 / 0.282; physics-only
loses too much cone-only engagement (733k / 0.169); **hard physics tie-breaks add
flipping noise** (hybrid 0.238). Conclusion: keep lead correction, distribute
mass softly, let soft-CE consume ambiguity — do **not** argmax. The offline
label-quality probes (`gbt_target_v2_vs_v3.py`, `mlp_target_v3.py`) were built to
grade v3 soft labels against v2 hard labels on held-out shards; **their numeric
results are not committed as sidecars** (only `runs/probe/_gbt_*.log` files from
an earlier pid-probe era exist) — *unverified numbers, do not cite*.

## 2. The head — soft pointer over entity tokens

The canonical production head ([`target.py:TargetPointer`](../qnn/model/target.py))
is a **per-entity MLP scoring head**: `logits_i = w2·gelu(W1·entity_out_i + b1) + b2`,
masked to enemy ∧ valid tokens, softmaxed to a blend `target_feat`. `d_target`
(MLP hidden width) is the sole knob. It serves two consumers: `target_logits`
(supervised by the labeler) and `target_feat` (the pointer-blended entity vector
handed to motor/weapon/attack heads). **Target is not a sampled action** — the
engine never consumes a target label; `target_logits` only gets supervised
gradient, `target_feat` propagates downstream. This MLP form was **promoted to
canonical** (commit `fd9703a9`); the older attention-style pointer + its five
flags moved to bench for ablation only.

The bench head [`target.py`](../qnn/model/bench/target.py) is the standalone
mirror: `target_soft_ce_loss` (present-weighted soft-CE,
[`:33-47`](../qnn/model/bench/target.py#L33-L47)) and `target_metrics`
(`loss_target`, `target_kl`, `target_present_mean`,
[`:50-81`](../qnn/model/bench/target.py#L50-L81)). Selection metric is
`target_kl`, lower-is-better ([`:145-152`](../qnn/model/bench/target.py#L145-L152)).

### 2.1 Bench pointer variants (registry [`heads.py`](../qnn/model/bench/heads.py))

All share the same scaffold: `ObsEmbedding(include_spatial=False)` →
`PreAttnEncoder` (no attention) → variant pointer → motor heads Off → canonical
BC soft-CE on `target_probs`. They differ only in the **scoring/query mechanism**:

| head | pointer | query / scoring | enemy mask |
|---|---|---|---|
| `target_constant_query` | [`ConstantQueryTargetPointer`](../qnn/model/bench/inputs/constant_query_target_pointer.py) | single learned `(d_model,)` vector = `Linear(d_model,1)` per entity | yes |
| `target_mlp_query` | [`MLPQueryTargetPointer`](../qnn/model/bench/inputs/mlp_query_target_pointer.py) | per-entity MLP (= canonical form) | yes |
| `target_self_query` | [`CanonicalTargetPointer`](../qnn/model/bench/inputs/canonical_target_pointer.py) | `self_readout · entity_out` dot product | **no** (entity_mask only) |
| `target_self_query_enemy` | `EnemyMaskedTargetPointer` | same, + enemy post-mask | yes |
| `target_weapon_query` | [`WeaponQueryTargetPointer`](../qnn/model/bench/inputs/weapon_query_target_pointer.py) | `weapon_proj(weapon_static) + weapon_embed(impulse)` query · entity_out | yes |
| `target_mlp_query_full_stack` | `MLPQueryTargetPointer` + `TransformerEncoder` + spatial tokens | per-entity MLP over **attended** tokens | yes |

The **oracle** [`GTTargetPointer`](../qnn/model/bench/inputs/gt_target_pointer.py)
soft-pools entity tokens by the labeler's GT distribution (read via
`target_supervision_context`); `target_logits` = log-prob of GT. It is **not a
learnable head** — it is the privileged input that feeds weapon/attack bench
scaffolds as an oracle (so those heads' learnability is isolated from pointer
error). `detach_entity_grad=True` by default (without it, backward through the
weighted pool is ~3× the forward — a real ROCm cost).

## 3. Bench results — KL, not accuracy

All runs: `act.target != 0` segment, input_mask=true, lr 0.006, 10 epochs (the
full-stack `d_target` sweep ran only 3–4 epochs), seed 17, n ≈ 893k val,
`target_present_mean ≈ 0.827`. Read `final_val_target_kl` from
`live_run_report.json → results.bc` (the full-stack sweep reads
`checkpoints/bc_history.json`). Lower KL is better.

| head | best ep | val `target_kl` | val `loss_target` | val `target_kl_multi` |
|---|---|---|---|---|
| `target_mlp_query_full_stack` (d_target=64) | 0 | **0.0427** | 0.157 | — |
| `target_mlp_query` (d_target=64, PreAttn) | 6 | **0.0678** | 0.182 | — |
| `target_constant_query` (null query) | 0 | 0.1288 | 0.243 | — |
| `target_self_query` (cls dot, no enemy mask) | 9 | 0.1231 | 0.238 | — |
| `target_self_query_enemy` (+ enemy mask) | 2 | 0.1240 | 0.238 | — |
| `target_weapon_query` (full 7 scalars) | 5 | 0.1271 | 0.242 | — |
| `target_weapon_query_damage_radius` ([0,6]) | 5 | 0.1270 | 0.242 | — |

Reads:

- **The MLP-per-entity scoring head is the lever**, not the query. `target_mlp_query`
  (0.068) roughly halves the KL of every single-linear-score variant
  (constant/self/weapon all cluster at 0.123–0.129). The constant-query null
  baseline confirms the design intent ([`target_constant_query.py`](../qnn/model/bench/target_constant_query.py)):
  a learned scoring *direction* with no per-frame conditioning lands at the same
  ~0.12–0.13 plateau as the weapon/self queries — so under PreAttn + no GRU, **the
  query mechanism does almost no per-frame work**; per-entity scoring capacity does.
- **Attention helps**: `target_mlp_query_full_stack` (TransformerEncoder + spatial
  tokens, 0.043) beats the PreAttn MLP (0.068) — attended entity tokens carry
  sharper target signal. But it **selects at epoch 0** (val KL rises every
  subsequent epoch — overfit onset), so 0.043 is an early-stop number on a short
  run, not a converged plateau. **Unverified that it survives a longer schedule.**
- **Weapon-query is a wash vs self-query** (0.127 vs 0.123) — handing the pointer
  the held-weapon signature directly does not beat a cls-readout query at this
  scaffold. Within-weapon the saliency is interpretable (next section) but it does
  not move aggregate KL.

### 3.1 full-stack `d_target` sweep (3-epoch, bc_history)

Per-epoch val `target_kl` / `target_kl_multi` (best is always epoch 0):

| d_target | ep0 KL | ep0 KL_multi | trajectory |
|---|---|---|---|
| 16 | 0.0345 | 0.0824 | rises (0.087, 0.071) — overfits |
| 32 | 0.0356 | 0.0816 | rises then dips (0.057, 0.048, 0.086) |
| 128 | 0.0560 | 0.1165 | rises monotonically |
| 256 | 0.0337 | 0.0780 | flat-ish (0.036, 0.036) |

`target_kl_multi` (the discrimination metric, frames with >1 live enemy) runs
~2× the aggregate KL — single-candidate frames are trivial wins, so the
multi-candidate metric is the honest one. d_target ∈ {16, 256} are
indistinguishable at epoch 0 (~0.034 KL / ~0.080 KL_multi); **capacity past
d_target=16 buys nothing here**. All overfit within 1–3 epochs, so these are
early-stop numbers. *(Unverified: no converged long-run KL for any d_target.)*

### 3.2 weapon-query input saliency (settled, interpretable)

Sidecar [`_weapon_query_input_ablation_head_probe_target_weapon_query_seed17.md`](../../runs/head_probe/_weapon_query_input_ablation_head_probe_target_weapon_query_seed17.md)
(baseline val soft-CE 0.2313, 497 rows / 8 val episodes — a tiny eval slice):

- Per-scalar ablation of the 7 static weapon physics inputs: **`damage`
  (+14.8% loss) and `max_dist` (−18.1%, i.e. removing it *helps*)** dominate;
  cooldown/v_vert/gravity are dead weight. The `max_dist` sign says the trained
  query was using range *backwards* — consistent with weapon-query being a wash.
- Path ablation: zeroing the **physics path (`weapon_proj`) costs +84.9%**;
  zeroing the identity path (`weapon_embed`) costs only +2.8%. The physics
  projection carries the query; the per-weapon embedding is nearly decorative.

This is *within-weapon-query* interpretability, not a win over the MLP head.

## 4. The off-by-one alignment (settled, critical)

`target_probs[:,0]` = NO_TARGET; **entity token `j` ↔ `target_probs[:,j+1]`**.
Verified consistent end-to-end:

- **Labeler**: `out[:, NO_TARGET_INDEX]=1` default, indices written to `out[:,1:]`
  ([`target_labeler.py:221-227`](../qnn/bc/target_labeler.py#L221-L227),
  [`:407`](../qnn/bc/target_labeler.py#L407)); `NO_TARGET_INDEX=0`,
  `TARGET_PROBS_CLASSES = MAX_TOKEN_OBJECTS+1 = 17`
  ([`:25-26`](../qnn/bc/target_labeler.py#L25-L26)).
- **Loss**: `present = 1 − target_probs[:,0]`, `idx_dist = target_probs[:,1:]`
  renormalized — both bench ([`target.py:42-46`](../qnn/model/bench/target.py#L42-L46))
  and canonical policy ([`policy.py:~820`](../qnn/model/policy.py)).
- **Segment mask**: the trainer derives `act.target = 1 − target_probs[:,0]` so
  `segment_mask: {"act.target": {"$ne": 0}}` filters engaged frames without
  column-indexing ([`train.py:452-457`](../qnn/bc/train.py#L452-L457)).
- **Token filter**: redistributes masked idx mass back to col 0
  ([`token_filter.py:136-140`](../qnn/bc/token_filter.py#L136-L140)).
- **Geometry validation**: `_target_move_correlation.py` documents and applies
  the `entity j ↔ [:,j+1]` mapping when soft-pooling rel vectors
  ([`scripts/analysis/_target_move_correlation.py:61`](../../scripts/analysis/_target_move_correlation.py#L61)),
  and `argmax(target_probs[:,1:])` returns an **entity index** used to gather
  `entity_rel` directly — i.e. argmax lands on actor tokens by construction.

A 16-dim head output (`output_dim=16`) maps to the 16 entity slots; the
labeler's 17th column (NO_TARGET) is a **loss gate** (`present`), not a head
logit. An off-by-one (treating `target_probs[:,0]` as entity-0) would shift every
soft-pooled rel/dist one slot and silently corrupt all downstream target geometry
— the alignment is load-bearing and verified, not assumed.

## 5. Why `acc_target` is the wrong metric (settled)

Entity slot index = engine emit order ≈ edict number, an **arbitrary** ordering.
The engine↔labeler alignment audit ([`target_labeler_engine_alignment.md`](target_labeler_engine_alignment.md))
quantifies the slot-0 confound:

| baseline | bucket A (target==slot 0) | bucket B (target≠slot 0) | overall |
|---|---|---|---|
| always-slot-0 (val, 683,504 labeled frames) | 100.0% | 0.00% | **97.43%** |
| causal TCN probe argmax | 94.12% | 84.21% | 93.87% |
| probe + τ=0.9 override | 99.80% | 15.73% | 97.63% |

A model that **always predicts slot 0** scores 97.43% accuracy. So per-run
`acc_target` numbers (0.91–0.99 across the bench variants) are dominated by the
majority slot and say almost nothing about target discrimination — the slot-0
share on the `act.target!=0` segment is lower (idx0 baseline ≈ 0.46 here) but the
confound is the same in kind. The TCN ablations also localize the real signal:
**per-slot `rel` vector and the enemy-mask flag** carry the confident overrides
(ablating `rel` → 29.9%, ablating `slot_enemy` → near-zero overrides); recency /
look / fire / self-scalars are noise at the high-confidence operating point.

This is exactly why the bench head **dropped the slot-keyed metric family** and
selects on `target_kl` / `target_kl_multi` ([`target.py:50-81`](../qnn/model/bench/target.py#L50-L81),
commit `ab79ae2a`). The slot-keyed metrics still appear in `live_run_report.json`
(legacy emission) — **ignore them for selection.**

## 6. Metric convention & the oracle's role

- **Selection / comparison**: `val_target_kl` (aggregate) for selection;
  `val_target_kl_multi` (>1 live enemy) for honest discrimination. Both are
  present-weighted, so they only score engaged frames.
- **Eval in-distribution**: all bench target runs match the training segment
  (`act.target != 0`, input_mask=true). Comparing to an OOD segment misleads.
- **Oracle as privileged input**: `GTTargetPointer` is *not* a target-head
  result — it is the labeler's GT distribution dropped into the pointer slot so
  downstream weapon/attack bench heads can be measured free of pointer error
  (the bench skill calls it the "ceiling" scaffold). The weapon-head findings
  ([`weapon-head.md`](weapon-head.md)) use it this way and report that adding
  `target_feat` to the weapon head **hurts** aggregate macro-F1 by ~0.010 (it is
  an LG-viability gate, not a weapon driver) — a downstream-consumer finding, not
  a target-head one.

## Settled vs open

**Settled:**
- KL-not-acc evaluation (`target_kl` / `target_kl_multi`); slot-keyed metrics are
  slot-0-confounded and were dropped.
- Off-by-one alignment (`[:,0]`=NO_TARGET, `j ↔ [:,j+1]`), verified across
  labeler / loss / segment-mask / geometry scripts.
- v3 lead-cone labeler design: per-weapon projectile speeds, sticky-robust
  acquire cone (208u/30°), noisy-OR cone+physics, fitted-logistic engagement
  confidence, death penalty, soft-mass (never argmax).
- The per-entity MLP scoring head is the lever (halves KL vs any single-linear
  query); the query mechanism does little per-frame work under PreAttn+no-GRU.
- Weapon-query is a wash vs self-query on aggregate KL.

**Open / unverified:**
- Every bench KL number selects at epoch 0–9 on a 3–10 epoch run; most variants
  overfit within 1–3 epochs. No converged long-schedule KL exists for any head.
- The full-stack attention win (KL 0.043 vs PreAttn 0.068) is an epoch-0
  early-stop number — unverified it survives a longer schedule or a held-out
  multi-candidate set.
- The offline label-quality probes (`gbt_target_v2_vs_v3.py`, `mlp_target_v3.py`)
  have **no committed result sidecars** — their v2-vs-v3 numbers could not be
  verified.
- `target_kl_multi` is only present for the full-stack `_postfix`/sweep runs; the
  PreAttn variants predate its emission, so multi-candidate KL is unverified for
  constant/self/weapon queries.
- The lead spec's `V_perp_max = 500 u/s` transverse cap is **not** applied as a
  clamp in live `label_enemy_target_probs` (lead uses raw obs velocity) — whether
  it was ever wired is unverified.
- `run.md` notebooks for the target runs are mostly unfilled templates; the
  weapon-query and full-stack notebooks have hypotheses but empty Findings.

## Regenerate

- **Bench KL sweep**: retrain via the bench daemon —
  `head_probe_target_constant_query_seed17`, `head_probe_target_mlp_query_seed17`,
  `head_probe_target_self_query{,_enemy}_seed17`,
  `head_probe_target_weapon_query{,_damage_radius}_seed17`,
  `head_probe_target_mlp_query_full_stack_d{16,32,64,128,256}_seed17`. Read
  `final_val_target_kl` from `live_run_report.json → results.bc` (or
  `checkpoints/bc_history.json → history[*].val_target_kl` for the sweep). Configs
  in each run dir's `config/{train,probe}.json`; see the
  [bench skill](../../agents/skills/bench/SKILL.md).
- **Weapon-query saliency**: the input-ablation grader producing
  `_weapon_query_input_ablation_*.md` (per-scalar + path ablation on
  `WeaponQueryTargetPointer`).
- **Label quality**: `scripts/analysis/gbt_target_v2_vs_v3.py` and
  `mlp_target_v3.py` over `artifacts/collect/qwd/precomputed_val` (`--train-shards
  8 --test-shards 4`) — these are not yet committed-as-sidecar; run and capture.
- **Off-by-one / geometry**: `scripts/analysis/_target_move_correlation.py` (soft
  pool + argmax-to-entity-index check).
- **Engine alignment / slot-0 confound**: the TCN target probe documented in
  [`target_labeler_engine_alignment.md`](target_labeler_engine_alignment.md).
