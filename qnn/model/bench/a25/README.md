# a25

> **Reading `move_seg_skill` mid-run:** skill = dll / H_marg over the onset
> 30-way joint; it crosses zero only when the seg NLL drops below the human
> onset-marginal entropy — **2.87 nats** on the qwd corpus (fb 2.865 / lr
> 2.874). Healthy runs sit NEGATIVE through ~epoch 4 of a 12ep schedule
> (moveud's losses: 4.0/4.85/4.17/3.12/2.8…). Early-negative is arithmetic,
> not damage; worry only if still negative past ~ep8 or the loss spikes >4.5
> after ep4 (the p3/p3b instability signature).


> **TRAINING STANDARD (2026-07-09): 12ep @ lr 0.018 (super-iso-mass, mass 0.216),
> lr_min 0.0003, warmup 1ep.** Baked into `templates/train.json`; per-probe
> overrides still win. Replaces the prior 30ep schedule. Chosen from the Arm B
> P2-shape epoch-reduction sweep (`runs/head_probe/head_probe_afmask_p2iso_*`):
> the LR-compressed rungs (16/12/8ep at iso-mass LR) stay stable with no
> divergence, and the `pk018` rung (12ep @ 0.018) recovers near-baseline head
> quality at 0.4× the epochs. All per-head deltas vs the 30ep P2 baseline are
> within seed noise (≤~1% skill), with target/pointer actually improving.
>
> | Head | P2 30ep@0.006 | pk018 12ep@0.018 | Δ |
> |---|---|---|---|
> | attack_skill | 0.5209 | 0.5152 | −0.006 |
> | look_skill | 0.5824 | 0.5787 | −0.004 |
> | move_skill | 0.5331 | 0.5235 | −0.010 |
> | weapon_skill | 0.7302 | 0.7223 | −0.008 |
> | target_skill | 0.7683 | 0.7765 | **+0.008** |
>
> Watch items for the per-weapon closed-loop A/B (offline aggregates don't
> decide attack changes): attack recall −0.060 (traded for precision +0.014) and
> super_shotgun f1 −0.041; nailgun f1 gained +0.031.
>
> **lr_min swept on-base and CLOSED (2026-07-09):** `lrmin0005`/`lrmin001` runs
> re-tested a hotter tail on the pk018 base (only `lr_min` changed). Result is
> monotonic in both directions — every lift raises the composite selection score
> yet lowers `attack_skill`, with no head benefit:
>
> | lr_min | sel_score | attack_skill | weapon_skill | target_skill |
> |---|---|---|---|---|
> | **0.0003 (standard)** | 1.8838 | **0.5152** | 0.7223 | **0.7765** |
> | 0.0005 | 1.9284 | 0.4883 | 0.7234 | 0.7730 |
> | 0.001 | 1.9528 | 0.4779 | 0.7230 | 0.7571 |
>
> The selection-score gain is an artifact (it rewards the tail behavior that
> guts attack). `lr_min 0.0003` stays. **Cross-check: judge tail changes on
> `attack_skill`/`weapon_skill`, never the composite `best_selection_score`.**

> **STATUS (2026-07-04): the move_hazard WHEN/WHAT-split line below is RETIRED**
> (failed attempt — WHEN/WHAT splits keep losing competitive ablations project-
> wide). The a25 generation's live directions are the **attack-with** 9-way head
> (`attack_with_head.py`, research/weapon-head.md §15) and the **segment-level
> move head** design (research/move-head.md §8), which replaces the entire
> hazard line. `move_hazard_head.py`/`hazard_labels.py` are kept for the
> full_6head reload path and the segment head's label derivation.

The a25-generation project: a learned **WHEN/termination law** for move, on top
of the promoted a24 base. The open thread it closes is the move "hazard gap"
(src/docs/move-head.md): the a24 decode reconstructs the human dwell/switch
rhythm with a *tabulated* semi-Markov hazard (`move_hazard_*` stamps + sticky
gate + switch-back watermark + stop-onset) — a fit-to-corpus model living
outside the network, which is fit-fragile (the none-row zero-tail "statue mode")
and can't condition on combat context (reactive dodge was never learned).

## `move_hazard_head.py` — the sixth head

Per axis it predicts `P(release the held class this tick | axis, held_class,
dwell_age, CLS)`. The move head keeps the **WHAT** (which direction, calibrated);
this head owns only the **WHEN**. On a release the new class is sampled from the
move softmax renormalized over non-held classes. **Jump onset = the ud none-row**
of this surface — no separate jump head.

| property | choice | why |
|---|---|---|
| arch | shared per-axis **MLP**, not GRU | dwell fed explicitly → nothing for recurrence to track; a per-head GRU re-opens the prev-action copy path (move-head.md §3) |
| inputs | `[CLS, one-hot(held_class)=3, one-hot(axis)=3, log1p(dwell_age)=1]` | dwell is the variable the human hazard is a function of; class selects the class-conditioned law; axis shares stats across fb/lr/ud while keeping them independent (§4) |
| context | via `CLS` only | the hazard gradient shapes the trunk to encode dodge context the static table can't |
| loss | calibrated BCE on `y = 1[class switches next tick]`, **no reweighting** | goal is a calibrated release prob; reweighting → over-switching. Judge on AUC/calibration, not F1 |
| labels | `held_class` / `dwell_age` / release target derived from the existing `act_move` stream **on the fly in the loader** (`hazard_labels.py`), **no recollect, no cache growth** | all three are deterministic functions of the move-action sequence; this head is temporal-only so the time axis is always present |

## Status / build phases

1. **head module + loss** — `move_hazard_head.py` ✅
2. **graph + pipeline wiring** — `move_hazard` type in
   `qnn.model.graph.build.HEAD_TYPES`, `HeadNodeSpec`, the `Network` slot, a
   `graph/bases/full_6head.json` base, and the calibrated-BCE dispatch in
   `QNNPolicy._compute_head_losses_and_metrics`. ✅
3. **labels** — `hazard_labels.py` derives the four columns episode-aware from
   `act_move`. The dispatch consumes them; the loss + derivation are unit-tested.
   ✅
4. **loader hook (remaining)** — derive the four columns on the fly in
   `qnn.bc.streaming_source.read_rows` (scanning from `EpisodeRef.row_start` for
   exact sub-window dwell) and add them to the requested keys when the model has
   a `move_hazard` head — mirroring the attack-bit derivation already in
   `read_rows`. Cache-free. Needs a trainer-container run on the corpus to
   validate. *(pending)*

Honest scope: this fixes WHEN/how-often (rate, dwell, strafe rhythm) and jump
*timing/rate*. It does not by itself fix jump *placement* (geometrically-correct
jump targeting) — that stays a parked training ceiling (move-head.md).
