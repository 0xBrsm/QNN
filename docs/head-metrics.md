# Head metrics: the canonical contract

The single source of truth for **how every action head is judged**. The five
heads (move, look, target, attack, weapon) had each independently drifted onto a
different headline metric — `f1`, `acc`, `kl`, `dll` — and the live trainer was
still selecting on the argmax metrics every head's own findings doc rejects. This
doc fixes the unit of measurement *once*; the per-head design notes
(`move-head`, `target-head`, `attack-head`, `weapon-head`,
`persistence-and-changepoints`) defer to it.

## TL;DR

Three metric classes, identical shape across all five heads:

1. **`<head>_skill` — selection.** A proper scoring rule normalised to a common
   ruler: the fraction of the head's marginal entropy it eliminates. The trainer
   selects checkpoints on the sum of these and **nothing else**.
2. **Distributional fidelity — the human-likeness verdict.** Dwell-time and
   switch-rate EMD of the committed/sampled stream vs the human, on the
   in-distribution segment, plus a leak-free honest-subset where the head cannot
   echo persistence. This is the *objective*; skill is the proxy the trainer can
   optimise per epoch.
3. **Argmax diagnostics (`<head>_f1`, `<head>_acc`).** Emitted for analysis,
   **never used for selection.** Under persistence-dominated streams argmax
   point-accuracy is blind to calibration and over/under-switching (see the
   `persistence-and-changepoints` design notes).

## The skill normalisation

For every head define a Δloglik gain over the marginal base-rate predictor and
normalise it by the marginal entropy:

```
<head>_dll   = H_marg − NLL_model           (nats; gain over the base rate)
<head>_skill = <head>_dll / H_marg          = 1 − NLL_model / H_marg
```

`<head>_skill ∈ (−∞, 1)`: **0** = no better than predicting the corpus base rate,
**→1** = the per-frame decision is fully determined by context, **<0** = worse
than the base rate (a real regression). Because every head is on the same
`fraction-of-entropy-explained` ruler, a 0.01 skill gain means the same thing for
every head and the additive composite weights all heads equally — unlike raw
nats/KL/BCE, whose scales differ ~20× and would let the highest-variance head
silently dominate selection.

`move_skill` and `look_skill` already exist on this definition
([supervised_loop.py:495-504](../qnn/bc/supervised_loop.py#L495-L504)); the move
machinery is the template (per-axis `H_marg − ce_model`, mean over axes, divided
by mean `H_marg`). `target_skill`, `attack_skill`, `weapon_skill` are defined
identically against their marginal label entropies.

### The composite

```
selection_score = Σ_head (1 − <head>_skill)        # lower is better
```

Each term is the fraction of that head's signal **not** captured. Heads absent
from a run contribute a neutral 0 (skill = 1) so subset runs still produce a
monotone signal. ([train.py:_selection_score](../qnn/bc/train.py))

## Reporting surfaces

| surface | what it shows | why |
|---|---|---|
| **Epoch report** (printed line) | `sel` (composite) + `overfit` + `reorg` + the five `<head>_skill` values | normalised, comparable, easy to track where the run is improving and whether it is generalising / still moving |
| **`bc_history.json`** | every raw metric (`*_dll`, `*_kl`, `*_loss`, `*_f1`, per-class, grad norms, weight drift, train-proxy / train-eval gaps, …) | full fidelity for careful post-hoc analysis |

The epoch line is intentionally **summary-only**; the raw core numbers live in
`bc_history.json`. Two derived scalars replace the old
`train_proxy`/`proxy_gap`/`train_eval`/`gap`/`grad`/`drift` clutter:

- **`overfit`** = val selection error − the train reference (held-out
  `train_eval` when it ran, else the noisier train proxy). `>0` = val worse
  than train (memorising); `~0` = generalising; `<0` = val ahead.
- **`reorg`** = this epoch's weight drift ÷ the running peak drift. `~1` =
  reorganising as hard as ever; `→0` = converged / stuck.

([train.py:_headline_keys](../qnn/bc/train.py))

## Naming convention

Every metric field is **head-first**: `<head>_<stat>`. So `move_skill`,
`look_dll`, `target_kl`, `attack_f1`, `weapon_loss` — never `f1_attack`,
`loss_weapon`, `acc_target`. New code emits head-first; there is **no
back-compat** for the old order (old run JSONs keep their legacy keys; ignore
them).

## Per-head contract

| head | Tier-1 selection | Tier-2 verdict (human-likeness) | leak-free honest subset | argmax diagnostic | purged (leak / confound) |
|---|---|---|---|---|---|
| **move** | `move_skill` | per-axis dwell/switch-rate EMD (fb/lr sticky, ud sampled) | — | `f1_move`, `acc_move` | — |
| **look** | `look_skill` | turn-magnitude EMD (turn ≥15° buckets), bout duration | — | `look_r2`, `look_ewa_deg` (regression-head diagnostics) | `cos_sim_look` (already not emitted — saturates ≈0.99) |
| **target** | `target_skill` (+ `target_kl`, `target_kl_multi` as analysis) | output switch-rate vs human 0.88% — **OPEN, unmeasured** | `target_kl_multi` (>1 live enemy) | — | `acc_target*` / slot-keyed F1 (slot-0 confound; already dropped) |
| **attack** | `attack_skill` | burst-initiation recall + event-tolerance (±2–3 frames) | burst-initiation vs continuation recall | `f1_attack` (exact-frame) | unmasked `*_f1_fire` / `*_fire` aliases (already not emitted) |
| **weapon** | `weapon_skill` | dwell EMD + switch-rate + weapon-marginal fidelity | switch acc (pick ≠ `self_weapon_id`), switch@attack | `f1_weapon`, `f1_weapon_{weighted,micro}_global`, `acc_weapon` | `f1_weapon_global` / `f1_weapon_macro_global` (pure duplicates of `f1_weapon`) |

Notes:

- **`target_kl` / `target_kl_multi` stay** as raw analysis metrics in
  `bc_history.json` (KL to the soft label / discrimination on multi-candidate
  frames). `target_skill` is the *selection* term (gain over the marginal target
  base rate) so target sits on the same ruler as the other heads — distinct
  reference from `target_kl`, both useful.
- **attack** is binary, so `attack_dll` is the BCE gain over the ~2.8% fire base
  rate; `H_marg` is the binary entropy of that base rate.
- **weapon** `H_marg` is the entropy of the 8-class weapon marginal (~1.35 nats);
  per-class F1 stays in history for rare-gun analysis.

## Open Tier-2 gaps (not yet measured)

These are the *only* real holes once the metrics are consistent — flagged here so
they are not mistaken for solved:

1. **Target output switch-dynamics.** The memoryless target head's chained
   output switch-rate vs the human 0.88% has never been measured; it may
   over-switch at ambiguous boundaries. (`target-head`,
   `persistence-and-changepoints` §Per-Head Status)
2. **Attack leak-free switch-on/off audit.** No attack-frame leak-free audit
   exists analogous to the weapon switch-vs-hold metric; the ~0.59 `attack_f1` is
   continuation-inflated and the genuine burst-initiation skill (~51%) is the
   consequential number. (`attack-head` §8)

## What changed (migration)

- `_selection_score` no longer keys on `acc_target` (which was dropped from
  emission — the term was a silent dead 0), `f1_attack`, or `f1_weapon_global`.
  All five heads now contribute `(1 − <head>_skill)`; `target` is no longer a
  dead term.
- New `target_skill`/`target_dll`, `attack_skill`/`attack_dll`,
  `weapon_skill`/`weapon_dll` emitted from the canonical metric path
  (`QNNPolicy._compute_head_losses_and_metrics` + `supervised_loop`), mirroring
  the existing `move_skill`/`look_skill`. attack uses a CLEAN unweighted BCE
  (not the focal/pos-weighted training loss).
- Bench `selection_metric`s align to the per-head Tier-1 column (`attack_skill`,
  `weapon_skill`, `target_skill`; target's `lower_is_better` flag flipped);
  `weapon_aim` moves off `look_r2` to `look_skill`. `f1`/`acc`/`kl` stay emitted
  as diagnostics.
- The epoch line is now `sel`/`overfit`/`reorg` + the five skills.
- Removed the pure-duplicate `f1_weapon_global` / `f1_weapon_macro_global`
  (identical to `f1_weapon`). `cos_sim_look` and the `*_fire` aliases were
  already not emitted by the current loop (they survive only in old run files —
  ignore them there). `look_r2`/`look_ewa_deg` remain as regression-head
  diagnostics, just out of selection and the headline.
