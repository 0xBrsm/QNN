# Project review — June 2026

Four independent review reports (architecture, methodology, objective
alignment, git trajectory) produced by parallel reviewer agents on 2026-06-10,
plus a synthesis. Snapshot taken on `feat/head-probes` at ~14 weeks of project
age, mid-way through the `full_5head` target-pointer experiment
(`runs/head_probe/head_probe_full_5head_seed17`). Corrections applied after
review are listed in [Corrections](#corrections-to-the-reports).

## Synthesis

**Verdict: converging, with one structural inversion to fix.** The project's
two strongest assets are process assets: distrust of its own metrics
(settled-vs-open bookkeeping, metric-confound hunting) and cheap comparable
experiments (the bench harness). The failure modes discovered the hard way —
slot-0 accuracy confound, copy-prev persistence leaks, argmax collapse,
offline/live divergence — are the canonical pathologies of behavior cloning
(copycat BC, causal confusion, covariate shift); they were found empirically
first and connected to literature after.

**The structural critique:** every major failure was discovered live, never
offline, yet the effort ratio is hundreds of offline runs to one committed
live eval. Offline metrics have measurably asymptoted as an information source
(move is Markov-1-capped, weapon at the linear-probe ceiling, attack floored
by label timing noise). The highest-leverage process change is inverting that
ratio: make the live battery (blind-fire rate, tracking-cos, dwell/switch
EMDs, time-to-frag) a routine gate after every model/decode change.

**Endgame ordering (opinionated):**

1. Closed-loop correction — DAgger-style supervision with the oracle
   lead-corrected aim from the existing labeler, model driving its own
   trajectory (`env/sim.py` is the seed). This is the durable fix for aim and
   projectile leading; `target_feat` provides the substrate but on-manifold BC
   cannot teach lead because humans bake it into the labels.
2. Longer-horizon behavior — item timing, armor control, engagement pacing.
   Unmeasured today; in rocket arena, armor timing is half of "plays like a
   human."
3. PPO only when instrumented — its win-rate reward with a weak KL anchor
   would degrade distributional fidelity invisibly; wire `obs_blind_fire_rate`
   / tracking-cos / switch-rate EMD into PPO eval before any retained run.

**Pre-deploy gate for full_5head:** measure target-head closed-loop switch
dynamics (chained switch rate vs human ~0.88%) from cached softmaxes before
trusting the live eval — an over-switching pointer would feed unstable
`target_feat` to look/attack and could be worse live with flat offline
metrics.

**Housekeeping:** checkpoint-merge stable infrastructure to `main`
periodically; split `policy.py` (checkpoint I/O / obs adapter / loop); add
tests for `tools/export_onnx.py`; bench registry needs a deprecation
mechanism.

## Corrections to the reports

- **Branch divergence (trajectory report).** `git rev-list` does report 530
  commits ahead of `main`, but the branch has been split/merged to `main`
  several times under a squash-merge flow, so raw ahead-counts overstate
  divergence; `main`'s tip (PR #17) is the merge-base, not an orphaned
  ancestor. The softer point stands: no checkpoint merge since 2026-05-18, and
  release tags 0.19.0–0.21.0 sit on the feature branch.
- **Weapon switch-process head (methodology report rec #3).** Stale. The
  WHEN/WHAT switch-process head was built, validated, and deliberately
  abandoned (2026-06-07) after the dense CLS+GRU head won per-frame and
  distributionally ([weapon-head.md](weapon-head.md) §5–6). Not a live
  recommendation.
- **Turing-test priority (objective report rec #1).** Right in principle,
  premature in practice: failures are still identifiable by the developer in
  an hour of play. Blind human judgment becomes the right instrument once
  obvious defects (aim, lead) stop being articulable.
- **MVD scale path.** Not covered by any report; see
  [Addendum](#addendum--mvd-scale-path-bitter-lesson).

## Report: architecture & engineering health

### Model / training stack

- `ModelConfig` is a frozen slots dataclass with no hidden defaults;
  construction-time validation prevents config/code drift.
- The `Network` slot-override pattern (`None | nn.Module | Off` per slot) with
  `slot_dims()` as the single dim authority is an unusually disciplined
  contract design for a solo project.
- The bench/canonical split is the right experimental infrastructure: probes
  share the full BC pipeline (loader, loop, checkpointing, history) via a
  `model_factory` hook, so probe results are directly comparable to production
  runs. Adding a probe is one file + one registry entry; 37 bench modules show
  active use.
- Run config management (`qnn.run.config`) uses strict require-key helpers, no
  silent defaults.

### Engine/C seam

- The three-axis contract system (wire / semantics / arch) is the standout
  engineering achievement: checkpoint as source of truth, refuse-don't-guess
  backfill, import-time parity assert between the MODERN registry row and live
  `engine_norm` ids, and `test_engine_norm_parity.py` parsing C headers to pin
  constant parity.
- `qnn_obs_codec_t` cleanly separates the stable tick/action contract from
  per-wire-contract packing; adding wire.10 is a codec struct + array entry.
- Fragility: `pack_scratch` in `qnn_onnx.c` clears entity slots with
  per-field `memset`s — a new field added without its memset silently bleeds
  previous-tick data. No compile- or test-time guard.

### Quality signals

- Tests: 29 files / ~6k lines covering contracts, C↔Python parity, dequant,
  wire parity, BC loop. Gaps: no tests for the bench system, PPO, `diag/`, or
  `tools/export_onnx.py` (651 lines, the most externally visible artifact).
- Zero TODO/FIXME markers; debt is tracked in `agents/plans/` instead. Legacy
  compat aliases span at least three config eras.
- Documentation density is exceptional (25 docs in `src/docs/`, full agents/
  conventions stack).

### Strengths / weaknesses / recommendations

Strengths: the contract system; the bench split; config discipline.
Weaknesses: `policy.py` god module (2,005 lines, 33 functions mixing
checkpoint I/O, migration, obs conversion, training loop); structural test
gaps (export, bench, PPO, diag); bench sprawl with no lifecycle management
(pycache entries for deleted modules, unbounded `HEADS` registry).
Recommendations: split `policy.py` into checkpoint I/O / obs adapter / loop;
replace the memset cascade with a region clear or static assert plus a C
test; add `deprecated`/`superseded_by` to `HeadSpec` and prune.

## Report: research methodology & experimental discipline

### What is genuinely good

- Head-probe isolation with a frozen data contract makes results comparable
  across ablations; configs committed per run.
- Settled-vs-open epistemics in every findings doc, with explicit flags on
  early-stop numbers, single-seed results, and metric-family
  incompatibilities.
- The metric regime overhaul ([head-metrics.md](head-metrics.md)): five heads
  consolidated onto `<head>_skill = 1 − NLL/H_marg`, argmax diagnostics
  demoted from selection — foundational self-correction.
- Triangulated ablations: cooldown dominance confirmed via three independent
  routes (geometry-only F1 0.0; oracle ceiling 0.34; +96% loss on cooldown
  zeroing, twice).

### Failure modes found and fixed

- **Persistence/copycat trap** in all three flavors: argmax collapse (jump
  P=0.000079 vs human 0.0135 — sample, don't argmax), the Markov-1 momentum
  cap (Markov-1 dll 0.407 beats trained GRU 0.378), and the `prev_look` leak
  (+0.09 nats traced to persistence echo, not aiming). Matches de Haan 2019 /
  Wen 2020; guards implemented.
- **Offline/live divergence** ([fire-discrimination.md](fire-discrimination.md)):
  model void-fire ratio offline (0.056) is *better* than human (0.087), yet
  live the bot fired off-crosshair 60% of fire ticks. Mechanism resolved to
  decode-induced crosshair instability (per-frame direction sampling tripling
  heading reversals), fixed by hybrid decode with a falsifiable live
  prediction. Complaint → refuted hypotheses → mechanism → fix → prediction is
  the right investigation shape.
- **Metric confounds caught:** slot-0 target accuracy (always-slot-0 scores
  97.43%), masked-vs-unmasked fire metric families, the `move_kl_joint` 3.26
  einsum ordering bug.

### Gaps

- One committed live eval vs hundreds of offline runs (the central asymmetry).
- Single-seed decisions on consequential comparisons; target-head leaderboard
  is entirely early-stop snapshots (no converged long-schedule KL exists).
- Most `run.md` hypotheses unfilled (post-hoc narrative bias risk); the
  weapon_aim notebooks are the model to propagate.
- Target-head closed-loop switch dynamics never measured — the upstream gate
  for the whole engagement subsystem.

### Effort allocation

attack/fire ~144 runs, weapon ~73, look ~50, target ~12, move ~9. The attack
investment concluded "the head was fine, cooldown dominates" — an expensive
detour that closed-loop diagnosis first would have shortened. Live evaluation
is the most under-invested area by any measure.

## Report: objective & evaluation alignment

### Operationalization of "human-like"

The metric arc (per-frame F1/acc → proper scoring rules → distributional
fidelity → live closed-loop) is genuine intellectual progress and lands in
the right place. Open gaps by the project's own admission: target switch
dynamics unmeasured; attack burst-initiation recall continuation-inflated
(genuine engage recall ~51%); projectile leading entirely unresolved — the
labeler lead-corrects attribution, but the look head has no mechanism to
produce lead in closed loop (humans bake it into the labels on-manifold).

### Eval stack

Offline bench mature; live eval (`qnn.eval.run`) emits the right
discrimination metrics (`obs_blind_fire_rate`, `engine_fire_tracking_cos_*`,
`engine_fire_by_los_angle`); human reference baselines exist for move/look.
Missing: a composite human-likeness verdict (Tier-2 metrics exist per head
but are never combined); human blind judgment (HNTT/BotPrize cited, never
run); longer-horizon behavior (item timing, map control, respawn positioning
— absent entirely); PPO eval emits only return/frag_delta, blind to
human-likeness.

### BC asymptote

Corpus 34.7M train rows / 482 h. Ceiling evidence is concrete (move
Markov-1-capped; weapon at linear-probe ceiling; attack floored by label
timing noise; look in-distribution resolved). More same-source data will not
move per-frame ceilings; the deployment target (Pi CPU, 20 Hz) caps model
scale. Next levers assessed: diverse data (real but bounded), bigger model
(deployment-blocked), DAgger-style closed-loop correction (highest-value;
`env/sim.py` exists for look), human-likeness-constrained RL (unexplored;
current PPO reward is structurally misaligned), hierarchical policies
(cautioned by the tactics-head failure; a lightweight commitment layer is
more defensible).

### Verdict and biggest risk

Conditional convergence: near-term decode fixes address the dominant live
complaint, but "human observers can't tell" additionally requires the live
baseline measurement, a closed-loop lead solution, and longer-horizon
metrics. Biggest risk: resuming PPO without human-likeness instrumentation —
a stronger win-rate bot that is less human-like, with no measurement to
detect it.

## Report: trajectory & velocity (git)

548 commits over ~14 weeks (2026-03-05 →). Eras: foundation sprint (Mar,
10 squashed releases), BC hardening (Apr), combat-objective BC + TargetPointer
(May 18, PR #17, 76 commits in two days), head-probe/bench era (May 19 →,
~520 commits). Velocity rose from ~2.5 squashed commits/week (Mar) to 20–48
micro-commits/active day (Jun).

- Churn: collect/labeler churn is problem-driven and progressive (v2→v3,
  MDP-correct obs/act alignment); decode churn concentrated in one June week
  and closing; look-metric iteration (4 rewrites in a week) is productive, not
  thrash. PPO is a recurring maintenance thread, not a designed effort —
  accumulating drift debt against the evolving BC arch.
- Convergence signals: VERDICT commits, documented dead ends, retired code
  ("drop dead labeler-v2"), increasingly causal commit messages, fix:bench
  ratio 8:48.
- Hygiene: release flow informally redefined as "deploy-ready ONNX from the
  research branch"; tags 0.19.0–0.21.0 stamped on the feature branch (see
  [Corrections](#corrections-to-the-reports) for the divergence framing).
- Observations: micro-commit style is an excellent lab notebook but obscures
  milestones — consider one RESOLVED wrap-up commit per ablation; PPO
  integration cost grows the longer it is deferred.

## Addendum — MVD scale path ("bitter lesson")

Context: QWD demos carry recorded usercmds (ground-truth labels); MVD demos
carry only server-side state, so labels must be inferred
([input-inference.md](input-inference.md)). The fire back-shift inference is
built and validated on 133 paired demos; switch/move/look inference is
partial. Locking down MVD parity would unlock a corpus in the hundreds of
millions of frames at 20 Hz — orders of magnitude past the current 34.7M
rows.

Assessment (synthesis author's opinion):

1. **The bet is directionally right, but scale fixes coverage, not the
   measured ceilings.** Move/weapon/attack per-frame ceilings are
   objective-structure ceilings (momentum, label noise), not data-starved
   ones. What MVD scale genuinely buys: tail-event coverage (≥15° turns,
   switches, multi-enemy frames — `target_kl_multi` is data-hungry),
   demonstrator diversity (the real "bitter lesson" ingredient; current
   corpus is one player population), and a wider manifold that shrinks the
   OOD gap live play exposes.
2. **A 184k-parameter model saturates long before 100M+ frames.** The
   bitter-lesson play requires scaling the trunk with the data and
   distilling back to the Pi-sized deployment model. Without that, MVD scale
   moves little beyond the tails.
3. **Label fidelity is the gating risk, and it is head-asymmetric.** Inferred
   fire timing is validated to ±1–2 frames; the look head is the danger —
   its supervision is sub-degree turn deltas and its human-likeness metrics
   are sub-degree EMDs, so MVD angle quantization/timing jitter directly
   pollutes the most quantization-sensitive label. Validate look-label parity
   on paired QWD/MVD demos (turn-magnitude EMD between the two label streams
   of the same play) before scaling collection.
4. **Closed-loop correction remains necessary at any corpus size** — no
   amount of demonstration data covers the bot's own trajectory manifold.
   MVD scale and DAgger-style correction are complements, not substitutes.

Practical note: at current bytes/frame, a 100M+-frame corpus exceeds the
~200 GB working budget of the dev container — collection would need to land
on the NAS with streaming or sharded staging. Also,
[input-inference.md](input-inference.md) links `agents/plans/mvd-reconstruction.md`,
which does not exist in `agents/plans/` — the plan doc should be restored or
the link repointed.
