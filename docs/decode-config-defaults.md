# Decode-Config Defaults — Key Registry & Fail-Loud Audit

Source-of-truth inventory of every decode-config `params.*` key recognized at
HEAD: its policy attribute / export kwarg, its code-side default (or `NONE —
REQUIRED` when there is none), and — for the keys removed in the 2026-08-26
decode-surface purge — why. The single source of truth is
`src/qnn/model/decode_config.py`: `DECODE_PARAMS` (the registry), `_NON_REGISTRY_KEYS`
(engine/guard-applied keys with no registry row), `REQUIRED_PARAM_KEYS` (the
shared BASE required set) and `MODULE_REQUIRED_PARAM_KEYS` (per-`decode_module`
extras). This file is a reader's index onto that code; when they disagree, the
code wins.

Motivation: the run-config philosophy — *"No code-level defaults fill in missing
training keys"* (`src/docs/run.md`) — was **not** originally enforced for the
decode config. A key applied only when present silently fell back to a
`policy.__init__` / kwarg default when omitted. That is exactly how the a25
decode line lost the a24 hazard-discounted lead caps (`look.lead_hold_cap_frames`
= 4 / `_radial` = 5) and silently ran a linear over-lead, and how the a25
`look.turn_mag_scale` dampener regressed to a cross-arch value.

## Mechanism

- **Per-decode-module manifest.** The required set = a shared **BASE**
  (`REQUIRED_PARAM_KEYS`, the `look.*` aim-geometry family every look-decode arch
  carries) **UNION** the extras registered for the config's own `decode_module`
  (`MODULE_REQUIRED_PARAM_KEYS`). This keeps arch-owned keys required **only**
  for that arch's configs. `_required_param_keys(decode_module)` returns BASE +
  module extras (`decode_config.py:145`).
- Validated once, inside `resolve_decode_config`, which **both** consumers
  (`qnn.eval.run` and `tools/export_onnx`) call — so they cannot drift.
- A config missing any required key raises `ValueError` naming the **config
  path**, the **decode_module**, and the **missing key(s)** (`_validate_required_params`,
  `decode_config.py:150`). No default is substituted. OFF, where wanted, must be
  an explicit `0.0`/`false`/`"off"` — never omission.

**No silently-ignored keys, of any kind (2026-08-26).** The required-key
manifest above only ever covered *missing* keys; a key that was *present* but
named a dead law used to load anyway and do nothing (`attack.crest_theta_vec`
reaching the exporter with no eval-side consumer was exactly this bug — see the
registry section below). `_validate_known_params` (`decode_config.py:296`) now
closes that hole from the other side: every key in a config's `params` must be
either a `DECODE_PARAMS` row or a `_NON_REGISTRY_KEYS` entry, or the config is
**refused at load**, naming the offending key(s). There is no third,
silently-tolerated category and no separate "rejected keys" list to maintain —
a key with no implementation is indistinguishable from a typo, and both fail
the same way. Brian: *"They either work or they fail. No exceptions."*

## The param registry

The *mapping* — config key → policy attribute / ExportWrapper kwarg → coercion
→ default — used to be hand-written at three sites and **drifted**:
`attack.crest_theta_vec` / `attack.crest_hold_ticks` reached the exporter but
were never applied in offline eval, so a crest-gated config baked the discharge
gate into the ONNX while `qnn.eval` ran without it. No pinned config had ever
set them, so nothing shipped wrong — but the divergence was live in the code.
(The crest gate itself was deleted outright on 2026-08-26 — see REMOVED below —
but the registry pattern this incident motivated stays load-bearing for every
key that replaced it.)

There is now ONE table, `DECODE_PARAMS` in `src/qnn/model/decode_config.py`
(`decode_config.py:394-484`):

| field | meaning |
|---|---|
| `key` | the decode-config `params.*` key |
| `name` | the `QNNPolicy` attribute **and** the `ExportWrapper` kwarg (identical by contract) |
| `coerce` | the raw-JSON → normalized-value function (shared by both consumers) |
| `default` | the value used when the config omits the key; `NO_DEFAULT` = fail loud |
| `export_name` | the ExportWrapper kwarg when it differs (only the tremor pair) |
| `graph` | `False` = eval-only law with no in-graph twin. Nothing currently sets this — the eval-only DOWN-band degraders that used it (`look.aim_degrade_sluggish_tau` / `_lag_frames` / `_jitter_mag`) were deleted 2026-08-26 (see REMOVED) |

Consumers read it through two accessors and nothing else:
`ResolvedDecode.policy_attrs()` (eval — one `setattr` loop, raises if the
attribute does not exist on the policy) and `ResolvedDecode.export_kwargs()`
(export — splatted straight into `ExportWrapper`, `graph=False` rows excluded).
`tests/test_decode_param_registry.py` fails if the registry, `QNNPolicy` and
`ExportWrapper` fall out of step, including on the defaults.

**Not in `DECODE_PARAMS`, by design** (`_NON_REGISTRY_KEYS`, `decode_config.py:225`):

- `weapon_ban`, `look_grid`, `move_hazard` — schema fields / file refs, handled
  explicitly by the callers.
- `guard.projectile_release`, `guard.lg_range` — bound directly by the guard
  module's own `make_guard(params)`, a separate default surface (see `guard.*`
  below).
- `attack.hold_tail_sec` — the ENGINE is its consumer (`src/engine/common/qnn_onnx.c`),
  not python; see the dedicated section below.

## Legend

- **Risk** — `SILENT` = a code-side default is substituted when the key is
  absent, i.e. omission changes behavior with no error; `INERT` = the key is a
  true no-op (paired with another key that's off, or has no live consumer at
  the current operating point); `REQUIRED` = omission raises `ValueError` at
  resolve time, no default exists.
- **`decode.base.json`** — the live `a25base` template (`REGIME_CONFIGS["a25base"]`,
  the ORACLE_TEMPLATE every fresh decode-fit config is built from) — ✓ set, ✗
  omitted (takes the registry default, or fails to resolve if required).

## `look.*` — aim geometry

| key | attribute (init default) | registry default | `decode.base.json` | risk |
|---|---|---|---|---|
| `look.aim_prior_gain` | `look_aim_prior_gain=None` (`policy.py:298`) | **NO_DEFAULT** | ✓ (`0.0`) | REQUIRED |
| `look.aim_ffwd_gain` | `look_aim_ffwd=None` (`policy.py:374`) | **NO_DEFAULT** | ✓ (`0.0`) | REQUIRED |
| `look.aim_mag_gain` | `look_aim_mag_gain=0.0` (`policy.py:380`) | **NO_DEFAULT** | ✓ (`0.0`) | REQUIRED |
| `look.turn_mag_scale` | `look_turn_mag_scale=1.0` (`policy.py:393`) | **NO_DEFAULT** | ✓ (`1.0`) | REQUIRED |
| `look.lead_hold_cap_frames` | `look_lead_hold_cap_frames=None` (`policy.py:407`) | **NO_DEFAULT** | ✓ (`4.0`) | REQUIRED |
| `look.lead_hold_cap_radial_frames` | `look_lead_hold_cap_radial_frames=None` (`policy.py:410`) | **NO_DEFAULT** | ✓ (`5.0`) | REQUIRED |
| `look.hold_passthrough` | `look_hold_passthrough=False` (`policy.py:404`) | **NO_DEFAULT** | ✓ (`true`) | REQUIRED — module extra for `qnn.model.decode_actions` |
| `look.aim_degrade_tremor_mag` | `look_aim_degrade_tremor_mag=0.0` (`policy.py:423`) | `0.0` (export kwarg `look_tremor_mag`) | ✓ (`0.0`) | SILENT while unset; the live DOWN-band demoter when armed |
| `look.aim_degrade_tremor_tau` | `look_aim_degrade_tremor_tau=5.0` (`policy.py:424`) | `5.0` (export kwarg `look_tremor_tau`) | ✓ (`5.0`) | INERT while tremor_mag is 0 |

**REMOVED 2026-08-26** (`look.weapon_pitch_gain/_bias/_mode/_shift_strength/_lock`,
`look.aim_degrade_sluggish_tau/_lag_frames/_jitter_mag`, `look.hold_drift_eps`) —
see REMOVED below.

## `move.*` — commitment decode

| key | attribute (init default) | registry default | `decode.base.json` | risk |
|---|---|---|---|---|
| `move.commit_dur_tilt` | `move_commit_dur_tilt=(0.0,0.0)` (`policy.py:338`) | `(0.0, 0.0)` | ✓ (`[0,0]`) | SILENT — fitted per checkpoint (a27rc1a `[0.0647, 0.1105]`, a28 `[0.1037, 0.1409]`, `[0,0]` on base) |
| `move.threat_break_hazard` | `move_threat_break_hazard=0.0` (`policy.py:352`) | `0.0` | ✓ (`0.0`) | SILENT — the dodge-reactivity assist, fitted per checkpoint (0.0409–0.0474 on live rc lines) |
| `move.idle_none_bias` | `move_idle_none_bias=(0.0,0.0)` (`policy.py:344`) | **NO_DEFAULT** | ✓ (`[0,0]`) | REQUIRED — module extra. Engagement-gated per-axis idle stillness; `(0,0)` = off |
| `move.idle_engagement_base` | `move_idle_engagement_base=0.5` (`policy.py:345`) | **NO_DEFAULT** | ✓ (`0.5`) | REQUIRED — module extra. E when an enemy is merely present |
| `move.idle_cooldown_ticks` | `move_idle_cooldown_ticks=20` (`policy.py:346`) | **NO_DEFAULT** | ✓ (`20`) | REQUIRED — module extra. Ticks E holds at 1 after combat |

All three `move.idle_*` keys were promoted from SILENT to REQUIRED on
2026-08-26 (`71617b945`): only the bias was ever config-stated; the two
triggers ran on code defaults no config named. 1492 stored configs were
backfilled at the values they were already running — behavior unchanged, but
now readable off the config.

**`move.commitment` is no longer a config key at all** (`42e173a5c`,
2026-08-26): `QNNPolicy.move_commitment` is a **read-only property** derived
from the checkpoint (`model._has_move_seg_head`), mirroring `look_commitment`,
which was always derived this way. The config key could never actually turn it
off — the export already ANDed it with the model's own head structure — so a
config saying otherwise was silently overridden. `move.commit_dur_tilt` is
unaffected (a real, still-config-stated operating point).

**`move.commit_interrupt` / `move.commit_recommit` REMOVED 2026-08-26**
(`da7b9368f`) — see REMOVED below.

### Fitting `move.idle_none_bias` — the a26rc1 ruler error

`move.idle_none_bias` is a per-checkpoint constant fit by
`qnn.decode_fit.move_gates` against the REALIZED occupancy of the decode itself
(`move_gates --statistic rollout`), never a hand-rolled proxy: the original a26rc1
fit scored a move class by summing its ten bucket logits (β entering the
none-score as `+10β`), but the shipped decode softmaxes the 30-way joint, where
the same β shifts `none`'s mass by `exp(β)` — i.e. `+β` on its log-sum-exp. The
fit reached its human target (39.7% out-of-combat stand-still) on a ruler ~10×
more sensitive than the shipped law; measured against the real decode,
`a26rc1b`/`c`/`d`'s `β = 0.31` moved realized stand-still from 19.1% to about
21.6% — roughly an eighth of the correction its provenance claims.

## `jump.*` — confidence gate

| key | attribute (init default) | registry default | `decode.base.json` | risk |
|---|---|---|---|---|
| `jump.threshold` | `jump_threshold=0.0` (`policy.py:371`) | `0.0` | ✗ (omitted — takes `0.0`) | SILENT while `0.0`; a hand-derived per-checkpoint deploy addition (`a26rc1b`: `0.60`), not part of the base template |

`0.0` = AS-IS decode (no deterministic confidence gate on the up/down jump
axis). `τ` is placed by **cut factor against the model's own sampled posterior
rate**, never human parity — `jump: no rate calibration` — reproducing a
historical placement needs the engaged population loaded with
`segment_mask={"act.target": {"$ne": 0}}` (masking before the forward gives the
GRU a spliced sequence and shifts `p_jump` from the same-frames-selected-after
convention the bot experiences in play).

## `attack.*` — attack-with operating point

| key | attribute (init default) | registry default | `decode.base.json` | risk |
|---|---|---|---|---|
| `attack.fire_bias_vec` | `attack_fire_bias_vec=None` (`policy.py:309`) | `None` | ✓ (zero 8-vector) | SILENT while unset. Fire-only calibration; validated as an 8-element numeric vector (`_validate_attack_vectors`, `decode_config.py:165`) |
| `attack.hold_tail_sec` | **none — ENGINE-applied**, `ctx->fire_hold_sec` in `src/engine/common/qnn_onnx.c` | `0.0` (`ATTACK_HOLD_TAIL_DEFAULT`), **always stamped** by the exporter | ✓ (`0.0`) | Continuous-weapon (NG/SNG/LG) hold-tail seconds; `0` = off. `_NON_REGISTRY_KEYS`, not `DECODE_PARAMS` — see the dedicated section below |

`weapon.preference_bias_vec` (selection) lives in this same operating point but
is keyed under `weapon.*` — see below, split-v1 keeps the two vectors
deliberately separate (fire-only vs. selection-only).

**REMOVED 2026-08-26**: `attack.bias`, `attack.bias_vec`, `attack.vector_semantics`
(the legacy JOINT attack law, `5b64a90bb`); `attack.crest_theta_vec`,
`attack.crest_hold_ticks` (the a25 discharge-quality/"crest" gate, `a29ba2697`);
`attack.fire_gather` (the choice-gathered fire test, `7fa8a23ee`/`6ae35eeb3`);
`attack.stick_bias` (superseded by `weapon.switch_margin`, was already being
silently ignored by five stale a25/a26 configs before this pass — `7fa8a23ee`).
See REMOVED below for the full rationale.

### `attack.hold_tail_sec` — the one engine-applied knob (2026-08-26)

Absence is **not** "off" here, and that asymmetry is deliberate. In python the
key does nothing (there has never been an offline hold-tail — `qnn.eval`
already models the tail-free world). In the live bin an **absent stamp** means
the historical `0.25` s tail (`QNN_FIRE_HOLD_SEC`), because archived `.onnx`
files cannot be re-exported (era-locked generations) and must keep the behavior
they shipped with. `tools/export_onnx` therefore stamps the resolved value on
**every** export — config value or the `0.0` default — so a fresh model always
states its law explicitly and only pre-key artifacts fall through to the engine
constant. To change an artifact already on the share:
`tools/stamp_onnx.py --attack-hold-tail-sec <sec>`. See
[`contracts/README.md`](contracts/README.md#decode-config--the-decodeguard-layer)
and the [`wire.11` correction note](contracts/wire/wire.11.md).

The hold-tail was never in-graph and never carried in `attack_state` — that
tensor's 4 lanes (`y`, `locked_weapon`, `af_prev`, `dt`) carry `weapon.af_lockout`
today (`qnn.model.decode_actions.ATTACK_STATE_DIM`). Two prior docs asserted the
opposite (`wire.11.md`, `src/qnn/engine_norm.py`'s wire-table comment); the
`wire.11` doc carries a dated correction, the `engine_norm.py` comment does not
(reported, not fixed here — outside this file's scope).

## `weapon.*` — selection & feasibility

| key | attribute (init default) | registry default | `decode.base.json` | risk |
|---|---|---|---|---|
| `weapon.preference_bias_vec` | `weapon_preference_bias_vec=None` (`policy.py:310`) | `None` | ✓ (zero 8-vector) | SILENT while unset. Selection-only additive bias; validated as an 8-vector |
| `weapon.switch_margin` | `weapon_switch_margin=0.0` (`policy.py:313`) | `0.0` (a **provable no-op**) | ✗ (omitted — takes `0.0`) | SILENT while `0.0`. Same-tick raw-argmax vs. preference-adjusted-ideal gate — see RESTORED below |
| `weapon.infeasible_vec` | `weapon_infeasible_vec=None` (`policy.py:316`) | `None` | ✗ (omitted — takes `None`) | SILENT while unset. Static per-run exclusion (`>0.5` masked to `-1e9`, upstream of selection/fire) — see RESTORED below |
| `weapon.af_lockout` | `weapon_af_lockout=0.0` (`policy.py:317`) | `0.0` | ✗ (omitted — takes `0.0`) | SILENT while `0.0`. Multiplier on the real observed `self_arsenal_scalars` `attack_finished` cooldown — see RESTORED below |
| `weapon.af_lockout_cap` | `weapon_af_lockout_cap=0.0` (`policy.py:318`) | `0.0` (uncapped) | ✗ (omitted — takes `0.0`) | INERT while `weapon.af_lockout` is `0.0`. Ceiling in seconds on the `af_lockout` extension |

**REMOVED 2026-08-26** (`7fa8a23ee`, `0c5e892bd`, `047bbae02`): the whole
post-a26 weapon-selection surface — `weapon.continue_prob_vec`,
`weapon.continue_bias_vec`, `weapon.choice_prob_matrix`,
`weapon.choice_temperature` (the human weapon-transition/continuation law),
`weapon.switch_evidence_decay` (the SPRT evidence accumulator's λ — its
`switch_margin` companion was removed the same day and restored, see below),
`weapon.switch_lockout_mult`, `weapon.switch_lockout_cap_ticks` (the a26-lineage
static per-weapon switch lockout, superseded by `weapon.af_lockout` /
`weapon.af_lockout_cap`). See REMOVED below for why.

## `guard.*` — world-geometry safety rails

Bound by `make_guard(params)` in `qnn.model.guard` — a separate default surface
from the `DECODE_PARAMS` policy/export path, and **only evaluated when the
config's `guard_module` is not `"none"`** (`resolve_decode_config` skips
`make_guard` entirely for a guard-less config).

| key | default source | required? | `decode.base.json` | risk |
|---|---|---|---|---|
| `guard.projectile_release` | `make_guard` (`guard.py:456`) | **NO_DEFAULT — REQUIRED** whenever `guard_module` is set | ✓ (`"rocket"`) | Gate B inbound-projectile move-hold-release MODE: `"off"` \| `"rocket"` \| `"any"`. A bool is refused (it absorbed the old `guard.projectile_release_mode` flag) |
| `guard.lg_range` | `make_guard` (`guard.py:473`) | **NO_DEFAULT — REQUIRED** whenever `guard_module` is set | ✓ (`0`, i.e. off) | LG beam range in Quake units; `0` disables the guard, any positive value arms it. A bool is refused (would silently mean `1u`) |

`guard.lg_range` marks LG **infeasible** for the tick when no enemy actor is
within range (`qnn.model.guard.lg_out_of_range`) — the same feasibility layer
ammo/ownership already ride. It does **not** veto the shot and does not decide
which weapon replaces LG; the network picks among what's left. The RL
self-splash veto (always on, no config knob) is unaffected by either key.

**REMOVED 2026-08-26, NOT restored** (`7fa8a23ee`, `269881170`):
`guard.lg_range_u` and `guard.lg_align_half_angle_deg` (merged into the single
`guard.lg_range` value — one key, one quantity, no separate on/off flag to
disagree with it), `guard.lg_range_mode` and `guard.lg_veto_unseen` (the
veto/selection-mask mode switch — one key deciding between two entirely
different mechanisms in two different families was the clearest instance of
the layering problem this whole cut addressed), and `guard.projectile_release_mode`
(merged into `guard.projectile_release`, the same shape of problem).

## REMOVED DECODE LAWS — the 2026-08-26 decode-surface purge

**BREAKING.** Brian, 2026-08-26: *"clean slate."* The post-a26 attack/weapon
surface had grown into four mutually-exclusive selection laws, three masking
layers and five guard vetoes across four stages, with enough cross-stage
leakage to need a dedicated key (`attack.fire_gather`) just to reconcile which
weapon two stages were discussing. Measured on the live RA venue, the human
weapon-transition sampler discarded the network's own weapon choice on 52–58%
of firing ticks (`chosen==argmax_raw` 0.482 armed vs. 0.742 under the plain
margin law), and with `continue_prob_vec=[0]*8` it could not reach LG at all
(equip 12.5% vs. 55.8%). A heuristic overriding the model, which this project
does not do.

A key naming a removed law is not silently ignored: it fails
`_validate_known_params` and refuses the config to load (see Mechanism above).
There is no parallel "rejected keys" registry — this section is prose, not
code, because the *rationale* is not recoverable from a key's mere absence.

**Wave 1 — the attack/weapon param registry** (`7fa8a23ee`, 15 keys):
`weapon.continue_prob_vec`, `weapon.continue_bias_vec`, `weapon.choice_prob_matrix`,
`weapon.choice_temperature`, `weapon.switch_evidence_decay`,
`weapon.switch_lockout_mult`, `weapon.switch_lockout_cap_ticks`, `attack.fire_gather`,
`guard.lg_range`, `guard.lg_range_u`, `guard.lg_range_mode`,
`guard.lg_align_half_angle_deg`, `guard.lg_veto_unseen`, `attack.stick_bias`,
and `weapon.infeasible_vec` (temporarily — restored the same day, see below).
`guard.lg_range` was cut here on the mistaken premise that the whole LG
range/alignment guard was post-a26 machinery; it is a24-lineage world-geometry
(14 a24 templates and `a28rc1c/d/e` set it) and was restored the same day in a
narrower form (Wave-1-restore, below).

**Wave 2 — the decode_actions selection laws** (`0c5e892bd`, `269881170`,
`047bbae02`, `6ae35eeb3`): the human transition/continuation sampler, the SPRT
evidence accumulator (`qnn.decode_fit.weapon_switch` — its own sweep had
already concluded "no chatter signal survives — lambda=0 is the answer"), the
static per-weapon switch lockout table, the choice-gathered fire selector
(`qnn.decode_fit.fire_gather`), and the LG range/alignment guard's veto/mask
mechanism (`lg_range_guard_mask`, `lg_unconnectable`).

**Wave 3 — `weapon.switch_margin`** (`cf2af1c14`, same day): *"there is no held
weapon"* — retired the anchor concept together with the key it gated. Restored
hours later (Wave-1-restore) once it was established the underlying math never
actually needed the anchor (see RESTORED below).

**Wave 4 — the legacy joint attack law** (`5b64a90bb`): `attack.bias_vec` (the
a26 joint selection+fire vector) and `attack.bias` (the global propensity
scalar) were zero in every a27/a28 config — new fits pinned them to zero by
contract. With the joint vector gone there was exactly one fire law, so
`attack.vector_semantics` — which existed only to declare "split, not joint" —
had nothing left to select and went with it.

**Wave 5 — the a25 discharge-quality ("crest") gate** (`a29ba2697`):
`attack.crest_theta_vec` / `attack.crest_hold_ticks` — no a27 or a28 config
ever armed it, and its countdown latch was the last live consumer of
`attack_state` before `weapon.af_lockout` (Wave-1-restore) reused the slot for
something new.

**Wave 6 — `move.commitment` / `move.commit_interrupt` / `move.commit_recommit`**
(`42e173a5c`, `da7b9368f`): `move.commitment` was never actually a config
choice (QNNPolicy forced it from the checkpoint's own head structure
regardless of what the key said); `commit_interrupt` was `false` in every
config and `commit_recommit` was set by none — both only gated code paths that
went with them.

**Wave 7 — the feet-aim pitch family and the DOWN-band degraders** (`7b6c2030d`):
`look.weapon_pitch_gain/_bias/_mode/_shift_strength/_lock` (a24 RL feet-aiming;
`_lock` was a back-compat alias of `_mode` — two keys for one knob),
`look.aim_degrade_sluggish_tau/_lag_frames/_jitter_mag` and `look.hold_drift_eps`
(eval-side research knobs). None were armed in any a27/a28 config; `_apply_aim_degrade`
drops to the tremor stage alone (tremor stays — it is live and fitted).

## RESTORED same day — feasibility, not selection

Weapon **selection** belongs to the network; a decode-side law that decides
WHICH weapon among the feasible set on grounds other than the network's own
logits is not coming back. **Feasibility** — narrowing the candidate set
without picking among what's left — is a different question, and four keys
were restored on 2026-08-26 (`ee92c9d8f`, `4bf4c5028`) after Wave 1 went
further than warranted:

- **`weapon.switch_margin`** — re-litigated as a pure **same-tick confidence
  gate**: the a28 obs contract carries no equip state, so the held-weapon
  anchor it originally gated could only ever be this tick's own raw argmax —
  it had already degenerated to that before the removal. `0.0` (default) is a
  provable no-op (always take the preference-adjusted ideal). Configs fitted
  under the pre-removal build resolve unchanged (it applied on 8.6% of ticks at
  margin 1.496 vs. 30% at margin 0 for the a28rc1e fit).
- **`weapon.infeasible_vec`** — static per-run exclusion. Unlike the removed
  static feasibility mask, this is one declared vector, not a law with its own
  accumulator/lockout/matrix machinery riding on it.
- **`weapon.af_lockout` / `weapon.af_lockout_cap`** — a switch-lockout, but
  keyed off the engine's own `self_arsenal_scalars` `attack_finished` countdown
  (real per-weapon refire truth already in the obs) rather than a re-derived
  held-weapon anchor, and with no mutual-exclusion with a transition/continuation
  law. `af_lockout` is a multiplier on the real per-discharge cooldown (`0` =
  none); `af_lockout_cap` a ceiling on the extension in seconds (`0` =
  uncapped) — the a26-lineage `switch_lockout_mult` / `switch_lockout_cap_ticks`
  roles, generalized to a real observed value instead of a static per-weapon
  table. Real precedent: `a28rc1h` shipped `switch_lockout_cap_ticks=6` (0.3 s
  at 20 Hz), formula `lockout = cd + min(cd, T)` (`agents/plans/rl-skill-finetune.md`).
- **`guard.lg_range`** — restored as a pure range-in-units feasibility mask
  (see `guard.*` above); the alignment cone and the veto/selection-mask switch
  were **not** restored (the cone was inert at 180° in every a28 config that
  shipped it; the mode switch put a fire veto and a selection mask behind one
  key, the exact layering problem the cut addressed).

Two live bugs in the restored `weapon.af_lockout` were found and fixed the same
week (`acc319410`): (1) `self_arsenal_scalars` is dequantized `/60`, so
`af_lockout_cap` — documented and written in configs as real seconds — never
bound at any realistic value until the read was rescaled; (2) the lockout's
decay window was keyed to the engine's `attack_finished` fully clearing instead
of to this tick's own fire decision, so a held continuous-weapon burst (whose
`attack_finished` never truly clears mid-burst) tacked the full extension onto
the end of however long the engine's own hold-tail kept the button down. Both
are fixed at HEAD; see `qnn.model.decode_actions.attack_with_decode`'s
`af_lockout` branch.

## Status

The 2026-07 "promote silent defaults to required" program (formerly tracked
here as a rolling checklist) is **complete**: every `look.*` / `move.idle_*`
key identified as SILENT-HIGH risk is now `REQUIRED_PARAM_KEYS` or
`MODULE_REQUIRED_PARAM_KEYS`, and every key with no working implementation
(promoted or not) fails to load outright (`_validate_known_params`). Nothing in
this file is a pending recommendation any more — a key not marked REQUIRED
above has a code-side default because it is a genuine, fitted-per-checkpoint
operating point (tremor, `move.commit_dur_tilt`, `move.threat_break_hazard`,
`jump.threshold`, the attack/weapon vectors, `weapon.af_lockout*`), not an
oversight.
