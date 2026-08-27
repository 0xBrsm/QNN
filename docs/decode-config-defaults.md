# Decode-Config Defaults — Fail-Loud Audit

Source-of-truth inventory of every decode-config `params.*` key that currently
has a **code-side default**, and whether that default is a *silent* fallback
(absence changes closed-loop behavior with no error) or genuinely inert.

Motivation: the run-config philosophy — *"No code-level defaults fill in missing
training keys"* (`src/docs/run.md`) — was **not** enforced for the decode config.
A key applied only when present (`if "look.X" in p:` in `qnn.eval.run`, or
`params.get("look.X", <default>)` in `tools/export_onnx`) silently falls back to
a `policy.__init__` / kwarg default when omitted. That is exactly how the a25
decode line lost the a24 hazard-discounted lead caps (`look.lead_hold_cap_frames`
= 4 / `_radial` = 5) and silently ran a linear over-lead, and how the a25
`look.turn_mag_scale` dampener regressed to a cross-arch value.

## Mechanism (implemented 2026-07-13)

- **Per-decode-module manifest** in `src/qnn/model/decode_config.py`. The required
  set = a shared **BASE** (`REQUIRED_PARAM_KEYS`, the look.* aim-geometry family
  every look-decode arch carries) **UNION** the extras registered for the config's
  own `decode_module` (`MODULE_REQUIRED_PARAM_KEYS`). This keeps arch-owned keys
  (e.g. the a25 segment-commitment move decode) required **only** for that arch's
  configs, so promoting one does not break a pre-a25 arch whose decode law never
  reads it. `_required_param_keys(decode_module)` returns BASE + module extras.
- Validated once, inside `resolve_decode_config`, which **both** consumers
  (`qnn.eval.run` and `tools/export_onnx`) call — so they cannot drift.
- A config missing any required key raises `ValueError` naming the **config path**,
  the **decode_module**, and the **missing key(s)**. No default is substituted.
  OFF, where wanted, must be an explicit `0.0`/`false` — never omission.
- The presence-gating (`if key in p` / `.get(key, default)`) for the required
  keys was replaced with direct `p[key]` access at the applier sites
  (`qnn.eval.run._apply_decode_config_params`, `tools/export_onnx.main`).

## The param registry (implemented 2026-07-25)

The *mapping* — config key → policy attribute / ExportWrapper kwarg → coercion →
default — used to be hand-written at **three** sites (`PARAM_TO_KWARG`, the
`if "key" in p:` chain in `qnn.eval.run`, the `.get(key, default)` block in
`tools/export_onnx`) and it **drifted**: `attack.crest_theta_vec` /
`attack.crest_hold_ticks` reached the exporter but were never applied in offline
eval, so a crest-gated config baked the discharge gate into the ONNX while
`qnn.eval` ran without it. No pinned config had ever set them, so nothing
shipped wrong — but the divergence was live in the code.

There is now ONE table, `DECODE_PARAMS` in `src/qnn/model/decode_config.py`:

| field | meaning |
|---|---|
| `key` | the decode-config `params.*` key |
| `name` | the `QNNPolicy` attribute **and** the `ExportWrapper` kwarg (identical by contract) |
| `coerce` | the raw-JSON → normalized-value function (shared by both consumers) |
| `default` | the value used when the config omits the key; `NO_DEFAULT` = fail loud |
| `export_name` | the ExportWrapper kwarg when it differs (only the tremor pair) |
| `graph` | `False` = eval-only law with no in-graph twin (see below) |

Consumers read it through two accessors and nothing else:
`ResolvedDecode.policy_attrs()` (eval — one `setattr` loop, raises if the
attribute does not exist on the policy) and `ResolvedDecode.export_kwargs()`
(export — splatted straight into `ExportWrapper`). Adding a knob = adding a row.
`tests/test_decode_param_registry.py` fails if the registry, `QNNPolicy` and
`ExportWrapper` fall out of step, including on the defaults.

`graph=False` knobs (`look.weapon_pitch_lock`, `look.aim_degrade_sluggish_tau`,
`look.aim_degrade_lag_frames`, `look.aim_degrade_jitter_mag`) are eval-only by
design — the retired/research DOWN-band degraders and the `_mode` back-compat
alias. They are marked in the table rather than merely missing from the export
block, and the test asserts `ExportWrapper` does **not** accept them.

### a25 module extras (`MODULE_REQUIRED_PARAM_KEYS["qnn.model.decode_actions"]`)

Promoted 2026-07-13 (this pass): `move.commitment`, `look.hold_passthrough`.
Both live a25 templates now carry them (`a25rc1` had omitted
`look.hold_passthrough` → silently ran passthrough OFF; backfilled to `true`).

## Scope note

a24 decode/guard modules are **deleted** (`test_a24_modules_are_gone`); a24
templates (`decode.a24rc1*`) can no longer resolve — module import fails before
the param check. The manifest therefore only needs to be satisfied by the **live
a25 configs** (`decode.base.json`, `decode.a25rc1.json`) and every
freshly-emitted a25 decode-fit config (built from `a25base` as `ORACLE_TEMPLATE`).
Both live templates were backfilled so they carry all required keys.

## Legend

- **Risk** — `SILENT` = absence changes behavior via a code default (the bug
  class); `INERT` = default is a true no-op / key never emitted / provenance-only.
- **a25base / a25rc1** — ✓ set in the committed template, ✗ omitted.
- Setter presence-gated = applied only `if key in p` (eval) / `.get(default)`
  (export) — i.e. silently defaulted when absent, unless promoted to required.

## `look.*` — aim geometry (in scope this pass)

| key | policy attr / default (`policy.py`) | export default | a25base | a25rc1 | status | risk |
|---|---|---|---|---|---|---|
| `look.aim_prior_gain` | `look_aim_prior_gain=None` (:270) | via `kw` | ✓ | ✓ | **REQUIRED** | SILENT |
| `look.aim_ffwd_gain` | `look_aim_ffwd=None` (:307) | via `kw` | ✓ | ✓ | **REQUIRED** | SILENT |
| `look.aim_mag_gain` | `look_aim_mag_gain=0.0` (:313) | `0.0` | ✓ | ✓ | **REQUIRED** | SILENT |
| `look.turn_mag_scale` | `look_turn_mag_scale=1.0` (:326) | `1.0` | ✓ | ✓ | **REQUIRED** | SILENT |
| `look.lead_hold_cap_frames` | `look_lead_hold_cap_frames=None` (:365) | `None` | ✓ (4.0) | ✓ (4.0) | **REQUIRED** | SILENT |
| `look.lead_hold_cap_radial_frames` | `look_lead_hold_cap_radial_frames=None` (:368) | `None` | ✓ (5.0) | ✓ (5.0) | **REQUIRED** | SILENT |
| `look.hold_passthrough` | `look_hold_passthrough=False` (:339) | `False` | ✓ (true) | ✓ (true) | **REQUIRED** (a25 module) | SILENT |
| `look.hold_drift_eps` | `look_hold_drift_eps=0.0` (:334) | `0.0` | ✗ | ✗ | optional (emitted only when >0) | SILENT-if-intended |
| `look.aim_degrade_tremor_mag` | `look_aim_degrade_tremor_mag=0.0` (:391) | `0.0` | ✓ | ✗ | optional — **NEEDS RULING** | SILENT |
| `look.aim_degrade_tremor_tau` | `look_aim_degrade_tremor_tau=5.0` (:392) | `5.0` | ✓ | ✗ | optional (paired with tremor_mag) | INERT when mag=0 |
| `look.aim_degrade_lag_frames` | `look_aim_degrade_lag_frames=0.0` (:386) | — | ✓ (dead) | ✗ | RETIRED 7/10 (popped on emit) | INERT |
| `look.aim_degrade_sluggish_tau` | `look_aim_degrade_sluggish_tau=0.0` (:383) | — | ✗ | ✗ | research-only, never emitted | INERT |
| `look.aim_degrade_jitter_mag` | `look_aim_degrade_jitter_mag=0.0` (:396) | — | ✗ | ✗ | rejected baseline, never emitted | INERT |
| `look.weapon_pitch_gain` | `look_weapon_pitch_gain=None` (:345) | `None` | ✗ | ✗ | optional — **NEEDS RULING** (a24 RL feet-aim; OFF on a25?) | SILENT |
| `look.weapon_pitch_bias` | `look_weapon_pitch_bias=None` (:349) | `None` | ✗ | ✗ | optional — **NEEDS RULING** | SILENT |
| `look.weapon_pitch_mode` | `look_weapon_pitch_mode="lock"` (:359) | `"lock"` | ✗ | ✗ | optional (gated by pitch_gain) | INERT when gain off |
| `look.weapon_pitch_shift_strength` | `=1.0` (:360) | `1.0` | ✗ | ✗ | optional (gated) | INERT when gain off |
| `look.weapon_pitch_lock` | `look_weapon_pitch_lock=True` (:358) | — | ✗ | ✗ | back-compat alias of `_mode` | INERT |

## `move.*` — commitment decode

| key | policy attr / default | export default | a25base | a25rc1 | risk |
|---|---|---|---|---|---|
| `move.commitment` | `move_commitment=False` (:288) | `False` | ✓ (true) | ✓ (true) | **REQUIRED** (a25 module) — was SILENT-HIGH (absent → seg-commitment move decode OFF); now fails loud in both eval + export |
| `move.commit_interrupt` | `move_commit_interrupt=True` (:298) | `True` | ✓ | ✓ | SILENT |
| `move.commit_dur_tilt` | `move_commit_dur_tilt=(0.0,0.0)` (:292) | `[0,0]` | ✓ | ✓ | SILENT |
| `move.threat_break_hazard` | `move_threat_break_hazard=0.0` (:304) | `0.0` | ✓ | ✗ | SILENT |
| `move.commit_recommit` | `move_commit_recommit=False` (:331) | `False` | ✗ | ✗ | optional — absent/`False` = maximal-run law (no re-commit to the held class before expiry); SILENT |
| `move.idle_none_bias` | `move_idle_none_bias=(0.0,0.0)` (:337) | `[0.0,0.0]` | ✗ | ✗ | optional — engagement-gated idle stillness; `(0,0)` = off; SILENT. **Fit it with `qnn.decode_fit.move_gates`, and read the a26rc1 values as ~10x too small** — see the ruler note below |
| `move.idle_engagement_base` | `move_idle_engagement_base=0.5` (:338) | `0.5` | ✗ | ✗ | optional — paired with `idle_none_bias`, inert when that is `(0,0)` | INERT when idle_none_bias off |
| `move.idle_cooldown_ticks` | `move_idle_cooldown_ticks=20` (:339) | `20` | ✗ | ✗ | optional — paired with `idle_none_bias` | INERT when idle_none_bias off |

### Fitting the two move gates — and the a26rc1 ruler error

`move.idle_none_bias` and `jump.threshold` are per-checkpoint constants: each
inverts THIS model's posterior against a human rate, so neither transfers
between checkpoints (the same rule `move.commit_dur_tilt` follows). Both are fit
by `qnn.decode_fit.move_gates`, which recovers and replaces the uncommitted
a26rc1b-era scripts.

**β (`move.idle_none_bias`) — the a26rc1 values are ~10x too small.** The
original fit scored a move class by SUMMING its ten bucket logits and taking an
argmax, adding β to each none-bucket logit first — so β entered the none score
as **+10β**. The decode does something else: it softmaxes the 30-way joint,
where the same β shifts `none`'s mass by `exp(β)`, i.e. `+β` on its log-sum-exp.
The fit therefore reached its human target (39.7% out-of-combat stand-still) on
a ruler ~10x more sensitive than the shipped law. Measured against the real
decode, `a26rc1b`/`c`/`d`'s `β = 0.31` moves realized stand-still from 19.1% to
about 21.6% — roughly an eighth of the correction its provenance claims. Fit β
against the REALIZED occupancy of the decode itself (`move_gates
--statistic rollout`), never a hand-rolled proxy of it.

**τ (`jump.threshold`) is placed by cut factor, not human parity.** `a26rc1b`'s
`0.60` is a ~4x cut against that model's OWN sampled posterior rate on
jump-feasible engaged frames, which lands it ~6.4x BELOW the human rate — the
human number is a diagnostic, never a target
(`jump: no rate calibration`). Reproducing the historical numbers needs the
engaged population loaded WITH `segment_mask={"act.target": {"$ne": 0}}`: masking
before the forward gives the GRU a spliced sequence and shifts p_jump ~0.1% from
the same-frames-selected-after convention (which is what the bot experiences in
play, since it never skips frames).

## `jump.*` — confidence gate

| key | policy attr / default | export default | a25base | a25rc1 | risk |
|---|---|---|---|---|---|
| `jump.threshold` | `jump_threshold=0.0` (:352) | `0.0` | ✗ | ✗ | optional — absent/`0.0` = AS-IS decode (no deterministic confidence gate on the up/down jump axis); a hand-derived deploy addition (`a26rc1b`: `0.60`, confident-only jump), not part of the base template | SILENT |

## `attack.*` — attack-with operating point

| key | policy attr / default | export default | a25base | a25rc1 | risk |
|---|---|---|---|---|---|
| `attack.bias` | `attack_bias=0.0` (:277) | `0.0` | ✓ | ✓ | SILENT |
| `attack.bias_vec` | `attack_bias_vec=None` | `None` | ✓ (zero) | ✗ | LEGACY JOINT — selection + fire on a26; never emit nonzero with split-v1 |
| `attack.fire_bias_vec` | `attack_fire_bias_vec=None` | `None` | ✓ (zero) | ✗ | split-v1 fire-only; validated as an 8-vector |
| `weapon.preference_bias_vec` | `weapon_preference_bias_vec=None` | `None` | ✓ (zero) | ✗ | split-v1 selection-only; validated as an 8-vector |
| `attack.vector_semantics` | provenance/validation | — | ✓ (`split_v1`) | ✗ | required when either explicit vector is present |
| `weapon.switch_margin` | `weapon_switch_margin=0.0` | `0.0` | ✓ | ✗ | held-weapon hysteresis; final-combat calibrated |
| `attack.crest_theta_vec` | `attack_crest_theta_vec=None` | `None` | ✗ | ✗ | discharge-quality gate; (8,) θ per impulse-1, ≤0 = OFF per weapon. **Was export-only until 2026-07-25** (inert in eval); no pinned config sets it | SILENT |
| `attack.crest_hold_ticks` | `attack_crest_hold_ticks=0` | `0` | ✗ | ✗ | shared max crest hold H; `0` = gate OFF globally. Same export-only drift, same fix | INERT while 0 |

Fresh decode-fit artifacts use `split_v1`. `resolve_decode_config` requires both
explicit vectors, an all-zero legacy vector, and the semantics marker. Old
configs without explicit vectors continue to resolve under their historical
law; the resolver never guesses which branch meaning was intended.

## `weapon_ban` / `guard.*` (DEFERRED)

| key | default source | a25base | a25rc1 | risk |
|---|---|---|---|---|
| `weapon_ban` | `weapon_ban=()` (policy :414); export **refuses** non-empty on a25 | ✓ ([]) | ✓ ([]) | guarded (export raises) |
| `guard.projectile_release` | `params.get(..., True)` in `a25/guard.make_guard` | ✓ | ✓ | SILENT (default True) |
| `guard.projectile_release_mode` | `params.get(..., "rocket")` in `make_guard` | ✓ (rocket) | ✓ (any) | SILENT (default "rocket") |
| `guard.attack_align_strength` | consumed by guard adapter | ✗ | ✓ | provenance/guard-local |

Guard params are bound by `make_guard(params)` (a separate default surface from
the policy/export kwarg path); on a25 the guard reads only the two
`guard.projectile_release*` keys above.

## Recommended full rollout (for Brian's ruling)

Promoting to **required** would close the remaining silent-default siblings, but
each needs a live-template backfill first (any promoted key must exist in *all*
live templates, or resolve breaks):

1. ~~**`move.commitment`**~~ — **DONE 2026-07-13** (a25-module required key;
   a24 templates fail earlier at module import so this does not affect them).
2. ~~**`look.hold_passthrough`**~~ — **DONE 2026-07-13** (a25rc1 backfilled to
   `true`; promoted as an a25-module required key).
3. **`look.aim_degrade_tremor_mag` / `_tremor_tau`** — a25base sets them, a25rc1
   omits. Reconcile then promote (tremor is the universal DOWN-band demoter).
4. **`attack.bias`** — both live templates set it; safe to promote. The retired
   `attack.stick_bias` is not part of the reconciled law.
5. **`look.weapon_pitch_gain` / `_bias`** — a24 RL feet-aim; **omitted on both a25
   templates.** Confirm feet-aim is intentionally OFF for a25 before either
   promoting (with explicit `[0]*9`) or leaving optional. Do **not** guess.

Genuinely inert (leave optional, no action): `look.aim_degrade_sluggish_tau`,
`look.aim_degrade_jitter_mag` (research-only, never emitted),
`look.aim_degrade_lag_frames` (retired, popped on emit),
`look.weapon_pitch_mode/_shift_strength/_lock` (gated by pitch_gain),
`look.aim_degrade_tremor_tau` (inert while mag=0).
