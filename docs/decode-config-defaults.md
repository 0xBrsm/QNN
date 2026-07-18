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

### a25 module extras (`MODULE_REQUIRED_PARAM_KEYS["qnn.model.bench.a25.decode"]`)

Promoted 2026-07-13 (this pass): `move.commitment`, `look.hold_passthrough`.
Both live a25 templates now carry them (`a25rc1` had omitted
`look.hold_passthrough` → silently ran passthrough OFF; backfilled to `true`).

## Scope note

a24 decode/guard modules are **deleted** (`test_a24_modules_are_gone`); a24
templates (`decode.a24rc1*`) can no longer resolve — module import fails before
the param check. The manifest therefore only needs to be satisfied by the **live
a25 configs** (`decode.a25base.json`, `decode.a25rc1.json`) and every
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

## `attack.*` — attack-with operating point (DEFERRED)

| key | policy attr / default | export default | a25base | a25rc1 | risk |
|---|---|---|---|---|---|
| `attack.bias` | `attack_bias=0.0` (:277) | `0.0` | ✓ | ✓ | SILENT |
| `attack.bias_vec` | `attack_bias_vec=None` (:283) | `None` | ✓ | ✗ | SILENT |
| `attack.stick_bias` | `attack_stick_bias=0.0` (:284) | `0.0` | ✓ | ✓ | SILENT |

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
4. **`attack.bias` / `attack.stick_bias`** — both live templates set them; safe
   to promote.
5. **`look.weapon_pitch_gain` / `_bias`** — a24 RL feet-aim; **omitted on both a25
   templates.** Confirm feet-aim is intentionally OFF for a25 before either
   promoting (with explicit `[0]*9`) or leaving optional. Do **not** guess.

Genuinely inert (leave optional, no action): `look.aim_degrade_sluggish_tau`,
`look.aim_degrade_jitter_mag` (research-only, never emitted),
`look.aim_degrade_lag_frames` (retired, popped on emit),
`look.weapon_pitch_mode/_shift_strength/_lock` (gated by pitch_gain),
`look.aim_degrade_tremor_tau` (inert while mag=0).
