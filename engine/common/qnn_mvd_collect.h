/*
 * qnn_mvd_collect.h — MVD reconstruction path for the QW demo worker.
 *
 * Used when no usercmd_t is available (real MVD demos OR `force_mvd_emit`
 * on QWD).  Three discrete-cmd inference paths plus one move path:
 *
 *   ATTACK sound (weapon-fire PHS multicast) → walkback by full ping
 *          → attack=1 → co-temporal dedup. One operative press per event;
 *          no hold tail.
 *   JUMP   sound (player/plyrjmp8.wav) → walkback by full ping →
 *          move[2]=jump_speed → grounded-count chain gate.
 *          One operative press per event; no hold tail.
 *   SWITCH per-emit action.weapon from snapshot; on weapon_id
 *          transitions, rewrite trailing K slots back to the press
 *          frame (pickup gate at call site suppresses server-forced
 *          touches)
 *   MOVE   per-emit fb/lr from view-relative position-delta sign;
 *          back-shifted into the ring by QNN_MvdBackShiftWriteMoveXY
 *
 * All MVD-private state (back-shift ring, per-weapon dedup tables)
 * lives as module-private static inside qnn_mvd_collect.c.  Callers
 * reset it at demo start via QNN_MvdCollectReset().
 */

#ifndef QNN_MVD_COLLECT_H
#define QNN_MVD_COLLECT_H

#include "qnn.h"
#include "qnn_collect_helpers.h"

/* ── Module reset (per demo) ──────────────────────────────────────── */

/* Reset the back-shift ring and per-weapon attack/jump dedup tables.
 * The demo_path_seed argument is retained for call-site compatibility
 * but is no longer used (the hold-sim RNG it seeded was removed). */
void QNN_MvdCollectReset(uintptr_t demo_path_seed);

/* ── Action inference ─────────────────────────────────────────────── */

/* Per-native-frame attack detection — emits attack=1 only on the native frame
 * where a self weapon-fire sound is detected.  Updates the module's
 * native_attack_this_window latch (mirrored in qnn_runtime for NQ compatibility). */
void QNN_MvdInferNativeAction(qnn_action_t *action,
	const qnn_snapshot_t *snapshot);

/* Per-emission look/switch action.  Movement is filled separately by
 * QNN_MvdInferEmitMove. */
void QNN_MvdInferEmitAction(qnn_action_t *action,
	const qnn_snapshot_t *snapshot);

/* Per-emission move label.  fb/lr from view-relative position-delta
 * sign; ud filled by the back-shift jump-event writer (water uses
 * position-delta swim up/down). */
void QNN_MvdInferEmitMove(qnn_action_t *action,
	const qnn_snapshot_t *snapshot, float emit_dt);

/* ── MVD sound/move back-shift writers ────────────────────────────────
 *
 * The generic back-shift ring (push/flush/slot-at/rewrite/accessors) now
 * lives in qnn_collect_helpers.h.  These remain MVD-specific: they walk
 * weapon-fire / jump sound events back to the press frame and stamp the
 * resolved slot via the shared ring. */

/* Per-event attack/jump back-shift driven by sound native_time. */
void QNN_MvdBackShiftWriteAttackEvents(const qnn_snapshot_t *snapshot,
	float ping_sec, float emit_start_native_time);
void QNN_MvdBackShiftWriteJumpEvents(const qnn_snapshot_t *snapshot,
	float ping_sec, float emit_start_native_time);

/* Copy the current emit's move XY to the back-shifted slot. */
void QNN_MvdBackShiftWriteMoveXY(uint8_t move, int shift_frames);

/* Reset the per-weapon attack dedup state for `weapon_id` (1..8).  Called
 * when the held weapon changes so a later same-weapon shot isn't
 * false-linked through a different-weapon interval. */
void QNN_MvdResetAttackChain(int weapon_id);

/* ── De-scripted intent label (MVD parity, weapon-head.md §12) ─────────
 *
 * MVD mirror of QNN_QwdIntentWeaponLabel: act.weapon = the player's
 * deliberate weapon carried forward, reconstructed from held-weapon
 * transitions + attack sounds instead of usercmd select edges (which .mvd
 * demos lack).  Per-transition classification:
 *
 *   deliberate  any transition not classified below → adopt the new
 *               held weapon; caller back-shift-rewrites the trailing
 *               ring slots (press-frame anticipation, same shift source
 *               as the old held rewrite).
 *   dump        attack-scripted demo + transition INTO the demo's release
 *               target + dump evidence: a recent attack of the outgoing
 *               weapon (the deferred release-half equip) or a recent
 *               respawn (the respawn press's release half dumps with no
 *               shot) → config churn, label frozen.  Without either the
 *               equip is a deliberate choice — the axe-release
 *               population genuinely fights with its release weapon,
 *               and axe swings are inaudible to the attack path, so
 *               over-suppression cannot be corrected back.
 *   forced      pickup auto-equip (IT_ bit 0→1, detected here from the
 *               ring's prev stat_items) or intent no longer QC-
 *               selectable (ammo-out auto-switch) → adopt held at the
 *               observed frame, no press lead.
 *   death       label 0 (masked) while dead; the first alive frame
 *               adopts the spawn weapon (respawning takes a button
 *               press, and QWD adopts held on that shot-less attack
 *               press — waiting for an attack sound would mask the whole
 *               post-spawn re-arm).  An attack sound also re-reveals at
 *               its back-shifted press slot for any other masked span.
 *
 * The attack path stamps slot->action.weapon only when the sound's weapon
 * AGREES with intent (pure timing fix) or when intent is unrevealed
 * (adopt).  A disagreeing sound leaves the label alone — QWD truth rides
 * revealed intent through unrealizable presses, so overriding from
 * sound would diverge from the labels this path is validated against. */

typedef enum
{
	QNN_MVD_WTRANS_NONE = 0,	/* no transition this emit */
	QNN_MVD_WTRANS_DELIBERATE,
	QNN_MVD_WTRANS_DUMP,
	QNN_MVD_WTRANS_FORCED,
} qnn_mvd_wtrans_t;

/* QC-feasibility callback: return nonzero iff `weapon` (raw 1..8) is
 * selectable for the snapshot's stats given `current` held.  Supplied by
 * the QW call site (QNN_ProgsEvalWeaponImpulse) so this common module
 * stays progs-free. */
typedef int (*qnn_mvd_select_feasible_fn)(const qnn_snapshot_t *snapshot,
	int current, int weapon);

/* Advance the intent state for this emit tick and return the act.weapon
 * label (raw 1..8, 0 = unrevealed/masked).  Call once per emit, after
 * the ring pushed last tick's slot and before this tick's push.
 * `out_trans`/`out_prev_intent` (either may be NULL) report the
 * transition class and the pre-transition intent so the caller can
 * back-shift-rewrite deliberate adoptions. */
int QNN_MvdIntentWeaponStep(const qnn_snapshot_t *snapshot,
	qnn_mvd_select_feasible_fn feasible,
	qnn_mvd_wtrans_t *out_trans, int *out_prev_intent);

#endif /* QNN_MVD_COLLECT_H */
