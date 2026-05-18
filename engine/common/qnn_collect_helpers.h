/*
 * qnn_collect_helpers.h — Shared types, constants, and prototypes for the
 * NQ and QW collect main loops.  Included only by qnn_collect_main.c on
 * each engine and by qnn_collect_helpers.c — NOT by the trainer worker,
 * which has its own qnn_runtime_t with different fields.
 */

#ifndef QNN_COLLECT_HELPERS_H
#define QNN_COLLECT_HELPERS_H

#include "qnn.h"

/* ── Worker protocol identity ────────────────────────────────────── */

#define QNN_WORKER_PROTOCOL "v6"
#define QNN_WORKER_UPSTREAM_COMMIT "bf4ac424ce754894ac8f1dae6a3981954bc9852d"
#define QNN_WORKER_MAX_LINE 8192
#define QNN_WORKER_MAX_COMMAND_TEXT 1024

/* QNN_WORKER_SERVER_NAME is engine-specific ("qw-demo-worker" /
 * "quake-demo-worker") and stays in each engine's main file. */

/* ── Per-native-frame buffers for entity interaction ─────────────── */

#define QNN_MAX_NATIVE_FRAMES 16
#define QNN_MAX_PUSH_TRIGGERS 8

/* ── Emission-rate frame filters ─────────────────────────────────── */

#define QNN_GOD_MODE_HEALTH     250
#define QNN_DEAD_MAX_EMIT       20   /* 1 s at 20 Hz */
#define QNN_FROZEN_MAX_EMIT     40   /* 2 s at 20 Hz */

/* ── MVD back-shift ring (deferred label emit) ───────────────────── */

/* Number of emit-tick slots held in the back-shift ring.  At 20 Hz
 * emit, 24 slots cover up to 1200 ms of total look-back.  The ring
 * must reach back FAR enough for two purposes:
 *   1. Press-time back-shift: ping + server tick/2 (p99 ≈ 250 ms).
 *   2. Fire chain-fill: when a same-weapon shot lands ≤ cd+slack after
 *      the previous shot, the gap between them is filled with fire=1
 *      so held-burst behaviour reads as a continuous run.  RL has the
 *      longest cooldown (16 emit frames = 800 ms); with up to ~5 emit
 *      frames of additional back-shift on top, the previous shot's
 *      slot can be 21 emit frames behind the current push.
 * K = 24 covers both with a safety margin.  The ring delays MVD-path
 * emission by exactly QNN_BACKSHIFT_K ticks; flushed at demo end. */
#define QNN_BACKSHIFT_K 24

/* Hard cap on the impulse walk-back when anchoring a QWD weapon
 * transition to its causing impulse.  10 emit frames = 500 ms at
 * 20 Hz, well above any plausible click→server latency; if the
 * matching impulse isn't within this window the transition is
 * almost certainly server-driven (pickup, auto-switch on ammo-out,
 * etc.) and we leave the label at the server-observed frame. */
#define QNN_BACKSHIFT_MAX_IMPULSE_WALKBACK 10

typedef struct
{
	uint8_t		obs[QNN_OBS_BUFFER_SIZE];
	qnn_action_t	action;
	qboolean	grounded;
	int		tick;
	int		steps;
	int		tick_hz;
	qboolean	done;
	qboolean	reset_flag;
	FILE		*out;
	qboolean	valid;
	qnn_fire_route_event_t fire_routes[QNN_MAX_FIRE_ROUTE_EVENTS];
	int		fire_route_count;
	/* Weapon id (1..8) the cmd-window impulse resolved to for the
	 * emit tick this slot represents.  0 = no impulse in window.
	 * Read by the QWD impulse-anchored back-shift; MVD slots always
	 * carry 0 (no cmd data available). */
	int		impulse_target_weapon;
} qnn_backshift_slot_t;

typedef struct
{
	qnn_backshift_slot_t	slots[QNN_BACKSHIFT_K];
	int			head;	/* index of next write (oldest when full) */
	int			count;
	int			prev_weapon_id;
	qboolean		has_prev_weapon_id;
	/* STAT_ITEMS at the previous push.  Used by the MVD pickup gate
	 * to suppress the back-shift when the new weapon's IT_ bit
	 * flipped 0→1 on the same frame as the transition (deterministic
	 * server-driven switch via weapon_touch — not player intent). */
	int			prev_stat_items;
	qboolean		has_prev_stat_items;
} qnn_backshift_ring_t;

/* ── Collect runtime state ───────────────────────────────────────── */

typedef struct
{
	int		fixed_tick_hz;
	float		fixed_dt;
	/* The tick_hz value the controller asked for in hello.  0 means
	 * "emit at the demo's native recording rate" — resolved per-demo
	 * after QNN_DetectNativeTickHz by overriding fixed_tick_hz /
	 * fixed_dt with native_hz_detected.  Captured separately so the
	 * resolution happens on every demo, not just the first. */
	int		requested_tick_hz;
	/* Detected native recording rate of the demo currently being played.
	 * Set by QNN_DetectNativeTickHz post-signon.  Used to convert
	 * manifest's native-frame play_start / play_end into emit-frame
	 * indices at our fixed_tick_hz, so the play gate fires at the
	 * correct demo time regardless of the rate gap.  0 if unknown. */
	int		native_hz_detected;
	/* QW-only: cls.netchan.outgoing_sequence at the start of the current
	 * Host_Frame call, captured BEFORE Host_Frame runs.  At demo rates
	 * higher than fixed_tick_hz (typical: 77 Hz QWD vs 20 Hz emit), the
	 * demo player drains multiple dem_cmd messages per Host_Frame,
	 * depositing their usercmds into cl.frames[outgoing_sequence].
	 * QNN_ExtractActionFromUsercmd walks [cmd_seq_window_start ..
	 * outgoing_sequence) to OR-merge fire/jump, edge-detect weapon
	 * switches across the window — the cmd-pipeline analogue of what
	 * kbutton_t does for live input. */
	int		cmd_seq_window_start;
	/* cl.time captured BEFORE the current Host_Frame runs — the
	 * playback wall-clock at the start of the current emit window.
	 * Used by the per-event MVD fire back-shift: a sound's
	 * native_time = cl.mtime[0] at parse (QW demo-record demotime),
	 * so estimated press offset from this emit start is
	 * (native_time - emit_start_native) - ping_sec. */
	float		emit_start_native;
	int		tick;
	int		steps;
	qboolean	has_reset;
	qboolean	done;
	vec3_t		prev_origin;
	vec3_t		prev_velocity;
	int		prev_ammo;
	int		prev_weapon_id;
	qboolean	has_prev;
	vec3_t		emit_view_angles;
	vec3_t		emit_origin;
	vec3_t		emit_velocity;
	qboolean	emit_grounded;
	int		emit_waterlevel;
	int		emit_weapon_id;
	qboolean	has_emit_anchor;
	int		native_frame_count;
	/* NQ-only: physics groundedness seed for inference. */
	qboolean	phys_grounded;
	/* Mover tracking. */
	int		mover_entity_nums[QNN_MAX_PHYS_MOVERS];
	int		mover_model_indices[QNN_MAX_PHYS_MOVERS];
	int		mover_count;
	vec3_t		mover_emit_origins[QNN_MAX_PHYS_MOVERS];
	vec3_t		mover_origins[QNN_MAX_NATIVE_FRAMES][QNN_MAX_PHYS_MOVERS];
	/* Other player positions for body-block collision. */
	vec3_t		player_origins[QNN_MAX_NATIVE_FRAMES][QNN_MAX_PHYS_PLAYERS];
	int		player_entity_nums[QNN_MAX_PHYS_PLAYERS];
	int		player_count;
	/* trigger_push tracking. */
	int		push_model_indices[QNN_MAX_PUSH_TRIGGERS];
	vec3_t		push_velocities[QNN_MAX_PUSH_TRIGGERS];
	int		push_count;
	/* Previous candidate direction for continuity bias. */
	int		prev_fwd_sign;
	int		prev_strafe_sign;
	float		prev_move[3];
	FILE		*store_dump;
	qnn_tick_emit_state_t tick_emit;
	/* Frame filter counters. */
	int		dead_emit_count;
	int		frozen_emit_count;
	/* Grounded state from the previous native tick (set by QNN_SavePrev).
	 * Used by QNN_InferNativeAction_MVD to detect ground→air transitions
	 * at native-tick resolution (necessary for bunny-hop jump labeling). */
	qboolean	prev_grounded;
	/* Set true by inferred-demo paths whenever a weapon-fire event is seen
	 * at native-tick resolution within the current emit window.  Read by
	 * QNN_InferEmitAction* for the fire label and cleared by
	 * QNN_SaveEmitAnchor after each emission. */
	qboolean	native_fire_this_window;
	/* Actual reported velocity from the previous native tick.  Unlike
	 * prev_velocity (origin-delta estimate), this is snapshot->player_velocity
	 * stored directly so QNN_InferNativeAction_MVD can detect the +270 u/s
	 * jump impulse via vel_z delta without relying on the grounded flag. */
	vec3_t		prev_snap_velocity;
	/* QW-only: when true, action emission uses the MVD inference path
	 * (9-candidate physics for move, sound-event fire, etc.) even on
	 * QWD demos where usercmd_t is available.  Used to validate label
	 * drift against ground-truth recorded inputs. */
	qboolean	force_mvd_emit;
	/* QW-only: labeler-mode collect.  When true, the worker writes a
	 * slim LOBS stream (qnn_collect_helpers.c QNN_EmitLabelerTick)
	 * instead of the full QOBS frame; the snapshot path is also held
	 * to MVD-faithful semantics (cl_nopred=1, zero player_velocity)
	 * so the labeler sees the same obs distribution at training time
	 * as it would on a real MVD apply path.  When the host demo is
	 * a QWD with usercmd_t available, the move target is extracted
	 * from the recorded usercmd so the labeler trains against truth;
	 * the inference path is unused. */
	qboolean	labeler_mode;
	char		demo_path[MAX_OSPATH];
	/* MVD-path back-shift ring / fire/jump chain-fill state lives in
	 * the module-private struct in qnn_mvd_collect.c.  Labeler-mode
	 * cooldown counter lives in qnn_labeler_collect.c.  Both modules
	 * expose Reset entry points called from the QW main loop's per-
	 * demo init.  See qnn_mvd_collect.h / qnn_labeler_collect.h. */
} qnn_runtime_t;

extern qnn_runtime_t qnn_runtime;

/* ── Shared collect helpers ──────────────────────────────────────── */

/* Reconstruct prev_velocity from origin delta and stash prev_origin /
 * prev_ammo for the next tick. */
void QNN_SavePrev(const qnn_snapshot_t *snapshot, float dt);

/* Single-frame fire-event detection used by both NQ and QW MVD paths.
 * Returns true if THIS native frame contains a shot event:
 *   - a weapon-fire sound matching the currently-held weapon, OR
 *   - an ammo decrement on the same weapon as the previous frame.
 * The weapon-id guard on the ammo check prevents weapon-switch
 * cascades (snapshot->ammo jumps between weapons' ammo pools and
 * would otherwise read as a "decrement"). */
qboolean QNN_DetectFireEvent(const qnn_snapshot_t *snapshot);

/* Emission-rate filter: returns stdout to emit, NULL to drop.  May
 * mutate snapshot->action_label (dead-frame fire injection). */
FILE *QNN_EmitFilter(qnn_snapshot_t *snapshot);

/* JSON op-key matcher: 1 if line contains `"op":"<value>"` (with or
 * without space after the colon), else 0. */
int QNN_OpIs(const char *line, const char *value);

/* Fill action->look (view-relative turn delta) and action->weapon
 * (per-frame held-weapon state) from the current snapshot.  Shared by
 * the QWD usercmd-truth path and the MVD inference path. */
void QNN_FillLookAndSwitch(qnn_action_t *action,
	const qnn_snapshot_t *snapshot);

#endif /* QNN_COLLECT_HELPERS_H */
