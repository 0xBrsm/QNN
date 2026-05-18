#ifndef QNN_H
#define QNN_H

#include <stdio.h>
#include <string.h>

#include "qnn_navmesh.h"
#include "qnn_vocab.h"
#include "quakedef.h"

#ifndef QNN_ROUTE_RUNTIME_FWD
#define QNN_ROUTE_RUNTIME_FWD
typedef struct qnn_route_runtime_s qnn_route_runtime_t;
#endif

#define QNN_OBS_BUFFER_SIZE 4096
#define QNN_MAX_PROPERTY_KEY 64
#define QNN_MAX_PROPERTY_VALUE 256
#define QNN_MAX_CLASSNAME 64
#define QNN_MAX_CATEGORY 16
#define QNN_MAX_OBJECT_ID 32
#define QNN_MAX_MAP_ID 64
#define QNN_MAX_MODEL_NAME 64
#define QNN_MAX_STATIC_PROPERTIES 16
#define QNN_MAX_SOUNDS 128
#define QNN_MAX_SOUND_NAME 64
#define QNN_MAX_VISIBLE 64
#define QNN_MAX_EVENTS 16
#define QNN_MAX_DYNAMIC_OBJECTS 128
#define QNN_MAX_TOKEN_OBJECTS 16
#define QNN_MAX_EVENT_ATOMS 256
#define QNN_SPATIAL_TOKEN_COUNT 9
#define QNN_MAX_TRAIN_DAMAGE 64
#define QNN_MAX_TRAIN_ITEMS 64
#define QNN_MAX_TRAIN_DEATHS 16
#define QNN_MAX_TRAIN_SPAWNS 16

typedef struct
{
	vec3_t	origin;
	float	volume;
	float	attenuation;
	int	entity_num;
	char	name[QNN_MAX_SOUND_NAME];
	/* cl.mtime[0] captured at S_StartSound — the native-frame
	 * (~13 ms at QW 77 Hz server tick) timestamp of when the sound
	 * multicast was sent by the server (svc_time embedded in the
	 * containing packet).  Used by the MVD per-event fire back-shift
	 * to compute press_phase within the current emit window and
	 * route the fire label to the right ring slot. */
	float	native_time;
} qnn_sound_event_t;

typedef struct
{
	char	key[QNN_MAX_PROPERTY_KEY];
	char	value[QNN_MAX_PROPERTY_VALUE];
} qnn_property_t;

typedef struct
{
	char	object_id[QNN_MAX_OBJECT_ID];
	char	category[QNN_MAX_CATEGORY];
	char	classname[QNN_MAX_CLASSNAME];
	int	entity_num;	/* BSP parse order — stable edict number */
	int	region_id;
	vec3_t	origin;
	vec3_t	angles;
	qnn_property_t *properties;
	int	property_count;
} qnn_static_object_t;

typedef struct
{
	int	region_id;
	vec3_t	center;
	vec3_t	bounds_min;
	vec3_t	bounds_max;
} qnn_region_t;

typedef struct
{
	char	requested_map_id[QNN_MAX_MAP_ID];
	char	map_name[QNN_MAX_MAP_ID];
	char	source[32];
	char	navmesh_status[16];
	char	navmesh_error[256];
	char	route_status[16];
	char	route_error_msg[256];
	qnn_navmesh_build_config_t navmesh_config;
	qnn_navmesh_summary_t navmesh_summary;
	qnn_navmesh_runtime_t *navmesh;
	qnn_route_runtime_t *route;
	int	route_area_count;
	int	route_cluster_count;
	int	route_min_cluster_area_count;
	int	route_max_cluster_area_count;
	float	route_avg_cluster_area_count;
	int	route_walk_link_count;
	int	route_teleport_link_count;
	int	route_lift_link_count;
	int	route_push_link_count;
	int	route_drop_link_count;
	qnn_region_t *regions;
	int	region_count;
	qnn_static_object_t *static_objects;
	int	static_object_count;
	void	*cached_worldmodel; /* track cl.worldmodel to detect map reload */
} qnn_map_state_t;

typedef struct
{
	float	move[3];	/* view-relative direction: (forward, right, up) */
	float	look[3];
	int	fire;
	int	weapon;		/* raw engine weapon byte: 0 = no switch this
				 * frame, 1..8 = Quake weapon id (axe..lightning)
				 * consumed as an impulse by the runtime engine. */
} qnn_action_t;

typedef struct
{
	char	event_type[32];
	int	region_id;
	int	has_delta;
	int	delta;
	int	has_weapon_id;
	int	weapon_id;
	int	source_entity_num;
	int	target_entity_num;
} qnn_event_t;

typedef struct
{
	vec3_t	player_origin;
	vec3_t	player_velocity;
	vec3_t	player_view_angles;
	int	health;
	int	armor;
	float	armor_type;
	int	ammo;
	int	ammo_shells;
	int	ammo_nails;
	int	ammo_rockets;
	int	ammo_cells;
	int	weapons_owned;
	int	items_owned;
	int	weapon_id;
	int	waterlevel;
	qboolean grounded;
	int	current_region_id;
	qboolean done;
	qnn_event_t events[QNN_MAX_EVENTS];
	int	event_count;
	int	damage_dealt;
	int	hit_count;
	int	shots_fired;
	int	damage_weapon_id;
	qnn_sound_event_t sounds[QNN_MAX_SOUNDS];
	int	sound_count;
	qnn_action_t action_label;
} qnn_snapshot_t;

extern qnn_action_t qnn_pending_action;

/* Binary step protocol: 1 opcode byte + action struct.
 * Replaces JSON for the hot step path.  Hello/reset/shutdown stay JSON. */
#define QNN_BINARY_OP_STEP 0x01
#define QNN_BINARY_ACTION_SIZE ((int)sizeof(qnn_action_t))

extern qnn_sound_event_t qnn_sound_buffer[QNN_MAX_SOUNDS];
extern int qnn_sound_count;

/* Shared globals (defined in qnn_sys.c) */
extern qnn_map_state_t qnn_map_state;
extern char qnn_basedir_storage[MAX_OSPATH];
extern char *basedir;
extern char *cachedir;

/* Utilities (qnn_sys.c) */
int QNN_JsonExtractInt(const char *line, const char *key, int fallback);
float QNN_JsonExtractFloat(const char *line, const char *key, float fallback);
qboolean QNN_JsonExtractBool(const char *line, const char *key, qboolean fallback);
qboolean QNN_JsonExtractString(const char *line, const char *key, char *out, size_t out_size);
qboolean QNN_JsonExtractVec2(const char *line, const char *key, float out[2]);
qboolean QNN_JsonExtractVec3(const char *line, const char *key, vec3_t out);
float QNN_LookAxisFromMouseCount(int mouse_count);
int QNN_MouseCountFromLookAxis(float axis);
void QNN_WriteJsonString(FILE *out, const char *text);
void QNN_WriteError(const char *message);
int QNN_HandleNavQuery(const char *line);

/* Map lifecycle (qnn_io.c) */
qboolean QNN_PrepareMap(const char *requested_map_id, char *error, size_t error_size);
qboolean QNN_BuildMapState(qnn_map_state_t *out, const char *requested_map_id, const char *map_name, char *error, size_t error_size);
void QNN_FreeMapState(qnn_map_state_t *map_state);
static inline void QNN_ClearAction(qnn_action_t *action)
{
	memset(action, 0, sizeof(*action));
}

/* Standard Quake player hull dimensions (from QuakeC VEC_HULL_MIN/MAX). */
#define QNN_PLAYER_MINS_X (-16.0f)
#define QNN_PLAYER_MINS_Y (-16.0f)
#define QNN_PLAYER_MINS_Z (-24.0f)
#define QNN_PLAYER_MAXS_X  (16.0f)
#define QNN_PLAYER_MAXS_Y  (16.0f)
#define QNN_PLAYER_MAXS_Z  (32.0f)

/* Continuous wishdir label inference — projects velocity delta onto
 * the view-relative frame to produce a 3D (forward, right, up)
 * movement direction vector.  No BSP simulation needed. */

/* Entity classification helpers (qnn_entity.c) */
int QNN_CategoryOrder(const char *category);
const char *QNN_Classify(const char *classname);

/* Basedir resolution (qnn_sys.c) */
void QNN_ResolveBasedir(char *out, size_t out_size);

/* Route build orchestration (qnn_route.cpp) */
qboolean QNN_RouteBuildFromWorldmodel(qnn_map_state_t *out, char *error, size_t error_size);

/* Object assembly (qnn_object.c) */
const char *QNN_ProgString(string_t value);
int QNN_WeaponId(void);
int QNN_CurrentFrags(void);
void QNN_CaptureBaseSnapshot(qnn_snapshot_t *snapshot);
void QNN_DrainSounds(qnn_snapshot_t *snapshot);
qboolean QNN_SnapshotHasSelfWeaponFireSound(const qnn_snapshot_t *snapshot);
qboolean QNN_SnapshotHasSelfJumpSound(const qnn_snapshot_t *snapshot);
/* Per-sound check used by the per-event MVD fire back-shift.  See
 * qnn_event.c — same rules as QNN_SnapshotHasSelfWeaponFireSound
 * but on one sound so the caller can route per native_time. */
qboolean QNN_IsSelfWeaponFireSound(const qnn_sound_event_t *sound);
qboolean QNN_IsSelfJumpSound(const qnn_sound_event_t *sound);

/* IO (qnn_io.c) — see qnn_io.h for full typed token API */

/* Physics sim (qnn_phys.c) — 9-candidate forward search for move labels. */
#define QNN_MAX_PHYS_MOVERS 8
#define QNN_MAX_PHYS_PLAYERS 16

typedef struct {
	int	model_index;		/* sv.models[] / cl.model_precache[] index */
	vec3_t	origin;			/* world position at start of step */
	vec3_t	velocity;		/* demo-observed velocity for SV_PushMove */
} qnn_mover_state_t;

void QNN_PhysInit(void);
void QNN_PhysSetupMovers(const qnn_mover_state_t *movers, int count);
void QNN_PhysSetupPlayers(const vec3_t *origins, int count);
void QNN_PhysBestCandidate(
	const vec3_t vel, const vec3_t origin,
	const vec3_t view_angles, qboolean grounded, int waterlevel,
	float dt, const vec3_t observed,
	int prev_forward, int prev_strafe,
	int *out_forward, int *out_strafe,
	qboolean *out_unreachable);

/* Collect helpers (qnn_collect_helpers.c) — engine-agnostic scanners
 * and post-processing shared by NQ and QW collect main loops. */
int QNN_BuildMoverRefs(int *out_entity_nums, int *out_model_indices,
	int max_count);
int QNN_BuildPushRefs(int *out_model_indices, vec3_t *out_velocities,
	int max_count);
int QNN_BuildPlayerRefs(int self_entity_num, int *out_entity_nums,
	int max_count);
void QNN_JitterFilter(qnn_action_t *mid, const float *prev_move,
	const float *next_move);
qboolean QNN_ActionIsFrozen(const qnn_action_t *a);
void QNN_EmitTick(FILE *out, const uint8_t *obs, const qnn_action_t *action,
	int tick, int steps, int tick_hz, uint16_t flags);

/* LOBS (Labeler OBS): slim per-native-tick stream for labeler training /
 * apply.  32-byte fixed-size frame, no obs buffer overhead.  Used only
 * when qnn_runtime.labeler_mode is set; the worker writes LOBS instead
 * of QOBS so the labeler reader is decoupled from the BC obs schema.
 *
 * Wire layout (little-endian, all fields contiguous):
 *   "LOBS"            4 bytes magic
 *   tick              u32        4
 *   tick_hz           u32        4
 *   flags             u16        2  (reserved, currently 0)
 *   pos_delta_vel[3]  fp16       6  (body-frame, normalized by QNN_VELOCITY_SCALE)
 *   movement_id       u8         1
 *   view_delta[3]     fp16       6  (per-tick cur_forward dot anchor_basis)
 *   c_rule_fire       u8         1  (engine-side sound+ammo fire detection)
 *   c_rule_jump       u8         1
 *   move_packed       u8         1  (fb | lr<<2 | ud<<4, target)
 *   target_valid_mask u8         1  (per-axis engine-effective bits;
 *                                    bit0=fb, bit1=lr, bit2=ud,
 *                                    bit3=fire, bit4=weapon (held);
 *                                    bits5..7 reserved.  See
 *                                    src/demo/sanitize.py for the
 *                                    rule definitions this mirrors.)
 *   usercmd_fire      u8         1  (true press signal: cmd-window-OR'd
 *                                    action_label.fire on QWD,
 *                                    MVD-inferred fire on real MVD)
 *   weapon_id         u8         1  (held-weapon byte 1..8 = axe..LG, 0
 *                                    if no weapon held — observable
 *                                    every alive frame from server-side
 *                                    STAT_ACTIVEWEAPON)
 *                              ---
 *                              33 bytes/tick
 */
void QNN_EmitLabelerTick(FILE *out,
	int tick, int tick_hz, uint16_t flags,
	const float pos_delta_vel[3],     /* body-frame, pre-normalized */
	int movement_id,
	const float view_delta[3],
	int c_rule_fire,
	int c_rule_jump,
	uint8_t move_packed,
	uint8_t target_valid_mask,
	uint8_t usercmd_fire,
	uint8_t weapon_id);

/* Shared tick-emission state for collect workers (NQ + QW).
 * Handles the two-level buffer (obs delay + 3-frame jitter filter on
 * move).  `has_prev_emitted` is a sentinel set on the first successful
 * emit so the demo-end fallback can decide whether to inject a done-
 * tick when the play gate skipped every frame. */
typedef struct
{
	int		source_tick;
	int		dest_tick;
	int		sound_index;
	int		weapon_id;
	float		native_time;
	float		emit_start_native;
	float		ping_sec;
	float		phase;
	float		press_offset;
	int		deterministic_offset;
	int		route_offset;
} qnn_fire_route_event_t;

#define QNN_MAX_FIRE_ROUTE_EVENTS 32

typedef struct
{
	uint8_t		buffered_obs[QNN_OBS_BUFFER_SIZE];
	qboolean	has_buffered_obs;
	uint8_t		jitter_obs[QNN_OBS_BUFFER_SIZE];
	qnn_action_t	jitter_action;
	int		jitter_tick;
	int		jitter_steps;
	int		jitter_tick_hz;
	uint16_t	jitter_flags;
	FILE		*jitter_out;
	qboolean	has_jitter_buf;
	qnn_fire_route_event_t jitter_fire_routes[QNN_MAX_FIRE_ROUTE_EVENTS];
	int		jitter_fire_route_count;
	float		prev_prev_move[3];
	qboolean	has_prev_prev_move;
	qboolean	has_prev_emitted;
	int		emitted_rows;
} qnn_tick_emit_state_t;

void QNN_TickEmitReset(qnn_tick_emit_state_t *st);
void QNN_WriteObsTick(qnn_tick_emit_state_t *st, FILE *out,
	const qnn_snapshot_t *snapshot, int tick, int steps, int tick_hz,
	qboolean reset_flag);

/* Pack obs bytes from a snapshot without writing — exposes the
 * QNN_IOEmit + QNN_IOPackObsBuffer combo used internally by
 * QNN_WriteObsTick.  Callers that need to defer the emit (MVD
 * back-shift ring buffer) pack at capture time and store bytes. */
void QNN_PackSnapshotObs(const qnn_snapshot_t *snapshot, uint8_t *obs_out);

/* Push a pre-packed obs+action through the obs/jitter pipeline.
 * Same buffer semantics as QNN_WriteObsTick, but accepts already-
 * packed obs bytes (so the caller can defer the call after capture
 * without losing access to global state read by QNN_IOEmit). */
void QNN_WriteObsTickPrepacked(qnn_tick_emit_state_t *st, FILE *out,
	const uint8_t *obs_bytes, const qnn_action_t *action,
	qboolean done, int tick, int steps, int tick_hz,
	qboolean reset_flag);

void QNN_WriteObsTickPrepackedWithFireRoutes(qnn_tick_emit_state_t *st,
	FILE *out, const uint8_t *obs_bytes, const qnn_action_t *action,
	qboolean done, int tick, int steps, int tick_hz,
	qboolean reset_flag, const qnn_fire_route_event_t *routes,
	int route_count);

void QNN_FlushTickEmit(qnn_tick_emit_state_t *st);

/* Tolerance on |look - identity| below which a look label counts as
 * "no turn" for the frozen-action predicate. */
#define QNN_FROZEN_LOOK_TOL     0.025f

/* Reward (qnn_reward.c) */
void QNN_TrainingResetEpisode(void);
void QNN_TrainingResetTick(void);
void QNN_TrainingParseRewardWeights(const char *line);
void QNN_WriteTrainingExtrasBinary(FILE *out, const qnn_snapshot_t *snapshot, int tick, int steps, qboolean reset_flag);
/* ── Engine physics constants (shared by collector, physics, inference) ── */

#define QNN_SV_MAXSPEED      320.0f
#define QNN_SV_ACCELERATE     10.0f
#define QNN_SV_GRAVITY       800.0f
#define QNN_SV_JUMP_SPEED    270.0f
#define QNN_SV_SWIM_SPEED    100.0f  /* water; slime=80, lava=50 */

/* ── Move snap — used by collector for keyboard demo label generation ── */

#define QNN_MEDIUM_GROUND 0
#define QNN_MEDIUM_AIR    1
#define QNN_MEDIUM_WATER  2

#define QNN_SNAP_THRESHOLD 0.1f

/* Snap a raw move vector (view-relative, scaled by maxspeed) to the
 * nearest legal key combination.
 * Ground: 9 XY directions from candidate search, Z from jump sound.
 * Air: strafe only (L/R/none).  Forward is zeroed because SV_AirAccelerate
 *      has no effect parallel to velocity at cruise speed (addspeed <= 0).
 *      The effective key is always pure strafe in standard strafejumping.
 * Water: 27 directions (XY + Z all keyboard-binary).
 *
 * in[3]:  raw input (e.g. candidate signs or vel / maxspeed)
 * medium: QNN_MEDIUM_GROUND, _AIR, or _WATER
 * has_jump_sound: true if jump sound detected this frame
 * out[3]: snapped result */
static inline void QNN_SnapMove(const float *in, int medium,
	qboolean has_jump_sound, float *out)
{
	float sx, sy;

	/* Air: only the perpendicular-to-velocity axis is resolvable.
	 * SV_AirAccelerate caps wishspeed at 30 — if the player is already
	 * moving >= 30 ups along the wish direction, addspeed <= 0 and
	 * nothing happens.  So keys parallel to velocity have zero effect;
	 * only the perpendicular component produces a measurable signal.
	 * Project the candidate result onto the perp axis and zero the
	 * parallel component. */
	if (medium == QNN_MEDIUM_AIR)
	{
		/* Air: only strafe has consistent effect.  SV_AirAccelerate
		 * has zero effect parallel to velocity (addspeed <= 0 at
		 * cruise).  The perpendicular axis IS the strafe axis when
		 * the player looks roughly where they're moving (standard
		 * strafejump).  Forward component is unresolvable — zero it
		 * and keep only the strafe sign from the candidate search. */
		out[0] = 0.0f;
		if (in[1] > QNN_SNAP_THRESHOLD) out[1] = 1.0f;
		else if (in[1] < -QNN_SNAP_THRESHOLD) out[1] = -1.0f;
		else out[1] = 0.0f;

		out[2] = 0.0f;
		if (has_jump_sound && in[2] > QNN_SNAP_THRESHOLD)
			out[2] = QNN_SV_JUMP_SPEED / QNN_SV_MAXSPEED;
		return;
	}

	/* XY: per-axis snap to {-1, 0, +1}. */
	if (in[0] > QNN_SNAP_THRESHOLD) sx = 1.0f;
	else if (in[0] < -QNN_SNAP_THRESHOLD) sx = -1.0f;
	else sx = 0.0f;

	if (in[1] > QNN_SNAP_THRESHOLD) sy = 1.0f;
	else if (in[1] < -QNN_SNAP_THRESHOLD) sy = -1.0f;
	else sy = 0.0f;

	/* Diagonal normalization: (1,1) → (0.707, 0.707). */
	if (sx != 0.0f && sy != 0.0f)
	{
		sx *= 0.70710678f;
		sy *= 0.70710678f;
	}
	out[0] = sx;
	out[1] = sy;

	/* Z: context-dependent. */
	if (medium == QNN_MEDIUM_WATER)
	{
		/* Water: snap Z to ±swim_speed or zero (27 directions). */
		if (in[2] > QNN_SNAP_THRESHOLD)
			out[2] = QNN_SV_SWIM_SPEED / QNN_SV_MAXSPEED;
		else if (in[2] < -QNN_SNAP_THRESHOLD)
			out[2] = -QNN_SV_SWIM_SPEED / QNN_SV_MAXSPEED;
		else
			out[2] = 0.0f;
	}
	else
	{
		/* Ground/air: Z only from jump sound detection. */
		out[2] = 0.0f;
		if (has_jump_sound && in[2] > QNN_SNAP_THRESHOLD)
			out[2] = QNN_SV_JUMP_SPEED / QNN_SV_MAXSPEED;
	}
}

/* ── Common math macros ──────────────────────────────────────────── */

#define QNN_Clamp(v, lo, hi) ((v) < (lo) ? (lo) : (v) > (hi) ? (hi) : (v))
#define QNN_Normalize(v, scale) (QNN_Clamp((v) / (scale), -1.0f, 1.0f))
#define QNN_AngleSinDeg(d) (sinf((d) * ((float)M_PI / 180.0f)))
#define QNN_AngleCosDeg(d) (cosf((d) * ((float)M_PI / 180.0f)))
#define QNN_DistSq(a, b) (((a)[0]-(b)[0])*((a)[0]-(b)[0]) + ((a)[1]-(b)[1])*((a)[1]-(b)[1]) + ((a)[2]-(b)[2])*((a)[2]-(b)[2]))
#define QNN_VecLength(v) ((float)sqrt((double)((v)[0]*(v)[0] + (v)[1]*(v)[1] + (v)[2]*(v)[2])))

/* Build a forward direction vector from Quake view angles (pitch, yaw). */
static inline void QNN_ForwardFromAngles(const vec3_t angles, vec3_t out)
{
	float cp = cosf(angles[0] * ((float)M_PI / 180.0f));
	float sp = sinf(angles[0] * ((float)M_PI / 180.0f));
	float cy = cosf(angles[1] * ((float)M_PI / 180.0f));
	float sy = sinf(angles[1] * ((float)M_PI / 180.0f));
	out[0] = cp * cy;
	out[1] = cp * sy;
	out[2] = -sp;
}

/* Transform a world-space vector into view-relative (forward/right/up). */
static inline void QNN_RelativeFrame(const vec3_t view_angles, const vec3_t world_delta, vec3_t out)
{
	vec3_t forward, right, up, angles_copy;
	VectorCopy(view_angles, angles_copy);
	AngleVectors(angles_copy, forward, right, up);
	out[0] = DotProduct(world_delta, forward);
	out[1] = DotProduct(world_delta, right);
	out[2] = DotProduct(world_delta, up);
}

/* ── Binary write helpers (little-endian) ────────────────────────── */

void QNN_WriteU16LE(FILE *out, uint16_t value);
void QNN_WriteU32LE(FILE *out, uint32_t value);
void QNN_WriteI16LE(FILE *out, int value);
void QNN_WriteI32LE(FILE *out, int32_t value);
void QNN_WriteF32LE(FILE *out, float value);

/* ── Weapon class mapping ────────────────────────────────────────── */
/* Maps Quake weapon_id (1-8) to subject_embed vocabulary.
   Values must match QNN_SUBJECT_* in qnn_vocab.h. */

static inline int qnn_weapon_subject_from_id(int weapon_id)
{
	switch (weapon_id)
	{
	case 1: return QNN_SUBJECT_AXE;
	case 2: return QNN_SUBJECT_SHOTGUN;
	case 3: return QNN_SUBJECT_SUPER_SHOTGUN;
	case 4: return QNN_SUBJECT_NAILGUN;
	case 5: return QNN_SUBJECT_SUPER_NAILGUN;
	case 6: return QNN_SUBJECT_GRENADE_LAUNCHER;
	case 7: return QNN_SUBJECT_ROCKET_LAUNCHER;
	case 8: return QNN_SUBJECT_THUNDERBOLT;
	default: return QNN_SUBJECT_NONE;
	}
}

#endif
