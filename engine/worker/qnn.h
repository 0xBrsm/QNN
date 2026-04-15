#ifndef QNN_H
#define QNN_H

#include <stdio.h>
#include <string.h>

#include "qnn_navmesh.h"
#include "quakedef.h"

#ifndef QNN_ROUTE_RUNTIME_FWD
#define QNN_ROUTE_RUNTIME_FWD
typedef struct qnn_route_runtime_s qnn_route_runtime_t;
#endif

#define QNN_ACTION_HISTORY 2
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
	int	switch_slot;
	int	recall[4];
} qnn_action_t;

typedef struct
{
	int	entity_key;
	char	entity_id[32];
	int	entity_num;
	char	classname[QNN_MAX_CLASSNAME];
	char	model_name[QNN_MAX_MODEL_NAME];
	int	region_id;
	vec3_t	origin;
	vec3_t	velocity;
	vec3_t	angles;
	int	model_id;
	int	frame;
	int	effects;
	int	skin;
	int	health;
	int	frags;
	float	half_extents[3];
	qboolean static_proxy;
} qnn_known_entity_t;

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
	qnn_known_entity_t known[QNN_MAX_VISIBLE];
	int	known_count;
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

/* ── Tick resampling gate ─────────────────────────────────────────
 * Accumulates engine frames and emits at a fixed target Hz.
 * Call QNN_ResampleInit() once, then QNN_ResampleShouldEmit()
 * every frame.  When it returns true, emit the token tick.
 * Action labels are merged across the window: fire uses OR,
 * move is a continuous wishdir inferred per emission, and
 * look/switch remain emission-to-emission labels computed by the
 * collector.
 */
typedef struct
{
	int	target_hz;		/* 0 = disabled (emit every frame) */
	float	target_dt;		/* 1.0 / target_hz */
	float	accumulated_dt;		/* time since last emission */
	int	fire_any;		/* OR accumulator for fire */
} qnn_resample_state_t;

extern qnn_resample_state_t qnn_resample;

void QNN_ResampleInit(int target_hz);
void QNN_ResampleAccumulate(const qnn_action_t *action, float frame_dt);
qboolean QNN_ResampleShouldEmit(void);
void QNN_ResampleApplyActionMerge(qnn_action_t *action);
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
int QNN_SwitchSlotFromWeaponId(int weapon_id);
int QNN_SwitchImpulseFromSlot(int switch_slot, int weapons_owned);
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
void QNN_PhysPushMovers(float dt);
void QNN_PhysSetupPlayers(const vec3_t *origins, int count);
void QNN_PhysBestCandidate(
	const vec3_t vel, const vec3_t origin,
	const vec3_t view_angles, qboolean grounded, int waterlevel,
	float dt, const vec3_t observed,
	int prev_forward, int prev_strafe,
	int *out_forward, int *out_strafe,
	qboolean *out_unreachable);

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
	case 1: case 2: return 4;   /* SG, SSG → SHOTGUN */
	case 3: case 4: return 5;   /* NG, SNG → NAILGUN */
	case 5: return 6;           /* GL → GRENADE_LAUNCHER */
	case 6: return 7;           /* RL → ROCKET_LAUNCHER */
	case 7: return 8;           /* LG → THUNDERBOLT */
	default: return 0;          /* Axe → NONE */
	}
}

#endif
