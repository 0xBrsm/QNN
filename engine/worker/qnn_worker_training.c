#include "qnn_worker.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#define QNN_TRAIN_BINARY_MAGIC "QTRN"
#define QNN_TRAIN_BINARY_VERSION 2

#define QNN_TRAIN_FLAG_RESET 0x0001
#define QNN_TRAIN_FLAG_DONE  0x0002

#define QNN_TRAIN_EVENT_PICKUP  1
#define QNN_TRAIN_EVENT_RESPAWN 2

#define QNN_TRAIN_PICKUP_HEALTH 1
#define QNN_TRAIN_PICKUP_ARMOR  2
#define QNN_TRAIN_PICKUP_AMMO   3
#define QNN_TRAIN_PICKUP_WEAPON 4
#define QNN_TRAIN_PICKUP_ITEM   5
#define QNN_TRAIN_PICKUP_PACK   6

#define QNN_TRAIN_DAMAGE_FLAG_SELF    0x0001
#define QNN_TRAIN_DAMAGE_FLAG_WORLD  0x0002
#define QNN_TRAIN_DAMAGE_FLAG_SPLASH 0x0004

#define QNN_TRAIN_DEATH_FLAG_GIB 0x0001

#define IT_AXE              4096.0f
#define IT_SHOTGUN          1.0f
#define IT_SUPER_SHOTGUN    2.0f
#define IT_NAILGUN          4.0f
#define IT_SUPER_NAILGUN    8.0f
#define IT_GRENADE_LAUNCHER 16.0f
#define IT_ROCKET_LAUNCHER  32.0f
#define IT_LIGHTNING        64.0f

typedef struct
{
	int attacker_entity_num;
	int target_entity_num;
	int weapon_id;
	int flags;
	float health_before;
	float health_after;
	float armor_before;
	float armor_after;
	float armor_type_before;
	float armor_type_after;
	float damage_health;
	float damage_armor;
} qnn_training_damage_record_t;

typedef struct
{
	int actor_entity_num;
	int item_entity_num;
	int event_kind;
	int category;
	int weapon_id;
	int flags;
	float amount;
	vec3_t origin;
} qnn_training_item_record_t;

typedef struct
{
	int victim_entity_num;
	int attacker_entity_num;
	int weapon_id;
	int flags;
} qnn_training_death_record_t;

typedef struct
{
	int player_entity_num;
	int flags;
	vec3_t origin;
} qnn_training_spawn_record_t;

typedef struct
{
	qnn_training_damage_record_t damage[QNN_WORKER_MAX_TRAIN_DAMAGE];
	int damage_count;
	qnn_training_item_record_t items[QNN_WORKER_MAX_TRAIN_ITEMS];
	int item_count;
	qnn_training_death_record_t deaths[QNN_WORKER_MAX_TRAIN_DEATHS];
	int death_count;
	qnn_training_spawn_record_t spawns[QNN_WORKER_MAX_TRAIN_SPAWNS];
	int spawn_count;
	int shot_counts[MAX_EDICTS];
	int last_shot_weapon_id[MAX_EDICTS];
	float total_damage_dealt[MAX_EDICTS];
	int total_hit_count[MAX_EDICTS];
	int total_shots_fired[MAX_EDICTS];
	qboolean prev_alive[MAX_EDICTS];
	qboolean prev_self_valid;
	int prev_self_entity_num;
	float prev_health;
	float prev_armor;
	float prev_armor_type;
	int prev_frags;
	/* Transient: computed during write, used for v2 extension fields */
	float _computed_reward;
	float _tracking_cos;
	float _damage_dealt_self;
} qnn_training_state_t;

static qnn_training_state_t qnn_training_state;

/* ── Reward computation (moved from Python for performance) ────── */

typedef struct
{
	float	frag_bonus;
	float	death_penalty;
	float	ehp_delta_weight;
	float	edp_delta_weight;
	float	fire_penalty;
	float	self_damage_penalty;
	float	tracking_weight;
	float	tracking_fov;		/* full FOV in degrees */
	qboolean tracking_penalty;	/* if false, clamp tracking to non-negative */
	qboolean valid;			/* true once weights have been received */
} qnn_reward_weights_t;

typedef struct
{
	float	prev_ehp;		/* persists across ticks, reset to 100 on death */
} qnn_reward_state_t;

static qnn_reward_weights_t qnn_reward_weights;
static qnn_reward_state_t qnn_reward_state;

/* Forward declaration — defined later in this file */
static float qnn_train_effective_hp(float health, float armor, float armor_type);

static void qnn_require_json_key(const char *line, const char *key)
{
	if (strstr(line, key) == NULL)
		Sys_Error("reward_weights is missing required key %s", key);
}

void qnn_worker_training_parse_reward_weights(const char *line)
{
	/* Only parse if reward_weights block is present in reset options.
	   When absent, valid stays false and reward computation is skipped. */
	if (strstr(line, "\"reward_weights\"") == NULL)
	{
		qnn_reward_weights.valid = false;
		return;
	}
	/* All keys are required when reward_weights is provided. */
	qnn_require_json_key(line, "\"frag_bonus\"");
	qnn_require_json_key(line, "\"death_penalty\"");
	qnn_require_json_key(line, "\"ehp_delta_weight\"");
	qnn_require_json_key(line, "\"edp_delta_weight\"");
	qnn_require_json_key(line, "\"fire_penalty\"");
	qnn_require_json_key(line, "\"self_damage_penalty\"");
	qnn_require_json_key(line, "\"tracking_weight\"");
	qnn_require_json_key(line, "\"tracking_fov\"");
	qnn_require_json_key(line, "\"tracking_penalty\"");
	qnn_reward_weights.frag_bonus = qnn_json_extract_float(line, "\"frag_bonus\"", 0.0f);
	qnn_reward_weights.death_penalty = qnn_json_extract_float(line, "\"death_penalty\"", 0.0f);
	qnn_reward_weights.ehp_delta_weight = qnn_json_extract_float(line, "\"ehp_delta_weight\"", 0.0f);
	qnn_reward_weights.edp_delta_weight = qnn_json_extract_float(line, "\"edp_delta_weight\"", 0.0f);
	qnn_reward_weights.fire_penalty = qnn_json_extract_float(line, "\"fire_penalty\"", 0.0f);
	qnn_reward_weights.self_damage_penalty = qnn_json_extract_float(line, "\"self_damage_penalty\"", 0.0f);
	qnn_reward_weights.tracking_weight = qnn_json_extract_float(line, "\"tracking_weight\"", 0.0f);
	qnn_reward_weights.tracking_fov = qnn_json_extract_float(line, "\"tracking_fov\"", 360.0f);
	qnn_reward_weights.tracking_penalty = qnn_json_extract_bool(line, "\"tracking_penalty\"", true);
	qnn_reward_weights.valid = true;
}

static float qnn_train_remap_tracking(float cos_val, float fov_deg)
{
	float half_rad, boundary;

	if (fov_deg >= 360.0f)
		return cos_val;
	half_rad = fov_deg * 0.5f * ((float)M_PI / 180.0f);
	boundary = cosf(half_rad);
	if (cos_val >= boundary)
		return (cos_val - boundary) / fmaxf(1.0f - boundary, 1e-8f);
	else
		return -(boundary - cos_val) / fmaxf(boundary + 1.0f, 1e-8f);
}

static float qnn_train_tracking_cosine(const qnn_worker_snapshot_t *snapshot)
{
	float best_cos = 0.0f;
	float best_dist_sq = 1e30f;
	int i;
	float yaw_rad, pitch_rad;
	float cy, sy, cp, sp;
	vec3_t forward;

	/* Build forward vector from view angles */
	yaw_rad = snapshot->player_view_angles[1] * ((float)M_PI / 180.0f);
	pitch_rad = snapshot->player_view_angles[0] * ((float)M_PI / 180.0f);
	cy = cosf(yaw_rad);
	sy = sinf(yaw_rad);
	cp = cosf(pitch_rad);
	sp = sinf(pitch_rad);
	forward[0] = cp * cy;
	forward[1] = cp * sy;
	forward[2] = -sp;

	for (i = 0; i < snapshot->visible_count; ++i)
	{
		const qnn_worker_visible_entity_t *ent = &snapshot->visible[i];
		float dx, dy, dz, dist_sq, inv_dist, cos_val;

		if (ent->entity_num == cl.viewentity || ent->entity_num <= 0)
			continue;
		if (ent->health <= 0)
			continue;

		dx = ent->origin[0] - snapshot->player_origin[0];
		dy = ent->origin[1] - snapshot->player_origin[1];
		dz = ent->origin[2] - snapshot->player_origin[2];

		dist_sq = dx * dx + dy * dy + dz * dz;
		if (dist_sq < 1e-8f)
			continue;

		inv_dist = 1.0f / sqrtf(dist_sq);
		cos_val = (forward[0] * dx + forward[1] * dy + forward[2] * dz) * inv_dist;

		if (dist_sq < best_dist_sq)
		{
			best_dist_sq = dist_sq;
			best_cos = cos_val;
		}
	}
	return best_cos;
}

static float qnn_train_compute_reward(
	const qnn_worker_snapshot_t *snapshot,
	float damage_dealt_self,
	float edp_raw,
	int frag_gain,
	int player_died,
	int shots_fired,
	float health_after,
	float armor_after,
	float armor_type_after,
	float *out_tracking_cos,
	float *out_damage_self)
{
	float reward = 0.0f;
	float ehp_now, tracking_cos, remapped;

	*out_tracking_cos = 0.0f;
	*out_damage_self = damage_dealt_self;

	if (!qnn_reward_weights.valid)
		return 0.0f;

	/* Frag bonus */
	reward += qnn_reward_weights.frag_bonus * (float)(frag_gain > 0 ? frag_gain : 0);

	/* Death penalty */
	reward += qnn_reward_weights.death_penalty * (player_died ? 1.0f : 0.0f);

	/* EHP delta (log-ratio) */
	ehp_now = qnn_train_effective_hp(health_after, armor_after, armor_type_after);
	if (!player_died && qnn_reward_state.prev_ehp > 0.0f)
		reward += qnn_reward_weights.ehp_delta_weight * logf(ehp_now / fmaxf(qnn_reward_state.prev_ehp, 1.0f));

	/* EDP delta */
	reward += qnn_reward_weights.edp_delta_weight * edp_raw;

	/* Fire penalty */
	reward += qnn_reward_weights.fire_penalty * (shots_fired > 0 ? 1.0f : 0.0f);

	/* Self-damage penalty */
	reward += qnn_reward_weights.self_damage_penalty * damage_dealt_self;

	/* Tracking */
	tracking_cos = qnn_train_tracking_cosine(snapshot);
	*out_tracking_cos = tracking_cos;
	remapped = qnn_train_remap_tracking(tracking_cos, qnn_reward_weights.tracking_fov);
	if (!qnn_reward_weights.tracking_penalty && remapped < 0.0f)
		remapped = 0.0f;
	reward += qnn_reward_weights.tracking_weight * remapped;

	/* Update prev_ehp state for next tick */
	qnn_reward_state.prev_ehp = player_died ? 100.0f : ehp_now;

	return reward;
}

static void qnn_train_write_u16_le(FILE *out, uint16_t value)
{
	uint8_t bytes[2];

	bytes[0] = (uint8_t)(value & 0xffu);
	bytes[1] = (uint8_t)((value >> 8) & 0xffu);
	fwrite(bytes, 1, sizeof(bytes), out);
}

static void qnn_train_write_i16_le(FILE *out, int value)
{
	qnn_train_write_u16_le(out, (uint16_t)(int16_t)value);
}

static void qnn_train_write_u32_le(FILE *out, uint32_t value)
{
	uint8_t bytes[4];

	bytes[0] = (uint8_t)(value & 0xffu);
	bytes[1] = (uint8_t)((value >> 8) & 0xffu);
	bytes[2] = (uint8_t)((value >> 16) & 0xffu);
	bytes[3] = (uint8_t)((value >> 24) & 0xffu);
	fwrite(bytes, 1, sizeof(bytes), out);
}

static void qnn_train_write_f32_le(FILE *out, float value)
{
	union
	{
		float f;
		uint32_t u;
	} bits;

	bits.f = value;
	qnn_train_write_u32_le(out, bits.u);
}

static qboolean qnn_train_is_player(edict_t *ed)
{
	if (ed == NULL || ed == sv.edicts || ed->free || ed->v.classname == 0)
		return false;
	return !strcmp(pr_strings + ed->v.classname, "player");
}

static int qnn_train_weapon_id_from_bits(float weapon_bits)
{
	if (weapon_bits == IT_AXE)
		return 1;
	if (weapon_bits == IT_SHOTGUN)
		return 2;
	if (weapon_bits == IT_SUPER_SHOTGUN)
		return 3;
	if (weapon_bits == IT_NAILGUN)
		return 4;
	if (weapon_bits == IT_SUPER_NAILGUN)
		return 5;
	if (weapon_bits == IT_GRENADE_LAUNCHER)
		return 6;
	if (weapon_bits == IT_ROCKET_LAUNCHER)
		return 7;
	if (weapon_bits == IT_LIGHTNING)
		return 8;
	return 0;
}

static float qnn_train_effective_hp(float health, float armor, float armor_type)
{
	float armor_first;
	float health_first;

	if (health < 1.0f)
		health = 1.0f;
	if (armor <= 0.0f || armor_type <= 0.0f)
		return health;
	armor_first = health + armor;
	health_first = health / (1.0f - armor_type);
	return armor_first < health_first ? armor_first : health_first;
}

static void qnn_train_clear_tick(void)
{
	qnn_training_state.damage_count = 0;
	qnn_training_state.item_count = 0;
	qnn_training_state.death_count = 0;
	qnn_training_state.spawn_count = 0;
	memset(qnn_training_state.shot_counts, 0, sizeof(qnn_training_state.shot_counts));
	memset(qnn_training_state.last_shot_weapon_id, 0, sizeof(qnn_training_state.last_shot_weapon_id));
}

void qnn_worker_training_reset_episode(void)
{
	memset(&qnn_training_state, 0, sizeof(qnn_training_state));
	qnn_reward_state.prev_ehp = 100.0f;
}

void qnn_worker_training_reset_tick(void)
{
	qnn_train_clear_tick();
}

static void qnn_train_append_spawn(int entity_num, edict_t *player)
{
	qnn_training_spawn_record_t *record;

	if (qnn_training_state.spawn_count >= QNN_WORKER_MAX_TRAIN_SPAWNS)
		return;
	record = &qnn_training_state.spawns[qnn_training_state.spawn_count++];
	record->player_entity_num = entity_num;
	record->flags = 0;
	VectorCopy(player->v.origin, record->origin);
}

static void qnn_train_detect_player_spawns(qboolean reset_flag)
{
	int client_idx;

	for (client_idx = 0; client_idx < svs.maxclients; ++client_idx)
	{
		edict_t *player;
		int entity_num;
		qboolean alive_now;

		if (!svs.clients[client_idx].active || svs.clients[client_idx].edict == NULL)
			continue;
		player = svs.clients[client_idx].edict;
		entity_num = NUM_FOR_EDICT(player);
		alive_now = qnn_train_is_player(player) && player->v.health > 0.0f;
		if (alive_now && (reset_flag || !qnn_training_state.prev_alive[entity_num]))
			qnn_train_append_spawn(entity_num, player);
		qnn_training_state.prev_alive[entity_num] = alive_now;
	}
}

void PF_qnn_training_note_shot(void)
{
	edict_t *shooter;
	int entity_num;
	int weapon_id;

	shooter = G_EDICT(OFS_PARM0);
	if (!qnn_train_is_player(shooter))
		return;
	entity_num = NUM_FOR_EDICT(shooter);
	if (entity_num <= 0 || entity_num >= MAX_EDICTS)
		return;
	weapon_id = (int)G_FLOAT(OFS_PARM1);
	qnn_training_state.shot_counts[entity_num] += 1;
	if (weapon_id > 0)
		qnn_training_state.last_shot_weapon_id[entity_num] = weapon_id;
}

void PF_qnn_training_note_damage(void)
{
	edict_t *attacker;
	edict_t *target;
	qnn_training_damage_record_t *record;

	attacker = G_EDICT(OFS_PARM0);
	target = G_EDICT(OFS_PARM1);
	if (!qnn_train_is_player(attacker) && !qnn_train_is_player(target))
		return;
	if (qnn_training_state.damage_count >= QNN_WORKER_MAX_TRAIN_DAMAGE)
		return;

	record = &qnn_training_state.damage[qnn_training_state.damage_count++];
	record->attacker_entity_num = attacker != NULL ? NUM_FOR_EDICT(attacker) : 0;
	record->target_entity_num = target != NULL ? NUM_FOR_EDICT(target) : 0;
	record->weapon_id = (int)G_FLOAT(OFS_PARM2);
	record->flags = 0;
	if (record->attacker_entity_num == record->target_entity_num && record->target_entity_num > 0)
		record->flags |= QNN_TRAIN_DAMAGE_FLAG_SELF;
	if (attacker == sv.edicts || record->attacker_entity_num == 0)
		record->flags |= QNN_TRAIN_DAMAGE_FLAG_WORLD;
	if (G_FLOAT(OFS_PARM6) != 0.0f)
		record->flags |= QNN_TRAIN_DAMAGE_FLAG_SPLASH;
	record->health_before = G_FLOAT(OFS_PARM3);
	record->armor_before = G_FLOAT(OFS_PARM4);
	record->armor_type_before = G_FLOAT(OFS_PARM5);
	record->health_after = target != NULL ? target->v.health : record->health_before;
	record->armor_after = target != NULL ? target->v.armorvalue : record->armor_before;
	record->armor_type_after = target != NULL ? target->v.armortype : record->armor_type_before;
	record->damage_health = record->health_before - record->health_after;
	record->damage_armor = record->armor_before - record->armor_after;
	if (record->damage_health < 0.0f)
		record->damage_health = 0.0f;
	if (record->damage_armor < 0.0f)
		record->damage_armor = 0.0f;
}

void PF_qnn_training_note_death(void)
{
	edict_t *victim;
	edict_t *attacker;
	qnn_training_death_record_t *record;

	victim = G_EDICT(OFS_PARM0);
	attacker = G_EDICT(OFS_PARM1);
	if (!qnn_train_is_player(victim))
		return;
	if (qnn_training_state.death_count >= QNN_WORKER_MAX_TRAIN_DEATHS)
		return;
	record = &qnn_training_state.deaths[qnn_training_state.death_count++];
	record->victim_entity_num = NUM_FOR_EDICT(victim);
	record->attacker_entity_num = attacker != NULL ? NUM_FOR_EDICT(attacker) : 0;
	record->weapon_id = (int)G_FLOAT(OFS_PARM2);
	record->flags = (int)G_FLOAT(OFS_PARM3);
}

void PF_qnn_training_note_item(void)
{
	edict_t *actor;
	edict_t *item;
	qnn_training_item_record_t *record;

	actor = G_EDICT(OFS_PARM0);
	item = G_EDICT(OFS_PARM1);
	if (item == NULL || item->free)
		return;
	if (qnn_training_state.item_count >= QNN_WORKER_MAX_TRAIN_ITEMS)
		return;
	record = &qnn_training_state.items[qnn_training_state.item_count++];
	record->actor_entity_num = (actor != NULL && actor != sv.edicts) ? NUM_FOR_EDICT(actor) : 0;
	record->item_entity_num = NUM_FOR_EDICT(item);
	record->event_kind = (int)G_FLOAT(OFS_PARM2);
	record->category = (int)G_FLOAT(OFS_PARM3);
	record->amount = G_FLOAT(OFS_PARM4);
	record->weapon_id = (int)G_FLOAT(OFS_PARM5);
	record->flags = 0;
	VectorCopy(item->v.origin, record->origin);
}

/* FrikBotNex calls frik_checkextension (#99) at startup.  Return 0 so
   the bot editor knows no engine extensions are available. */

void PF_qnn_checkextension(void)
{
	G_FLOAT(OFS_RETURN) = 0;
}

void qnn_worker_write_training_extras_binary(FILE *out, const qnn_worker_snapshot_t *snapshot, int tick, int steps, qboolean reset_flag)
{
	float damage_dealt;
	float damage_taken;
	float edp_raw;
	float pickup_health;
	float pickup_armor;
	float pickup_ammo;
	float weapon_pickups;
	float item_pickups;
	float health_before;
	float armor_before;
	float armor_type_before;
	float health_after;
	float armor_after;
	float armor_type_after;
	int self_entity_num;
	int weapon_id;
	int frag_gain;
	int frag_loss;
	int player_died;
	int hit_count;
	int shots_fired;
	int current_frags;
	int idx;
	uint16_t flags;

	qnn_train_detect_player_spawns(reset_flag);

	self_entity_num = cl.viewentity;
	health_after = (float)snapshot->health;
	armor_after = (float)snapshot->armor;
	armor_type_after = snapshot->armor_type;
	current_frags = 0;
	if (self_entity_num > 0 && self_entity_num < sv.num_edicts)
		current_frags = (int)EDICT_NUM(self_entity_num)->v.frags;

	if (!qnn_training_state.prev_self_valid || qnn_training_state.prev_self_entity_num != self_entity_num)
	{
		health_before = health_after;
		armor_before = armor_after;
		armor_type_before = armor_type_after;
		frag_gain = 0;
		frag_loss = 0;
	}
	else
	{
		health_before = qnn_training_state.prev_health;
		armor_before = qnn_training_state.prev_armor;
		armor_type_before = qnn_training_state.prev_armor_type;
		frag_gain = current_frags > qnn_training_state.prev_frags ? current_frags - qnn_training_state.prev_frags : 0;
		frag_loss = current_frags < qnn_training_state.prev_frags ? qnn_training_state.prev_frags - current_frags : 0;
	}

	damage_dealt = 0.0f;
	damage_taken = 0.0f;
	edp_raw = 0.0f;
	pickup_health = 0.0f;
	pickup_armor = 0.0f;
	pickup_ammo = 0.0f;
	weapon_pickups = 0.0f;
	item_pickups = 0.0f;
	player_died = 0;
	hit_count = 0;
	shots_fired = (self_entity_num > 0 && self_entity_num < MAX_EDICTS) ? qnn_training_state.shot_counts[self_entity_num] : 0;
	weapon_id = (self_entity_num > 0 && self_entity_num < MAX_EDICTS) ? qnn_training_state.last_shot_weapon_id[self_entity_num] : 0;
	if (weapon_id <= 0)
		weapon_id = snapshot->weapon_id;

	for (idx = 0; idx < qnn_training_state.damage_count; ++idx)
	{
		qnn_training_damage_record_t *record = &qnn_training_state.damage[idx];
		float total_delta = record->damage_health + record->damage_armor;

		if (record->attacker_entity_num == self_entity_num)
		{
			float ehp_before = qnn_train_effective_hp(record->health_before, record->armor_before, record->armor_type_before);
			float ehp_after = qnn_train_effective_hp(record->health_after, record->armor_after, record->armor_type_after);

			damage_dealt += total_delta;
			if (record->target_entity_num != self_entity_num && total_delta > 0.0f)
				hit_count += 1;
			if (record->target_entity_num != self_entity_num && ehp_before > ehp_after && ehp_after > 0.0f)
				edp_raw += logf(ehp_before / ehp_after);
		}
		if (record->target_entity_num == self_entity_num)
			damage_taken += total_delta;
	}

	for (idx = 0; idx < qnn_training_state.item_count; ++idx)
	{
		qnn_training_item_record_t *record = &qnn_training_state.items[idx];

		if (record->actor_entity_num != self_entity_num || record->event_kind != QNN_TRAIN_EVENT_PICKUP)
			continue;
		if (record->category == QNN_TRAIN_PICKUP_HEALTH)
			pickup_health += record->amount;
		else if (record->category == QNN_TRAIN_PICKUP_ARMOR)
			pickup_armor += record->amount;
		else if (record->category == QNN_TRAIN_PICKUP_AMMO)
			pickup_ammo += record->amount;
		else if (record->category == QNN_TRAIN_PICKUP_WEAPON)
			weapon_pickups += 1.0f;
		else
			item_pickups += 1.0f;
	}

	for (idx = 0; idx < qnn_training_state.death_count; ++idx)
	{
		if (qnn_training_state.deaths[idx].victim_entity_num == self_entity_num)
			player_died = 1;
	}

	/* Compute self-damage (attacker == target == self) for reward */
	{
		float damage_dealt_self = 0.0f;
		float computed_reward;
		float tracking_cos;
		float out_damage_self;

		for (idx = 0; idx < qnn_training_state.damage_count; ++idx)
		{
			qnn_training_damage_record_t *record = &qnn_training_state.damage[idx];
			if (record->attacker_entity_num == self_entity_num
				&& record->target_entity_num == self_entity_num)
				damage_dealt_self += record->damage_health + record->damage_armor;
		}

		computed_reward = qnn_train_compute_reward(
			snapshot, damage_dealt_self, edp_raw, frag_gain, player_died,
			shots_fired, health_after, armor_after, armor_type_after,
			&tracking_cos, &out_damage_self);

		/* Store for writing after header (version 2 extension) */
		qnn_training_state._computed_reward = computed_reward;
		qnn_training_state._tracking_cos = tracking_cos;
		qnn_training_state._damage_dealt_self = out_damage_self;
	}

	flags = 0;
	if (reset_flag)
		flags |= QNN_TRAIN_FLAG_RESET;
	if (snapshot->done)
		flags |= QNN_TRAIN_FLAG_DONE;

	fwrite(QNN_TRAIN_BINARY_MAGIC, 1, 4, out);
	qnn_train_write_u16_le(out, (uint16_t)QNN_TRAIN_BINARY_VERSION);
	qnn_train_write_u16_le(out, flags);
	qnn_train_write_u32_le(out, (uint32_t)tick);
	qnn_train_write_u32_le(out, (uint32_t)steps);
	qnn_train_write_i16_le(out, self_entity_num);
	qnn_train_write_i16_le(out, weapon_id);
	qnn_train_write_u16_le(out, (uint16_t)qnn_training_state.damage_count);
	qnn_train_write_u16_le(out, (uint16_t)qnn_training_state.item_count);
	qnn_train_write_u16_le(out, (uint16_t)qnn_training_state.death_count);
	qnn_train_write_u16_le(out, (uint16_t)qnn_training_state.spawn_count);
	qnn_train_write_i16_le(out, frag_gain);
	qnn_train_write_i16_le(out, frag_loss);
	qnn_train_write_u16_le(out, (uint16_t)player_died);
	qnn_train_write_u16_le(out, (uint16_t)hit_count);
	qnn_train_write_u16_le(out, (uint16_t)shots_fired);
	qnn_train_write_u16_le(out, 0);
	qnn_train_write_f32_le(out, damage_dealt);
	qnn_train_write_f32_le(out, damage_taken);
	qnn_train_write_f32_le(out, health_before);
	qnn_train_write_f32_le(out, health_after);
	qnn_train_write_f32_le(out, armor_before);
	qnn_train_write_f32_le(out, armor_after);
	qnn_train_write_f32_le(out, armor_type_before);
	qnn_train_write_f32_le(out, armor_type_after);
	qnn_train_write_f32_le(out, edp_raw);
	qnn_train_write_f32_le(out, pickup_health);
	qnn_train_write_f32_le(out, pickup_armor);
	qnn_train_write_f32_le(out, pickup_ammo);
	qnn_train_write_f32_le(out, weapon_pickups);
	qnn_train_write_f32_le(out, item_pickups);
	qnn_train_write_f32_le(out, (float)((self_entity_num > 0 && self_entity_num < MAX_EDICTS) ? qnn_training_state.total_damage_dealt[self_entity_num] + damage_dealt : damage_dealt));
	qnn_train_write_f32_le(out, (float)((self_entity_num > 0 && self_entity_num < MAX_EDICTS) ? qnn_training_state.total_hit_count[self_entity_num] + hit_count : hit_count));
	qnn_train_write_f32_le(out, (float)((self_entity_num > 0 && self_entity_num < MAX_EDICTS) ? qnn_training_state.total_shots_fired[self_entity_num] + shots_fired : shots_fired));

	/* Version 2 extension: computed reward, tracking cosine, self-damage */
	qnn_train_write_f32_le(out, qnn_training_state._computed_reward);
	qnn_train_write_f32_le(out, qnn_training_state._tracking_cos);
	qnn_train_write_f32_le(out, qnn_training_state._damage_dealt_self);

	for (idx = 0; idx < qnn_training_state.damage_count; ++idx)
	{
		qnn_training_damage_record_t *record = &qnn_training_state.damage[idx];

		qnn_train_write_i16_le(out, record->attacker_entity_num);
		qnn_train_write_i16_le(out, record->target_entity_num);
		qnn_train_write_u16_le(out, (uint16_t)record->weapon_id);
		qnn_train_write_u16_le(out, (uint16_t)record->flags);
		qnn_train_write_f32_le(out, record->health_before);
		qnn_train_write_f32_le(out, record->health_after);
		qnn_train_write_f32_le(out, record->armor_before);
		qnn_train_write_f32_le(out, record->armor_after);
		qnn_train_write_f32_le(out, record->armor_type_before);
		qnn_train_write_f32_le(out, record->armor_type_after);
		qnn_train_write_f32_le(out, record->damage_health);
		qnn_train_write_f32_le(out, record->damage_armor);
	}

	for (idx = 0; idx < qnn_training_state.item_count; ++idx)
	{
		qnn_training_item_record_t *record = &qnn_training_state.items[idx];

		qnn_train_write_i16_le(out, record->actor_entity_num);
		qnn_train_write_i16_le(out, record->item_entity_num);
		qnn_train_write_u16_le(out, (uint16_t)record->event_kind);
		qnn_train_write_u16_le(out, (uint16_t)record->category);
		qnn_train_write_u16_le(out, (uint16_t)record->weapon_id);
		qnn_train_write_u16_le(out, (uint16_t)record->flags);
		qnn_train_write_f32_le(out, record->amount);
		qnn_train_write_f32_le(out, record->origin[0]);
		qnn_train_write_f32_le(out, record->origin[1]);
		qnn_train_write_f32_le(out, record->origin[2]);
	}

	for (idx = 0; idx < qnn_training_state.death_count; ++idx)
	{
		qnn_training_death_record_t *record = &qnn_training_state.deaths[idx];

		qnn_train_write_i16_le(out, record->victim_entity_num);
		qnn_train_write_i16_le(out, record->attacker_entity_num);
		qnn_train_write_u16_le(out, (uint16_t)record->weapon_id);
		qnn_train_write_u16_le(out, (uint16_t)record->flags);
	}

	for (idx = 0; idx < qnn_training_state.spawn_count; ++idx)
	{
		qnn_training_spawn_record_t *record = &qnn_training_state.spawns[idx];

		qnn_train_write_i16_le(out, record->player_entity_num);
		qnn_train_write_u16_le(out, (uint16_t)record->flags);
		qnn_train_write_f32_le(out, record->origin[0]);
		qnn_train_write_f32_le(out, record->origin[1]);
		qnn_train_write_f32_le(out, record->origin[2]);
	}

	fflush(out);

	if (self_entity_num > 0 && self_entity_num < MAX_EDICTS)
	{
		qnn_training_state.total_damage_dealt[self_entity_num] += damage_dealt;
		qnn_training_state.total_hit_count[self_entity_num] += hit_count;
		qnn_training_state.total_shots_fired[self_entity_num] += shots_fired;
	}
	qnn_training_state.prev_self_valid = true;
	qnn_training_state.prev_self_entity_num = self_entity_num;
	qnn_training_state.prev_health = health_after;
	qnn_training_state.prev_armor = armor_after;
	qnn_training_state.prev_armor_type = armor_type_after;
	qnn_training_state.prev_frags = current_frags;
}
