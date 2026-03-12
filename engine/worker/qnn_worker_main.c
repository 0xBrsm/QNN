#include "qnn_worker.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define QNN_WORKER_PROTOCOL "v5"
#define QNN_WORKER_SERVER_NAME "quake-worker"
#define QNN_WORKER_UPSTREAM_COMMIT "bf4ac424ce754894ac8f1dae6a3981954bc9852d"
#define QNN_WORKER_MAX_LINE 8192
#define QNN_WORKER_MAX_EPISODE_ID 128
#define QNN_WORKER_MAX_COMMAND_TEXT 1024
#define QNN_WORKER_MAX_BINARY_STRINGS 128

#define QNN_STEP_BINARY_MAGIC "QWLD"
#define QNN_STEP_BINARY_VERSION 1

#define QNN_STEP_FLAG_DONE         0x0001
#define QNN_STEP_FLAG_GOAL_REACHED 0x0002

#define QNN_OUTPUT_WORLD_JSON 0
#define QNN_OUTPUT_WORLD_BINARY_V1 1
#define QNN_OUTPUT_TOKEN_BINARY_V2 2

#define QNN_TRAIN_OUTPUT_NONE 0
#define QNN_TRAIN_OUTPUT_BINARY_V1 1

#define QNN_ENTITY_FLAG_STATIC_PROXY 0x0001

#define QNN_EVENT_FLAG_HAS_DELTA     0x0001
#define QNN_EVENT_FLAG_HAS_WEAPON_ID 0x0002

#define QNN_DONE_REASON_NONE         0
#define QNN_DONE_REASON_GOAL_REACHED 1
#define QNN_DONE_REASON_PLAYER_DIED  2
#define QNN_DONE_REASON_TIMEOUT      3

#define QNN_EVENT_DAMAGE_TAKEN  1
#define QNN_EVENT_PICKUP_HEALTH 2
#define QNN_EVENT_PICKUP_ARMOR  3
#define QNN_EVENT_PICKUP_AMMO   4
#define QNN_EVENT_PICKUP_WEAPON 5
#define QNN_EVENT_PICKUP_ITEM   6
#define QNN_EVENT_FRAG_GAINED   7
#define QNN_EVENT_FRAG_LOST     8
#define QNN_EVENT_MONSTER_KILL  9
#define QNN_EVENT_DAMAGE_DEALT  10
#define QNN_EVENT_HIT_CONFIRMED 11
#define QNN_EVENT_SHOTS_FIRED   12
#define QNN_EVENT_GOAL_REACHED  13
#define QNN_EVENT_PLAYER_DIED   14

typedef struct
{
	int	fixed_tick_hz;
	float	fixed_dt;
	int	seed;
	int	episode_index;
	int	tick;
	int	steps;
	qboolean has_reset;
	qboolean done;
	int	prev_health;
	int	prev_armor;
	int	prev_ammo;
	int	prev_weapon;
	int	prev_items;
	int	prev_intermission;
	int	prev_frags;
	int	prev_monster_kills;
	int	recent_fire_steps;
	int	last_fire_weapon_id;
	int	total_damage_dealt;
	int	total_hit_count;
	int	total_shots_fired;
	int	weapon_damage_dealt[9];
	int	weapon_hits_landed[9];
	int	weapon_shots_fired[9];
	int	prev_entity_health[MAX_EDICTS];
	qboolean prev_entity_active[MAX_EDICTS];
	qnn_worker_action_t history[QNN_WORKER_ACTION_HISTORY];
	int	history_count;
	char	episode_id[QNN_WORKER_MAX_EPISODE_ID];
	int	output_mode;
	int	training_output_mode;
} qnn_worker_runtime_t;

typedef struct
{
	int	maxplayers;
	int	skill;
	int	deathmatch;
	int	coop;
	int	teamplay;
	int	fraglimit;
	int	timelimit;
	int	samelevel;
	char	pre_map_commands[QNN_WORKER_MAX_COMMAND_TEXT];
	char	post_map_commands[QNN_WORKER_MAX_COMMAND_TEXT];
} qnn_worker_reset_options_t;

typedef struct
{
	const char	*values[QNN_WORKER_MAX_BINARY_STRINGS];
	int	count;
} qnn_worker_binary_string_table_t;

static qnn_worker_runtime_t qnn_worker_runtime;
static qnn_worker_reset_options_t qnn_worker_reset_options;

static void qnn_worker_write_action_json(FILE *out, const qnn_worker_action_t *action)
{
	fprintf(out, "{\"move\":%d,\"strafe\":%d,\"look_yaw\":%d,\"look_pitch\":%d,\"fire\":%d,\"jump\":%d,\"weapon\":%d}",
		action->move,
		action->strafe,
		action->look_yaw,
		action->look_pitch,
		action->fire,
		action->jump,
		action->weapon);
}

static qboolean qnn_worker_client_ready(void)
{
	return (sv.active
		&& cls.state == ca_connected
		&& cls.signon == SIGNONS
		&& cl.worldmodel != NULL
		&& cl.viewentity > 0
		&& cl.viewentity < MAX_EDICTS) ? true : false;
}

static int qnn_worker_current_monster_kills(void)
{
	return cl.stats[STAT_MONSTERS];
}

static int qnn_worker_total_monsters(void)
{
	return cl.stats[STAT_TOTALMONSTERS];
}

static void qnn_worker_reset_options_defaults(qnn_worker_reset_options_t *options)
{
	memset(options, 0, sizeof(*options));
	options->maxplayers = 1;
	options->skill = 0;
	options->deathmatch = 0;
	options->coop = 0;
	options->teamplay = 0;
	options->fraglimit = 0;
	options->timelimit = 0;
	options->samelevel = 1;
}

static void qnn_worker_parse_reset_options(const char *line, qnn_worker_reset_options_t *options)
{
	qnn_worker_reset_options_defaults(options);
	options->maxplayers = qnn_json_extract_int(line, "\"maxplayers\"", options->maxplayers);
	if (options->maxplayers < 1)
		options->maxplayers = 1;
	options->skill = qnn_json_extract_int(line, "\"skill\"", options->skill);
	options->deathmatch = qnn_json_extract_int(line, "\"deathmatch\"", options->deathmatch);
	options->coop = qnn_json_extract_int(line, "\"coop\"", options->coop);
	options->teamplay = qnn_json_extract_int(line, "\"teamplay\"", options->teamplay);
	options->fraglimit = qnn_json_extract_int(line, "\"fraglimit\"", options->fraglimit);
	options->timelimit = qnn_json_extract_int(line, "\"timelimit\"", options->timelimit);
	options->samelevel = qnn_json_extract_int(line, "\"samelevel\"", options->samelevel);
	qnn_json_extract_string(line, "\"pre_map_commands\"", options->pre_map_commands, sizeof(options->pre_map_commands));
	qnn_json_extract_string(line, "\"post_map_commands\"", options->post_map_commands, sizeof(options->post_map_commands));
}

static void qnn_worker_append_command(char *buffer, size_t buffer_size, const char *command_text)
{
	size_t used;
	size_t available;

	if (command_text == NULL || command_text[0] == 0)
		return;
	used = strlen(buffer);
	if (used >= buffer_size - 1)
		return;
	available = buffer_size - used - 1;
	strncat(buffer, command_text, available);
	used = strlen(buffer);
	if (used == 0 || buffer[used - 1] != '\n')
		strncat(buffer, "\n", buffer_size - strlen(buffer) - 1);
}

static int qnn_worker_weapon_index(int weapon_id)
{
	if (weapon_id < 0)
		return 0;
	if (weapon_id > 8)
		return 8;
	return weapon_id;
}

static void qnn_worker_entity_id_string(int entity_num, char *out, size_t out_size)
{
	if (entity_num <= 0)
	{
		out[0] = 0;
		return;
	}
	snprintf(out, out_size, "entity_%04d", entity_num);
}

static qboolean qnn_worker_track_damage_entity(int entity_num, const edict_t *edict)
{
	const char *classname;

	if (!edict || edict->free || entity_num == cl.viewentity || !edict->v.classname)
		return false;
	classname = qnn_worker_prog_string(edict->v.classname);
	if (!strcmp(classname, "player"))
		return true;
	if (!strncmp(classname, "monster_", 8))
		return true;
	return false;
}

static void qnn_worker_cache_entity_state(void)
{
	int entity_num;

	memset(qnn_worker_runtime.prev_entity_health, 0, sizeof(qnn_worker_runtime.prev_entity_health));
	memset(qnn_worker_runtime.prev_entity_active, 0, sizeof(qnn_worker_runtime.prev_entity_active));
	if (!sv.active)
		return;
	for (entity_num = 0; entity_num < sv.num_edicts && entity_num < MAX_EDICTS; ++entity_num)
	{
		edict_t *edict;

		edict = EDICT_NUM(entity_num);
		if (!qnn_worker_track_damage_entity(entity_num, edict))
			continue;
		qnn_worker_runtime.prev_entity_health[entity_num] = (int)edict->v.health;
		qnn_worker_runtime.prev_entity_active[entity_num] = true;
	}
}

static void qnn_worker_write_int_array(FILE *out, const int *values, int count)
{
	int i;

	fputc('[', out);
	for (i = 0; i < count; ++i)
	{
		if (i > 0)
			fputc(',', out);
		fprintf(out, "%d", values[i]);
	}
	fputc(']', out);
}

static void qnn_worker_write_server_players(FILE *out)
{
	int entity_num;
	int wrote;

	fputc('[', out);
	wrote = 0;
	for (entity_num = 1; entity_num < sv.num_edicts && entity_num <= 16; ++entity_num)
	{
		edict_t *edict;
		const char *classname;
		const char *netname;

		edict = EDICT_NUM(entity_num);
		if (edict->free)
			continue;
		classname = qnn_worker_prog_string(edict->v.classname);
		netname = qnn_worker_prog_string(edict->v.netname);
		if (strcmp(classname, "player") && !netname[0])
			continue;
		if (wrote)
			fputc(',', out);
		wrote = 1;
		fprintf(out, "{\"entity_num\":%d,\"classname\":", entity_num);
		qnn_worker_write_json_string(out, classname);
		fprintf(out, ",\"netname\":");
		qnn_worker_write_json_string(out, netname);
		fprintf(out, ",\"frags\":%d,\"health\":%.0f,\"origin\":[%.1f,%.1f,%.1f]}",
			(int)edict->v.frags,
			edict->v.health,
			edict->v.origin[0],
			edict->v.origin[1],
			edict->v.origin[2]);
	}
	fputc(']', out);
}

static void qnn_worker_write_bytes(FILE *out, const void *data, size_t size)
{
	if (size > 0)
		fwrite(data, 1, size, out);
}

static void qnn_worker_write_u16_le(FILE *out, uint16_t value)
{
	unsigned char bytes[2];

	bytes[0] = (unsigned char)(value & 0xff);
	bytes[1] = (unsigned char)((value >> 8) & 0xff);
	qnn_worker_write_bytes(out, bytes, sizeof(bytes));
}

static void qnn_worker_write_i16_le(FILE *out, int value)
{
	qnn_worker_write_u16_le(out, (uint16_t)(int16_t)value);
}

static void qnn_worker_write_u32_le(FILE *out, uint32_t value)
{
	unsigned char bytes[4];

	bytes[0] = (unsigned char)(value & 0xff);
	bytes[1] = (unsigned char)((value >> 8) & 0xff);
	bytes[2] = (unsigned char)((value >> 16) & 0xff);
	bytes[3] = (unsigned char)((value >> 24) & 0xff);
	qnn_worker_write_bytes(out, bytes, sizeof(bytes));
}

static void qnn_worker_write_i32_le(FILE *out, int value)
{
	qnn_worker_write_u32_le(out, (uint32_t)(int32_t)value);
}

static void qnn_worker_write_f32_le(FILE *out, float value)
{
	union
	{
		float	f;
		uint32_t u;
	} bits;

	bits.f = value;
	qnn_worker_write_u32_le(out, bits.u);
}

static int qnn_worker_binary_string_index(qnn_worker_binary_string_table_t *table, const char *value)
{
	int i;

	if (value == NULL || value[0] == 0)
		return 0;
	for (i = 0; i < table->count; ++i)
	{
		if (!strcmp(table->values[i], value))
			return i + 1;
	}
	if (table->count >= QNN_WORKER_MAX_BINARY_STRINGS)
		return 0;
	table->values[table->count] = value;
	table->count += 1;
	return table->count;
}

static int qnn_worker_done_reason_code(const char *done_reason)
{
	if (done_reason == NULL || done_reason[0] == 0)
		return QNN_DONE_REASON_NONE;
	if (!strcmp(done_reason, "goal_reached"))
		return QNN_DONE_REASON_GOAL_REACHED;
	if (!strcmp(done_reason, "player_died"))
		return QNN_DONE_REASON_PLAYER_DIED;
	if (!strcmp(done_reason, "timeout"))
		return QNN_DONE_REASON_TIMEOUT;
	return QNN_DONE_REASON_NONE;
}

static int qnn_worker_event_code(const char *event_type)
{
	if (event_type == NULL || event_type[0] == 0)
		return 0;
	if (!strcmp(event_type, "damage_taken"))
		return QNN_EVENT_DAMAGE_TAKEN;
	if (!strcmp(event_type, "pickup_health"))
		return QNN_EVENT_PICKUP_HEALTH;
	if (!strcmp(event_type, "pickup_armor"))
		return QNN_EVENT_PICKUP_ARMOR;
	if (!strcmp(event_type, "pickup_ammo"))
		return QNN_EVENT_PICKUP_AMMO;
	if (!strcmp(event_type, "pickup_weapon"))
		return QNN_EVENT_PICKUP_WEAPON;
	if (!strcmp(event_type, "pickup_item"))
		return QNN_EVENT_PICKUP_ITEM;
	if (!strcmp(event_type, "frag_gained"))
		return QNN_EVENT_FRAG_GAINED;
	if (!strcmp(event_type, "frag_lost"))
		return QNN_EVENT_FRAG_LOST;
	if (!strcmp(event_type, "monster_kill"))
		return QNN_EVENT_MONSTER_KILL;
	if (!strcmp(event_type, "damage_dealt"))
		return QNN_EVENT_DAMAGE_DEALT;
	if (!strcmp(event_type, "hit_confirmed"))
		return QNN_EVENT_HIT_CONFIRMED;
	if (!strcmp(event_type, "shots_fired"))
		return QNN_EVENT_SHOTS_FIRED;
	if (!strcmp(event_type, "goal_reached"))
		return QNN_EVENT_GOAL_REACHED;
	if (!strcmp(event_type, "player_died"))
		return QNN_EVENT_PLAYER_DIED;
	return 0;
}

static int qnn_worker_parse_protocol_version(const char *value, int fallback)
{
	const char *cursor;
	int parsed;

	if (value == NULL || value[0] == 0)
		return fallback;
	cursor = value;
	if (*cursor == 'v' || *cursor == 'V')
		cursor += 1;
	if (*cursor == 0)
		return fallback;
	parsed = atoi(cursor);
	return parsed > 0 ? parsed : fallback;
}

static void qnn_worker_collect_binary_strings(qnn_worker_binary_string_table_t *strings, const qnn_worker_snapshot_t *snapshot)
{
	int i;

	memset(strings, 0, sizeof(*strings));
	for (i = 0; i < snapshot->visible_count; ++i)
	{
		const qnn_worker_visible_entity_t *entity;

		entity = &snapshot->visible[i];
		qnn_worker_binary_string_index(strings, entity->classname);
		if (entity->entity_num <= 0 || entity->static_proxy)
			qnn_worker_binary_string_index(strings, entity->entity_id);
	}
	for (i = 0; i < snapshot->sound_count; ++i)
		qnn_worker_binary_string_index(strings, snapshot->sounds[i].name);
}

static void qnn_worker_write_binary_strings(FILE *out, const qnn_worker_binary_string_table_t *strings)
{
	int i;

	for (i = 0; i < strings->count; ++i)
	{
		const char *value;
		size_t length;

		value = strings->values[i];
		length = strlen(value);
		if (length > 0xffff)
			length = 0xffff;
		qnn_worker_write_u16_le(out, (uint16_t)length);
		qnn_worker_write_bytes(out, value, length);
	}
}

static void qnn_worker_write_action_binary(FILE *out, const qnn_worker_action_t *action)
{
	qnn_worker_write_i16_le(out, action->move);
	qnn_worker_write_i16_le(out, action->strafe);
	qnn_worker_write_i16_le(out, action->look_yaw);
	qnn_worker_write_i16_le(out, action->look_pitch);
	qnn_worker_write_i16_le(out, action->fire);
	qnn_worker_write_i16_le(out, action->jump);
	qnn_worker_write_i16_le(out, action->weapon);
}

static void qnn_worker_write_step_binary(FILE *out, const qnn_worker_snapshot_t *snapshot, const qnn_worker_action_t *current_action, float reward)
{
	qnn_worker_binary_string_table_t strings;
	uint16_t flags;
	int i;

	qnn_worker_collect_binary_strings(&strings, snapshot);
	flags = 0;
	if (snapshot->done)
		flags |= QNN_STEP_FLAG_DONE;
	if (snapshot->goal_reached)
		flags |= QNN_STEP_FLAG_GOAL_REACHED;

	qnn_worker_write_bytes(out, QNN_STEP_BINARY_MAGIC, 4);
	qnn_worker_write_u16_le(out, QNN_STEP_BINARY_VERSION);
	qnn_worker_write_u16_le(out, flags);
	qnn_worker_write_f32_le(out, reward);
	qnn_worker_write_i32_le(out, qnn_worker_runtime.tick);
	qnn_worker_write_i32_le(out, qnn_worker_runtime.steps);
	qnn_worker_write_i32_le(out, snapshot->current_region_id);
	qnn_worker_write_i32_le(out, qnn_worker_current_frags());
	qnn_worker_write_i32_le(out, qnn_worker_current_monster_kills());
	qnn_worker_write_i32_le(out, qnn_worker_total_monsters());
	qnn_worker_write_i32_le(out, snapshot->health);
	qnn_worker_write_i32_le(out, snapshot->armor);
	qnn_worker_write_i32_le(out, snapshot->ammo);
	qnn_worker_write_i32_le(out, snapshot->weapon_id);
	qnn_worker_write_i32_le(out, snapshot->weapons_owned);
	qnn_worker_write_i32_le(out, snapshot->ammo_shells);
	qnn_worker_write_i32_le(out, snapshot->ammo_nails);
	qnn_worker_write_i32_le(out, snapshot->ammo_rockets);
	qnn_worker_write_i32_le(out, snapshot->ammo_cells);
	qnn_worker_write_i32_le(out, snapshot->grounded ? 1 : 0);
	qnn_worker_write_f32_le(out, snapshot->armor_type);
	for (i = 0; i < 3; ++i)
		qnn_worker_write_f32_le(out, snapshot->player_origin[i]);
	for (i = 0; i < 3; ++i)
		qnn_worker_write_f32_le(out, snapshot->player_velocity[i]);
	for (i = 0; i < 3; ++i)
		qnn_worker_write_f32_le(out, snapshot->player_view_angles[i]);
	qnn_worker_write_i32_le(out, snapshot->damage_dealt);
	qnn_worker_write_i32_le(out, qnn_worker_runtime.total_damage_dealt);
	qnn_worker_write_i32_le(out, snapshot->damage_weapon_id);
	qnn_worker_write_i32_le(out, snapshot->hit_count);
	qnn_worker_write_i32_le(out, qnn_worker_runtime.total_hit_count);
	qnn_worker_write_i32_le(out, snapshot->shots_fired);
	qnn_worker_write_i32_le(out, qnn_worker_runtime.total_shots_fired);
	qnn_worker_write_i32_le(out, qnn_worker_done_reason_code(snapshot->done_reason));
	for (i = 0; i < 9; ++i)
		qnn_worker_write_i32_le(out, qnn_worker_runtime.weapon_damage_dealt[i]);
	for (i = 0; i < 9; ++i)
		qnn_worker_write_i32_le(out, qnn_worker_runtime.weapon_hits_landed[i]);
	for (i = 0; i < 9; ++i)
		qnn_worker_write_i32_le(out, qnn_worker_runtime.weapon_shots_fired[i]);
	qnn_worker_write_action_binary(out, current_action);
	qnn_worker_write_u16_le(out, (uint16_t)qnn_worker_runtime.history_count);
	qnn_worker_write_u16_le(out, (uint16_t)snapshot->visible_count);
	qnn_worker_write_u16_le(out, (uint16_t)snapshot->event_count);
	qnn_worker_write_u16_le(out, (uint16_t)snapshot->sound_count);
	qnn_worker_write_u16_le(out, (uint16_t)strings.count);

	for (i = 0; i < qnn_worker_runtime.history_count; ++i)
		qnn_worker_write_action_binary(out, &qnn_worker_runtime.history[i]);

	for (i = 0; i < snapshot->visible_count; ++i)
	{
		const qnn_worker_visible_entity_t *entity;
		int class_idx;
		int entity_id_idx;
		uint16_t entity_flags;

		entity = &snapshot->visible[i];
		class_idx = qnn_worker_binary_string_index(&strings, entity->classname);
		entity_id_idx = (entity->entity_num <= 0 || entity->static_proxy)
			? qnn_worker_binary_string_index(&strings, entity->entity_id)
			: 0;
		entity_flags = entity->static_proxy ? QNN_ENTITY_FLAG_STATIC_PROXY : 0;

		qnn_worker_write_i32_le(out, entity->entity_key);
		qnn_worker_write_i32_le(out, entity->entity_num);
		qnn_worker_write_i32_le(out, entity->region_id);
		qnn_worker_write_u16_le(out, (uint16_t)class_idx);
		qnn_worker_write_u16_le(out, (uint16_t)entity_id_idx);
		qnn_worker_write_u16_le(out, entity_flags);
		qnn_worker_write_f32_le(out, entity->origin[0]);
		qnn_worker_write_f32_le(out, entity->origin[1]);
		qnn_worker_write_f32_le(out, entity->origin[2]);
	}

	for (i = 0; i < snapshot->event_count; ++i)
	{
		const qnn_worker_event_t *event;
		uint16_t event_flags;

		event = &snapshot->events[i];
		event_flags = 0;
		if (event->has_delta)
			event_flags |= QNN_EVENT_FLAG_HAS_DELTA;
		if (event->has_weapon_id)
			event_flags |= QNN_EVENT_FLAG_HAS_WEAPON_ID;
		qnn_worker_write_u16_le(out, (uint16_t)qnn_worker_event_code(event->event_type));
		qnn_worker_write_u16_le(out, event_flags);
		qnn_worker_write_i32_le(out, event->region_id);
		qnn_worker_write_i32_le(out, event->delta);
		qnn_worker_write_i32_le(out, event->weapon_id);
		qnn_worker_write_i32_le(out, event->source_entity_num);
		qnn_worker_write_i32_le(out, event->target_entity_num);
	}

	for (i = 0; i < snapshot->sound_count; ++i)
	{
		const qnn_worker_sound_event_t *sound;
		int name_idx;

		sound = &snapshot->sounds[i];
		name_idx = qnn_worker_binary_string_index(&strings, sound->name);
		qnn_worker_write_u16_le(out, (uint16_t)name_idx);
		qnn_worker_write_u16_le(out, (uint16_t)sound->category);
		qnn_worker_write_i32_le(out, sound->entity_num);
		qnn_worker_write_f32_le(out, sound->origin[0]);
		qnn_worker_write_f32_le(out, sound->origin[1]);
		qnn_worker_write_f32_le(out, sound->origin[2]);
		qnn_worker_write_f32_le(out, sound->volume);
		qnn_worker_write_f32_le(out, sound->attenuation);
	}

	qnn_worker_write_binary_strings(out, &strings);
	fflush(out);
}

static void qnn_worker_runtime_reset(void)
{
	memset(&qnn_worker_runtime, 0, sizeof(qnn_worker_runtime));
	qnn_worker_runtime.fixed_tick_hz = 20;
	qnn_worker_runtime.fixed_dt = 1.0f / 20.0f;
	qnn_worker_runtime.output_mode = QNN_OUTPUT_WORLD_JSON;
	qnn_worker_runtime.training_output_mode = QNN_TRAIN_OUTPUT_NONE;
	qnn_worker_reset_options_defaults(&qnn_worker_reset_options);
}

static void qnn_worker_push_history(const qnn_worker_action_t *action)
{
	if (qnn_worker_runtime.history_count < QNN_WORKER_ACTION_HISTORY)
	{
		qnn_worker_runtime.history[qnn_worker_runtime.history_count] = *action;
		qnn_worker_runtime.history_count += 1;
		return;
	}
	qnn_worker_runtime.history[0] = qnn_worker_runtime.history[1];
	qnn_worker_runtime.history[1] = *action;
}

static void qnn_worker_add_event(
	qnn_worker_snapshot_t *snapshot,
	const char *event_type,
	int region_id,
	int has_delta,
	int delta,
	int has_weapon_id,
	int weapon_id,
	int source_entity_num,
	int target_entity_num)
{
	qnn_worker_event_t *event;

	if (snapshot->event_count >= QNN_WORKER_MAX_EVENTS)
		return;
	event = &snapshot->events[snapshot->event_count];
	memset(event, 0, sizeof(*event));
	snprintf(event->event_type, sizeof(event->event_type), "%s", event_type);
	event->region_id = region_id;
	event->has_delta = has_delta;
	event->delta = delta;
	event->has_weapon_id = has_weapon_id;
	event->weapon_id = weapon_id;
	event->source_entity_num = source_entity_num > 0 ? source_entity_num : 0;
	event->target_entity_num = target_entity_num > 0 ? target_entity_num : 0;
	snapshot->event_count += 1;
}

static void qnn_worker_add_flag_event(qnn_worker_snapshot_t *snapshot, const char *event_type, int region_id)
{
	qnn_worker_add_event(snapshot, event_type, region_id, 0, 0, 0, 0, 0, 0);
}

static void qnn_worker_add_delta_event(qnn_worker_snapshot_t *snapshot, const char *event_type, int region_id, int delta)
{
	qnn_worker_add_event(snapshot, event_type, region_id, 1, delta, 0, 0, 0, 0);
}

static void qnn_worker_add_weapon_event(qnn_worker_snapshot_t *snapshot, const char *event_type, int region_id, int weapon_id)
{
	qnn_worker_add_event(snapshot, event_type, region_id, 0, 0, 1, weapon_id, 0, 0);
}

static void qnn_worker_add_combat_event(
	qnn_worker_snapshot_t *snapshot,
	const char *event_type,
	int region_id,
	int delta,
	int weapon_id,
	int source_entity_num,
	int target_entity_num)
{
	qnn_worker_add_event(snapshot, event_type, region_id, 1, delta, 1, weapon_id, source_entity_num, target_entity_num);
}

static void qnn_worker_add_static_proxy(qnn_worker_snapshot_t *snapshot, const qnn_worker_static_object_t *object)
{
	qnn_worker_visible_entity_t *entity;

	if (snapshot->visible_count >= QNN_WORKER_MAX_VISIBLE)
		return;
	entity = &snapshot->visible[snapshot->visible_count];
	memset(entity, 0, sizeof(*entity));
	entity->entity_key = -((int)(object - qnn_worker_map_state.static_objects) + 1);
	snprintf(entity->entity_id, sizeof(entity->entity_id), "%s", object->object_id);
	snprintf(entity->classname, sizeof(entity->classname), "%s", object->classname);
	entity->entity_num = 0;
	entity->region_id = object->region_id;
	VectorCopy(object->origin, entity->origin);
	VectorCopy(vec3_origin, entity->velocity);
	VectorCopy(object->angles, entity->angles);
	entity->model_id = 0;
	entity->frame = 0;
	entity->static_proxy = true;
	snapshot->visible_count += 1;
}

static void qnn_worker_capture_snapshot(qnn_worker_snapshot_t *snapshot, const qnn_worker_action_t *current_action, qboolean reset_flag)
{
	int entity_num;
	int current_region_id;
	int health;
	int armor;
	int ammo;
	int weapon_id;
	int frags;
	int monster_kills;
	int damage_dealt;
	int hit_count;
	int shots_fired;
	int damage_weapon_id;
	int damage_target_entity;
	int damage_target_entity_num;
	int damage_target_count;
	qboolean goal_reached;

	qnn_worker_capture_base_snapshot(snapshot);

	health = snapshot->health;
	armor = snapshot->armor;
	ammo = snapshot->ammo;
	weapon_id = snapshot->weapon_id;
	current_region_id = snapshot->current_region_id;
	frags = qnn_worker_current_frags();
	monster_kills = qnn_worker_current_monster_kills();
	goal_reached = cl.intermission ? true : false;

	snapshot->goal_reached = goal_reached;
	snapshot->done = false;
	snapshot->done_reason[0] = 0;
	damage_dealt = 0;
	hit_count = 0;
	shots_fired = 0;
	damage_weapon_id = 0;
	damage_target_entity_num = 0;
	damage_target_count = 0;

	if (!reset_flag)
	{
		qboolean fire_window_active;

		if (current_action->fire)
		{
			shots_fired = 1;
			damage_weapon_id = weapon_id > 0 ? weapon_id : qnn_worker_runtime.last_fire_weapon_id;
			qnn_worker_runtime.last_fire_weapon_id = damage_weapon_id;
			qnn_worker_runtime.total_shots_fired += 1;
			qnn_worker_runtime.weapon_shots_fired[qnn_worker_weapon_index(damage_weapon_id)] += 1;
		}

		if (health < qnn_worker_runtime.prev_health)
			qnn_worker_add_delta_event(snapshot, "damage_taken", current_region_id, qnn_worker_runtime.prev_health - health);
		else if (health > qnn_worker_runtime.prev_health)
			qnn_worker_add_delta_event(snapshot, "pickup_health", current_region_id, health - qnn_worker_runtime.prev_health);

		if (armor > qnn_worker_runtime.prev_armor)
			qnn_worker_add_delta_event(snapshot, "pickup_armor", current_region_id, armor - qnn_worker_runtime.prev_armor);
		if (ammo > qnn_worker_runtime.prev_ammo)
			qnn_worker_add_delta_event(snapshot, "pickup_ammo", current_region_id, ammo - qnn_worker_runtime.prev_ammo);
		if (weapon_id > 0 && weapon_id != qnn_worker_runtime.prev_weapon)
			qnn_worker_add_weapon_event(snapshot, "pickup_weapon", current_region_id, weapon_id);
		if (cl.items != qnn_worker_runtime.prev_items
			&& snapshot->event_count == 0
			&& cl.items > qnn_worker_runtime.prev_items)
			qnn_worker_add_flag_event(snapshot, "pickup_item", current_region_id);
		if (frags > qnn_worker_runtime.prev_frags)
			qnn_worker_add_delta_event(snapshot, "frag_gained", current_region_id, frags - qnn_worker_runtime.prev_frags);
		else if (frags < qnn_worker_runtime.prev_frags)
			qnn_worker_add_delta_event(snapshot, "frag_lost", current_region_id, qnn_worker_runtime.prev_frags - frags);
		if (monster_kills > qnn_worker_runtime.prev_monster_kills)
			qnn_worker_add_delta_event(snapshot, "monster_kill", current_region_id, monster_kills - qnn_worker_runtime.prev_monster_kills);

		fire_window_active = current_action->fire || qnn_worker_runtime.recent_fire_steps > 0;
		if (fire_window_active && damage_weapon_id <= 0)
			damage_weapon_id = qnn_worker_runtime.last_fire_weapon_id > 0 ? qnn_worker_runtime.last_fire_weapon_id : weapon_id;
		if (fire_window_active)
		{
			for (entity_num = 0; entity_num < sv.num_edicts && entity_num < MAX_EDICTS; ++entity_num)
			{
				edict_t *edict;
				int prev_health;
				int current_health;

				edict = EDICT_NUM(entity_num);
				current_health = qnn_worker_track_damage_entity(entity_num, edict) ? (int)edict->v.health : 0;
				prev_health = qnn_worker_runtime.prev_entity_active[entity_num] ? qnn_worker_runtime.prev_entity_health[entity_num] : current_health;
				if (prev_health <= current_health)
					continue;
				damage_dealt += prev_health - current_health;
				hit_count += 1;
				damage_target_count += 1;
				damage_target_entity_num = damage_target_count == 1 ? entity_num : -1;
			}
		}
	}

	damage_target_entity = damage_target_count == 1 ? damage_target_entity_num : 0;
	if (damage_dealt > 0)
	{
		qnn_worker_runtime.total_damage_dealt += damage_dealt;
		qnn_worker_runtime.total_hit_count += hit_count;
		qnn_worker_runtime.weapon_damage_dealt[qnn_worker_weapon_index(damage_weapon_id)] += damage_dealt;
		qnn_worker_runtime.weapon_hits_landed[qnn_worker_weapon_index(damage_weapon_id)] += hit_count;
		qnn_worker_add_combat_event(snapshot, "damage_dealt", current_region_id, damage_dealt, damage_weapon_id, cl.viewentity, damage_target_entity);
		qnn_worker_add_combat_event(snapshot, "hit_confirmed", current_region_id, hit_count, damage_weapon_id, cl.viewentity, damage_target_entity);
	}
	if (shots_fired > 0)
		qnn_worker_add_combat_event(snapshot, "shots_fired", current_region_id, shots_fired, damage_weapon_id, cl.viewentity, 0);

	snapshot->damage_dealt = damage_dealt;
	snapshot->hit_count = hit_count;
	snapshot->shots_fired = shots_fired;
	snapshot->damage_weapon_id = damage_weapon_id;

	if (goal_reached && !qnn_worker_runtime.prev_intermission)
		qnn_worker_add_flag_event(snapshot, "goal_reached", current_region_id);
	if (goal_reached)
	{
		snapshot->done = true;
		snprintf(snapshot->done_reason, sizeof(snapshot->done_reason), "goal_reached");
	}
	else if (health <= 0)
	{
		qnn_worker_add_flag_event(snapshot, "player_died", current_region_id);
		snapshot->done = true;
		snprintf(snapshot->done_reason, sizeof(snapshot->done_reason), "player_died");
	}

	qnn_worker_capture_visible_entities(snapshot, qnn_worker_runtime.fixed_dt);

	/* Static proxy fallback: synthesise entities from map objects when
	   the engine reports no visible entities in this frame. */
	if (snapshot->visible_count == 0)
	{
		int i;

		for (i = 0; i < qnn_worker_map_state.static_object_count && snapshot->visible_count < 4; ++i)
		{
			const qnn_worker_static_object_t *object;

			object = &qnn_worker_map_state.static_objects[i];
			if (object->region_id != snapshot->current_region_id && !qnn_worker_is_goal_region(&qnn_worker_map_state, object->region_id))
				continue;
			if (strcmp(object->category, "item")
				&& strcmp(object->category, "goal")
				&& strcmp(object->category, "trigger"))
				continue;
			qnn_worker_add_static_proxy(snapshot, object);
		}
	}

	/* Drain the global sound ring buffer into this snapshot. */
	qnn_worker_drain_sounds(snapshot);
}

static void qnn_worker_commit_snapshot(const qnn_worker_snapshot_t *snapshot, const qnn_worker_action_t *current_action, qboolean reset_flag)
{
	qnn_worker_runtime.prev_health = snapshot->health;
	qnn_worker_runtime.prev_armor = snapshot->armor;
	qnn_worker_runtime.prev_ammo = snapshot->ammo;
	qnn_worker_runtime.prev_weapon = snapshot->weapon_id;
	qnn_worker_runtime.prev_items = cl.items;
	qnn_worker_runtime.prev_intermission = cl.intermission;
	qnn_worker_runtime.prev_frags = qnn_worker_current_frags();
	qnn_worker_runtime.prev_monster_kills = qnn_worker_current_monster_kills();
	qnn_worker_runtime.done = snapshot->done;
	if (current_action->fire)
		qnn_worker_runtime.recent_fire_steps = 2;
	else if (qnn_worker_runtime.recent_fire_steps > 0)
		qnn_worker_runtime.recent_fire_steps -= 1;
	if (snapshot->damage_weapon_id > 0)
		qnn_worker_runtime.last_fire_weapon_id = snapshot->damage_weapon_id;
	qnn_worker_cache_entity_state();
	if (!reset_flag)
		qnn_worker_push_history(current_action);
}

static void qnn_worker_write_events(FILE *out, const qnn_worker_snapshot_t *snapshot)
{
	int i;

	fputc('[', out);
	for (i = 0; i < snapshot->event_count; ++i)
	{
		const qnn_worker_event_t *event;
		char source_id[32];
		char target_id[32];
		int wrote_payload;

		event = &snapshot->events[i];
		qnn_worker_entity_id_string(event->source_entity_num, source_id, sizeof(source_id));
		qnn_worker_entity_id_string(event->target_entity_num, target_id, sizeof(target_id));
		if (i > 0)
			fputc(',', out);
		fprintf(out, "{\"event_type\":");
		qnn_worker_write_json_string(out, event->event_type);
		fprintf(out, ",\"payload\":{");
		wrote_payload = 0;
		if (event->has_delta)
		{
			fprintf(out, "\"delta\":%d", event->delta);
			wrote_payload = 1;
		}
		if (event->has_weapon_id)
		{
			if (wrote_payload)
				fputc(',', out);
			fprintf(out, "\"weapon_id\":%d", event->weapon_id);
		}
		fprintf(out, "},\"region_id\":");
		if (event->region_id >= 0)
			fprintf(out, "%d", event->region_id);
		else
			fputs("null", out);
		fprintf(out, ",\"source_id\":");
		qnn_worker_write_json_string(out, source_id);
		fprintf(out, ",\"target_id\":");
		qnn_worker_write_json_string(out, target_id);
		fputc('}', out);
	}
	fputc(']', out);
}

static void qnn_worker_write_sounds(FILE *out, const qnn_worker_snapshot_t *snapshot)
{
	int i;

	fputc('[', out);
	for (i = 0; i < snapshot->sound_count; ++i)
	{
		const qnn_worker_sound_event_t *snd;

		snd = &snapshot->sounds[i];
		if (i > 0)
			fputc(',', out);
		fprintf(out, "{\"origin\":[%.1f,%.1f,%.1f],\"volume\":%.3f,\"attenuation\":%.3f,\"entity_num\":%d,\"category\":%d,\"name\":",
			snd->origin[0], snd->origin[1], snd->origin[2],
			snd->volume, snd->attenuation,
			snd->entity_num, snd->category);
		qnn_worker_write_json_string(out, snd->name);
		fputc('}', out);
	}
	fputc(']', out);
}

static void qnn_worker_write_visible_entities(FILE *out, const qnn_worker_snapshot_t *snapshot)
{
	int i;

	fputc('[', out);
	for (i = 0; i < snapshot->visible_count; ++i)
	{
		const qnn_worker_visible_entity_t *entity;

		entity = &snapshot->visible[i];
		if (i > 0)
			fputc(',', out);
		fprintf(out, "{\"angles\":[%.1f,%.1f,%.1f],\"classname\":",
			entity->angles[0], entity->angles[1], entity->angles[2]);
		qnn_worker_write_json_string(out, entity->classname);
		fprintf(out, ",\"entity_id\":");
		qnn_worker_write_json_string(out, entity->entity_id);
		fprintf(out, ",\"entity_num\":%d,\"frame\":%d,\"model_id\":%d,\"origin\":[%.1f,%.1f,%.1f],\"properties\":{",
			entity->entity_num,
			entity->frame,
			entity->model_id,
			entity->origin[0],
			entity->origin[1],
			entity->origin[2]);
		if (entity->static_proxy)
		{
			fprintf(out, "\"source\":\"static_proxy\"");
		}
		else
		{
			fprintf(out, "\"effects\":%d,\"frags\":%d,\"health\":%d,\"skin\":%d",
				entity->effects,
				entity->frags,
				entity->health,
				entity->skin);
		}
		fprintf(out, "},\"region_id\":");
		if (entity->region_id >= 0)
			fprintf(out, "%d", entity->region_id);
		else
			fputs("null", out);
		fprintf(out, ",\"velocity\":[%.1f,%.1f,%.1f],\"visible\":true}",
			entity->velocity[0],
			entity->velocity[1],
			entity->velocity[2]);
	}
	fputc(']', out);
}

static void qnn_worker_write_world_tick(FILE *out, const qnn_worker_snapshot_t *snapshot, const qnn_worker_action_t *current_action, qboolean reset_flag)
{
	int i;

	fprintf(out, "{\"action_history\":[");
	for (i = 0; i < qnn_worker_runtime.history_count; ++i)
	{
		if (i > 0)
			fputc(',', out);
		qnn_worker_write_action_json(out, &qnn_worker_runtime.history[i]);
	}
	fprintf(out, "],\"action_label\":");
	qnn_worker_write_action_json(out, current_action);
	fprintf(out, ",\"current_region_id\":%d,\"debug\":{\"client_maxclients\":%d,\"damage_dealt\":%d,\"damage_dealt_total\":%d,\"damage_weapon_id\":%d,\"frags\":%d,\"goal_progress\":%.6f,\"hit_count\":%d,\"hit_count_total\":%d,\"monster_kills\":%d,\"monster_total\":%d,\"player_entity_num\":%d,\"seed\":%d,\"shots_fired\":%d,\"shots_fired_total\":%d,\"weapon_damage_dealt_total\":",
		snapshot->current_region_id,
		cl.maxclients,
		snapshot->damage_dealt,
		qnn_worker_runtime.total_damage_dealt,
		snapshot->damage_weapon_id,
		qnn_worker_current_frags(),
		qnn_worker_goal_progress(&qnn_worker_map_state, snapshot->current_region_id),
		snapshot->hit_count,
		qnn_worker_runtime.total_hit_count,
		qnn_worker_current_monster_kills(),
		qnn_worker_total_monsters(),
		cl.viewentity,
		qnn_worker_runtime.seed,
		snapshot->shots_fired,
		qnn_worker_runtime.total_shots_fired);
	qnn_worker_write_int_array(out, qnn_worker_runtime.weapon_damage_dealt, 9);
	fprintf(out, ",\"weapon_hits_landed_total\":");
	qnn_worker_write_int_array(out, qnn_worker_runtime.weapon_hits_landed, 9);
	fprintf(out, ",\"weapon_shots_fired_total\":");
	qnn_worker_write_int_array(out, qnn_worker_runtime.weapon_shots_fired, 9);
	fprintf(out, ",\"server_players\":");
	qnn_worker_write_server_players(out);
	fprintf(out, "},\"done\":%s,\"done_reason\":",
		snapshot->done ? "true" : "false");
	qnn_worker_write_json_string(out, snapshot->done_reason);
	fprintf(out, ",\"episode_id\":");
	qnn_worker_write_json_string(out, qnn_worker_runtime.episode_id);
	fprintf(out, ",\"events\":");
	qnn_worker_write_events(out, snapshot);
	fprintf(out, ",\"sounds\":");
	qnn_worker_write_sounds(out, snapshot);
	fprintf(out, ",\"map_id\":");
	qnn_worker_write_json_string(out, qnn_worker_map_state.requested_map_id);
	fprintf(out, ",\"player\":{\"ammo\":%d,\"ammo_cells\":%d,\"ammo_nails\":%d,\"ammo_rockets\":%d,\"ammo_shells\":%d,\"armor\":%d,\"armor_type\":%.1f,\"grounded\":%s,\"health\":%d,\"origin\":[%.1f,%.1f,%.1f],\"velocity\":[%.1f,%.1f,%.1f],\"view_angles\":[%.1f,%.1f,%.1f],\"weapon_id\":%d,\"weapons_owned\":%d},\"reset\":%s,\"tick\":%d,\"visible_entities\":",
		snapshot->ammo,
		snapshot->ammo_cells,
		snapshot->ammo_nails,
		snapshot->ammo_rockets,
		snapshot->ammo_shells,
		snapshot->armor,
		snapshot->armor_type,
		snapshot->grounded ? "true" : "false",
		snapshot->health,
		snapshot->player_origin[0],
		snapshot->player_origin[1],
		snapshot->player_origin[2],
		snapshot->player_velocity[0],
		snapshot->player_velocity[1],
		snapshot->player_velocity[2],
		snapshot->player_view_angles[0],
		snapshot->player_view_angles[1],
		snapshot->player_view_angles[2],
		snapshot->weapon_id,
		snapshot->weapons_owned,
		reset_flag ? "true" : "false",
		qnn_worker_runtime.tick);
	qnn_worker_write_visible_entities(out, snapshot);
	fputc('}', out);
}

static qboolean qnn_worker_reset_world(int seed, char *error, size_t error_size)
{
	char command[2048];
	int frame;

	if (qnn_worker_map_state.map_name[0] == 0)
	{
		snprintf(error, error_size, "Call hello first so the worker knows which map to load");
		return false;
	}

	qnn_worker_clear_action(&qnn_worker_pending_action);
	qnn_worker_runtime.seed = seed >= 0 ? seed : 0;
	srand((unsigned int)qnn_worker_runtime.seed);
	DEFAULTnet_hostport = 0;
	net_hostport = 0;
	cls.demonum = -1;
	qnn_worker_runtime.done = false;

	snprintf(command, sizeof(command),
		"disconnect\n"
		"maxplayers %d\n"
		"listen 1\n"
		"skill %d\n"
		"deathmatch %d\n"
		"coop %d\n"
		"teamplay %d\n"
		"fraglimit %d\n"
		"timelimit %d\n"
		"samelevel %d\n",
		qnn_worker_reset_options.maxplayers,
		qnn_worker_reset_options.skill,
		qnn_worker_reset_options.deathmatch,
		qnn_worker_reset_options.coop,
		qnn_worker_reset_options.teamplay,
		qnn_worker_reset_options.fraglimit,
		qnn_worker_reset_options.timelimit,
		qnn_worker_reset_options.samelevel);
	qnn_worker_append_command(command, sizeof(command), qnn_worker_reset_options.pre_map_commands);
	snprintf(command + strlen(command), sizeof(command) - strlen(command),
		"map %s\n",
		qnn_worker_map_state.map_name);
	Cbuf_AddText(command);

	for (frame = 0; frame < 2048; ++frame)
	{
		Host_Frame(qnn_worker_runtime.fixed_dt);
		if (qnn_worker_client_ready())
			break;
	}
	if (!qnn_worker_client_ready())
	{
		snprintf(error, error_size, "Timed out waiting for local client signon on %s", qnn_worker_map_state.map_name);
		return false;
	}
	cl.movemessages = 2;
	if (qnn_worker_reset_options.post_map_commands[0])
	{
		Cbuf_AddText(qnn_worker_reset_options.post_map_commands);
		Cbuf_AddText("\n");
		for (frame = 0; frame < 128; ++frame)
			Host_Frame(qnn_worker_runtime.fixed_dt);
	}

	qnn_worker_runtime.tick = 0;
	qnn_worker_runtime.steps = 0;
	qnn_worker_runtime.history_count = 0;
	qnn_worker_runtime.recent_fire_steps = 0;
	qnn_worker_runtime.last_fire_weapon_id = 0;
	qnn_worker_runtime.total_damage_dealt = 0;
	qnn_worker_runtime.total_hit_count = 0;
	qnn_worker_runtime.total_shots_fired = 0;
	memset(qnn_worker_runtime.weapon_damage_dealt, 0, sizeof(qnn_worker_runtime.weapon_damage_dealt));
	memset(qnn_worker_runtime.weapon_hits_landed, 0, sizeof(qnn_worker_runtime.weapon_hits_landed));
	memset(qnn_worker_runtime.weapon_shots_fired, 0, sizeof(qnn_worker_runtime.weapon_shots_fired));
	memset(qnn_worker_runtime.prev_entity_health, 0, sizeof(qnn_worker_runtime.prev_entity_health));
	memset(qnn_worker_runtime.prev_entity_active, 0, sizeof(qnn_worker_runtime.prev_entity_active));
	qnn_worker_runtime.episode_index += 1;
	qnn_worker_runtime.has_reset = true;
	snprintf(qnn_worker_runtime.episode_id, sizeof(qnn_worker_runtime.episode_id), "%s-%d-%04d",
		qnn_worker_map_state.requested_map_id,
		qnn_worker_runtime.seed,
		qnn_worker_runtime.episode_index);
	qnn_worker_runtime.prev_health = cl.stats[STAT_HEALTH];
	qnn_worker_runtime.prev_armor = cl.stats[STAT_ARMOR];
	qnn_worker_runtime.prev_ammo = cl.stats[STAT_AMMO];
	qnn_worker_runtime.prev_weapon = qnn_worker_weapon_id();
	qnn_worker_runtime.prev_items = cl.items;
	qnn_worker_runtime.prev_intermission = cl.intermission;
	qnn_worker_runtime.prev_frags = qnn_worker_current_frags();
	qnn_worker_runtime.prev_monster_kills = qnn_worker_current_monster_kills();
	qnn_worker_cache_entity_state();
	qnn_worker_semantic_reset(&qnn_worker_map_state);
	qnn_worker_training_reset_episode();
	return true;
}

static void qnn_worker_write_hello_response(void)
{
	fprintf(stdout, "{\"capabilities\":[\"binary_step_v1\",\"listen_local\",\"reset_options\",\"token_step_v2\",\"training_extras_v1\",\"udp_networking\",\"world_tick_only\"],\"map_id\":");
	qnn_worker_write_json_string(stdout, qnn_worker_map_state.requested_map_id);
	fprintf(stdout, ",\"map_state\":");
	qnn_worker_write_map_state_json(stdout, &qnn_worker_map_state);
	fprintf(stdout, ",\"ok\":true,\"protocol_version\":");
	qnn_worker_write_json_string(stdout, QNN_WORKER_PROTOCOL);
	fprintf(stdout, ",\"server\":");
	qnn_worker_write_json_string(stdout, QNN_WORKER_SERVER_NAME);
	fprintf(stdout, ",\"tick_hz\":%d,\"worker_build\":{\"basedir\":", qnn_worker_runtime.fixed_tick_hz);
	qnn_worker_write_json_string(stdout, basedir);
	fprintf(stdout, ",\"upstream_commit\":");
	qnn_worker_write_json_string(stdout, QNN_WORKER_UPSTREAM_COMMIT);
	fprintf(stdout, "}}\n");
	fflush(stdout);
}

static void qnn_worker_write_reset_response(const qnn_worker_snapshot_t *snapshot, const qnn_worker_action_t *action)
{
	if (qnn_worker_runtime.output_mode == QNN_OUTPUT_TOKEN_BINARY_V2)
	{
		qnn_worker_write_token_step_binary(stdout, snapshot, qnn_worker_runtime.tick, qnn_worker_runtime.steps, qnn_worker_runtime.fixed_tick_hz, true);
		if (qnn_worker_runtime.training_output_mode == QNN_TRAIN_OUTPUT_BINARY_V1)
			qnn_worker_write_training_extras_binary(stdout, snapshot, qnn_worker_runtime.tick, qnn_worker_runtime.steps, true);
		return;
	}
	fprintf(stdout, "{\"info\":{\"map_id\":");
	qnn_worker_write_json_string(stdout, qnn_worker_map_state.requested_map_id);
	fprintf(stdout, ",\"deathmatch\":%d,\"maxplayers\":%d,\"seed\":%d,\"teamplay\":%d},\"ok\":true,\"world_tick\":",
		qnn_worker_reset_options.deathmatch,
		qnn_worker_reset_options.maxplayers,
		qnn_worker_runtime.seed,
		qnn_worker_reset_options.teamplay);
	qnn_worker_write_world_tick(stdout, snapshot, action, true);
	fprintf(stdout, "}\n");
	fflush(stdout);
	if (qnn_worker_runtime.training_output_mode == QNN_TRAIN_OUTPUT_BINARY_V1)
		qnn_worker_write_training_extras_binary(stdout, snapshot, qnn_worker_runtime.tick, qnn_worker_runtime.steps, true);
}

static void qnn_worker_write_step_response(const qnn_worker_snapshot_t *snapshot, const qnn_worker_action_t *action)
{
	if (qnn_worker_runtime.output_mode == QNN_OUTPUT_WORLD_BINARY_V1)
	{
		qnn_worker_write_step_binary(stdout, snapshot, action, 0.0f);
		if (qnn_worker_runtime.training_output_mode == QNN_TRAIN_OUTPUT_BINARY_V1)
			qnn_worker_write_training_extras_binary(stdout, snapshot, qnn_worker_runtime.tick, qnn_worker_runtime.steps, false);
		return;
	}
	if (qnn_worker_runtime.output_mode == QNN_OUTPUT_TOKEN_BINARY_V2)
	{
		qnn_worker_write_token_step_binary(stdout, snapshot, qnn_worker_runtime.tick, qnn_worker_runtime.steps, qnn_worker_runtime.fixed_tick_hz, false);
		if (qnn_worker_runtime.training_output_mode == QNN_TRAIN_OUTPUT_BINARY_V1)
			qnn_worker_write_training_extras_binary(stdout, snapshot, qnn_worker_runtime.tick, qnn_worker_runtime.steps, false);
		return;
	}

	fprintf(stdout, "{\"done\":%s,\"info\":{\"deathmatch\":%d,\"goal_reached\":%s,\"maxplayers\":%d,\"seed\":%d,\"steps\":%d,\"teamplay\":%d},\"ok\":true,\"reward\":0.0,\"world_tick\":",
		snapshot->done ? "true" : "false",
		qnn_worker_reset_options.deathmatch,
		snapshot->goal_reached ? "true" : "false",
		qnn_worker_reset_options.maxplayers,
		qnn_worker_runtime.seed,
		qnn_worker_runtime.steps,
		qnn_worker_reset_options.teamplay);
	qnn_worker_write_world_tick(stdout, snapshot, action, false);
	fprintf(stdout, "}\n");
	fflush(stdout);
	if (qnn_worker_runtime.training_output_mode == QNN_TRAIN_OUTPUT_BINARY_V1)
		qnn_worker_write_training_extras_binary(stdout, snapshot, qnn_worker_runtime.tick, qnn_worker_runtime.steps, false);
}

static int qnn_worker_handle_hello(const char *line)
{
	char map_id[QNN_WORKER_MAX_MAP_ID];
	char protocol_version[16];
	char step_format[32];
	char training_format[32];
	char error[256];
	int requested_protocol_version;

	memset(map_id, 0, sizeof(map_id));
	memset(protocol_version, 0, sizeof(protocol_version));
	memset(step_format, 0, sizeof(step_format));
	memset(training_format, 0, sizeof(training_format));
	memset(error, 0, sizeof(error));
	if (!qnn_json_extract_string(line, "\"map_id\"", map_id, sizeof(map_id)))
	{
		snprintf(map_id, sizeof(map_id), "E1M1");
	}
	requested_protocol_version = 3;
	if (qnn_json_extract_string(line, "\"protocol_version\"", protocol_version, sizeof(protocol_version)))
		requested_protocol_version = qnn_worker_parse_protocol_version(protocol_version, 3);
	qnn_worker_runtime.output_mode = QNN_OUTPUT_WORLD_JSON;
	qnn_worker_runtime.training_output_mode = QNN_TRAIN_OUTPUT_NONE;
	if (qnn_json_extract_string(line, "\"step_format\"", step_format, sizeof(step_format)))
	{
		if (!strcmp(step_format, "binary_v1") && requested_protocol_version >= 3)
			qnn_worker_runtime.output_mode = QNN_OUTPUT_WORLD_BINARY_V1;
		else if (!strcmp(step_format, "token_binary_v2") && requested_protocol_version >= 4)
			qnn_worker_runtime.output_mode = QNN_OUTPUT_TOKEN_BINARY_V2;
	}
	if (qnn_json_extract_string(line, "\"training_format\"", training_format, sizeof(training_format)))
	{
		if (!strcmp(training_format, "binary_v1") && requested_protocol_version >= 5)
			qnn_worker_runtime.training_output_mode = QNN_TRAIN_OUTPUT_BINARY_V1;
	}
	qnn_worker_runtime.fixed_tick_hz = qnn_json_extract_int(line, "\"tick_hz\"", qnn_worker_runtime.fixed_tick_hz > 0 ? qnn_worker_runtime.fixed_tick_hz : 20);
	if (qnn_worker_runtime.fixed_tick_hz <= 0)
		qnn_worker_runtime.fixed_tick_hz = 20;
	qnn_worker_runtime.fixed_dt = 1.0f / (float)qnn_worker_runtime.fixed_tick_hz;

	if (!qnn_worker_prepare_map(map_id, error, sizeof(error)))
	{
		qnn_worker_write_error(error);
		return 0;
	}

	qnn_worker_write_hello_response();
	return 0;
}

static int qnn_worker_handle_reset(const char *line)
{
	qnn_worker_snapshot_t snapshot;
	qnn_worker_action_t action;
	char error[256];
	int seed;

	qnn_worker_clear_action(&action);
	memset(error, 0, sizeof(error));
	seed = qnn_json_extract_int(line, "\"seed\"", -1);
	qnn_worker_parse_reset_options(line, &qnn_worker_reset_options);
	if (!qnn_worker_reset_world(seed, error, sizeof(error)))
	{
		qnn_worker_write_error(error);
		return 0;
	}

	qnn_worker_capture_snapshot(&snapshot, &action, true);
	snapshot.action_label = action;
	qnn_worker_semantic_update(&qnn_worker_map_state, &snapshot, qnn_worker_runtime.fixed_dt, true);
	qnn_worker_commit_snapshot(&snapshot, &action, true);
	qnn_worker_write_reset_response(&snapshot, &action);
	return 0;
}

static int qnn_worker_handle_step(const char *line)
{
	qnn_worker_snapshot_t snapshot;
	qnn_worker_action_t action;

	if (!qnn_worker_runtime.has_reset)
	{
		qnn_worker_write_error("Call reset before step");
		return 0;
	}

	qnn_worker_clear_action(&action);
	action.move = qnn_json_extract_int(line, "\"move\"", 0);
	action.strafe = qnn_json_extract_int(line, "\"strafe\"", 0);
	action.look_yaw = qnn_json_extract_int(line, "\"look_yaw\"", QNN_WORKER_LOOK_NEUTRAL_LABEL);
	action.look_pitch = qnn_json_extract_int(line, "\"look_pitch\"", QNN_WORKER_LOOK_NEUTRAL_LABEL);
	action.look_yaw_count = qnn_json_extract_int(line, "\"look_yaw_count\"", 0);
	action.look_pitch_count = qnn_json_extract_int(line, "\"look_pitch_count\"", 0);
	action.fire = qnn_json_extract_int(line, "\"fire\"", 0);
	action.jump = qnn_json_extract_int(line, "\"jump\"", 0);
	action.weapon = qnn_json_extract_int(line, "\"weapon\"", 0);
	qnn_worker_pending_action = action;
	qnn_worker_training_reset_tick();

	if (!qnn_worker_runtime.done)
	{
		Host_Frame(qnn_worker_runtime.fixed_dt);
		qnn_worker_runtime.tick += 1;
		qnn_worker_runtime.steps += 1;
	}
	qnn_worker_capture_snapshot(&snapshot, &action, false);
	snapshot.action_label = action;
	qnn_worker_semantic_update(&qnn_worker_map_state, &snapshot, qnn_worker_runtime.fixed_dt, false);
	qnn_worker_commit_snapshot(&snapshot, &action, false);
	qnn_worker_clear_action(&qnn_worker_pending_action);
	qnn_worker_write_step_response(&snapshot, &action);
	return 0;
}

int main(int argc, char **argv)
{
	quakeparms_t parms;
	char line[QNN_WORKER_MAX_LINE];

	qnn_worker_resolve_basedir(qnn_worker_basedir_storage, sizeof(qnn_worker_basedir_storage));
	qnn_worker_runtime_reset();
	qnn_worker_clear_action(&qnn_worker_pending_action);
	memset(&parms, 0, sizeof(parms));
	COM_InitArgv(argc, argv);
	parms.argc = com_argc;
	parms.argv = com_argv;
	parms.memsize = 32 * 1024 * 1024;
	parms.membase = malloc(parms.memsize);
	parms.basedir = basedir;
	Host_Init(&parms);
	cls.demonum = -1;

	while (fgets(line, sizeof(line), stdin) != NULL)
	{
		if (strstr(line, "\"op\"") != NULL && strstr(line, "hello") != NULL)
		{
			qnn_worker_handle_hello(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "reset") != NULL)
		{
			qnn_worker_handle_reset(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "step") != NULL)
		{
			qnn_worker_handle_step(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "shutdown") != NULL)
		{
			fprintf(stdout, "{\"ok\":true}\n");
			fflush(stdout);
			qnn_worker_free_map_state(&qnn_worker_map_state);
			Host_Shutdown();
			return 0;
		}

		qnn_worker_write_error("unsupported op");
	}

	qnn_worker_free_map_state(&qnn_worker_map_state);
	return 0;
}
