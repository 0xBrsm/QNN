#include "qnn_worker.h"

#include <stdlib.h>
#include <string.h>

#define QNN_WORKER_PROTOCOL "v5"
#define QNN_WORKER_SERVER_NAME "quake-worker"
#define QNN_WORKER_UPSTREAM_COMMIT "bf4ac424ce754894ac8f1dae6a3981954bc9852d"
#define QNN_WORKER_MAX_LINE 8192
#define QNN_WORKER_MAX_EPISODE_ID 128
#define QNN_WORKER_MAX_COMMAND_TEXT 1024

#define QNN_TRAIN_OUTPUT_NONE 0
#define QNN_TRAIN_OUTPUT_BINARY_V1 1

#define QNN_STEP_OUTPUT_TOKEN_BINARY_V2 0
#define QNN_STEP_OUTPUT_OBS_BUFFER_V1   1

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
	int	training_output_mode;
	int	step_output_mode;
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

static qnn_worker_runtime_t qnn_worker_runtime;
static qnn_worker_reset_options_t qnn_worker_reset_options;

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

static void qnn_worker_runtime_reset(void)
{
	memset(&qnn_worker_runtime, 0, sizeof(qnn_worker_runtime));
	qnn_worker_runtime.fixed_tick_hz = 20;
	qnn_worker_runtime.fixed_dt = 1.0f / 20.0f;
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
	qboolean intermission_done;

	qnn_worker_capture_base_snapshot(snapshot);

	health = snapshot->health;
	armor = snapshot->armor;
	ammo = snapshot->ammo;
	weapon_id = snapshot->weapon_id;
	current_region_id = snapshot->current_region_id;
	frags = qnn_worker_current_frags();
	monster_kills = qnn_worker_current_monster_kills();
	intermission_done = cl.intermission ? true : false;

	snapshot->done = false;
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

	if (intermission_done)
	{
		snapshot->done = true;
	}
	else if (health <= 0)
	{
		qnn_worker_add_flag_event(snapshot, "player_died", current_region_id);
		/* In deathmatch, death is part of the reward signal, not a terminal
		   state.  The player respawns and the episode continues until
		   max_steps (Python-side timeout).  Only end on intermission. */
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
			if (object->region_id != snapshot->current_region_id)
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
	/* Run extra frames so the server processes the client "begin" command
	   and QuakeC PutClientInServer executes, which sets self.weapon and
	   other spawn state.  Without this, the player spawns with weapon=0. */
	for (frame = 0; frame < 4; ++frame)
		Host_Frame(qnn_worker_runtime.fixed_dt);

	cl.movemessages = 2;
	if (qnn_worker_reset_options.post_map_commands[0])
	{
		/* Execute post-map commands one line at a time, running settle
		   frames between each line.  This prevents signon conflicts when
		   multiple bots connect via repeated "impulse 100" commands. */
		const char *p = qnn_worker_reset_options.post_map_commands;
		while (*p)
		{
			const char *nl = strchr(p, '\n');
			size_t len = nl ? (size_t)(nl - p) : strlen(p);
			if (len > 0)
			{
				char line[256];
				if (len >= sizeof(line))
					len = sizeof(line) - 1;
				memcpy(line, p, len);
				line[len] = '\0';
				Cbuf_AddText(line);
				Cbuf_AddText("\n");
				for (frame = 0; frame < 32; ++frame)
					Host_Frame(qnn_worker_runtime.fixed_dt);
			}
			p += len;
			if (*p == '\n')
				p++;
		}
	}

	/* Bot spawns via ``impulse 100`` can kill the player (FrikBot
	   side-effect).  If health <= 0 after post_map_commands, restart
	   the map and re-issue the bot connect commands so the player
	   spawns alive with bots present. */
	if (cl.stats[STAT_HEALTH] <= 0)
	{
		Cbuf_AddText("restart\n");
		for (frame = 0; frame < 64; ++frame)
			Host_Frame(qnn_worker_runtime.fixed_dt);
		/* Re-issue post_map_commands (bot connects) after restart. */
		{
			const char *p = qnn_worker_reset_options.post_map_commands;
			while (*p)
			{
				const char *nl = strchr(p, '\n');
				size_t len = nl ? (size_t)(nl - p) : strlen(p);
				if (len > 0)
				{
					char line[256];
					if (len >= sizeof(line))
						len = sizeof(line) - 1;
					memcpy(line, p, len);
					line[len] = '\0';
					Cbuf_AddText(line);
					Cbuf_AddText("\n");
					for (frame = 0; frame < 32; ++frame)
						Host_Frame(qnn_worker_runtime.fixed_dt);
				}
				p += len;
				if (*p == '\n')
					p++;
			}
		}
		/* If still dead after second attempt, accept it. */
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
	qnn_worker_runtime.prev_frags = qnn_worker_current_frags();
	qnn_worker_runtime.prev_monster_kills = qnn_worker_current_monster_kills();
	qnn_worker_cache_entity_state();
	qnn_worker_semantic_reset(&qnn_worker_map_state);
	qnn_worker_training_reset_episode();
	return true;
}

static void qnn_worker_write_hello_response(void)
{
	fprintf(stdout, "{\"capabilities\":[\"listen_local\",\"navmesh_query_v1\",\"obs_buffer_v1\",\"reset_options\",\"token_step_v2\",\"training_extras_v1\",\"udp_networking\"],\"map_id\":");
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

static void qnn_worker_write_reset_response(const qnn_worker_snapshot_t *snapshot)
{
	if (qnn_worker_runtime.step_output_mode == QNN_STEP_OUTPUT_OBS_BUFFER_V1)
		qnn_worker_write_obs_buffer(stdout, snapshot, qnn_worker_runtime.tick, qnn_worker_runtime.steps, qnn_worker_runtime.fixed_tick_hz, true);
	else
		qnn_worker_write_token_step_binary(stdout, snapshot, qnn_worker_runtime.tick, qnn_worker_runtime.steps, qnn_worker_runtime.fixed_tick_hz, true);
	if (qnn_worker_runtime.training_output_mode == QNN_TRAIN_OUTPUT_BINARY_V1)
		qnn_worker_write_training_extras_binary(stdout, snapshot, qnn_worker_runtime.tick, qnn_worker_runtime.steps, true);
}

static void qnn_worker_write_step_response(const qnn_worker_snapshot_t *snapshot)
{
	if (qnn_worker_runtime.step_output_mode == QNN_STEP_OUTPUT_OBS_BUFFER_V1)
		qnn_worker_write_obs_buffer(stdout, snapshot, qnn_worker_runtime.tick, qnn_worker_runtime.steps, qnn_worker_runtime.fixed_tick_hz, false);
	else
		qnn_worker_write_token_step_binary(stdout, snapshot, qnn_worker_runtime.tick, qnn_worker_runtime.steps, qnn_worker_runtime.fixed_tick_hz, false);
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
	requested_protocol_version = 5;
	if (qnn_json_extract_string(line, "\"protocol_version\"", protocol_version, sizeof(protocol_version)))
		requested_protocol_version = qnn_worker_parse_protocol_version(protocol_version, 5);
	qnn_worker_runtime.training_output_mode = QNN_TRAIN_OUTPUT_NONE;
	qnn_worker_runtime.step_output_mode = QNN_STEP_OUTPUT_TOKEN_BINARY_V2;
	if (!qnn_json_extract_string(line, "\"step_format\"", step_format, sizeof(step_format))
		|| requested_protocol_version < 4)
	{
		qnn_worker_write_error("Worker requires step_format with protocol_version>=4");
		return 0;
	}
	if (strcmp(step_format, "obs_buffer_v1") == 0)
	{
		qnn_worker_runtime.step_output_mode = QNN_STEP_OUTPUT_OBS_BUFFER_V1;
	}
	else if (strcmp(step_format, "token_binary_v2") != 0)
	{
		qnn_worker_write_error("Worker requires step_format=token_binary_v2 or obs_buffer_v1");
		return 0;
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
	qnn_worker_write_reset_response(&snapshot);
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
	action.recall[0] = qnn_json_extract_int(line, "\"recall_0\"", 0);
	action.recall[1] = qnn_json_extract_int(line, "\"recall_1\"", 0);
	action.recall[2] = qnn_json_extract_int(line, "\"recall_2\"", 0);
	action.recall[3] = qnn_json_extract_int(line, "\"recall_3\"", 0);
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
	qnn_worker_write_step_response(&snapshot);
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
		if (strstr(line, "\"op\"") != NULL && strstr(line, "nav_query") != NULL)
		{
			qnn_worker_handle_nav_query(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "shutdown") != NULL)
		{
			fprintf(stdout, "{\"ok\":true}\n");
			fflush(stdout);
			qnn_worker_free_map_state(&qnn_worker_map_state);
			CL_Disconnect();
			Host_Shutdown();
			return 0;
		}

		qnn_worker_write_error("unsupported op");
	}

	qnn_worker_free_map_state(&qnn_worker_map_state);
	return 0;
}
