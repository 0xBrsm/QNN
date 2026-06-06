#include "qnn.h"
#include "qnn_fault.h"
#include "qnn_io.h"
#include "qnn_tick.h"

#include <stdlib.h>
#include <string.h>

#define QNN_WORKER_PROTOCOL "v6"
#define QNN_WORKER_SERVER_NAME "quake-worker"
#define QNN_WORKER_UPSTREAM_COMMIT "bf4ac424ce754894ac8f1dae6a3981954bc9852d"
#define QNN_WORKER_MAX_LINE 8192
#define QNN_WORKER_MAX_EPISODE_ID 128
#define QNN_WORKER_MAX_COMMAND_TEXT 1024

#define QNN_TRAIN_OUTPUT_NONE 0
#define QNN_TRAIN_OUTPUT_BINARY_V1 1

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
	char	episode_id[QNN_WORKER_MAX_EPISODE_ID];
	int	training_output_mode;
	int	step_output_mode;
} qnn_runtime_t;

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
	int	train;
	char	pre_map_commands[QNN_WORKER_MAX_COMMAND_TEXT];
	char	post_map_commands[QNN_WORKER_MAX_COMMAND_TEXT];
} qnn_reset_options_t;

static qnn_runtime_t qnn_runtime;
static qnn_reset_options_t qnn_reset_options;

/* Train mode cvar (numeric for QuakeC consumption).  JSON config carries
   a string name which QNN_TrainModeFromName() maps to one of these values. */
#define QNN_TRAIN_OFF    0
#define QNN_TRAIN_ARENA  1  /* bots get full loadout via CheatCommand */
static cvar_t qnn_train_cvar = {"qnn_train", "0", false, false};

static int QNN_TrainModeFromName(const char *name)
{
	if (name == NULL || name[0] == 0) return QNN_TRAIN_OFF;
	if (!strcmp(name, "arena"))       return QNN_TRAIN_ARENA;
	return QNN_TRAIN_OFF;
}
/* Inventory cvars: set by C worker from scenario.json, read by QuakeC PlayerPreThink. */
static cvar_t qnn_inv_weapons_cvar    = {"qnn_inv_weapons",    "0", false, false};
static cvar_t qnn_inv_shells_cvar     = {"qnn_inv_shells",     "0", false, false};
static cvar_t qnn_inv_nails_cvar      = {"qnn_inv_nails",      "0", false, false};
static cvar_t qnn_inv_rockets_cvar    = {"qnn_inv_rockets",    "0", false, false};
static cvar_t qnn_inv_cells_cvar      = {"qnn_inv_cells",      "0", false, false};
static cvar_t qnn_inv_armor_cvar      = {"qnn_inv_armor",      "0", false, false};
static cvar_t qnn_inv_armor_type_cvar = {"qnn_inv_armor_type", "0", false, false};
static cvar_t qnn_inv_health_cvar     = {"qnn_inv_health",     "0", false, false};
static cvar_t qnn_inv_powerups_cvar   = {"qnn_inv_powerups",   "0", false, false};
static cvar_t qnn_inv_selected_cvar   = {"qnn_inv_selected",   "0", false, false};

static qboolean QNN_ClientReady(void)
{
	return (sv.active
		&& cls.state == ca_connected
		&& cls.signon == SIGNONS
		&& cl.worldmodel != NULL
		&& cl.viewentity > 0
		&& cl.viewentity < MAX_EDICTS) ? true : false;
}

static int QNN_CurrentMonsterKills(void)
{
	return cl.stats[STAT_MONSTERS];
}

static void QNN_ResetOptionsDefaults(qnn_reset_options_t *options)
{
	memset(options, 0, sizeof(*options));
	options->maxplayers = 1;
	options->skill = 0;
	options->deathmatch = 1;
	options->coop = 0;
	options->teamplay = 0;
	options->fraglimit = 0;
	options->timelimit = 0;
	options->samelevel = 1;
	options->train = 0;
}

/* Weapon name → IT_ bitmask. */
static int QNN_WeaponBit(const char *name)
{
	if (!strcmp(name, "axe"))              return 4096;
	if (!strcmp(name, "shotgun"))          return 1;
	if (!strcmp(name, "super_shotgun"))    return 2;
	if (!strcmp(name, "nailgun"))          return 4;
	if (!strcmp(name, "super_nailgun"))    return 8;
	if (!strcmp(name, "grenade_launcher")) return 16;
	if (!strcmp(name, "rocket_launcher"))  return 32;
	if (!strcmp(name, "lightning"))        return 64;
	return 0;
}

/* Powerup name → IT_ bitmask. */
static int QNN_PowerupBit(const char *name)
{
	if (!strcmp(name, "quad"))             return 4194304;
	if (!strcmp(name, "pentagram"))        return 1048576;
	if (!strcmp(name, "ring"))             return 524288;
	if (!strcmp(name, "envirosuit"))       return 2097152;
	return 0;
}

static void QNN_CvarSetInt(const char *name, int value)
{
	char buf[32];
	snprintf(buf, sizeof(buf), "%d", value);
	Cvar_Set(name, buf);
}

static void QNN_CvarSetFloat(const char *name, float value)
{
	char buf[32];
	snprintf(buf, sizeof(buf), "%g", value);
	Cvar_Set(name, buf);
}

/* Parse "inventory" from reset JSON and set qnn_inv_* cvars.
   Format: {"inventory": {"weapons": ["axe","lightning"], "ammo": {"shells":100,...},
            "armor": {"value":200,"type":0.8}, "health":100, "powerups": ["quad"]}} */
static void QNN_ParseInventory(const char *line)
{
	int weapons = 0;
	int powerups = 0;

	/* Default: no custom inventory (cvars = 0 → QC fallback to train-mode behavior) */
	QNN_CvarSetInt("qnn_inv_weapons", 0);
	QNN_CvarSetInt("qnn_inv_shells", 0);
	QNN_CvarSetInt("qnn_inv_nails", 0);
	QNN_CvarSetInt("qnn_inv_rockets", 0);
	QNN_CvarSetInt("qnn_inv_cells", 0);
	QNN_CvarSetInt("qnn_inv_armor", 0);
	QNN_CvarSetFloat("qnn_inv_armor_type", 0.0f);
	QNN_CvarSetInt("qnn_inv_health", 0);
	QNN_CvarSetInt("qnn_inv_powerups", 0);
	QNN_CvarSetInt("qnn_inv_selected", 0);

	if (strstr(line, "\"inventory\"") == NULL)
		return;

	/* Weapons: scan for known weapon names in the JSON string. */
	{
		static const char *weapon_names[] = {
			"axe", "shotgun", "super_shotgun", "nailgun", "super_nailgun",
			"grenade_launcher", "rocket_launcher", "lightning", NULL
		};
		int i;
		for (i = 0; weapon_names[i]; ++i)
		{
			char needle[64];
			snprintf(needle, sizeof(needle), "\"%s\"", weapon_names[i]);
			if (strstr(line, needle))
				weapons |= QNN_WeaponBit(weapon_names[i]);
		}
	}
	QNN_CvarSetInt("qnn_inv_weapons", weapons);

	/* Ammo */
	QNN_CvarSetInt("qnn_inv_shells", QNN_JsonExtractInt(line, "\"shells\"", 0));
	QNN_CvarSetInt("qnn_inv_nails", QNN_JsonExtractInt(line, "\"nails\"", 0));
	QNN_CvarSetInt("qnn_inv_rockets", QNN_JsonExtractInt(line, "\"rockets\"", 0));
	QNN_CvarSetInt("qnn_inv_cells", QNN_JsonExtractInt(line, "\"cells\"", 0));

	/* Armor */
	QNN_CvarSetInt("qnn_inv_armor", QNN_JsonExtractInt(line, "\"armor_value\"", 0));
	QNN_CvarSetFloat("qnn_inv_armor_type", QNN_JsonExtractFloat(line, "\"armor_type\"", 0.0f));

	/* Health */
	QNN_CvarSetInt("qnn_inv_health", QNN_JsonExtractInt(line, "\"health\"", 0));

	/* Powerups */
	{
		static const char *powerup_names[] = {"quad", "pentagram", "ring", "envirosuit", NULL};
		int i;
		for (i = 0; powerup_names[i]; ++i)
		{
			char needle[64];
			snprintf(needle, sizeof(needle), "\"%s\"", powerup_names[i]);
			if (strstr(line, needle))
				powerups |= QNN_PowerupBit(powerup_names[i]);
		}
	}
	QNN_CvarSetInt("qnn_inv_powerups", powerups);

	/* Selected weapon: which IT_* bit the model wields at every spawn.
	   Read by QC PutClientInServer's inv_weapons branch on every respawn. */
	{
		char selected[32] = {0};
		if (QNN_JsonExtractString(line, "\"selected_weapon\"", selected, sizeof(selected)))
			QNN_CvarSetInt("qnn_inv_selected", QNN_WeaponBit(selected));
	}
}

static void QNN_ParseResetOptions(const char *line, qnn_reset_options_t *options)
{
	QNN_ResetOptionsDefaults(options);
	options->maxplayers = QNN_JsonExtractInt(line, "\"maxplayers\"", options->maxplayers);
	if (options->maxplayers < 1)
		options->maxplayers = 1;
	options->skill = QNN_JsonExtractInt(line, "\"skill\"", options->skill);
	options->deathmatch = QNN_JsonExtractInt(line, "\"deathmatch\"", options->deathmatch);
	options->coop = QNN_JsonExtractInt(line, "\"coop\"", options->coop);
	options->teamplay = QNN_JsonExtractInt(line, "\"teamplay\"", options->teamplay);
	options->fraglimit = QNN_JsonExtractInt(line, "\"fraglimit\"", options->fraglimit);
	options->timelimit = QNN_JsonExtractInt(line, "\"timelimit\"", options->timelimit);
	options->samelevel = QNN_JsonExtractInt(line, "\"samelevel\"", options->samelevel);
	{
		char train_name[32] = {0};
		if (QNN_JsonExtractString(line, "\"train\"", train_name, sizeof(train_name)))
			options->train = QNN_TrainModeFromName(train_name);
	}
	QNN_JsonExtractString(line, "\"pre_map_commands\"", options->pre_map_commands, sizeof(options->pre_map_commands));
	QNN_JsonExtractString(line, "\"post_map_commands\"", options->post_map_commands, sizeof(options->post_map_commands));

	/* Parse inventory section and set qnn_inv_* cvars for QuakeC. */
	QNN_ParseInventory(line);

}

static void QNN_AppendCommand(char *buffer, size_t buffer_size, const char *command_text)
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

/* QNN_WeaponIndex removed — inline QNN_Clamp(id, 0, 8) at call sites */

static qboolean QNN_TrackDamageEntity(int entity_num, const edict_t *edict)
{
	const char *classname;

	if (!edict || edict->free || entity_num == cl.viewentity || !edict->v.classname)
		return false;
	classname = QNN_ProgString(edict->v.classname);
	if (!strcmp(classname, "player"))
		return true;
	if (!strncmp(classname, "monster_", 8))
		return true;
	return false;
}

static void QNN_CacheEntityState(void)
{
	int entity_num;

	memset(qnn_runtime.prev_entity_health, 0, sizeof(qnn_runtime.prev_entity_health));
	memset(qnn_runtime.prev_entity_active, 0, sizeof(qnn_runtime.prev_entity_active));
	if (!sv.active)
		return;
	for (entity_num = 0; entity_num < sv.num_edicts && entity_num < MAX_EDICTS; ++entity_num)
	{
		edict_t *edict;

		edict = EDICT_NUM(entity_num);
		if (!QNN_TrackDamageEntity(entity_num, edict))
			continue;
		qnn_runtime.prev_entity_health[entity_num] = (int)edict->v.health;
		qnn_runtime.prev_entity_active[entity_num] = true;
	}
}

static int QNN_ParseProtocolVersion(const char *value, int fallback)
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

static void QNN_RuntimeReset(void)
{
	memset(&qnn_runtime, 0, sizeof(qnn_runtime));
	qnn_runtime.fixed_tick_hz = 20;
	qnn_runtime.fixed_dt = 1.0f / 20.0f;
	qnn_runtime.training_output_mode = QNN_TRAIN_OUTPUT_NONE;
	QNN_ResetOptionsDefaults(&qnn_reset_options);
}

static void QNN_AddEvent(
	qnn_snapshot_t *snapshot,
	const char *event_type,
	int region_id,
	int has_delta,
	int delta,
	int has_weapon_id,
	int weapon_id,
	int source_entity_num,
	int target_entity_num)
{
	qnn_event_t *event;

	if (snapshot->event_count >= QNN_MAX_EVENTS)
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

static void QNN_AddFlagEvent(qnn_snapshot_t *snapshot, const char *event_type, int region_id)
{
	QNN_AddEvent(snapshot, event_type, region_id, 0, 0, 0, 0, 0, 0);
}

static void QNN_AddDeltaEvent(qnn_snapshot_t *snapshot, const char *event_type, int region_id, int delta)
{
	QNN_AddEvent(snapshot, event_type, region_id, 1, delta, 0, 0, 0, 0);
}

static void QNN_AddWeaponEvent(qnn_snapshot_t *snapshot, const char *event_type, int region_id, int weapon_id)
{
	QNN_AddEvent(snapshot, event_type, region_id, 0, 0, 1, weapon_id, 0, 0);
}

static void QNN_AddCombatEvent(
	qnn_snapshot_t *snapshot,
	const char *event_type,
	int region_id,
	int delta,
	int weapon_id,
	int source_entity_num,
	int target_entity_num)
{
	QNN_AddEvent(snapshot, event_type, region_id, 1, delta, 1, weapon_id, source_entity_num, target_entity_num);
}

static void QNN_CaptureSnapshotLocal(qnn_snapshot_t *snapshot, const qnn_action_t *current_action, qboolean reset_flag)
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

	QNN_CaptureBaseSnapshot(snapshot);

	health = snapshot->health;
	armor = snapshot->armor;
	ammo = snapshot->ammo;
	weapon_id = snapshot->weapon_id;
	current_region_id = snapshot->current_region_id;
	frags = QNN_CurrentFrags();
	monster_kills = QNN_CurrentMonsterKills();
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

		if (QNN_ActionAttack(current_action->move))
		{
			shots_fired = 1;
			damage_weapon_id = weapon_id > 0 ? weapon_id : qnn_runtime.last_fire_weapon_id;
			qnn_runtime.last_fire_weapon_id = damage_weapon_id;
			qnn_runtime.total_shots_fired += 1;
			qnn_runtime.weapon_shots_fired[QNN_Clamp(damage_weapon_id, 0, 8)] += 1;
		}

		if (health < qnn_runtime.prev_health)
			QNN_AddDeltaEvent(snapshot, "damage_taken", current_region_id, qnn_runtime.prev_health - health);
		else if (health > qnn_runtime.prev_health)
			QNN_AddDeltaEvent(snapshot, "pickup_health", current_region_id, health - qnn_runtime.prev_health);

		if (armor > qnn_runtime.prev_armor)
			QNN_AddDeltaEvent(snapshot, "pickup_armor", current_region_id, armor - qnn_runtime.prev_armor);
		if (ammo > qnn_runtime.prev_ammo)
			QNN_AddDeltaEvent(snapshot, "pickup_ammo", current_region_id, ammo - qnn_runtime.prev_ammo);
		if (weapon_id > 0 && weapon_id != qnn_runtime.prev_weapon)
			QNN_AddWeaponEvent(snapshot, "pickup_weapon", current_region_id, weapon_id);
		if (cl.items != qnn_runtime.prev_items
			&& snapshot->event_count == 0
			&& cl.items > qnn_runtime.prev_items)
			QNN_AddFlagEvent(snapshot, "pickup_item", current_region_id);
		if (frags > qnn_runtime.prev_frags)
			QNN_AddDeltaEvent(snapshot, "frag_gained", current_region_id, frags - qnn_runtime.prev_frags);
		else if (frags < qnn_runtime.prev_frags)
			QNN_AddDeltaEvent(snapshot, "frag_lost", current_region_id, qnn_runtime.prev_frags - frags);
		if (monster_kills > qnn_runtime.prev_monster_kills)
			QNN_AddDeltaEvent(snapshot, "monster_kill", current_region_id, monster_kills - qnn_runtime.prev_monster_kills);

		fire_window_active = QNN_ActionAttack(current_action->move) || qnn_runtime.recent_fire_steps > 0;
		if (fire_window_active && damage_weapon_id <= 0)
			damage_weapon_id = qnn_runtime.last_fire_weapon_id > 0 ? qnn_runtime.last_fire_weapon_id : weapon_id;
		if (fire_window_active)
		{
			for (entity_num = 0; entity_num < sv.num_edicts && entity_num < MAX_EDICTS; ++entity_num)
			{
				edict_t *edict;
				int prev_health;
				int current_health;

				edict = EDICT_NUM(entity_num);
				current_health = QNN_TrackDamageEntity(entity_num, edict) ? (int)edict->v.health : 0;
				prev_health = qnn_runtime.prev_entity_active[entity_num] ? qnn_runtime.prev_entity_health[entity_num] : current_health;
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
		qnn_runtime.total_damage_dealt += damage_dealt;
		qnn_runtime.total_hit_count += hit_count;
		qnn_runtime.weapon_damage_dealt[QNN_Clamp(damage_weapon_id, 0, 8)] += damage_dealt;
		qnn_runtime.weapon_hits_landed[QNN_Clamp(damage_weapon_id, 0, 8)] += hit_count;
		QNN_AddCombatEvent(snapshot, "damage_dealt", current_region_id, damage_dealt, damage_weapon_id, cl.viewentity, damage_target_entity);
		QNN_AddCombatEvent(snapshot, "hit_confirmed", current_region_id, hit_count, damage_weapon_id, cl.viewentity, damage_target_entity);
	}
	if (shots_fired > 0)
		QNN_AddCombatEvent(snapshot, "shots_fired", current_region_id, shots_fired, damage_weapon_id, cl.viewentity, 0);

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
		QNN_AddFlagEvent(snapshot, "player_died", current_region_id);
		/* In deathmatch, death is part of the reward signal, not a terminal
		   state.  The player respawns and the episode continues until
		   max_steps (Python-side timeout).  Only end on intermission. */
	}

	/* Drain the global sound ring buffer into this snapshot. */
	QNN_DrainSounds(snapshot);
}

static void QNN_CommitSnapshot(const qnn_snapshot_t *snapshot, const qnn_action_t *current_action, qboolean reset_flag)
{
	qnn_runtime.prev_health = snapshot->health;
	qnn_runtime.prev_armor = snapshot->armor;
	qnn_runtime.prev_ammo = snapshot->ammo;
	qnn_runtime.prev_weapon = snapshot->weapon_id;
	qnn_runtime.prev_items = cl.items;
	qnn_runtime.prev_frags = QNN_CurrentFrags();
	qnn_runtime.prev_monster_kills = QNN_CurrentMonsterKills();
	qnn_runtime.done = snapshot->done;
	if (QNN_ActionAttack(current_action->move))
		qnn_runtime.recent_fire_steps = 2;
	else if (qnn_runtime.recent_fire_steps > 0)
		qnn_runtime.recent_fire_steps -= 1;
	if (snapshot->damage_weapon_id > 0)
		qnn_runtime.last_fire_weapon_id = snapshot->damage_weapon_id;
	QNN_CacheEntityState();
}

static qboolean QNN_ResetWorldLocal(int seed, char *error, size_t error_size)
{
	char command[2048];
	int frame;

	if (qnn_map_state.map_name[0] == 0)
	{
		snprintf(error, error_size, "Call hello first so the worker knows which map to load");
		return false;
	}

	QNN_ClearAction(&qnn_pending_action);
	qnn_runtime.seed = seed >= 0 ? seed : 0;
	srand((unsigned int)qnn_runtime.seed);
	DEFAULTnet_hostport = 0;
	net_hostport = 0;
	cls.demonum = -1;
	qnn_runtime.done = false;

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
		qnn_reset_options.maxplayers,
		qnn_reset_options.skill,
		qnn_reset_options.deathmatch,
		qnn_reset_options.coop,
		qnn_reset_options.teamplay,
		qnn_reset_options.fraglimit,
		qnn_reset_options.timelimit,
		qnn_reset_options.samelevel);
	/* Publish train mode so QuakeC PlayerPreThink can branch on it.
	   Cvar persists across map changes (unlike console commands in pre_map). */
	QNN_CvarSetInt("qnn_train", qnn_reset_options.train);

	QNN_AppendCommand(command, sizeof(command), qnn_reset_options.pre_map_commands);
	snprintf(command + strlen(command), sizeof(command) - strlen(command),
		"map %s\n",
		qnn_map_state.map_name);
	Cbuf_AddText(command);

	for (frame = 0; frame < 2048; ++frame)
	{
		Host_Frame(qnn_runtime.fixed_dt);
		if (QNN_ClientReady())
			break;
	}
	if (!QNN_ClientReady())
	{
		snprintf(error, error_size, "Timed out waiting for local client signon on %s", qnn_map_state.map_name);
		return false;
	}
	/* Run extra frames so the server processes the client "begin" command
	   and QuakeC PutClientInServer executes, which sets self.weapon and
	   other spawn state.  Without this, the player spawns with weapon=0. */
	for (frame = 0; frame < 4; ++frame)
		Host_Frame(qnn_runtime.fixed_dt);

	cl.movemessages = 2;
	if (qnn_reset_options.post_map_commands[0])
	{
		/* Execute post-map commands one line at a time, running settle
		   frames between each line.  This prevents signon conflicts when
		   multiple bots connect via repeated "impulse 100" commands. */
		const char *p = qnn_reset_options.post_map_commands;
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
					Host_Frame(qnn_runtime.fixed_dt);
			}
			p += len;
			if (*p == '\n')
				p++;
		}
	}

	/* Bot spawns via ``impulse 100`` can telefrag the player.
	   Just let the player respawn naturally — bots are already
	   connected and demo recording continues uninterrupted. */

	qnn_runtime.tick = 0;
	qnn_runtime.steps = 0;
	qnn_runtime.recent_fire_steps = 0;
	qnn_runtime.last_fire_weapon_id = 0;
	qnn_runtime.total_damage_dealt = 0;
	qnn_runtime.total_hit_count = 0;
	qnn_runtime.total_shots_fired = 0;
	memset(qnn_runtime.weapon_damage_dealt, 0, sizeof(qnn_runtime.weapon_damage_dealt));
	memset(qnn_runtime.weapon_hits_landed, 0, sizeof(qnn_runtime.weapon_hits_landed));
	memset(qnn_runtime.weapon_shots_fired, 0, sizeof(qnn_runtime.weapon_shots_fired));
	memset(qnn_runtime.prev_entity_health, 0, sizeof(qnn_runtime.prev_entity_health));
	memset(qnn_runtime.prev_entity_active, 0, sizeof(qnn_runtime.prev_entity_active));
	qnn_runtime.episode_index += 1;
	qnn_runtime.has_reset = true;
	snprintf(qnn_runtime.episode_id, sizeof(qnn_runtime.episode_id), "%s-%d-%04d",
		qnn_map_state.requested_map_id,
		qnn_runtime.seed,
		qnn_runtime.episode_index);
	qnn_runtime.prev_health = cl.stats[STAT_HEALTH];
	qnn_runtime.prev_armor = cl.stats[STAT_ARMOR];
	qnn_runtime.prev_ammo = cl.stats[STAT_AMMO];
	qnn_runtime.prev_weapon = QNN_WeaponId();
	qnn_runtime.prev_items = cl.items;
	qnn_runtime.prev_frags = QNN_CurrentFrags();
	qnn_runtime.prev_monster_kills = QNN_CurrentMonsterKills();
	QNN_CacheEntityState();
	QNN_IOInit(&qnn_map_state);
	QNN_TrainingResetEpisode();
	return true;
}

static void QNN_WriteHelloResponse(void)
{
	fprintf(stdout, "{\"capabilities\":[\"binary_step_v1\",\"listen_local\",\"navmesh_query_v1\",\"obs_buffer_v1\",\"reset_options\",\"training_extras_v1\",\"udp_networking\"],\"map_id\":");
	QNN_WriteJsonString(stdout, qnn_map_state.requested_map_id);
	fprintf(stdout, ",\"ok\":true,\"protocol_version\":");
	QNN_WriteJsonString(stdout, QNN_WORKER_PROTOCOL);
	fprintf(stdout, ",\"server\":");
	QNN_WriteJsonString(stdout, QNN_WORKER_SERVER_NAME);
	fprintf(stdout, ",\"tick_hz\":%d,\"worker_build\":{\"basedir\":", qnn_runtime.fixed_tick_hz);
	QNN_WriteJsonString(stdout, basedir);
	fprintf(stdout, ",\"upstream_commit\":");
	QNN_WriteJsonString(stdout, QNN_WORKER_UPSTREAM_COMMIT);
	fprintf(stdout, "}}\n");
	fflush(stdout);
}

static void QNN_WriteObsToStdout(const qnn_snapshot_t *snapshot)
{
	static uint8_t obs[QNN_OBS_BUFFER_SIZE];
	qnn_tick_result_t result;
	QNN_IOEmit(snapshot, &result);
	QNN_IOPackObsBuffer(obs, &result);
	fwrite(obs, 1, QNN_OBS_BUFFER_SIZE, stdout);
	fflush(stdout);
}

static void QNN_WriteResetResponse(const qnn_snapshot_t *snapshot)
{
	QNN_WriteObsToStdout(snapshot);
	if (qnn_runtime.training_output_mode == QNN_TRAIN_OUTPUT_BINARY_V1)
		QNN_WriteTrainingExtrasBinary(stdout, snapshot, qnn_runtime.tick, qnn_runtime.steps, true);
}

static void QNN_WriteStepResponse(const qnn_snapshot_t *snapshot)
{
	QNN_WriteObsToStdout(snapshot);
	if (qnn_runtime.training_output_mode == QNN_TRAIN_OUTPUT_BINARY_V1)
		QNN_WriteTrainingExtrasBinary(stdout, snapshot, qnn_runtime.tick, qnn_runtime.steps, false);
}

static int QNN_HandleHello(const char *line)
{
	char map_id[QNN_MAX_MAP_ID];
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
	if (!QNN_JsonExtractString(line, "\"map_id\"", map_id, sizeof(map_id)))
	{
		snprintf(map_id, sizeof(map_id), "E1M1");
	}
	requested_protocol_version = 5;
	if (QNN_JsonExtractString(line, "\"protocol_version\"", protocol_version, sizeof(protocol_version)))
		requested_protocol_version = QNN_ParseProtocolVersion(protocol_version, 5);
	qnn_runtime.training_output_mode = QNN_TRAIN_OUTPUT_NONE;
	qnn_runtime.step_output_mode = QNN_STEP_OUTPUT_OBS_BUFFER_V1;
	if (!QNN_JsonExtractString(line, "\"step_format\"", step_format, sizeof(step_format))
		|| requested_protocol_version < 4)
	{
		QNN_WriteError("Worker requires step_format=obs_buffer_v1 with protocol_version>=4");
		return 0;
	}
	if (strcmp(step_format, "obs_buffer_v1") != 0)
	{
		QNN_WriteError("Worker requires step_format=obs_buffer_v1");
		return 0;
	}
	if (QNN_JsonExtractString(line, "\"training_format\"", training_format, sizeof(training_format)))
	{
		if (!strcmp(training_format, "binary_v1") && requested_protocol_version >= 5)
			qnn_runtime.training_output_mode = QNN_TRAIN_OUTPUT_BINARY_V1;
	}
	qnn_runtime.fixed_tick_hz = QNN_JsonExtractInt(line, "\"tick_hz\"", qnn_runtime.fixed_tick_hz > 0 ? qnn_runtime.fixed_tick_hz : 20);
	if (qnn_runtime.fixed_tick_hz <= 0)
		qnn_runtime.fixed_tick_hz = 20;
	qnn_runtime.fixed_dt = 1.0f / (float)qnn_runtime.fixed_tick_hz;
	Cvar_SetValue("qnn_tick_hz", (float)qnn_runtime.fixed_tick_hz);

	if (!QNN_PrepareMap(map_id, error, sizeof(error)))
	{
		QNN_WriteError(error);
		return 0;
	}

	QNN_WriteHelloResponse();
	return 0;
}

static int QNN_HandleReset(const char *line)
{
	qnn_snapshot_t snapshot;
	qnn_action_t action;
	char error[256];
	int seed;

	QNN_ClearAction(&action);
	memset(error, 0, sizeof(error));
	seed = QNN_JsonExtractInt(line, "\"seed\"", -1);
	QNN_ParseResetOptions(line, &qnn_reset_options);
	QNN_TrainingParseRewardWeights(line);
	if (!QNN_ResetWorldLocal(seed, error, sizeof(error)))
	{
		QNN_WriteError(error);
		return 0;
	}

	/* Rebuild navmesh/route if worldmodel changed (first reset after hello,
	   or map change).  QNN_PrepareMap detects the change via cached_worldmodel. */
	if (!QNN_PrepareMap(qnn_map_state.requested_map_id, error, sizeof(error)))
	{
		QNN_WriteError(error);
		return 0;
	}

	/* Tag subsequent crashes with the map name so parallel-worker
	 * stack traces can be grouped by scenario. */
	QNN_FaultSetContext(qnn_map_state.requested_map_id);

	QNN_CaptureSnapshotLocal(&snapshot, &action, true);
	snapshot.action_label = action;
	QNN_IOUpdate(&snapshot, qnn_runtime.fixed_dt, true);
	QNN_CommitSnapshot(&snapshot, &action, true);
	QNN_WriteResetResponse(&snapshot);
	return 0;
}

static int QNN_HandleStep(const char *line)
{
	qnn_snapshot_t snapshot;
	qnn_action_t action;

	if (!qnn_runtime.has_reset)
	{
		QNN_WriteError("Call reset before step");
		return 0;
	}

	QNN_ClearAction(&action);
	{
		vec3_t move_vec = {0.0f, 0.0f, 0.0f};
		int attack_press;
		int jump_press;
		int fb_neg, fb_pos, lr_neg, lr_pos, up_neg, up_pos;
		QNN_JsonExtractVec3(line, "\"move\"", move_vec);
		QNN_JsonExtractVec3(line, "\"look\"", action.look);
		attack_press = QNN_JsonExtractInt(line, "\"attack\"", 0) ? 1 : 0;
		jump_press = QNN_JsonExtractInt(line, "\"jump\"", 0) ? 1 : 0;
		action.weapon = (uint8_t)QNN_JsonExtractInt(line, "\"weapon\"", 0);
		fb_neg = (move_vec[0] < -QNN_SNAP_THRESHOLD) ? 1 : 0;
		fb_pos = (move_vec[0] >  QNN_SNAP_THRESHOLD) ? 1 : 0;
		lr_neg = (move_vec[1] < -QNN_SNAP_THRESHOLD) ? 1 : 0;
		lr_pos = (move_vec[1] >  QNN_SNAP_THRESHOLD) ? 1 : 0;
		up_neg = (move_vec[2] < -QNN_SNAP_THRESHOLD) ? 1 : 0;
		up_pos = (move_vec[2] >  QNN_SNAP_THRESHOLD) ? 1 : 0;
		action.move = QNN_PackInputMask(
			/*alive=*/1,
			fb_neg, fb_pos, lr_neg, lr_pos,
			up_neg, up_pos,
			jump_press, attack_press);
	}
	qnn_pending_action = action;
	QNN_TrainingResetTick();

	if (!qnn_runtime.done)
	{
		Host_Frame(qnn_runtime.fixed_dt);
		qnn_runtime.tick += 1;
		qnn_runtime.steps += 1;
	}

	QNN_CaptureSnapshotLocal(&snapshot, &action, false);
	snapshot.action_label = action;
	QNN_IOUpdate(&snapshot, qnn_runtime.fixed_dt, false);
	QNN_CommitSnapshot(&snapshot, &action, false);
	QNN_ClearAction(&qnn_pending_action);
	QNN_WriteStepResponse(&snapshot);
	return 0;
}

int main(int argc, char **argv)
{
	quakeparms_t parms;
	char line[QNN_WORKER_MAX_LINE];

	QNN_FaultInit("ppo_worker");
	QNN_ResolveBasedir(qnn_basedir_storage, sizeof(qnn_basedir_storage));
	QNN_RuntimeReset();
	QNN_ClearAction(&qnn_pending_action);
	memset(&parms, 0, sizeof(parms));
	COM_InitArgv(argc, argv);
	parms.argc = com_argc;
	parms.argv = com_argv;
	parms.memsize = 32 * 1024 * 1024;
	parms.membase = malloc(parms.memsize);
	parms.basedir = basedir;
	Host_Init(&parms);
	QNN_TickRegister();
	Cvar_RegisterVariable(&qnn_train_cvar);
	Cvar_RegisterVariable(&qnn_inv_weapons_cvar);
	Cvar_RegisterVariable(&qnn_inv_shells_cvar);
	Cvar_RegisterVariable(&qnn_inv_nails_cvar);
	Cvar_RegisterVariable(&qnn_inv_rockets_cvar);
	Cvar_RegisterVariable(&qnn_inv_cells_cvar);
	Cvar_RegisterVariable(&qnn_inv_armor_cvar);
	Cvar_RegisterVariable(&qnn_inv_armor_type_cvar);
	Cvar_RegisterVariable(&qnn_inv_health_cvar);
	Cvar_RegisterVariable(&qnn_inv_powerups_cvar);
	Cvar_RegisterVariable(&qnn_inv_selected_cvar);
	cls.demonum = -1;

	for (;;)
	{
		int first_byte;

		/* Peek at first byte to decide protocol: binary opcode or JSON line. */
		first_byte = fgetc(stdin);
		if (first_byte == EOF)
			break;

		if (first_byte == QNN_BINARY_OP_STEP)
		{
			/* Binary step: read action struct directly (no JSON parsing). */
			qnn_action_t action;
			if (fread(&action, 1, QNN_BINARY_ACTION_SIZE, stdin) != (size_t)QNN_BINARY_ACTION_SIZE)
				break;
			qnn_pending_action = action;
			QNN_TrainingResetTick();
			if (!qnn_runtime.has_reset)
			{
				QNN_WriteError("Call reset before step");
				continue;
			}
			if (!qnn_runtime.done)
			{
				Host_Frame(qnn_runtime.fixed_dt);
				qnn_runtime.tick += 1;
				qnn_runtime.steps += 1;
			}
			{
				qnn_snapshot_t snapshot;
				QNN_CaptureSnapshotLocal(&snapshot, &action, false);
				snapshot.action_label = action;
				QNN_IOUpdate(&snapshot, qnn_runtime.fixed_dt, false);
				QNN_CommitSnapshot(&snapshot, &action, false);
				QNN_ClearAction(&qnn_pending_action);
				QNN_WriteStepResponse(&snapshot);
			}
			continue;
		}

		/* JSON protocol: put first byte back and read the full line. */
		ungetc(first_byte, stdin);
		if (fgets(line, sizeof(line), stdin) == NULL)
			break;

		if (strstr(line, "\"op\"") != NULL && strstr(line, "hello") != NULL)
		{
			QNN_HandleHello(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "reset") != NULL)
		{
			QNN_HandleReset(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "step") != NULL)
		{
			QNN_HandleStep(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "nav_query") != NULL)
		{
			QNN_HandleNavQuery(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "shutdown") != NULL)
		{
			fprintf(stdout, "{\"ok\":true}\n");
			fflush(stdout);
			QNN_FreeMapState(&qnn_map_state);
			CL_Disconnect();
			Host_Shutdown();
			return 0;
		}

		QNN_WriteError("unsupported op");
	}

	QNN_FreeMapState(&qnn_map_state);
	return 0;
}
