#include "nq_worker.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#define NQ_WORKER_PROTOCOL "v2"
#define NQ_WORKER_SERVER_NAME "quake-worker"
#define NQ_WORKER_UPSTREAM_COMMIT "bf4ac424ce754894ac8f1dae6a3981954bc9852d"
#define NQ_WORKER_MAX_LINE 8192
#define NQ_WORKER_MAX_EPISODE_ID 128
#define NQ_WORKER_MAX_VISIBLE 64
#define NQ_WORKER_MAX_EVENTS 16
#define NQ_WORKER_MAX_COMMAND_TEXT 1024

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct
{
	char	entity_id[32];
	int	entity_num;
	char	classname[NQ_WORKER_MAX_CLASSNAME];
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
	qboolean static_proxy;
} nq_worker_visible_entity_t;

typedef struct
{
	char	event_type[32];
	int	region_id;
	int	has_delta;
	int	delta;
	int	has_weapon_id;
	int	weapon_id;
	char	source_id[32];
	char	target_id[32];
} nq_worker_event_t;

typedef struct
{
	vec3_t	player_origin;
	vec3_t	player_velocity;
	vec3_t	player_view_angles;
	int	health;
	int	armor;
	int	ammo;
	int	weapon_id;
	qboolean grounded;
	int	current_region_id;
	qboolean goal_reached;
	qboolean done;
	char	done_reason[32];
	nq_worker_visible_entity_t visible[NQ_WORKER_MAX_VISIBLE];
	int	visible_count;
	nq_worker_event_t events[NQ_WORKER_MAX_EVENTS];
	int	event_count;
	int	damage_dealt;
	int	hit_count;
	int	shots_fired;
	int	damage_weapon_id;
} nq_worker_snapshot_t;

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
	nq_worker_action_t history[NQ_WORKER_ACTION_HISTORY];
	int	history_count;
	char	episode_id[NQ_WORKER_MAX_EPISODE_ID];
} nq_worker_runtime_t;

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
	char	pre_map_commands[NQ_WORKER_MAX_COMMAND_TEXT];
	char	post_map_commands[NQ_WORKER_MAX_COMMAND_TEXT];
} nq_worker_reset_options_t;

static nq_worker_map_state_t nq_worker_map_state;
static nq_worker_runtime_t nq_worker_runtime;
static nq_worker_reset_options_t nq_worker_reset_options;
static char nq_worker_basedir_storage[MAX_OSPATH] = ".";

qboolean isDedicated;
int nostdout = 1;
char *basedir = nq_worker_basedir_storage;
char *cachedir = "/tmp";
cvar_t sys_linerefresh = {"sys_linerefresh", "0"};

void Sys_DebugNumber(int y, int val)
{
	(void)y;
	(void)val;
}

void Sys_Printf(char *fmt, ...)
{
	(void)fmt;
}

void Sys_Quit(void)
{
	Host_Shutdown();
	exit(0);
}

void Sys_Init(void)
{
}

void Sys_Error(char *error, ...)
{
	va_list argptr;

	va_start(argptr, error);
	vfprintf(stderr, error, argptr);
	va_end(argptr);
	fputc('\n', stderr);
	Host_Shutdown();
	exit(1);
}

void Sys_Warn(char *warning, ...)
{
	va_list argptr;

	va_start(argptr, warning);
	vfprintf(stderr, warning, argptr);
	va_end(argptr);
	fputc('\n', stderr);
}

int Sys_FileTime(char *path)
{
	struct stat st;

	return stat(path, &st) == -1 ? -1 : (int)st.st_mtime;
}

void Sys_mkdir(char *path)
{
	mkdir(path, 0777);
}

int Sys_FileOpenRead(char *path, int *handle)
{
	int h;
	struct stat st;

	h = open(path, O_RDONLY, 0666);
	*handle = h;
	if (h == -1)
		return -1;
	if (fstat(h, &st) == -1)
		Sys_Error("Error fstating %s", path);
	return (int)st.st_size;
}

int Sys_FileOpenWrite(char *path)
{
	int handle;

	umask(0);
	handle = open(path, O_RDWR | O_CREAT | O_TRUNC, 0666);
	if (handle == -1)
		Sys_Error("Error opening %s: %s", path, strerror(errno));
	return handle;
}

int Sys_FileWrite(int handle, void *src, int count)
{
	return (int)write(handle, src, count);
}

void Sys_FileClose(int handle)
{
	close(handle);
}

void Sys_FileSeek(int handle, int position)
{
	lseek(handle, position, SEEK_SET);
}

int Sys_FileRead(int handle, void *dest, int count)
{
	return (int)read(handle, dest, count);
}

void Sys_DebugLog(char *file, char *fmt, ...)
{
	(void)file;
	(void)fmt;
}

void Sys_EditFile(char *filename)
{
	(void)filename;
}

double Sys_FloatTime(void)
{
	struct timeval tv;
	static int initialized;
	static double base;
	double now;

	gettimeofday(&tv, NULL);
	now = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
	if (!initialized)
	{
		base = now;
		initialized = 1;
	}
	return now - base;
}

void Sys_LineRefresh(void)
{
}

void Sys_SendKeyEvents(void)
{
}

void Sys_Sleep(void)
{
}

char *Sys_ConsoleInput(void)
{
	return NULL;
}

void Sys_HighFPPrecision(void)
{
}

void Sys_LowFPPrecision(void)
{
}

void Sys_MakeCodeWriteable(unsigned long startaddr, unsigned long length)
{
	long page_size;
	unsigned long addr;

	page_size = sysconf(_SC_PAGESIZE);
	addr = startaddr & ~(unsigned long)(page_size - 1);
	mprotect((void *)addr, length + startaddr - addr, PROT_READ | PROT_WRITE | PROT_EXEC);
}

static qboolean nq_worker_dir_exists(const char *path)
{
	struct stat st;

	if (stat(path, &st) != 0)
		return false;
	return S_ISDIR(st.st_mode) ? true : false;
}

static qboolean nq_worker_has_id1(const char *root)
{
	char path[MAX_OSPATH];

	snprintf(path, sizeof(path), "%s/id1", root);
	return nq_worker_dir_exists(path);
}

static void nq_worker_try_basedir(char *out, size_t out_size, const char *candidate)
{
	if (!candidate || !candidate[0])
		return;
	if (!nq_worker_has_id1(candidate))
		return;
	snprintf(out, out_size, "%s", candidate);
}

static void nq_worker_resolve_basedir(char *out, size_t out_size)
{
	const char *env;
	char cwd[MAX_OSPATH];
	char candidate[MAX_OSPATH];

	env = getenv("QUAKE_BASEDIR");
	out[0] = 0;

	if (env && env[0])
		nq_worker_try_basedir(out, out_size, env);
	if (out[0])
		return;

	nq_worker_try_basedir(out, out_size, "/assets");
	if (out[0])
		return;

	if (getcwd(cwd, sizeof(cwd)) == NULL)
	{
		snprintf(out, out_size, ".");
		return;
	}

	snprintf(candidate, sizeof(candidate), "%s/assets", cwd);
	nq_worker_try_basedir(out, out_size, candidate);
	if (out[0])
		return;

	snprintf(candidate, sizeof(candidate), "%s/../assets", cwd);
	nq_worker_try_basedir(out, out_size, candidate);
	if (out[0])
		return;

	snprintf(candidate, sizeof(candidate), "%s/../../assets", cwd);
	nq_worker_try_basedir(out, out_size, candidate);
	if (out[0])
		return;

	snprintf(out, out_size, "%s", cwd);
}

static int nq_json_extract_int(const char *line, const char *key, int fallback)
{
	const char *match;
	const char *colon;

	match = strstr(line, key);
	if (match == NULL)
		return fallback;
	colon = strchr(match, ':');
	if (colon == NULL)
		return fallback;
	return atoi(colon + 1);
}

static qboolean nq_json_extract_string(const char *line, const char *key, char *out, size_t out_size)
{
	const char *match;
	const char *colon;
	const char *start;
	const char *cursor;
	size_t index;

	match = strstr(line, key);
	if (match == NULL)
		return false;
	colon = strchr(match, ':');
	if (colon == NULL)
		return false;
	start = strchr(colon, '"');
	if (start == NULL)
		return false;
	start += 1;
	index = 0;
	for (cursor = start; *cursor && *cursor != '"'; ++cursor)
	{
		char ch;

		ch = *cursor;
		if (ch == '\\' && cursor[1])
		{
			cursor += 1;
			ch = *cursor;
			if (ch == 'n')
				ch = '\n';
			else if (ch == 'r')
				ch = '\r';
			else if (ch == 't')
				ch = '\t';
		}
		if (index + 1 < out_size)
			out[index++] = ch;
	}
	if (*cursor != '"')
		return false;
	out[index] = 0;
	return true;
}

static void nq_worker_canonicalize_map(char *out, size_t out_size, const char *requested)
{
	size_t i;

	snprintf(out, out_size, "%s", requested);
	for (i = 0; i < strlen(out); ++i)
		out[i] = (char)tolower((unsigned char)out[i]);
}

static qboolean nq_worker_prepare_map(const char *requested_map_id, char *error, size_t error_size)
{
	char map_name[NQ_WORKER_MAX_MAP_ID];

	if (!requested_map_id || !requested_map_id[0])
	{
		snprintf(error, error_size, "map_id is required");
		return false;
	}

	nq_worker_canonicalize_map(map_name, sizeof(map_name), requested_map_id);
	if (!strcmp(nq_worker_map_state.requested_map_id, requested_map_id)
		&& !strcmp(nq_worker_map_state.map_name, map_name)
		&& nq_worker_map_state.region_count > 0)
		return true;

	nq_worker_free_map_state(&nq_worker_map_state);
	if (!nq_worker_build_map_state(&nq_worker_map_state, requested_map_id, map_name, error, error_size))
		return false;
	return true;
}

static void nq_worker_write_json_string(FILE *out, const char *text)
{
	const unsigned char *cursor;

	fputc('"', out);
	for (cursor = (const unsigned char *)text; *cursor; ++cursor)
	{
		if (*cursor == '\\' || *cursor == '"')
		{
			fputc('\\', out);
			fputc(*cursor, out);
		}
		else if (*cursor == '\n')
			fputs("\\n", out);
		else if (*cursor == '\r')
			fputs("\\r", out);
		else if (*cursor == '\t')
			fputs("\\t", out);
		else if (*cursor < 32)
			fprintf(out, "\\u%04x", (unsigned int)*cursor);
		else
			fputc(*cursor, out);
	}
	fputc('"', out);
}

static void nq_worker_write_action_json(FILE *out, const nq_worker_action_t *action)
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

static qboolean nq_worker_client_ready(void)
{
	return (sv.active
		&& cls.state == ca_connected
		&& cls.signon == SIGNONS
		&& cl.worldmodel != NULL
		&& cl.viewentity > 0
		&& cl.viewentity < MAX_EDICTS) ? true : false;
}

static const char *nq_worker_prog_string(string_t value)
{
	if (!value)
		return "";
	return pr_strings + value;
}

static int nq_worker_weapon_id(void)
{
	int active;
	int weapon_id;

	active = cl.stats[STAT_ACTIVEWEAPON];
	if (active > 0)
	{
		weapon_id = 1;
		while (active > 1)
		{
			active >>= 1;
			weapon_id += 1;
		}
		return weapon_id;
	}
	if (cl.stats[STAT_WEAPON] > 0)
		return cl.stats[STAT_WEAPON];
	return 0;
}

static int nq_worker_current_frags(void)
{
	if (cl.viewentity > 0 && cl.scores != NULL && cl.viewentity - 1 < cl.maxclients)
		return cl.scores[cl.viewentity - 1].frags;
	return cl.stats[STAT_FRAGS];
}

static int nq_worker_current_monster_kills(void)
{
	return cl.stats[STAT_MONSTERS];
}

static int nq_worker_total_monsters(void)
{
	return cl.stats[STAT_TOTALMONSTERS];
}

static void nq_worker_reset_options_defaults(nq_worker_reset_options_t *options)
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

static void nq_worker_parse_reset_options(const char *line, nq_worker_reset_options_t *options)
{
	nq_worker_reset_options_defaults(options);
	options->maxplayers = nq_json_extract_int(line, "\"maxplayers\"", options->maxplayers);
	if (options->maxplayers < 1)
		options->maxplayers = 1;
	options->skill = nq_json_extract_int(line, "\"skill\"", options->skill);
	options->deathmatch = nq_json_extract_int(line, "\"deathmatch\"", options->deathmatch);
	options->coop = nq_json_extract_int(line, "\"coop\"", options->coop);
	options->teamplay = nq_json_extract_int(line, "\"teamplay\"", options->teamplay);
	options->fraglimit = nq_json_extract_int(line, "\"fraglimit\"", options->fraglimit);
	options->timelimit = nq_json_extract_int(line, "\"timelimit\"", options->timelimit);
	options->samelevel = nq_json_extract_int(line, "\"samelevel\"", options->samelevel);
	nq_json_extract_string(line, "\"pre_map_commands\"", options->pre_map_commands, sizeof(options->pre_map_commands));
	nq_json_extract_string(line, "\"post_map_commands\"", options->post_map_commands, sizeof(options->post_map_commands));
}

static void nq_worker_append_command(char *buffer, size_t buffer_size, const char *command_text)
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

static const char *nq_worker_server_classname(int entity_num)
{
	edict_t *edict;

	if (!sv.active)
		return "";
	if (entity_num < 0 || entity_num >= sv.num_edicts)
		return "";
	edict = EDICT_NUM(entity_num);
	if (edict->free || !edict->v.classname)
		return "";
	return nq_worker_prog_string(edict->v.classname);
}

static int nq_worker_weapon_index(int weapon_id)
{
	if (weapon_id < 0)
		return 0;
	if (weapon_id > 8)
		return 8;
	return weapon_id;
}

static void nq_worker_entity_id_string(int entity_num, char *out, size_t out_size)
{
	if (entity_num <= 0)
	{
		out[0] = 0;
		return;
	}
	snprintf(out, out_size, "entity_%04d", entity_num);
}

static qboolean nq_worker_track_damage_entity(int entity_num, const edict_t *edict)
{
	const char *classname;

	if (!edict || edict->free || entity_num == cl.viewentity || !edict->v.classname)
		return false;
	classname = nq_worker_prog_string(edict->v.classname);
	if (!strcmp(classname, "player"))
		return true;
	if (!strncmp(classname, "monster_", 8))
		return true;
	return false;
}

static void nq_worker_cache_entity_state(void)
{
	int entity_num;

	memset(nq_worker_runtime.prev_entity_health, 0, sizeof(nq_worker_runtime.prev_entity_health));
	memset(nq_worker_runtime.prev_entity_active, 0, sizeof(nq_worker_runtime.prev_entity_active));
	if (!sv.active)
		return;
	for (entity_num = 0; entity_num < sv.num_edicts && entity_num < MAX_EDICTS; ++entity_num)
	{
		edict_t *edict;

		edict = EDICT_NUM(entity_num);
		if (!nq_worker_track_damage_entity(entity_num, edict))
			continue;
		nq_worker_runtime.prev_entity_health[entity_num] = (int)edict->v.health;
		nq_worker_runtime.prev_entity_active[entity_num] = true;
	}
}

static void nq_worker_write_int_array(FILE *out, const int *values, int count)
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

static void nq_worker_write_server_players(FILE *out)
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
		classname = nq_worker_prog_string(edict->v.classname);
		netname = nq_worker_prog_string(edict->v.netname);
		if (strcmp(classname, "player") && !netname[0])
			continue;
		if (wrote)
			fputc(',', out);
		wrote = 1;
		fprintf(out, "{\"entity_num\":%d,\"classname\":", entity_num);
		nq_worker_write_json_string(out, classname);
		fprintf(out, ",\"netname\":");
		nq_worker_write_json_string(out, netname);
		fprintf(out, ",\"frags\":%d,\"health\":%.0f,\"origin\":[%.1f,%.1f,%.1f]}",
			(int)edict->v.frags,
			edict->v.health,
			edict->v.origin[0],
			edict->v.origin[1],
			edict->v.origin[2]);
	}
	fputc(']', out);
}

static void nq_worker_runtime_reset(void)
{
	memset(&nq_worker_runtime, 0, sizeof(nq_worker_runtime));
	nq_worker_runtime.fixed_tick_hz = 20;
	nq_worker_runtime.fixed_dt = 1.0f / 20.0f;
	nq_worker_reset_options_defaults(&nq_worker_reset_options);
}

static void nq_worker_push_history(const nq_worker_action_t *action)
{
	if (nq_worker_runtime.history_count < NQ_WORKER_ACTION_HISTORY)
	{
		nq_worker_runtime.history[nq_worker_runtime.history_count] = *action;
		nq_worker_runtime.history_count += 1;
		return;
	}
	nq_worker_runtime.history[0] = nq_worker_runtime.history[1];
	nq_worker_runtime.history[1] = *action;
}

static void nq_worker_add_event(
	nq_worker_snapshot_t *snapshot,
	const char *event_type,
	int region_id,
	int has_delta,
	int delta,
	int has_weapon_id,
	int weapon_id,
	const char *source_id,
	const char *target_id)
{
	nq_worker_event_t *event;

	if (snapshot->event_count >= NQ_WORKER_MAX_EVENTS)
		return;
	event = &snapshot->events[snapshot->event_count];
	memset(event, 0, sizeof(*event));
	snprintf(event->event_type, sizeof(event->event_type), "%s", event_type);
	event->region_id = region_id;
	event->has_delta = has_delta;
	event->delta = delta;
	event->has_weapon_id = has_weapon_id;
	event->weapon_id = weapon_id;
	snprintf(event->source_id, sizeof(event->source_id), "%s", source_id ? source_id : "");
	snprintf(event->target_id, sizeof(event->target_id), "%s", target_id ? target_id : "");
	snapshot->event_count += 1;
}

static void nq_worker_add_static_proxy(nq_worker_snapshot_t *snapshot, const nq_worker_static_object_t *object)
{
	nq_worker_visible_entity_t *entity;

	if (snapshot->visible_count >= NQ_WORKER_MAX_VISIBLE)
		return;
	entity = &snapshot->visible[snapshot->visible_count];
	memset(entity, 0, sizeof(*entity));
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

static void nq_worker_capture_visible_entities(nq_worker_snapshot_t *snapshot)
{
	int entity_num;

	for (entity_num = 1; entity_num < cl.num_entities && snapshot->visible_count < NQ_WORKER_MAX_VISIBLE; ++entity_num)
	{
		entity_t *entity;
		edict_t *server_edict;
		nq_worker_visible_entity_t *out_entity;
		vec3_t delta;

		entity = &cl_entities[entity_num];
		server_edict = (entity_num >= 0 && entity_num < sv.num_edicts) ? EDICT_NUM(entity_num) : NULL;
		if (entity_num == cl.viewentity)
			continue;
		if (entity->model == NULL)
			continue;

		out_entity = &snapshot->visible[snapshot->visible_count];
		memset(out_entity, 0, sizeof(*out_entity));
		snprintf(out_entity->entity_id, sizeof(out_entity->entity_id), "entity_%04d", entity_num);
		snprintf(out_entity->classname, sizeof(out_entity->classname), "%s", nq_worker_server_classname(entity_num));
		out_entity->entity_num = entity_num;
		VectorCopy(entity->origin, out_entity->origin);
		VectorSubtract(entity->msg_origins[0], entity->msg_origins[1], delta);
		if (nq_worker_runtime.fixed_dt > 0.0f)
			VectorScale(delta, 1.0f / nq_worker_runtime.fixed_dt, out_entity->velocity);
		else
			VectorCopy(vec3_origin, out_entity->velocity);
		VectorCopy(entity->angles, out_entity->angles);
		out_entity->model_id = entity->baseline.modelindex > 0 ? entity->baseline.modelindex : 0;
		out_entity->frame = entity->frame;
		out_entity->effects = entity->effects;
		out_entity->skin = entity->skinnum;
		out_entity->health = (server_edict != NULL && !server_edict->free) ? (int)server_edict->v.health : 0;
		out_entity->frags = (server_edict != NULL && !server_edict->free) ? (int)server_edict->v.frags : 0;
		out_entity->region_id = nq_worker_nearest_region_id(&nq_worker_map_state, out_entity->origin);
		snapshot->visible_count += 1;
	}

	if (snapshot->visible_count == 0)
	{
		int i;

		for (i = 0; i < nq_worker_map_state.static_object_count && snapshot->visible_count < 4; ++i)
		{
			const nq_worker_static_object_t *object;

			object = &nq_worker_map_state.static_objects[i];
			if (object->region_id != snapshot->current_region_id && !nq_worker_is_goal_region(&nq_worker_map_state, object->region_id))
				continue;
			if (strcmp(object->category, "item")
				&& strcmp(object->category, "goal")
				&& strcmp(object->category, "trigger"))
				continue;
			nq_worker_add_static_proxy(snapshot, object);
		}
	}
}

static void nq_worker_capture_snapshot(nq_worker_snapshot_t *snapshot, const nq_worker_action_t *current_action, qboolean reset_flag)
{
	entity_t *player_entity;
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
	int damage_target_entity_num;
	int damage_target_count;
	qboolean goal_reached;

	memset(snapshot, 0, sizeof(*snapshot));
	player_entity = (cl.viewentity > 0 && cl.viewentity < MAX_EDICTS) ? &cl_entities[cl.viewentity] : NULL;
	if (player_entity != NULL)
	{
		VectorCopy(player_entity->origin, snapshot->player_origin);
	}
	else
	{
		VectorCopy(vec3_origin, snapshot->player_origin);
	}
	VectorCopy(cl.velocity, snapshot->player_velocity);
	VectorCopy(cl.viewangles, snapshot->player_view_angles);

	health = cl.stats[STAT_HEALTH];
	armor = cl.stats[STAT_ARMOR];
	ammo = cl.stats[STAT_AMMO];
	weapon_id = nq_worker_weapon_id();
	frags = nq_worker_current_frags();
	monster_kills = nq_worker_current_monster_kills();
	goal_reached = cl.intermission ? true : false;
	current_region_id = nq_worker_nearest_region_id(&nq_worker_map_state, snapshot->player_origin);

	snapshot->health = health;
	snapshot->armor = armor;
	snapshot->ammo = ammo;
	snapshot->weapon_id = weapon_id;
	snapshot->grounded = cl.onground ? true : false;
	snapshot->current_region_id = current_region_id;
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
			damage_weapon_id = weapon_id > 0 ? weapon_id : nq_worker_runtime.last_fire_weapon_id;
			nq_worker_runtime.last_fire_weapon_id = damage_weapon_id;
			nq_worker_runtime.total_shots_fired += 1;
			nq_worker_runtime.weapon_shots_fired[nq_worker_weapon_index(damage_weapon_id)] += 1;
		}

		if (health < nq_worker_runtime.prev_health)
			nq_worker_add_event(snapshot, "damage_taken", current_region_id, 1, nq_worker_runtime.prev_health - health, 0, 0, "", "");
		else if (health > nq_worker_runtime.prev_health)
			nq_worker_add_event(snapshot, "pickup_health", current_region_id, 1, health - nq_worker_runtime.prev_health, 0, 0, "", "");

		if (armor > nq_worker_runtime.prev_armor)
			nq_worker_add_event(snapshot, "pickup_armor", current_region_id, 1, armor - nq_worker_runtime.prev_armor, 0, 0, "", "");
		if (ammo > nq_worker_runtime.prev_ammo)
			nq_worker_add_event(snapshot, "pickup_ammo", current_region_id, 1, ammo - nq_worker_runtime.prev_ammo, 0, 0, "", "");
		if (weapon_id > 0 && weapon_id != nq_worker_runtime.prev_weapon)
			nq_worker_add_event(snapshot, "pickup_weapon", current_region_id, 0, 0, 1, weapon_id, "", "");
		if (cl.items != nq_worker_runtime.prev_items
			&& snapshot->event_count == 0
			&& cl.items > nq_worker_runtime.prev_items)
			nq_worker_add_event(snapshot, "pickup_item", current_region_id, 0, 0, 0, 0, "", "");
		if (frags > nq_worker_runtime.prev_frags)
			nq_worker_add_event(snapshot, "frag_gained", current_region_id, 1, frags - nq_worker_runtime.prev_frags, 0, 0, "", "");
		else if (frags < nq_worker_runtime.prev_frags)
			nq_worker_add_event(snapshot, "frag_lost", current_region_id, 1, nq_worker_runtime.prev_frags - frags, 0, 0, "", "");
		if (monster_kills > nq_worker_runtime.prev_monster_kills)
			nq_worker_add_event(snapshot, "monster_kill", current_region_id, 1, monster_kills - nq_worker_runtime.prev_monster_kills, 0, 0, "", "");

		fire_window_active = current_action->fire || nq_worker_runtime.recent_fire_steps > 0;
		if (fire_window_active && damage_weapon_id <= 0)
			damage_weapon_id = nq_worker_runtime.last_fire_weapon_id > 0 ? nq_worker_runtime.last_fire_weapon_id : weapon_id;
		if (fire_window_active)
		{
			for (entity_num = 0; entity_num < sv.num_edicts && entity_num < MAX_EDICTS; ++entity_num)
			{
				edict_t *edict;
				int prev_health;
				int current_health;

				edict = EDICT_NUM(entity_num);
				current_health = nq_worker_track_damage_entity(entity_num, edict) ? (int)edict->v.health : 0;
				prev_health = nq_worker_runtime.prev_entity_active[entity_num] ? nq_worker_runtime.prev_entity_health[entity_num] : current_health;
				if (prev_health <= current_health)
					continue;
				damage_dealt += prev_health - current_health;
				hit_count += 1;
				damage_target_count += 1;
				damage_target_entity_num = damage_target_count == 1 ? entity_num : -1;
			}
		}
	}

	if (damage_dealt > 0)
	{
		char source_id[32];
		char target_id[32];

		nq_worker_entity_id_string(cl.viewentity, source_id, sizeof(source_id));
		target_id[0] = 0;
		if (damage_target_count == 1 && damage_target_entity_num > 0)
			nq_worker_entity_id_string(damage_target_entity_num, target_id, sizeof(target_id));
		nq_worker_runtime.total_damage_dealt += damage_dealt;
		nq_worker_runtime.total_hit_count += hit_count;
		nq_worker_runtime.weapon_damage_dealt[nq_worker_weapon_index(damage_weapon_id)] += damage_dealt;
		nq_worker_runtime.weapon_hits_landed[nq_worker_weapon_index(damage_weapon_id)] += hit_count;
		nq_worker_add_event(snapshot, "damage_dealt", current_region_id, 1, damage_dealt, 1, damage_weapon_id, source_id, target_id);
		nq_worker_add_event(snapshot, "hit_confirmed", current_region_id, 1, hit_count, 1, damage_weapon_id, source_id, target_id);
	}
	if (shots_fired > 0)
	{
		char source_id[32];

		nq_worker_entity_id_string(cl.viewentity, source_id, sizeof(source_id));
		nq_worker_add_event(snapshot, "shots_fired", current_region_id, 1, shots_fired, 1, damage_weapon_id, source_id, "");
	}

	snapshot->damage_dealt = damage_dealt;
	snapshot->hit_count = hit_count;
	snapshot->shots_fired = shots_fired;
	snapshot->damage_weapon_id = damage_weapon_id;

	if (goal_reached && !nq_worker_runtime.prev_intermission)
		nq_worker_add_event(snapshot, "goal_reached", current_region_id, 0, 0, 0, 0, "", "");
	if (goal_reached)
	{
		snapshot->done = true;
		snprintf(snapshot->done_reason, sizeof(snapshot->done_reason), "goal_reached");
	}
	else if (health <= 0)
	{
		nq_worker_add_event(snapshot, "player_died", current_region_id, 0, 0, 0, 0, "", "");
		snapshot->done = true;
		snprintf(snapshot->done_reason, sizeof(snapshot->done_reason), "player_died");
	}

	nq_worker_capture_visible_entities(snapshot);
}

static void nq_worker_commit_snapshot(const nq_worker_snapshot_t *snapshot, const nq_worker_action_t *current_action, qboolean reset_flag)
{
	nq_worker_runtime.prev_health = snapshot->health;
	nq_worker_runtime.prev_armor = snapshot->armor;
	nq_worker_runtime.prev_ammo = snapshot->ammo;
	nq_worker_runtime.prev_weapon = snapshot->weapon_id;
	nq_worker_runtime.prev_items = cl.items;
	nq_worker_runtime.prev_intermission = cl.intermission;
	nq_worker_runtime.prev_frags = nq_worker_current_frags();
	nq_worker_runtime.prev_monster_kills = nq_worker_current_monster_kills();
	nq_worker_runtime.done = snapshot->done;
	if (current_action->fire)
		nq_worker_runtime.recent_fire_steps = 2;
	else if (nq_worker_runtime.recent_fire_steps > 0)
		nq_worker_runtime.recent_fire_steps -= 1;
	if (snapshot->damage_weapon_id > 0)
		nq_worker_runtime.last_fire_weapon_id = snapshot->damage_weapon_id;
	nq_worker_cache_entity_state();
	if (!reset_flag)
		nq_worker_push_history(current_action);
}

static void nq_worker_write_obs(FILE *out, const nq_worker_snapshot_t *snapshot, const nq_worker_action_t *current_action)
{
	float goal_progress;
	float region_distance;
	int i;
	float obs[20];

	goal_progress = nq_worker_goal_progress(&nq_worker_map_state, snapshot->current_region_id);
	region_distance = 1.0f - goal_progress;

	for (i = 0; i < 20; ++i)
		obs[i] = 0.0f;

	obs[0] = snapshot->player_origin[0] / 2048.0f;
	obs[1] = snapshot->player_origin[1] / 2048.0f;
	obs[2] = snapshot->player_origin[2] / 512.0f;
	obs[3] = snapshot->player_velocity[0] / 320.0f;
	obs[4] = snapshot->player_velocity[1] / 320.0f;
	obs[5] = snapshot->player_velocity[2] / 320.0f;
	obs[6] = cosf(snapshot->player_view_angles[1] * (float)M_PI / 180.0f);
	obs[7] = sinf(snapshot->player_view_angles[1] * (float)M_PI / 180.0f);
	obs[8] = snapshot->health / 100.0f;
	obs[9] = snapshot->armor / 100.0f;
	obs[10] = snapshot->ammo / 100.0f;
	obs[11] = snapshot->weapon_id / 8.0f;
	obs[12] = goal_progress;
	obs[13] = nq_worker_is_goal_region(&nq_worker_map_state, snapshot->current_region_id) ? 1.0f : 0.0f;
	obs[14] = snapshot->grounded ? 1.0f : 0.0f;
	obs[15] = snapshot->visible_count / 8.0f;
	obs[16] = snapshot->event_count / 8.0f;
	obs[17] = region_distance;
	obs[18] = current_action->fire ? 1.0f : 0.0f;
	obs[19] = snapshot->done ? 1.0f : 0.0f;

	fputc('[', out);
	for (i = 0; i < 20; ++i)
	{
		if (i > 0)
			fputc(',', out);
		fprintf(out, "%.6f", obs[i]);
	}
	fputc(']', out);
}

static void nq_worker_write_events(FILE *out, const nq_worker_snapshot_t *snapshot)
{
	int i;

	fputc('[', out);
	for (i = 0; i < snapshot->event_count; ++i)
	{
		const nq_worker_event_t *event;
		int wrote_payload;

		event = &snapshot->events[i];
		if (i > 0)
			fputc(',', out);
		fprintf(out, "{\"event_type\":");
		nq_worker_write_json_string(out, event->event_type);
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
		nq_worker_write_json_string(out, event->source_id);
		fprintf(out, ",\"target_id\":");
		nq_worker_write_json_string(out, event->target_id);
		fputc('}', out);
	}
	fputc(']', out);
}

static void nq_worker_write_visible_entities(FILE *out, const nq_worker_snapshot_t *snapshot)
{
	int i;

	fputc('[', out);
	for (i = 0; i < snapshot->visible_count; ++i)
	{
		const nq_worker_visible_entity_t *entity;

		entity = &snapshot->visible[i];
		if (i > 0)
			fputc(',', out);
		fprintf(out, "{\"angles\":[%.1f,%.1f,%.1f],\"classname\":",
			entity->angles[0], entity->angles[1], entity->angles[2]);
		nq_worker_write_json_string(out, entity->classname);
		fprintf(out, ",\"entity_id\":");
		nq_worker_write_json_string(out, entity->entity_id);
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

static void nq_worker_write_world_tick(FILE *out, const nq_worker_snapshot_t *snapshot, const nq_worker_action_t *current_action, qboolean reset_flag)
{
	int i;

	fprintf(out, "{\"action_history\":[");
	for (i = 0; i < nq_worker_runtime.history_count; ++i)
	{
		if (i > 0)
			fputc(',', out);
		nq_worker_write_action_json(out, &nq_worker_runtime.history[i]);
	}
	fprintf(out, "],\"action_label\":");
	nq_worker_write_action_json(out, current_action);
	fprintf(out, ",\"current_region_id\":%d,\"debug\":{\"client_maxclients\":%d,\"damage_dealt\":%d,\"damage_dealt_total\":%d,\"damage_weapon_id\":%d,\"frags\":%d,\"goal_progress\":%.6f,\"hit_count\":%d,\"hit_count_total\":%d,\"monster_kills\":%d,\"monster_total\":%d,\"player_entity_num\":%d,\"seed\":%d,\"shots_fired\":%d,\"shots_fired_total\":%d,\"weapon_damage_dealt_total\":",
		snapshot->current_region_id,
		cl.maxclients,
		snapshot->damage_dealt,
		nq_worker_runtime.total_damage_dealt,
		snapshot->damage_weapon_id,
		nq_worker_current_frags(),
		nq_worker_goal_progress(&nq_worker_map_state, snapshot->current_region_id),
		snapshot->hit_count,
		nq_worker_runtime.total_hit_count,
		nq_worker_current_monster_kills(),
		nq_worker_total_monsters(),
		cl.viewentity,
		nq_worker_runtime.seed,
		snapshot->shots_fired,
		nq_worker_runtime.total_shots_fired);
	nq_worker_write_int_array(out, nq_worker_runtime.weapon_damage_dealt, 9);
	fprintf(out, ",\"weapon_hits_landed_total\":");
	nq_worker_write_int_array(out, nq_worker_runtime.weapon_hits_landed, 9);
	fprintf(out, ",\"weapon_shots_fired_total\":");
	nq_worker_write_int_array(out, nq_worker_runtime.weapon_shots_fired, 9);
	fprintf(out, ",\"server_players\":");
	nq_worker_write_server_players(out);
	fprintf(out, "},\"done\":%s,\"done_reason\":",
		snapshot->done ? "true" : "false");
	nq_worker_write_json_string(out, snapshot->done_reason);
	fprintf(out, ",\"episode_id\":");
	nq_worker_write_json_string(out, nq_worker_runtime.episode_id);
	fprintf(out, ",\"events\":");
	nq_worker_write_events(out, snapshot);
	fprintf(out, ",\"map_id\":");
	nq_worker_write_json_string(out, nq_worker_map_state.requested_map_id);
	fprintf(out, ",\"player\":{\"ammo\":%d,\"armor\":%d,\"grounded\":%s,\"health\":%d,\"origin\":[%.1f,%.1f,%.1f],\"velocity\":[%.1f,%.1f,%.1f],\"view_angles\":[%.1f,%.1f,%.1f],\"weapon_id\":%d},\"reset\":%s,\"tick\":%d,\"visible_entities\":",
		snapshot->ammo,
		snapshot->armor,
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
		reset_flag ? "true" : "false",
		nq_worker_runtime.tick);
	nq_worker_write_visible_entities(out, snapshot);
	fputc('}', out);
}

static qboolean nq_worker_reset_world(int seed, char *error, size_t error_size)
{
	char command[2048];
	int frame;

	if (nq_worker_map_state.map_name[0] == 0)
	{
		snprintf(error, error_size, "Call hello first so the worker knows which map to load");
		return false;
	}

	nq_worker_clear_action(&nq_worker_pending_action);
	nq_worker_runtime.seed = seed >= 0 ? seed : 0;
	srand((unsigned int)nq_worker_runtime.seed);
	DEFAULTnet_hostport = 0;
	net_hostport = 0;
	cls.demonum = -1;
	nq_worker_runtime.done = false;

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
		nq_worker_reset_options.maxplayers,
		nq_worker_reset_options.skill,
		nq_worker_reset_options.deathmatch,
		nq_worker_reset_options.coop,
		nq_worker_reset_options.teamplay,
		nq_worker_reset_options.fraglimit,
		nq_worker_reset_options.timelimit,
		nq_worker_reset_options.samelevel);
	nq_worker_append_command(command, sizeof(command), nq_worker_reset_options.pre_map_commands);
	snprintf(command + strlen(command), sizeof(command) - strlen(command),
		"map %s\n",
		nq_worker_map_state.map_name);
	Cbuf_AddText(command);

	for (frame = 0; frame < 2048; ++frame)
	{
		Host_Frame(nq_worker_runtime.fixed_dt);
		if (nq_worker_client_ready())
			break;
	}
	if (!nq_worker_client_ready())
	{
		snprintf(error, error_size, "Timed out waiting for local client signon on %s", nq_worker_map_state.map_name);
		return false;
	}
	cl.movemessages = 2;
	if (nq_worker_reset_options.post_map_commands[0])
	{
		Cbuf_AddText(nq_worker_reset_options.post_map_commands);
		Cbuf_AddText("\n");
		for (frame = 0; frame < 128; ++frame)
			Host_Frame(nq_worker_runtime.fixed_dt);
	}

	nq_worker_runtime.tick = 0;
	nq_worker_runtime.steps = 0;
	nq_worker_runtime.history_count = 0;
	nq_worker_runtime.recent_fire_steps = 0;
	nq_worker_runtime.last_fire_weapon_id = 0;
	nq_worker_runtime.total_damage_dealt = 0;
	nq_worker_runtime.total_hit_count = 0;
	nq_worker_runtime.total_shots_fired = 0;
	memset(nq_worker_runtime.weapon_damage_dealt, 0, sizeof(nq_worker_runtime.weapon_damage_dealt));
	memset(nq_worker_runtime.weapon_hits_landed, 0, sizeof(nq_worker_runtime.weapon_hits_landed));
	memset(nq_worker_runtime.weapon_shots_fired, 0, sizeof(nq_worker_runtime.weapon_shots_fired));
	memset(nq_worker_runtime.prev_entity_health, 0, sizeof(nq_worker_runtime.prev_entity_health));
	memset(nq_worker_runtime.prev_entity_active, 0, sizeof(nq_worker_runtime.prev_entity_active));
	nq_worker_runtime.episode_index += 1;
	nq_worker_runtime.has_reset = true;
	snprintf(nq_worker_runtime.episode_id, sizeof(nq_worker_runtime.episode_id), "%s-%d-%04d",
		nq_worker_map_state.requested_map_id,
		nq_worker_runtime.seed,
		nq_worker_runtime.episode_index);
	nq_worker_runtime.prev_health = cl.stats[STAT_HEALTH];
	nq_worker_runtime.prev_armor = cl.stats[STAT_ARMOR];
	nq_worker_runtime.prev_ammo = cl.stats[STAT_AMMO];
	nq_worker_runtime.prev_weapon = nq_worker_weapon_id();
	nq_worker_runtime.prev_items = cl.items;
	nq_worker_runtime.prev_intermission = cl.intermission;
	nq_worker_runtime.prev_frags = nq_worker_current_frags();
	nq_worker_runtime.prev_monster_kills = nq_worker_current_monster_kills();
	nq_worker_cache_entity_state();
	return true;
}

static void nq_worker_write_error(const char *message)
{
	fprintf(stdout, "{\"error\":");
	nq_worker_write_json_string(stdout, message);
	fprintf(stdout, ",\"ok\":false}\n");
	fflush(stdout);
}

static void nq_worker_write_hello_response(void)
{
	fprintf(stdout, "{\"capabilities\":[\"legacy_obs\",\"listen_local\",\"reset_options\",\"udp_networking\",\"world_v2\"],\"map_id\":");
	nq_worker_write_json_string(stdout, nq_worker_map_state.requested_map_id);
	fprintf(stdout, ",\"map_state\":");
	nq_worker_write_map_state_json(stdout, &nq_worker_map_state);
	fprintf(stdout, ",\"ok\":true,\"protocol_version\":");
	nq_worker_write_json_string(stdout, NQ_WORKER_PROTOCOL);
	fprintf(stdout, ",\"server\":");
	nq_worker_write_json_string(stdout, NQ_WORKER_SERVER_NAME);
	fprintf(stdout, ",\"tick_hz\":%d,\"worker_build\":{\"basedir\":", nq_worker_runtime.fixed_tick_hz);
	nq_worker_write_json_string(stdout, basedir);
	fprintf(stdout, ",\"upstream_commit\":");
	nq_worker_write_json_string(stdout, NQ_WORKER_UPSTREAM_COMMIT);
	fprintf(stdout, "}}\n");
	fflush(stdout);
}

static void nq_worker_write_reset_response(const nq_worker_snapshot_t *snapshot, const nq_worker_action_t *action)
{
	fprintf(stdout, "{\"info\":{\"map_id\":");
	nq_worker_write_json_string(stdout, nq_worker_map_state.requested_map_id);
	fprintf(stdout, ",\"deathmatch\":%d,\"maxplayers\":%d,\"seed\":%d,\"teamplay\":%d},\"obs\":",
		nq_worker_reset_options.deathmatch,
		nq_worker_reset_options.maxplayers,
		nq_worker_runtime.seed,
		nq_worker_reset_options.teamplay);
	nq_worker_write_obs(stdout, snapshot, action);
	fprintf(stdout, ",\"ok\":true,\"world_tick\":");
	nq_worker_write_world_tick(stdout, snapshot, action, true);
	fprintf(stdout, "}\n");
	fflush(stdout);
}

static void nq_worker_write_step_response(const nq_worker_snapshot_t *snapshot, const nq_worker_action_t *action)
{
	fprintf(stdout, "{\"done\":%s,\"info\":{\"deathmatch\":%d,\"goal_reached\":%s,\"maxplayers\":%d,\"seed\":%d,\"steps\":%d,\"teamplay\":%d},\"obs\":",
		snapshot->done ? "true" : "false",
		nq_worker_reset_options.deathmatch,
		snapshot->goal_reached ? "true" : "false",
		nq_worker_reset_options.maxplayers,
		nq_worker_runtime.seed,
		nq_worker_runtime.steps,
		nq_worker_reset_options.teamplay);
	nq_worker_write_obs(stdout, snapshot, action);
	fprintf(stdout, ",\"ok\":true,\"reward\":0.0,\"world_tick\":");
	nq_worker_write_world_tick(stdout, snapshot, action, false);
	fprintf(stdout, "}\n");
	fflush(stdout);
}

static int nq_worker_handle_hello(const char *line)
{
	char map_id[NQ_WORKER_MAX_MAP_ID];
	char error[256];

	memset(map_id, 0, sizeof(map_id));
	memset(error, 0, sizeof(error));
	if (!nq_json_extract_string(line, "\"map_id\"", map_id, sizeof(map_id)))
	{
		snprintf(map_id, sizeof(map_id), "E1M1");
	}
	nq_worker_runtime.fixed_tick_hz = nq_json_extract_int(line, "\"tick_hz\"", nq_worker_runtime.fixed_tick_hz > 0 ? nq_worker_runtime.fixed_tick_hz : 20);
	if (nq_worker_runtime.fixed_tick_hz <= 0)
		nq_worker_runtime.fixed_tick_hz = 20;
	nq_worker_runtime.fixed_dt = 1.0f / (float)nq_worker_runtime.fixed_tick_hz;

	if (!nq_worker_prepare_map(map_id, error, sizeof(error)))
	{
		nq_worker_write_error(error);
		return 0;
	}

	nq_worker_write_hello_response();
	return 0;
}

static int nq_worker_handle_reset(const char *line)
{
	nq_worker_snapshot_t snapshot;
	nq_worker_action_t action;
	char error[256];
	int seed;

	nq_worker_clear_action(&action);
	memset(error, 0, sizeof(error));
	seed = nq_json_extract_int(line, "\"seed\"", -1);
	nq_worker_parse_reset_options(line, &nq_worker_reset_options);
	if (!nq_worker_reset_world(seed, error, sizeof(error)))
	{
		nq_worker_write_error(error);
		return 0;
	}

	nq_worker_capture_snapshot(&snapshot, &action, true);
	nq_worker_commit_snapshot(&snapshot, &action, true);
	nq_worker_write_reset_response(&snapshot, &action);
	return 0;
}

static int nq_worker_handle_step(const char *line)
{
	nq_worker_snapshot_t snapshot;
	nq_worker_action_t action;

	if (!nq_worker_runtime.has_reset)
	{
		nq_worker_write_error("Call reset before step");
		return 0;
	}

	nq_worker_clear_action(&action);
	action.move = nq_json_extract_int(line, "\"move\"", 0);
	action.strafe = nq_json_extract_int(line, "\"strafe\"", 0);
	action.look_yaw = nq_json_extract_int(line, "\"look_yaw\"", NQ_WORKER_LOOK_NEUTRAL_LABEL);
	action.look_pitch = nq_json_extract_int(line, "\"look_pitch\"", NQ_WORKER_LOOK_NEUTRAL_LABEL);
	action.look_yaw_count = nq_json_extract_int(line, "\"look_yaw_count\"", 0);
	action.look_pitch_count = nq_json_extract_int(line, "\"look_pitch_count\"", 0);
	action.fire = nq_json_extract_int(line, "\"fire\"", 0);
	action.jump = nq_json_extract_int(line, "\"jump\"", 0);
	action.weapon = nq_json_extract_int(line, "\"weapon\"", 0);
	nq_worker_pending_action = action;

	if (!nq_worker_runtime.done)
	{
		Host_Frame(nq_worker_runtime.fixed_dt);
		nq_worker_runtime.tick += 1;
		nq_worker_runtime.steps += 1;
	}
	nq_worker_capture_snapshot(&snapshot, &action, false);
	nq_worker_commit_snapshot(&snapshot, &action, false);
	nq_worker_clear_action(&nq_worker_pending_action);
	nq_worker_write_step_response(&snapshot, &action);
	return 0;
}

int main(int argc, char **argv)
{
	quakeparms_t parms;
	char line[NQ_WORKER_MAX_LINE];

	nq_worker_resolve_basedir(nq_worker_basedir_storage, sizeof(nq_worker_basedir_storage));
	nq_worker_runtime_reset();
	nq_worker_clear_action(&nq_worker_pending_action);
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
			nq_worker_handle_hello(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "reset") != NULL)
		{
			nq_worker_handle_reset(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "step") != NULL)
		{
			nq_worker_handle_step(line);
			continue;
		}
		if (strstr(line, "\"op\"") != NULL && strstr(line, "shutdown") != NULL)
		{
			fprintf(stdout, "{\"ok\":true}\n");
			fflush(stdout);
			nq_worker_free_map_state(&nq_worker_map_state);
			Host_Shutdown();
			return 0;
		}

		nq_worker_write_error("unsupported op");
	}

	nq_worker_free_map_state(&nq_worker_map_state);
	return 0;
}
