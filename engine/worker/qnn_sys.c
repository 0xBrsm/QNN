#include "qnn.h"
#include "qnn_nav_oracle.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

/* ── shared globals ──────────────────────────────────────────────── */

qnn_map_state_t qnn_map_state;
qnn_resample_state_t qnn_resample;
char qnn_basedir_storage[MAX_OSPATH] = ".";

qboolean isDedicated;
int nostdout = 1;
char *basedir = qnn_basedir_storage;
char *cachedir = "/tmp";
cvar_t sys_linerefresh = {"sys_linerefresh", "0"};

/* ── Sys_* stubs (Quake engine system layer) ─────────────────────── */

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

/* ── basedir resolution ──────────────────────────────────────────── */

static qboolean QNN_DirExists(const char *path)
{
	struct stat st;

	if (stat(path, &st) != 0)
		return false;
	return S_ISDIR(st.st_mode) ? true : false;
}

static qboolean QNN_HasId1(const char *root)
{
	char path[MAX_OSPATH];

	snprintf(path, sizeof(path), "%s/id1", root);
	return QNN_DirExists(path);
}

static void QNN_TryBasedir(char *out, size_t out_size, const char *candidate)
{
	if (!candidate || !candidate[0])
		return;
	if (!QNN_HasId1(candidate))
		return;
	snprintf(out, out_size, "%s", candidate);
}

void QNN_ResolveBasedir(char *out, size_t out_size)
{
	const char *env;
	char cwd[MAX_OSPATH];
	char candidate[MAX_OSPATH];

	env = getenv("QUAKE_BASEDIR");
	out[0] = 0;

	if (env && env[0])
		QNN_TryBasedir(out, out_size, env);
	if (out[0])
		return;

	QNN_TryBasedir(out, out_size, "/assets");
	if (out[0])
		return;

	if (getcwd(cwd, sizeof(cwd)) == NULL)
	{
		snprintf(out, out_size, ".");
		return;
	}

	snprintf(candidate, sizeof(candidate), "%s/assets", cwd);
	QNN_TryBasedir(out, out_size, candidate);
	if (out[0])
		return;

	snprintf(candidate, sizeof(candidate), "%s/../assets", cwd);
	QNN_TryBasedir(out, out_size, candidate);
	if (out[0])
		return;

	snprintf(candidate, sizeof(candidate), "%s/../../assets", cwd);
	QNN_TryBasedir(out, out_size, candidate);
	if (out[0])
		return;

	snprintf(out, out_size, "%s", cwd);
}

/* ── JSON extraction utilities ───────────────────────────────────── */

int QNN_JsonExtractInt(const char *line, const char *key, int fallback)
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

float QNN_JsonExtractFloat(const char *line, const char *key, float fallback)
{
	const char *match;
	const char *colon;

	match = strstr(line, key);
	if (match == NULL)
		return fallback;
	colon = strchr(match, ':');
	if (colon == NULL)
		return fallback;
	return (float)atof(colon + 1);
}

qboolean QNN_JsonExtractBool(const char *line, const char *key, qboolean fallback)
{
	const char *match;
	const char *colon;
	const char *val;

	match = strstr(line, key);
	if (match == NULL)
		return fallback;
	colon = strchr(match, ':');
	if (colon == NULL)
		return fallback;
	val = colon + 1;
	while (*val == ' ' || *val == '\t')
		val++;
	if (*val == 't' || *val == 'T' || *val == '1')
		return true;
	if (*val == 'f' || *val == 'F' || *val == '0')
		return false;
	return fallback;
}

qboolean QNN_JsonExtractString(const char *line, const char *key, char *out, size_t out_size)
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

qboolean QNN_JsonExtractVec2(const char *line, const char *key, float out[2])
{
	const char *match;
	const char *colon;
	const char *cursor;
	char *endptr;
	int axis;

	match = strstr(line, key);
	if (match == NULL)
		return false;
	colon = strchr(match, ':');
	if (colon == NULL)
		return false;
	cursor = strchr(colon, '[');
	if (cursor == NULL)
		return false;
	cursor += 1;

	for (axis = 0; axis < 2; ++axis)
	{
		while (*cursor && isspace((unsigned char)*cursor))
			cursor += 1;
		out[axis] = (float)strtod(cursor, &endptr);
		if (endptr == cursor)
			return false;
		cursor = endptr;
		while (*cursor && isspace((unsigned char)*cursor))
			cursor += 1;
		if (axis < 1)
		{
			if (*cursor != ',')
				return false;
			cursor += 1;
		}
	}

	while (*cursor && isspace((unsigned char)*cursor))
		cursor += 1;
	return *cursor == ']' ? true : false;
}

qboolean QNN_JsonExtractVec3(const char *line, const char *key, vec3_t out)
{
	const char *match;
	const char *colon;
	const char *cursor;
	char *endptr;
	int axis;

	match = strstr(line, key);
	if (match == NULL)
		return false;
	colon = strchr(match, ':');
	if (colon == NULL)
		return false;
	cursor = strchr(colon, '[');
	if (cursor == NULL)
		return false;
	cursor += 1;

	for (axis = 0; axis < 3; ++axis)
	{
		while (*cursor && isspace((unsigned char)*cursor))
			cursor += 1;
		out[axis] = (float)strtod(cursor, &endptr);
		if (endptr == cursor)
			return false;
		cursor = endptr;
		while (*cursor && isspace((unsigned char)*cursor))
			cursor += 1;
		if (axis < 2)
		{
			if (*cursor != ',')
				return false;
			cursor += 1;
		}
	}

	while (*cursor && isspace((unsigned char)*cursor))
		cursor += 1;
	return *cursor == ']' ? true : false;
}

static float QNN_ClampUnit(float value)
{
	if (value < -1.0f)
		return -1.0f;
	if (value > 1.0f)
		return 1.0f;
	return value;
}

#define QNN_LOOK_DEADZONE 0.03f
#define QNN_LOOK_BASE_COUNT 256.0f
#define QNN_LOOK_HIGH_GAIN 2.0f

static float QNN_LookCountCurve(float magnitude)
{
	float clamped;

	clamped = magnitude;
	if (clamped < 0.0f)
		clamped = 0.0f;
	if (clamped > 1.0f)
		clamped = 1.0f;
	return QNN_LOOK_BASE_COUNT * clamped * (1.0f + ((QNN_LOOK_HIGH_GAIN - 1.0f) * clamped * clamped));
}

static float QNN_LookMagnitudeFromCount(float count_magnitude)
{
	float target;
	float lo;
	float hi;
	int i;

	target = count_magnitude / QNN_LOOK_BASE_COUNT;
	if (target < 0.0f)
		target = 0.0f;
	if (target > QNN_LOOK_HIGH_GAIN)
		target = QNN_LOOK_HIGH_GAIN;
	lo = 0.0f;
	hi = 1.0f;
	for (i = 0; i < 24; ++i)
	{
		float mid = 0.5f * (lo + hi);
		float value = mid * (1.0f + ((QNN_LOOK_HIGH_GAIN - 1.0f) * mid * mid));
		if (value < target)
			lo = mid;
		else
			hi = mid;
	}
	return 0.5f * (lo + hi);
}

int QNN_MouseCountFromLookAxis(float axis)
{
	float clamped;
	float magnitude;
	float normalized;
	float sign;

	clamped = QNN_ClampUnit(axis);
	sign = clamped < 0.0f ? -1.0f : 1.0f;
	magnitude = fabsf(clamped);
	if (magnitude <= QNN_LOOK_DEADZONE)
		return 0;
	normalized = (magnitude - QNN_LOOK_DEADZONE) / (1.0f - QNN_LOOK_DEADZONE);
	return (int)roundf(sign * QNN_LookCountCurve(normalized));
}

float QNN_LookAxisFromMouseCount(int mouse_count)
{
	float normalized;
	float axis;
	float sign;

	if (mouse_count == 0)
		return 0.0f;
	sign = mouse_count < 0 ? -1.0f : 1.0f;
	normalized = QNN_LookMagnitudeFromCount((float)abs(mouse_count));
	axis = QNN_LOOK_DEADZONE + ((1.0f - QNN_LOOK_DEADZONE) * normalized);
	return sign * QNN_ClampUnit(axis);
}

int QNN_SwitchSlotFromWeaponId(int weapon_id)
{
	if (weapon_id <= 0)
		return 0;
	if (weapon_id == 2 || weapon_id == 3)
		return 1;
	if (weapon_id == 4 || weapon_id == 5)
		return 2;
	if (weapon_id == 6)
		return 3;
	if (weapon_id == 7)
		return 4;
	if (weapon_id == 8)
		return 5;
	return 0;
}

int QNN_SwitchImpulseFromSlot(int switch_slot, int weapons_owned)
{
	switch (switch_slot)
	{
	case 1:
		if (weapons_owned & IT_SUPER_SHOTGUN) return 3;
		if (weapons_owned & IT_SHOTGUN) return 2;
		return 0;
	case 2:
		if (weapons_owned & IT_SUPER_NAILGUN) return 5;
		if (weapons_owned & IT_NAILGUN) return 4;
		return 0;
	case 3:
		return (weapons_owned & IT_GRENADE_LAUNCHER) ? 6 : 0;
	case 4:
		return (weapons_owned & IT_ROCKET_LAUNCHER) ? 7 : 0;
	case 5:
		return (weapons_owned & IT_LIGHTNING) ? 8 : 0;
	default:
		return 0;
	}
}

/* ── map preparation ─────────────────────────────────────────────── */

static void QNN_CanonicalizeMap(char *out, size_t out_size, const char *requested)
{
	size_t i;

	snprintf(out, out_size, "%s", requested);
	for (i = 0; i < strlen(out); ++i)
		out[i] = (char)tolower((unsigned char)out[i]);
}

qboolean QNN_PrepareMap(const char *requested_map_id, char *error, size_t error_size)
{
	char map_name[QNN_MAX_MAP_ID];

	if (!requested_map_id || !requested_map_id[0])
	{
		snprintf(error, error_size, "map_id is required");
		return false;
	}

	QNN_CanonicalizeMap(map_name, sizeof(map_name), requested_map_id);
	if (!strcmp(qnn_map_state.requested_map_id, requested_map_id)
		&& !strcmp(qnn_map_state.map_name, map_name)
		&& qnn_map_state.region_count > 0)
		return true;

	QNN_FreeMapState(&qnn_map_state);
	if (!QNN_BuildMapState(&qnn_map_state, requested_map_id, map_name, error, error_size))
		return false;
	return true;
}

/* ── JSON output helpers ─────────────────────────────────────────── */

void QNN_WriteJsonString(FILE *out, const char *text)
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

void QNN_WriteError(const char *message)
{
	fprintf(stdout, "{\"error\":");
	QNN_WriteJsonString(stdout, message);
	fprintf(stdout, ",\"ok\":false}\n");
	fflush(stdout);
}

/* ── engine state helpers ────────────────────────────────────────── */

const char *QNN_ProgString(string_t value)
{
	if (!value)
		return "";
	return pr_strings + value;
}

static const char *QNN_ServerClassname(int entity_num)
{
	edict_t *edict;

	if (!sv.active)
		return "";
	if (entity_num < 0 || entity_num >= sv.num_edicts)
		return "";
	edict = EDICT_NUM(entity_num);
	if (edict->free || !edict->v.classname)
		return "";
	return QNN_ProgString(edict->v.classname);
}

int QNN_WeaponId(void)
{
	int active;
	int weapon_id;

	active = cl.stats[STAT_ACTIVEWEAPON];
	/* In listen server mode cl.stats[STAT_ACTIVEWEAPON] can stay zero even
	   though the server-side edict has the correct weapon field.  Fall back
	   to the authoritative server edict when the client stat is missing. */
	if (sv.active && cl.viewentity > 0 && cl.viewentity < sv.num_edicts)
	{
		edict_t *ent = EDICT_NUM(cl.viewentity);
		if (ent && !ent->free)
		{
			if (active <= 0)
				active = (int)ent->v.weapon;
			(void)0; /* weapon fallback checked */
		}
	}
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

int QNN_CurrentFrags(void)
{
	if (cl.viewentity > 0 && cl.scores != NULL && cl.viewentity - 1 < cl.maxclients)
		return cl.scores[cl.viewentity - 1].frags;
	return cl.stats[STAT_FRAGS];
}

static float QNN_CurrentArmortype(void)
{
	int items;

	items = cl.items;
	if (items & IT_ARMOR3) return 0.8f;
	if (items & IT_ARMOR2) return 0.6f;
	if (items & IT_ARMOR1) return 0.3f;
	return 0.0f;
}

/* ── observation capture ─────────────────────────────────────────── */

/* Returns true if a server edict is an "actor" — anything that moves
   autonomously and can take damage (players, bots, monsters).  Excludes
   projectiles, doors, platforms, items, and corpses. */
static qboolean QNN_IsActorEdict(const edict_t *ed)
{
	int mt;
	if (ed->free)
		return false;
	if (ed->v.takedamage == DAMAGE_NO)
		return false;
	if (ed->v.health <= 0)
		return false;
	mt = (int)ed->v.movetype;
	return mt == MOVETYPE_WALK || mt == MOVETYPE_STEP;
}

void QNN_CaptureVisibleEntities(qnn_snapshot_t *snapshot, float fixed_dt)
{
	int entity_num;
	/* Track which entity numbers were already captured from the client list
	   so the server-side actor pass doesn't duplicate them. */
	unsigned char captured[MAX_EDICTS / 8 + 1];
	memset(captured, 0, sizeof(captured));

	/* Pass 1: client-side entities (normal network-visible entities). */
	for (entity_num = 1; entity_num < cl.num_entities && snapshot->visible_count < QNN_MAX_VISIBLE; ++entity_num)
	{
		entity_t *entity;
		edict_t *server_edict;
		qnn_visible_entity_t *out_entity;
		vec3_t delta;

		entity = &cl_entities[entity_num];
		server_edict = (sv.active && entity_num >= 0 && entity_num < sv.num_edicts) ? EDICT_NUM(entity_num) : NULL;
		if (entity_num == cl.viewentity)
			continue;
		if (entity->model == NULL)
			continue;

		out_entity = &snapshot->visible[snapshot->visible_count];
		memset(out_entity, 0, sizeof(*out_entity));
		out_entity->entity_key = entity_num;
		snprintf(out_entity->entity_id, sizeof(out_entity->entity_id), "entity_%04d", entity_num);
		snprintf(out_entity->classname, sizeof(out_entity->classname), "%s", QNN_ServerClassname(entity_num));
		snprintf(out_entity->model_name, sizeof(out_entity->model_name), "%s", entity->model != NULL ? entity->model->name : "");
		out_entity->entity_num = entity_num;
		VectorCopy(entity->origin, out_entity->origin);
		VectorSubtract(entity->msg_origins[0], entity->msg_origins[1], delta);
		if (fixed_dt > 0.0f)
			VectorScale(delta, 1.0f / fixed_dt, out_entity->velocity);
		else
			VectorCopy(vec3_origin, out_entity->velocity);
		VectorCopy(entity->angles, out_entity->angles);
		out_entity->model_id = entity->baseline.modelindex > 0 ? entity->baseline.modelindex : 0;
		out_entity->frame = entity->frame;
		out_entity->effects = entity->effects;
		out_entity->skin = entity->skinnum;
		out_entity->health = (server_edict != NULL && !server_edict->free) ? (int)server_edict->v.health : 0;
		out_entity->frags = (server_edict != NULL && !server_edict->free) ? (int)server_edict->v.frags : 0;
		/* Shift origin to bounding box center and store half-extents.
		   Prefer server edict bounds (exact per-entity), fall back to
		   model bounds (works during demo playback without a server). */
		{
			vec3_t bmins = {0,0,0}, bmaxs = {0,0,0};
			qboolean have_bounds = false;
			if (server_edict != NULL && !server_edict->free) {
				VectorCopy(server_edict->v.mins, bmins);
				VectorCopy(server_edict->v.maxs, bmaxs);
				have_bounds = (bmins[0] != 0 || bmins[1] != 0 || bmins[2] != 0 ||
				               bmaxs[0] != 0 || bmaxs[1] != 0 || bmaxs[2] != 0);
			}
			if (!have_bounds && entity->model != NULL) {
				VectorCopy(entity->model->mins, bmins);
				VectorCopy(entity->model->maxs, bmaxs);
				have_bounds = (bmins[0] != 0 || bmins[1] != 0 || bmins[2] != 0 ||
				               bmaxs[0] != 0 || bmaxs[1] != 0 || bmaxs[2] != 0);
			}
			if (have_bounds) {
				out_entity->origin[0] += (bmins[0] + bmaxs[0]) * 0.5f;
				out_entity->origin[1] += (bmins[1] + bmaxs[1]) * 0.5f;
				out_entity->origin[2] += (bmins[2] + bmaxs[2]) * 0.5f;
				out_entity->half_extents[0] = (bmaxs[0] - bmins[0]) * 0.5f;
				out_entity->half_extents[1] = (bmaxs[1] - bmins[1]) * 0.5f;
				out_entity->half_extents[2] = (bmaxs[2] - bmins[2]) * 0.5f;
			}
		}
		out_entity->region_id = QNN_NearestRegionId(&qnn_map_state, out_entity->origin);
		snapshot->visible_count += 1;
		captured[entity_num / 8] |= (1 << (entity_num % 8));
	}

	/* Pass 2: server-side actors not in the client entity list.
	   FrikBots and other fake clients bypass the network protocol so they
	   never appear in cl_entities.  We read their state directly from the
	   server edict array. */
	if (sv.active)
	{
		for (entity_num = 1; entity_num < sv.num_edicts && snapshot->visible_count < QNN_MAX_VISIBLE; ++entity_num)
		{
			edict_t *ed;
			qnn_visible_entity_t *out_entity;
			const char *model_name;
			int model_idx;

			if (captured[entity_num / 8] & (1 << (entity_num % 8)))
				continue;
			if (entity_num == cl.viewentity)
				continue;

			ed = EDICT_NUM(entity_num);
			if (!QNN_IsActorEdict(ed))
				continue;

			model_idx = (int)ed->v.modelindex;
			model_name = (model_idx > 0 && model_idx < MAX_MODELS && sv.model_precache[model_idx])
				? sv.model_precache[model_idx] : "";

			out_entity = &snapshot->visible[snapshot->visible_count];
			memset(out_entity, 0, sizeof(*out_entity));
			out_entity->entity_key = entity_num;
			snprintf(out_entity->entity_id, sizeof(out_entity->entity_id), "entity_%04d", entity_num);
			snprintf(out_entity->classname, sizeof(out_entity->classname), "%s", QNN_ServerClassname(entity_num));
			snprintf(out_entity->model_name, sizeof(out_entity->model_name), "%s", model_name);
			out_entity->entity_num = entity_num;
			VectorCopy(ed->v.origin, out_entity->origin);
			VectorCopy(ed->v.velocity, out_entity->velocity);
			VectorCopy(ed->v.angles, out_entity->angles);
			out_entity->model_id = model_idx;
			out_entity->frame = (int)ed->v.frame;
			out_entity->effects = (int)ed->v.effects;
			out_entity->skin = (int)ed->v.skin;
			out_entity->health = (int)ed->v.health;
			out_entity->frags = (int)ed->v.frags;
			/* Shift origin to bounding box center and store half-extents. */
			{
				vec3_t bmins = {0,0,0}, bmaxs = {0,0,0};
				qboolean have_bounds = false;
				VectorCopy(ed->v.mins, bmins);
				VectorCopy(ed->v.maxs, bmaxs);
				have_bounds = (bmins[0] != 0 || bmins[1] != 0 || bmins[2] != 0 ||
				               bmaxs[0] != 0 || bmaxs[1] != 0 || bmaxs[2] != 0);
				if (have_bounds) {
					out_entity->origin[0] += (bmins[0] + bmaxs[0]) * 0.5f;
					out_entity->origin[1] += (bmins[1] + bmaxs[1]) * 0.5f;
					out_entity->origin[2] += (bmins[2] + bmaxs[2]) * 0.5f;
					out_entity->half_extents[0] = (bmaxs[0] - bmins[0]) * 0.5f;
					out_entity->half_extents[1] = (bmaxs[1] - bmins[1]) * 0.5f;
					out_entity->half_extents[2] = (bmaxs[2] - bmins[2]) * 0.5f;
				}
			}
			out_entity->region_id = QNN_NearestRegionId(&qnn_map_state, out_entity->origin);
			snapshot->visible_count += 1;
		}
	}
}

void QNN_CaptureBaseSnapshot(qnn_snapshot_t *snapshot)
{
	entity_t *player_entity;
	edict_t *server_edict;

	memset(snapshot, 0, sizeof(*snapshot));
	player_entity = (cl.viewentity > 0 && cl.viewentity < MAX_EDICTS) ? &cl_entities[cl.viewentity] : NULL;
	server_edict = (sv.active && cl.viewentity > 0 && cl.viewentity < sv.num_edicts) ? EDICT_NUM(cl.viewentity) : NULL;
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

	snapshot->health = cl.stats[STAT_HEALTH];
	snapshot->armor = cl.stats[STAT_ARMOR];
	snapshot->armor_type = QNN_CurrentArmortype();
	snapshot->ammo = cl.stats[STAT_AMMO];
	snapshot->ammo_shells = cl.stats[STAT_SHELLS];
	snapshot->ammo_nails = cl.stats[STAT_NAILS];
	snapshot->ammo_rockets = cl.stats[STAT_ROCKETS];
	snapshot->ammo_cells = cl.stats[STAT_CELLS];
	snapshot->weapons_owned = cl.items & (IT_SHOTGUN | IT_SUPER_SHOTGUN | IT_NAILGUN | IT_SUPER_NAILGUN | IT_GRENADE_LAUNCHER | IT_ROCKET_LAUNCHER | IT_LIGHTNING);
	snapshot->weapon_id = QNN_WeaponId();
	snapshot->waterlevel = (server_edict != NULL && !server_edict->free) ? (int)server_edict->v.waterlevel : (cl.inwater ? 2 : 0);
	snapshot->grounded = (server_edict != NULL && !server_edict->free)
		? ((((int)server_edict->v.flags) & FL_ONGROUND) ? true : false)
		: (cl.onground ? true : false);
	snapshot->current_region_id = QNN_NearestRegionId(&qnn_map_state, snapshot->player_origin);
}

void QNN_DrainSounds(qnn_snapshot_t *snapshot)
{
	snapshot->sound_count = qnn_sound_count < QNN_MAX_SOUNDS
		? qnn_sound_count : QNN_MAX_SOUNDS;
	if (snapshot->sound_count > 0)
		memcpy(snapshot->sounds, qnn_sound_buffer, snapshot->sound_count * sizeof(qnn_sound_event_t));
	qnn_sound_count = 0;
}

/* ── shared nav query handler ───────────────────────────────────── */

int QNN_HandleNavQuery(const char *line)
{
	char kind[32];
	char error[256];

	memset(kind, 0, sizeof(kind));
	memset(error, 0, sizeof(error));
	if (qnn_map_state.navmesh == NULL)
	{
		QNN_WriteError("Navmesh is unavailable for this map");
		return 0;
	}
	if (!QNN_JsonExtractString(line, "\"kind\"", kind, sizeof(kind)))
	{
		QNN_WriteError("nav_query requires kind");
		return 0;
	}

	if (!strcmp(kind, "nearest"))
	{
		vec3_t point;
		qnn_navmesh_nearest_result_t result;
		int found;

		if (!QNN_JsonExtractVec3(line, "\"point\"", point))
		{
			QNN_WriteError("nav_query nearest requires point=[x,y,z]");
			return 0;
		}
		found = qnn_navmesh_find_nearest(qnn_map_state.navmesh, point, &result, error, sizeof(error));
		if (!found && error[0] != 0)
		{
			QNN_WriteError(error);
			return 0;
		}
		fprintf(stdout, "{\"ok\":true,\"query\":\"nearest\",\"result\":");
		qnn_navmesh_write_nearest_json(stdout, &result);
		fprintf(stdout, "}\n");
		fflush(stdout);
		return 0;
	}

	if (!strcmp(kind, "path"))
	{
		vec3_t start;
		vec3_t end;
		qnn_navmesh_path_result_t result;
		int found;

		if (!QNN_JsonExtractVec3(line, "\"start\"", start)
			|| !QNN_JsonExtractVec3(line, "\"end\"", end))
		{
			QNN_WriteError("nav_query path requires start=[x,y,z] and end=[x,y,z]");
			return 0;
		}
		found = qnn_navmesh_find_path(qnn_map_state.navmesh, start, end, &result, error, sizeof(error));
		if (!found && error[0] != 0)
		{
			QNN_WriteError(error);
			return 0;
		}
		fprintf(stdout, "{\"ok\":true,\"query\":\"path\",\"result\":");
		qnn_navmesh_write_path_json(stdout, &result);
		fprintf(stdout, "}\n");
		fflush(stdout);
		return 0;
	}

	if (!strcmp(kind, "area"))
	{
		vec3_t point;
		qnn_nav_area_result_t result;
		int found;

		if (qnn_map_state.nav_oracle == NULL)
		{
			QNN_WriteError("Navigation oracle is unavailable for this map");
			return 0;
		}
		if (!QNN_JsonExtractVec3(line, "\"point\"", point))
		{
			QNN_WriteError("nav_query area requires point=[x,y,z]");
			return 0;
		}
		found = qnn_nav_oracle_find_area(qnn_map_state.nav_oracle, point, &result, error, sizeof(error));
		if (!found && error[0] != 0)
		{
			QNN_WriteError(error);
			return 0;
		}
		fprintf(stdout, "{\"ok\":true,\"query\":\"area\",\"result\":");
		qnn_nav_oracle_write_area_json(stdout, &result);
		fprintf(stdout, "}\n");
		fflush(stdout);
		return 0;
	}

	if (!strcmp(kind, "cluster"))
	{
		vec3_t point;
		qnn_nav_cluster_result_t result;
		int found;

		if (qnn_map_state.nav_oracle == NULL)
		{
			QNN_WriteError("Navigation oracle is unavailable for this map");
			return 0;
		}
		if (!QNN_JsonExtractVec3(line, "\"point\"", point))
		{
			QNN_WriteError("nav_query cluster requires point=[x,y,z]");
			return 0;
		}
		found = qnn_nav_oracle_find_cluster(qnn_map_state.nav_oracle, point, &result, error, sizeof(error));
		if (!found && error[0] != 0)
		{
			QNN_WriteError(error);
			return 0;
		}
		fprintf(stdout, "{\"ok\":true,\"query\":\"cluster\",\"result\":");
		qnn_nav_oracle_write_cluster_json(stdout, &result);
		fprintf(stdout, "}\n");
		fflush(stdout);
		return 0;
	}

	if (!strcmp(kind, "route"))
	{
		vec3_t start;
		vec3_t end;
		qnn_nav_route_result_t result;
		int found;

		if (qnn_map_state.nav_oracle == NULL)
		{
			QNN_WriteError("Navigation oracle is unavailable for this map");
			return 0;
		}
		if (!QNN_JsonExtractVec3(line, "\"start\"", start)
			|| !QNN_JsonExtractVec3(line, "\"end\"", end))
		{
			QNN_WriteError("nav_query route requires start=[x,y,z] and end=[x,y,z]");
			return 0;
		}
		found = qnn_nav_oracle_find_route(qnn_map_state.nav_oracle, start, end, &result, error, sizeof(error));
		if (!found && error[0] != 0)
		{
			QNN_WriteError(error);
			return 0;
		}
		fprintf(stdout, "{\"ok\":true,\"query\":\"route\",\"result\":");
		qnn_nav_oracle_write_route_json(stdout, &result);
		fprintf(stdout, "}\n");
		fflush(stdout);
		return 0;
	}

	QNN_WriteError("unsupported nav_query kind");
	return 0;
}

/* ── Tick resampling gate ─────────────────────────────────────────── */

void QNN_ResampleInit(int target_hz)
{
	memset(&qnn_resample, 0, sizeof(qnn_resample));
	if (target_hz > 0)
	{
		qnn_resample.target_hz = target_hz;
		qnn_resample.target_dt = 1.0f / (float)target_hz;
	}
}

void QNN_ResampleAccumulate(const qnn_snapshot_t *snapshot, float frame_dt)
{
	qnn_resample.accumulated_dt += frame_dt;

	/* Merge discrete actions across the window (OR — any press counts). */
	if (snapshot->action_label.fire)
		qnn_resample.fire_any = 1;
	if (snapshot->action_label.jump)
		qnn_resample.jump_any = 1;
}

void QNN_ResampleAccumulateLook(float yaw_degrees, float pitch_degrees)
{
	qnn_resample.look_yaw_degrees += yaw_degrees;
	qnn_resample.look_pitch_degrees += pitch_degrees;
}

qboolean QNN_ResampleShouldEmit(void)
{
	if (qnn_resample.target_hz <= 0)
		return true; /* disabled — emit every frame */

	if (qnn_resample.accumulated_dt >= qnn_resample.target_dt)
		return true;

	return false;
}

void QNN_ResampleApplyActionMerge(qnn_snapshot_t *snapshot)
{
	/* Apply the OR-merged discrete actions to the snapshot being emitted. */
	if (qnn_resample.fire_any)
		snapshot->action_label.fire = 1;
	if (qnn_resample.jump_any)
		snapshot->action_label.jump = 1;

	/* Look deltas are now computed at emission time by infer_action using
	 * the full emission-to-emission window.  No accumulator override needed. */

	/* Reset accumulators for next window. */
	qnn_resample.accumulated_dt = 0.0f;
	qnn_resample.fire_any = 0;
	qnn_resample.jump_any = 0;
	qnn_resample.look_yaw_degrees = 0.0f;
	qnn_resample.look_pitch_degrees = 0.0f;
}

/* ── Binary write helpers (little-endian) ────────────────────────── */

void QNN_WriteU16LE(FILE *out, uint16_t value)
{
	uint8_t b[2];
	b[0] = (uint8_t)(value & 0xff);
	b[1] = (uint8_t)((value >> 8) & 0xff);
	fwrite(b, 1, 2, out);
}

void QNN_WriteI16LE(FILE *out, int value)
{
	QNN_WriteU16LE(out, (uint16_t)(int16_t)value);
}

void QNN_WriteU32LE(FILE *out, uint32_t value)
{
	uint8_t b[4];
	b[0] = (uint8_t)(value & 0xff);
	b[1] = (uint8_t)((value >> 8) & 0xff);
	b[2] = (uint8_t)((value >> 16) & 0xff);
	b[3] = (uint8_t)((value >> 24) & 0xff);
	fwrite(b, 1, 4, out);
}

void QNN_WriteI32LE(FILE *out, int32_t value)
{
	QNN_WriteU32LE(out, (uint32_t)value);
}

void QNN_WriteF32LE(FILE *out, float value)
{
	union { float f; uint32_t u; } bits;
	bits.f = value;
	QNN_WriteU32LE(out, bits.u);
}
