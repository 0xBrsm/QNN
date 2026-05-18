/*
 * qnn_sys_common.c — Engine-agnostic system/IO utilities shared by
 * nq/qnn_sys.c and qw/qnn_sys.c.
 *
 * Contents: JSON extraction, look/switch helpers, JSON string output,
 * nav query handler, resample state machine, little-endian binary
 * writers.  Engine-specific Sys_* stubs, QNN_ProgString, and the
 * global declarations stay in the per-engine qnn_sys.c files.
 */

#include "qnn.h"
#include "qnn_route.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

/* ── JSON extraction utilities ───────────────────────────────────── */

static const char *QNN_JsonFindKey(const char *line, const char *key)
{
	const char *match;
	const char *next;

	match = NULL;
	next = line;
	while ((next = strstr(next, key)) != NULL)
	{
		match = next;
		next += 1;
	}
	return match;
}

int QNN_JsonExtractInt(const char *line, const char *key, int fallback)
{
	const char *match;
	const char *colon;

	match = QNN_JsonFindKey(line, key);
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

	match = QNN_JsonFindKey(line, key);
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

	match = QNN_JsonFindKey(line, key);
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

	match = QNN_JsonFindKey(line, key);
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

/* TODO(ppo): tune deadzone as eval-time parameter sweep once PPO is
   training.  0.03 ate 43% of non-zero yaw and 72% of non-zero pitch
   ticks.  Zero for now so BC fine-aim signal passes through. */
#define QNN_LOOK_DEADZONE 0.0f
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

/* Map weapon-select impulse (1-8) → IT_* bit-flag.  Game-agnostic per
 * QC convention: impulse 1 is always axe, 8 always lightning, etc. */
int QNN_ItemFlagFromImpulse(int impulse)
{
	switch (impulse)
	{
		case 1: return IT_AXE;
		case 2: return IT_SHOTGUN;
		case 3: return IT_SUPER_SHOTGUN;
		case 4: return IT_NAILGUN;
		case 5: return IT_SUPER_NAILGUN;
		case 6: return IT_GRENADE_LAUNCHER;
		case 7: return IT_ROCKET_LAUNCHER;
		case 8: return IT_LIGHTNING;
		default: return 0;
	}
}

/* QNN_NextWeaponId lives in per-game qnn_self.c — items source
 * differs between NQ (cl.items) and QW (cl.stats[STAT_ITEMS]). */

/* map preparation moved to qnn_io.c */

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

/* Engine state helpers (QNN_ProgString, QNN_WeaponId, QNN_CurrentFrags,
   QNN_CaptureBaseSnapshot, QNN_CaptureKnownEntities, QNN_DrainSounds)
   moved to qnn_object.c */

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
		qnn_route_area_result_t result;
		int found;

		if (qnn_map_state.route == NULL)
		{
			QNN_WriteError("Navigation oracle is unavailable for this map");
			return 0;
		}
		if (!QNN_JsonExtractVec3(line, "\"point\"", point))
		{
			QNN_WriteError("nav_query area requires point=[x,y,z]");
			return 0;
		}
		found = QNN_RouteFindArea(qnn_map_state.route, point, &result, error, sizeof(error));
		if (!found && error[0] != 0)
		{
			QNN_WriteError(error);
			return 0;
		}
		fprintf(stdout, "{\"ok\":true,\"query\":\"area\",\"result\":");
		QNN_RouteWriteAreaJson(stdout, &result);
		fprintf(stdout, "}\n");
		fflush(stdout);
		return 0;
	}

	if (!strcmp(kind, "cluster"))
	{
		vec3_t point;
		qnn_route_cluster_result_t result;
		int found;

		if (qnn_map_state.route == NULL)
		{
			QNN_WriteError("Navigation oracle is unavailable for this map");
			return 0;
		}
		if (!QNN_JsonExtractVec3(line, "\"point\"", point))
		{
			QNN_WriteError("nav_query cluster requires point=[x,y,z]");
			return 0;
		}
		found = QNN_RouteFindCluster(qnn_map_state.route, point, &result, error, sizeof(error));
		if (!found && error[0] != 0)
		{
			QNN_WriteError(error);
			return 0;
		}
		fprintf(stdout, "{\"ok\":true,\"query\":\"cluster\",\"result\":");
		QNN_RouteWriteClusterJson(stdout, &result);
		fprintf(stdout, "}\n");
		fflush(stdout);
		return 0;
	}

	if (!strcmp(kind, "route"))
	{
		vec3_t start;
		vec3_t end;
		qnn_route_route_result_t result;
		int found;

		if (qnn_map_state.route == NULL)
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
		found = QNN_RouteFind(qnn_map_state.route, start, end, &result, error, sizeof(error));
		if (!found && error[0] != 0)
		{
			QNN_WriteError(error);
			return 0;
		}
		fprintf(stdout, "{\"ok\":true,\"query\":\"route\",\"result\":");
		QNN_RouteWriteRouteJson(stdout, &result);
		fprintf(stdout, "}\n");
		fflush(stdout);
		return 0;
	}

	QNN_WriteError("unsupported nav_query kind");
	return 0;
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
