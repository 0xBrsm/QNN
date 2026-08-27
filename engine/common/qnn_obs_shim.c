/*
 * qnn_obs_shim.c — Obs-API protocol layer + legacy wire-identity shim.
 *
 * See qnn_obs_shim.h for the schema, the reply format and the framing.
 * The JSON parser below is deliberately minimal — this ONE fixed
 * obs_api v1 schema, nothing generic — but strict: unknown keys,
 * duplicates, wrong types, missing required params and trailing bytes
 * are all hard errors.  No engine dependencies (standalone-linkable
 * with qnn_obs_registry.c for the registry test).
 */

#include "qnn_obs_shim.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* ── Minimal strict JSON cursor ─────────────────────────────────── */

typedef struct {
	const char *start;
	const char *p;
	const char *end;
	char       *error;
	size_t      error_size;
} qnn_obs_json_t;

static qboolean QNN_ObsJsonFail(qnn_obs_json_t *j, const char *fmt, ...)
{
	va_list args;

	va_start(args, fmt);
	vsnprintf(j->error, j->error_size, fmt, args);
	va_end(args);
	return false;
}

static void QNN_ObsJsonSkipWs(qnn_obs_json_t *j)
{
	while (j->p < j->end && (*j->p == ' ' || *j->p == '\t'
		|| *j->p == '\n' || *j->p == '\r'))
		j->p++;
}

static int QNN_ObsJsonPeek(qnn_obs_json_t *j)
{
	QNN_ObsJsonSkipWs(j);
	return j->p < j->end ? (unsigned char)*j->p : -1;
}

static qboolean QNN_ObsJsonExpect(qnn_obs_json_t *j, char c)
{
	if (QNN_ObsJsonPeek(j) != (unsigned char)c)
		return QNN_ObsJsonFail(j, "declaration JSON: expected '%c' at "
			"offset %d", c, (int)(j->p - j->start));
	j->p++;
	return true;
}

/* Plain string, no escape support (the schema's strings are field /
 * policy identifiers; an escape means a malformed declaration). */
static qboolean QNN_ObsJsonString(qnn_obs_json_t *j, char *dst,
	size_t dst_size, const char *what)
{
	size_t n = 0;

	if (QNN_ObsJsonPeek(j) != '"')
		return QNN_ObsJsonFail(j, "declaration JSON: %s must be a string",
			what);
	j->p++;
	while (j->p < j->end && *j->p != '"')
	{
		unsigned char c = (unsigned char)*j->p;

		if (c == '\\' || c < 0x20)
			return QNN_ObsJsonFail(j, "declaration JSON: unsupported "
				"escape/control byte in %s", what);
		if (n + 1 >= dst_size)
			return QNN_ObsJsonFail(j, "declaration JSON: %s longer than "
				"%d bytes", what, (int)dst_size - 1);
		dst[n++] = (char)c;
		j->p++;
	}
	if (j->p >= j->end)
		return QNN_ObsJsonFail(j, "declaration JSON: unterminated %s",
			what);
	j->p++;   /* closing quote */
	dst[n] = '\0';
	return true;
}

static qboolean QNN_ObsJsonInt(qnn_obs_json_t *j, int *out, const char *what)
{
	long value = 0;
	int negative = 0;
	int digits = 0;

	QNN_ObsJsonSkipWs(j);
	if (j->p < j->end && *j->p == '-')
	{
		negative = 1;
		j->p++;
	}
	while (j->p < j->end && *j->p >= '0' && *j->p <= '9')
	{
		value = value * 10 + (*j->p - '0');
		if (value > 0x7fffffffL)
			return QNN_ObsJsonFail(j, "declaration JSON: %s out of range",
				what);
		digits++;
		j->p++;
	}
	if (digits == 0)
		return QNN_ObsJsonFail(j, "declaration JSON: %s must be an integer",
			what);
	if (j->p < j->end && (*j->p == '.' || *j->p == 'e' || *j->p == 'E'))
		return QNN_ObsJsonFail(j, "declaration JSON: %s must be an integer "
			"(got a float)", what);
	*out = (int)(negative ? -value : value);
	return true;
}

static qboolean QNN_ObsJsonLiteral(qnn_obs_json_t *j, const char *literal)
{
	size_t n = strlen(literal);

	QNN_ObsJsonSkipWs(j);
	if ((size_t)(j->end - j->p) < n || strncmp(j->p, literal, n) != 0)
		return false;
	j->p += n;
	return true;
}

static qboolean QNN_ObsJsonBool(qnn_obs_json_t *j, qboolean *out,
	const char *what)
{
	if (QNN_ObsJsonLiteral(j, "true"))  { *out = true;  return true; }
	if (QNN_ObsJsonLiteral(j, "false")) { *out = false; return true; }
	return QNN_ObsJsonFail(j, "declaration JSON: %s must be true or false",
		what);
}

/* ── Sub-object parsers ─────────────────────────────────────────── */

static qboolean QNN_ObsJsonAtlas(qnn_obs_json_t *j, qnn_obs_decl_t *out)
{
	qboolean seen_yaw = false, seen_bands = false, seen_packed = false;
	char key[32];

	if (!QNN_ObsJsonExpect(j, '{'))
		return false;
	if (QNN_ObsJsonPeek(j) == '}')
		{ j->p++; goto done; }
	for (;;)
	{
		if (!QNN_ObsJsonString(j, key, sizeof(key), "atlas param name"))
			return false;
		if (!QNN_ObsJsonExpect(j, ':'))
			return false;
		if (strcmp(key, "yaw") == 0)
		{
			if (seen_yaw)
				return QNN_ObsJsonFail(j, "declaration JSON: duplicate "
					"atlas param \"yaw\"");
			seen_yaw = true;
			if (!QNN_ObsJsonInt(j, &out->atlas.yaw, "atlas.yaw"))
				return false;
		}
		else if (strcmp(key, "bands") == 0)
		{
			if (seen_bands)
				return QNN_ObsJsonFail(j, "declaration JSON: duplicate "
					"atlas param \"bands\"");
			seen_bands = true;
			if (!QNN_ObsJsonInt(j, &out->atlas.bands, "atlas.bands"))
				return false;
		}
		else if (strcmp(key, "packed") == 0)
		{
			if (seen_packed)
				return QNN_ObsJsonFail(j, "declaration JSON: duplicate "
					"atlas param \"packed\"");
			seen_packed = true;
			if (!QNN_ObsJsonBool(j, &out->atlas.packed, "atlas.packed"))
				return false;
		}
		else
			return QNN_ObsJsonFail(j, "declaration JSON: unknown atlas "
				"param \"%s\"", key);
		if (QNN_ObsJsonPeek(j) == ',') { j->p++; continue; }
		if (!QNN_ObsJsonExpect(j, '}'))
			return false;
		break;
	}
done:
	if (!seen_yaw || !seen_bands || !seen_packed)
		return QNN_ObsJsonFail(j, "declaration JSON: atlas requires "
			"yaw, bands and packed (no defaults)");
	out->atlas_requested = true;
	return true;
}

static qboolean QNN_ObsJsonEntities(qnn_obs_json_t *j, qnn_obs_decl_t *out)
{
	qboolean seen_policy = false, seen_max = false, seen_paths = false;
	char key[32];

	if (!QNN_ObsJsonExpect(j, '{'))
		return false;
	if (QNN_ObsJsonPeek(j) == '}')
		{ j->p++; goto done; }
	for (;;)
	{
		if (!QNN_ObsJsonString(j, key, sizeof(key), "entities param name"))
			return false;
		if (!QNN_ObsJsonExpect(j, ':'))
			return false;
		if (strcmp(key, "policy") == 0)
		{
			if (seen_policy)
				return QNN_ObsJsonFail(j, "declaration JSON: duplicate "
					"entities param \"policy\"");
			seen_policy = true;
			if (!QNN_ObsJsonString(j, out->entities.policy,
					sizeof(out->entities.policy), "entities.policy"))
				return false;
		}
		else if (strcmp(key, "max_tokens") == 0)
		{
			if (seen_max)
				return QNN_ObsJsonFail(j, "declaration JSON: duplicate "
					"entities param \"max_tokens\"");
			seen_max = true;
			if (!QNN_ObsJsonInt(j, &out->entities.max_tokens,
					"entities.max_tokens"))
				return false;
		}
		else if (strcmp(key, "paths") == 0)
		{
			if (seen_paths)
				return QNN_ObsJsonFail(j, "declaration JSON: duplicate "
					"entities param \"paths\"");
			seen_paths = true;
			if (!QNN_ObsJsonBool(j, &out->entities.paths, "entities.paths"))
				return false;
		}
		else
			return QNN_ObsJsonFail(j, "declaration JSON: unknown entities "
				"param \"%s\"", key);
		if (QNN_ObsJsonPeek(j) == ',') { j->p++; continue; }
		if (!QNN_ObsJsonExpect(j, '}'))
			return false;
		break;
	}
done:
	if (!seen_policy || !seen_max || !seen_paths)
		return QNN_ObsJsonFail(j, "declaration JSON: entities requires "
			"policy, max_tokens and paths (no defaults)");
	out->entities_requested = true;
	return true;
}

static qboolean QNN_ObsJsonStateList(qnn_obs_json_t *j, qnn_obs_decl_t *out)
{
	if (!QNN_ObsJsonExpect(j, '['))
		return false;
	if (QNN_ObsJsonPeek(j) == ']')
		{ j->p++; return true; }
	for (;;)
	{
		if (out->state_count >= QNN_OBS_MAX_STATE_FIELDS)
			return QNN_ObsJsonFail(j, "declaration JSON: state list longer "
				"than %d entries", QNN_OBS_MAX_STATE_FIELDS);
		if (!QNN_ObsJsonString(j, out->state[out->state_count],
				QNN_OBS_MAX_FIELD_NAME, "state field name"))
			return false;
		out->state_count++;
		if (QNN_ObsJsonPeek(j) == ',') { j->p++; continue; }
		if (!QNN_ObsJsonExpect(j, ']'))
			return false;
		return true;
	}
}

/* ── The declaration parser ─────────────────────────────────────── */

qboolean QNN_ObsDeclParseJson(const char *json, int len,
	qnn_obs_decl_t *out, char *error, size_t error_size)
{
	qnn_obs_json_t j;
	qboolean seen_obs_api = false, seen_state = false;
	qboolean seen_atlas = false, seen_entities = false, seen_frame = false;
	char key[32];

	memset(out, 0, sizeof(*out));
	if (json == NULL)
	{
		snprintf(error, error_size, "declaration JSON: NULL payload");
		return false;
	}
	j.start = json;
	j.p = json;
	j.end = json + (len < 0 ? (int)strlen(json) : len);
	j.error = error;
	j.error_size = error_size;

	if (!QNN_ObsJsonExpect(&j, '{'))
		return false;
	if (QNN_ObsJsonPeek(&j) == '}')
	{
		j.p++;
		goto closed;
	}
	for (;;)
	{
		if (!QNN_ObsJsonString(&j, key, sizeof(key), "declaration key"))
			return false;
		if (!QNN_ObsJsonExpect(&j, ':'))
			return false;
		if (strcmp(key, "obs_api") == 0)
		{
			if (seen_obs_api)
				return QNN_ObsJsonFail(&j, "declaration JSON: duplicate "
					"key \"obs_api\"");
			seen_obs_api = true;
			if (!QNN_ObsJsonInt(&j, &out->obs_api, "obs_api"))
				return false;
		}
		else if (strcmp(key, "state") == 0)
		{
			if (seen_state)
				return QNN_ObsJsonFail(&j, "declaration JSON: duplicate "
					"key \"state\"");
			seen_state = true;
			if (!QNN_ObsJsonStateList(&j, out))
				return false;
		}
		else if (strcmp(key, "atlas") == 0)
		{
			if (seen_atlas)
				return QNN_ObsJsonFail(&j, "declaration JSON: duplicate "
					"key \"atlas\"");
			seen_atlas = true;
			if (QNN_ObsJsonLiteral(&j, "null"))
				out->atlas_requested = false;
			else if (!QNN_ObsJsonAtlas(&j, out))
				return false;
		}
		else if (strcmp(key, "entities") == 0)
		{
			if (seen_entities)
				return QNN_ObsJsonFail(&j, "declaration JSON: duplicate "
					"key \"entities\"");
			seen_entities = true;
			if (QNN_ObsJsonLiteral(&j, "null"))
				out->entities_requested = false;
			else if (!QNN_ObsJsonEntities(&j, out))
				return false;
		}
		else if (strcmp(key, "frame_bytes") == 0)
		{
			if (seen_frame)
				return QNN_ObsJsonFail(&j, "declaration JSON: duplicate "
					"key \"frame_bytes\"");
			seen_frame = true;
			if (!QNN_ObsJsonInt(&j, &out->frame_bytes_override,
					"frame_bytes"))
				return false;
			if (out->frame_bytes_override <= 0)
				return QNN_ObsJsonFail(&j, "declaration JSON: frame_bytes "
					"must be a positive integer (got %d)",
					out->frame_bytes_override);
		}
		else
			return QNN_ObsJsonFail(&j, "declaration JSON: unknown key "
				"\"%s\"", key);
		if (QNN_ObsJsonPeek(&j) == ',') { j.p++; continue; }
		if (!QNN_ObsJsonExpect(&j, '}'))
			return false;
		break;
	}
closed:
	if (QNN_ObsJsonPeek(&j) != -1)
		return QNN_ObsJsonFail(&j, "declaration JSON: trailing bytes after "
			"the closing brace");
	if (!seen_obs_api)
		return QNN_ObsJsonFail(&j, "declaration JSON: missing \"obs_api\"");
	if (!seen_state)
		return QNN_ObsJsonFail(&j, "declaration JSON: missing \"state\"");
	return true;
}

/* ── Layout-reply serialization ─────────────────────────────────── */

static qboolean QNN_ObsReplyAppend(char *out, size_t out_size, size_t *off,
	char *error, size_t error_size, const char *fmt, ...)
{
	va_list args;
	int wrote;

	if (*off >= out_size)
		goto overflow;
	va_start(args, fmt);
	wrote = vsnprintf(out + *off, out_size - *off, fmt, args);
	va_end(args);
	if (wrote < 0 || (size_t)wrote >= out_size - *off)
		goto overflow;
	*off += (size_t)wrote;
	return true;
overflow:
	snprintf(error, error_size,
		"layout reply exceeds its %d-byte buffer", (int)out_size);
	return false;
}

qboolean QNN_ObsLayoutReplyJson(const qnn_obs_plan_t *plan,
	char *out, size_t out_size, char *error, size_t error_size)
{
	size_t off = 0;
	int i, d;

#define APPEND(...) \
	do { \
		if (!QNN_ObsReplyAppend(out, out_size, &off, error, error_size, \
				__VA_ARGS__)) \
			return false; \
	} while (0)

	APPEND("{\"ok\":true,\"layout\":{\"frame_bytes\":%d,\"fields\":[",
		plan->frame_bytes);
	for (i = 0; i < plan->step_count; ++i)
	{
		const qnn_obs_plan_step_t *step = &plan->steps[i];
		const qnn_obs_registry_entry_t *entry = step->entry;

		APPEND("%s{\"name\":\"%s\",\"kind\":", i ? "," : "", entry->name);
		switch (entry->kind)
		{
		case QNN_OBS_KIND_STATE:
			APPEND("\"state\",\"params\":{}");
			break;
		case QNN_OBS_KIND_SENSOR:
			APPEND("\"sensor\",\"params\":{\"yaw\":%d,\"bands\":%d,"
				"\"packed\":%s}",
				step->params.atlas.yaw, step->params.atlas.bands,
				step->params.atlas.packed ? "true" : "false");
			break;
		case QNN_OBS_KIND_PERCEPT:
			APPEND("\"percept\",\"params\":{\"policy\":\"%s\","
				"\"max_tokens\":%d,\"paths\":%s}",
				step->params.entities.policy,
				step->params.entities.max_tokens,
				step->params.entities.paths ? "true" : "false");
			break;
		}
		APPEND(",\"offset\":%d,\"bytes\":%d,\"dtype\":",
			step->offset, step->bytes);
		if (entry->kind == QNN_OBS_KIND_PERCEPT)
		{
			/* Variable-length stream — no wire dtype/shape (matches
			 * qnn/obs_api.py compile_layout). */
			APPEND("null,\"shape\":null}");
		}
		else
		{
			int shape[QNN_OBS_MAX_SHAPE_DIMS];
			int ndim = 0;

			entry->shape_fn(&step->params, shape, &ndim);
			APPEND("\"%s\",\"shape\":[", entry->dtype);
			for (d = 0; d < ndim; ++d)
				APPEND("%s%d", d ? "," : "", shape[d]);
			APPEND("]}");
		}
	}
	APPEND("]}}");
#undef APPEND
	return true;
}

/* ── The wire-identity shim table ────────────────────────────────────
 *
 * The four stamped legacy frame identities and the declarations they
 * are equivalent to.  Sources of truth: the codec registry in
 * qnn_onnx.c (which spatial block + entity stream each id consumes)
 * and the pre-obs-api flat-frame constants:
 *
 *   wire.12.1  a26 rc1  — 72×11 unpacked atlas, FULL entity stream
 *              (policy "v1"), all 13 state fields.  Its generation's
 *              flat frame was the fixed pre-packing
 *              QNN_OBS_BUFFER_SIZE of 4096 (payload variable, tail
 *              zero) — not derivable from field sizes, hence the
 *              explicit frame_bytes override.
 *   wire.12.2  a26 rc2  — 24×11 nibble-packed atlas, FULL entity
 *              stream, all 13 state fields; compiles to today's
 *              864-byte frame with no override.
 *   wire.13.1  a27 rc1  — 72×11 unpacked atlas, COMBAT entity stream
 *              (policy "v3"), state WITHOUT self_weapon_id (the A27
 *              pure-combat substrate folds select-and-fire into the
 *              9-way attack head).  Same 4096 fixed-frame generation
 *              as wire.12.1.
 *   wire.13.2  a27 frontier — 24×11 packed atlas, COMBAT entity
 *              stream, state without self_weapon_id.
 *
 * Semantics pairing mirrors the codec table: wire.12.x carry
 * semantics.1, wire.13.x semantics.2.  Bare "wire.12"/"wire.13" are
 * deliberately ABSENT — they stay ambiguous and refused (the
 * qnn_onnx.c retired-wire table names the re-stamp fix). */

typedef struct {
	const char *wire_id;
	const char *semantics_contract;
	int         atlas_yaw;
	qboolean    atlas_packed;
	const char *entity_policy;
	qboolean    include_weapon_id;
	int         frame_bytes_override;   /* 0 = compiled minimum */
} qnn_obs_wire_shim_t;

static const qnn_obs_wire_shim_t QNN_OBS_WIRE_SHIMS[] = {
	{ "wire.12.1", "semantics.1", 72, false, "v1", true,  4096 },
	{ "wire.12.2", "semantics.1", 24, true,  "v1", true,  0    },
	{ "wire.13.1", "semantics.2", 72, false, "v3", false, 4096 },
	{ "wire.13.2", "semantics.2", 24, true,  "v3", false, 0    },
};
#define QNN_OBS_N_WIRE_SHIMS \
	((int)(sizeof(QNN_OBS_WIRE_SHIMS) / sizeof(QNN_OBS_WIRE_SHIMS[0])))

qboolean QNN_ObsShimDeclForWire(const char *wire_id, qnn_obs_decl_t *out,
	const char **semantics_contract, char *error, size_t error_size)
{
	const qnn_obs_wire_shim_t *shim = NULL;
	qnn_obs_decl_t decl;
	int i;

	for (i = 0; i < QNN_OBS_N_WIRE_SHIMS; ++i)
	{
		if (wire_id != NULL
			&& strcmp(QNN_OBS_WIRE_SHIMS[i].wire_id, wire_id) == 0)
		{
			shim = &QNN_OBS_WIRE_SHIMS[i];
			break;
		}
	}
	if (shim == NULL)
	{
		snprintf(error, error_size,
			"no wire-shim declaration for wire_contract=%s (shimmed: "
			"wire.12.1, wire.12.2, wire.13.1, wire.13.2; new models "
			"carry an obs_declaration instead)",
			wire_id != NULL ? wire_id : "(none)");
		return false;
	}

	/* Start from the default declaration (all 13 state rows in wire
	 * order) and specialize. */
	QNN_ObsDeclDefault(&decl);
	if (!shim->include_weapon_id)
	{
		int w = 0;

		for (i = 0; i < decl.state_count; ++i)
		{
			if (strcmp(decl.state[i], "self_weapon_id") == 0)
				continue;
			if (w != i)
				memcpy(decl.state[w], decl.state[i], QNN_OBS_MAX_FIELD_NAME);
			w++;
		}
		decl.state_count = w;
	}
	decl.atlas.yaw = shim->atlas_yaw;
	decl.atlas.packed = shim->atlas_packed;
	snprintf(decl.entities.policy, sizeof(decl.entities.policy), "%s",
		shim->entity_policy);
	decl.frame_bytes_override = shim->frame_bytes_override;

	*out = decl;
	if (semantics_contract != NULL)
		*semantics_contract = shim->semantics_contract;
	return true;
}
