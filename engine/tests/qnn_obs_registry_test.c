/*
 * qnn_obs_registry_test.c — Gate 1 for the obs API (agents/plans/obs-api.md).
 *
 * Proves the plan-compiled emit of a policy-v1 plan is byte-identical
 * to the pre-refactor hardcoded QNN_IOPackObsBuffer (kept verbatim
 * below as Gate1_LegacyPackObsBuffer — the FULL, wire-shim row shape)
 * over synthetic tick results, AND that the DEFAULT plan (policy v3,
 * the A27 pure-combat stream) is byte-identical to the ported
 * bb27a296 combat row writer (Gate1_CombatPackObsBuffer — no recency,
 * actor/projectile only).  v1 and v3 diverged from one shared (v1-
 * shaped) row writer at the WS1/WS2 obs-api landing; this file gates
 * both shapes independently now that qnn_obs_registry.c serializes
 * them with their own writers (QNN_ObsEmitEntityRowV1/V3).  Also
 * exercises the compiler's fail-loud paths, the restored atlas
 * (72, unpacked) constructor, and the demand-driven compute property
 * (a plan without entities never reaches the oracle → never the
 * pathfinder).
 *
 * Standalone main — links only qnn_obs_registry.c; built by
 * src/engine/build/build_obs_registry_test.sh.
 */

#include "qnn_obs_registry.h"
#include "qnn_obs_shim.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks_run;
static int checks_failed;

#define CHECK(cond, ...) \
	do { \
		checks_run++; \
		if (!(cond)) { \
			checks_failed++; \
			fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
			fprintf(stderr, __VA_ARGS__); \
			fprintf(stderr, "\n"); \
		} \
	} while (0)

/* ── Deterministic PRNG (no libc rand — reproducible everywhere) ── */

static uint32_t rng_state = 0x5eed1234u;

static uint32_t RngU32(void)
{
	/* xorshift32 */
	uint32_t x = rng_state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	rng_state = x;
	return x;
}

static int RngRange(int lo, int hi)   /* inclusive */
{
	return lo + (int)(RngU32() % (uint32_t)(hi - lo + 1));
}

static float RngFloat(float lo, float hi)
{
	return lo + (hi - lo) * ((float)(RngU32() & 0xffffffu) / 16777215.0f);
}

/* ══════════════════════════════════════════════════════════════════
 * The pre-refactor packer, VERBATIM (feat/a27 qnn_io.c before the
 * plan refactor).  This is the Gate-1 reference — do not "improve".
 * ══════════════════════════════════════════════════════════════════ */

static int Gate1_PackEventSlots(uint8_t *obs, int pos,
	const qnn_token_event_t *events, int event_count)
{
	int j;
	int cap = event_count;
	if (cap > QNN_MAX_ENTITY_EVENTS) cap = QNN_MAX_ENTITY_EVENTS;
	if (cap < 0) cap = 0;

	obs[pos++] = (uint8_t)cap;
	for (j = 0; j < cap; ++j)
	{
		obs[pos++] = (uint8_t)events[j].action_id;
		obs[pos++] = (uint8_t)events[j].source_id;
	}
	return pos;
}

static void Gate1_LegacyPackObsBuffer(uint8_t *obs, const qnn_tick_result_t *r)
{
	int i;
	int pos;

	memset(obs, 0, QNN_OBS_BUFFER_SIZE);

	/* ── Self block (27 B) ───────────────────────────────────── */
	{
		const qnn_self_token_t *tok = &r->self;
		float eff_armor = (float)tok->raw_armor * tok->armor_type;

		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_HEALTH,
			QNN_QuantizeU8Saturating((float)tok->health));
		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_EFF_ARMOR,
			QNN_QuantizeU8Saturating(eff_armor));
		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_AMMO_SHELLS,
			QNN_QuantizeU8Saturating((float)tok->ammo_shells));
		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_AMMO_NAILS,
			QNN_QuantizeU8Saturating((float)tok->ammo_nails));
		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_AMMO_ROCKETS,
			QNN_QuantizeU8Saturating((float)tok->ammo_rockets));
		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_AMMO_CELLS,
			QNN_QuantizeU8Saturating((float)tok->ammo_cells));

		QNN_BufWriteI16(obs, QNN_OBS_OFF_SELF_VEL + 0,
			QNN_QuantizeI16Clamped(tok->vel[0], QNN_VELOCITY_SCALE));
		QNN_BufWriteI16(obs, QNN_OBS_OFF_SELF_VEL + 2,
			QNN_QuantizeI16Clamped(tok->vel[1], QNN_VELOCITY_SCALE));
		QNN_BufWriteI16(obs, QNN_OBS_OFF_SELF_VEL + 4,
			QNN_QuantizeI16Clamped(tok->vel[2], QNN_VELOCITY_SCALE));

		QNN_BufWriteF16(obs, QNN_OBS_OFF_SELF_ATTACK_FIN, tok->attack_finished);

		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_WEAPON_ID,
			(uint8_t)tok->weapon_id);
		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_MOVEMENT_ID,
			(uint8_t)tok->movement_id);

		QNN_BufWriteI32(obs, QNN_OBS_OFF_SELF_ITEMS, tok->items);

		QNN_BufWriteI8 (obs, QNN_OBS_OFF_SELF_VIEW_PITCH,
			QNN_QuantizeI8(tok->view_pitch));

		QNN_BufWriteF16(obs, QNN_OBS_OFF_SELF_LOOK_DELTA + 0, tok->look_delta[0]);
		QNN_BufWriteF16(obs, QNN_OBS_OFF_SELF_LOOK_DELTA + 2, tok->look_delta[1]);
		QNN_BufWriteF16(obs, QNN_OBS_OFF_SELF_LOOK_DELTA + 4, tok->look_delta[2]);
	}

	/* ── Spatial block (132 B — 24x11 depth-atlas codes, nibble-packed).
	 * Low nibble = even yaw cell; high nibble = odd yaw cell. Matches
	 * qnn/wire.py's _unpack_native_spatial and the ONNX scratch packer. */
	for (i = 0; i < QNN_OBS_ATLAS_ELEVS; ++i)
		QNN_AtlasPackRow(
			&obs[QNN_OBS_OFF_SPATIAL + i * QNN_OBS_ATLAS_PACKED_BYTES],
			r->spatial_atlas[i]);

	/* ── Entity stream (variable-length, native widths) ──────── */
	{
		int pack_count = r->entity_count < QNN_MAX_TOKEN_OBJECTS
			? r->entity_count : QNN_MAX_TOKEN_OBJECTS;
		pos = QNN_OBS_OFF_ENTITY_STREAM;
		obs[pos++] = (uint8_t)pack_count;

		for (i = 0; i < pack_count; ++i)
		{
			const qnn_tagged_token_t *tt = &r->entities[i];
			obs[pos++] = (uint8_t)tt->type;

			switch (tt->type)
			{
			case QNN_TOKEN_PROJECTILE:
				{
					const qnn_projectile_token_t *tok = &tt->projectile;
					obs[pos++] = (uint8_t)tok->subject_id;
					obs[pos++] = (uint8_t)tok->modality_id;
					pos = Gate1_PackEventSlots(obs, pos, tok->events, tok->event_count);
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[0], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[1], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[2], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->vel[0], QNN_VELOCITY_SCALE)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->vel[1], QNN_VELOCITY_SCALE)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->vel[2], QNN_VELOCITY_SCALE)); pos += 2;
					QNN_BufWriteF16(obs, pos, tok->recency); pos += 2;
				}
				break;

			case QNN_TOKEN_ACTOR:
				{
					const qnn_actor_token_t *tok = &tt->actor;
					obs[pos++] = (uint8_t)tok->subject_id;
					obs[pos++] = (uint8_t)tok->modality_id;
					obs[pos++] = (uint8_t)tok->player_id;
					pos = Gate1_PackEventSlots(obs, pos, tok->events, tok->event_count);
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[0])); pos += 1;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[1])); pos += 1;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[2])); pos += 1;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[0], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[1], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[2], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->vel[0], QNN_VELOCITY_SCALE)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->vel[1], QNN_VELOCITY_SCALE)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->vel[2], QNN_VELOCITY_SCALE)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->path[0], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->path[1], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->path[2], 32767.0f)); pos += 2;
					QNN_BufWriteU16(obs, pos, QNN_QuantizeU16Saturating(tok->path_dist)); pos += 2;
					QNN_BufWriteF16(obs, pos, tok->eta); pos += 2;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Unit(tok->facing)); pos += 1;
					QNN_BufWriteU8 (obs, pos, (uint8_t)(tok->team > 0.5f ? 1 : 0)); pos += 1;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Unit(tok->score)); pos += 1;
					QNN_BufWriteF16(obs, pos, tok->recency); pos += 2;
				}
				break;

			case QNN_TOKEN_ITEM:
				{
					const qnn_item_token_t *tok = &tt->item;
					obs[pos++] = (uint8_t)tok->subject_id;
					obs[pos++] = (uint8_t)tok->modality_id;
					pos = Gate1_PackEventSlots(obs, pos, tok->events, tok->event_count);
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[0])); pos += 1;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[1])); pos += 1;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[2])); pos += 1;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[0], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[1], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[2], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->path[0], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->path[1], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->path[2], 32767.0f)); pos += 2;
					QNN_BufWriteU16(obs, pos, QNN_QuantizeU16Saturating(tok->path_dist)); pos += 2;
					QNN_BufWriteF16(obs, pos, tok->eta); pos += 2;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->amount)); pos += 1;
					QNN_BufWriteF16(obs, pos, tok->regen); pos += 2;
					QNN_BufWriteF16(obs, pos, tok->recency); pos += 2;
				}
				break;

			case QNN_TOKEN_MOVER:
				{
					const qnn_mover_token_t *tok = &tt->mover;
					obs[pos++] = (uint8_t)tok->subject_id;
					obs[pos++] = (uint8_t)tok->modality_id;
					pos = Gate1_PackEventSlots(obs, pos, tok->events, tok->event_count);
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[0])); pos += 1;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[1])); pos += 1;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[2])); pos += 1;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[0], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[1], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[2], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->path[0], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->path[1], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->path[2], 32767.0f)); pos += 2;
					QNN_BufWriteU16(obs, pos, QNN_QuantizeU16Saturating(tok->path_dist)); pos += 2;
					QNN_BufWriteF16(obs, pos, tok->eta); pos += 2;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Unit(tok->state)); pos += 1;
					QNN_BufWriteF16(obs, pos, tok->recency); pos += 2;
				}
				break;
			}
		}
	}
}

/* ══════════════════════════════════════════════════════════════════
 * The A27 pure-combat packer, ported from bb27a296 ("feat(a27):
 * pure-combat substrate refactor") — the last commit where the
 * native emit still wrote this shape, before the obs-api WS1 merge
 * (5471d10e) folded v1 and v3 onto one (v1-shaped) writer.  This is
 * the Gate-1 reference for policy v3 (the DEFAULT plan): actor and
 * projectile only, no recency.  Extends the bb27a296 shape with the
 * `paths` gate the registry refactor added (zeroes path/path_dist/eta
 * when false — compute-side pathfinder skip, byte layout unchanged).
 * ══════════════════════════════════════════════════════════════════ */

static void Gate1_CombatPackObsBuffer(uint8_t *obs, const qnn_tick_result_t *r,
	int paths)
{
	int i;
	int pos;

	memset(obs, 0, QNN_OBS_BUFFER_SIZE);

	/* ── Self block (27 B) — identical to the legacy packer. ──── */
	{
		const qnn_self_token_t *tok = &r->self;
		float eff_armor = (float)tok->raw_armor * tok->armor_type;

		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_HEALTH,
			QNN_QuantizeU8Saturating((float)tok->health));
		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_EFF_ARMOR,
			QNN_QuantizeU8Saturating(eff_armor));
		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_AMMO_SHELLS,
			QNN_QuantizeU8Saturating((float)tok->ammo_shells));
		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_AMMO_NAILS,
			QNN_QuantizeU8Saturating((float)tok->ammo_nails));
		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_AMMO_ROCKETS,
			QNN_QuantizeU8Saturating((float)tok->ammo_rockets));
		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_AMMO_CELLS,
			QNN_QuantizeU8Saturating((float)tok->ammo_cells));

		QNN_BufWriteI16(obs, QNN_OBS_OFF_SELF_VEL + 0,
			QNN_QuantizeI16Clamped(tok->vel[0], QNN_VELOCITY_SCALE));
		QNN_BufWriteI16(obs, QNN_OBS_OFF_SELF_VEL + 2,
			QNN_QuantizeI16Clamped(tok->vel[1], QNN_VELOCITY_SCALE));
		QNN_BufWriteI16(obs, QNN_OBS_OFF_SELF_VEL + 4,
			QNN_QuantizeI16Clamped(tok->vel[2], QNN_VELOCITY_SCALE));

		QNN_BufWriteF16(obs, QNN_OBS_OFF_SELF_ATTACK_FIN, tok->attack_finished);

		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_WEAPON_ID,
			(uint8_t)tok->weapon_id);
		QNN_BufWriteU8 (obs, QNN_OBS_OFF_SELF_MOVEMENT_ID,
			(uint8_t)tok->movement_id);

		QNN_BufWriteI32(obs, QNN_OBS_OFF_SELF_ITEMS, tok->items);

		QNN_BufWriteI8 (obs, QNN_OBS_OFF_SELF_VIEW_PITCH,
			QNN_QuantizeI8(tok->view_pitch));

		QNN_BufWriteF16(obs, QNN_OBS_OFF_SELF_LOOK_DELTA + 0, tok->look_delta[0]);
		QNN_BufWriteF16(obs, QNN_OBS_OFF_SELF_LOOK_DELTA + 2, tok->look_delta[1]);
		QNN_BufWriteF16(obs, QNN_OBS_OFF_SELF_LOOK_DELTA + 4, tok->look_delta[2]);
	}

	/* ── Spatial block — identical to the legacy packer. ──────── */
	for (i = 0; i < QNN_OBS_ATLAS_ELEVS; ++i)
		QNN_AtlasPackRow(
			&obs[QNN_OBS_OFF_SPATIAL + i * QNN_OBS_ATLAS_PACKED_BYTES],
			r->spatial_atlas[i]);

	/* ── Entity stream: actor/projectile only, no recency ────── */
	{
		int pack_count = r->entity_count < QNN_MAX_TOKEN_OBJECTS
			? r->entity_count : QNN_MAX_TOKEN_OBJECTS;
		pos = QNN_OBS_OFF_ENTITY_STREAM;
		obs[pos++] = (uint8_t)pack_count;

		for (i = 0; i < pack_count; ++i)
		{
			const qnn_tagged_token_t *tt = &r->entities[i];
			obs[pos++] = (uint8_t)tt->type;

			switch (tt->type)
			{
			case QNN_TOKEN_PROJECTILE:
				{
					const qnn_projectile_token_t *tok = &tt->projectile;
					obs[pos++] = (uint8_t)tok->subject_id;
					obs[pos++] = (uint8_t)tok->modality_id;
					pos = Gate1_PackEventSlots(obs, pos, tok->events, tok->event_count);
					/* Per-type scalars (12 B): rel i16×3, vel i16×3 */
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[0], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[1], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[2], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->vel[0], QNN_VELOCITY_SCALE)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->vel[1], QNN_VELOCITY_SCALE)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->vel[2], QNN_VELOCITY_SCALE)); pos += 2;
				}
				break;

			case QNN_TOKEN_ACTOR:
				{
					const qnn_actor_token_t *tok = &tt->actor;
					obs[pos++] = (uint8_t)tok->subject_id;
					obs[pos++] = (uint8_t)tok->modality_id;
					obs[pos++] = (uint8_t)tok->player_id;
					pos = Gate1_PackEventSlots(obs, pos, tok->events, tok->event_count);
					/* Per-type (28 B): half u8×3, rel i16×3, vel i16×3, path i16×3,
					 * path_dist u16, eta f16, facing u8, team u8, score u8 */
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[0])); pos += 1;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[1])); pos += 1;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Saturating(tok->half_extents[2])); pos += 1;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[0], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[1], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->rel[2], 32767.0f)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->vel[0], QNN_VELOCITY_SCALE)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->vel[1], QNN_VELOCITY_SCALE)); pos += 2;
					QNN_BufWriteI16(obs, pos, QNN_QuantizeI16Clamped(tok->vel[2], QNN_VELOCITY_SCALE)); pos += 2;
					QNN_BufWriteI16(obs, pos, paths ? QNN_QuantizeI16Clamped(tok->path[0], 32767.0f) : 0); pos += 2;
					QNN_BufWriteI16(obs, pos, paths ? QNN_QuantizeI16Clamped(tok->path[1], 32767.0f) : 0); pos += 2;
					QNN_BufWriteI16(obs, pos, paths ? QNN_QuantizeI16Clamped(tok->path[2], 32767.0f) : 0); pos += 2;
					QNN_BufWriteU16(obs, pos, paths ? QNN_QuantizeU16Saturating(tok->path_dist) : 0); pos += 2;
					QNN_BufWriteF16(obs, pos, paths ? tok->eta : 0.0f); pos += 2;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Unit(tok->facing)); pos += 1;
					QNN_BufWriteU8 (obs, pos, (uint8_t)(tok->team > 0.5f ? 1 : 0)); pos += 1;
					QNN_BufWriteU8 (obs, pos, QNN_QuantizeU8Unit(tok->score)); pos += 1;
				}
				break;
			}
		}
	}
}

/* ══════════════════════════════════════════════════════════════════
 * Synthetic tick results
 * ══════════════════════════════════════════════════════════════════ */

static void FillEvents(qnn_token_event_t events[QNN_MAX_ENTITY_EVENTS],
	int *event_count)
{
	int i;

	/* Over-range counts exercise the pack-time cap. */
	*event_count = RngRange(0, QNN_MAX_ENTITY_EVENTS + 2);
	for (i = 0; i < QNN_MAX_ENTITY_EVENTS; ++i)
	{
		events[i].action_id = RngRange(0, 255);
		events[i].source_id = RngRange(0, 255);
	}
}

static void FillVec3(float v[3], float lo, float hi)
{
	v[0] = RngFloat(lo, hi);
	v[1] = RngFloat(lo, hi);
	v[2] = RngFloat(lo, hi);
}

static void BuildSyntheticResult(qnn_tick_result_t *r, int entity_count)
{
	int i, j;

	memset(r, 0, sizeof(*r));

	r->self.health = RngRange(-10, 400);
	r->self.raw_armor = RngRange(0, 300);
	r->self.armor_type = (float)RngRange(0, 8) / 10.0f;
	r->self.ammo_shells = RngRange(0, 300);
	r->self.ammo_nails = RngRange(0, 300);
	r->self.ammo_rockets = RngRange(0, 300);
	r->self.ammo_cells = RngRange(0, 300);
	FillVec3(r->self.vel, -3000.0f, 3000.0f);
	r->self.attack_finished = RngFloat(0.0f, 5.0f);
	r->self.weapon_id = RngRange(0, 8);
	r->self.movement_id = RngRange(0, 4);
	r->self.items = (int32_t)RngU32();
	r->self.view_pitch = RngFloat(-1.2f, 1.2f);
	FillVec3(r->self.look_delta, -2.0f, 2.0f);

	for (i = 0; i < QNN_OBS_ATLAS_ELEVS; ++i)
		for (j = 0; j < QNN_OBS_ATLAS_YAWS_MAX; ++j)
			r->spatial_atlas[i][j] = (uint8_t)RngRange(0, 15);

	r->entity_count = entity_count;
	for (i = 0; i < QNN_MAX_TOKEN_OBJECTS; ++i)
	{
		qnn_tagged_token_t *tt = &r->entities[i];

		tt->type = RngRange(QNN_TOKEN_PROJECTILE, QNN_TOKEN_MOVER);
		switch (tt->type)
		{
		case QNN_TOKEN_PROJECTILE:
			tt->projectile.subject_id = RngRange(0, 255);
			tt->projectile.modality_id = RngRange(0, 5);
			FillVec3(tt->projectile.rel, -40000.0f, 40000.0f);
			FillVec3(tt->projectile.vel, -3000.0f, 3000.0f);
			tt->projectile.recency = RngFloat(0.0f, 2.5f);
			FillEvents(tt->projectile.events, &tt->projectile.event_count);
			break;
		case QNN_TOKEN_ACTOR:
			tt->actor.subject_id = RngRange(0, 255);
			tt->actor.modality_id = RngRange(0, 5);
			tt->actor.player_id = RngRange(0, 32);
			FillVec3(tt->actor.half_extents, 0.0f, 300.0f);
			FillVec3(tt->actor.rel, -40000.0f, 40000.0f);
			FillVec3(tt->actor.vel, -3000.0f, 3000.0f);
			FillVec3(tt->actor.path, -40000.0f, 40000.0f);
			tt->actor.path_dist = RngFloat(0.0f, 70000.0f);
			tt->actor.eta = RngFloat(0.0f, 30.0f);
			tt->actor.facing = RngFloat(-0.2f, 1.2f);
			tt->actor.team = RngFloat(0.0f, 1.0f);
			tt->actor.score = RngFloat(-0.2f, 1.2f);
			tt->actor.recency = RngFloat(0.0f, 2.5f);
			FillEvents(tt->actor.events, &tt->actor.event_count);
			break;
		case QNN_TOKEN_ITEM:
			tt->item.subject_id = RngRange(0, 255);
			tt->item.modality_id = RngRange(0, 5);
			FillVec3(tt->item.half_extents, 0.0f, 300.0f);
			FillVec3(tt->item.rel, -40000.0f, 40000.0f);
			FillVec3(tt->item.path, -40000.0f, 40000.0f);
			tt->item.path_dist = RngFloat(0.0f, 70000.0f);
			tt->item.eta = RngFloat(0.0f, 30.0f);
			tt->item.amount = RngFloat(0.0f, 300.0f);
			tt->item.regen = RngFloat(0.0f, 60.0f);
			tt->item.recency = RngFloat(0.0f, 2.5f);
			FillEvents(tt->item.events, &tt->item.event_count);
			break;
		case QNN_TOKEN_MOVER:
			tt->mover.subject_id = RngRange(0, 255);
			tt->mover.modality_id = RngRange(0, 5);
			FillVec3(tt->mover.half_extents, 0.0f, 300.0f);
			FillVec3(tt->mover.rel, -40000.0f, 40000.0f);
			FillVec3(tt->mover.path, -40000.0f, 40000.0f);
			tt->mover.path_dist = RngFloat(0.0f, 70000.0f);
			tt->mover.eta = RngFloat(0.0f, 30.0f);
			tt->mover.state = RngFloat(0.0f, 1.0f);
			tt->mover.recency = RngFloat(0.0f, 2.5f);
			FillEvents(tt->mover.events, &tt->mover.event_count);
			break;
		}
	}
}

/* v3/COMBAT-mode candidates: QNN_QualifyCombatEntity (qnn_oracle.c)
 * only ever qualifies ACTOR/PROJECTILE, so a real combat stream never
 * carries ITEM/MOVER — restrict the synthetic fixture the same way. */
static void BuildCombatSyntheticResult(qnn_tick_result_t *r, int entity_count)
{
	int i;

	memset(r, 0, sizeof(*r));

	r->self.health = RngRange(-10, 400);
	r->self.raw_armor = RngRange(0, 300);
	r->self.armor_type = (float)RngRange(0, 8) / 10.0f;
	r->self.ammo_shells = RngRange(0, 300);
	r->self.ammo_nails = RngRange(0, 300);
	r->self.ammo_rockets = RngRange(0, 300);
	r->self.ammo_cells = RngRange(0, 300);
	FillVec3(r->self.vel, -3000.0f, 3000.0f);
	r->self.attack_finished = RngFloat(0.0f, 5.0f);
	r->self.weapon_id = RngRange(0, 8);
	r->self.movement_id = RngRange(0, 4);
	r->self.items = (int32_t)RngU32();
	r->self.view_pitch = RngFloat(-1.2f, 1.2f);
	FillVec3(r->self.look_delta, -2.0f, 2.0f);

	for (i = 0; i < QNN_OBS_ATLAS_ELEVS; ++i)
	{
		int j;
		for (j = 0; j < QNN_OBS_ATLAS_YAWS_MAX; ++j)
			r->spatial_atlas[i][j] = (uint8_t)RngRange(0, 15);
	}

	r->entity_count = entity_count;
	for (i = 0; i < QNN_MAX_TOKEN_OBJECTS; ++i)
	{
		qnn_tagged_token_t *tt = &r->entities[i];

		tt->type = RngRange(QNN_TOKEN_PROJECTILE, QNN_TOKEN_ACTOR);
		switch (tt->type)
		{
		case QNN_TOKEN_PROJECTILE:
			tt->projectile.subject_id = RngRange(0, 255);
			tt->projectile.modality_id = RngRange(0, 5);
			FillVec3(tt->projectile.rel, -40000.0f, 40000.0f);
			FillVec3(tt->projectile.vel, -3000.0f, 3000.0f);
			tt->projectile.recency = RngFloat(0.0f, 2.5f);
			FillEvents(tt->projectile.events, &tt->projectile.event_count);
			break;
		case QNN_TOKEN_ACTOR:
			tt->actor.subject_id = RngRange(0, 255);
			tt->actor.modality_id = RngRange(0, 5);
			tt->actor.player_id = RngRange(0, 32);
			FillVec3(tt->actor.half_extents, 0.0f, 300.0f);
			FillVec3(tt->actor.rel, -40000.0f, 40000.0f);
			FillVec3(tt->actor.vel, -3000.0f, 3000.0f);
			FillVec3(tt->actor.path, -40000.0f, 40000.0f);
			tt->actor.path_dist = RngFloat(0.0f, 70000.0f);
			tt->actor.eta = RngFloat(0.0f, 30.0f);
			tt->actor.facing = RngFloat(-0.2f, 1.2f);
			tt->actor.team = RngFloat(0.0f, 1.0f);
			tt->actor.score = RngFloat(-0.2f, 1.2f);
			tt->actor.recency = RngFloat(0.0f, 2.5f);
			FillEvents(tt->actor.events, &tt->actor.event_count);
			break;
		}
	}
}

/* ══════════════════════════════════════════════════════════════════
 * Tests
 * ══════════════════════════════════════════════════════════════════ */

static void CompileDefault(qnn_obs_plan_t *plan)
{
	qnn_obs_decl_t decl;
	char error[256] = "";

	QNN_ObsDeclDefault(&decl);
	if (!QNN_ObsPlanCompile(&decl, plan, error, sizeof(error)))
	{
		fprintf(stderr, "FATAL: default plan failed to compile: %s\n", error);
		exit(1);
	}
}

static void TestDefaultPlanLayout(void)
{
	qnn_obs_plan_t plan;
	const qnn_obs_plan_step_t *atlas = NULL, *entities = NULL;
	int i;

	CompileDefault(&plan);

	CHECK(plan.step_count == 15, "default plan has %d steps, want 15",
		plan.step_count);
	CHECK(plan.payload_bytes == QNN_OBS_MAX_PAYLOAD_BYTES,
		"default payload %d, want %d",
		plan.payload_bytes, QNN_OBS_MAX_PAYLOAD_BYTES);
	CHECK(plan.frame_bytes == QNN_OBS_BUFFER_SIZE,
		"default frame %d, want %d", plan.frame_bytes, QNN_OBS_BUFFER_SIZE);

	for (i = 0; i < plan.step_count; ++i)
	{
		if (strcmp(plan.steps[i].entry->name, "atlas") == 0)
			atlas = &plan.steps[i];
		if (strcmp(plan.steps[i].entry->name, "entities") == 0)
			entities = &plan.steps[i];
	}
	CHECK(atlas != NULL && atlas->offset == QNN_OBS_OFF_SPATIAL
		&& atlas->bytes == QNN_OBS_SPATIAL_BLOCK_BYTES,
		"default atlas step off/bytes mismatch");
	CHECK(entities != NULL && entities->offset == QNN_OBS_OFF_ENTITY_STREAM,
		"default entity step offset mismatch");

	/* Spot-check the self-block offsets against the pinned table. */
	CHECK(plan.steps[0].offset == QNN_OBS_OFF_SELF_HEALTH
		&& strcmp(plan.steps[0].entry->name, "health") == 0,
		"health not at offset 0");
	CHECK(plan.steps[6].offset == QNN_OBS_OFF_SELF_VEL
		&& strcmp(plan.steps[6].entry->name, "vel") == 0,
		"vel not at offset %d", QNN_OBS_OFF_SELF_VEL);
	CHECK(plan.steps[12].offset == QNN_OBS_OFF_SELF_LOOK_DELTA
		&& strcmp(plan.steps[12].entry->name, "look_delta") == 0,
		"look_delta not at offset %d", QNN_OBS_OFF_SELF_LOOK_DELTA);
}

/* Gate 1a: a policy-v1 plan (the wire-shim FULL stream) must still
 * pack byte-identical to the pre-refactor legacy packer.  Same self/
 * atlas shape as the default plan — only the entities policy differs
 * — so a plain override on top of QNN_ObsDeclDefault exercises it. */
static void TestGate1FullBitParity(void)
{
	qnn_obs_decl_t decl;
	qnn_obs_plan_t plan;
	char error[256] = "";
	static uint8_t legacy[QNN_OBS_BUFFER_SIZE];
	static uint8_t planned[QNN_OBS_BUFFER_SIZE];
	qnn_tick_result_t r;
	int frame;
	int mismatches = 0;

	QNN_ObsDeclDefault(&decl);
	strncpy(decl.entities.policy, "v1", QNN_OBS_MAX_POLICY_NAME - 1);
	CHECK(QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"v1 plan failed to compile: %s", error);

	for (frame = 0; frame < 2000; ++frame)
	{
		/* Sweep entity counts through empty / partial / full / over-cap
		 * (over-cap exercises the pack-time clamp to 16). */
		int entity_count = frame % 21;

		BuildSyntheticResult(&r, entity_count);
		Gate1_LegacyPackObsBuffer(legacy, &r);
		QNN_ObsPlanPack(&plan, planned, QNN_OBS_BUFFER_SIZE, &r);
		if (memcmp(legacy, planned, QNN_OBS_BUFFER_SIZE) != 0)
		{
			int b;
			mismatches++;
			if (mismatches == 1)
			{
				for (b = 0; b < QNN_OBS_BUFFER_SIZE; ++b)
					if (legacy[b] != planned[b])
						fprintf(stderr,
							"  first diff frame %d byte %d: %02x != %02x\n",
							frame, b, legacy[b], planned[b]);
			}
		}
	}
	CHECK(mismatches == 0,
		"gate 1a (v1/FULL) FAILED: %d/2000 frames differ from the legacy packer",
		mismatches);
	if (mismatches == 0)
		printf("gate 1a: v1 (FULL) plan bit-identical to legacy packer "
			"over 2000 synthetic frames\n");
}

/* Gate 1b: the DEFAULT plan (policy v3, the A27 pure-combat stream)
 * must pack byte-identical to the ported bb27a296 combat writer.
 * Fixture is restricted to actor/projectile candidates — the only
 * types QNN_QualifyCombatEntity ever qualifies for COMBAT mode. */
static void TestGate1CombatBitParity(void)
{
	qnn_obs_plan_t plan;
	static uint8_t reference[QNN_OBS_BUFFER_SIZE];
	static uint8_t planned[QNN_OBS_BUFFER_SIZE];
	qnn_tick_result_t r;
	int frame;
	int mismatches = 0;

	CompileDefault(&plan);   /* the default plan IS policy v3 */

	for (frame = 0; frame < 2000; ++frame)
	{
		int entity_count = frame % 21;

		BuildCombatSyntheticResult(&r, entity_count);
		Gate1_CombatPackObsBuffer(reference, &r, 1 /* paths on, matches default decl */);
		QNN_ObsPlanPack(&plan, planned, QNN_OBS_BUFFER_SIZE, &r);
		if (memcmp(reference, planned, QNN_OBS_BUFFER_SIZE) != 0)
		{
			int b;
			mismatches++;
			if (mismatches == 1)
			{
				for (b = 0; b < QNN_OBS_BUFFER_SIZE; ++b)
					if (reference[b] != planned[b])
						fprintf(stderr,
							"  first diff frame %d byte %d: %02x != %02x\n",
							frame, b, reference[b], planned[b]);
			}
		}
	}
	CHECK(mismatches == 0,
		"gate 1b (v3/COMBAT) FAILED: %d/2000 frames differ from the "
		"ported bb27a296 combat packer", mismatches);
	if (mismatches == 0)
		printf("gate 1b: default (v3/COMBAT) plan bit-identical to the "
			"ported bb27a296 combat packer over 2000 synthetic frames\n");
}

static void TestCompileErrors(void)
{
	qnn_obs_decl_t decl;
	qnn_obs_plan_t plan;
	char error[256];

	/* Unknown state field → hard error naming it. */
	QNN_ObsDeclDefault(&decl);
	strncpy(decl.state[0], "helth", QNN_OBS_MAX_FIELD_NAME - 1);
	error[0] = '\0';
	CHECK(!QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"unknown field accepted");
	CHECK(strstr(error, "helth") != NULL,
		"error does not name the field: \"%s\"", error);

	/* Non-state name in the state list. */
	QNN_ObsDeclDefault(&decl);
	strncpy(decl.state[1], "atlas", QNN_OBS_MAX_FIELD_NAME - 1);
	CHECK(!QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"sensor accepted in the state list");

	/* Duplicate state field. */
	QNN_ObsDeclDefault(&decl);
	strncpy(decl.state[2], "health", QNN_OBS_MAX_FIELD_NAME - 1);
	CHECK(!QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"duplicate state field accepted");

	/* Unsupported atlas parameterizations. */
	QNN_ObsDeclDefault(&decl);
	decl.atlas.yaw = 36;
	CHECK(!QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"atlas yaw 36 accepted");
	QNN_ObsDeclDefault(&decl);
	decl.atlas.yaw = 72;   /* packed stays true → unsupported combo */
	CHECK(!QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"atlas (72, packed) accepted");
	QNN_ObsDeclDefault(&decl);
	decl.atlas.packed = false;   /* (24, unpacked) → unsupported combo */
	CHECK(!QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"atlas (24, unpacked) accepted");
	QNN_ObsDeclDefault(&decl);
	decl.atlas.bands = 12;
	CHECK(!QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"atlas bands 12 accepted");

	/* Unknown percept policy / bad token budget. */
	QNN_ObsDeclDefault(&decl);
	strncpy(decl.entities.policy, "v2", QNN_OBS_MAX_POLICY_NAME - 1);
	error[0] = '\0';
	CHECK(!QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"unknown policy accepted");
	CHECK(strstr(error, "v2") != NULL,
		"error does not name the policy: \"%s\"", error);
	QNN_ObsDeclDefault(&decl);
	decl.entities.max_tokens = 0;
	CHECK(!QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"max_tokens 0 accepted");
	QNN_ObsDeclDefault(&decl);
	decl.entities.max_tokens = QNN_MAX_TOKEN_OBJECTS + 1;
	CHECK(!QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"max_tokens %d accepted", QNN_MAX_TOKEN_OBJECTS + 1);

	/* Wrong schema version / empty declaration. */
	QNN_ObsDeclDefault(&decl);
	decl.obs_api = 2;
	CHECK(!QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"obs_api 2 accepted");
	memset(&decl, 0, sizeof(decl));
	decl.obs_api = QNN_OBS_API_VERSION;
	CHECK(!QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"empty declaration accepted");
}

static void TestLegacyAtlas72(void)
{
	qnn_obs_decl_t decl;
	qnn_obs_plan_t plan;
	char error[256] = "";
	qnn_tick_result_t r;
	static uint8_t obs[4096];
	const qnn_obs_plan_step_t *atlas = NULL, *entities = NULL;
	int i, j, ok;

	/* The a26 rc1 line: same state list, atlas (72, 11, unpacked). */
	QNN_ObsDeclDefault(&decl);
	decl.atlas.yaw = QNN_OBS_ATLAS_YAWS_LEGACY;
	decl.atlas.packed = false;
	CHECK(QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"legacy 72-unpacked plan failed to compile: %s", error);

	for (i = 0; i < plan.step_count; ++i)
	{
		if (strcmp(plan.steps[i].entry->name, "atlas") == 0)
			atlas = &plan.steps[i];
		if (strcmp(plan.steps[i].entry->name, "entities") == 0)
			entities = &plan.steps[i];
	}
	/* The f84c36cd^ layout: 792 B flat atlas at 27, entity stream at 819. */
	CHECK(atlas != NULL && atlas->offset == 27 && atlas->bytes == 792,
		"legacy atlas at %d/%d B, want 27/792",
		atlas ? atlas->offset : -1, atlas ? atlas->bytes : -1);
	CHECK(entities != NULL && entities->offset == 819,
		"legacy entity stream at %d, want 819",
		entities ? entities->offset : -1);
	CHECK(plan.frame_bytes == 819 + 689 + QNN_OBS_POSE_TAIL_BYTES,
		"legacy frame %d, want %d",
		plan.frame_bytes, 819 + 689 + QNN_OBS_POSE_TAIL_BYTES);

	/* Flat emit = the tick scratch verbatim, one code per byte,
	 * elevation-major. This decl keeps the default policy (v3/COMBAT),
	 * so the entity fixture must stay actor/projectile-only (the v3
	 * row writer fails loud on ITEM/MOVER — real COMBAT-mode
	 * qualification never produces them). Atlas bytes are all this
	 * test inspects. */
	BuildCombatSyntheticResult(&r, 4);
	QNN_ObsPlanPack(&plan, obs, (int)sizeof(obs), &r);
	ok = 1;
	for (i = 0; i < QNN_OBS_ATLAS_ELEVS; ++i)
		for (j = 0; j < QNN_OBS_ATLAS_YAWS_LEGACY; ++j)
			if (obs[27 + i * QNN_OBS_ATLAS_YAWS_LEGACY + j]
				!= r.spatial_atlas[i][j])
				ok = 0;
	CHECK(ok, "legacy flat atlas bytes differ from the tick scratch");
}

/* ── Demand-driven compute (the no-entities → no-pathfinder proof) ──
 * The oracle is the only consumer of the pathfinder on the emit path;
 * counting provider calls proves a plan without entities performs
 * zero oracle work, hence zero pathfinder queries. */

static int stub_self_calls;
static int stub_entity_calls;
static int stub_atlas_calls;

static void StubSelf(const qnn_snapshot_t *snapshot, qnn_self_token_t *out)
{
	(void)snapshot; (void)out;
	stub_self_calls++;
}

static int StubEntities(const qnn_snapshot_t *snapshot,
	const qnn_obs_entity_params_t *params, qnn_tagged_token_t *out,
	int max_tokens)
{
	(void)snapshot; (void)params; (void)out; (void)max_tokens;
	stub_entity_calls++;
	return 0;
}

static void StubAtlas(const qnn_snapshot_t *snapshot,
	const qnn_obs_atlas_params_t *params,
	uint8_t atlas[QNN_OBS_ATLAS_ELEVS][QNN_OBS_ATLAS_YAWS_MAX])
{
	(void)snapshot; (void)params; (void)atlas;
	stub_atlas_calls++;
}

static void TestDemandDrivenCompute(void)
{
	static const qnn_obs_compute_fns_t stubs = {
		StubSelf, StubEntities, StubAtlas,
	};
	qnn_obs_decl_t decl;
	qnn_obs_plan_t plan;
	char error[256] = "";
	qnn_tick_result_t r;

	/* No entities requested → the oracle (and with it the pathfinder)
	 * must never be touched. */
	QNN_ObsDeclDefault(&decl);
	decl.entities_requested = false;
	CHECK(QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"entities-less plan failed to compile: %s", error);
	stub_self_calls = stub_entity_calls = stub_atlas_calls = 0;
	QNN_ObsPlanCompute(&plan, &stubs, NULL, &r);
	CHECK(stub_entity_calls == 0,
		"entities-less plan reached the oracle (%d calls)",
		stub_entity_calls);
	CHECK(stub_self_calls == 1 && stub_atlas_calls == 1,
		"entities-less plan providers: self %d atlas %d, want 1/1",
		stub_self_calls, stub_atlas_calls);

	/* State-only plan → neither oracle nor atlas rays. */
	QNN_ObsDeclDefault(&decl);
	decl.entities_requested = false;
	decl.atlas_requested = false;
	CHECK(QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"state-only plan failed to compile: %s", error);
	stub_self_calls = stub_entity_calls = stub_atlas_calls = 0;
	QNN_ObsPlanCompute(&plan, &stubs, NULL, &r);
	CHECK(stub_entity_calls == 0 && stub_atlas_calls == 0,
		"state-only plan ran sensors/percepts (entities %d atlas %d)",
		stub_entity_calls, stub_atlas_calls);
	/* Undemanded atlas scratch must read all-miss. */
	CHECK(r.spatial_atlas[0][0] == QNN_OBS_ATLAS_MISS_CODE,
		"undemanded atlas scratch not prefilled with the miss code");

	/* Full plan → each provider exactly once. */
	QNN_ObsDeclDefault(&decl);
	CHECK(QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"default plan failed to compile: %s", error);
	stub_self_calls = stub_entity_calls = stub_atlas_calls = 0;
	QNN_ObsPlanCompute(&plan, &stubs, NULL, &r);
	CHECK(stub_self_calls == 1 && stub_entity_calls == 1
		&& stub_atlas_calls == 1,
		"full plan providers: self %d entities %d atlas %d, want 1/1/1",
		stub_self_calls, stub_entity_calls, stub_atlas_calls);
}

static void TestPathsOffZeroesPathFields(void)
{
	qnn_obs_decl_t decl;
	qnn_obs_plan_t plan;
	char error[256] = "";
	qnn_tick_result_t r;
	static uint8_t obs[QNN_OBS_BUFFER_SIZE];
	int row, k, zero_ok, rel_nonzero;

	QNN_ObsDeclDefault(&decl);
	decl.entities.paths = false;
	CHECK(QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"paths-off plan failed to compile: %s", error);

	/* One actor token, no events → fixed row layout:
	 * n(1) tag(1) ids(3) evcnt(1) half(3) rel(6) vel(6) path(6)
	 * path_dist(2) eta(2) ... */
	BuildSyntheticResult(&r, 1);
	memset(&r.entities[0], 0, sizeof(r.entities[0]));
	r.entities[0].type = QNN_TOKEN_ACTOR;
	r.entities[0].actor.event_count = 0;
	r.entities[0].actor.rel[0] = 512.0f;
	r.entities[0].actor.path[0] = 512.0f;
	r.entities[0].actor.path_dist = 512.0f;
	r.entities[0].actor.eta = 3.0f;
	QNN_ObsPlanPack(&plan, obs, QNN_OBS_BUFFER_SIZE, &r);

	row = QNN_OBS_OFF_ENTITY_STREAM + 1;   /* past n_tokens */
	rel_nonzero = obs[row + 1 + 3 + 1 + 3] != 0
		|| obs[row + 1 + 3 + 1 + 3 + 1] != 0;
	zero_ok = 1;
	for (k = 0; k < 6 + 2 + 2; ++k)   /* path + path_dist + eta */
		if (obs[row + 1 + 3 + 1 + 3 + 6 + 6 + k] != 0)
			zero_ok = 0;
	CHECK(rel_nonzero, "paths=false zeroed non-path fields");
	CHECK(zero_ok, "paths=false left path-derived bytes non-zero");
}

/* ══════════════════════════════════════════════════════════════════
 * WS2: declaration JSON parsing, the OP_ATTACH_DECL layout reply, and
 * the wire-identity shim table (qnn_obs_shim.{h,c}).
 * ══════════════════════════════════════════════════════════════════ */

/* The default declaration exactly as the python mirror serializes it
 * (qnn/obs_api.py DEFAULT_DECLARATION.to_json()). */
static const char *QNN_TEST_DEFAULT_DECL_JSON =
	"{\"obs_api\":1,\"state\":[\"health\",\"effective_armor\","
	"\"ammo_shells\",\"ammo_nails\",\"ammo_rockets\",\"ammo_cells\","
	"\"vel\",\"attack_finished\",\"self_weapon_id\",\"self_movement_id\","
	"\"self_items\",\"view_pitch\",\"look_delta\"],"
	"\"atlas\":{\"yaw\":24,\"bands\":11,\"packed\":true},"
	"\"entities\":{\"policy\":\"v3\",\"max_tokens\":16,\"paths\":true}}";

/* The legacy 72-unpacked / full-stream / pinned-4096 generation. */
static const char *QNN_TEST_LEGACY_DECL_JSON =
	"{\"obs_api\":1,\"state\":[\"health\",\"effective_armor\","
	"\"ammo_shells\",\"ammo_nails\",\"ammo_rockets\",\"ammo_cells\","
	"\"vel\",\"attack_finished\",\"self_weapon_id\",\"self_movement_id\","
	"\"self_items\",\"view_pitch\",\"look_delta\"],"
	"\"atlas\":{\"yaw\":72,\"bands\":11,\"packed\":false},"
	"\"entities\":{\"policy\":\"v1\",\"max_tokens\":16,\"paths\":true},"
	"\"frame_bytes\":4096}";

static void TestDeclJsonRoundTrip(void)
{
	qnn_obs_decl_t decl;
	qnn_obs_decl_t reference;
	qnn_obs_plan_t plan;
	char error[256] = "";

	/* The python-serialized default declaration must parse into
	 * exactly QNN_ObsDeclDefault and compile to the 864 frame. */
	CHECK(QNN_ObsDeclParseJson(QNN_TEST_DEFAULT_DECL_JSON, -1, &decl,
			error, sizeof(error)),
		"default decl JSON failed to parse: %s", error);
	QNN_ObsDeclDefault(&reference);
	CHECK(decl.obs_api == reference.obs_api
			&& decl.state_count == reference.state_count
			&& memcmp(decl.state, reference.state, sizeof(decl.state)) == 0
			&& decl.atlas_requested && decl.entities_requested
			&& decl.atlas.yaw == 24 && decl.atlas.bands == 11
			&& decl.atlas.packed
			&& strcmp(decl.entities.policy, "v3") == 0
			&& decl.entities.max_tokens == QNN_MAX_TOKEN_OBJECTS
			&& decl.entities.paths
			&& decl.frame_bytes_override == 0,
		"parsed default decl != QNN_ObsDeclDefault");
	CHECK(QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"parsed default decl failed to compile: %s", error);
	CHECK(plan.frame_bytes == QNN_OBS_BUFFER_SIZE,
		"parsed default decl frame is %d, want %d",
		plan.frame_bytes, QNN_OBS_BUFFER_SIZE);

	/* The legacy declaration: 72-unpacked atlas, policy v1, explicit
	 * 4096 frame override.  Offsets are dense; the override only pads
	 * the frame. */
	CHECK(QNN_ObsDeclParseJson(QNN_TEST_LEGACY_DECL_JSON, -1, &decl,
			error, sizeof(error)),
		"legacy decl JSON failed to parse: %s", error);
	CHECK(decl.atlas.yaw == 72 && !decl.atlas.packed
			&& strcmp(decl.entities.policy, "v1") == 0
			&& decl.frame_bytes_override == 4096,
		"legacy decl fields parsed wrong");
	CHECK(QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"legacy decl failed to compile: %s", error);
	CHECK(plan.frame_bytes == 4096,
		"legacy decl frame is %d, want 4096", plan.frame_bytes);
	CHECK(plan.payload_bytes == 27 + 792 + 689,
		"legacy decl payload is %d, want %d",
		plan.payload_bytes, 27 + 792 + 689);
	CHECK(plan.steps[14].offset == 819,
		"legacy entities offset %d, want 819", plan.steps[14].offset);

	/* Whitespace tolerance (json.dumps with spaces). */
	CHECK(QNN_ObsDeclParseJson(
			"{ \"obs_api\": 1, \"state\": [ \"health\" ], "
			"\"atlas\": null, \"entities\": null }",
			-1, &decl, error, sizeof(error)),
		"spaced decl JSON failed to parse: %s", error);
	CHECK(decl.state_count == 1 && !decl.atlas_requested
			&& !decl.entities_requested,
		"spaced decl parsed wrong");
}

static void TestDeclJsonErrors(void)
{
	qnn_obs_decl_t decl;
	qnn_obs_plan_t plan;
	char error[256];
	static const char *bad_parse[] = {
		"",                                             /* empty */
		"not json",                                     /* garbage */
		"{\"obs_api\":1,\"state\":[]",                  /* truncated */
		"{\"obs_api\":1,\"state\":[]}x",                /* trailing bytes */
		"{\"state\":[]}",                               /* missing obs_api */
		"{\"obs_api\":1}",                              /* missing state */
		"{\"obs_api\":1,\"state\":[],\"bogus\":1}",     /* unknown key */
		"{\"obs_api\":1,\"obs_api\":1,\"state\":[]}",   /* duplicate key */
		"{\"obs_api\":1.5,\"state\":[]}",               /* float version */
		"{\"obs_api\":1,\"state\":[1]}",                /* non-string field */
		/* atlas params are all-required — no silent defaults */
		"{\"obs_api\":1,\"state\":[],"
		"\"atlas\":{\"yaw\":24,\"bands\":11}}",
		"{\"obs_api\":1,\"state\":[],"
		"\"atlas\":{\"yaw\":24,\"bands\":11,\"packed\":true,\"x\":1}}",
		/* entities params are all-required too */
		"{\"obs_api\":1,\"state\":[],"
		"\"entities\":{\"policy\":\"v3\",\"paths\":true}}",
		/* frame_bytes must be a positive integer */
		"{\"obs_api\":1,\"state\":[],\"frame_bytes\":0}",
		"{\"obs_api\":1,\"state\":[],\"frame_bytes\":-4}",
	};
	int i;

	for (i = 0; i < (int)(sizeof(bad_parse) / sizeof(bad_parse[0])); ++i)
	{
		error[0] = 0;
		CHECK(!QNN_ObsDeclParseJson(bad_parse[i], -1, &decl,
				error, sizeof(error)),
			"malformed decl #%d parsed anyway: %s", i, bad_parse[i]);
		CHECK(error[0] != 0, "malformed decl #%d rejected without a "
			"message", i);
	}

	/* Parse-clean but compile-rejected: content errors belong to the
	 * compiler and must still fail loud end-to-end. */
	error[0] = 0;
	CHECK(QNN_ObsDeclParseJson(
			"{\"obs_api\":1,\"state\":[\"warp_factor\"]}",
			-1, &decl, error, sizeof(error)),
		"unknown-field decl failed at PARSE (want compile): %s", error);
	CHECK(!QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error))
			&& strstr(error, "warp_factor") != NULL,
		"unknown field not named by the compiler: %s", error);

	error[0] = 0;
	CHECK(QNN_ObsDeclParseJson(
			"{\"obs_api\":1,\"state\":[\"health\"],"
			"\"entities\":{\"policy\":\"v9\",\"max_tokens\":16,"
			"\"paths\":true}}",
			-1, &decl, error, sizeof(error)),
		"unknown-policy decl failed at PARSE (want compile): %s", error);
	CHECK(!QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error))
			&& strstr(error, "v9") != NULL,
		"unknown policy not named by the compiler: %s", error);

	/* frame_bytes override below the compiled minimum. */
	error[0] = 0;
	CHECK(QNN_ObsDeclParseJson(
			"{\"obs_api\":1,\"state\":[\"health\"],\"frame_bytes\":8}",
			-1, &decl, error, sizeof(error)),
		"short-override decl failed at PARSE (want compile): %s", error);
	CHECK(!QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error))
			&& strstr(error, "override") != NULL,
		"short frame_bytes override not rejected: %s", error);
}

static void TestLayoutReplyJson(void)
{
	qnn_obs_decl_t decl;
	qnn_obs_plan_t plan;
	char error[256] = "";
	static char reply[4096];

	/* Packed 24×11 parameterization (the default plan). */
	CHECK(QNN_ObsDeclParseJson(QNN_TEST_DEFAULT_DECL_JSON, -1, &decl,
			error, sizeof(error))
			&& QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"default decl setup failed: %s", error);
	CHECK(QNN_ObsLayoutReplyJson(&plan, reply, sizeof(reply),
			error, sizeof(error)),
		"default layout reply failed: %s", error);
	CHECK(strncmp(reply,
			"{\"ok\":true,\"layout\":{\"frame_bytes\":864,\"fields\":["
			"{\"name\":\"health\",\"kind\":\"state\",\"params\":{},"
			"\"offset\":0,\"bytes\":1,\"dtype\":\"uint8\",\"shape\":[]}",
			strlen("{\"ok\":true,\"layout\":{\"frame_bytes\":864,"
			"\"fields\":[{\"name\":\"health\",\"kind\":\"state\","
			"\"params\":{},\"offset\":0,\"bytes\":1,\"dtype\":\"uint8\","
			"\"shape\":[]}")) == 0,
		"default reply head wrong: %.120s", reply);
	CHECK(strstr(reply,
			"{\"name\":\"vel\",\"kind\":\"state\",\"params\":{},"
			"\"offset\":6,\"bytes\":6,\"dtype\":\"int16\","
			"\"shape\":[3]}") != NULL,
		"vel field wrong in default reply");
	CHECK(strstr(reply,
			"{\"name\":\"self_items\",\"kind\":\"state\",\"params\":{},"
			"\"offset\":16,\"bytes\":4,\"dtype\":\"int32\","
			"\"shape\":[]}") != NULL,
		"self_items field wrong in default reply");
	CHECK(strstr(reply,
			"{\"name\":\"atlas\",\"kind\":\"sensor\",\"params\":"
			"{\"yaw\":24,\"bands\":11,\"packed\":true},\"offset\":27,"
			"\"bytes\":132,\"dtype\":\"uint8\",\"shape\":[11,12]}") != NULL,
		"atlas field wrong in default reply");
	CHECK(strstr(reply,
			"{\"name\":\"entities\",\"kind\":\"percept\",\"params\":"
			"{\"policy\":\"v3\",\"max_tokens\":16,\"paths\":true},"
			"\"offset\":159,\"bytes\":689,\"dtype\":null,"
			"\"shape\":null}]}}") != NULL,
		"entities field / reply tail wrong in default reply");

	/* Unpacked 72×11 parameterization (the legacy shim frame). */
	CHECK(QNN_ObsDeclParseJson(QNN_TEST_LEGACY_DECL_JSON, -1, &decl,
			error, sizeof(error))
			&& QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error)),
		"legacy decl setup failed: %s", error);
	CHECK(QNN_ObsLayoutReplyJson(&plan, reply, sizeof(reply),
			error, sizeof(error)),
		"legacy layout reply failed: %s", error);
	CHECK(strstr(reply, "\"frame_bytes\":4096") != NULL,
		"legacy reply frame_bytes wrong");
	CHECK(strstr(reply,
			"{\"name\":\"atlas\",\"kind\":\"sensor\",\"params\":"
			"{\"yaw\":72,\"bands\":11,\"packed\":false},\"offset\":27,"
			"\"bytes\":792,\"dtype\":\"uint8\",\"shape\":[11,72]}") != NULL,
		"legacy atlas field wrong in reply");
	CHECK(strstr(reply,
			"{\"name\":\"entities\",\"kind\":\"percept\",\"params\":"
			"{\"policy\":\"v1\",\"max_tokens\":16,\"paths\":true},"
			"\"offset\":819,\"bytes\":689,\"dtype\":null,"
			"\"shape\":null}]}}") != NULL,
		"legacy entities field wrong in reply");
}

static void TestWireShimTable(void)
{
	qnn_obs_decl_t decl;
	qnn_obs_plan_t plan;
	const char *semantics = NULL;
	char error[256] = "";
	int i;
	static const char *refused[] =
		{ "wire.12", "wire.13", "wire.10", "wire.11", "wire.9", "wire.7",
		  "wire.14.1", "" };

	/* wire.12.1 — a26 rc1: 13 state fields, 72-unpacked atlas, FULL
	 * stream (v1), pinned 4096 frame. */
	CHECK(QNN_ObsShimDeclForWire("wire.12.1", &decl, &semantics,
			error, sizeof(error)),
		"wire.12.1 shim failed: %s", error);
	CHECK(strcmp(semantics, "semantics.1") == 0,
		"wire.12.1 semantics %s, want semantics.1", semantics);
	CHECK(decl.state_count == 13 && decl.atlas.yaw == 72
			&& !decl.atlas.packed
			&& strcmp(decl.entities.policy, "v1") == 0
			&& decl.frame_bytes_override == 4096,
		"wire.12.1 shim declaration wrong");
	CHECK(QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error))
			&& plan.frame_bytes == 4096,
		"wire.12.1 shim plan: %s (frame %d)", error, plan.frame_bytes);

	/* wire.12.2 — a26 rc2: today's 864 frame with the FULL stream. */
	CHECK(QNN_ObsShimDeclForWire("wire.12.2", &decl, &semantics,
			error, sizeof(error)),
		"wire.12.2 shim failed: %s", error);
	CHECK(strcmp(semantics, "semantics.1") == 0
			&& decl.state_count == 13 && decl.atlas.yaw == 24
			&& decl.atlas.packed
			&& strcmp(decl.entities.policy, "v1") == 0
			&& decl.frame_bytes_override == 0,
		"wire.12.2 shim declaration wrong");
	CHECK(QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error))
			&& plan.frame_bytes == QNN_OBS_BUFFER_SIZE,
		"wire.12.2 shim plan: %s (frame %d)", error, plan.frame_bytes);

	/* wire.13.1 — a27 rc1: NO self_weapon_id, combat stream (v3),
	 * 72-unpacked atlas, pinned 4096 frame. */
	CHECK(QNN_ObsShimDeclForWire("wire.13.1", &decl, &semantics,
			error, sizeof(error)),
		"wire.13.1 shim failed: %s", error);
	CHECK(strcmp(semantics, "semantics.2") == 0
			&& decl.state_count == 12 && decl.atlas.yaw == 72
			&& !decl.atlas.packed
			&& strcmp(decl.entities.policy, "v3") == 0
			&& decl.frame_bytes_override == 4096,
		"wire.13.1 shim declaration wrong");
	for (i = 0; i < decl.state_count; ++i)
		CHECK(strcmp(decl.state[i], "self_weapon_id") != 0,
			"wire.13.1 shim kept self_weapon_id at %d", i);
	CHECK(QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error))
			&& plan.frame_bytes == 4096
			&& plan.payload_bytes == 26 + 792 + 689,
		"wire.13.1 shim plan: %s (frame %d payload %d)",
		error, plan.frame_bytes, plan.payload_bytes);

	/* wire.13.2 — a27 frontier: 26-byte self block + packed atlas. */
	CHECK(QNN_ObsShimDeclForWire("wire.13.2", &decl, &semantics,
			error, sizeof(error)),
		"wire.13.2 shim failed: %s", error);
	CHECK(strcmp(semantics, "semantics.2") == 0
			&& decl.state_count == 12 && decl.atlas.yaw == 24
			&& decl.atlas.packed
			&& strcmp(decl.entities.policy, "v3") == 0
			&& decl.frame_bytes_override == 0,
		"wire.13.2 shim declaration wrong");
	CHECK(QNN_ObsPlanCompile(&decl, &plan, error, sizeof(error))
			&& plan.frame_bytes == 26 + 132 + 689 + QNN_OBS_POSE_TAIL_BYTES,
		"wire.13.2 shim plan: %s (frame %d)", error, plan.frame_bytes);

	/* Anything else — including the ambiguous bare ids — is refused
	 * with an error naming the id. */
	for (i = 0; i < (int)(sizeof(refused) / sizeof(refused[0])); ++i)
	{
		error[0] = 0;
		CHECK(!QNN_ObsShimDeclForWire(refused[i], &decl, NULL,
				error, sizeof(error)),
			"bare/foreign id \"%s\" resolved via the shim table",
			refused[i]);
		CHECK(refused[i][0] == 0 || strstr(error, refused[i]) != NULL,
			"shim refusal for \"%s\" does not name the id: %s",
			refused[i], error);
	}
	CHECK(!QNN_ObsShimDeclForWire(NULL, &decl, NULL,
			error, sizeof(error)),
		"NULL wire id resolved via the shim table");
}

int main(void)
{
	TestDefaultPlanLayout();
	TestGate1FullBitParity();
	TestGate1CombatBitParity();
	TestCompileErrors();
	TestLegacyAtlas72();
	TestDemandDrivenCompute();
	TestPathsOffZeroesPathFields();
	TestDeclJsonRoundTrip();
	TestDeclJsonErrors();
	TestLayoutReplyJson();
	TestWireShimTable();

	printf("%d checks, %d failed\n", checks_run, checks_failed);
	return checks_failed == 0 ? 0 : 1;
}
