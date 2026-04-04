/*
 * qnn_self.c — Self state capture and token emission.
 *
 * Reads client engine state (cl.stats, cl.items, cl.velocity, etc.)
 * and emits a typed qnn_self_token_t.  Also owns QNN_CaptureBaseSnapshot
 * which populates the snapshot struct with self state for all modules.
 */

#include "qnn_io.h"

#include <string.h>

/* ── Engine state helpers ──────────────────────────────────────── */

int QNN_WeaponId(void)
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

int QNN_CurrentFrags(void)
{
	if (cl.viewentity > 0 && cl.scores != NULL && cl.viewentity - 1 < cl.maxclients)
		return cl.scores[cl.viewentity - 1].frags;
	return cl.stats[STAT_FRAGS];
}

static float QNN_CurrentArmortype(void)
{
	int items = cl.items;
	if (items & IT_ARMOR3) return 0.8f;
	if (items & IT_ARMOR2) return 0.6f;
	if (items & IT_ARMOR1) return 0.3f;
	return 0.0f;
}

/* ── Snapshot capture ──────────────────────────────────────────── */

void QNN_CaptureBaseSnapshot(qnn_snapshot_t *snapshot)
{
	entity_t *player_entity;

	memset(snapshot, 0, sizeof(*snapshot));
	player_entity = (cl.viewentity > 0 && cl.viewentity < MAX_EDICTS) ? &cl_entities[cl.viewentity] : NULL;
	if (player_entity != NULL) {
		VectorCopy(player_entity->origin, snapshot->player_origin);
	} else {
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
	snapshot->items_owned = cl.items;
	snapshot->weapon_id = QNN_WeaponId();
	snapshot->waterlevel = cl.inwater ? 2 : 0;
	snapshot->grounded = cl.onground ? true : false;
	snapshot->current_region_id = 0; /* legacy region system — unused */
}

/* ── Token emission ────────────────────────────────────────────── */

void QNN_SelfEmitToken(qnn_self_token_t *out, const qnn_snapshot_t *snapshot, int player_cluster_id)
{
	int items = snapshot->items_owned;
	vec3_t vel_view;
	int i;

	memset(out, 0, sizeof(*out));

	out->health = (float)snapshot->health / QNN_SELF_HEALTH_CAP;
	out->armor = (float)snapshot->armor * snapshot->armor_type / QNN_SELF_HEALTH_CAP;
	out->weapon_sg = (items & IT_SUPER_SHOTGUN) ? 1.0f : (items & IT_SHOTGUN) ? 0.5f : 0.0f;
	out->weapon_ng = (items & IT_SUPER_NAILGUN) ? 1.0f : (items & IT_NAILGUN) ? 0.5f : 0.0f;
	out->weapon_gl = (items & IT_GRENADE_LAUNCHER) ? 1.0f : 0.0f;
	out->weapon_rl = (items & IT_ROCKET_LAUNCHER) ? 1.0f : 0.0f;
	out->weapon_lg = (items & IT_LIGHTNING) ? 1.0f : 0.0f;
	out->ammo_shells = QNN_Clamp((float)snapshot->ammo_shells / QNN_SELF_SHELLS_CAP, 0.0f, 1.0f);
	out->ammo_nails = QNN_Clamp((float)snapshot->ammo_nails / QNN_SELF_NAILS_CAP, 0.0f, 1.0f);
	out->ammo_rockets = QNN_Clamp((float)snapshot->ammo_rockets / QNN_SELF_ROCKETS_CAP, 0.0f, 1.0f);
	out->ammo_cells = QNN_Clamp((float)snapshot->ammo_cells / QNN_SELF_CELLS_CAP, 0.0f, 1.0f);

	QNN_RelativeFrame(snapshot->player_view_angles, snapshot->player_velocity, vel_view);
	out->vel[0] = QNN_Normalize(vel_view[0], QNN_VELOCITY_SCALE);
	out->vel[1] = QNN_Normalize(vel_view[1], QNN_VELOCITY_SCALE);
	out->vel[2] = QNN_Normalize(vel_view[2], QNN_VELOCITY_SCALE);

	out->weapon_id = qnn_weapon_subject_from_id(snapshot->weapon_id);

	if (items & IT_ARMOR3)
		out->armor_type_id = QNN_SUBJECT_ARMOR_RED;
	else if (items & IT_ARMOR2)
		out->armor_type_id = QNN_SUBJECT_ARMOR_YELLOW;
	else if (items & IT_ARMOR1)
		out->armor_type_id = QNN_SUBJECT_ARMOR_GREEN;

	switch (snapshot->waterlevel)
	{
	case 1: out->movement_id = 2; break;
	case 2: out->movement_id = 3; break;
	case 3: out->movement_id = 4; break;
	default: out->movement_id = snapshot->grounded ? 0 : 1; break;
	}

	out->cluster_id = player_cluster_id;

	i = 0;
	if (items & IT_QUAD)
		out->powerup_ids[i++] = QNN_SUBJECT_QUAD;
	if (items & IT_INVULNERABILITY)
		out->powerup_ids[i++] = QNN_SUBJECT_PENT;
	if (items & IT_INVISIBILITY)
		out->powerup_ids[i++] = QNN_SUBJECT_RING;
	if (items & IT_SUIT)
		out->powerup_ids[i++] = QNN_SUBJECT_SUIT;
	if (snapshot->health > 100)
		out->powerup_ids[i++] = QNN_SUBJECT_MEGAHEALTH;
	out->powerup_count = i;
}
