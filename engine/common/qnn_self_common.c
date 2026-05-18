/*
 * qnn_self.c (common) — Game-agnostic self-token helpers.
 *
 * QNN_WeaponId, QNN_CurrentArmortype, and QNN_SelfEmitToken read from the
 * shared engine state (cl.stats[*]) or the snapshot struct, both of which
 * are populated by the per-game QNN_CaptureBaseSnapshot.  They have no
 * NQ/QW divergence and live here so they exist in one place.
 */

#include "qnn_io.h"

#include <string.h>

/* ── Engine state helpers ──────────────────────────────────────── */

int QNN_WeaponId(void)
{
	int active;

	active = cl.stats[STAT_ACTIVEWEAPON];
	if (active > 0)
	{
		if (active == IT_AXE) return 1;
		if (active == IT_SHOTGUN) return 2;
		if (active == IT_SUPER_SHOTGUN) return 3;
		if (active == IT_NAILGUN) return 4;
		if (active == IT_SUPER_NAILGUN) return 5;
		if (active == IT_GRENADE_LAUNCHER) return 6;
		if (active == IT_ROCKET_LAUNCHER) return 7;
		if (active == IT_LIGHTNING) return 8;
	}
	if (cl.stats[STAT_WEAPON] > 0)
	{
		active = cl.stats[STAT_WEAPON];
		if (active == IT_AXE) return 1;
		if (active == IT_SHOTGUN) return 2;
		if (active == IT_SUPER_SHOTGUN) return 3;
		if (active == IT_NAILGUN) return 4;
		if (active == IT_SUPER_NAILGUN) return 5;
		if (active == IT_GRENADE_LAUNCHER) return 6;
		if (active == IT_ROCKET_LAUNCHER) return 7;
		if (active == IT_LIGHTNING) return 8;
		return cl.stats[STAT_WEAPON];
	}
	return 0;
}

float QNN_CurrentArmortype(void)
{
	/* NQ: cl.items is the native field; QW: synthesized from cl.stats[STAT_ITEMS]
	 * by the compat shim, so cl.items works in both engines. */
	int items = cl.items;
	if (items & IT_ARMOR3) return 0.8f;
	if (items & IT_ARMOR2) return 0.6f;
	if (items & IT_ARMOR1) return 0.3f;
	return 0.0f;
}

/* ── Token emission (shared format) ────────────────────────────── */

void QNN_SelfEmitToken(qnn_self_token_t *out, const qnn_snapshot_t *snapshot)
{
	int items = snapshot->items_owned;
	vec3_t vel_view;
	int i;

	memset(out, 0, sizeof(*out));

	out->health = (float)snapshot->health / QNN_MAX_HEALTH;
	out->armor = (float)snapshot->armor * snapshot->armor_type / QNN_MAX_ARMOR;
	out->weapon_sg = (items & IT_SHOTGUN) ? 1.0f : 0.0f;
	out->weapon_ssg = (items & IT_SUPER_SHOTGUN) ? 1.0f : 0.0f;
	out->weapon_ng = (items & IT_NAILGUN) ? 1.0f : 0.0f;
	out->weapon_sng = (items & IT_SUPER_NAILGUN) ? 1.0f : 0.0f;
	out->weapon_gl = (items & IT_GRENADE_LAUNCHER) ? 1.0f : 0.0f;
	out->weapon_rl = (items & IT_ROCKET_LAUNCHER) ? 1.0f : 0.0f;
	out->weapon_lg = (items & IT_LIGHTNING) ? 1.0f : 0.0f;
	out->ammo_shells = QNN_Clamp((float)snapshot->ammo_shells / QNN_MAX_SHELLS, 0.0f, 1.0f);
	out->ammo_nails = QNN_Clamp((float)snapshot->ammo_nails / QNN_MAX_NAILS, 0.0f, 1.0f);
	out->ammo_rockets = QNN_Clamp((float)snapshot->ammo_rockets / QNN_MAX_ROCKETS, 0.0f, 1.0f);
	out->ammo_cells = QNN_Clamp((float)snapshot->ammo_cells / QNN_MAX_CELLS, 0.0f, 1.0f);

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
}
