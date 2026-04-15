/*
 * qnn_self.c (qw) — Self state capture for the QW demo worker.
 *
 * QW equivalent of qnn_self.c.  Key differences from NQ:
 *   - Player origin: cl.simorg (predicted) or playerstate[].origin
 *   - Player velocity: cl.simvel or playerstate[].velocity
 *   - View angles: cl.viewangles (same as NQ)
 *   - Stats: cl.stats[] (same STAT_* constants)
 *   - Items: cl.stats[STAT_ITEMS] (not cl.items — QW has no cl.items)
 *   - Ground state: playerstate[].onground
 *   - No cl.viewentity — QW uses cl.playernum
 *   - Frags: cl.players[cl.playernum].frags
 */

#include "qnn_io.h"

#include <string.h>

/* QW pmove globals — updated by CL_PredictMove each frame.  We read
 * waterlevel here because QW's player_state_t doesn't carry it. */
extern int waterlevel;

/* Latest-known per-player state, maintained by CL_ParsePlayerinfo_MVD
 * in the patched QW client sources.  Used as the MVD source of truth
 * for playerstate instead of cl.frames[], whose circular-buffer slots
 * alias every UPDATE_BACKUP packets and get polluted by delta gaps. */
extern player_state_t qnn_mvd_latest_playerstate[MAX_CLIENTS];

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
	if (cl.playernum >= 0 && cl.playernum < MAX_CLIENTS)
		return cl.players[cl.playernum].frags;
	return 0;
}

static float QNN_CurrentArmortype(void)
{
	int items = cl.stats[STAT_ITEMS];
	if (items & IT_ARMOR3) return 0.8f;
	if (items & IT_ARMOR2) return 0.6f;
	if (items & IT_ARMOR1) return 0.3f;
	return 0.0f;
}

/* ── Get best player state for current frame ──────────────────── */

static player_state_t *QNN_GetPlayerState(void)
{
	if (cl.playernum < 0 || cl.playernum >= MAX_CLIENTS)
		return NULL;

	/* MVD: read from the dedicated latest-state array maintained by
	 * CL_ParsePlayerinfo_MVD.  QWD: read from cl.frames[validsequence]
	 * as normal — QWD demos include the real client/server handshake
	 * so validsequence advances correctly. */
	if (cls.mvdplayback)
		return &qnn_mvd_latest_playerstate[cl.playernum];

	if (cl.validsequence <= 0)
		return NULL;
	return &cl.frames[cl.validsequence & UPDATE_MASK].playerstate[cl.playernum];
}

/* ── Snapshot capture ──────────────────────────────────────────── */

void QNN_CaptureBaseSnapshot(qnn_snapshot_t *snapshot)
{
	player_state_t *ps;
	int items;

	memset(snapshot, 0, sizeof(*snapshot));

	ps = QNN_GetPlayerState();

	/* Origin: prefer simorg (predicted), fallback to playerstate */
	if (cl.simorg[0] != 0.0f || cl.simorg[1] != 0.0f || cl.simorg[2] != 0.0f)
	{
		VectorCopy(cl.simorg, snapshot->player_origin);
	}
	else if (ps != NULL)
	{
		VectorCopy(ps->origin, snapshot->player_origin);
	}

	/* Velocity: prefer simvel (predicted), fallback to playerstate */
	if (cl.simvel[0] != 0.0f || cl.simvel[1] != 0.0f || cl.simvel[2] != 0.0f)
	{
		VectorCopy(cl.simvel, snapshot->player_velocity);
	}
	else if (ps != NULL)
	{
		VectorCopy(ps->velocity, snapshot->player_velocity);
	}

	/* View angles: same as NQ */
	VectorCopy(cl.viewangles, snapshot->player_view_angles);

	/* Stats: same as NQ */
	snapshot->health = cl.stats[STAT_HEALTH];
	snapshot->armor = cl.stats[STAT_ARMOR];
	snapshot->armor_type = QNN_CurrentArmortype();
	snapshot->ammo = cl.stats[STAT_AMMO];
	snapshot->ammo_shells = cl.stats[STAT_SHELLS];
	snapshot->ammo_nails = cl.stats[STAT_NAILS];
	snapshot->ammo_rockets = cl.stats[STAT_ROCKETS];
	snapshot->ammo_cells = cl.stats[STAT_CELLS];

	/* Items: QW stores items in stats[STAT_ITEMS] */
	items = cl.stats[STAT_ITEMS];
	snapshot->weapons_owned = items & (IT_SHOTGUN | IT_SUPER_SHOTGUN | IT_NAILGUN | IT_SUPER_NAILGUN | IT_GRENADE_LAUNCHER | IT_ROCKET_LAUNCHER | IT_LIGHTNING);
	snapshot->items_owned = items;
	snapshot->weapon_id = QNN_WeaponId();

	/* Ground/water: from playerstate + pmove globals.  The QW client
	 * runs CL_PredictMove every Host_Frame, which updates the module-
	 * level `waterlevel` to the tracked player's current value.  We
	 * capture it here before any candidate physics sim clobbers it. */
	if (ps != NULL)
		snapshot->grounded = (ps->onground != -1) ? true : false;
	snapshot->waterlevel = waterlevel;

	snapshot->current_region_id = 0;
}

/* ── Token emission (shared format with NQ) ───────────────────── */

void QNN_SelfEmitToken(qnn_self_token_t *out, const qnn_snapshot_t *snapshot)
{
	int items = snapshot->items_owned;
	vec3_t vel_view;
	int i;

	memset(out, 0, sizeof(*out));

	out->health = (float)snapshot->health / QNN_MAX_HEALTH;
	out->armor = (float)snapshot->armor * snapshot->armor_type / QNN_MAX_ARMOR;
	out->weapon_sg = (items & IT_SUPER_SHOTGUN) ? 1.0f : (items & IT_SHOTGUN) ? 0.5f : 0.0f;
	out->weapon_ng = (items & IT_SUPER_NAILGUN) ? 1.0f : (items & IT_NAILGUN) ? 0.5f : 0.0f;
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
