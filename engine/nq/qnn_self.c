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

/* QNN_WeaponId, QNN_CurrentArmortype, QNN_SelfEmitToken live in
 * common/qnn_self.c — they read cl.stats[*] / snapshot only and have
 * no NQ/QW divergence. */

/* NQ has no per-cmd RTT plumbing: the protocol records server→client
 * messages only (no outgoing usercmd records, no netchan ack), so
 * client-side latency is unmeasurable from a demo.  Stubbed so
 * callers in shared collect code link cleanly.  Same story for
 * the MVD-style back-shift helper — NQ has no svc_updateping
 * broadcast either. */
float QNN_LatencySeconds(void) { return -1.0f; }
int QNN_LatencyFrames(int emit_hz) { (void)emit_hz; return 0; }
void QNN_ObservePings(void) {}
void QNN_ResetPingEstimator(void) {}
int QNN_PressBackShiftFrames(int player_slot, int emit_hz)
{
	(void)player_slot; (void)emit_hz;
	return 0;
}

int QNN_PressPingMs(int player_slot) { (void)player_slot; return 0; }
int QNN_SelfPingMs(void) { return 0; }

/* NQ inventory lives in cl.items directly (not cl.stats[STAT_ITEMS]).
 * Same cycle order as QW since the IT_* constants and weapons.qc
 * priority match between the two engines. */
static int qnn_id_from_item(int item)
{
	if (item == IT_AXE) return 1;
	if (item == IT_SHOTGUN) return 2;
	if (item == IT_SUPER_SHOTGUN) return 3;
	if (item == IT_NAILGUN) return 4;
	if (item == IT_SUPER_NAILGUN) return 5;
	if (item == IT_GRENADE_LAUNCHER) return 6;
	if (item == IT_ROCKET_LAUNCHER) return 7;
	if (item == IT_LIGHTNING) return 8;
	return 0;
}

static int qnn_has_ammo_for(int item)
{
	if (item == IT_AXE) return 1;
	if (item == IT_SHOTGUN) return cl.stats[STAT_SHELLS] >= 1;
	if (item == IT_SUPER_SHOTGUN) return cl.stats[STAT_SHELLS] >= 2;
	if (item == IT_NAILGUN) return cl.stats[STAT_NAILS] >= 1;
	if (item == IT_SUPER_NAILGUN) return cl.stats[STAT_NAILS] >= 2;
	if (item == IT_GRENADE_LAUNCHER) return cl.stats[STAT_ROCKETS] >= 1;
	if (item == IT_ROCKET_LAUNCHER) return cl.stats[STAT_ROCKETS] >= 1;
	if (item == IT_LIGHTNING) return cl.stats[STAT_CELLS] >= 1;
	return 0;
}

int QNN_NextWeaponId(int reverse)
{
	static const int ring[8] = {
		IT_AXE, IT_SHOTGUN, IT_SUPER_SHOTGUN, IT_NAILGUN,
		IT_SUPER_NAILGUN, IT_GRENADE_LAUNCHER, IT_ROCKET_LAUNCHER,
		IT_LIGHTNING
	};
	int active = cl.stats[STAT_ACTIVEWEAPON];
	int items = cl.items;
	int idx = -1;
	int i, step;
	for (i = 0; i < 8; ++i)
		if (ring[i] == active) { idx = i; break; }
	if (idx < 0)
		return 0;
	step = reverse ? -1 : 1;
	for (i = 0; i < 8; ++i)
	{
		int candidate;
		idx = (idx + step + 8) & 7;
		candidate = ring[idx];
		if ((items & candidate) && qnn_has_ammo_for(candidate))
			return qnn_id_from_item(candidate);
	}
	return 0;
}

int QNN_CurrentFrags(void)
{
	if (cl.viewentity > 0 && cl.scores != NULL && cl.viewentity - 1 < cl.maxclients)
		return cl.scores[cl.viewentity - 1].frags;
	return cl.stats[STAT_FRAGS];
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
