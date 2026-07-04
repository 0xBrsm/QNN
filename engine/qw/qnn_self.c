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
#include "qnn_collect_helpers.h"   /* qnn_runtime: force_mvd_emit */

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

/* QNN_WeaponId, QNN_CurrentArmortype, QNN_SelfEmitToken live in
 * common/qnn_self.c — they read cl.stats[*] / snapshot only and have
 * no NQ/QW divergence. */

/* Most recent measured round-trip latency for an acked usercmd, in
 * the engine's wall-clock seconds.  cl_demo.c sets senttime when a
 * DEM_CMD record is processed; cl_parse.c sets receivedtime when a
 * server packet arrives carrying that cmd's sequence in its netchan
 * ack.  Single source of truth for ping — prefer this over the
 * smoothed svc_updateping or the manifest's per-demo avg_ping_ms
 * when per-event ping is needed.  Returns -1.0f if no ack yet. */
float QNN_LatencySeconds(void)
{
	int ack = cls.netchan.incoming_acknowledged & UPDATE_MASK;
	float sent = cl.frames[ack].senttime;
	float recv = cl.frames[ack].receivedtime;
	if (recv < 0.0f)
		return -1.0f;
	return recv - sent;
}

int QNN_LatencyFrames(int emit_hz)
{
	float sec = QNN_LatencySeconds();
	if (sec < 0.0f || emit_hz <= 0)
		return 0;
	return (int)(sec * (float)emit_hz + 0.5f);
}

/* QNN_NextWeaponId — QW reads inventory from cl.stats[STAT_ITEMS]
 * (NQ uses cl.items directly; see nq/qnn_self.c).  Simulates the
 * server's CycleWeaponCommand (forward) / CycleWeaponReverseCommand
 * (reverse) ring walk, gated on ownership + per-weapon ammo. */
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

/* ── MVD back-shift helper ───────────────────────────────────────
 *
 * QNN_ObservePings() should be called once per emit (or per host_frame)
 * to keep a per-demo running median of svc_updateping values.  The
 * median is used by QNN_PressBackShiftFrames as a fallback when the
 * raw cl.players[slot].ping reading is out of range — the engine never
 * validates the signed-short field, so spectator-slot memory can leak
 * uninitialized values (-13057, 999) into our consumers.
 *
 * Formula:
 *   shift = ping_ms / (1000 / emit_hz)
 *
 * Integer floor over half of round-trip ping.  Sub-frame components
 * (client cmd quantization ~7 ms, server cmd dispatch ~6 ms, emit-window
 * phase) are well below our 25 ms half-frame boundary so they don't
 * affect integer output.  Also used by the QWD walk-back to derive a
 * per-event ping-driven cap (max plausible delay), so stale impulses
 * past that distance don't get falsely anchored to subsequent server-
 * driven transitions. */

#define QNN_PING_RING 32
#define QNN_PING_MAX 500          /* hard cap; nothing real exceeds this */
#define QNN_PING_OUTLIER_K 5      /* relative cap = k × running median */
static int s_ping_ring[QNN_PING_RING];
static int s_ping_count = 0;
static int s_ping_cursor = 0;
/* Per-slot last-observed ping for change detection.  svc_updateping
 * broadcasts on the server's ping cycle (~1 s); the same value sits in
 * cl.players[].ping across many emit ticks.  Push only fresh values. */
static int s_last_slot_ping[MAX_CLIENTS];
static qboolean s_last_slot_ping_valid[MAX_CLIENTS];

void QNN_ObservePings(void)
{
	/* Sweep all slots for fresh, plausible svc_updateping values.
	 * Spectator-recorded QWDs never get pings broadcast for cl.playernum
	 * (the recorder's slot), so scoping to that slot misses the actual
	 * players' pings.  Pooling across slots gives a robust per-demo
	 * baseline regardless of whether the demo is player or spectator. */
	int slot;
	for (slot = 0; slot < MAX_CLIENTS; ++slot)
	{
		int p = cl.players[slot].ping;
		if (p <= 0 || p > QNN_PING_MAX)
			continue;
		if (s_last_slot_ping_valid[slot] && s_last_slot_ping[slot] == p)
			continue;
		s_last_slot_ping[slot] = p;
		s_last_slot_ping_valid[slot] = true;
		s_ping_ring[s_ping_cursor & (QNN_PING_RING - 1)] = p;
		s_ping_cursor++;
		if (s_ping_count < QNN_PING_RING) s_ping_count++;
	}
}

void QNN_ResetPingEstimator(void)
{
	memset(s_ping_ring, 0, sizeof(s_ping_ring));
	s_ping_count = 0;
	s_ping_cursor = 0;
	memset(s_last_slot_ping, 0, sizeof(s_last_slot_ping));
	memset(s_last_slot_ping_valid, 0, sizeof(s_last_slot_ping_valid));
}

static int qnn_median_ping_ms(void)
{
	int buf[QNN_PING_RING];
	int i, j, v;
	if (s_ping_count == 0) return 0;  /* no observations yet */
	memcpy(buf, s_ping_ring, sizeof(int) * (size_t)s_ping_count);
	for (i = 1; i < s_ping_count; ++i)
	{
		v = buf[i];
		j = i - 1;
		while (j >= 0 && buf[j] > v) { buf[j+1] = buf[j]; --j; }
		buf[j+1] = v;
	}
	return buf[s_ping_count / 2];
}

int QNN_PressPingMs(int player_slot)
{
	int raw;
	int median;

	if (player_slot < 0 || player_slot >= MAX_CLIENTS)
		return 0;
	raw = cl.players[player_slot].ping;
	median = qnn_median_ping_ms();
	if (raw <= 0 || raw > QNN_PING_MAX
		|| (median > 0 && raw > QNN_PING_OUTLIER_K * median))
		return median;
	return raw;
}

float QNN_PressPingSec(int player_slot)
{
	return (float)QNN_PressPingMs(player_slot) / 1000.0f;
}

int QNN_SelfPingMs(void)
{
	if (cl.playernum < 0 || cl.playernum >= MAX_CLIENTS)
		return 0;
	return QNN_PressPingMs(cl.playernum);
}

int QNN_PressBackShiftFrames(int player_slot, int emit_hz)
{
	int ping_ms;
	int emit_period_ms;

	if (emit_hz <= 0) return 0;
	ping_ms = QNN_PressPingMs(player_slot);
	emit_period_ms = 1000 / emit_hz;
	return ping_ms / emit_period_ms;
}

int QNN_NextWeaponId(int reverse)
{
	static const int ring[8] = {
		IT_AXE, IT_SHOTGUN, IT_SUPER_SHOTGUN, IT_NAILGUN,
		IT_SUPER_NAILGUN, IT_GRENADE_LAUNCHER, IT_ROCKET_LAUNCHER,
		IT_LIGHTNING
	};
	int active = cl.stats[STAT_ACTIVEWEAPON];
	int items = cl.stats[STAT_ITEMS];
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
	if (cl.playernum >= 0 && cl.playernum < MAX_CLIENTS)
		return cl.players[cl.playernum].frags;
	return 0;
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

	/* Origin / velocity: prefer cl.simorg / cl.simvel (predicted via
	 * CL_PredictMove on QWD's usercmds) and fall back to the server
	 * playerstate.  EXCEPT under force_mvd_emit, where we must not leak
	 * QWD-specific pmove integration into the obs.  Real MVD playback
	 * delivers only server-authoritative origin / velocity to the
	 * client; this branch takes the same path on QWD so the inference /
	 * training pipeline operates on the same distribution it would see
	 * during true MVD playback. */
	{
		qboolean mvd_faithful = qnn_runtime.force_mvd_emit;
		if (!mvd_faithful
			&& (cl.simorg[0] != 0.0f || cl.simorg[1] != 0.0f
				|| cl.simorg[2] != 0.0f))
		{
			VectorCopy(cl.simorg, snapshot->player_origin);
		}
		else if (ps != NULL)
		{
			VectorCopy(ps->origin, snapshot->player_origin);
		}

		if (!mvd_faithful
			&& (cl.simvel[0] != 0.0f || cl.simvel[1] != 0.0f
				|| cl.simvel[2] != 0.0f))
		{
			VectorCopy(cl.simvel, snapshot->player_velocity);
		}
		else if (ps != NULL)
		{
			VectorCopy(ps->velocity, snapshot->player_velocity);
		}
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
	 * capture it here before any candidate physics sim clobbers it.
	 *
	 * MVD-path exception: the MVD playerinfo parser never writes
	 * onground (memset 0 reads as "!= -1" = always grounded — real
	 * collects came out 100% ground), and CL_PredictMove's waterlevel
	 * tracks the idle local client, not the tracked player.  Derive
	 * both from the BSP instead: standing = the player hull is blocked
	 * one unit below the origin; water = point contents at the origin.
	 * pmove.physents are maintained by QNN_PhysInit on these paths.
	 * force_mvd_emit uses the same derivation so paired evals and the
	 * GBT labeler's features see real-MVD semantics. */
	if ((cls.mvdplayback || qnn_runtime.force_mvd_emit)
		&& cl.worldmodel != NULL)
	{
		/* Query the worldmodel hulls directly — pmove.physents are
		 * not reliably populated at snapshot-capture time.  hull 1 is
		 * the player-size expanded hull, so a point query at the
		 * origin answers "would a player whose origin is here clip
		 * solid"; one unit below the origin that means standing. */
		hull_t *player_hull = &cl.worldmodel->hulls[1];
		hull_t *point_hull = &cl.worldmodel->hulls[0];
		vec3_t below;
		int contents;

		VectorCopy(snapshot->player_origin, below);
		below[2] -= 1.0f;
		snapshot->grounded =
			(PM_HullPointContents(player_hull,
				player_hull->firstclipnode, below) == CONTENTS_SOLID);
		contents = PM_HullPointContents(point_hull,
			point_hull->firstclipnode, snapshot->player_origin);
		snapshot->waterlevel =
			(contents == CONTENTS_WATER || contents == CONTENTS_SLIME
			 || contents == CONTENTS_LAVA) ? 2 : 0;
	}
	else
	{
		if (ps != NULL)
			snapshot->grounded = (ps->onground != -1) ? true : false;
		snapshot->waterlevel = waterlevel;
	}

	snapshot->current_region_id = 0;
}

/* QNN_SelfEmitToken lives in common/qnn_self.c. */
