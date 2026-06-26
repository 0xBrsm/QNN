/*
 * qnn_self.c (common) — Game-agnostic self-token helpers.
 *
 * QNN_WeaponId, QNN_CurrentArmortype, and QNN_SelfEmitToken read from the
 * shared engine state (cl.stats[*]) or the snapshot struct, both of which
 * are populated by the per-game QNN_CaptureBaseSnapshot.  They have no
 * NQ/QW divergence and live here so they exist in one place.
 *
 * Native-width policy (post-engine_norm): emit raw engine values.  All
 * scaling / normalization happens in QNN_IOPackObsBuffer (wire) or the
 * model-side SelfDequantizer (qnn.model.dequant).
 */

#include "qnn_io.h"               /* QNN_TIME_SCALE — canonical time normalization */
#include "qnn.h"                  /* QNN_ProgsGetAttackCdRemaining */
#include "qnn_collect_helpers.h"  /* qnn_runtime */

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

/* ── Token emission (raw engine values; wire pack handles widths) ── */

void QNN_SelfEmitToken(qnn_self_token_t *out, const qnn_snapshot_t *snapshot)
{
	memset(out, 0, sizeof(*out));

	out->health       = snapshot->health;
	out->raw_armor    = snapshot->armor;
	out->armor_type   = snapshot->armor_type;
	out->ammo_shells  = snapshot->ammo_shells;
	out->ammo_nails   = snapshot->ammo_nails;
	out->ammo_rockets = snapshot->ammo_rockets;
	out->ammo_cells   = snapshot->ammo_cells;

	{
		vec3_t vel_view;
		QNN_RelativeFrame(snapshot->player_view_angles, snapshot->player_velocity, vel_view);
		out->vel[0] = vel_view[0];
		out->vel[1] = vel_view[1];
		out->vel[2] = vel_view[2];
	}

	{
		/* attack_finished cooldown — read native-frame remainder from
		 * the QC VM evaluator, convert to seconds.  The model
		 * dequantizer divides by QNN_TIME_SCALE (60s) like every other
		 * time scalar in the obs.  NQ build's stub returns 0 (no QC
		 * VM emulation), so this is a no-op on NQ paths and live for QW. */
		int af_frames = QNN_ProgsGetAttackCdRemaining(
			qnn_runtime.tick, qnn_runtime.fixed_tick_hz);
		float af_sec = (qnn_runtime.fixed_tick_hz > 0)
			? (float)af_frames / (float)qnn_runtime.fixed_tick_hz
			: 0.0f;
		if (af_sec < 0.0f) af_sec = 0.0f;
		out->attack_finished = af_sec;
	}

	out->weapon_id = qnn_weapon_subject_from_id(snapshot->weapon_id);

	switch (snapshot->waterlevel)
	{
	case 1: out->movement_id = 2; break;
	case 2: out->movement_id = 3; break;
	case 3: out->movement_id = 4; break;
	default: out->movement_id = snapshot->grounded ? 0 : 1; break;
	}

	/* Raw cl.items bitfield — model-side dequantizer extracts the
	 * weapon-owned flags, armor-type ID, and powerup IDs via the
	 * IT_* bit positions documented in qnn_vocab.h / engine_norm.py. */
	out->items = (int32_t)snapshot->items_owned;

	/* view_pitch normalized to ~[-1, 1] via deg/90. Engine clamps pitch
	 * to ±70° (vendor/quake/QW/client/cl_input.c CL_AdjustAngles), so
	 * the result stays well inside the i8 range. Used by the model's
	 * self.motion subtoken; spatial dirs no longer carry pitch. */
	out->view_pitch = snapshot->player_view_angles[0] * (1.0f / 90.0f);

	/* look_delta = realized_look[t] - realized_look[t-1], where
	 * realized_look = forward(current view) expressed in the PREVIOUS
	 * emit's view basis — i.e. the turn just made (== the act_look label
	 * one frame back). The difference is the change in that turn (~0 under
	 * steady rotation). Self-contained 2-deep carry in qnn_runtime. Live
	 * inference calls QNN_IOEmit once per Host_Frame, so the carry advances
	 * once per frame and look_delta is exact — verified byte-for-byte against
	 * the BC preload's look[t-1]-look[t-2] on a real demo (99.94% of frames).
	 * CAVEAT: the demo-COLLECTION loop runs several snapshot/emit passes per
	 * frame (label snapshots, pmove re-emits) which can double-advance the
	 * carry and transiently corrupt look_delta for ~2 frames around those
	 * events. Harmless: collection drops look_delta before caching
	 * (qnn.bc.collect) and the preload re-derives it per-episode from the
	 * cached look column — so don't trust the collection-path look_delta, use
	 * the preload. Reset across episodes via QNN_IOUpdate's reset_flag; out is
	 * memset to 0 above, so the first frame(s) emit a zero look_delta. */
	{
		vec3_t cur_forward, realized;
		QNN_ForwardFromAngles(snapshot->player_view_angles, cur_forward);
		if (qnn_runtime.ld_has_prev_view)
		{
			QNN_RelativeFrame(qnn_runtime.ld_prev_view, cur_forward, realized);
			if (qnn_runtime.ld_has_prev_realized)
				VectorSubtract(realized, qnn_runtime.ld_prev_realized,
					out->look_delta);
			VectorCopy(realized, qnn_runtime.ld_prev_realized);
			qnn_runtime.ld_has_prev_realized = true;
		}
		VectorCopy(snapshot->player_view_angles, qnn_runtime.ld_prev_view);
		qnn_runtime.ld_has_prev_view = true;
	}
}
