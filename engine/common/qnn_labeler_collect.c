/*
 * qnn_labeler_collect.c — Labeler-mode LOBS emit path.
 *
 * Lifecycle:
 *   - QNN_LabelerCollectReset() called from QW main per demo.
 *   - QNN_LabelerHandleTick() called from the per-tick play loop when
 *     qnn_runtime.labeler_mode is set.  Bypasses the QOBS pipeline.
 *
 * One LOBS frame per native tick.
 *
 * Wire layout puts two parallel columns side-by-side per tick:
 *   `cmd_*` — the player's aggregated usercmd bytes (angles, move,
 *             buttons, impulse), in QW wire-format precision (int16
 *             for angles/move, raw u8 for buttons/impulse).
 *   `op_input` — strict per-axis op mask of that usercmd: bit i set
 *                iff the engine acted on the player's press on axis i
 *                this tick.  No-input frames → bit clear (caller
 *                derives training-keep as `no_press | op_input_bit`).
 *
 * Source of truth for ops:
 *   bit 0 (fb)      : `cmd_move.fb != 0` (pmove integrates fb unconditionally)
 *   bit 1 (lr)      : `cmd_move.lr != 0`
 *   bit 2 (ud)      : QC PlayerJump when (button2 OR umove>0),
 *                     waterlevel>=2 when umove<0
 *   bit 3 (fire)    : QC W_WeaponFrame on button0
 *   bit 4 (impulse) : QC ImpulseCommands on last-non-zero impulse
 *
 * The QC predicates run via qnn_progs.c (qwprogs.dat in the real
 * server VM) — the C side never transcribes engine rules.
 */

#include "qnn_labeler_collect.h"
#include "qnn_collect_helpers.h"
#include "qnn_io.h"

#include <stdint.h>
#include <string.h>

/* ── Module-private state ──────────────────────────────────────────── */

/* No labeler-local state — the pmove jump driver and QC predicate-VM
 * state both live in their own modules and reset at demo boundaries
 * via QNN_QwdCollectReset / QNN_ProgsInit. */

void QNN_LabelerCollectReset(void)
{
	/* No-op: cross-tick state for the labeler lives in qwd_state
	 * (qnn_qwd_collect.c) and qnn_progs.c statics. */
}

/* ── fp16 conversion (used by LOBS writer) ─────────────────────────── */

/* IEEE-754 binary32 → binary16 conversion.  Round-to-nearest-even.
 * Sufficient for the labeler obs fields (velocity is normalized into
 * ~[-1, 1] so the exponent never overflows binary16). */
static uint16_t QNN_F32ToF16(float f)
{
	uint32_t u;
	uint32_t sign, exp32, mant32;
	int e16;
	uint16_t mant16;

	memcpy(&u, &f, 4);
	sign   = (u >> 31) & 0x1u;
	exp32  = (u >> 23) & 0xFFu;
	mant32 = u & 0x7FFFFFu;

	if (exp32 == 0xFFu)
		return (uint16_t)((sign << 15) | 0x7C00u
			| (mant32 ? 0x0200u : 0u));  /* inf / nan */
	if (exp32 == 0)
		return (uint16_t)(sign << 15);   /* zero / subnormal → 0 */

	e16 = (int)exp32 - 127 + 15;
	if (e16 >= 0x1F)
		return (uint16_t)((sign << 15) | 0x7C00u);  /* overflow → inf */
	if (e16 <= 0)
		return (uint16_t)(sign << 15);              /* underflow → 0 */

	mant16 = (uint16_t)(mant32 >> 13);
	/* Round-to-nearest-even on the dropped low bits. */
	if ((mant32 & 0x1000u)
		&& ((mant32 & 0x2FFFu) != 0 || (mant16 & 0x1u)))
	{
		if (mant16 == 0x3FFu)
		{
			mant16 = 0;
			e16++;
			if (e16 >= 0x1F)
				return (uint16_t)((sign << 15) | 0x7C00u);
		}
		else
		{
			mant16++;
		}
	}
	return (uint16_t)((sign << 15) | ((uint16_t)e16 << 10) | mant16);
}

/* ── LOBS writer ───────────────────────────────────────────────────── */

void QNN_EmitLabelerTick(FILE *out,
	int tick, int tick_hz, uint16_t flags,
	const float pos_delta_vel[3],
	int movement_id,
	const int16_t cmd_angles[3],
	const int16_t cmd_move[3],
	uint8_t cmd_buttons,
	uint8_t cmd_impulse,
	uint8_t op_input,
	uint8_t weapon_id,
	uint8_t c_rule_fire,
	uint8_t c_rule_jump)
{
	uint8_t header[10];
	uint16_t vel_h[3];
	uint8_t  mid_u8;

	memcpy(header + 0, &tick,    4);
	memcpy(header + 4, &tick_hz, 4);
	memcpy(header + 8, &flags,   2);

	vel_h[0] = QNN_F32ToF16(pos_delta_vel[0]);
	vel_h[1] = QNN_F32ToF16(pos_delta_vel[1]);
	vel_h[2] = QNN_F32ToF16(pos_delta_vel[2]);

	mid_u8 = (uint8_t)(movement_id & 0xFF);

	fwrite("LOBS",  1, 4, out);
	fwrite(header,  1, sizeof(header), out);
	fwrite(vel_h,   1, sizeof(vel_h),  out);   /* offset 0..5   */
	fwrite(&mid_u8, 1, 1, out);                 /* offset 6      */
	fwrite(cmd_angles, 1, 6, out);              /* offset 7..12  */
	fwrite(cmd_move,   1, 6, out);              /* offset 13..18 */
	fwrite(&cmd_buttons, 1, 1, out);            /* offset 19     */
	fwrite(&cmd_impulse, 1, 1, out);            /* offset 20     */
	fwrite(&op_input, 1, 1, out);               /* offset 21     */
	fwrite(&weapon_id, 1, 1, out);              /* offset 22     */
	fwrite(&c_rule_fire, 1, 1, out);            /* offset 23     */
	fwrite(&c_rule_jump, 1, 1, out);            /* offset 24     */
	fflush(out);
}

/* ── Per-tick labeler emit ─────────────────────────────────────────── */

void QNN_LabelerHandleTick(const qnn_snapshot_t *snapshot, FILE *out)
{
	vec3_t pos_delta_body;
	int mid;
	int16_t cmd_angles[3];
	int16_t cmd_move[3];
	uint8_t cmd_buttons;
	uint8_t cmd_impulse;
	int op_fire = 0;
	int op_jump = 0;
	int op_impulse = 0;
	int fmove_int = 0, smove_int = 0, umove_int = 0;
	int buttons_int = 0, impulse_int = 0;
	uint8_t op_input;
	int i;

	/* World-frame origin-delta velocity over the previous native tick —
	 * already computed by QNN_SavePrev. */
	QNN_RelativeFrame(snapshot->player_view_angles,
		qnn_runtime.prev_velocity, pos_delta_body);
	pos_delta_body[0] /= QNN_VELOCITY_SCALE;
	pos_delta_body[1] /= QNN_VELOCITY_SCALE;
	pos_delta_body[2] /= QNN_VELOCITY_SCALE;

	switch (snapshot->waterlevel)
	{
	case 1: mid = 2; break;
	case 2: mid = 3; break;
	case 3: mid = 4; break;
	default: mid = snapshot->grounded ? 0 : 1; break;
	}

	/* Per-cmd fire predicate + cmd-block aggregation. */
	QNN_QwdEvalOperativePerCmd(snapshot,
		&op_fire,
		&fmove_int, &smove_int, &umove_int,
		&buttons_int, &impulse_int);

	/* Per-cmd pmove driver for jump operativeness — replaces the QC
	 * PlayerJump predicate's K-gated workaround with direct observation
	 * of pmove's ground-jump success branch.  Frame-exact press
	 * attribution; no snapshot.grounded lag artifact. */
	op_jump = QNN_QwdEvalPmoveJump(snapshot);

	/* Pack cmd_move as int16 (raw QW units).  Clamp to int16 range —
	 * QW values rarely exceed ±400 so headroom is huge, but defensive. */
	{
		int vals[3];
		vals[0] = fmove_int;
		vals[1] = smove_int;
		vals[2] = umove_int;
		for (i = 0; i < 3; i++)
		{
			int v = vals[i];
			if (v >  32767) v =  32767;
			if (v < -32768) v = -32768;
			cmd_move[i] = (int16_t)v;
		}
	}
	cmd_buttons = (uint8_t)(buttons_int & 0xFF);
	cmd_impulse = (uint8_t)(impulse_int & 0xFF);

	/* Encode view angles in QW wire format: int16 = degrees × 65536/360,
	 * modular-wrapped to 16 bits.  Reader recovers degrees via
	 * (uint16)cmd_angles * (360/65536).  Matches MSG_WriteAngle16 /
	 * MSG_ReadAngle16 in vendor/quake/QW/client/common.c. */
	for (i = 0; i < 3; i++)
	{
		float deg = snapshot->player_view_angles[i];
		int q = (int)(deg * 65536.0f / 360.0f);
		cmd_angles[i] = (int16_t)(q & 0xFFFF);
	}

	/* op_impulse: stateful QC ImpulseCommands eval — maintains
	 * persistent self.weapon across calls so sticky-impulse cmds (held
	 * across the snapshot.weapon_id lag window) only fire op_impulse=1
	 * on the actual press tick.  Called unconditionally every tick so
	 * the predicate's snapshot-sync logic stays current even on
	 * cmd_impulse=0 ticks where server-forced switches may occur. */
	op_impulse = QNN_ProgsEvalWeaponImpulseOperative(
		snapshot->health, snapshot->items_owned,
		snapshot->ammo_shells, snapshot->ammo_nails,
		snapshot->ammo_rockets, snapshot->ammo_cells,
		snapshot->weapon_id, (int)cmd_impulse);

	/* Strict per-axis op-of-usercmd.  bit i set iff there was a press
	 * on axis i AND the engine acted on it this tick.  Same packer is
	 * used by the BC QWD path via QNN_QwdPackOpInput. */
	op_input = QNN_PackOpInput(
		(snapshot->health > 0),
		(cmd_move[0] != 0),
		(cmd_move[1] != 0),
		((cmd_buttons & 2) != 0) || (cmd_move[2] > 0),
		(cmd_move[2] < 0),
		(cmd_buttons & 1) != 0,
		(snapshot->waterlevel >= 2),
		op_jump, op_fire, op_impulse,
		(cmd_impulse != 0));

	{
		uint16_t lobs_flags = snapshot->done ? 0x02u : 0u;
		uint8_t weapon_id_u8 = (uint8_t)
			(snapshot->weapon_id & 0xFF);
		/* Sound-derived discrete-cmd reconstructions — same generators
		 * as the MVD inference path uses at apply time
		 * (QNN_SnapshotHasSelfWeaponFireSound / SelfJumpSound).  Sparse
		 * one-tick-per-event: a 1 here means the server multicast a
		 * fire/jump sound for the self entity at this snapshot.  The
		 * labeler trainer feeds these as features so train-time and
		 * apply-time input distributions match. */
		uint8_t c_rule_fire = QNN_SnapshotHasSelfWeaponFireSound(snapshot)
			? 1 : 0;
		uint8_t c_rule_jump = QNN_SnapshotHasSelfJumpSound(snapshot)
			? 1 : 0;

		QNN_EmitLabelerTick(out,
			qnn_runtime.tick,
			qnn_runtime.fixed_tick_hz,
			lobs_flags,
			pos_delta_body, mid,
			cmd_angles, cmd_move,
			cmd_buttons, cmd_impulse,
			op_input,
			weapon_id_u8,
			c_rule_fire,
			c_rule_jump);
	}

	QNN_SavePrev(snapshot, qnn_runtime.fixed_dt);
}
