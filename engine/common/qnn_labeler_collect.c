/*
 * qnn_labeler_collect.c — Labeler-mode LOBS emit path.
 *
 * Lifecycle:
 *   - QNN_LabelerCollectReset() called from QW main per demo.
 *   - QNN_LabelerHandleTick() called from the per-tick play loop when
 *     qnn_runtime.labeler_mode is set.  Bypasses the QOBS pipeline.
 *
 * One LOBS frame per native tick.  Target labels come from the
 * action_label filled upstream — usercmd truth on QWD, MVD-rule
 * inference on real MVD.  See `target_valid_mask` rules in the header
 * comment (mirrors src/demo/sanitize.py).
 *
 * Single global fire cooldown counter (NOT per-weapon): vanilla Quake
 * stores attack_finished on the player edict, not per-weapon, so a
 * press carries cooldown across weapon switches.
 */

#include "qnn_labeler_collect.h"
#include "qnn_collect_helpers.h"
#include "qnn_io.h"

#include <string.h>

/* ── Module-private state ──────────────────────────────────────────── */

typedef struct
{
	/* Earliest qnn_runtime.tick at which a NEXT attack press would clear
	 * attack_finished — i.e., be "engine-effective".  Updated each time a
	 * press is accepted in the emit branch.  Used only to fill bit3 of
	 * target_valid_mask in the LOBS payload. */
	int fire_next_ok;
} qnn_labeler_state_t;

static qnn_labeler_state_t lab_state;

void QNN_LabelerCollectReset(void)
{
	memset(&lab_state, 0, sizeof(lab_state));
}

/* ── fp16 conversion (used by LOBS writer) ─────────────────────────── */

/* IEEE-754 binary32 → binary16 conversion.  Round-to-nearest-even.
 * Sufficient for the labeler obs fields (velocity / view-delta, both
 * normalized into ~[-1, 1] so the exponent never overflows binary16). */
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
	const float view_delta[3],
	int c_rule_fire,
	int c_rule_jump,
	uint8_t move_packed,
	uint8_t target_valid_mask,
	uint8_t usercmd_fire,
	uint8_t weapon_id)
{
	uint8_t header[10];
	uint16_t vel_h[3], view_h[3];
	uint8_t  mid_u8, fire_u8, jump_u8;

	memcpy(header + 0, &tick,    4);
	memcpy(header + 4, &tick_hz, 4);
	memcpy(header + 8, &flags,   2);

	vel_h[0] = QNN_F32ToF16(pos_delta_vel[0]);
	vel_h[1] = QNN_F32ToF16(pos_delta_vel[1]);
	vel_h[2] = QNN_F32ToF16(pos_delta_vel[2]);

	view_h[0] = QNN_F32ToF16(view_delta[0]);
	view_h[1] = QNN_F32ToF16(view_delta[1]);
	view_h[2] = QNN_F32ToF16(view_delta[2]);

	mid_u8  = (uint8_t)(movement_id & 0xFF);
	fire_u8 = (uint8_t)(c_rule_fire ? 1 : 0);
	jump_u8 = (uint8_t)(c_rule_jump ? 1 : 0);

	fwrite("LOBS",  1, 4, out);
	fwrite(header,  1, sizeof(header), out);
	fwrite(vel_h,   1, sizeof(vel_h),  out);
	fwrite(&mid_u8, 1, 1, out);
	fwrite(view_h,  1, sizeof(view_h), out);
	fwrite(&fire_u8, 1, 1, out);
	fwrite(&jump_u8, 1, 1, out);
	fwrite(&move_packed, 1, 1, out);
	fwrite(&target_valid_mask, 1, 1, out);
	fwrite(&usercmd_fire, 1, 1, out);
	fwrite(&weapon_id, 1, 1, out);
	fflush(out);
}

/* ── Per-tick labeler emit ─────────────────────────────────────────── */

void QNN_LabelerHandleTick(const qnn_snapshot_t *snapshot, FILE *out)
{
	vec3_t pos_delta_body;
	int mid;
	int c_fire, c_jump;
	uint8_t mp;
	uint8_t target_valid_mask;
	int ud_class;

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

	/* C-rule fire: sound + ammo decrement; velocity-free, so it survives
	 * the player_velocity zeroing in labeler-mode. */
	c_fire = QNN_DetectFireEvent(snapshot) ? 1 : 0;

	/* C-rule jump: ground→air transition with world-frame upward velocity
	 * above the engine-jump impulse floor (~270 u/s).  prev_velocity[2]
	 * is the per-tick origin z-delta / dt — non-zero on the impulse tick
	 * even with snapshot velocity zeroed. */
	c_jump = (qnn_runtime.prev_grounded
		&& !snapshot->grounded
		&& qnn_runtime.prev_velocity[2] > 100.0f) ? 1 : 0;

	/* Pack move target.  Threshold 0.1 matches MOVE_AXIS_THRESHOLD on
	 * the Python side.  ud_class is the resulting per-tick ud class
	 * (0=neg, 1=none, 2=pos) — held for the jump-acceptance check below. */
	ud_class = 1;
	{
		int i;
		mp = 0;
		for (i = 0; i < 3; i++)
		{
			float v = snapshot->action_label.move[i];
			uint8_t cls = (v >  0.1f) ? 2
				: (v < -0.1f) ? 0 : 1;
			mp |= (uint8_t)((cls & 0x3) << (i * 2));
			if (i == 2)
				ud_class = cls;
		}
	}

	/* ── target_valid_mask ─────────────────────────────
	 * Engine-effectiveness bits per axis (mirrors
	 * src/demo/sanitize.py rules):
	 *   bit0 = fb       (alive)
	 *   bit1 = lr       (alive)
	 *   bit2 = ud       (alive; if pressed up, grounded or in water)
	 *   bit3 = fire     (alive; if pressed, held weapon off cooldown)
	 *   bit4 = weapon   (alive — dense held-weapon target, no
	 *                    engine-effect gate)
	 * Bits 5..7 reserved (zero). */
	target_valid_mask = 0;
	if (snapshot->health > 0)
	{
		int wid = snapshot->weapon_id;
		int fire_press = snapshot->action_label.fire ? 1 : 0;
		qboolean in_water = (snapshot->waterlevel >= 2);
		qboolean ud_pressed_up = (ud_class == 2);
		qboolean ud_pressed_down = (ud_class == 0);
		/* fb / lr always pass through while alive. */
		target_valid_mask |= 0x01;
		target_valid_mask |= 0x02;

		/* ud: pressed-up is accepted only when grounded or in water;
		 * pressed-down only in water (no crouch in vanilla QW).  The
		 * "none" class (ud_class == 1) is the unconditional pass-
		 * through majority. */
		if (ud_pressed_up)
		{
			if (snapshot->grounded || in_water)
				target_valid_mask |= 0x04;
		}
		else if (ud_pressed_down)
		{
			if (in_water)
				target_valid_mask |= 0x04;
		}
		else
		{
			target_valid_mask |= 0x04;
		}

		/* fire: check per-weapon attack_finished cooldown.  Native-tick
		 * cooldowns match src/demo/sanitize.py::FIRE_COOLDOWN_NATIVE and
		 * scripts/compare_collects.py — Python-rounded (banker's) from
		 * the QC attack_finished delays:
		 *   axe 0.5s→38  sg 0.5s→38  ssg 0.7s→54
		 *   ng  0.2s→15  sng 0.2s→15  gl  0.6s→46
		 *   rl  0.8s→62  lg  0.1s→8
		 * fixed_tick_hz is the actual emit cadence (matches native rate
		 * when hello.tick_hz=0; matches the requested rate otherwise).
		 * When fixed_tick_hz != 77 (e.g., demo recorded at 70 Hz, or
		 * non-native emit) the table is scaled proportionally. */
		if (fire_press == 0)
		{
			target_valid_mask |= 0x08;
		}
		else if (wid >= 1 && wid <= 8
			&& qnn_runtime.fixed_tick_hz > 0
			&& qnn_runtime.tick >= lab_state.fire_next_ok)
		{
			static const int k_fire_cd_native[9] = {
				0,
				38, 38, 54, 15,
				15, 46, 62,  8,
			};
			int base_cd = k_fire_cd_native[wid];
			int cd_ticks = (qnn_runtime.fixed_tick_hz == 77)
				? base_cd
				: (int)((float)base_cd
					* (float)qnn_runtime.fixed_tick_hz
					/ 77.0f + 0.5f);
			target_valid_mask |= 0x08;
			lab_state.fire_next_ok =
				qnn_runtime.tick + cd_ticks;
		}
		/* else: held weapon out of {1..8} (e.g., 0 = no weapon held
		 * mid-respawn); press is a no-op. */

		/* weapon (held-weapon target): dense per-frame — action_label
		 * .weapon is the currently-held weapon byte, which is observable
		 * and unambiguous on every alive frame.  The labeler isn't
		 * predicting a sparse "switch event"; it predicts "what weapon
		 * is held right now."  No engine-effect filtering applies.  Bit
		 * is on iff alive. */
		target_valid_mask |= 0x10;
	}
	/* else: dead — all bits remain 0 (masked out). */

	{
		uint16_t lobs_flags = snapshot->done ? 0x02u : 0u;
		uint8_t usercmd_fire = (uint8_t)
			(snapshot->action_label.fire ? 1 : 0);
		uint8_t weapon_id_u8 = (uint8_t)
			(snapshot->weapon_id & 0xFF);
		QNN_EmitLabelerTick(out,
			qnn_runtime.tick,
			qnn_runtime.fixed_tick_hz,
			lobs_flags,
			pos_delta_body, mid,
			snapshot->action_label.look,
			c_fire, c_jump, mp,
			target_valid_mask,
			usercmd_fire,
			weapon_id_u8);
	}

	QNN_SavePrev(snapshot, qnn_runtime.fixed_dt);
}
