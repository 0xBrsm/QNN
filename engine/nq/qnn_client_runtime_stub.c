/* qnn_client_runtime_stub.c — minimal qnn_runtime definition for the
 * NQ live client.
 *
 * qnn_runtime is normally provided by qnn_collect_helpers.c, which the
 * client explicitly excludes (touches sv.* and pulls in the whole
 * collect/labeler stack). After commit a834d756, qnn_self_common.c
 * began reading qnn_runtime.tick / qnn_runtime.fixed_tick_hz to
 * normalize the attack_finished cooldown scalar, so the client needs
 * a definition too — just enough to satisfy the link and supply
 * fixed_tick_hz at the canonical 20 Hz.
 *
 * Set by qnn_client_main.c at init; advanced per frame from the
 * main step loop.
 *
 * QNN_PackInputMask lives in qnn_collect_helpers.c too (same excluded
 * source); qnn_onnx.c calls it to pack the model's per-axis move/attack
 * decision into the action press byte. It's pure bit-packing with no sv.*
 * dependency, so we provide a local copy here rather than pulling in the
 * whole collect stack. Keep this byte-identical to the canonical definition
 * (qnn_collect_helpers.c) — they must agree on the bit layout.
 */
#include <stdint.h>

#include "qnn_collect_helpers.h"

qnn_runtime_t qnn_runtime;

uint8_t QNN_PackInputMask(
	int alive,
	int fb_act_neg,  int fb_act_pos,
	int lr_act_neg,  int lr_act_pos,
	int up_act_neg,  int up_act_pos,
	int jump_act,
	int attack_act)
{
	uint8_t m;

	if (!alive)
		return 0;
	m = 0;
	if (attack_act)  m |= 0x01;	/* bit 0 */
	if (fb_act_neg)  m |= 0x02;	/* bit 1 */
	if (fb_act_pos)  m |= 0x04;	/* bit 2 */
	if (lr_act_neg)  m |= 0x08;	/* bit 3 */
	if (lr_act_pos)  m |= 0x10;	/* bit 4 */
	if (up_act_neg)  m |= 0x20;	/* bit 5 */
	if (up_act_pos)  m |= 0x40;	/* bit 6 */
	if (jump_act)    m |= 0x80;	/* bit 7 */
	return m;
}
