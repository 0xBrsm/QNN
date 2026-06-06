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
 */
#include "qnn_collect_helpers.h"

qnn_runtime_t qnn_runtime;
