/*
 * qnn_obs_shim.h — Obs-API protocol layer + legacy wire-identity shim.
 *
 * WS2 of the demand-driven obs API (agents/plans/obs-api.md).  Three
 * concerns, all pure functions over qnn_obs_registry.h types (no
 * engine state, standalone-linkable for tests):
 *
 *   1. Declaration JSON parsing — the OP_ATTACH_DECL payload and the
 *      ONNX `obs_declaration` metadata prop share one fixed obs_api v1
 *      schema (mirrored in src/qnn/obs_api.py Declaration.to_dict):
 *        {"obs_api": 1,
 *         "state": ["health", ...],
 *         "atlas": {"yaw": 24, "bands": 11, "packed": true} | null,
 *         "entities": {"policy": "v3", "max_tokens": 16,
 *                      "paths": true} | null,
 *         "frame_bytes": 4096}          (optional, wire-shim only)
 *      Unknown keys, duplicate keys, missing sub-object params, type
 *      errors and trailing bytes are all hard parse errors — fail
 *      loud, no silent defaults.
 *
 *   2. Layout-reply serialization — the OP_ATTACH_DECL success reply,
 *      one JSON line:
 *        {"ok":true,"layout":{"frame_bytes":N,"fields":[{"name":...,
 *         "kind":...,"params":{...},"offset":N,"bytes":N,"dtype":...,
 *         "shape":[...]},...]}}
 *      Field order is plan (= registry) order.  dtype/shape are JSON
 *      null for percept fields (variable-length streams).  The dict
 *      must compare equal to the python mirror's compile_layout
 *      (qnn/obs_api.py parse_layout_reply) — the drivers hard-error
 *      on any divergence.
 *
 *   3. The wire-identity shim table — maps the four stamped legacy
 *      frame identities (wire.12.1 / wire.12.2 / wire.13.1 /
 *      wire.13.2, each pinned to its semantics contract) onto
 *      equivalent built-in declarations, so existing stamped ONNX
 *      models resolve to compiled emit plans without re-stamping.
 *      Bare `wire.12` / `wire.13` / anything else is NOT in the
 *      table — those stay refused exactly as before (qnn_onnx.c's
 *      retired-wire table names the fix).
 *
 * Bridge-protocol framing (shared by the worker binary channel and
 * the arena server/client channels; mirrored in qnn/obs_api.py):
 *   request: u8 opcode (QNN_OBS_OP_ATTACH_DECL) | u8 seat_index |
 *            u32le declaration-JSON byte length | the JSON bytes
 *   reply:   one JSON line on the existing response channel — the
 *            layout reply above, or {"ok":false,"error":"..."} and
 *            the engine refuses to start.
 * The framing reads live in the three process mains; this module
 * only parses the JSON payload and renders the reply.
 */

#ifndef QNN_OBS_SHIM_H
#define QNN_OBS_SHIM_H

#include "qnn_obs_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bridge opcode for the declaration handshake.  Chosen clear of every
 * existing opcode space it shares: the worker binary channel
 * (QNN_BINARY_OP_STEP = 1), the arena server channel (2/3/7/255) and
 * the arena client channel (1/4/5/6/7/255).  Mirrored as
 * OP_ATTACH_DECL in src/qnn/obs_api.py. */
#define QNN_OBS_OP_ATTACH_DECL        8

/* Upper bound accepted for the length prefix of an attach request —
 * a declaration JSON blob is a few hundred bytes; anything near this
 * bound is a corrupt frame. */
#define QNN_OBS_DECL_JSON_MAX         65536

/* Parse an obs_api v1 declaration JSON blob into *out.  `len` < 0
 * means NUL-terminated.  Returns false + fills `error` on any schema
 * violation; on failure *out is undefined.  Parsing validates SHAPE
 * (keys, types); QNN_ObsPlanCompile validates CONTENT (field names,
 * parameter values, frame_bytes bounds). */
qboolean QNN_ObsDeclParseJson(const char *json, int len,
	qnn_obs_decl_t *out, char *error, size_t error_size);

/* Render the OP_ATTACH_DECL success reply (no trailing newline) for a
 * compiled plan.  Returns false + fills `error` if `out_size` cannot
 * hold the reply (sizing bug — treat as fatal). */
qboolean QNN_ObsLayoutReplyJson(const qnn_obs_plan_t *plan,
	char *out, size_t out_size, char *error, size_t error_size);

/* Resolve a stamped legacy wire identity to its equivalent built-in
 * declaration.  Exactly four rows: wire.12.1, wire.12.2, wire.13.1,
 * wire.13.2.  `semantics_contract` (optional, may be NULL) receives
 * the semantics contract the identity pairs with ("semantics.1" for
 * wire.12.x, "semantics.2" for wire.13.x).  Any other id — including
 * bare "wire.12"/"wire.13" — returns false with `error` naming it. */
qboolean QNN_ObsShimDeclForWire(const char *wire_id, qnn_obs_decl_t *out,
	const char **semantics_contract, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* QNN_OBS_SHIM_H */
