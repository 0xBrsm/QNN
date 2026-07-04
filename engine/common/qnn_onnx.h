/*
 * qnn_onnx.h — in-process ONNX inference for the live NQ client.
 *
 * Loads an .onnx policy via the ONNX Runtime C API, runs one inference
 * per game tick from the engine's qnn_tick_result_t, decodes the model
 * outputs into the engine's qnn_action_t (sticky-weapon controller +
 * greedy argmax). All schema concepts (qnn_tick_result_t, qnn_action_t,
 * QNN_MAX_TOKEN_OBJECTS, ...) come from the engine headers — there's no
 * parallel obs/action shape.
 *
 * Multiple wire contracts are supported through ONE binary via a
 * per-model obs codec (qnn_obs_codec_t, internal to qnn_onnx.c). A
 * codec is the bin-side implementation of a wire contract (see
 * src/docs/contracts/README.md — "wire vs codec"). The codec is
 * selected at model-load time (QNN_OnnxInit) from the model's
 * `wire_contract` metadata or its ORT input-name set, and every tick
 * dispatches through it. The stable seam is qnn_tick_result_t in /
 * qnn_action_t out; emit, packing, and decode are the codec's job.
 *
 * One context per bot. Not thread-safe; QNN_OnnxLastError is
 * thread-local.
 */
#ifndef QNN_ONNX_H
#define QNN_ONNX_H

#include "qnn.h"          /* qnn_action_t */
#include "qnn_io.h"       /* qnn_tick_result_t */

typedef struct qnn_onnx_ctx qnn_onnx_ctx_t;

/* Create a context bound to an .onnx export. Selects the obs codec for
 * the model's wire contract; returns NULL on error (including "no codec
 * handles this model") — call QNN_OnnxLastError for the reason. */
qnn_onnx_ctx_t *QNN_OnnxInit(const char *onnx_path);

/* Reset the recurrent (GRU) hidden state. Call between episodes. */
void QNN_OnnxReset(qnn_onnx_ctx_t *ctx);

/* One frame of inference. Reads obs from *result*, advances GRU
 * hidden in *ctx* in place, writes the decoded engine action into
 * *out*. Returns 0 on success, non-zero on error. */
int QNN_OnnxStep(qnn_onnx_ctx_t *ctx, const qnn_tick_result_t *result, qnn_action_t *out);

/* Free the context. Safe to pass NULL. */
void QNN_OnnxFree(qnn_onnx_ctx_t *ctx);

/* The model's REQUIRED decision cadence (Hz) from its `tick_hz` stamp — the
 * rate the weights were trained at. Always > 0 for a successfully-loaded model
 * (QNN_OnnxInit refuses an unstamped model). The live client runs inference at
 * this rate. Returns 0 if ctx is NULL. */
int QNN_OnnxTickHz(const qnn_onnx_ctx_t *ctx);

/* Thread-local last error message. "" if no error has been raised
 * on this thread. */
const char *QNN_OnnxLastError(void);

#endif /* QNN_ONNX_H */
