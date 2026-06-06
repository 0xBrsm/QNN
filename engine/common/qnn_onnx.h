/*
 * qnn_onnx.h — in-process ONNX inference for the live NQ client.
 *
 * Loads a v23 .onnx policy via the ONNX Runtime C API, runs one
 * inference per game tick from the engine's qnn_tick_result_t,
 * decodes the model outputs into the engine's qnn_action_t
 * (sticky-weapon controller + greedy argmax). All schema concepts
 * (qnn_tick_result_t, qnn_action_t, QNN_MAX_TOKEN_OBJECTS, ...) come
 * from the engine headers — there's no parallel obs/action shape.
 *
 * One context per bot. Not thread-safe; QNN_OnnxLastError is
 * thread-local.
 */
#ifndef QNN_ONNX_H
#define QNN_ONNX_H

#include "qnn.h"          /* qnn_action_t */
#include "qnn_io.h"       /* qnn_tick_result_t */

typedef struct qnn_onnx_ctx qnn_onnx_ctx_t;

/* Create a context bound to a v23 .onnx export. Returns NULL on error;
 * call QNN_OnnxLastError for the reason. */
qnn_onnx_ctx_t *QNN_OnnxInit(const char *onnx_path);

/* Reset the recurrent (GRU) hidden state. Call between episodes. */
void QNN_OnnxReset(qnn_onnx_ctx_t *ctx);

/* One frame of inference. Reads obs from *result*, advances GRU
 * hidden in *ctx* in place, writes the decoded engine action into
 * *out*. Returns 0 on success, non-zero on error. */
int QNN_OnnxStep(qnn_onnx_ctx_t *ctx, const qnn_tick_result_t *result, qnn_action_t *out);

/* Free the context. Safe to pass NULL. */
void QNN_OnnxFree(qnn_onnx_ctx_t *ctx);

/* Thread-local last error message. "" if no error has been raised
 * on this thread. */
const char *QNN_OnnxLastError(void);

#endif /* QNN_ONNX_H */
