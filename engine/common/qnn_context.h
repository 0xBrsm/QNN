#ifndef QNN_CONTEXT_H
#define QNN_CONTEXT_H

#include <stddef.h>

typedef struct
{
	unsigned char *data;
	size_t size;
} qnn_context_t;

void QNN_ContextRegister(void *address, size_t size);
size_t QNN_ContextSize(void);
void QNN_ContextInit(qnn_context_t *context);
void QNN_ContextCapture(qnn_context_t *context);
void QNN_ContextRestore(const qnn_context_t *context);
void QNN_ContextDestroy(qnn_context_t *context);

#endif
