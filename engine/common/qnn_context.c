#include "qnn_context.h"

#include "quakedef.h"

#include <stdlib.h>
#include <string.h>

#define QNN_CONTEXT_MAX_RANGES 64

typedef struct
{
	void *address;
	size_t size;
} qnn_context_range_t;

static qnn_context_range_t qnn_context_ranges[QNN_CONTEXT_MAX_RANGES];
static int qnn_context_range_count;
static size_t qnn_context_size;

void QNN_ContextRegister(void *address, size_t size)
{
	if (address == NULL || size == 0)
		Sys_Error("QNN_ContextRegister: invalid range");
	if (qnn_context_range_count >= QNN_CONTEXT_MAX_RANGES)
		Sys_Error("QNN_ContextRegister: too many ranges");
	qnn_context_ranges[qnn_context_range_count].address = address;
	qnn_context_ranges[qnn_context_range_count].size = size;
	qnn_context_range_count += 1;
	qnn_context_size += size;
}

size_t QNN_ContextSize(void)
{
	return qnn_context_size;
}

void QNN_ContextInit(qnn_context_t *context)
{
	if (context == NULL || qnn_context_size == 0)
		Sys_Error("QNN_ContextInit: observer state is not registered");
	context->data = (unsigned char *)malloc(qnn_context_size);
	if (context->data == NULL)
		Sys_Error("QNN_ContextInit: out of memory");
	context->size = qnn_context_size;
	QNN_ContextCapture(context);
}

void QNN_ContextCapture(qnn_context_t *context)
{
	unsigned char *out;
	int index;

	if (context == NULL || context->data == NULL
		|| context->size != qnn_context_size)
		Sys_Error("QNN_ContextCapture: invalid context");
	out = context->data;
	for (index = 0; index < qnn_context_range_count; ++index)
	{
		memcpy(out, qnn_context_ranges[index].address,
			qnn_context_ranges[index].size);
		out += qnn_context_ranges[index].size;
	}
}

void QNN_ContextRestore(const qnn_context_t *context)
{
	const unsigned char *in;
	int index;

	if (context == NULL || context->data == NULL
		|| context->size != qnn_context_size)
		Sys_Error("QNN_ContextRestore: invalid context");
	in = context->data;
	for (index = 0; index < qnn_context_range_count; ++index)
	{
		memcpy(qnn_context_ranges[index].address, in,
			qnn_context_ranges[index].size);
		in += qnn_context_ranges[index].size;
	}
}

void QNN_ContextDestroy(qnn_context_t *context)
{
	if (context == NULL)
		return;
	free(context->data);
	context->data = NULL;
	context->size = 0;
}
