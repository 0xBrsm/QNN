#ifndef QNN_ARENA_VIRTUAL_H
#define QNN_ARENA_VIRTUAL_H

#include "qnn.h"

void QNN_ArenaVirtualConfigure(int client_count, qboolean selfplay,
	qboolean shadow, const char *reward_json);
void QNN_ArenaVirtualAttach(client_t *server_client);
void QNN_ArenaVirtualMirrorMessage(client_t *server_client,
	const sizebuf_t *message, int message_type);
qboolean QNN_ArenaVirtualMirrorActive(void);
int QNN_ArenaVirtualGetMessage(void);
void QNN_ArenaVirtualPumpSignon(float dt);
int QNN_ArenaVirtualAssignSeats(void);
qboolean QNN_ArenaVirtualReady(void);
void QNN_ArenaVirtualPrepare(void);
void QNN_ArenaVirtualStageActions(const qnn_action_t *actions, int action_count);
void QNN_ArenaVirtualWriteInitial(FILE *out);
void QNN_ArenaVirtualReceive(FILE *out, float dt, qboolean reset_receive);
void QNN_ArenaVirtualShutdown(void);

#endif
