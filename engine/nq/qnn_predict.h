/* qnn_predict.h — client-side self-state prediction (see qnn_predict.c). */
#ifndef QNN_PREDICT_H
#define QNN_PREDICT_H

#include "qnn.h"

void QNN_PredictInit(void);
void QNN_PredictReset(void);
void QNN_PredictRecordCmd(float fwd, float side, float yaw_deg);
void QNN_PredictTick(float dt);
void QNN_PredictSelfVelocity(vec3_t vel);

#endif
