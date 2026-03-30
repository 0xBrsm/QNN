/*
 * qnn_obs.h — Observation buffer layout and packing API.
 *
 * The obs buffer is the single source of truth for what the model sees.
 * Both the training process and demo replay emit through QNN_PackObsBuffer.
 *
 * Wire format: fixed-size little-endian byte buffer (QNN_OBS_BUFFER_SIZE).
 * Python reads it with np.frombuffer. The offset table below is the contract.
 */

#ifndef QNN_OBS_H
#define QNN_OBS_H

#include <stdint.h>

/* ---- Observation dimensions (must match observation.py) ---- */

#define QNN_OBS_SELF_SCALAR_DIM      23
#define QNN_OBS_SELF_ID_DIM           3
#define QNN_OBS_MAX_OBJECTS          64
#define QNN_OBS_OBJECT_ID_DIM         5
#define QNN_OBS_OBJECT_SCALAR_DIM    13
#define QNN_OBS_MAX_ROUTE_CLUSTERS    8
#define QNN_OBS_MAX_EVENTS          256
#define QNN_OBS_EVENT_ID_DIM          4
#define QNN_OBS_EVENT_SCALAR_DIM      3
#define QNN_OBS_SPATIAL_COUNT         9
#define QNN_OBS_SPATIAL_SCALAR_DIM   10
#define QNN_OBS_ACTION_HISTORY_LEN    8
#define QNN_OBS_ACTION_HISTORY_DIM    7

/* ---- Observation buffer layout (15892 bytes, little-endian) ---- */

/*  Offset  Field                        Type       Shape          Bytes */
/*       0  self_scalars                 float32    [23]              92 */
/*      92  self_weapon_id               int32      [1]                4 */
/*      96  self_movement_id             int32      [1]                4 */
/*     100  self_cluster_id              int32      [1]                4 */
/*     104  object_ids                   int32      [64,5]          1280 */
/*    1384  object_scalars               float32    [64,13]         3328 */
/*    4712  object_mask                  uint8      [64]              64 */
/*    4776  object_route_cluster_ids     int32      [64,8]          2048 */
/*    6824  event_ids                    int32      [256,4]         4096 */
/*   10920  event_scalars                float32    [256,3]         3072 */
/*   13992  event_owner                  int32      [256]           1024 */
/*   15016  event_mask                   uint8      [256]            256 */
/*   15272  spatial_ids                  int32      [9]               36 */
/*   15308  spatial_scalars              float32    [9,10]           360 */
/*   15668  action_history               float32    [8,7]            224 */
/*   Total:                                                        15892 */

#define QNN_OBS_BUFFER_SIZE 15892

#define QNN_OBS_OFF_SELF_SCALARS            0
#define QNN_OBS_OFF_SELF_WEAPON_ID         92
#define QNN_OBS_OFF_SELF_MOVEMENT_ID       96
#define QNN_OBS_OFF_SELF_CLUSTER_ID       100
#define QNN_OBS_OFF_OBJECT_IDS            104
#define QNN_OBS_OFF_OBJECT_SCALARS       1384
#define QNN_OBS_OFF_OBJECT_MASK          4712
#define QNN_OBS_OFF_OBJECT_ROUTE_IDS     4776
#define QNN_OBS_OFF_EVENT_IDS            6824
#define QNN_OBS_OFF_EVENT_SCALARS       10920
#define QNN_OBS_OFF_EVENT_OWNER         13992
#define QNN_OBS_OFF_EVENT_MASK          15016
#define QNN_OBS_OFF_SPATIAL_IDS         15272
#define QNN_OBS_OFF_SPATIAL_SCALARS     15308
#define QNN_OBS_OFF_ACTION_HISTORY      15668

/* ---- Action history normalization (must match actions.py) ---- */

#define QNN_ACTION_SWITCH_SLOTS 5

/* ---- API ---- */

/* Requires qnn.h to be included first for qnn_snapshot_t. */

/* Pack a snapshot into a flat obs buffer. Single source of truth for
   observation encoding. Both trainer and demo call this. */
void QNN_PackObsBuffer(uint8_t *buf, const qnn_snapshot_t *snapshot, int tick_hz, qboolean reset_flag);

/* Reset action history ring buffer (call on episode reset). */
void QNN_ObsResetActionHistory(void);

#endif /* QNN_OBS_H */
