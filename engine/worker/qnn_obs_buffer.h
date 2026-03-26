/*
 * qnn_obs_buffer.h — Fixed-size observation + training buffer layout.
 *
 * This is the wire contract between the C worker and Python's np.frombuffer().
 * Every field offset and size must match the Python ObservationBufferSpec exactly.
 *
 * Total per-step payload: OBS_BUFFER_SIZE + TRAIN_BUFFER_SIZE = 14668 bytes.
 */

#ifndef QNN_OBS_BUFFER_H
#define QNN_OBS_BUFFER_H

#include <stdint.h>

/* ---- Observation buffer dimensions (must match observation.py) ---- */

#define QNN_OBS_SELF_SCALAR_DIM      23
#define QNN_OBS_SELF_ID_DIM           3
#define QNN_OBS_MAX_OBJECTS          64
#define QNN_OBS_OBJECT_ID_DIM         5
#define QNN_OBS_OBJECT_SCALAR_DIM     8
#define QNN_OBS_MAX_ROUTE_CLUSTERS    8
#define QNN_OBS_MAX_EVENTS          256
#define QNN_OBS_EVENT_ID_DIM          4
#define QNN_OBS_EVENT_SCALAR_DIM      3
#define QNN_OBS_SPATIAL_COUNT         9
#define QNN_OBS_SPATIAL_SCALAR_DIM   10
#define QNN_OBS_ACTION_HISTORY_LEN    8
#define QNN_OBS_ACTION_HISTORY_DIM    7

/* ---- Observation buffer layout (14612 bytes, little-endian) ---- */

/*  Offset  Field                        Type       Shape          Bytes */
/*       0  self_scalars                 float32    [23]              92 */
/*      92  self_weapon_id               int32      [1]                4 */
/*      96  self_movement_id             int32      [1]                4 */
/*     100  self_cluster_id              int32      [1]                4 */
/*     104  object_ids                   int32      [64,5]          1280 */
/*    1384  object_scalars               float32    [64,8]          2048 */
/*    3432  object_mask                  uint8      [64]              64 */
/*    3496  object_route_cluster_ids     int32      [64,8]          2048 */
/*    5544  event_ids                    int32      [256,4]         4096 */
/*    9640  event_scalars                float32    [256,3]         3072 */
/*   12712  event_owner                  int32      [256]           1024 */
/*   13736  event_mask                   uint8      [256]            256 */
/*   13992  spatial_ids                  int32      [9]               36 */
/*   14028  spatial_scalars              float32    [9,10]           360 */
/*   14388  action_history               float32    [8,7]            224 */
/*   Total:                                                        14612 */

#define QNN_OBS_BUFFER_SIZE 14612

#define QNN_OBS_OFF_SELF_SCALARS            0
#define QNN_OBS_OFF_SELF_WEAPON_ID         92
#define QNN_OBS_OFF_SELF_MOVEMENT_ID       96
#define QNN_OBS_OFF_SELF_CLUSTER_ID       100
#define QNN_OBS_OFF_OBJECT_IDS            104
#define QNN_OBS_OFF_OBJECT_SCALARS       1384
#define QNN_OBS_OFF_OBJECT_MASK          3432
#define QNN_OBS_OFF_OBJECT_ROUTE_IDS     3496
#define QNN_OBS_OFF_EVENT_IDS            5544
#define QNN_OBS_OFF_EVENT_SCALARS        9640
#define QNN_OBS_OFF_EVENT_OWNER         12712
#define QNN_OBS_OFF_EVENT_MASK          13736
#define QNN_OBS_OFF_SPATIAL_IDS         13992
#define QNN_OBS_OFF_SPATIAL_SCALARS     14028
#define QNN_OBS_OFF_ACTION_HISTORY      14388

/* ---- Training extras flat buffer (56 bytes) ---- */

#define QNN_TRAIN_FLAT_BUFFER_SIZE 56

#define QNN_TRAIN_OFF_TICK                  0
#define QNN_TRAIN_OFF_STEPS                 4
#define QNN_TRAIN_OFF_FLAGS                 8
#define QNN_TRAIN_OFF_FRAG_GAIN            12
#define QNN_TRAIN_OFF_FRAG_LOSS            16
#define QNN_TRAIN_OFF_PLAYER_DIED          20
#define QNN_TRAIN_OFF_HIT_COUNT            21
#define QNN_TRAIN_OFF_SHOTS_FIRED          22
#define QNN_TRAIN_OFF_PAD0                 23
#define QNN_TRAIN_OFF_DAMAGE_DEALT         24
#define QNN_TRAIN_OFF_DAMAGE_TAKEN         28
#define QNN_TRAIN_OFF_HEALTH_BEFORE        32
#define QNN_TRAIN_OFF_HEALTH_AFTER         36
#define QNN_TRAIN_OFF_ARMOR_BEFORE         40
#define QNN_TRAIN_OFF_ARMOR_AFTER          44
#define QNN_TRAIN_OFF_PICKUP_HEALTH        48
#define QNN_TRAIN_OFF_PICKUP_ARMOR         49
#define QNN_TRAIN_OFF_PICKUP_AMMO          50
#define QNN_TRAIN_OFF_PICKUP_WEAPON        51
#define QNN_TRAIN_OFF_EDP_RAW              52

#define QNN_TRAIN_FLAG_RESET_BIT  0x01
#define QNN_TRAIN_FLAG_DONE_BIT   0x02

/* ---- Combined per-step payload ---- */

#define QNN_STEP_PAYLOAD_SIZE (QNN_OBS_BUFFER_SIZE + QNN_TRAIN_FLAT_BUFFER_SIZE)

/* ---- Action history normalization (must match actions.py) ---- */

#define QNN_ACTION_SWITCH_SLOTS 5

#endif /* QNN_OBS_BUFFER_H */
