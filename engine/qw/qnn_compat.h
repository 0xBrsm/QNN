/*
 * qnn_compat.h (qw) — Compatibility shims mapping NQ client state names
 * to their QW equivalents.
 *
 * Included by qnn.h (via QNN_QW_BUILD define) so that shared worker
 * code compiles against the QW engine headers.
 *
 * NQ -> QW mapping:
 *   cl.viewentity     -> (cl.playernum + 1)
 *   cl.maxclients     -> MAX_CLIENTS
 *   cl.scores         -> cl.players
 *   cl.scores[i].frags -> cl.players[i].frags
 *   cl.scores[i].name -> cl.players[i].name
 *   cl.scores[i].colors -> (cl.players[i].topcolor << 4 | cl.players[i].bottomcolor)
 *   cl.items          -> cl.stats[STAT_ITEMS]
 *   cl.velocity       -> cl.simvel
 *   cl.onground       -> (from playerstate)
 *   cl.inwater        -> 0 (approximated)
 *   cl.mtime[0]       -> cl.time
 *   cl.mtime[1]       -> cl.time (no second timestamp in QW)
 *   cl_entities[i]    -> (static entity array, limited)
 */

#ifndef QNN_QW_COMPAT_H
#define QNN_QW_COMPAT_H

#ifdef QNN_QW_BUILD

/* ── NQ scoreboard_t emulation ────────────────────────────────── */
/* NQ's cl.scores is an array of scoreboard_t { char name[16]; float entertime; int frags; int colors; }
 * QW's cl.players is an array of player_info_t { int userid; char name[16]; float entertime; int frags; ... }
 * We need to provide `colors` from topcolor/bottomcolor. */

/* Provide a `colors` field via macro for the shared code that accesses
 * cl.scores[i].colors.  This is tricky because QW player_info_t has
 * separate topcolor/bottomcolor fields.
 *
 * Actually, for the shared code, we'll add a wrapper macro. */

/* ── Direct equivalences ──────────────────────────────────────── */

/* cl.viewentity -> cl.playernum + 1 (1-indexed entity number) */
#define qnn_viewentity() (cl.playernum + 1)

/* cl.maxclients -> MAX_CLIENTS (always 32 in QW) */
#define qnn_maxclients() MAX_CLIENTS

/* cl.mtime[0] -> cl.time */
#define qnn_server_time() ((float)cl.time)

/* cl.mtime[1] -> cl.time (no prev timestamp in QW client) */
/* For dt computation, the caller should use host_frametime instead */
#define qnn_server_time_prev() ((float)(cl.time - host_frametime))

/* cl.items -> cl.stats[STAT_ITEMS] */
#define qnn_items() (cl.stats[STAT_ITEMS])

/* cl.velocity -> cl.simvel */
#define qnn_velocity() (cl.simvel)

/* ── Compatibility entity_t shim ─────────────────────────────── */
/* QW doesn't have a flat cl_entities[MAX_EDICTS] array for all entities.
 * Static entities are in cl_static_entities[].  Packet entities are
 * in cl.frames[].packet_entities.  For the shared code that accesses
 * cl_entities[i].origin, we provide a limited shim.
 *
 * This is only used for map entity extraction (qnn_map.c) which
 * needs baseline entity positions.  QW stores these in cl_baselines[]. */

/* Note: cl_baselines[] is entity_state_t, not entity_t. The shared code
 * accesses entity_t.origin which is entity_state_t.origin for baselines. */

#endif /* QNN_QW_BUILD */
#endif /* QNN_QW_COMPAT_H */
