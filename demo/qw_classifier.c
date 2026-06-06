/*
 * qw_classifier.c — Standalone Quake demo classifier (QWD + NQ .dem).
 *
 * Mirrors src/demo/classify.py byte-for-byte:
 *   - QWD path: _classify_qw (DEM_* record walk, usercmd byte reads,
 *     match-text byte regex, signon spec-bit, fullserverinfo dict).
 *   - NQ path:  _classify_nq (16-byte msg header walk, full svc_*
 *     opcode dispatch, STAT_HEALTH + clientdata for ammo tracking,
 *     match text from svc_print/stufftext/centerprint/finale/cutscene).
 *
 * Outputs total_frames + label intervals + qwd_usercmd tallies (QWD) + serverinfo dict; Python
 * wrapper applies mode-classification and bc_exclude policy on top.
 *
 * Persistent worker: reads demo paths from stdin (one per line),
 * writes one JSON object per demo to stdout, exits on EOF.
 *
 * Build: src/engine/build/build_qw_classifier.sh
 */

#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "qnn_demo_sounds.h"  /* canonical fire/jump sound name lists */

/* ── DEM_* record types (low 3 bits of QWD type byte) ──────────────── */

#define DEM_CMD       0
#define DEM_READ      1
#define DEM_SET       2
#define DEM_MULTIPLE  3
#define DEM_SINGLE    4
#define DEM_STATS     5
#define DEM_ALL       6
#define DEM_MASK      7

/* QWD per-record header: 4-byte demotime float + 1-byte type byte. */
#define QWD_REC_HEADER 5

/* DEM_CMD payload size (usercmd_t with natural alignment + 3 viewangles).
 * Layout: 1 msec + 3 padding + 12 angles + 6 move shorts + 1 buttons +
 * 1 impulse + ... = 24 byte usercmd, then 12 byte viewangles = 36. */
#define DEM_CMD_PAYLOAD 36

/* svc_serverdata opcode (QW). Full QW svc table is defined further
 * down; this one is needed early by parse_signon. */
#define QW_SVC_SERVERDATA 11

/* Netchan header at the start of each DEM_READ-class message. */
#define QWD_NETCHAN_HEADER 8

/* ── NQ svc_* opcodes (mirrors src/demo/protocol.py) ───────────────── */

#define SVC_NOP              1
#define SVC_DISCONNECT       2
#define SVC_UPDATESTAT       3
#define SVC_VERSION          4
#define SVC_SETVIEW          5
#define SVC_SOUND            6
#define SVC_TIME             7
#define SVC_PRINT            8
#define SVC_STUFFTEXT        9
#define SVC_SETANGLE         10
#define SVC_SERVERINFO       11
#define SVC_LIGHTSTYLE       12
#define SVC_UPDATENAME       13
#define SVC_UPDATEFRAGS      14
#define SVC_CLIENTDATA       15
#define SVC_STOPSOUND        16
#define SVC_UPDATECOLORS     17
#define SVC_PARTICLE         18
#define SVC_DAMAGE           19
#define SVC_SPAWNSTATIC      20
#define SVC_SPAWNBASELINE    22
#define SVC_TEMP_ENTITY      23
#define SVC_SETPAUSE         24
#define SVC_SIGNONNUM        25
#define SVC_CENTERPRINT      26
#define SVC_KILLEDMONSTER    27
#define SVC_FOUNDSECRET      28
#define SVC_SPAWNSTATICSOUND 29
#define SVC_INTERMISSION     30
#define SVC_FINALE           31
#define SVC_CDTRACK          32
#define SVC_SELLSCREEN       33
#define SVC_CUTSCENE         34

/* Entity update bits (U_*). */
#define U_MOREBITS    (1 << 0)
#define U_ORIGIN1     (1 << 1)
#define U_ORIGIN2     (1 << 2)
#define U_ORIGIN3     (1 << 3)
#define U_ANGLE2      (1 << 4)
#define U_FRAME       (1 << 6)
#define U_ANGLE1      (1 << 8)
#define U_ANGLE3      (1 << 9)
#define U_MODEL       (1 << 10)
#define U_COLORMAP    (1 << 11)
#define U_SKIN        (1 << 12)
#define U_EFFECTS     (1 << 13)
#define U_LONGENTITY  (1 << 14)

/* SVC_CLIENTDATA bits — match vendor/quake/WinQuake/protocol.h.  NQ
 * layout differs from QW: SU_ONGROUND is bit 10, not bit 14. */
#define SU_VIEWHEIGHT  (1 << 0)
#define SU_IDEALPITCH  (1 << 1)
#define SU_PUNCH1      (1 << 2)
#define SU_VELOCITY1   (1 << 5)
/* bit 8 is AVAILABLE per vendor (was SU_AIMENT) */
#define SU_ITEMS       (1 << 9)
#define SU_ONGROUND    (1 << 10)
#define SU_INWATER     (1 << 11)
#define SU_WEAPONFRAME (1 << 12)
#define SU_ARMOR       (1 << 13)
#define SU_WEAPON      (1 << 14)

/* NQ stat indices. */
#define STAT_HEALTH 0

/* ── Match-text patterns (literals in lowercase + the one regex) ───── */

static const char *START_LITERALS[] = {
	"match has begun",
	"match started",
	"match has started",
	"game is starting",
};
static const int START_LITERAL_COUNT = 4;

static const char *END_LITERALS[] = {
	"match is over",
	"match over",
	"game over",
	"has won over",
};
static const int END_LITERAL_COUNT = 4;

/* "match is NvN" — hand-rolled scan; POSIX regex can't handle binary
 * (stops at NUL bytes which appear throughout demo files). */
static int find_match_is_num(const char *hay, size_t hay_n)
{
	const char *needle = "match is ";
	size_t needle_n = 9;
	const char *limit = hay + hay_n;
	const char *p = hay;
	while ((size_t)(limit - p) >= needle_n + 3)
	{
		const char *m = memmem(p, (size_t)(limit - p), needle, needle_n);
		if (m == NULL) return -1;
		const char *q = m + needle_n;
		if (q < limit && *q >= '0' && *q <= '9')
		{
			while (q < limit && *q >= '0' && *q <= '9') q++;
			if (q < limit && *q == 'v')
			{
				q++;
				if (q < limit && *q >= '0' && *q <= '9')
					return (int)(m - hay);
			}
		}
		p = m + 1;
	}
	return -1;
}

/* ── Active-input tallies (format-agnostic) ────────────────────────── */

/* Per-frame active-input tallies.  Each counter is the number of
 * frames where that input channel had activity — a frame counts as
 * "active" for a channel if any signal in that frame implies the
 * recorder exercised that input.
 *
 * Provenance differs by format and matters for downstream interpretation:
 *   QWD: literal — read directly from the recorded usercmd_t
 *        (vendor protocol.h:274-281); angles + weaponswitch use a
 *        delta-vs-prev-usercmd test so the stream is treated as a
 *        sequence of transitions rather than absolute state.
 *   MVD/NQ: inferred from server-observed state (origin deltas for
 *        movement, TE/sound events for attack, etc.) — the recorder's
 *        actual key presses are not in the wire format.  Same struct
 *        shape, different signal sources.
 *
 * `none` counts frames where no channel showed activity (idle frames). */
typedef struct {
	int forwardmove;
	int sidemove;
	int upmove;
	int pitch;
	int yaw;
	int roll;
	int attack;
	int jump;
	int use;
	int weaponswitch;
	int none;
} active_input_t;

/* Per-frame accumulator: caller sets a bit when the channel showed
 * activity at any point in the current frame; active_input_commit
 * folds those bits into the totals and resets.  Walker-agnostic —
 * every format walker uses the same accumulator and commit; only the
 * signal-extraction step (where the bits are set) differs per format. */
typedef struct {
	unsigned forwardmove : 1;
	unsigned sidemove : 1;
	unsigned upmove : 1;
	unsigned pitch : 1;
	unsigned yaw : 1;
	unsigned roll : 1;
	unsigned attack : 1;
	unsigned jump : 1;
	unsigned use : 1;
	unsigned weaponswitch : 1;
} active_input_accum_t;

static void active_input_accum_reset(active_input_accum_t *a)
{
	memset(a, 0, sizeof(*a));
}

static void active_input_commit(active_input_t *out, active_input_accum_t *a)
{
	out->forwardmove  += a->forwardmove;
	out->sidemove     += a->sidemove;
	out->upmove       += a->upmove;
	out->pitch        += a->pitch;
	out->yaw          += a->yaw;
	out->roll         += a->roll;
	out->attack       += a->attack;
	out->jump         += a->jump;
	out->use          += a->use;
	out->weaponswitch += a->weaponswitch;
	if (!(a->forwardmove | a->sidemove | a->upmove
	      | a->pitch | a->yaw | a->roll
	      | a->attack | a->jump | a->use
	      | a->weaponswitch))
		out->none += 1;
	active_input_accum_reset(a);
}

/* Per-frame counters for inventory/state deltas — what the player
 * RECEIVED/LOST this frame.  Distinct from active_input (recorder's
 * intent): an ammo_down without an attack signal indicates a spectator
 * triggering attack-bind without firing; a fire-only frame with no
 * ammo_down is impossible for real play.
 *
 * weapon_up counts new bits set in IT_AXE..IT_LIGHTNING (items bits 0-8).
 * special_up counts new bits in IT_SUPERHEALTH, IT_KEY1/2, IT_INVIS,
 * IT_INVULN, IT_SUIT, IT_QUAD, IT_SIGIL1..4 (bits 16-22 + 28-31).
 * Ammo-flag bits (9-12) and armor-flag bits (13-15) are skipped —
 * already covered by ammo_up / armor_up deltas off the scalar counters. */
typedef struct {
	int health_up;
	int health_down;
	int armor_up;
	int armor_down;
	int ammo_up;
	int ammo_down;
	int frag_up;
	int weapon_up;
	int special_up;
} active_state_t;

typedef struct {
	unsigned health_up : 1, health_down : 1;
	unsigned armor_up  : 1, armor_down  : 1;
	unsigned ammo_up   : 1, ammo_down   : 1;
	unsigned frag_up   : 1;
	unsigned weapon_up : 1;
	unsigned special_up : 1;
} active_state_accum_t;

#define ITEMS_WEAPON_MASK  0x000001ff   /* bits 0-8: IT_AXE..IT_EXTRA_WEAPON */
#define ITEMS_SPECIAL_MASK 0xf07f0000U  /* bits 16-22 + 28-31 */

static void active_state_accum_reset(active_state_accum_t *a)
{
	memset(a, 0, sizeof(*a));
}

static void active_state_commit(active_state_t *out, active_state_accum_t *a)
{
	out->health_up   += a->health_up;
	out->health_down += a->health_down;
	out->armor_up    += a->armor_up;
	out->armor_down  += a->armor_down;
	out->ammo_up     += a->ammo_up;
	out->ammo_down   += a->ammo_down;
	out->frag_up     += a->frag_up;
	out->weapon_up   += a->weapon_up;
	out->special_up  += a->special_up;
	active_state_accum_reset(a);
}

typedef struct {
	int total_frames;
	int match_start_frame;  /* -1 if absent — feeds labels.match */
	int match_end_frame;    /* -1 if absent — feeds labels.match */
	/* Frame index where the walker bailed on an unknown / corrupt
	 * record before reaching EOF.  -1 = walked the file cleanly. */
	int error_frame;
	/* Average recorder-slot ping (svc_updateping samples).  ping_sum_ms
	 * accumulates valid samples; ping_count is the divisor.  999 ('unknown'
	 * sentinel) and 0 are excluded. */
	uint64_t ping_sum_ms;
	int ping_count;
	/* Debug-only: per-event ping samples (filled when
	 * QNN_EMIT_PING_HISTORY=1).  ph_n is the count. */
	uint32_t ph_frame[8192];
	uint16_t ph_ping[8192];
	int ph_n;
	/* Per-frame other-player histogram (QWD only).  Bucket k = number of
	 * server frames where exactly k OTHER players (alive, slot != self)
	 * had a svc_playerinfo update.  Counted over every DEM_READ frame.
	 * QW supports up to 32 client slots; bucket 32 catches overflow. */
	uint32_t actors_per_frame[33];
} bounds_t;

/* Global flag set in main() from QNN_EMIT_PING_HISTORY env var.  No
 * runtime cost when off; capture path is gated. */
static int g_emit_ping_history = 0;

/* Debug: emit per-event (demotime, event_type, value) to stderr for
 * weapon-timing analysis.  Toggled by QNN_EMIT_WEAPON_TIMING env. */
static int g_emit_weapon_timing = 0;

/* Debug: emit attack-button rising edges (`FT ... press`), fire sound
 * events (`FT ... sound`), ammo decrements (`FT ... ammo`) and per-ping
 * updates (`FT ... ping`) to stderr for fire-timing analysis.  Toggled
 * by QNN_EMIT_FIRE_TIMING env. */
static int g_emit_fire_timing = 0;

/* ── Per-label interval tracking ─────────────────────────────────────
 *
 * Each label has an independent list of [start, end] frame ranges.
 * Labels can overlap.  pov_N uses N = player slot from svc_setview.
 *
 * Memory layout is fixed-cap rather than dynamic; demos rarely have
 * more than a handful of intervals per label.  Overflow is a hard
 * stop (not a silent drop) so we notice if a real demo blows the cap. */

#define MAX_INTERVALS_PER_LABEL 64
#define MAX_LABELS              48
#define MAX_LABEL_NAME          24

typedef struct {
	int start;
	int end;
} interval_t;

typedef struct {
	char name[MAX_LABEL_NAME];
	int open_start;      /* frame where current open interval began, -1 if none */
	interval_t intervals[MAX_INTERVALS_PER_LABEL];
	int count;
} label_track_t;

typedef struct {
	label_track_t tracks[MAX_LABELS];
	int count;
} labels_t;

static void labels_init(labels_t *L) { L->count = 0; }

static label_track_t *labels_find_or_create(labels_t *L, const char *name)
{
	for (int i = 0; i < L->count; ++i)
		if (strcmp(L->tracks[i].name, name) == 0)
			return &L->tracks[i];
	if (L->count >= MAX_LABELS) return NULL;
	label_track_t *t = &L->tracks[L->count++];
	memset(t, 0, sizeof(*t));
	strncpy(t->name, name, MAX_LABEL_NAME - 1);
	t->open_start = -1;
	return t;
}

static void label_open(labels_t *L, const char *name, int frame)
{
	label_track_t *t = labels_find_or_create(L, name);
	if (t == NULL) return;
	if (t->open_start < 0) t->open_start = frame;
}

static void label_close(labels_t *L, const char *name, int frame)
{
	label_track_t *t = labels_find_or_create(L, name);
	if (t == NULL || t->open_start < 0) return;
	if (t->count >= MAX_INTERVALS_PER_LABEL)
	{
		t->open_start = -1;
		return;
	}
	t->intervals[t->count].start = t->open_start;
	t->intervals[t->count].end = frame;
	t->count += 1;
	t->open_start = -1;
}

/* Close any open intervals at demo end. */
static void labels_close_all(labels_t *L, int final_frame)
{
	for (int i = 0; i < L->count; ++i)
		if (L->tracks[i].open_start >= 0)
			label_close(L, L->tracks[i].name, final_frame);
}

/* ── Shared label state machine ──────────────────────────────────────
 *
 * Adapter pattern: per-format walkers parse their byte stream, extract
 * events (setview, health, pause, intermission, disconnect), and call
 * the hooks below.  The state machine — open/close transitions, slot
 * tracking, the rule that intermission closes on the next match-start
 * text — lives in exactly one place so QWD/NQ/MVD all behave identically.
 *
 * The walker is still responsible for the format-specific parts:
 * frame counting, message boundaries, and which opcodes carry which
 * events (those opcode numbers and payload layouts differ per format). */

/* NQ inference adapter: per-state we need to derive active_input bits from
 * server-observed signals (vendor doesn't expose the player's usercmd in NQ
 * demos).  Sound-based attack/jump detection requires matching sound_num
 * against a small bitset of precached fire/jump sound indices, captured at
 * signon (SVC_SERVERINFO precache list). */
#define NQ_MAX_PRECACHED_SOUNDS 512

typedef struct {
	labels_t *labels;
	int impossible_open;
	int dead_open;
	int paused_open;
	/* `signon`: open from the start of label tracking, closed on the
	 * first HEALTH stat broadcast.  Marks the head-of-demo window
	 * where personal stats haven't landed yet and the worker emits
	 * stale obs scalars.  Filterable via drop_tick_labels=["signon"]. */
	int signon_open;
	/*   qw_protocol:    Wire-format version (26/27/28); captured from
	 *                   svc_serverdata.  Drives delta-usercmd field
	 *                   widths in MSG_ReadDeltaUsercmd (Threewave CTF
	 *                   demos ship at 26 with a different byte layout).
	 *   is_spectator:   signon playernum high bit; drives the
	 *                   top-level `recorder` JSON field. */
	int qw_protocol;
	int is_spectator;
	/* Recorder's player slot for filtering svc_updatefrags (frag_up
	 * detection).  QWD: signon playernum & 0x7f.  NQ: nq_self_entity - 1
	 * (vendor encodes slot = edict_num - 1).  -1 until known. */
	int self_slot;
	/* Prev values for active_state delta detection — shared across
	 * formats since both NQ and QWD expose these (via svc_clientdata
	 * and svc_updatestat respectively).  -1 sentinel = not yet seen. */
	int prev_health;
	int prev_armor;
	int prev_frags;
	int prev_ammo[4];              /* shells, nails, rockets, cells; -1 until first read */
	int32_t prev_items;
	int have_prev_items;
	/* NQ inference state.  All NQ-only; QWD/MVD walkers leave these
	 * at defaults (-1 / 0). */
	int nq_self_entity;            /* recorder's edict#, from svc_setview */
	int nq_prev_active_weapon;     /* -1 until first SVC_CLIENTDATA */
	/* Recorder's local view angles — from the NQ demo message header
	 * (vendor cl_demo.c:74-78 writes cl.viewangles per CL_WriteDemoMessage).
	 * Used for look-delta detection and to decompose self-entity origin
	 * deltas into the player's facing frame for forwardmove/sidemove. */
	float nq_view_pitch;
	float nq_view_yaw;
	float nq_view_roll;
	int nq_have_view_angles;
	/* Recorder's own origin from the most recent self-entity update —
	 * vendor sv_main.c:451 always sends the receiving client's own entity,
	 * so origin reaches us even when svc_clientdata omits velocity bits. */
	float nq_prev_origin[3];
	int nq_have_prev_origin;
	/* Per-sound-num classification: 1 = weapon fire, 2 = jump.
	 * Other bytes left zero.  Indexed by sound_num as emitted in
	 * SVC_SOUND.  Populated during SVC_SERVERINFO precache walk. */
	uint8_t nq_sound_kind[NQ_MAX_PRECACHED_SOUNDS];
	/* Running sum/count of recorder-slot ping samples; copied into
	 * bounds_t at end of walk_qwd.  Initial 999 sentinel and 0 are
	 * excluded from the average. */
	uint64_t ping_sum_ms;
	int ping_count;
	/* Optional debug capture of per-event ping samples for the
	 * recorder slot.  Only populated when emit_ping_history != 0.
	 * Triggered by QNN_EMIT_PING_HISTORY env var in main(). */
	int emit_ping_history;
	uint32_t ph_frame[8192];
	uint16_t ph_ping[8192];
	int ph_n;
	/* Current demotime, set by walker before dispatching each record.
	 * Used by the QNN_EMIT_WEAPON_TIMING debug path to attach a
	 * wall-clock timestamp to each impulse / stat event. */
	float current_demotime;
	const char *current_demo_path;
	/* Per-DEM_READ-frame player accounting (QWD only).  qw_parse_playerinfo
	 * sets the slot bit on each svc_playerinfo; PF_DEAD/PF_GIB also sets
	 * the dead bit.  walker resets both after each DEM_READ frame after
	 * folding popcount(seen & ~dead & ~self) into bounds.actors_per_frame. */
	uint32_t frame_player_seen;
	uint32_t frame_player_dead;
} label_state_t;

static void lstate_init(label_state_t *s, labels_t *L)
{
	s->labels = L;
	s->impossible_open = 0;
	s->dead_open = 0;
	s->paused_open = 0;
	/* Signon opens immediately at frame 0 and closes when the first
	 * HEALTH stat update lands.  Opened here so it's already active
	 * by the time the walker emits its first frame. */
	label_open(L, "signon", 0);
	s->signon_open = 1;
	s->qw_protocol = 28;  /* assume modern until serverdata says otherwise */
	s->is_spectator = 0;
	s->self_slot = -1;
	s->prev_health = -1;
	s->prev_armor = -1;
	s->prev_frags = -1;
	s->prev_items = 0;
	s->have_prev_items = 0;
	s->nq_self_entity = -1;
	s->nq_prev_active_weapon = -1;
	for (int i = 0; i < 4; ++i) s->prev_ammo[i] = -1;
	s->nq_view_pitch = 0.0f;
	s->nq_view_yaw = 0.0f;
	s->nq_view_roll = 0.0f;
	s->nq_have_view_angles = 0;
	s->nq_prev_origin[0] = s->nq_prev_origin[1] = s->nq_prev_origin[2] = 0.0f;
	s->nq_have_prev_origin = 0;
	memset(s->nq_sound_kind, 0, sizeof(s->nq_sound_kind));
	s->ping_sum_ms = 0;
	s->ping_count = 0;
	s->emit_ping_history = g_emit_ping_history;
	s->ph_n = 0;
	s->current_demotime = 0.0f;
	s->current_demo_path = NULL;
	s->frame_player_seen = 0;
	s->frame_player_dead = 0;
}

static void lstate_health(label_state_t *s, int health, int frame)
{
	/* First HEALTH broadcast closes the signon interval.  After this
	 * point the worker's personal-stat obs scalars carry real values
	 * instead of the C-side init defaults. */
	if (s->signon_open)
	{
		label_close(s->labels, "signon", frame);
		s->signon_open = 0;
	}
	if (health > 250 && !s->impossible_open)
	{
		label_open(s->labels, "impossible_health", frame);
		s->impossible_open = 1;
	}
	else if (health <= 250 && s->impossible_open)
	{
		label_close(s->labels, "impossible_health", frame);
		s->impossible_open = 0;
	}
	if (health <= 0 && !s->dead_open)
	{
		label_open(s->labels, "dead", frame);
		s->dead_open = 1;
	}
	else if (health > 0 && s->dead_open)
	{
		label_close(s->labels, "dead", frame);
		s->dead_open = 0;
	}
}

static void lstate_setpause(label_state_t *s, int paused, int frame)
{
	if (paused && !s->paused_open)
	{
		label_open(s->labels, "paused", frame);
		s->paused_open = 1;
	}
	else if (!paused && s->paused_open)
	{
		label_close(s->labels, "paused", frame);
		s->paused_open = 0;
	}
}

static void lstate_intermission(label_state_t *s, int frame)
{
	/* No end opcode; closed at next match_start text or demo end. */
	label_open(s->labels, "intermission", frame);
}

/* Called when a fresh match-start text is detected (NOT for repeated
 * detections at the same frame).  Closes intermission if open. */
static void lstate_match_started(label_state_t *s, int frame)
{
	label_close(s->labels, "intermission", frame);
}

static void lstate_close_all(label_state_t *s, int final_frame)
{
	labels_close_all(s->labels, final_frame);
}

/* ── Whole-file match-text scan ────────────────────────────────────── */

/* Strip high bit + lowercase (in-place into out_buf, must be size >= n). */
static void strip_highbit_lower(const uint8_t *data, size_t n, char *out_buf)
{
	size_t i;
	for (i = 0; i < n; ++i)
		out_buf[i] = (char)tolower(data[i] & 0x7f);
}

static int memmem_int_offset(const char *hay, size_t hay_n,
                              const char *needle, size_t needle_n)
{
	const char *p = memmem(hay, hay_n, needle, needle_n);
	if (p == NULL)
		return -1;
	return (int)(p - hay);
}

static int find_earliest(const char *hay, size_t hay_n,
                          const char **needles, int needle_count)
{
	int best = -1;
	int i;
	for (i = 0; i < needle_count; ++i)
	{
		int off = memmem_int_offset(hay, hay_n, needles[i], strlen(needles[i]));
		if (off >= 0 && (best < 0 || off < best))
			best = off;
	}
	return best;
}

static void find_match_positions(const char *lower, size_t n,
                                  int *start_out, int *end_out)
{
	int start = find_earliest(lower, n, START_LITERALS, START_LITERAL_COUNT);
	int num_off = find_match_is_num(lower, n);
	if (num_off >= 0 && (start < 0 || num_off < start))
		start = num_off;
	int end = find_earliest(lower, n, END_LITERALS, END_LITERAL_COUNT);
	*start_out = start;
	*end_out = end;
}

/* Signon (playernum / is_spectator) is extracted by the dispatcher when
 * it encounters svc_serverdata in the message stream.  Earlier code had
 * a separate parse_signon that walked the first DEM_READ records — but
 * that approach fails when the signon message has svc_print before
 * svc_serverdata (common in real QWDs).  Dispatching is the canonical
 * walk and gets it right by construction. */

/* ── Binary reader (forward-hoisted for the walkers) ──────────────────
 *
 * Full rd_* helpers + skip_* / scan_* live further down; walk_qwd and
 * walk_mvd only need to construct a reader and pass it to the QW svc
 * dispatcher, which has access to the full reader API. */

typedef struct {
	const uint8_t *data;
	size_t pos;
	size_t end;     /* exclusive upper bound for current message */
	int overflowed;
} reader_t;

static void rd_init(reader_t *r, const uint8_t *data, size_t pos, size_t end)
{
	r->data = data;
	r->pos = pos;
	r->end = end;
	r->overflowed = 0;
}

/* Forward decl: QW svc dispatch lives further down to keep QW protocol
 * constants and skip helpers grouped.  walk_qwd / walk_mvd call it on
 * each message body. */
static int dispatch_qw_message(reader_t *r, label_state_t *ls,
                                int frame, active_state_accum_t *acc_state,
                                int *match_start_frame, int *match_end_frame,
                                const uint8_t *base);

/* ── QWD adapter: DEM_* record walk + QW dispatch per DEM_READ msg ── */

static void walk_qwd(const uint8_t *data, size_t n,
                      int match_start_off, int match_end_off,
                      bounds_t *bounds, active_input_t *ai,
                      active_state_t *as,
                      labels_t *labels, int *out_is_spectator)
{
	size_t pos = 0;
	int frame = 0;
	int match_start_frame = -1;
	int match_end_frame = -1;
	int error_frame = -1;

	/* Self-entity and spec-track flags are populated by the dispatcher
	 * when it parses svc_serverdata — typically in the first big
	 * DEM_READ message.  Initial -1/0 disables POV switching until then. */
	label_state_t ls;
	lstate_init(&ls, labels);

	/* Per-frame input accumulator.  Multiple DEM_CMDs can share a frame
	 * (DEM_READ-paced, ~12-15 Hz vs usercmd ~77 Hz); a channel's bit
	 * sets if any usercmd in the frame showed activity.  Format-specific
	 * delta state (prev angles, prev impulse) is local to the QWD walker
	 * since MVD/NQ infer the same channels from different signals. */
	active_input_accum_t acc;
	active_input_accum_reset(&acc);
	active_state_accum_t acc_state;
	active_state_accum_reset(&acc_state);
	float prev_pitch = 0.0f, prev_yaw = 0.0f, prev_roll = 0.0f;
	int have_prev_angles = 0;
	/* impulse is sticky in many client demos — treat a value as
	 * "user pressed a weapon key" only when it changes vs the previous
	 * usercmd (matches src/demo/analyze.py's impulse_changes count). */
	int prev_impulse = -1;
	int prev_attack = 0;
	/* True when ≥1 DEM_CMD has arrived since the last DEM_READ flushed
	 * the accumulator.  Drives the end-of-walk flush: tiny end-of-match
	 * snippets sometimes have a final shot arriving in DEM_CMDs after
	 * the last DEM_READ — without flushing here their bits silently drop. */
	int dem_cmd_pending = 0;

	while (pos + QWD_REC_HEADER <= n)
	{
		float demotime_f;
		memcpy(&demotime_f, data + pos, 4);
		ls.current_demotime = demotime_f;
		uint8_t type_byte = data[pos + 4];
		pos += QWD_REC_HEADER;
		int dem_type = type_byte & DEM_MASK;

		if (dem_type == DEM_CMD)
		{
			if (pos + DEM_CMD_PAYLOAD <= n)
			{
				/* usercmd_t (vendor protocol.h:274): msec(1) + pad(3)
				 * + angles[3](12) + fwd(2) + side(2) + up(2)
				 * + buttons(1) + impulse(1) = 24 bytes; followed by
				 * 12 bytes of viewangles → DEM_CMD_PAYLOAD = 36. */
				float pitch, yaw, roll;
				memcpy(&pitch, data + pos + 4, 4);
				memcpy(&yaw,   data + pos + 8, 4);
				memcpy(&roll,  data + pos + 12, 4);
				int16_t fwd, side, up;
				memcpy(&fwd, data + pos + 16, 2);
				memcpy(&side, data + pos + 18, 2);
				memcpy(&up, data + pos + 20, 2);
				uint8_t buttons = data[pos + 22];
				uint8_t impulse = data[pos + 23];

				if (fwd != 0)  acc.forwardmove = 1;
				if (side != 0) acc.sidemove    = 1;
				if (up != 0)   acc.upmove      = 1;
				if (have_prev_angles)
				{
					if (pitch != prev_pitch) acc.pitch = 1;
					if (yaw   != prev_yaw)   acc.yaw   = 1;
					if (roll  != prev_roll)  acc.roll  = 1;
				}
				prev_pitch = pitch;
				prev_yaw   = yaw;
				prev_roll  = roll;
				have_prev_angles = 1;
				if (buttons & 0x01) acc.attack = 1;
				if (buttons & 0x02) acc.jump   = 1;
				if (buttons & 0x04) acc.use    = 1;
				/* Attack rising edge for fire-timing analysis. */
				int attack_now = (buttons & 0x01) ? 1 : 0;
				if (g_emit_fire_timing && !prev_attack && attack_now)
					fprintf(stderr, "FT %.3f press frame %d\n",
						ls.current_demotime, frame);
				prev_attack = attack_now;
				/* Weapon-select impulses per vendor cl_input.c: 1-8 =
				 * direct select, 10 = nextweapon, 12 = prevweapon.
				 * Only count on transition (impulse is sticky in many
				 * client demos). */
				int is_weap_imp = (impulse >= 1 && impulse <= 8)
				                  || impulse == 10 || impulse == 12;
				int is_edge = prev_impulse >= 0 && impulse != prev_impulse;
				if (is_edge && is_weap_imp)
					acc.weaponswitch = 1;
				if (g_emit_weapon_timing && is_edge && is_weap_imp)
					fprintf(stderr, "WT %.3f impulse %d frame %d\n",
					        ls.current_demotime, (int)impulse, frame);
				prev_impulse = impulse;
				dem_cmd_pending = 1;
			}
			pos += DEM_CMD_PAYLOAD;
			continue;
		}

		if (dem_type == DEM_SET)
		{
			pos += 8;
			continue;
		}

		if (dem_type == DEM_READ)
		{
			if (pos + 4 > n) { error_frame = frame; break; }
			int32_t msg_len;
			memcpy(&msg_len, data + pos, 4);
			pos += 4;
			if (msg_len < 0 || msg_len > 65536) { error_frame = frame; break; }
			size_t msg_start = pos;
			size_t msg_end = pos + (size_t)msg_len;
			if (msg_end > n) { error_frame = frame; break; }  /* truncated payload */
			pos = msg_end;

			int prev_msf = match_start_frame;
			if (match_start_frame < 0 && match_start_off >= 0
				&& (size_t)match_start_off >= msg_start
				&& (size_t)match_start_off < msg_end)
				match_start_frame = frame;
			if (match_end_frame < 0 && match_end_off >= 0
				&& (size_t)match_end_off >= msg_start
				&& (size_t)match_end_off < msg_end)
				match_end_frame = frame;

			/* Dispatch QW svc opcodes for per-frame labels.  Skip the
			 * 8-byte netchan header at the start of every DEM_READ
			 * message.  Match-text detection stays with the whole-file
			 * scan (preserves byte-for-byte parity with prior baseline);
			 * pass NULL pointers so dispatch only runs label tracking. */
			if ((size_t)msg_len > QWD_NETCHAN_HEADER)
			{
				reader_t r;
				rd_init(&r, data, msg_start + QWD_NETCHAN_HEADER, msg_end);
				int dummy_ms = -1, dummy_me = -1;
				(void)dispatch_qw_message(&r, &ls, frame, &acc_state,
					&dummy_ms, &dummy_me, data);
			}

			/* If this is the frame match-start first appears, close
			 * any open intermission (consistent with NQ behavior). */
			if (match_start_frame == frame && prev_msf < 0)
				lstate_match_started(&ls, frame);

			active_input_commit(ai, &acc);
			active_state_commit(as, &acc_state);
			dem_cmd_pending = 0;

			/* Fold the frame's playerinfo mask into the histogram:
			 * count OTHER players that were alive this frame (seen &
			 * ~dead, excluding self_slot).  Cap at bucket 32 — QW
			 * supports up to 32 client slots so this only fires on
			 * a malformed demo. */
			{
				uint32_t alive = ls.frame_player_seen & ~ls.frame_player_dead;
				if (ls.self_slot >= 0 && ls.self_slot < 32)
					alive &= ~(1u << ls.self_slot);
				int n_others = __builtin_popcount(alive);
				if (n_others > 32) n_others = 32;
				bounds->actors_per_frame[n_others] += 1;
				ls.frame_player_seen = 0;
				ls.frame_player_dead = 0;
			}

			frame += 1;
			continue;
		}

		/* Per vendor cl_demo.c:194-247, real QWD files only contain
		 * DEM_CMD/READ/SET records.  Anything else (DEM_MULTIPLE/
		 * SINGLE/STATS/ALL is MVD; type 7 is reserved) means the file
		 * is mislabeled or corrupt — bail to avoid misaligned reads. */
		error_frame = frame;
		break;
	}

	/* Flush any per-frame accumulator bits left from DEM_CMDs that
	 * trailed the last DEM_READ (frame boundary never advanced past
	 * them).  Counts that activity as one synthetic trailing frame so
	 * `attack` / `forwardmove` / etc. don't silently drop trailing
	 * input — seen in tiny end-of-match snippets where the recorder's
	 * final shot landed after the server stopped sending updates. */
	if (dem_cmd_pending)
	{
		active_input_commit(ai, &acc);
		active_state_commit(as, &acc_state);
		/* Synthetic trailing frame: no DEM_READ landed, so the
		 * playerinfo mask has no fresh data — count as 0 others. */
		bounds->actors_per_frame[0] += 1;
		frame += 1;
	}

	bounds->total_frames = frame;
	bounds->match_start_frame = match_start_frame;
	bounds->match_end_frame = match_end_frame;
	bounds->error_frame = error_frame;
	bounds->ping_sum_ms = ls.ping_sum_ms;
	bounds->ping_count = ls.ping_count;
	if (ls.emit_ping_history) {
		bounds->ph_n = ls.ph_n;
		memcpy(bounds->ph_frame, ls.ph_frame,
		       sizeof(uint32_t) * (size_t)ls.ph_n);
		memcpy(bounds->ph_ping, ls.ph_ping,
		       sizeof(uint16_t) * (size_t)ls.ph_n);
	} else {
		bounds->ph_n = 0;
	}

	int final_frame = frame > 0 ? frame - 1 : 0;
	lstate_close_all(&ls, final_frame);

	*out_is_spectator = ls.is_spectator;
}

/* ── fullserverinfo extraction (first 64 KB) ───────────────────────── */

static void emit_json_string(FILE *out, const char *s);

/* Walks a fullserverinfo string of the form "\key1\val1\key2\val2..."
 * and writes a JSON object. Lowercases keys, strips leading '*' (like
 * Python: `k.lower().strip("*")`). Keeps values verbatim. */
static void emit_serverinfo_json(FILE *out, const char *info, size_t info_n)
{
	fputc('{', out);
	int first = 1;
	size_t i = 0;
	while (i < info_n)
	{
		while (i < info_n && info[i] != '\\') i++;
		if (i >= info_n) break;
		i++;  /* skip leading backslash */
		size_t key_start = i;
		while (i < info_n && info[i] != '\\') i++;
		size_t key_end = i;
		if (i >= info_n) break;
		i++;  /* skip mid backslash */
		size_t val_start = i;
		while (i < info_n && info[i] != '\\' && info[i] != '"' && info[i] != 0) i++;
		size_t val_end = i;

		/* lowercase + strip-leading-* into a stack buffer */
		char keybuf[128];
		size_t klen = key_end - key_start;
		if (klen >= sizeof(keybuf)) klen = sizeof(keybuf) - 1;
		size_t kk = 0, kj = 0;
		/* strip leading * */
		while (kk < klen && info[key_start + kk] == '*') kk++;
		for (; kk < klen; ++kk)
			keybuf[kj++] = (char)tolower((unsigned char)info[key_start + kk]);
		keybuf[kj] = 0;
		if (kj == 0) continue;

		size_t vlen = val_end - val_start;
		char valbuf[1024];
		if (vlen >= sizeof(valbuf)) vlen = sizeof(valbuf) - 1;
		memcpy(valbuf, info + val_start, vlen);
		valbuf[vlen] = 0;

		if (!first) fputc(',', out);
		first = 0;
		emit_json_string(out, keybuf);
		fputc(':', out);
		emit_json_string(out, valbuf);
	}
	fputc('}', out);
}

static void parse_fullserverinfo(const uint8_t *data, size_t n,
                                  FILE *out)
{
	size_t scan_n = n < 65536 ? n : 65536;
	const char *needle = "fullserverinfo";
	size_t needle_n = 14;
	const char *p = memmem(data, scan_n, needle, needle_n);
	if (p == NULL)
	{
		fputs("{}", out);
		return;
	}
	const char *limit = (const char *)data + scan_n;
	const char *q = p + needle_n;
	while (q < limit && (*q == ' ' || *q == '\t')) q++;
	if (q < limit && *q == '"') q++;
	if (q >= limit || *q != '\\')
	{
		fputs("{}", out);
		return;
	}
	const char *end = q;
	while (end < limit && *end != 0 && *end != '"') end++;
	emit_serverinfo_json(out, q, (size_t)(end - q));
}

/* ── Binary reader: rd_* helpers (struct + rd_init are forward-hoisted) ── */

static int rd_overflow(reader_t *r, size_t n)
{
	if (r->overflowed) return 1;
	if (r->pos + n > r->end) { r->overflowed = 1; return 1; }
	return 0;
}

static uint8_t rd_byte(reader_t *r)
{
	if (rd_overflow(r, 1)) return 0;
	return r->data[r->pos++];
}

static int8_t rd_char(reader_t *r)
{
	if (rd_overflow(r, 1)) return 0;
	return (int8_t)r->data[r->pos++];
}

static int16_t rd_short(reader_t *r)
{
	if (rd_overflow(r, 2)) return 0;
	int16_t v;
	memcpy(&v, r->data + r->pos, 2);
	r->pos += 2;
	return v;
}

static uint16_t rd_ushort(reader_t *r)
{
	if (rd_overflow(r, 2)) return 0;
	uint16_t v;
	memcpy(&v, r->data + r->pos, 2);
	r->pos += 2;
	return v;
}

static int32_t rd_long(reader_t *r)
{
	if (rd_overflow(r, 4)) return 0;
	int32_t v;
	memcpy(&v, r->data + r->pos, 4);
	r->pos += 4;
	return v;
}

static float rd_float(reader_t *r)
{
	if (rd_overflow(r, 4)) return 0.0f;
	float v;
	memcpy(&v, r->data + r->pos, 4);
	r->pos += 4;
	return v;
}

/* coord = short / 8.0f, angle = char * (360/256) — but we discard the
 * value, so just consume the bytes. */
static void rd_coord(reader_t *r) { (void)rd_short(r); }
/* NQ coords are MSG_ReadShort * 0.125 (vendor common.c). */
static float rd_coord_val(reader_t *r) { return (float)rd_short(r) * 0.125f; }
static void rd_angle(reader_t *r) { (void)rd_char(r); }

/* Strings are NUL-or-0xFF terminated.  Returns offset of string start
 * and length (excluding terminator); caller copies into a stack buffer. */
static void rd_string(reader_t *r, size_t *start_out, size_t *len_out)
{
	*start_out = r->pos;
	while (r->pos < r->end)
	{
		uint8_t b = r->data[r->pos++];
		if (b == 0 || b == 255)
		{
			*len_out = r->pos - 1 - *start_out;
			return;
		}
	}
	*len_out = r->pos - *start_out;
	r->overflowed = 1;
}

static void rd_skip_string(reader_t *r)
{
	size_t s, n;
	rd_string(r, &s, &n);
	(void)s; (void)n;
}

/* Returns 1 if the string was non-empty. */
static int rd_string_nonempty(reader_t *r)
{
	size_t before = r->pos;
	rd_skip_string(r);
	return (r->pos > before + 1);
}

/* ── NQ skip helpers (port of protocol.py) ─────────────────────────── */

static void skip_particle(reader_t *r)
{
	rd_coord(r); rd_coord(r); rd_coord(r);
	rd_char(r); rd_char(r); rd_char(r);
	rd_byte(r); rd_byte(r);
}

static void skip_damage(reader_t *r)
{
	rd_byte(r); rd_byte(r);
	rd_coord(r); rd_coord(r); rd_coord(r);
}

static void skip_baseline(reader_t *r)
{
	rd_byte(r); rd_byte(r); rd_byte(r); rd_byte(r);
	for (int i = 0; i < 3; ++i) { rd_coord(r); rd_angle(r); }
}

static void skip_temp_entity(reader_t *r)
{
	uint8_t t = rd_byte(r);
	/* TE_SIMPLE_COORD = {0,1,2,3,4,7,8,10,11} */
	if (t == 0 || t == 1 || t == 2 || t == 3 || t == 4
		|| t == 7 || t == 8 || t == 10 || t == 11)
	{
		rd_coord(r); rd_coord(r); rd_coord(r);
	}
	/* TE_BEAM = {5,6,9,13} */
	else if (t == 5 || t == 6 || t == 9 || t == 13)
	{
		rd_short(r);
		for (int i = 0; i < 6; ++i) rd_coord(r);
	}
	else if (t == 12)
	{
		rd_coord(r); rd_coord(r); rd_coord(r);
		rd_byte(r); rd_byte(r);
	}
	else
	{
		r->overflowed = 1; /* unknown TE — bail */
	}
}

/* SVC_CLIENTDATA: walks the variable-length payload, returning fields the
 * label/inference code uses.  Velocity / currentammo / onground are also
 * in the wire format but unused — origin-delta is the movement signal
 * (see nq_track_entity_update) and per-pool ammo decrement the fire
 * fallback.  `armor_out` is -1 when SU_ARMOR bit is unset (caller carries
 * forward prior value); `items_out` is always read (vendor cl_parse.c:541). */
static int read_clientdata(reader_t *r, int *health, int *armor_out,
                            int *shells, int *nails, int *rockets, int *cells,
                            int *active_weapon, int32_t *items_out)
{
	uint16_t bits = rd_ushort(r);
	if (bits & SU_VIEWHEIGHT) rd_char(r);
	if (bits & SU_IDEALPITCH) rd_char(r);
	for (int axis = 0; axis < 3; ++axis)
	{
		if (bits & (SU_PUNCH1 << axis))    rd_char(r);
		if (bits & (SU_VELOCITY1 << axis)) rd_char(r);
	}
	*items_out = rd_long(r);
	if (bits & SU_WEAPONFRAME) rd_byte(r);
	*armor_out = (bits & SU_ARMOR) ? (int)rd_byte(r) : -1;
	if (bits & SU_WEAPON)      rd_byte(r);
	*health  = rd_short(r);
	rd_byte(r);  /* currentammo — per-pool counters below are the useful signal */
	*shells  = rd_byte(r);
	*nails   = rd_byte(r);
	*rockets = rd_byte(r);
	*cells   = rd_byte(r);
	*active_weapon = rd_byte(r);
	return r->overflowed ? 0 : 1;
}

/* Classify a precached NQ sound path as weapon-fire (1), jump (2), or
 * neither (0).  Canonical lists live in qnn_demo_sounds.h, shared with
 * the engine's qnn_event.c sound-rule tables — single source of truth. */
static uint8_t nq_classify_sound_name(const char *s, size_t len)
{
#define X(path, subject) \
	if (len == sizeof(path) - 1 && memcmp(s, path, len) == 0) return 1;
	QNN_FIRE_SOUND_LIST(X)
#undef X
#define X(path) \
	if (len == sizeof(path) - 1 && memcmp(s, path, len) == 0) return 2;
	QNN_JUMP_SOUND_LIST(X)
#undef X
	return 0;
}

/* Walk an NQ entity-update payload, with optional self-origin tracking.
 *
 * Vendor sv_main.c:451 marks the receiving client's own entity as ALWAYS
 * sent — origin reaches the recorder reliably even when svc_clientdata
 * omits the SU_VELOCITY bits, so origin-delta is the movement signal. */
#define NQ_MOVEMENT_THRESH_UNITS 4.0f  /* per-message world-unit delta; clears gravity / friction jitter */

static void nq_track_entity_update(reader_t *r, int bits, label_state_t *ls,
                                    active_input_accum_t *acc)
{
	if (bits & U_MOREBITS) bits |= rd_byte(r) << 8;
	int num = (bits & U_LONGENTITY) ? rd_ushort(r) : rd_byte(r);
	if (bits & U_MODEL)    rd_byte(r);
	if (bits & U_FRAME)    rd_byte(r);
	if (bits & U_COLORMAP) rd_byte(r);
	if (bits & U_SKIN)     rd_byte(r);
	if (bits & U_EFFECTS)  rd_byte(r);

	int is_self = (num == ls->nq_self_entity);
	float origin[3] = {0, 0, 0};
	int have_origin_bit[3] = {0, 0, 0};

	/* Origin/angle pairs interleave on the wire (vendor cl_parse.c:446-471).
	 * On non-self entities we only need to skip the bytes; rd_coord_val()
	 * is identical to rd_coord() but returns the world value. */
	const uint32_t origin_masks[3] = {U_ORIGIN1, U_ORIGIN2, U_ORIGIN3};
	const uint32_t angle_masks[3]  = {U_ANGLE1,  U_ANGLE2,  U_ANGLE3};
	for (int i = 0; i < 3; ++i)
	{
		if (bits & origin_masks[i])
		{
			if (is_self) origin[i] = rd_coord_val(r);
			else         rd_coord(r);
			have_origin_bit[i] = 1;
		}
		if (bits & angle_masks[i]) rd_byte(r);
	}

	if (!is_self || acc == NULL) return;

	/* Entity update is delta-style: omitted axis carries forward from the
	 * last seen value (or baseline on first read, which we treat as 0). */
	for (int i = 0; i < 3; ++i)
		if (!have_origin_bit[i])
			origin[i] = ls->nq_prev_origin[i];

	if (ls->nq_have_prev_origin && ls->nq_have_view_angles)
	{
		float dx = origin[0] - ls->nq_prev_origin[0];
		float dy = origin[1] - ls->nq_prev_origin[1];
		float dz = origin[2] - ls->nq_prev_origin[2];
		float yaw_rad = ls->nq_view_yaw * (3.14159265358979f / 180.0f);
		float cy = (float)cos(yaw_rad), sy = (float)sin(yaw_rad);
		float fwd_d  = dx * cy + dy * sy;
		float side_d = dx * sy - dy * cy;
		const float T = NQ_MOVEMENT_THRESH_UNITS;
		if (fwd_d  >  T || fwd_d  < -T) acc->forwardmove = 1;
		if (side_d >  T || side_d < -T) acc->sidemove    = 1;
		if (dz     >  T)                acc->upmove      = 1;
	}
	memcpy(ls->nq_prev_origin, origin, sizeof(origin));
	ls->nq_have_prev_origin = 1;
}

/* Parse NQ SVC_SERVERINFO payload and populate ls->nq_sound_kind from
 * the precache list.  Sound indices start at 1 in vendor's
 * cl.sound_precache (slot 0 is null), so the loop tracks that. */
static void nq_parse_serverinfo(reader_t *r, label_state_t *ls)
{
	rd_long(r);   /* protocol */
	rd_byte(r);   /* maxclients */
	rd_byte(r);   /* gametype */
	rd_skip_string(r);  /* level name */
	/* Model precache list — skip names, count not needed. */
	while (rd_string_nonempty(r) && !r->overflowed) {}
	/* Sound precache list — record each name's slot (1-indexed). */
	int sound_idx = 1;
	while (!r->overflowed && sound_idx < NQ_MAX_PRECACHED_SOUNDS)
	{
		size_t s, slen;
		rd_string(r, &s, &slen);
		if (slen == 0) break;  /* empty terminator */
		ls->nq_sound_kind[sound_idx] =
			nq_classify_sound_name((const char *)(r->data + s), slen);
		sound_idx += 1;
	}
}

/* Scan a string payload's bytes for match-start/end text.
 *
 * Important parity note: the Python NQ path passes the raw latin-1
 * decoded string to a re.IGNORECASE regex, WITHOUT stripping the
 * high-bit Quake color codes.  So a colored "GAME OVER" (each char
 * | 0x80) does NOT match.  We mirror that: only lowercase ASCII
 * letters, leave bytes >= 0x80 alone so they don't match the
 * lowercase ASCII needles. */
static void scan_text_for_match(const uint8_t *src, size_t src_n,
                                  int frame, int *start_frame, int *end_frame)
{
	if (src_n == 0) return;
	if (src_n > 4096) src_n = 4096;
	char buf[4096];
	for (size_t i = 0; i < src_n; ++i)
	{
		uint8_t c = src[i];
		buf[i] = (c < 0x80) ? (char)tolower(c) : (char)c;
	}
	if (*start_frame < 0)
	{
		int s = find_earliest(buf, src_n, START_LITERALS, START_LITERAL_COUNT);
		if (s < 0) s = find_match_is_num(buf, src_n);
		if (s >= 0) *start_frame = frame;
	}
	if (*end_frame < 0)
	{
		int e = find_earliest(buf, src_n, END_LITERALS, END_LITERAL_COUNT);
		if (e >= 0) *end_frame = frame;
	}
}

/* ── MVD per-payload match-text scan (high-bit STRIPPED, lowercased) ── */

static void scan_payload_strip_highbit(const uint8_t *src, size_t src_n,
                                         int frame, int *start_frame, int *end_frame)
{
	if (src_n == 0) return;
	/* Match-start/end literals are all <40 chars — scanning the first
	 * 4 KB of each payload is sufficient. Avoids per-message malloc on
	 * the hot path. */
	char buf[4096];
	if (src_n > sizeof(buf)) src_n = sizeof(buf);
	for (size_t i = 0; i < src_n; ++i)
		buf[i] = (char)tolower(src[i] & 0x7f);
	if (*start_frame < 0)
	{
		int s = find_earliest(buf, src_n, START_LITERALS, START_LITERAL_COUNT);
		if (s < 0) s = find_match_is_num(buf, src_n);
		if (s >= 0) *start_frame = frame;
	}
	if (*end_frame < 0)
	{
		int e = find_earliest(buf, src_n, END_LITERALS, END_LITERAL_COUNT);
		if (e >= 0) *end_frame = frame;
	}
}

/* ── MVD message walk ──────────────────────────────────────────────── */

/* ── MVD adapter: DEM_* record walk + QW dispatch per message ─────── */

static void walk_mvd(const uint8_t *data, size_t n,
                      bounds_t *bounds, labels_t *labels)
{
	size_t pos = 0;
	int frame = 0;
	int match_start_frame = -1, match_end_frame = -1;
	int error_frame = -1;
	/* MVD has no inference adapter yet; pass an unused accumulator so the
	 * dispatch helper keeps its uniform signature. */
	active_state_accum_t mvd_state_dummy;
	active_state_accum_reset(&mvd_state_dummy);

	/* MVD POV via dem_single/dem_multiple recipient masks is a separate
	 * mechanism from QWD's PF_WEAPONFRAME and is deferred. */
	label_state_t ls;
	lstate_init(&ls, labels);

	while (pos + 2 <= n)
	{
		/* msec at data[pos] is unused for frame-counting; just advance. */
		uint8_t type_byte = data[pos + 1];
		int dem_type = type_byte & DEM_MASK;
		pos += 2;

		if (dem_type == DEM_CMD)
		{ error_frame = frame; break; }  /* shouldn't appear in MVD */

		if (dem_type == DEM_SET)
		{
			pos += 8;
			continue;
		}

		if (dem_type == DEM_MULTIPLE)
		{
			if (pos + 4 > n) { error_frame = frame; break; }
			pos += 4;  /* player bitmask */
		}

		if (dem_type == DEM_READ || dem_type == DEM_MULTIPLE
			|| dem_type == DEM_SINGLE || dem_type == DEM_STATS
			|| dem_type == DEM_ALL)
		{
			if (pos + 4 > n) { error_frame = frame; break; }
			int32_t msg_len;
			memcpy(&msg_len, data + pos, 4);
			pos += 4;
			if (msg_len < 0 || msg_len > 65536) { error_frame = frame; break; }
			size_t msg_start = pos;
			size_t msg_end = pos + (size_t)msg_len;
			if (msg_end > n) { error_frame = frame; break; }
			pos = msg_end;

			int prev_msf = match_start_frame;
			scan_payload_strip_highbit(data + msg_start,
				msg_end - msg_start, frame,
				&match_start_frame, &match_end_frame);

			/* QW svc dispatch — MVD payload is raw QW svc opcodes
			 * (no netchan header).  Match-text detection stays with
			 * the existing high-bit-stripped per-payload scan. */
			reader_t r;
			rd_init(&r, data, msg_start, msg_end);
			int dummy_ms = -1, dummy_me = -1;
			(void)dispatch_qw_message(&r, &ls, frame, &mvd_state_dummy,
				&dummy_ms, &dummy_me, data);

			if (match_start_frame == frame && prev_msf < 0)
				lstate_match_started(&ls, frame);

			frame += 1;
			continue;
		}

		/* Unknown dem_type — payload length unknown, bail. */
		error_frame = frame;
		break;
	}

	bounds->total_frames = frame;
	bounds->match_start_frame = match_start_frame;
	bounds->match_end_frame = match_end_frame;
	bounds->error_frame = error_frame;

	int final_frame = frame > 0 ? frame - 1 : 0;
	lstate_close_all(&ls, final_frame);
}

/* ── QW protocol constants (server-to-client opcodes) ────────────────
 *
 * QW shares opcodes 1-31 with NQ but with different payload layouts:
 *   svc_updatestat (3): [byte stat][byte val]   (NQ has [byte][long])
 *   svc_print (8):      [byte level][string]    (NQ has just [string])
 *   svc_intermission (30): [3 coord][3 angle]   (NQ has no payload)
 * QW lacks svc_clientdata (15) and svc_cutscene (34) — uses svc_playerinfo
 * + svc_updatestatlong instead.
 *
 * QW-specific opcodes 33-53. */

#define QW_SVC_NOP                1
#define QW_SVC_DISCONNECT         2
#define QW_SVC_UPDATESTAT         3
#define QW_SVC_VERSION            4
#define QW_SVC_SETVIEW            5
#define QW_SVC_SOUND              6
#define QW_SVC_TIME               7
#define QW_SVC_PRINT              8
#define QW_SVC_STUFFTEXT          9
#define QW_SVC_SETANGLE          10
/* QW_SVC_SERVERDATA = 11 — defined earlier (needed by parse_signon). */
#define QW_SVC_LIGHTSTYLE        12
#define QW_SVC_UPDATENAME        13
#define QW_SVC_UPDATEFRAGS       14
#define QW_SVC_STOPSOUND         16
#define QW_SVC_UPDATECOLORS      17
#define QW_SVC_PARTICLE          18
#define QW_SVC_DAMAGE            19
#define QW_SVC_SPAWNSTATIC       20
#define QW_SVC_SPAWNBASELINE     22
#define QW_SVC_TEMP_ENTITY       23
#define QW_SVC_SETPAUSE          24
#define QW_SVC_SIGNONNUM         25
#define QW_SVC_CENTERPRINT       26
#define QW_SVC_KILLEDMONSTER     27
#define QW_SVC_FOUNDSECRET       28
#define QW_SVC_SPAWNSTATICSOUND  29
#define QW_SVC_INTERMISSION      30
#define QW_SVC_FINALE            31
#define QW_SVC_CDTRACK           32
#define QW_SVC_SELLSCREEN        33
#define QW_SVC_SMALLKICK         34
#define QW_SVC_BIGKICK           35
#define QW_SVC_UPDATEPING        36
#define QW_SVC_UPDATEENTERTIME   37
#define QW_SVC_UPDATESTATLONG    38
#define QW_SVC_MUZZLEFLASH       39
#define QW_SVC_UPDATEUSERINFO    40
#define QW_SVC_DOWNLOAD          41
#define QW_SVC_PLAYERINFO        42
#define QW_SVC_NAILS             43
#define QW_SVC_CHOKECOUNT        44
#define QW_SVC_MODELLIST         45
#define QW_SVC_SOUNDLIST         46
#define QW_SVC_PACKETENTITIES    47
#define QW_SVC_DELTAPACKETENTITIES 48
#define QW_SVC_MAXSPEED          49
#define QW_SVC_ENTGRAVITY        50
#define QW_SVC_SETINFO           51
#define QW_SVC_SERVERINFO_KV     52
#define QW_SVC_UPDATEPL          53

/* QW entity update bits — bit layout from
 * vendor/quake/QW/client/protocol.h:184-199.
 *
 * First 16-bit word: low 9 bits = entity number, high 7 bits = these flags:
 *   U_ORIGIN1=9, U_ORIGIN2=10, U_ORIGIN3=11, U_ANGLE2=12, U_FRAME=13,
 *   U_REMOVE=14, U_MOREBITS=15.
 *
 * If U_MOREBITS set, read one more byte; its low 7 bits are these flags:
 *   U_ANGLE1=0, U_ANGLE3=1, U_MODEL=2, U_COLORMAP=3, U_SKIN=4,
 *   U_EFFECTS=5, U_SOLID=6.  No data is read for U_SOLID (FIXME in
 *   vendor cl_ents.c:216). */

#define QWU_ANGLE1      (1<<0)
#define QWU_ANGLE3      (1<<1)
#define QWU_MODEL       (1<<2)
#define QWU_COLORMAP    (1<<3)
#define QWU_SKIN        (1<<4)
#define QWU_EFFECTS     (1<<5)
#define QWU_SOLID       (1<<6)
#define QWU_ORIGIN1     (1<<9)
#define QWU_ORIGIN2     (1<<10)
#define QWU_ORIGIN3     (1<<11)
#define QWU_ANGLE2      (1<<12)
#define QWU_FRAME       (1<<13)
#define QWU_REMOVE      (1<<14)
#define QWU_MOREBITS    (1<<15)

/* QW client move command bits (svc_playerinfo PF_COMMAND payload). */
#define CM_ANGLE1   (1<<0)
#define CM_ANGLE3   (1<<1)
#define CM_FORWARD  (1<<2)
#define CM_SIDE     (1<<3)
#define CM_UP       (1<<4)
#define CM_BUTTONS  (1<<5)
#define CM_IMPULSE  (1<<6)
#define CM_ANGLE2   (1<<7)

/* QW sound channel field bits. */
#define SND_VOLUME      (1<<15)
#define SND_ATTENUATION (1<<14)

/* QW player info flags. */
#define PF_MSEC         (1<<0)
#define PF_COMMAND      (1<<1)
#define PF_VELOCITY1    (1<<2)
#define PF_VELOCITY2    (1<<3)
#define PF_VELOCITY3    (1<<4)
#define PF_MODEL        (1<<5)
#define PF_SKINNUM      (1<<6)
#define PF_EFFECTS      (1<<7)
#define PF_WEAPONFRAME  (1<<8)
#define PF_DEAD         (1<<9)
#define PF_GIB          (1<<10)
#define PF_NOGRAV       (1<<11)

/* QW stat IDs — vendor QW protocol.h.  Same numbering as NQ for the
 * shared subset.  Used to drive active_state deltas off svc_updatestat
 * (and svc_updatestatlong for high-range counters). */
#define QW_STAT_HEALTH        0
#define QW_STAT_ARMOR         4
#define QW_STAT_SHELLS        6
#define QW_STAT_NAILS         7
#define QW_STAT_ROCKETS       8
#define QW_STAT_CELLS         9
#define QW_STAT_ITEMS        15

/* Apply a single QW stat update to label state + active_state.  Drives
 * the same delta semantics dispatch_nq_message gets from svc_clientdata
 * (NQ bundles all stats per tick; QW sends them individually).
 *
 * Note: QWD active_input.attack is sourced from usercmd buttons in
 * walk_qwd, so this helper doesn't touch the active_input accumulator
 * — only active_state. */
static void qw_apply_stat(label_state_t *ls, int stat_id, int32_t val,
                          int frame, active_state_accum_t *acc_state)
{
	if (stat_id == QW_STAT_HEALTH)
	{
		lstate_health(ls, val, frame);
		if (ls->prev_health >= 0)
		{
			if (val > ls->prev_health) acc_state->health_up   = 1;
			if (val < ls->prev_health) acc_state->health_down = 1;
		}
		ls->prev_health = val;
	}
	else if (stat_id == QW_STAT_ARMOR)
	{
		if (ls->prev_armor >= 0)
		{
			if (val > ls->prev_armor) acc_state->armor_up   = 1;
			if (val < ls->prev_armor) acc_state->armor_down = 1;
		}
		ls->prev_armor = val;
	}
	else if (stat_id >= QW_STAT_SHELLS && stat_id <= QW_STAT_CELLS)
	{
		int idx = stat_id - QW_STAT_SHELLS;  /* 0..3 */
		if (ls->prev_ammo[idx] >= 0)
		{
			if (val > ls->prev_ammo[idx]) acc_state->ammo_up   = 1;
			if (val < ls->prev_ammo[idx]) acc_state->ammo_down = 1;
		}
		ls->prev_ammo[idx] = val;
	}
	else if (stat_id == QW_STAT_ITEMS)
	{
		if (ls->have_prev_items)
		{
			uint32_t gained = (uint32_t)val & ~(uint32_t)ls->prev_items;
			if (gained & ITEMS_WEAPON_MASK)  acc_state->weapon_up  = 1;
			if (gained & ITEMS_SPECIAL_MASK) acc_state->special_up = 1;
		}
		ls->prev_items = val;
		ls->have_prev_items = 1;
	}
}

/* ── QW skip helpers (variable payload sizes) ─────────────────────── */

/* QW entities are always inside svc_packetentities / svc_deltapacketentities
 * — no inline U_-style updates in the message stream like NQ.
 *
 * Read order mirrors vendor/quake/QW/client/cl_ents.c:160 CL_ParseDelta. */

static void qw_skip_packetentities(reader_t *r, int is_delta)
{
	if (is_delta) rd_byte(r);  /* from-sequence */
	while (!r->overflowed)
	{
		uint16_t word = rd_ushort(r);
		if (word == 0) return;
		/* word's high 7 bits hold these flags; entity num is low 9. */
		uint16_t bits = word & ~511u;
		/* U_REMOVE entities consume only the leading short — no
		 * MOREBITS byte, no flag payloads.  Vendor CL_ParsePacketEntities
		 * (cl_ents.c:360-388) checks U_REMOVE BEFORE calling
		 * CL_ParseDelta and continues. */
		if (bits & QWU_REMOVE) continue;
		if (bits & QWU_MOREBITS) bits |= rd_byte(r);  /* low 7 bits of extra byte */
		if (bits & QWU_MODEL)    rd_byte(r);
		if (bits & QWU_FRAME)    rd_byte(r);
		if (bits & QWU_COLORMAP) rd_byte(r);
		if (bits & QWU_SKIN)     rd_byte(r);
		if (bits & QWU_EFFECTS)  rd_byte(r);
		if (bits & QWU_ORIGIN1)  rd_short(r);
		if (bits & QWU_ANGLE1)   rd_char(r);
		if (bits & QWU_ORIGIN2)  rd_short(r);
		if (bits & QWU_ANGLE2)   rd_char(r);
		if (bits & QWU_ORIGIN3)  rd_short(r);
		if (bits & QWU_ANGLE3)   rd_char(r);
		/* U_SOLID: no payload (FIXME in vendor cl_ents.c). */
	}
}

/* MSG_ReadDeltaUsercmd — variable-length.  Wire format changed between
 * protocol 26 and 27; see ezQuake com_msg.c:679-731.
 *
 * Protocol 26 (Threewave CTF and other older demos):
 *   - angles[1] is ALWAYS read (no CM_ANGLE2 gate)
 *   - bit (1<<7) is repurposed as CM_MSEC — gates the msec byte
 *   - forwardmove/sidemove/upmove are 1-byte chars (not 2-byte shorts)
 *
 * Protocol 27/28 (modern):
 *   - angles[1] gated by CM_ANGLE2 (1<<7)
 *   - forwardmove/sidemove/upmove are 2-byte shorts
 *   - msec is always read (last byte) */
#define CM_MSEC  (1<<7)  /* proto-26 alias for CM_ANGLE2 */

static void qw_skip_delta_usercmd(reader_t *r, int protocol)
{
	uint8_t bits = rd_byte(r);
	if (protocol <= 26)
	{
		if (bits & CM_ANGLE1) rd_short(r);
		rd_short(r);                         /* angles[1] always */
		if (bits & CM_ANGLE3) rd_short(r);
		if (bits & CM_FORWARD) rd_char(r);   /* 1-byte char */
		if (bits & CM_SIDE)    rd_char(r);
		if (bits & CM_UP)      rd_char(r);
		if (bits & CM_BUTTONS) rd_byte(r);
		if (bits & CM_IMPULSE) rd_byte(r);
		if (bits & CM_MSEC)    rd_byte(r);   /* gated, not always */
	}
	else
	{
		if (bits & CM_ANGLE1) rd_short(r);
		if (bits & CM_ANGLE2) rd_short(r);
		if (bits & CM_ANGLE3) rd_short(r);
		if (bits & CM_FORWARD) rd_short(r);
		if (bits & CM_SIDE)    rd_short(r);
		if (bits & CM_UP)      rd_short(r);
		if (bits & CM_BUTTONS) rd_byte(r);
		if (bits & CM_IMPULSE) rd_byte(r);
		rd_byte(r);                          /* msec always */
	}
}

/* CL_ParsePlayerinfo, vendor cl_ents.c:652.  Parses the payload
 * and returns (slot, flags) via out-params so dispatch can apply
 * PF_WEAPONFRAME-based spec-POV tracking. */
static void qw_parse_playerinfo(reader_t *r, int *slot_out, uint16_t *flags_out,
                                  int protocol)
{
	*slot_out = rd_byte(r);                  /* player num */
	uint16_t flags = rd_ushort(r);
	*flags_out = flags;
	rd_short(r); rd_short(r); rd_short(r);   /* origin (3 coords) */
	rd_byte(r);                               /* frame */
	if (flags & PF_MSEC)        rd_byte(r);
	if (flags & PF_COMMAND)     qw_skip_delta_usercmd(r, protocol);
	for (int axis = 0; axis < 3; ++axis)
		if (flags & (PF_VELOCITY1 << axis)) rd_short(r);
	if (flags & PF_MODEL)       rd_byte(r);
	if (flags & PF_SKINNUM)     rd_byte(r);
	if (flags & PF_EFFECTS)     rd_byte(r);
	if (flags & PF_WEAPONFRAME) rd_byte(r);
}

/* CL_ParseProjectiles ("nails"), vendor cl_ents.c:584. */
static void qw_skip_nails(reader_t *r)
{
	uint8_t num = rd_byte(r);
	for (int i = 0; i < num && !r->overflowed; ++i)
	{
		rd_byte(r); rd_byte(r); rd_byte(r); rd_byte(r); rd_byte(r); rd_byte(r);
	}
}

/* CL_ParseModellist / CL_ParseSoundlist, vendor cl_parse.c:656. */
static void qw_skip_stringlist(reader_t *r)
{
	rd_byte(r);  /* first index */
	while (rd_string_nonempty(r) && !r->overflowed) { /* names */ }
	rd_byte(r);  /* continuation index */
}

/* CL_ParseDownload, vendor cl_parse.c:334.  size==-1 means "file not
 * found" — no payload bytes follow; size>=0 means size bytes follow. */
static void qw_skip_download(reader_t *r)
{
	int16_t size = rd_short(r);
	rd_byte(r);  /* percent */
	if (size > 0 && (size_t)size <= r->end - r->pos)
		r->pos += (size_t)size;
	else if (size > 0)
		r->overflowed = 1;
}

/* CL_ParseServerData, vendor cl_parse.c:525.  Extracts the playernum
 * (with spec bit) so the dispatcher can seed pov_N + is_spectator
 * from the byte stream itself — no separate signon parse needed.
 *
 * Also captures the QW wire protocol version: Threewave CTF and other
 * older demos ship at protocol 26 which has a different delta-usercmd
 * layout than the protocol-28 layout id's stock parser assumes
 * (ezQuake com_msg.c:679 has the protover branch). */
static void qw_parse_serverdata(reader_t *r, int *playernum_out,
                                  int *is_spec_out, int *protocol_out)
{
	*protocol_out = rd_long(r);   /* protocol */
	rd_long(r);                   /* servercount */
	rd_skip_string(r);            /* gamedir */
	uint8_t playernum = rd_byte(r);
	*is_spec_out = (playernum & 0x80) ? 1 : 0;
	*playernum_out = playernum & 0x7f;
	rd_skip_string(r);            /* level name */
	/* movevars: gravity, stopspeed, maxspeed, spectatormaxspeed,
	 * accelerate, airaccelerate, wateraccelerate, friction,
	 * waterfriction, entgravity — 10 floats, always (vendor
	 * cl_parse.c:586-595).  We only accept protocols 26-28 anyway. */
	for (int i = 0; i < 10; ++i) rd_float(r);
}

/* CL_ParseStartSoundPacket, vendor cl_parse.c:788.  Note: byte order
 * differs from NQ — QW reads a short whose top 2 bits gate optional
 * vol/atten bytes. */
static void qw_parse_sound(reader_t *r, label_state_t *ls, int frame)
{
	uint16_t channel = rd_ushort(r);
	if (channel & SND_VOLUME)      rd_byte(r);
	if (channel & SND_ATTENUATION) rd_byte(r);
	uint8_t sound_num = rd_byte(r);
	rd_coord(r); rd_coord(r); rd_coord(r);      /* position */
	if (g_emit_fire_timing
		&& sound_num < NQ_MAX_PRECACHED_SOUNDS
		&& ls->nq_sound_kind[sound_num] == 1)
	{
		/* Vendor cl_parse.c:814: ent = (channel >> 3) & 1023. */
		int ent = (channel >> 3) & 1023;
		/* Self entity = self_slot + 1 (slot 0 = entity 1, etc.).
		 * Filter sound events to self only — we want press→sound delay
		 * for OUR shots, not teammates'. */
		if (ls->self_slot >= 0 && ent == ls->self_slot + 1)
			fprintf(stderr, "FT %.3f sound num=%d ent=%d frame %d\n",
				ls->current_demotime, sound_num, ent, frame);
	}
}

/* CL_ParseSoundlist, vendor cl_parse.c:686.  First byte is the
 * cumulative name count seen so far (server may split list across
 * multiple svc_soundlist messages).  Names are stored at indices
 * [start+1 .. start+N]. */
static void qw_parse_soundlist(reader_t *r, label_state_t *ls)
{
	int idx = rd_byte(r);
	while (!r->overflowed && idx < NQ_MAX_PRECACHED_SOUNDS - 1)
	{
		size_t s, slen;
		rd_string(r, &s, &slen);
		if (slen == 0) break;
		idx += 1;
		ls->nq_sound_kind[idx] = nq_classify_sound_name(
			(const char *)(r->data + s), slen);
	}
	rd_byte(r);  /* continuation index, ignored */
}

/* CL_ParseTEnt, vendor cl_tent.c:164.  Only types 0..13 are defined in
 * QW; anything else is a parse error. */
static void qw_skip_temp_entity(reader_t *r)
{
	uint8_t t = rd_byte(r);
	switch (t)
	{
		/* point-only: TE_SPIKE, TE_SUPERSPIKE, TE_EXPLOSION,
		 * TE_TAREXPLOSION, TE_WIZSPIKE, TE_KNIGHTSPIKE, TE_LAVASPLASH,
		 * TE_TELEPORT, TE_LIGHTNINGBLOOD. */
		case 0: case 1: case 3: case 4: case 7: case 8: case 10:
		case 11: case 13:
			rd_coord(r); rd_coord(r); rd_coord(r);
			break;
		/* beam: short ent + 6 coords (TE_LIGHTNING1, TE_LIGHTNING2,
		 * TE_LIGHTNING3). */
		case 5: case 6: case 9:
			rd_short(r);
			for (int i = 0; i < 6; ++i) rd_coord(r);
			break;
		/* byte count + 3 coords (TE_GUNSHOT, TE_BLOOD). */
		case 2: case 12:
			rd_byte(r);
			rd_coord(r); rd_coord(r); rd_coord(r);
			break;
		default:
			r->overflowed = 1;
	}
}

/* QW print: [byte level][string] */
static void qw_read_print(reader_t *r, size_t *str_start, size_t *str_len)
{
	rd_byte(r);  /* print level */
	rd_string(r, str_start, str_len);
}

/* QW intermission: [3 coord origin][3 angle viewangle] */
static void qw_skip_intermission(reader_t *r)
{
	rd_coord(r); rd_coord(r); rd_coord(r);
	rd_angle(r); rd_angle(r); rd_angle(r);
}

/* ── QW svc dispatch (single message) ──────────────────────────────── */

static int dispatch_qw_message(reader_t *r, label_state_t *ls,
                                int frame, active_state_accum_t *acc_state,
                                int *match_start_frame, int *match_end_frame,
                                const uint8_t *base)
{
	while (r->pos < r->end && !r->overflowed)
	{
		uint8_t cmd = rd_byte(r);
		switch (cmd)
		{
			case QW_SVC_NOP: continue;
			case QW_SVC_DISCONNECT: return 1;
			case QW_SVC_UPDATESTAT:
			{
				uint8_t stat_id = rd_byte(r);
				/* QW stat bytes are unsigned 0-255 (vendor common.c
				 * MSG_ReadByte).  Negative HP (corpse state) goes via
				 * the 4-byte signed svc_updatestatlong. */
				int32_t val = (int32_t)rd_byte(r);
				qw_apply_stat(ls, stat_id, val, frame, acc_state);
				if (g_emit_weapon_timing && (stat_id == 2 || stat_id == 10))
					fprintf(stderr, "WT %.3f stat%d %d frame %d\n",
					        ls->current_demotime, (int)stat_id, val, frame);
				continue;
			}
			case QW_SVC_UPDATESTATLONG:
			{
				uint8_t stat_id = rd_byte(r);
				int32_t val = rd_long(r);
				qw_apply_stat(ls, stat_id, val, frame, acc_state);
				if (g_emit_weapon_timing && (stat_id == 2 || stat_id == 10))
					fprintf(stderr, "WT %.3f stat%d %d frame %d\n",
					        ls->current_demotime, (int)stat_id, val, frame);
				continue;
			}
			case QW_SVC_VERSION: rd_long(r); continue;
			case QW_SVC_SETVIEW:
				rd_short(r);
				continue;
			case QW_SVC_SOUND: qw_parse_sound(r, ls, frame); continue;
			case QW_SVC_TIME:  rd_float(r); continue;
			case QW_SVC_PRINT:
			case QW_SVC_STUFFTEXT:
			case QW_SVC_CENTERPRINT:
			case QW_SVC_FINALE:
			{
				size_t s, slen;
				if (cmd == QW_SVC_PRINT)
					qw_read_print(r, &s, &slen);
				else
					rd_string(r, &s, &slen);
				int prev_msf = *match_start_frame;
				scan_text_for_match(base + s, slen, frame,
					match_start_frame, match_end_frame);
				if (*match_start_frame == frame && prev_msf < 0)
					lstate_match_started(ls, frame);
				continue;
			}
			case QW_SVC_SETANGLE:
				rd_angle(r); rd_angle(r); rd_angle(r);
				continue;
			case QW_SVC_SERVERDATA:
			{
				int playernum, is_spec, protocol;
				qw_parse_serverdata(r, &playernum, &is_spec, &protocol);
				ls->is_spectator = is_spec;
				ls->qw_protocol = protocol;
				ls->self_slot = playernum;
				continue;
			}
			case QW_SVC_LIGHTSTYLE: rd_byte(r); rd_skip_string(r); continue;
			case QW_SVC_UPDATENAME: rd_byte(r); rd_skip_string(r); continue;
			case QW_SVC_UPDATEFRAGS:
			{
				int slot = (int)rd_byte(r);
				int frags = (int)rd_short(r);
				if (slot == ls->self_slot)
				{
					if (ls->prev_frags >= 0 && frags > ls->prev_frags)
						acc_state->frag_up = 1;
					ls->prev_frags = frags;
				}
				continue;
			}
			case QW_SVC_STOPSOUND: rd_short(r); continue;
			case QW_SVC_UPDATECOLORS: rd_byte(r); rd_byte(r); continue;
			case QW_SVC_PARTICLE: skip_particle(r); continue;
			case QW_SVC_DAMAGE: skip_damage(r); continue;
			case QW_SVC_SPAWNSTATIC: skip_baseline(r); continue;
			case QW_SVC_SPAWNBASELINE:
				rd_short(r);
				skip_baseline(r);
				continue;
			case QW_SVC_TEMP_ENTITY: qw_skip_temp_entity(r); continue;
			case QW_SVC_SETPAUSE:
				lstate_setpause(ls, rd_byte(r), frame);
				continue;
			case QW_SVC_SIGNONNUM: rd_byte(r); continue;
			case QW_SVC_KILLEDMONSTER:
			case QW_SVC_FOUNDSECRET:
			case QW_SVC_SELLSCREEN:
			case QW_SVC_SMALLKICK:
			case QW_SVC_BIGKICK:
				continue;
			case QW_SVC_SPAWNSTATICSOUND:
				rd_coord(r); rd_coord(r); rd_coord(r);
				rd_byte(r); rd_byte(r); rd_byte(r);
				continue;
			case QW_SVC_INTERMISSION:
				lstate_intermission(ls, frame);
				qw_skip_intermission(r);
				continue;
			case QW_SVC_CDTRACK: rd_byte(r); continue;  /* QW: 1 byte */
			case QW_SVC_UPDATEPING:
			{
				uint8_t slot = rd_byte(r);
				uint16_t ping_ms = rd_ushort(r);
				if (ls->self_slot >= 0 && (int)slot == ls->self_slot
				    && ping_ms > 0 && ping_ms < 999)
				{
					ls->ping_sum_ms += ping_ms;
					ls->ping_count++;
					if (ls->emit_ping_history && ls->ph_n < 8192)
					{
						ls->ph_frame[ls->ph_n] = (uint32_t)frame;
						ls->ph_ping[ls->ph_n] = ping_ms;
						ls->ph_n++;
					}
				}
				if (g_emit_fire_timing
					&& ls->self_slot >= 0 && (int)slot == ls->self_slot
					&& ping_ms > 0 && ping_ms < 999)
					fprintf(stderr, "FT %.3f ping ms=%d frame %d\n",
						ls->current_demotime, (int)ping_ms, frame);
				continue;
			}
			case QW_SVC_UPDATEENTERTIME: rd_byte(r); rd_float(r); continue;
			case QW_SVC_MUZZLEFLASH: rd_short(r); continue;
			case QW_SVC_UPDATEUSERINFO:
				rd_byte(r); rd_long(r); rd_skip_string(r);
				continue;
			case QW_SVC_DOWNLOAD: qw_skip_download(r); continue;
			case QW_SVC_PLAYERINFO:
			{
				/* Variable-length opcode — must still parse to keep
				 * the byte stream aligned.  Protocol-version-aware
				 * skip lives in qw_parse_playerinfo.  We also fold
				 * the slot into the per-frame seen mask so the walker
				 * can emit an "other actors visible per frame"
				 * histogram alongside the other tallies. */
				int slot;
				uint16_t pf;
				qw_parse_playerinfo(r, &slot, &pf, ls->qw_protocol);
				if (slot >= 0 && slot < 32) {
					ls->frame_player_seen |= (1u << slot);
					if (pf & (PF_DEAD | PF_GIB))
						ls->frame_player_dead |= (1u << slot);
				}
				continue;
			}
			case QW_SVC_NAILS: qw_skip_nails(r); continue;
			case QW_SVC_CHOKECOUNT: rd_byte(r); continue;
			case QW_SVC_MODELLIST: qw_skip_stringlist(r); continue;
			case QW_SVC_SOUNDLIST: qw_parse_soundlist(r, ls); continue;
			case QW_SVC_PACKETENTITIES: qw_skip_packetentities(r, 0); continue;
			case QW_SVC_DELTAPACKETENTITIES: qw_skip_packetentities(r, 1); continue;
			case QW_SVC_MAXSPEED:
			case QW_SVC_ENTGRAVITY: rd_float(r); continue;
			case QW_SVC_SETINFO: rd_byte(r); rd_skip_string(r); rd_skip_string(r); continue;
			case QW_SVC_SERVERINFO_KV: rd_skip_string(r); rd_skip_string(r); continue;
			case QW_SVC_UPDATEPL: rd_byte(r); rd_byte(r); continue;
			default:
				return 1;  /* unknown opcode */
		}
	}
	return 0;
}

/* ── NQ svc dispatch (single message) ──────────────────────────────── */

/* Dispatches NQ svc opcodes inside one message payload, calling label
 * hooks for events we care about and folding per-frame inference bits
 * into `acc` for the NQ active_input adapter.  Returns 0 on clean
 * parse, 1 if disconnect / unknown opcode terminated parsing. */
static int dispatch_nq_message(reader_t *r, label_state_t *ls,
                                int frame, active_input_accum_t *acc,
                                active_state_accum_t *acc_state,
                                int *match_start_frame, int *match_end_frame,
                                const uint8_t *base)
{
	while (r->pos < r->end && !r->overflowed)
	{
		uint8_t cmd = rd_byte(r);
		/* Vendor's "end of message" is MSG_ReadByte returning -1
		 * (i.e. overflow), not a literal 0xFF in the stream — 0xFF
		 * with bit 7 set is a valid entity update.  The loop guard
		 * (r->overflowed) handles real EOM. */
		if (cmd & 128)
		{
			nq_track_entity_update(r, cmd & 127, ls, acc);
			continue;
		}
		if (cmd == SVC_NOP) continue;
		if (cmd == SVC_DISCONNECT) return 1;
		if (cmd == SVC_UPDATESTAT)
		{
			uint8_t stat_id = rd_byte(r);
			int32_t val = rd_long(r);
			if (stat_id == STAT_HEALTH)
				lstate_health(ls, val, frame);
			continue;
		}
		if (cmd == SVC_VERSION)   { rd_long(r); continue; }
		if (cmd == SVC_SETVIEW)
		{
			ls->nq_self_entity = (int)(int16_t)rd_short(r);
			ls->self_slot = ls->nq_self_entity - 1;
			continue;
		}
		if (cmd == SVC_SOUND)
		{
			/* Parse svc_sound (vendor cl_parse.c:101) and fold per-self
			 * fire/jump events into the active_input accumulator. */
			uint8_t mask = rd_byte(r);
			if (mask & 1) rd_byte(r);   /* volume */
			if (mask & 2) rd_byte(r);   /* attenuation */
			uint16_t chan_ent = rd_ushort(r);
			uint8_t sound_num = rd_byte(r);
			rd_coord(r); rd_coord(r); rd_coord(r);
			int sound_entity = (int)(chan_ent >> 3);
			if (sound_entity == ls->nq_self_entity
			    && sound_num < NQ_MAX_PRECACHED_SOUNDS)
			{
				uint8_t kind = ls->nq_sound_kind[sound_num];
				if (kind == 1) acc->attack = 1;
				else if (kind == 2) acc->jump = 1;
			}
			continue;
		}
		if (cmd == SVC_TIME)      { rd_float(r); continue; }
		if (cmd == SVC_PRINT || cmd == SVC_STUFFTEXT
			|| cmd == SVC_CENTERPRINT
			|| cmd == SVC_FINALE  || cmd == SVC_CUTSCENE)
		{
			size_t s, slen;
			rd_string(r, &s, &slen);
			int prev_msf = *match_start_frame;
			scan_text_for_match(base + s, slen, frame,
				match_start_frame, match_end_frame);
			if (*match_start_frame == frame && prev_msf < 0)
				lstate_match_started(ls, frame);
			continue;
		}
		if (cmd == SVC_SETANGLE)
		{
			rd_angle(r); rd_angle(r); rd_angle(r);
			continue;
		}
		if (cmd == SVC_SERVERINFO) { nq_parse_serverinfo(r, ls); continue; }
		if (cmd == SVC_LIGHTSTYLE)
		{
			rd_byte(r); rd_skip_string(r);
			continue;
		}
		if (cmd == SVC_UPDATENAME)
		{
			rd_byte(r); rd_skip_string(r);
			continue;
		}
		if (cmd == SVC_UPDATEFRAGS)
		{
			int slot = (int)rd_byte(r);
			int frags = (int)rd_short(r);
			if (slot == ls->self_slot)
			{
				if (ls->prev_frags >= 0 && frags > ls->prev_frags)
					acc_state->frag_up = 1;
				ls->prev_frags = frags;
			}
			continue;
		}
		if (cmd == SVC_CLIENTDATA)
		{
			int health, armor, active_weapon;
			int32_t items;
			int ammo_pools[4];  /* shells, nails, rockets, cells */
			if (!read_clientdata(r, &health, &armor,
				&ammo_pools[0], &ammo_pools[1], &ammo_pools[2], &ammo_pools[3],
				&active_weapon, &items)) return 1;
			lstate_health(ls, health, frame);
			/* active_input.attack fallback: per-pool ammo decrement.
			 * active_state.ammo_down/up: per-pool delta in either direction. */
			if (ls->prev_ammo[0] >= 0)
			{
				for (int i = 0; i < 4; ++i)
				{
					if (ammo_pools[i] < ls->prev_ammo[i])
					{ acc->attack = 1; acc_state->ammo_down = 1; }
					else if (ammo_pools[i] > ls->prev_ammo[i])
						acc_state->ammo_up = 1;
				}
			}
			for (int i = 0; i < 4; ++i)
				ls->prev_ammo[i] = ammo_pools[i];
			if (ls->nq_prev_active_weapon >= 0
			    && active_weapon != ls->nq_prev_active_weapon)
				acc->weaponswitch = 1;
			ls->nq_prev_active_weapon = active_weapon;
			/* active_state: health / armor / items deltas. */
			if (ls->prev_health >= 0)
			{
				if (health > ls->prev_health) acc_state->health_up   = 1;
				if (health < ls->prev_health) acc_state->health_down = 1;
			}
			ls->prev_health = health;
			if (armor >= 0)
			{
				if (ls->prev_armor >= 0)
				{
					if (armor > ls->prev_armor) acc_state->armor_up   = 1;
					if (armor < ls->prev_armor) acc_state->armor_down = 1;
				}
				ls->prev_armor = armor;
			}
			if (ls->have_prev_items)
			{
				uint32_t gained = (uint32_t)items & ~(uint32_t)ls->prev_items;
				if (gained & ITEMS_WEAPON_MASK)  acc_state->weapon_up  = 1;
				if (gained & ITEMS_SPECIAL_MASK) acc_state->special_up = 1;
			}
			ls->prev_items = items;
			ls->have_prev_items = 1;
			continue;
		}
		if (cmd == SVC_STOPSOUND)    { rd_short(r); continue; }
		if (cmd == SVC_UPDATECOLORS) { rd_byte(r); rd_byte(r); continue; }
		if (cmd == SVC_PARTICLE)     { skip_particle(r); continue; }
		if (cmd == SVC_DAMAGE)       { skip_damage(r); continue; }
		if (cmd == SVC_SPAWNSTATIC)  { skip_baseline(r); continue; }
		if (cmd == SVC_SPAWNBASELINE)
		{
			rd_short(r);
			skip_baseline(r);
			continue;
		}
		if (cmd == SVC_TEMP_ENTITY) { skip_temp_entity(r); continue; }
		if (cmd == SVC_SETPAUSE)
		{
			lstate_setpause(ls, rd_byte(r), frame);
			continue;
		}
		if (cmd == SVC_SIGNONNUM)   { rd_byte(r); continue; }
		if (cmd == SVC_INTERMISSION)
		{
			lstate_intermission(ls, frame);
			continue;
		}
		if (cmd == SVC_KILLEDMONSTER || cmd == SVC_FOUNDSECRET
			|| cmd == SVC_SELLSCREEN)
			continue;
		if (cmd == SVC_SPAWNSTATICSOUND)
		{
			rd_coord(r); rd_coord(r); rd_coord(r);
			rd_byte(r); rd_byte(r); rd_byte(r);
			continue;
		}
		if (cmd == SVC_CDTRACK) { rd_byte(r); rd_byte(r); continue; }

		return 1;  /* unknown opcode */
	}
	return 0;
}

/* ── NQ adapter: 16-byte msg-header walk + per-message NQ dispatch ── */

static void walk_nq(const uint8_t *data, size_t n,
                     bounds_t *bounds, active_input_t *ai,
                     active_state_t *as,
                     labels_t *labels)
{
	/* Strip optional CD-track header (first text line ending in '\n'). */
	const uint8_t *nl = memchr(data, '\n', n);
	if (nl == NULL)
	{
		bounds->total_frames = 0;
		bounds->error_frame = 0;
		return;
	}
	size_t pos = (size_t)(nl - data) + 1;

	label_state_t ls;
	lstate_init(&ls, labels);

	/* Per-frame active-input accumulator.  NQ has no recorded usercmd
	 * stream — channel bits are inferred from server-emitted signals
	 * during dispatch (sound → attack/jump, clientdata velocity →
	 * fwd/side/up, clientdata weapon → weaponswitch, entity-update
	 * angles for self → pitch/yaw). */
	active_input_accum_t acc;
	active_input_accum_reset(&acc);
	active_state_accum_t acc_state;
	active_state_accum_reset(&acc_state);

	int frame = 0;
	int match_start_frame = -1, match_end_frame = -1;
	int error_frame = -1;

	while (pos + 16 <= n)
	{
		int32_t msg_len;
		memcpy(&msg_len, data + pos, 4);
		/* Vendor cl_demo.c:74-78: each NQ DEM message header is
		 *   4 bytes msg_len
		 *  12 bytes viewangles[3] (3 little-endian floats)
		 *   N bytes payload
		 * The viewangles are cl.viewangles at the moment the message
		 * was received — i.e. the recorder's local aim. */
		float view[3];
		memcpy(view, data + pos + 4, 12);
		size_t msg_start = pos + 16;
		if (msg_len < 0 || msg_len > 65536) { error_frame = frame; break; }
		size_t msg_end = msg_start + (size_t)msg_len;
		if (msg_end > n) { error_frame = frame; break; }

		/* Look-delta detection — pitch/yaw/roll active if changed vs
		 * the previous message's viewangle.  Skip the very first
		 * message (no prior). */
		if (ls.nq_have_view_angles)
		{
			if (view[0] != ls.nq_view_pitch) acc.pitch = 1;
			if (view[1] != ls.nq_view_yaw)   acc.yaw   = 1;
			if (view[2] != ls.nq_view_roll)  acc.roll  = 1;
		}
		ls.nq_view_pitch = view[0];
		ls.nq_view_yaw   = view[1];
		ls.nq_view_roll  = view[2];
		ls.nq_have_view_angles = 1;

		reader_t r;
		rd_init(&r, data, msg_start, msg_end);

		(void)dispatch_nq_message(&r, &ls, frame, &acc, &acc_state,
			&match_start_frame, &match_end_frame, data);

		active_input_commit(ai, &acc);
		active_state_commit(as, &acc_state);

		pos = msg_end;
		frame += 1;
	}

	bounds->total_frames = frame;
	bounds->match_start_frame = match_start_frame;
	bounds->match_end_frame = match_end_frame;
	bounds->error_frame = error_frame;

	int final_frame = frame > 0 ? frame - 1 : 0;
	lstate_close_all(&ls, final_frame);
}

/* ── JSON output helpers ───────────────────────────────────────────── */

static void emit_json_string(FILE *out, const char *s)
{
	fputc('"', out);
	for (; *s; ++s)
	{
		unsigned char c = (unsigned char)*s;
		if (c == '"' || c == '\\') { fputc('\\', out); fputc(c, out); }
		else if (c == '\n') fputs("\\n", out);
		else if (c == '\r') fputs("\\r", out);
		else if (c == '\t') fputs("\\t", out);
		else if (c < 0x20 || c >= 0x7f) fprintf(out, "\\u%04x", c);
		else fputc(c, out);
	}
	fputc('"', out);
}

/* Emit the labels dict as JSON: only includes labels with non-empty
 * interval lists.  Empty labels (label never fired) are omitted. */
static void emit_labels_json(FILE *out, const labels_t *L)
{
	fputc('{', out);
	int first = 1;
	for (int i = 0; i < L->count; ++i)
	{
		const label_track_t *t = &L->tracks[i];
		if (t->count == 0) continue;
		if (!first) fputc(',', out);
		first = 0;
		fputc('"', out);
		fputs(t->name, out);
		fputc('"', out);
		fputc(':', out);
		fputc('[', out);
		for (int j = 0; j < t->count; ++j)
		{
			if (j > 0) fputc(',', out);
			fprintf(out, "[%d,%d]", t->intervals[j].start, t->intervals[j].end);
		}
		fputc(']', out);
	}
	fputc('}', out);
}

/* ── Per-demo classify ──────────────────────────────────────────────── */

static const char *path_extension_lower(const char *path, char buf[8])
{
	const char *dot = strrchr(path, '.');
	if (dot == NULL) { buf[0] = 0; return buf; }
	int i = 0;
	for (const char *p = dot + 1; *p && i < 7; ++p, ++i)
		buf[i] = (char)tolower((unsigned char)*p);
	buf[i] = 0;
	return buf;
}

static int classify_one(const char *demo_path, FILE *out)
{
	int fd = open(demo_path, O_RDONLY);
	if (fd < 0)
	{
		fprintf(out, "{\"ok\":false,\"error\":\"open_failed\",\"demo\":");
		emit_json_string(out, demo_path);
		fputs("}\n", out);
		return -1;
	}
	struct stat st;
	if (fstat(fd, &st) < 0 || st.st_size <= 0)
	{
		close(fd);
		fprintf(out, "{\"ok\":false,\"error\":\"stat_failed\",\"demo\":");
		emit_json_string(out, demo_path);
		fputs("}\n", out);
		return -1;
	}
	size_t n = (size_t)st.st_size;
	void *map = mmap(NULL, n, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if (map == MAP_FAILED)
	{
		fprintf(out, "{\"ok\":false,\"error\":\"mmap_failed\",\"demo\":");
		emit_json_string(out, demo_path);
		fputs("}\n", out);
		return -1;
	}
	const uint8_t *data = (const uint8_t *)map;

	char ext[8];
	path_extension_lower(demo_path, ext);
	int is_nq = (strcmp(ext, "dem") == 0);
	int is_mvd = (strcmp(ext, "mvd") == 0);

	bounds_t bounds;
	memset(&bounds, 0, sizeof(bounds));
	bounds.error_frame = -1;
	bounds.match_start_frame = -1;
	bounds.match_end_frame = -1;

	labels_t labels;
	labels_init(&labels);

	/* Per-frame active-input + active-state tallies — emitted on every
	 * format for a stable schema; MVD walker leaves both zero (no inference
	 * adapter yet). */
	active_input_t ai;
	memset(&ai, 0, sizeof(ai));
	active_state_t as;
	memset(&as, 0, sizeof(as));
	int is_spectator = 0;  /* signon playernum high bit (QWD only) */

	const char *format = "qwd";
	int extract_serverinfo = 0;

	if (is_nq)
	{
		format = "dem";
		walk_nq(data, n, &bounds, &ai, &as, &labels);
	}
	else if (is_mvd)
	{
		format = "mvd";
		walk_mvd(data, n, &bounds, &labels);
		extract_serverinfo = 1;  /* MVD has fullserverinfo too */
	}
	else
	{
		/* QWD path: lowercased copy for whole-file text scan. */
		char *lower = (char *)malloc(n);
		if (lower == NULL)
		{
			munmap(map, n);
			fprintf(out, "{\"ok\":false,\"error\":\"alloc_failed\",\"demo\":");
			emit_json_string(out, demo_path);
			fputs("}\n", out);
			return -1;
		}
		strip_highbit_lower(data, n, lower);

		int match_start_off = -1, match_end_off = -1;
		find_match_positions(lower, n, &match_start_off, &match_end_off);

		walk_qwd(data, n, match_start_off, match_end_off,
		         &bounds, &ai, &as, &labels, &is_spectator);
		free(lower);
		extract_serverinfo = 1;
	}

	/* Derive `match` interval from bounds (uniform across formats).
	 * If the first-detected end-text predates the first-detected
	 * start-text (multi-match demo where first match ended before a
	 * second one began), the two events aren't a pair — drop the end
	 * and run the interval to demo end. */
	if (bounds.match_start_frame >= 0)
	{
		int end;
		if (bounds.match_end_frame >= bounds.match_start_frame)
			end = bounds.match_end_frame;
		else
			end = bounds.total_frames > 0 ? bounds.total_frames - 1 : 0;
		label_open(&labels, "match", bounds.match_start_frame);
		label_close(&labels, "match", end);
	}

	/* Recorder role — single top-level value per demo.
	 *   "player"    — QWD/NQ recording from a real player's client
	 *   "spectator" — QWD spec-mode recording (signon playernum high bit)
	 *   "server"    — MVD server-side multi-perspective recording */
	const char *recorder;
	if (is_mvd)
		recorder = "server";
	else if (is_spectator)
		recorder = "spectator";
	else
		recorder = "player";

	/* ── emit JSON ── */
	fputs("{\"ok\":true,\"demo\":", out);
	emit_json_string(out, demo_path);
	fprintf(out, ",\"format\":\"%s\",\"recorder\":\"%s\"", format, recorder);
	fprintf(out, ",\"total_frames\":%d", bounds.total_frames);
	if (bounds.error_frame >= 0)
		fprintf(out, ",\"error_frame\":%d", bounds.error_frame);
	/* Per-frame active-input + active-state tallies — emitted on every
	 * format.  MVD blocks are all-zero (no inference adapter yet). */
	fputs(",\"active_input\":{", out);
	fprintf(out,
		"\"forwardmove\":%d,\"sidemove\":%d,\"upmove\":%d,"
		"\"pitch\":%d,\"yaw\":%d,\"roll\":%d,"
		"\"attack\":%d,\"jump\":%d,\"use\":%d,"
		"\"weaponswitch\":%d,\"none\":%d",
		ai.forwardmove, ai.sidemove, ai.upmove,
		ai.pitch, ai.yaw, ai.roll,
		ai.attack, ai.jump, ai.use,
		ai.weaponswitch, ai.none);
	fputc('}', out);
	fputs(",\"active_state\":{", out);
	fprintf(out,
		"\"health_up\":%d,\"health_down\":%d,"
		"\"armor_up\":%d,\"armor_down\":%d,"
		"\"ammo_up\":%d,\"ammo_down\":%d,"
		"\"frag_up\":%d,\"weapon_up\":%d,\"special_up\":%d",
		as.health_up, as.health_down,
		as.armor_up, as.armor_down,
		as.ammo_up, as.ammo_down,
		as.frag_up, as.weapon_up, as.special_up);
	fputc('}', out);
	fputs(",\"info\":", out);
	if (extract_serverinfo)
		parse_fullserverinfo(data, n, out);
	else
		fputs("{}", out);
	fputs(",\"labels\":", out);
	emit_labels_json(out, &labels);
	if (bounds.ping_count > 0)
		fprintf(out, ",\"avg_ping_ms\":%.1f",
		        (double)bounds.ping_sum_ms / (double)bounds.ping_count);
	else
		fputs(",\"avg_ping_ms\":null", out);
	if (g_emit_ping_history) {
		fputs(",\"ping_history\":[", out);
		for (int pi = 0; pi < bounds.ph_n; ++pi) {
			if (pi > 0) fputc(',', out);
			fprintf(out, "[%u,%u]", bounds.ph_frame[pi], bounds.ph_ping[pi]);
		}
		fputc(']', out);
	}
	/* Per-frame other-player histogram (QWD only; MVD/NQ leave the
	 * array zeroed).  Bucket k = number of DEM_READ frames where
	 * exactly k OTHER players (alive, slot != self_slot) had a
	 * svc_playerinfo update this frame. */
	fputs(",\"actors_per_frame\":[", out);
	for (int k = 0; k < 33; ++k) {
		if (k > 0) fputc(',', out);
		fprintf(out, "%u", bounds.actors_per_frame[k]);
	}
	fputc(']', out);
	fputs("}\n", out);

	munmap(map, n);
	return 0;
}

/* ── Main loop: read demo paths from stdin, one per line ───────────── */

int main(int argc, char **argv)
{
	(void)argc;
	(void)argv;

	const char *env = getenv("QNN_EMIT_PING_HISTORY");
	if (env != NULL && env[0] != '\0' && env[0] != '0')
		g_emit_ping_history = 1;
	const char *wt_env = getenv("QNN_EMIT_WEAPON_TIMING");
	if (wt_env != NULL && wt_env[0] != '\0' && wt_env[0] != '0')
		g_emit_weapon_timing = 1;
	const char *ft_env = getenv("QNN_EMIT_FIRE_TIMING");
	if (ft_env != NULL && ft_env[0] != '\0' && ft_env[0] != '0')
		g_emit_fire_timing = 1;

	char buf[4096];
	setvbuf(stdout, NULL, _IOLBF, 0);
	while (fgets(buf, sizeof(buf), stdin) != NULL)
	{
		size_t len = strlen(buf);
		while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
			buf[--len] = 0;
		if (len == 0) continue;
		classify_one(buf, stdout);
		fflush(stdout);
	}
	return 0;
}
