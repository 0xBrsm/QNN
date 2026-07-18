/*
 * qnn_sound.c — Sound capture ring buffer and S_* engine API stubs.
 *
 * The demo worker replaces the real sound system with a capture shim:
 * every S_StartSound writes the event (entity, origin, name) into a
 * ring buffer.  QNN_DrainSounds snapshots the buffer into the per-tick
 * snapshot struct where event classification can inspect it.
 *
 * Sound names are resolved through S_PrecacheSound: each call stores
 * the sample name into a parallel array indexed by sfx pointer offset.
 * Callers of S_StartSound pass a sfx_t* — we convert pointer → index
 * → name to recover the original "weapons/sgun1.wav" string.
 *
 * Shared by common/qnn_event.c across NQ and QW builds.  The event
 * classification itself (sound rules, QNN_SnapshotHasSelf* helpers,
 * event atom pool) all live in qnn_event.c; this file owns only the
 * engine-facing S_* shim + ring buffer drain.  QW's compat layer
 * aliases cl.viewentity = cl.playernum + 1 so self-identification
 * works uniformly in the shared classifier.
 */

#include "qnn.h"
#include "qnn_context.h"

#include <string.h>

#define QNN_SND_MAX_PRECACHE 512

static char qnn_snd_precache_names[QNN_SND_MAX_PRECACHE][MAX_QPATH];
static sfx_t qnn_snd_precache_sfx[QNN_SND_MAX_PRECACHE];
static int qnn_snd_precache_count = 0;

cvar_t bgmvolume = {"bgmvolume", "1"};
cvar_t volume = {"volume", "0.7"};

qnn_sound_event_t qnn_sound_buffer[QNN_MAX_SOUNDS];
int qnn_sound_count = 0;

/* Optional debug dump file — set by NQ's collect/trainer main when
 * QNN_SOUND_DUMP env var is set.  QW currently leaves it NULL. */
FILE *qnn_sound_dump = NULL;

void QNN_SoundRegisterContext(void)
{
	QNN_ContextRegister(qnn_sound_buffer, sizeof(qnn_sound_buffer));
	QNN_ContextRegister(&qnn_sound_count, sizeof(qnn_sound_count));
}

void QNN_DrainSounds(qnn_snapshot_t *snapshot)
{
	snapshot->sound_count = qnn_sound_count < QNN_MAX_SOUNDS
		? qnn_sound_count : QNN_MAX_SOUNDS;
	if (snapshot->sound_count > 0)
		memcpy(snapshot->sounds, qnn_sound_buffer,
			snapshot->sound_count * sizeof(qnn_sound_event_t));
	qnn_sound_count = 0;
}

/* ── Engine API stubs ────────────────────────────────────────────── */

void S_Init(void)
{
	Cvar_RegisterVariable(&bgmvolume);
	Cvar_RegisterVariable(&volume);
}

void S_AmbientOff(void) {}
void S_AmbientOn(void) {}
void S_Shutdown(void) {}
void S_TouchSound(char *sample) { (void)sample; }
void S_ClearBuffer(void) {}

void S_StaticSound(sfx_t *sfx, vec3_t origin, float vol, float attenuation)
{
	(void)sfx; (void)origin; (void)vol; (void)attenuation;
}

void S_StartSound(int entnum, int entchannel, sfx_t *sfx, vec3_t origin,
	float fvol, float attenuation)
{
	qnn_sound_event_t *snd;
	const char *name;
	int i;
	float now;

	(void)entchannel;

	if (qnn_sound_count >= QNN_MAX_SOUNDS)
		return;

	/* Resolve sound name from sfx pointer offset into our precache pool */
	name = "";
	if (sfx != NULL)
	{
		int index = (int)(sfx - qnn_snd_precache_sfx);
		if (index >= 0 && index < qnn_snd_precache_count)
			name = qnn_snd_precache_names[index];
		else
			name = sfx->name;
	}

	now = (float)cl.mtime[0];

	if (qnn_sound_dump != NULL)
		fprintf(qnn_sound_dump, "%.3f\t%d\t%s\n", now, entnum, name);

	/* Dedupe within the current native tick.  qnn_sound_buffer is reset
	 * every QNN_DrainSounds (once per native tick), so any event
	 * already in the buffer is from this tick.  If a new event matches
	 * an existing one on (entity, sample), drop it — some demos carry
	 * duplicate svc_sound messages for the same emit (root cause is in
	 * the QW server / recorder / protocol stack and not yet pinned
	 * down).  We deliberately do NOT compare native_time because the
	 * duplicates we've measured come at near-identical (but not
	 * bit-identical) times within the same Host_Frame, and a single
	 * tick can't physically contain two distinct emits of the same
	 * sample on the same entity. */
	for (i = 0; i < qnn_sound_count; ++i)
	{
		qnn_sound_event_t *prev = &qnn_sound_buffer[i];
		if (prev->entity_num == entnum
			&& strncmp(prev->name, name, sizeof(prev->name)) == 0)
			return;
	}

	snd = &qnn_sound_buffer[qnn_sound_count];
	VectorCopy(origin, snd->origin);
	snd->volume = fvol;
	snd->attenuation = attenuation;
	snd->entity_num = entnum;
	strncpy(snd->name, name, sizeof(snd->name) - 1);
	snd->name[sizeof(snd->name) - 1] = '\0';
	snd->native_time = (float)cl.mtime[0];
	qnn_sound_count += 1;
}

void S_StopSound(int entnum, int entchannel)
{
	(void)entnum; (void)entchannel;
}

sfx_t *S_PrecacheSound(char *sample)
{
	sfx_t *sfx;
	int index;

	/* In-process arena observers share immutable sound definitions.  Reuse
	 * the first parser's entry so sixteen client contexts do not consume the
	 * fixed precache table sixteen times. */
	for (index = 0; index < qnn_snd_precache_count; ++index)
		if (!strcmp(qnn_snd_precache_names[index], sample))
			return &qnn_snd_precache_sfx[index];

	if (qnn_snd_precache_count >= QNN_SND_MAX_PRECACHE)
		return NULL;
	sfx = &qnn_snd_precache_sfx[qnn_snd_precache_count];
	memset(sfx, 0, sizeof(*sfx));
	strncpy(sfx->name, sample, sizeof(sfx->name) - 1);
	sfx->name[sizeof(sfx->name) - 1] = '\0';
	strncpy(qnn_snd_precache_names[qnn_snd_precache_count], sample, MAX_QPATH - 1);
	qnn_snd_precache_names[qnn_snd_precache_count][MAX_QPATH - 1] = '\0';
	qnn_snd_precache_count += 1;
	return sfx;
}

void S_ClearPrecache(void) { qnn_snd_precache_count = 0; }
void S_Update(vec3_t origin, vec3_t forward, vec3_t right, vec3_t up)
	{ (void)origin; (void)forward; (void)right; (void)up; }
void S_StopAllSounds(qboolean clear) { (void)clear; qnn_sound_count = 0; }
void S_BeginPrecaching(void) {}
void S_EndPrecaching(void) {}
void S_ExtraUpdate(void) {}
void S_LocalSound(char *s) { (void)s; }
