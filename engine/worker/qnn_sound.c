/*
 * qnn_sound.c — Replaces snd_null.c for the Quake worker.
 *
 * All playback is still null (no audio output), but S_PrecacheSound stores
 * sound names and S_StartSound captures spatial sound events into a ring
 * buffer that the worker drains each tick for the observation stream.
 *
 * This gives the RL agent the same spatial audio signals a human client
 * receives: direction, distance (attenuation), and sound type — without
 * requiring a real audio subsystem.
 */

#include "qnn.h"

/* ---- Precache name table ------------------------------------------------ */

#define QNN_SND_MAX_PRECACHE 512

static char qnn_snd_precache_names[QNN_SND_MAX_PRECACHE][MAX_QPATH];

/* Dummy sfx_t entries — the engine only needs non-NULL pointers back from
   S_PrecacheSound so that cl.sound_precache[] slots are populated.  We store
   the name inside the sfx_t struct (which starts with char name[MAX_QPATH])
   for later lookup in S_StartSound. */
static sfx_t qnn_snd_precache_sfx[QNN_SND_MAX_PRECACHE];
static int qnn_snd_precache_count = 0;

cvar_t bgmvolume = {"bgmvolume", "1"};
cvar_t volume = {"volume", "0.7"};

/* ---- Sound event ring buffer -------------------------------------------- */

qnn_sound_event_t qnn_sound_buffer[QNN_MAX_SOUNDS];
int qnn_sound_count = 0;

/* ---- Sound category classification -------------------------------------- */

static int QNN_SndClassify(const char *name)
{
	if (!name || !name[0])
		return QNN_SND_CAT_AMBIENT;

	/* Weapon fire */
	if (strstr(name, "weapons/"))
	{
		if (strstr(name, "rocket") || strstr(name, "sgun") ||
		    strstr(name, "grenade") || strstr(name, "nail") ||
		    strstr(name, "spike") || strstr(name, "lstart") ||
		    strstr(name, "lhit") || strstr(name, "bounce"))
			return QNN_SND_CAT_WEAPON;
		return QNN_SND_CAT_WEAPON;
	}

	/* Player sounds */
	if (strstr(name, "player/"))
	{
		if (strstr(name, "pain") || strstr(name, "death") ||
		    strstr(name, "gib") || strstr(name, "udeath"))
			return QNN_SND_CAT_PAIN;
		if (strstr(name, "land") || strstr(name, "plyrjmp") ||
		    strstr(name, "step"))
			return QNN_SND_CAT_FOOTSTEP;
		return QNN_SND_CAT_FOOTSTEP;
	}

	/* Item pickups */
	if (strstr(name, "items/"))
		return QNN_SND_CAT_PICKUP;

	/* Door/platform/button */
	if (strstr(name, "doors/") || strstr(name, "plats/") ||
	    strstr(name, "buttons/"))
		return QNN_SND_CAT_AMBIENT;

	return QNN_SND_CAT_AMBIENT;
}

/* ---- Engine API stubs --------------------------------------------------- */

void S_Init(void)
{
	Cvar_RegisterVariable(&bgmvolume);
	Cvar_RegisterVariable(&volume);
}

void S_AmbientOff(void)
{
}

void S_AmbientOn(void)
{
}

void S_Shutdown(void)
{
}

void S_TouchSound(char *sample)
{
	(void)sample;
}

void S_ClearBuffer(void)
{
}

void S_StaticSound(sfx_t *sfx, vec3_t origin, float vol, float attenuation)
{
	(void)sfx;
	(void)origin;
	(void)vol;
	(void)attenuation;
}

void S_StartSound(int entnum, int entchannel, sfx_t *sfx, vec3_t origin, float fvol, float attenuation)
{
	qnn_sound_event_t *snd;
	const char *name;

	(void)entchannel;

	if (qnn_sound_count >= QNN_MAX_SOUNDS)
		return;

	/* Determine sound name from sfx pointer offset into our precache pool */
	name = "";
	if (sfx != NULL)
	{
		int index = (int)(sfx - qnn_snd_precache_sfx);
		if (index >= 0 && index < qnn_snd_precache_count)
			name = qnn_snd_precache_names[index];
		else
			name = sfx->name;
	}

	snd = &qnn_sound_buffer[qnn_sound_count];
	VectorCopy(origin, snd->origin);
	snd->volume = fvol;
	snd->attenuation = attenuation;
	snd->entity_num = entnum;
	snd->category = QNN_SndClassify(name);
	strncpy(snd->name, name, sizeof(snd->name) - 1);
	snd->name[sizeof(snd->name) - 1] = '\0';
	qnn_sound_count += 1;
}

void S_StopSound(int entnum, int entchannel)
{
	(void)entnum;
	(void)entchannel;
}

sfx_t *S_PrecacheSound(char *sample)
{
	sfx_t *sfx;

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

void S_ClearPrecache(void)
{
	qnn_snd_precache_count = 0;
}

void S_Update(vec3_t origin, vec3_t v_forward, vec3_t v_right, vec3_t v_up)
{
	(void)origin;
	(void)v_forward;
	(void)v_right;
	(void)v_up;
}

void S_StopAllSounds(qboolean clear)
{
	(void)clear;
	qnn_sound_count = 0;
}

void S_BeginPrecaching(void)
{
}

void S_EndPrecaching(void)
{
}

void S_ExtraUpdate(void)
{
}

void S_LocalSound(char *s)
{
	(void)s;
}
