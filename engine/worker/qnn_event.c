/*
 * qnn_event.c — Auditory perception: sound classification, event
 * routing, and sound capture.
 *
 * Merged from qnn_object.c (sound rule tables + event handlers) and
 * qnn_sound.c (engine S_* stubs + ring buffer capture).
 *
 * All playback is still null (no audio output), but S_PrecacheSound stores
 * sound names and S_StartSound captures spatial sound events into a ring
 * buffer that the worker drains each tick for the observation stream.
 */

#include "qnn_object.h"
#include "qnn_store.h"
#include "qnn_io.h"

#include <ctype.h>
#include <math.h>
#include <string.h>
#include <strings.h>

/* ══════════════════════════════════════════════════════════════════
 * Sound rule tables
 * ══════════════════════════════════════════════════════════════════ */

static const qnn_sound_rule_t qnn_player_sound_rules[] = {
	{"player/h2odeath.wav", QNN_SUBJECT_PLAYER, QNN_ACTION_DEATH, QNN_QUAL_DROWN, 0.0f},
	{"player/plyrjmp8.wav", QNN_SUBJECT_PLAYER, QNN_ACTION_JUMP, QNN_QUAL_NONE, 0.0f},
	{"player/land.wav", QNN_SUBJECT_PLAYER, QNN_ACTION_LAND, QNN_QUAL_NONE, 0.0f},
	{"player/land2.wav", QNN_SUBJECT_PLAYER, QNN_ACTION_LAND, QNN_QUAL_NONE, 1.0f},
	{"player/h2ojump.wav", QNN_SUBJECT_PLAYER, QNN_ACTION_LAND, QNN_QUAL_WATER, 0.0f},
	/* misc/h2ohit1.wav removed — engine plays it for ANY entity hitting
	   water (projectiles, gibs, debris), not just players.  Was creating
	   ghost player objects from teleporter effect entities on DM3. */
	{"player/gasp1.wav", QNN_SUBJECT_PLAYER, QNN_ACTION_BREATH, QNN_QUAL_WATER, 0.0f},
	{"player/gasp2.wav", QNN_SUBJECT_PLAYER, QNN_ACTION_BREATH, QNN_QUAL_WATER, 0.0f},
	{"player/inh2o.wav", QNN_SUBJECT_PLAYER, QNN_ACTION_ENTER, QNN_QUAL_WATER, 0.0f},
	{"misc/outwater.wav", QNN_SUBJECT_PLAYER, QNN_ACTION_EXIT, QNN_QUAL_WATER, 0.0f},
	{"player/inlava.wav", QNN_SUBJECT_PLAYER, QNN_ACTION_ENTER, QNN_QUAL_LAVA, 0.0f},
	{"player/slimbrn2.wav", QNN_SUBJECT_PLAYER, QNN_ACTION_ENTER, QNN_QUAL_SLIME, 0.0f},
	{"player/axhit1.wav", QNN_SUBJECT_AXE, QNN_ACTION_IMPACT, QNN_QUAL_FLESH, 0.0f},
	{"player/axhit2.wav", QNN_SUBJECT_AXE, QNN_ACTION_IMPACT, QNN_QUAL_WORLD, 0.0f},
	{NULL, 0, 0, 0, 0.0f},
};

static const qnn_sound_rule_t qnn_weapon_sound_rules[] = {
	{"weapons/ax1.wav", QNN_SUBJECT_AXE, QNN_ACTION_FIRE, QNN_QUAL_NONE, 0.0f},
	{"weapons/guncock.wav", QNN_SUBJECT_SHOTGUN, QNN_ACTION_FIRE, QNN_QUAL_NONE, 0.0f},
	{"weapons/shotgn2.wav", QNN_SUBJECT_SHOTGUN, QNN_ACTION_FIRE, QNN_QUAL_NONE, 1.0f},
	{"weapons/rocket1i.wav", QNN_SUBJECT_NAILGUN, QNN_ACTION_FIRE, QNN_QUAL_NONE, 0.0f},
	{"weapons/spike2.wav", QNN_SUBJECT_NAILGUN, QNN_ACTION_FIRE, QNN_QUAL_NONE, 1.0f},
	{"weapons/grenade.wav", QNN_SUBJECT_GRENADE_LAUNCHER, QNN_ACTION_FIRE, QNN_QUAL_NONE, 0.0f},
	{"weapons/sgun1.wav", QNN_SUBJECT_ROCKET_LAUNCHER, QNN_ACTION_FIRE, QNN_QUAL_NONE, 0.0f},
	{"weapons/lstart.wav", QNN_SUBJECT_THUNDERBOLT, QNN_ACTION_FIRE, QNN_QUAL_NONE, 0.0f},
	{NULL, 0, 0, 0, 0.0f},
};

static const qnn_sound_rule_t qnn_projectile_sound_rules[] = {
	{"weapons/bounce.wav", QNN_SUBJECT_PROJECTILE_GRENADE, QNN_ACTION_BOUNCE, QNN_QUAL_WORLD, 0.0f},
	{"weapons/tink1.wav", QNN_SUBJECT_PROJECTILE_NAIL, QNN_ACTION_IMPACT, QNN_QUAL_WORLD, 0.0f},
	{"weapons/ric1.wav", QNN_SUBJECT_PROJECTILE_NAIL, QNN_ACTION_IMPACT, QNN_QUAL_WORLD, 0.0f},
	{"weapons/ric2.wav", QNN_SUBJECT_PROJECTILE_NAIL, QNN_ACTION_IMPACT, QNN_QUAL_WORLD, 0.0f},
	{"weapons/ric3.wav", QNN_SUBJECT_PROJECTILE_NAIL, QNN_ACTION_IMPACT, QNN_QUAL_WORLD, 0.0f},
	{"weapons/r_exp3.wav", QNN_SUBJECT_PROJECTILE_ROCKET, QNN_ACTION_IMPACT, QNN_QUAL_WORLD, 0.0f},
	{"weapons/lhit.wav", QNN_SUBJECT_LIGHTNING_BEAM, QNN_ACTION_IMPACT, QNN_QUAL_NONE, 0.0f},
	{NULL, 0, 0, 0, 0.0f},
};

static const qnn_sound_rule_t qnn_static_sound_rules[] = {
	{"misc/r_tele1.wav", QNN_SUBJECT_TELEPORTER, QNN_ACTION_TELEPORT, QNN_QUAL_NONE, 0.0f},
	{"misc/r_tele2.wav", QNN_SUBJECT_TELEPORTER, QNN_ACTION_TELEPORT, QNN_QUAL_NONE, 0.0f},
	{"misc/r_tele3.wav", QNN_SUBJECT_TELEPORTER, QNN_ACTION_TELEPORT, QNN_QUAL_NONE, 0.0f},
	{"misc/r_tele4.wav", QNN_SUBJECT_TELEPORTER, QNN_ACTION_TELEPORT, QNN_QUAL_NONE, 0.0f},
	{"misc/r_tele5.wav", QNN_SUBJECT_TELEPORTER, QNN_ACTION_TELEPORT, QNN_QUAL_NONE, 0.0f},
	{"doors/medtry.wav", QNN_SUBJECT_DOOR, QNN_ACTION_REJECT, QNN_QUAL_KEYED, 0.0f},
	{"doors/runetry.wav", QNN_SUBJECT_DOOR, QNN_ACTION_REJECT, QNN_QUAL_KEYED, 0.0f},
	{"doors/basetry.wav", QNN_SUBJECT_DOOR, QNN_ACTION_REJECT, QNN_QUAL_KEYED, 0.0f},
	{"doors/meduse.wav", QNN_SUBJECT_DOOR, QNN_ACTION_ACTIVATE, QNN_QUAL_KEYED, 0.0f},
	{"doors/runeuse.wav", QNN_SUBJECT_DOOR, QNN_ACTION_ACTIVATE, QNN_QUAL_KEYED, 0.0f},
	{"doors/baseuse.wav", QNN_SUBJECT_DOOR, QNN_ACTION_ACTIVATE, QNN_QUAL_KEYED, 0.0f},
	{"doors/drclos4.wav", QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"doors/doormv1.wav", QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"doors/hydro1.wav", QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"doors/hydro2.wav", QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"doors/stndr1.wav", QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"doors/stndr2.wav", QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"doors/ddoor1.wav", QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"doors/ddoor2.wav", QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"doors/latch2.wav", QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_SECRET, 0.0f},
	{"doors/winch2.wav", QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_SECRET, 0.0f},
	{"doors/airdoor1.wav", QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_SECRET, 0.0f},
	{"doors/airdoor2.wav", QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_SECRET, 0.0f},
	{"doors/basesec1.wav", QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_SECRET, 0.0f},
	{"doors/basesec2.wav", QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_SECRET, 0.0f},
	{"plats/plat1.wav", QNN_SUBJECT_PLATFORM, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"plats/plat2.wav", QNN_SUBJECT_PLATFORM, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"plats/medplat1.wav", QNN_SUBJECT_PLATFORM, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"plats/medplat2.wav", QNN_SUBJECT_PLATFORM, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"plats/train1.wav", QNN_SUBJECT_TRAIN, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"plats/train2.wav", QNN_SUBJECT_TRAIN, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"buttons/airbut1.wav", QNN_SUBJECT_BUTTON, QNN_ACTION_ACTIVATE, QNN_QUAL_NONE, 0.0f},
	{"buttons/switch21.wav", QNN_SUBJECT_BUTTON, QNN_ACTION_ACTIVATE, QNN_QUAL_NONE, 0.0f},
	{"buttons/switch02.wav", QNN_SUBJECT_BUTTON, QNN_ACTION_ACTIVATE, QNN_QUAL_NONE, 0.0f},
	{"buttons/switch04.wav", QNN_SUBJECT_BUTTON, QNN_ACTION_ACTIVATE, QNN_QUAL_NONE, 0.0f},
	{NULL, 0, 0, 0, 0.0f},
};

/* ══════════════════════════════════════════════════════════════════
 * Sound classification helpers
 * ══════════════════════════════════════════════════════════════════ */

static int QNN_SoundPickupSubject(const char *name)
{
	if (!strcmp(name, "items/r_item1.wav") || !strcmp(name, "items/health1.wav"))
		return QNN_SUBJECT_HEALTH;
	if (!strcmp(name, "items/r_item2.wav"))
		return QNN_SUBJECT_MEGAHEALTH;
	if (!strcmp(name, "items/armor1.wav"))
		return QNN_SUBJECT_ARMOR_GREEN;	/* generic — can't distinguish tier from sound */
	if (!strcmp(name, "weapons/pkup.wav"))
		return QNN_SUBJECT_BACKPACK;	/* weapon pickup — generic */
	if (!strcmp(name, "weapons/lock4.wav"))
		return QNN_SUBJECT_BACKPACK;	/* ammo pickup — generic */
	if (!strcmp(name, "items/damage.wav"))
		return QNN_SUBJECT_QUAD;
	if (!strcmp(name, "items/protect.wav"))
		return QNN_SUBJECT_PENT;
	if (!strcmp(name, "items/inv1.wav"))
		return QNN_SUBJECT_RING;
	if (!strcmp(name, "items/suit.wav"))
		return QNN_SUBJECT_SUIT;
	return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * Sound event classification — produces qnn_event_record_t records
 * ══════════════════════════════════════════════════════════════════ */

static const qnn_sound_rule_t *QNN_FindSoundRule(const qnn_sound_rule_t *rules, const char *name)
{
	int i;
	for (i = 0; rules[i].name != NULL; ++i)
	{
		if (!strcmp(rules[i].name, name))
			return &rules[i];
	}
	return NULL;
}

static qboolean QNN_EmitRecord(qnn_event_record_t *out, int *count, int max,
	const qnn_sound_event_t *snd, int subject_id, int action_id, int qualifier_id, float magnitude)
{
	if (*count >= max)
		return false;
	memset(&out[*count], 0, sizeof(out[*count]));
	out[*count].entity_num = snd->entity_num;
	out[*count].subject_id = subject_id;
	out[*count].action_id = action_id;
	out[*count].qualifier_id = qualifier_id;
	out[*count].magnitude = magnitude;
	VectorCopy(snd->origin, out[*count].origin);
	(*count)++;
	return true;
}

int QNN_EventClassifySounds(const qnn_snapshot_t *snapshot, qnn_event_record_t *out, int max)
{
	int count = 0;
	int i;

	for (i = 0; i < snapshot->sound_count; ++i)
	{
		char name[QNN_MAX_SOUND_NAME];
		const qnn_sound_rule_t *rule;
		int j;

		name[0] = 0;
		for (j = 0; snapshot->sounds[i].name[j] && j < QNN_MAX_SOUND_NAME - 1; ++j)
			name[j] = (char)tolower((unsigned char)snapshot->sounds[i].name[j]);
		name[j] = 0;
		if (name[0] == 0)
			continue;

		/* Item respawn */
		if (!strcmp(name, "items/itembk2.wav"))
		{
			if (count < max)
			{
				memset(&out[count], 0, sizeof(out[count]));
				out[count].entity_num = snapshot->sounds[i].entity_num;
				VectorCopy(snapshot->sounds[i].origin, out[count].origin);
				out[count].action_id = QNN_ACTION_RESPAWN;
				out[count].is_respawn = true;
				count++;
			}
			continue;
		}

		/* Item pickup */
		{
			int pickup_sub = QNN_SoundPickupSubject(name);
			int pickup_cat = pickup_sub > 0 ? QNN_SubjectPickupCategory(pickup_sub) : 0;
			if (pickup_cat > 0 && snapshot->sounds[i].entity_num > 0)
			{
				int sub = QNN_SoundPickupSubject(name);
				if (sub > 0 && count < max)
				{
					memset(&out[count], 0, sizeof(out[count]));
					out[count].entity_num = snapshot->sounds[i].entity_num;
					out[count].subject_id = sub;
					out[count].action_id = QNN_ACTION_PICKUP;
					out[count].pickup_category = pickup_cat;
					VectorCopy(snapshot->sounds[i].origin, out[count].origin);
					count++;
				}
				continue;
			}
		}

		/* Powerup hum/warning */
		{
			int pu_sub = 0, pu_act = 0;
			if (!strcmp(name, "items/damage3.wav"))       { pu_sub = QNN_SUBJECT_QUAD; pu_act = QNN_ACTION_ACTIVE; }
			else if (!strcmp(name, "items/protect3.wav"))  { pu_sub = QNN_SUBJECT_PENT; pu_act = QNN_ACTION_ACTIVE; }
			else if (!strcmp(name, "items/inv3.wav"))      { pu_sub = QNN_SUBJECT_RING; pu_act = QNN_ACTION_ACTIVE; }
			else if (!strcmp(name, "items/damage2.wav"))   { pu_sub = QNN_SUBJECT_QUAD; pu_act = QNN_ACTION_WARNING; }
			else if (!strcmp(name, "items/protect2.wav"))  { pu_sub = QNN_SUBJECT_PENT; pu_act = QNN_ACTION_WARNING; }
			else if (!strcmp(name, "items/inv2.wav"))      { pu_sub = QNN_SUBJECT_RING; pu_act = QNN_ACTION_WARNING; }
			else if (!strcmp(name, "items/suit2.wav"))     { pu_sub = QNN_SUBJECT_SUIT; pu_act = QNN_ACTION_WARNING; }
			if (pu_sub > 0)
			{
				if (count < max)
				{
					memset(&out[count], 0, sizeof(out[count]));
					out[count].entity_num = snapshot->sounds[i].entity_num;
					out[count].subject_id = pu_sub;
					out[count].action_id = pu_act;
					out[count].powerup_subject_id = pu_sub;
					VectorCopy(snapshot->sounds[i].origin, out[count].origin);
					count++;
				}
				continue;
			}
		}

		/* Player sounds (rule table + pattern matches) */
		rule = QNN_FindSoundRule(qnn_player_sound_rules, name);
		if (rule != NULL)
		{
			QNN_EmitRecord(out, &count, max, &snapshot->sounds[i], rule->subject_id, rule->action_id, rule->qualifier_id, rule->magnitude);
			continue;
		}
		if (!strncmp(name, "player/pain", 11))
		{
			QNN_EmitRecord(out, &count, max, &snapshot->sounds[i], QNN_SUBJECT_PLAYER, QNN_ACTION_PAIN, QNN_QUAL_NONE, 0.0f);
			continue;
		}
		if (!strncmp(name, "player/drown", 12))
		{
			QNN_EmitRecord(out, &count, max, &snapshot->sounds[i], QNN_SUBJECT_PLAYER, QNN_ACTION_PAIN, QNN_QUAL_DROWN, 0.0f);
			continue;
		}
		if (!strncmp(name, "player/lburn", 12))
		{
			QNN_EmitRecord(out, &count, max, &snapshot->sounds[i], QNN_SUBJECT_PLAYER, QNN_ACTION_PAIN, QNN_QUAL_LAVA, 0.0f);
			continue;
		}
		if (!strncmp(name, "player/death", 12))
		{
			QNN_EmitRecord(out, &count, max, &snapshot->sounds[i], QNN_SUBJECT_PLAYER, QNN_ACTION_DEATH, QNN_QUAL_NONE, 0.0f);
			continue;
		}
		if (!strcmp(name, "player/gib.wav") || !strcmp(name, "player/udeath.wav"))
		{
			QNN_EmitRecord(out, &count, max, &snapshot->sounds[i], QNN_SUBJECT_PLAYER, QNN_ACTION_DEATH, QNN_QUAL_NONE, 1.0f);
			continue;
		}
		if (!strcmp(name, "player/tornoff2.wav"))
		{
			/* tornoff2.wav is only played by ClientDisconnect — always a disconnect. */
			QNN_EmitRecord(out, &count, max, &snapshot->sounds[i], QNN_SUBJECT_PLAYER, QNN_ACTION_DISCONNECT, QNN_QUAL_NONE, 0.0f);
			continue;
		}

		/* Weapon fire */
		rule = QNN_FindSoundRule(qnn_weapon_sound_rules, name);
		if (rule != NULL)
		{
			if (count < max)
			{
				QNN_EmitRecord(out, &count, max, &snapshot->sounds[i], rule->subject_id, rule->action_id, rule->qualifier_id, rule->magnitude);
				out[count - 1].weapon_subject_id = rule->subject_id;
			}
			continue;
		}

		/* Projectile impact/bounce */
		rule = QNN_FindSoundRule(qnn_projectile_sound_rules, name);
		if (rule != NULL)
		{
			QNN_EmitRecord(out, &count, max, &snapshot->sounds[i], rule->subject_id, rule->action_id, rule->qualifier_id, rule->magnitude);
			continue;
		}

		/* Static objects (doors, platforms, teleporters) */
		rule = QNN_FindSoundRule(qnn_static_sound_rules, name);
		if (rule != NULL)
			QNN_EmitRecord(out, &count, max, &snapshot->sounds[i], rule->subject_id, rule->action_id, rule->qualifier_id, rule->magnitude);
	}

	return count;
}

/* ══════════════════════════════════════════════════════════════════
 * Sound capture (merged from qnn_sound.c)
 *
 * Replaces snd_null.c for the Quake worker.  All playback is null
 * (no audio output), but S_PrecacheSound stores sound names and
 * S_StartSound captures spatial sound events into a ring buffer.
 * ══════════════════════════════════════════════════════════════════ */

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

void QNN_DrainSounds(qnn_snapshot_t *snapshot)
{
	snapshot->sound_count = qnn_sound_count < QNN_MAX_SOUNDS
		? qnn_sound_count : QNN_MAX_SOUNDS;
	if (snapshot->sound_count > 0)
		memcpy(snapshot->sounds, qnn_sound_buffer, snapshot->sound_count * sizeof(qnn_sound_event_t));
	qnn_sound_count = 0;
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

/* ══════════════════════════════════════════════════════════════════
 * Event atom pool — persistent per-entity event history.
 *
 * Atoms are created from classified sound records.  They decay
 * over QNN_RECENCY_DECAY_S and are read by the oracle at emission
 * time to attach per-entity event tokens.
 * ══════════════════════════════════════════════════════════════════ */

qnn_semantic_event_atom_t qnn_semantic_events[QNN_MAX_EVENT_ATOMS];
int qnn_event_head[QNN_EVENT_HEAD_CAPACITY];
int qnn_prev_object_indices[QNN_MAX_TOKEN_OBJECTS];
int qnn_prev_object_count = 0;

static void QNN_AppendEvent(int owner_index, int subject_id, int action_id, int qualifier_id, int modality_id, float confidence, float magnitude)
{
	int i;
	int free_index;

	free_index = -1;
	for (i = 0; i < QNN_MAX_EVENT_ATOMS; ++i)
	{
		if (!qnn_semantic_events[i].active)
		{
			if (free_index < 0)
				free_index = i;
			continue;
		}
		if (qnn_semantic_events[i].owner_index == owner_index
			&& qnn_semantic_events[i].subject_id == subject_id
			&& qnn_semantic_events[i].action_id == action_id
			&& qnn_semantic_events[i].qualifier_id == qualifier_id
			&& qnn_semantic_events[i].modality_id == modality_id)
		{
			qnn_semantic_events[i].recency = 1.0f;
			if (confidence > qnn_semantic_events[i].confidence)
				qnn_semantic_events[i].confidence = confidence;
			if (magnitude > qnn_semantic_events[i].magnitude)
				qnn_semantic_events[i].magnitude = magnitude;
			return;
		}
	}
	if (free_index < 0)
		return;
	memset(&qnn_semantic_events[free_index], 0, sizeof(qnn_semantic_events[free_index]));
	qnn_semantic_events[free_index].active = true;
	qnn_semantic_events[free_index].owner_index = owner_index;
	qnn_semantic_events[free_index].subject_id = subject_id;
	qnn_semantic_events[free_index].action_id = action_id;
	qnn_semantic_events[free_index].qualifier_id = qualifier_id;
	qnn_semantic_events[free_index].modality_id = modality_id;
	qnn_semantic_events[free_index].recency = 1.0f;
	qnn_semantic_events[free_index].confidence = confidence;
	qnn_semantic_events[free_index].magnitude = magnitude;

	qnn_semantic_events[free_index].next_for_owner = -1;
	if (owner_index >= 0 && owner_index < QNN_EVENT_HEAD_CAPACITY)
	{
		qnn_semantic_events[free_index].next_for_owner = qnn_event_head[owner_index];
		qnn_event_head[owner_index] = free_index;
	}
}

static void QNN_DecayEvents(float dt)
{
	int i;

	for (i = 0; i < QNN_MAX_EVENT_ATOMS; ++i)
	{
		if (!qnn_semantic_events[i].active)
			continue;
		qnn_semantic_events[i].recency = QNN_Clamp(qnn_semantic_events[i].recency - (dt / QNN_RECENCY_DECAY_S), 0.0f, 1.0f);
		if (qnn_semantic_events[i].recency <= 0.0f)
			qnn_semantic_events[i].active = false;
	}

	{
		int cap = QNN_StoreSize() < QNN_EVENT_HEAD_CAPACITY
			? QNN_StoreSize() : QNN_EVENT_HEAD_CAPACITY;
		memset(qnn_event_head, -1, (size_t)cap * sizeof(int));
		for (i = 0; i < QNN_MAX_EVENT_ATOMS; ++i)
		{
			int oi;
			if (!qnn_semantic_events[i].active)
				continue;
			oi = qnn_semantic_events[i].owner_index;
			if (oi >= 0 && oi < QNN_EVENT_HEAD_CAPACITY)
			{
				qnn_semantic_events[i].next_for_owner = qnn_event_head[oi];
				qnn_event_head[oi] = i;
			}
		}
	}
}

/* ══════════════════════════════════════════════════════════════════
 * Recall
 * ══════════════════════════════════════════════════════════════════ */

static void QNN_ApplyRecall(const qnn_action_t *action, int prev_count, int *prev_indices)
{
	int r;
	for (r = 0; r < 4; ++r)
	{
		int target = action->recall[r];
		int idx;
		if (target <= 0 || target > prev_count)
			continue;
		idx = prev_indices[target - 1];
		if (idx >= 0 && idx < QNN_StoreSize() && qnn_store[idx].active)
			qnn_store[idx].mem = (float)cl.mtime[0];
	}
}

/* ══════════════════════════════════════════════════════════════════
 * Per-tick event processing.
 *
 * Classifies sounds once, then:
 *   1. Updates store state (items, movers, actors) from records
 *   2. Creates event atoms from the same records
 *
 * Single consumer of QNN_EventClassifySounds — the store no longer
 * calls the classifier directly.
 * ══════════════════════════════════════════════════════════════════ */

static void QNN_EventProcessTick(const qnn_snapshot_t *snapshot)
{
	qnn_event_record_t event_records[QNN_MAX_EVENT_RECORDS];
	int event_count, i;
	float now = (float)cl.mtime[0];

	event_count = QNN_EventClassifySounds(snapshot, event_records, QNN_MAX_EVENT_RECORDS);

	for (i = 0; i < event_count; ++i)
	{
		qnn_event_record_t *ev = &event_records[i];

		/* ---- Item respawn ---- */
		if (ev->is_respawn)
		{
			if (ev->entity_num > 0 && ev->entity_num < MAX_EDICTS)
			{
				qnn_entity_t *e = &qnn_store[ev->entity_num];
				if (e->active && e->type == QNN_ENT_ITEM)
				{
					e->regen = 0.0f;
					e->snd = now;
					QNN_AppendEvent(ev->entity_num, e->subject_id,
						QNN_ACTION_RESPAWN, QNN_QUAL_NONE, QNN_MODALITY_SOUND, 1.0f, 0.0f);
				}
			}
			continue;
		}

		/* ---- Item pickup ---- */
		if (ev->pickup_category > 0)
		{
			int j;
			float best_dsq = QNN_ITEM_PICKUP_PLAYER_SQ;
			qnn_entity_t *best = NULL;
			int best_idx = -1;

			for (j = 1; j < MAX_EDICTS; ++j)
			{
				qnn_entity_t *e = &qnn_store[j];
				float dsq;
				if (!e->active || e->type != QNN_ENT_ITEM || e->regen > 0.0f)
					continue;
				if (QNN_SubjectPickupCategory(e->subject_id) != ev->pickup_category)
					continue;
				dsq = QNN_DistSq(e->origin, ev->origin);
				if (dsq < best_dsq)
				{
					best = e;
					best_dsq = dsq;
					best_idx = j;
				}
			}
			if (best)
			{
				best->regen = best->regen_time;
				best->snd = now;
			}
			if (ev->entity_num > 0)
				QNN_AppendEvent(ev->entity_num, ev->subject_id, QNN_ACTION_PICKUP,
					ev->qualifier_id, QNN_MODALITY_SOUND, 1.0f, ev->magnitude);
			continue;
		}

		/* ---- Player disconnect ---- */
		if (ev->action_id == QNN_ACTION_DISCONNECT)
		{
			if (ev->entity_num > 0 && ev->entity_num < MAX_EDICTS
				&& qnn_store[ev->entity_num].active
				&& qnn_store[ev->entity_num].type == QNN_ENT_ACTOR)
				memset(&qnn_store[ev->entity_num], 0, sizeof(qnn_store[ev->entity_num]));
			continue;
		}

		/* ---- Player-attributed sounds ---- */
		if (ev->entity_num > 0 && ev->entity_num <= cl.maxclients)
		{
			int owner_subject = ev->subject_id;
			qnn_entity_t *e;

			if (ev->weapon_subject_id > 0
				|| ev->subject_id == QNN_SUBJECT_PLAYER
				|| ev->powerup_subject_id > 0)
				owner_subject = QNN_SUBJECT_PLAYER;
			if (owner_subject != QNN_SUBJECT_PLAYER)
				continue;
			if (ev->entity_num == cl.viewentity)
				continue;
			if (cl.scores != NULL && cl.scores[ev->entity_num - 1].name[0] == '\0')
				continue;

			e = &qnn_store[ev->entity_num];
			e->snd = now;

			if (e->pvs < now - 0.001f)
			{
				e->active = true;
				e->type = QNN_ENT_ACTOR;
				e->subject_id = QNN_SUBJECT_PLAYER;
				e->entity_num = ev->entity_num;
				VectorCopy(ev->origin, e->origin);
			}

			if (ev->weapon_subject_id > 0)
				e->weapon_subject_id = ev->weapon_subject_id;
			if (ev->powerup_subject_id > 0)
			{
				e->powerup_subject_id = ev->powerup_subject_id;
				if (ev->action_id == QNN_ACTION_WARNING)
					e->powerup_warning_elapsed = 0.001f;
				else if (ev->action_id == QNN_ACTION_ACTIVE)
					e->powerup_warning_elapsed = 0.0f;
			}
			if (ev->action_id == QNN_ACTION_DEATH)
			{
				e->powerup_subject_id = 0;
				e->powerup_warning_elapsed = 0.0f;
				e->weapon_subject_id = 0;
			}

			QNN_AppendEvent(ev->entity_num, ev->subject_id, ev->action_id,
				ev->qualifier_id, QNN_MODALITY_SOUND, 1.0f, ev->magnitude);
			continue;
		}

		/* ---- Mover sounds (entity_num <= 0, spatial match) ---- */
		if (ev->entity_num <= 0)
		{
			int j;
			float best_dsq = QNN_STATIC_SOUND_MATCH_SQ;
			qnn_entity_t *best = NULL;
			int best_idx = -1;

			for (j = 1; j < MAX_EDICTS; ++j)
			{
				qnn_entity_t *e = &qnn_store[j];
				float dsq;
				if (!e->active || e->type != QNN_ENT_MOVER || e->subject_id != ev->subject_id)
					continue;
				dsq = QNN_DistSq(e->origin, ev->origin);
				if (dsq < best_dsq)
				{
					best = e;
					best_dsq = dsq;
					best_idx = j;
				}
			}
			if (best)
			{
				best->snd = now;
				if (best->pvs < now - 0.001f)
				{
					if (ev->subject_id == QNN_SUBJECT_DOOR
						|| ev->subject_id == QNN_SUBJECT_PLATFORM
						|| ev->subject_id == QNN_SUBJECT_TRAIN)
						best->state = (best->state >= 1.0f) ? 0.0f : 1.0f;
					else if (ev->subject_id == QNN_SUBJECT_BUTTON)
						best->state = 1.0f;
				}
				QNN_AppendEvent(best_idx, ev->subject_id, ev->action_id,
					ev->qualifier_id, QNN_MODALITY_SOUND, 1.0f, ev->magnitude);
			}
		}
	}
}

/* ══════════════════════════════════════════════════════════════════
 * Public API — called from qnn_io.c
 * ══════════════════════════════════════════════════════════════════ */

void QNN_EventInit(const qnn_map_state_t *map_state)
{
	memset(qnn_semantic_events, 0, sizeof(qnn_semantic_events));
	memset(qnn_event_head, -1, sizeof(qnn_event_head));
	qnn_prev_object_count = 0;
	QNN_StoreInit(map_state);
}

void QNN_EventTick(const qnn_snapshot_t *snapshot, float dt, qboolean reset_flag)
{
	float sdt;

	if (reset_flag)
	{
		memset(qnn_semantic_events, 0, sizeof(qnn_semantic_events));
		memset(qnn_event_head, -1, sizeof(qnn_event_head));
	}

	QNN_DecayEvents(dt);

	sdt = (float)(cl.mtime[0] - cl.mtime[1]);
	if (sdt < 0.001f || sdt > 0.5f)
		sdt = dt;
	QNN_StoreUpdate(snapshot, sdt);

	QNN_EventProcessTick(snapshot);

	if (!reset_flag)
		QNN_ApplyRecall(&snapshot->action_label, qnn_prev_object_count, qnn_prev_object_indices);
}
