/*
 * qnn_event.c — Auditory perception: sound classification, event
 * routing, and the per-entity event atom pool.
 *
 * Shared by the NQ and QW demo workers.  All engine-specific state
 * (self entity, scoreboard, server time) is accessed through fields
 * that exist natively on NQ (cl.viewentity, cl.scores, cl.mtime[0])
 * and are synthesized on QW by QNN_SyncEngineCompat() each frame — see
 * qw/qnn_compat.h and qnn_engine_compat.c.
 *
 * The engine-facing S_* shim and the shared sound ring buffer live
 * in common/qnn_sound.c; this file owns only the tick-time pipeline:
 * classify → route to owner → append atom → expire old atoms.
 */

#include "qnn_object.h"
#include "qnn_store.h"
#include "qnn_io.h"
#include "qnn_demo_sounds.h"
#include "qnn_context.h"

/* Defined in common/qnn_sound.c */
extern FILE *qnn_sound_dump;

#include <ctype.h>
#include <math.h>
#include <string.h>
#include <strings.h>

/* ══════════════════════════════════════════════════════════════════
 * Sound rule tables
 * ══════════════════════════════════════════════════════════════════ */

static const qnn_sound_rule_t qnn_player_sound_rules[] = {
	/* Death */
	{"player/death1.wav",   QNN_ACTION_DEATH, QNN_SOURCE_NONE, 0},
	{"player/death2.wav",   QNN_ACTION_DEATH, QNN_SOURCE_NONE, 0},
	{"player/death3.wav",   QNN_ACTION_DEATH, QNN_SOURCE_NONE, 0},
	{"player/death4.wav",   QNN_ACTION_DEATH, QNN_SOURCE_NONE, 0},
	{"player/death5.wav",   QNN_ACTION_DEATH, QNN_SOURCE_NONE, 0},
	{"player/teledth1.wav", QNN_ACTION_DEATH, QNN_SOURCE_NONE, 0},
	{"player/gib.wav",      QNN_ACTION_DEATH, QNN_SOURCE_GIB, 0},
	{"player/udeath.wav",   QNN_ACTION_DEATH, QNN_SOURCE_GIB, 0},
	{"player/h2odeath.wav", QNN_ACTION_DEATH, QNN_SOURCE_WATER, 0},
	/* Pain */
	{"player/pain1.wav",    QNN_ACTION_PAIN, QNN_SOURCE_NONE, 0},
	{"player/pain2.wav",    QNN_ACTION_PAIN, QNN_SOURCE_NONE, 0},
	{"player/pain3.wav",    QNN_ACTION_PAIN, QNN_SOURCE_NONE, 0},
	{"player/pain4.wav",    QNN_ACTION_PAIN, QNN_SOURCE_NONE, 0},
	{"player/pain5.wav",    QNN_ACTION_PAIN, QNN_SOURCE_NONE, 0},
	{"player/pain6.wav",    QNN_ACTION_PAIN, QNN_SOURCE_NONE, 0},
	{"player/drown1.wav",   QNN_ACTION_PAIN, QNN_SOURCE_WATER, 0},
	{"player/drown2.wav",   QNN_ACTION_PAIN, QNN_SOURCE_WATER, 0},
	{"player/lburn1.wav",   QNN_ACTION_PAIN, QNN_SOURCE_LAVA, 0},
	{"player/lburn2.wav",   QNN_ACTION_PAIN, QNN_SOURCE_LAVA, 0},
	{"player/axhit1.wav",   QNN_ACTION_PAIN, QNN_SUBJECT_AXE, 0},
	/* Movement — jump path from qnn_demo_sounds.h (shared with classifier). */
#define X(path) {path, QNN_ACTION_JUMP, QNN_SOURCE_NONE, 0},
	QNN_JUMP_SOUND_LIST(X)
#undef X
	{"player/land.wav",     QNN_ACTION_LAND, QNN_SOURCE_NONE, 0},
	{"player/land2.wav",    QNN_ACTION_LAND, QNN_SOURCE_NONE, 0},
	{"player/h2ojump.wav",  QNN_ACTION_LAND, QNN_SOURCE_WATER, 0},
	{"player/gasp1.wav",    QNN_ACTION_BREATH, QNN_SOURCE_WATER, 0},
	{"player/gasp2.wav",    QNN_ACTION_BREATH, QNN_SOURCE_WATER, 0},
	{"player/inh2o.wav",    QNN_ACTION_ENTER, QNN_SOURCE_WATER, 0},
	{"misc/outwater.wav",   QNN_ACTION_EXIT, QNN_SOURCE_WATER, 0},
	{"player/inlava.wav",   QNN_ACTION_ENTER, QNN_SOURCE_LAVA, 0},
	{"player/slimbrn2.wav", QNN_ACTION_ENTER, QNN_SOURCE_SLIME, 0},
	/* Disconnect */
	{"player/tornoff2.wav", QNN_ACTION_DISCONNECT, QNN_SOURCE_NONE, 0},
	{NULL, 0, 0, 0},
};

/* Weapon-fire sound rules: paths + subject metadata sourced from
 * qnn_demo_sounds.h (shared with src/demo/qw_classifier.c). */
static const qnn_sound_rule_t qnn_weapon_sound_rules[] = {
#define X(path, subject) {path, QNN_ACTION_ATTACK, subject, 0},
	QNN_ATTACK_SOUND_LIST(X)
#undef X
	{NULL, 0, 0, 0},
};

static const qnn_sound_rule_t qnn_projectile_sound_rules[] = {
	{"weapons/bounce.wav", QNN_ACTION_BOUNCE, QNN_SOURCE_NONE, 0},
	{NULL, 0, 0, 0},
};

static const qnn_sound_rule_t qnn_static_sound_rules[] = {
	{"misc/r_tele1.wav", QNN_ACTION_TELEPORT, QNN_SOURCE_NONE, QNN_SUBJECT_TELEPORTER},
	{"misc/r_tele2.wav", QNN_ACTION_TELEPORT, QNN_SOURCE_NONE, QNN_SUBJECT_TELEPORTER},
	{"misc/r_tele3.wav", QNN_ACTION_TELEPORT, QNN_SOURCE_NONE, QNN_SUBJECT_TELEPORTER},
	{"misc/r_tele4.wav", QNN_ACTION_TELEPORT, QNN_SOURCE_NONE, QNN_SUBJECT_TELEPORTER},
	{"misc/r_tele5.wav", QNN_ACTION_TELEPORT, QNN_SOURCE_NONE, QNN_SUBJECT_TELEPORTER},
	{"doors/medtry.wav", QNN_ACTION_REJECT, QNN_SOURCE_KEYED, QNN_SUBJECT_DOOR},
	{"doors/runetry.wav", QNN_ACTION_REJECT, QNN_SOURCE_KEYED, QNN_SUBJECT_DOOR},
	{"doors/basetry.wav", QNN_ACTION_REJECT, QNN_SOURCE_KEYED, QNN_SUBJECT_DOOR},
	{"doors/meduse.wav", QNN_ACTION_ACTIVATE, QNN_SOURCE_KEYED, QNN_SUBJECT_DOOR},
	{"doors/runeuse.wav", QNN_ACTION_ACTIVATE, QNN_SOURCE_KEYED, QNN_SUBJECT_DOOR},
	{"doors/baseuse.wav", QNN_ACTION_ACTIVATE, QNN_SOURCE_KEYED, QNN_SUBJECT_DOOR},
	{"doors/drclos4.wav", QNN_ACTION_MOVE, QNN_SOURCE_NONE, QNN_SUBJECT_DOOR},
	{"doors/doormv1.wav", QNN_ACTION_MOVE, QNN_SOURCE_NONE, QNN_SUBJECT_DOOR},
	{"doors/hydro1.wav", QNN_ACTION_MOVE, QNN_SOURCE_NONE, QNN_SUBJECT_DOOR},
	{"doors/hydro2.wav", QNN_ACTION_MOVE, QNN_SOURCE_NONE, QNN_SUBJECT_DOOR},
	{"doors/stndr1.wav", QNN_ACTION_MOVE, QNN_SOURCE_NONE, QNN_SUBJECT_DOOR},
	{"doors/stndr2.wav", QNN_ACTION_MOVE, QNN_SOURCE_NONE, QNN_SUBJECT_DOOR},
	{"doors/ddoor1.wav", QNN_ACTION_MOVE, QNN_SOURCE_NONE, QNN_SUBJECT_DOOR},
	{"doors/ddoor2.wav", QNN_ACTION_MOVE, QNN_SOURCE_NONE, QNN_SUBJECT_DOOR},
	{"doors/latch2.wav", QNN_ACTION_MOVE, QNN_SOURCE_SECRET, QNN_SUBJECT_DOOR},
	{"doors/winch2.wav", QNN_ACTION_MOVE, QNN_SOURCE_SECRET, QNN_SUBJECT_DOOR},
	{"doors/airdoor1.wav", QNN_ACTION_MOVE, QNN_SOURCE_SECRET, QNN_SUBJECT_DOOR},
	{"doors/airdoor2.wav", QNN_ACTION_MOVE, QNN_SOURCE_SECRET, QNN_SUBJECT_DOOR},
	{"doors/basesec1.wav", QNN_ACTION_MOVE, QNN_SOURCE_SECRET, QNN_SUBJECT_DOOR},
	{"doors/basesec2.wav", QNN_ACTION_MOVE, QNN_SOURCE_SECRET, QNN_SUBJECT_DOOR},
	{"plats/plat1.wav", QNN_ACTION_MOVE, QNN_SOURCE_NONE, QNN_SUBJECT_PLATFORM},
	{"plats/plat2.wav", QNN_ACTION_MOVE, QNN_SOURCE_NONE, QNN_SUBJECT_PLATFORM},
	{"plats/medplat1.wav", QNN_ACTION_MOVE, QNN_SOURCE_NONE, QNN_SUBJECT_PLATFORM},
	{"plats/medplat2.wav", QNN_ACTION_MOVE, QNN_SOURCE_NONE, QNN_SUBJECT_PLATFORM},
	{"plats/train1.wav", QNN_ACTION_MOVE, QNN_SOURCE_NONE, QNN_SUBJECT_TRAIN},
	{"plats/train2.wav", QNN_ACTION_MOVE, QNN_SOURCE_NONE, QNN_SUBJECT_TRAIN},
	{"buttons/airbut1.wav", QNN_ACTION_ACTIVATE, QNN_SOURCE_NONE, QNN_SUBJECT_BUTTON},
	{"buttons/switch21.wav", QNN_ACTION_ACTIVATE, QNN_SOURCE_NONE, QNN_SUBJECT_BUTTON},
	{"buttons/switch02.wav", QNN_ACTION_ACTIVATE, QNN_SOURCE_NONE, QNN_SUBJECT_BUTTON},
	{"buttons/switch04.wav", QNN_ACTION_ACTIVATE, QNN_SOURCE_NONE, QNN_SUBJECT_BUTTON},
	{NULL, 0, 0, 0},
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

static qboolean QNN_SoundMatchesAction(const qnn_sound_rule_t *rules, const char *name, int action_id)
{
	const qnn_sound_rule_t *rule = QNN_FindSoundRule(rules, name);
	return (rule != NULL && rule->action_id == action_id) ? true : false;
}

static qboolean QNN_SnapshotHasSelfActionSound(const qnn_snapshot_t *snapshot,
	const qnn_sound_rule_t *rules, int action_id)
{
	int i;

	for (i = 0; i < snapshot->sound_count; ++i)
	{
		const qnn_sound_event_t *sound = &snapshot->sounds[i];
		if (sound->entity_num != cl.viewentity)
			continue;
		if (QNN_SoundMatchesAction(rules, sound->name, action_id))
			return true;
	}
	return false;
}

qboolean QNN_SnapshotHasSelfWeaponAttackSound(const qnn_snapshot_t *snapshot)
{
	return QNN_SnapshotHasSelfActionSound(snapshot, qnn_weapon_sound_rules, QNN_ACTION_ATTACK);
}

/* Per-sound check: true iff `sound` is a weapon-fire multicast for
 * the self entity.  Same rule set as QNN_SnapshotHasSelfWeaponAttackSound
 * but evaluated one sound at a time so the caller can use the sound's
 * native_time for per-event back-shift decisions. */
qboolean QNN_IsSelfWeaponAttackSound(const qnn_sound_event_t *sound)
{
	if (sound->entity_num != cl.viewentity)
		return false;
	return QNN_SoundMatchesAction(qnn_weapon_sound_rules,
		sound->name, QNN_ACTION_ATTACK);
}

/* Raw weapon id (1..8) for a self weapon-fire sound, else
 * QNN_WEAPON_NONE.  The fire rule table carries each sound's subject
 * (qnn_demo_sounds.h); map it to the raw weapon id so the MVD fire
 * back-shift can attribute the shot's weapon from the SOUND itself —
 * the demo's own byte truth — rather than the held-weapon snapshot at a
 * separately-shifted frame. */
int QNN_WeaponIdFromAttackSound(const qnn_sound_event_t *sound)
{
	const qnn_sound_rule_t *rule;

	if (sound->entity_num != cl.viewentity)
		return QNN_WEAPON_NONE;
	rule = QNN_FindSoundRule(qnn_weapon_sound_rules, sound->name);
	if (rule == NULL || rule->action_id != QNN_ACTION_ATTACK)
		return QNN_WEAPON_NONE;
	/* The fire rules store the weapon SUBJECT in the source_id column
	 * (X-macro: {path, QNN_ACTION_ATTACK, subject, 0}). */
	return QNN_RawWeaponIdFromSubject(rule->source_id);
}

qboolean QNN_SnapshotHasSelfJumpSound(const qnn_snapshot_t *snapshot)
{
	return QNN_SnapshotHasSelfActionSound(snapshot, qnn_player_sound_rules, QNN_ACTION_JUMP);
}

/* Per-sound check: true iff `sound` is a jump sound for the self entity.
 * Evaluated one sound at a time so the caller can use the sound's
 * native_time for per-event back-shift decisions. */
qboolean QNN_IsSelfJumpSound(const qnn_sound_event_t *sound)
{
	if (sound->entity_num != cl.viewentity)
		return false;
	return QNN_SoundMatchesAction(qnn_player_sound_rules,
		sound->name, QNN_ACTION_JUMP);
}

static qboolean QNN_EmitRecord(qnn_event_record_t *out, int *count, int max,
	const qnn_sound_event_t *snd, int action_id, int source_id)
{
	if (*count >= max)
		return false;
	memset(&out[*count], 0, sizeof(out[*count]));
	out[*count].entity_num = snd->entity_num;
	out[*count].action_id = action_id;
	out[*count].source_id = source_id;
	VectorCopy(snd->origin, out[*count].origin);
	if (qnn_sound_dump)
		fprintf(qnn_sound_dump, "CLASSIFY\t%.3f\tent=%d\tact=%d\tsrc=%d\n",
			(float)cl.mtime[0], snd->entity_num, action_id, source_id);
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
			if (QNN_EmitRecord(out, &count, max, &snapshot->sounds[i], QNN_ACTION_RESPAWN, QNN_SOURCE_NONE))
				out[count - 1].is_respawn = true;
			continue;
		}

		/* Item pickup */
		{
			int pickup_sub = QNN_SoundPickupSubject(name);
			int pickup_cat = pickup_sub > 0 ? QNN_SubjectPickupCategory(pickup_sub) : 0;
			if (pickup_cat > 0 && snapshot->sounds[i].entity_num > 0)
			{
				if (pickup_sub > 0)
				{
					/* Map pickup subject to source ID */
					int src = pickup_sub; /* reuse subject ID for specific items */
					if (pickup_sub == QNN_SUBJECT_ARMOR_GREEN)
						src = QNN_SOURCE_ARMOR; /* generic armor */
					else if (pickup_sub == QNN_SUBJECT_BACKPACK && pickup_cat == 4)
						src = QNN_SOURCE_WEAPON;
					else if (pickup_sub == QNN_SUBJECT_BACKPACK && pickup_cat == 5)
						src = QNN_SOURCE_AMMO;

					if (QNN_EmitRecord(out, &count, max, &snapshot->sounds[i], QNN_ACTION_PICKUP, src))
						out[count - 1].pickup_category = pickup_cat;
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
			else if (!strcmp(name, "items/damage2.wav"))   { pu_sub = QNN_SUBJECT_QUAD; pu_act = QNN_ACTION_ENDING; }
			else if (!strcmp(name, "items/protect2.wav"))  { pu_sub = QNN_SUBJECT_PENT; pu_act = QNN_ACTION_ENDING; }
			else if (!strcmp(name, "items/inv2.wav"))      { pu_sub = QNN_SUBJECT_RING; pu_act = QNN_ACTION_ENDING; }
			else if (!strcmp(name, "items/suit2.wav"))     { pu_sub = QNN_SUBJECT_SUIT; pu_act = QNN_ACTION_ENDING; }
			if (pu_sub > 0)
			{
				if (QNN_EmitRecord(out, &count, max, &snapshot->sounds[i], pu_act, pu_sub))
					out[count - 1].powerup_subject_id = pu_sub;
				continue;
			}
		}

		/* Player sounds (rule table + pattern matches) */
		rule = QNN_FindSoundRule(qnn_player_sound_rules, name);
		if (rule != NULL)
		{
			QNN_EmitRecord(out, &count, max, &snapshot->sounds[i], rule->action_id, rule->source_id);
			continue;
		}
		/* Weapon fire */
		rule = QNN_FindSoundRule(qnn_weapon_sound_rules, name);
		if (rule != NULL)
		{
			if (QNN_EmitRecord(out, &count, max, &snapshot->sounds[i], rule->action_id, rule->source_id))
				out[count - 1].weapon_subject_id = rule->source_id;
			continue;
		}

		/* Projectile bounce */
		rule = QNN_FindSoundRule(qnn_projectile_sound_rules, name);
		if (rule != NULL)
		{
			QNN_EmitRecord(out, &count, max, &snapshot->sounds[i], rule->action_id, rule->source_id);
			continue;
		}

		/* Static objects (doors, platforms, teleporters) */
		rule = QNN_FindSoundRule(qnn_static_sound_rules, name);
		if (rule != NULL)
		{
			QNN_EmitRecord(out, &count, max, &snapshot->sounds[i], rule->action_id, rule->source_id);
			if (count > 0)
				out[count - 1].match_subject_id = rule->match_subject_id;
		}
	}

	return count;
}

/* ══════════════════════════════════════════════════════════════════
 * Event atom pool — persistent per-entity event history.
 *
 * Atoms are created from classified sound records.  They decay
 * over QNN_RECENCY_MAX_EVENT and are read by the oracle at emission
 * time to attach per-entity event tokens.
 * ══════════════════════════════════════════════════════════════════ */

qnn_semantic_event_atom_t qnn_semantic_events[QNN_MAX_EVENT_ATOMS];
int qnn_event_head[QNN_EVENT_HEAD_CAPACITY];

void QNN_EventRegisterContext(void)
{
	QNN_ContextRegister(qnn_semantic_events, sizeof(qnn_semantic_events));
	QNN_ContextRegister(qnn_event_head, sizeof(qnn_event_head));
}

static void QNN_AppendEvent(int owner_index, int action_id, int source_id)
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
			&& qnn_semantic_events[i].action_id == action_id
			&& qnn_semantic_events[i].source_id == source_id)
		{
			qnn_semantic_events[i].timestamp = (float)cl.mtime[0];
			if (qnn_sound_dump)
				fprintf(qnn_sound_dump, "APPEND_DEDUP\t%.3f\town=%d\tact=%d\tsrc=%d\n",
					(float)cl.mtime[0], owner_index, action_id, source_id);
			return;
		}
	}
	if (free_index < 0)
	{
		if (qnn_sound_dump)
			fprintf(qnn_sound_dump, "APPEND_FULL\t%.3f\town=%d\tact=%d\tsrc=%d\n",
				(float)cl.mtime[0], owner_index, action_id, source_id);
		return;
	}
	memset(&qnn_semantic_events[free_index], 0, sizeof(qnn_semantic_events[free_index]));
	qnn_semantic_events[free_index].active = true;
	qnn_semantic_events[free_index].owner_index = owner_index;
	qnn_semantic_events[free_index].action_id = action_id;
	qnn_semantic_events[free_index].source_id = source_id;
	qnn_semantic_events[free_index].timestamp = (float)cl.mtime[0];
	if (qnn_sound_dump)
		fprintf(qnn_sound_dump, "APPEND_NEW\t%.3f\town=%d\tact=%d\tsrc=%d\n",
			(float)cl.mtime[0], owner_index, action_id, source_id);

	qnn_semantic_events[free_index].next_for_owner = -1;
	if (owner_index >= 0 && owner_index < QNN_EVENT_HEAD_CAPACITY)
	{
		qnn_semantic_events[free_index].next_for_owner = qnn_event_head[owner_index];
		qnn_event_head[owner_index] = free_index;
	}
}

static void QNN_ExpireEvents(float now)
{
	int i;
	int cap = QNN_StoreCapacity() < QNN_EVENT_HEAD_CAPACITY
		? QNN_StoreCapacity() : QNN_EVENT_HEAD_CAPACITY;

	/* Deactivate expired events */
	for (i = 0; i < QNN_MAX_EVENT_ATOMS; ++i)
	{
		if (!qnn_semantic_events[i].active)
			continue;
		if (now - qnn_semantic_events[i].timestamp > QNN_RECENCY_MAX_EVENT)
			qnn_semantic_events[i].active = false;
	}

	/* Rebuild per-entity event head linked list */
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

/* Allocate an overflow slot for an unknown player heard at origin.
   Reuses expired slots (snd outside QNN_RECENCY_MAX_SOUND).
   Returns the store index, or -1 if pool is full. */
static int QNN_AllocUnknownPlayer(const vec3_t origin, float now)
{
	int base = MAX_EDICTS + qnn_store_overflow_count;
	int limit = MAX_EDICTS + QNN_STORE_OVERFLOW;
	int j, oldest = -1;
	float oldest_snd = now;

	for (j = base; j < limit; ++j)
	{
		qnn_entity_t *e = &qnn_store[j];
		/* Free slot */
		if (e->type == QNN_ENT_NONE)
		{
			oldest = j;
			break;
		}
		/* Expired unknown player — recycle */
		if (e->type == QNN_ENT_ACTOR && e->snd < oldest_snd)
		{
			oldest_snd = e->snd;
			oldest = j;
		}
	}
	if (oldest < 0)
		return -1;

	memset(&qnn_store[oldest], 0, sizeof(qnn_store[oldest]));
	qnn_store[oldest].type = QNN_ENT_ACTOR;
	qnn_store[oldest].subject_id = QNN_SUBJECT_PLAYER;
	qnn_store[oldest].entity_num = 0;
	qnn_store[oldest].snd = now;
	VectorCopy(origin, qnn_store[oldest].origin);
	return oldest;
}

/* Find the nearest player entity to a position.
   Returns the store index, or -1 if no player is within max_dsq. */
static int QNN_FindNearestPlayer(const vec3_t origin, float max_dsq)
{
	int j, best = -1;
	float best_dsq = max_dsq;
	for (j = 1; j <= cl.maxclients && j < MAX_EDICTS; ++j)
	{
		qnn_entity_t *e = &qnn_store[j];
		float dsq;
		if (e->type != QNN_ENT_ACTOR)
			continue;
		dsq = QNN_DistSq(e->origin, origin);
		if (dsq < best_dsq)
		{
			best_dsq = dsq;
			best = j;
		}
	}
	return best;
}

static void QNN_EventProcessTick(const qnn_snapshot_t *snapshot)
{
	qnn_event_record_t event_records[QNN_MAX_EVENT_RECORDS];
	int event_count, i;
	float now = (float)cl.mtime[0];

	event_count = QNN_EventClassifySounds(snapshot, event_records, QNN_MAX_EVENT_RECORDS);

	for (i = 0; i < event_count; ++i)
	{
		qnn_event_record_t *ev = &event_records[i];
		int owner = -1;  /* entity index that owns this event */

		/* Stock NQ broadcasts sound datagrams globally.  In the grouped arena
		 * that would leak every other match back into the token stream even
		 * though geometry and combat are isolated. */
		if (!strncmp(qnn_map_state.requested_map_id, "qnn_arena", 9)
			&& !QNN_EntityInPvs(snapshot->player_origin, ev->origin))
			continue;

		/* ---- Side effects by event type ---- */

		if (ev->is_respawn)
		{
			if (ev->entity_num > 0 && ev->entity_num < MAX_EDICTS)
			{
				qnn_entity_t *e = &qnn_store[ev->entity_num];
				if (e->type == QNN_ENT_ITEM)
				{
					e->regen = 0.0f;
					e->snd = now;
				}
			}
			owner = ev->entity_num;
			ev->action_id = QNN_ACTION_RESPAWN;
			ev->source_id = QNN_SOURCE_NONE;
		}
		else if (ev->pickup_category > 0)
		{
			int j;
			float best_dsq = QNN_ITEM_PICKUP_PLAYER_SQ;
			qnn_entity_t *best = NULL;

			for (j = 1; j < MAX_EDICTS; ++j)
			{
				qnn_entity_t *e = &qnn_store[j];
				float dsq;
				if (e->type != QNN_ENT_ITEM || e->regen > 0.0f)
					continue;
				if (QNN_SubjectPickupCategory(e->subject_id) != ev->pickup_category)
					continue;
				dsq = QNN_DistSq(e->origin, ev->origin);
				if (dsq < best_dsq)
				{
					best = e;
					best_dsq = dsq;
				}
			}
			if (best)
			{
				best->regen = best->regen_time;
				best->snd = now;
			}
			owner = ev->entity_num;
		}
		else if (ev->action_id == QNN_ACTION_DISCONNECT)
			continue; /* scoreboard check in StoreUpdate handles cleanup */
		else if (ev->action_id == QNN_ACTION_TELEPORT)
		{
			/* Teleport sounds play at the destination marker, not a player.
			   Match to the nearest player, or allocate an unknown. */
			owner = QNN_FindNearestPlayer(ev->origin, QNN_ITEM_PICKUP_PLAYER_SQ);
			if (owner <= 0)
				owner = QNN_AllocUnknownPlayer(ev->origin, now);
		}
		else if (ev->entity_num <= 0)
		{
			/* Mover sounds: no entity_num in demo, match by position.
			   Includes overflow slots for BSP-only entities (teleporters, push). */
			int j;
			int store_size = QNN_StoreCapacity();
			float best_dsq = QNN_STATIC_SOUND_MATCH_SQ;
			qnn_entity_t *best = NULL;
			int best_idx = -1;

			for (j = 1; j < store_size; ++j)
			{
				qnn_entity_t *e = &qnn_store[j];
				float dsq;
				if (e->type != QNN_ENT_MOVER && e->type != QNN_ENT_TELEPORTER)
					continue;
				if (e->subject_id != ev->match_subject_id)
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
				/* Mover not visible this HF → infer state change from sound. */
				if (best->pvs < now)
				{
					if (ev->match_subject_id == QNN_SUBJECT_DOOR
						|| ev->match_subject_id == QNN_SUBJECT_PLATFORM
						|| ev->match_subject_id == QNN_SUBJECT_TRAIN)
						best->state = (best->state >= 1.0f) ? 0.0f : 1.0f;
					else if (ev->match_subject_id == QNN_SUBJECT_BUTTON)
						best->state = 1.0f;
				}
				owner = best_idx;
			}
		}
		else
		{
			/* Entity-attributed sound (player, projectile, etc.).
			   If the slot isn't a live first-person player (empty,
			   spectator, or already promoted to a non-actor), try
			   nearest-player attribution and finally the overflow
			   "unknown player" pool. */
			owner = ev->entity_num;
			if (!QNN_IsLivePlayerSlot(owner)
				|| qnn_store[owner].type != QNN_ENT_ACTOR)
			{
				int nearest = QNN_FindNearestPlayer(ev->origin, QNN_ITEM_PICKUP_PLAYER_SQ);
				if (nearest > 0)
					owner = nearest;
				else
					owner = QNN_AllocUnknownPlayer(ev->origin, now);
			}
		}

		/* ---- Player-specific side effects ----
		 * Only promote the slot to an ACTOR entity when it is a live
		 * first-person player.  Spectator slots and unknown-player
		 * overflow indices skip this block (overflow indices are
		 * outside [1, cl.maxclients] and IsLivePlayerSlot rejects). */
		if (QNN_IsLivePlayerSlot(owner))
		{
			qnn_entity_t *e = &qnn_store[owner];

			e->snd = now;
			if (qnn_sound_dump)
				fprintf(qnn_sound_dump, "SET_SND\t%.3f\town=%d\tact=%d\n", now, owner, ev->action_id);

			/* Player slot not visible this HF → re-promote from sound. */
			if (e->pvs < now)
			{
				e->type = QNN_ENT_ACTOR;
				e->subject_id = QNN_SUBJECT_PLAYER;
				e->entity_num = owner;
				VectorCopy(ev->origin, e->origin);
			}

			if (ev->weapon_subject_id > 0)
				e->weapon_subject_id = ev->weapon_subject_id;
			if (ev->powerup_subject_id > 0)
			{
				e->powerup_subject_id = ev->powerup_subject_id;
				if (ev->action_id == QNN_ACTION_ENDING)
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
		}

		/* ---- Stamp snd for non-player entities (movers, teleporters) ---- */
		if (owner > 0 && owner < QNN_StoreCapacity()
			&& qnn_store[owner].type != QNN_ENT_NONE
			&& qnn_store[owner].type != QNN_ENT_ACTOR)
			qnn_store[owner].snd = now;

		/* ---- Universal event emit ---- */
		if (owner > 0 && owner < QNN_StoreCapacity() && qnn_store[owner].type != QNN_ENT_NONE)
			QNN_AppendEvent(owner, ev->action_id, ev->source_id);
	}

	/* ---- Sight-derived events: dimlight = active powerup ----
	 * pvs == now means StoreUpdate stamped this entity this HF (i.e.,
	 * it was visible to the engine in the most recent demo packet).
	 * StoreUpdate runs before EventTick in IOUpdate, so the comparison
	 * is exact — same now value used for both stamp and read. */
	for (i = 1; i < MAX_EDICTS; ++i)
	{
		qnn_entity_t *e = &qnn_store[i];
		if (e->type == QNN_ENT_ACTOR
			&& e->effects & QNN_EF_DIMLIGHT
			&& e->pvs == now)
		{
			/* Player is glowing — emit ACTIVE/POWERUP (generic, unknown type).
			   If a specific powerup hum/ending sound also fires this tick,
			   both events will be on the actor — the sound event carries
			   the specific powerup, this one carries the visual confirmation. */
			QNN_AppendEvent(i, QNN_ACTION_ACTIVE, QNN_SUBJECT_POWERUP);
		}
	}
}

/* ══════════════════════════════════════════════════════════════════
 * Public API — called from qnn_io.c
 * ══════════════════════════════════════════════════════════════════ */

void QNN_EventInit(const qnn_map_state_t *map_state)
{
	const char *dump_path;

	memset(qnn_semantic_events, 0, sizeof(qnn_semantic_events));
	memset(qnn_event_head, -1, sizeof(qnn_event_head));

	if (qnn_sound_dump != NULL)
	{
		fclose(qnn_sound_dump);
		qnn_sound_dump = NULL;
	}
	dump_path = getenv("QNN_SOUND_DUMP");
	if (dump_path != NULL && dump_path[0] != '\0')
	{
		qnn_sound_dump = fopen(dump_path, "w");
		if (qnn_sound_dump != NULL)
			fprintf(stderr, "[demo] sound dump: %s\n", dump_path);
	}

	QNN_StoreInit(map_state);
}

void QNN_EventTick(const qnn_snapshot_t *snapshot, float dt, qboolean reset_flag)
{
	if (reset_flag)
	{
		memset(qnn_semantic_events, 0, sizeof(qnn_semantic_events));
		memset(qnn_event_head, -1, sizeof(qnn_event_head));
	}

	{
		float now = (float)cl.mtime[0];

		/* 1. Expire old events before processing new ones */
		QNN_ExpireEvents(now);

		/* 2. Process new sounds — updates timestamps, adds events */
		QNN_EventProcessTick(snapshot);

	}
}
