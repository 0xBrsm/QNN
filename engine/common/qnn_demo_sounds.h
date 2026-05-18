/*
 * qnn_demo_sounds.h — canonical vanilla-NQ sound paths for fire and
 * jump detection.  Shared by two consumers:
 *
 *   1. The engine's qnn_event.c uses these names to populate the
 *      qnn_weapon_sound_rules[] and qnn_player_sound_rules[] tables
 *      (alongside action / source / subject metadata).
 *
 *   2. The standalone QWD demo classifier (src/demo/qw_classifier.c)
 *      uses them to classify precached sound names into
 *      fire / jump / other when emitting active_input fields for
 *      the NQ corpus manifest.
 *
 * Pattern is the standard X-macro: caller defines an X() macro,
 * invokes QNN_FIRE_SOUND_LIST(X) or QNN_JUMP_SOUND_LIST(X), then
 * #undefs X.  Each consumer expands the entries differently.
 *
 * Names extracted from vendor/quakec/qc/{weapons,client}.qc top-level
 * `sound (self, ...)` emissions — i.e. sounds the player's own entity
 * makes during attack / jump.  Mod sounds (e.g. KTeams custom paths)
 * are out of scope and fall through to "other" in the classifier.
 */
#ifndef QNN_DEMO_SOUNDS_H
#define QNN_DEMO_SOUNDS_H

/* (path, subject) — subject is one of QNN_SUBJECT_<WEAPON> from qnn.h.
 * The classifier discards the subject; the engine table uses it. */
#define QNN_FIRE_SOUND_LIST(X) \
	X("weapons/ax1.wav",      QNN_SUBJECT_AXE) \
	X("weapons/guncock.wav",  QNN_SUBJECT_SHOTGUN) \
	X("weapons/shotgn2.wav",  QNN_SUBJECT_SUPER_SHOTGUN) \
	X("weapons/rocket1i.wav", QNN_SUBJECT_NAILGUN) \
	X("weapons/spike2.wav",   QNN_SUBJECT_SUPER_NAILGUN) \
	X("weapons/grenade.wav",  QNN_SUBJECT_GRENADE_LAUNCHER) \
	X("weapons/sgun1.wav",    QNN_SUBJECT_ROCKET_LAUNCHER) \
	X("weapons/lstart.wav",   QNN_SUBJECT_THUNDERBOLT) \
	X("weapons/lhit.wav",     QNN_SUBJECT_THUNDERBOLT)

/* (path) — jump only.  Vendor client.qc line 56:
 *   sound (self, CHAN_BODY, "player/plyrjmp8.wav", 1, ATTN_NORM);
 * `player/h2ojump.wav` is the water-LAND sound per vendor, not a
 * jump initiation — not included. */
#define QNN_JUMP_SOUND_LIST(X) \
	X("player/plyrjmp8.wav")

#endif  /* QNN_DEMO_SOUNDS_H */
