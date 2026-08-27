/*
 * qnn_weapon.h — the RAW weapon id space (1..8) and its per-weapon data.
 *
 * ── Two weapon-id spaces, do not confuse them ──────────────────────────
 *
 *   RAW weapon id     1..8   Axe=1 SG=2 SSG=3 NG=4 SNG=5 GL=6 RL=7 LG=8
 *                            (QC impulse order). This is what every engine
 *                            and label path uses: snapshot->weapon_id, the
 *                            categorical action.attack label, the QC
 *                            predicates (QNN_ProgsEvalAttack guards 1..8),
 *                            cooldown/dedup tables. Use qnn_weapon_id_t /
 *                            QNN_WEAPON_* for these.
 *
 *   SUBJECT/vocab id  3..10  QNN_SUBJECT_AXE=3 .. THUNDERBOLT=10 (qnn_vocab.h).
 *                            Exists ONLY because the embedding vocab reserves
 *                            rows 0/1/2 (NONE/PLAYER/WEAPON-source); weapons
 *                            are shifted +2 so SG/SSG and NG/SNG embed rows
 *                            sit adjacent. It is an embedding-layout concern,
 *                            nothing more.
 *
 * The +2 is the source of a recurring class of bugs: weapon-indexed tables
 * written with the 3..10 mental model but indexed by a raw 1..8 id (and vice
 * versa). RULE: the subject space appears at EXACTLY ONE boundary — the
 * obs-token emit (qnn_weapon_subject_from_id in qnn.h, mirrored by
 * token_builder.py / vocab.py). Everywhere else is raw 1..8. Never write a
 * bare +2 / -2 weapon literal, and never index a raw table with QNN_SUBJECT_*.
 *
 * Per-weapon DATA (cooldown, label) is reached only through the accessors
 * below — there is intentionally no exposed raw array to mis-index.
 */

#ifndef QNN_WEAPON_H
#define QNN_WEAPON_H

/* Canonical raw-weapon table — the single source of truth. Order IS the raw
 * id order (Axe first = 1). Columns: (enum, short label, attack cooldown in
 * seconds per vanilla weapons.qc). Add a weapon here and every table/accessor
 * below grows with it. */
#define QNN_WEAPON_LIST(X) \
	X(QNN_WEAPON_AXE,           "Axe", 0.5f) \
	X(QNN_WEAPON_SHOTGUN,       "SG",  0.5f) \
	X(QNN_WEAPON_SUPER_SHOTGUN, "SSG", 0.7f) \
	X(QNN_WEAPON_NAILGUN,       "NG",  0.2f) \
	X(QNN_WEAPON_SUPER_NAILGUN, "SNG", 0.2f) \
	X(QNN_WEAPON_GRENADE,       "GL",  0.6f) \
	X(QNN_WEAPON_ROCKET,        "RL",  0.8f) \
	X(QNN_WEAPON_LIGHTNING,     "LG",  0.2f)  /* effective op-fire cadence: W_Attack literal is +0.1, but the player_light think-chain makes the measured re-fire 0.2s (== nailguns). Feeds QNN_MvdStampAttackFinished (real-MVD feasibility). See qnn_onnx.c hold-tail note. */

typedef enum
{
	QNN_WEAPON_NONE = 0,
#define QNN_WEAPON_ENUM_ROW(e, label, cd) e,
	QNN_WEAPON_LIST(QNN_WEAPON_ENUM_ROW)
#undef QNN_WEAPON_ENUM_ROW
	QNN_WEAPON_COUNT   /* = 9 (NONE + 8 weapons) */
} qnn_weapon_id_t;

/* QNN_WEAPON_AXE == 1 and QNN_WEAPON_LIGHTNING == 8 by construction. */

/* 1 iff weapon_id is a real raw weapon (QNN_WEAPON_AXE..QNN_WEAPON_LIGHTNING).
 * Inline so any path (QW or NQ) can guard a raw weapon id without linking
 * qnn_weapon.c. This is THE raw-id range check — use it instead of bare
 * `id >= 1 && id <= 8` literals. */
static inline int QNN_WeaponIsValid(int weapon_id)
{
	return (weapon_id >= QNN_WEAPON_AXE
		&& weapon_id <= QNN_WEAPON_LIGHTNING) ? 1 : 0;
}

/* Vanilla attack_finished cooldown for a raw weapon id, in seconds.
 * 0 for QNN_WEAPON_NONE or any out-of-range id. */
float QNN_WeaponCooldownSec(int weapon_id);

/* Short label ("Axe", "SG", ...) for a raw weapon id; "none"/"?" for
 * NONE / out-of-range. Never NULL. */
const char *QNN_WeaponName(int weapon_id);

#endif /* QNN_WEAPON_H */
